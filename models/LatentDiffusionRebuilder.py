import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.0):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive")

        dims = [input_dim]
        if num_layers == 1:
            dims.append(output_dim)
        else:
            dims.extend([hidden_dim] * (num_layers - 1))
            dims.append(output_dim)

        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x):
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx != len(self.layers) - 1:
                x = F.relu(x, inplace=False)
                x = self.dropout(x)
        return x


class LatentDiffusionRebuilder(nn.Module):
    def __init__(self, conf, user_embeddings, bundle_embeddings):
        super().__init__()

        if user_embeddings.ndim != 2 or bundle_embeddings.ndim != 2:
            raise ValueError("External user and bundle embeddings must be 2D")
        if user_embeddings.shape[1] != bundle_embeddings.shape[1]:
            raise ValueError("User and bundle embeddings must share the same embedding dimension")

        self.embedding_dim = user_embeddings.shape[1]
        self.hidden_dim = conf["latent_diffusion_hidden_dim"]
        self.time_dim = conf["latent_diffusion_time_dim"]
        self.num_steps = conf["latent_diffusion_num_steps"]
        self.dropout = conf["latent_diffusion_dropout"]
        self.set_loss_weight = conf["latent_diffusion_set_loss_weight"]
        self.num_slots = conf.get("latent_diffusion_num_slots", 4)
        self.score_agg = conf.get("latent_diffusion_score_agg", "max")
        self.diversity_weight = conf.get("latent_diffusion_diversity_weight", 1.0e-3)

        if self.num_slots <= 0:
            raise ValueError("latent_diffusion_num_slots must be positive")

        model_input_dim = self.embedding_dim * 4 + self.time_dim
        self.denoiser = MLP(
            model_input_dim,
            self.hidden_dim,
            self.embedding_dim,
            num_layers=conf["latent_diffusion_denoiser_layers"],
            dropout=self.dropout,
        )
        self.slot_queries = nn.Parameter(torch.empty(self.num_slots, self.embedding_dim))
        self.slot_key = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.slot_value = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.user_to_slot = nn.Linear(
            self.embedding_dim,
            self.num_slots * self.embedding_dim,
        )
        self.slot_fusion = MLP(
            self.embedding_dim * 4,
            self.hidden_dim,
            self.embedding_dim,
            num_layers=2,
            dropout=self.dropout,
        )

        self.register_buffer("user_embeddings", user_embeddings.detach().float())
        self.register_buffer("bundle_embeddings", bundle_embeddings.detach().float())

        betas = torch.linspace(
            conf["latent_diffusion_beta_start"],
            conf["latent_diffusion_beta_end"],
            self.num_steps,
            dtype=torch.float32,
        )
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(torch.clamp(1.0 - alphas_cumprod, min=1.0e-8)),
        )

    def reset_parameters(self):
        self.denoiser.reset_parameters()
        nn.init.normal_(self.slot_queries, mean=0.0, std=0.02)
        self.slot_key.reset_parameters()
        self.slot_value.reset_parameters()
        self.user_to_slot.reset_parameters()
        self.slot_fusion.reset_parameters()

    def _time_embedding(self, timesteps):
        device = timesteps.device
        half_dim = self.time_dim // 2
        if half_dim == 0:
            return torch.zeros((timesteps.shape[0], 0), device=device)
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half_dim, dtype=torch.float32, device=device)
            / max(half_dim, 1)
        )
        args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
        if self.time_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros((emb.shape[0], 1), device=device)], dim=1)
        return emb

    def _encode_observed_slots(self, observed_bundle_indices, observed_mask, user_vec):
        safe_indices = observed_bundle_indices.clamp(min=0)
        observed_vecs = F.normalize(self.bundle_embeddings[safe_indices], dim=-1)
        user_vec = F.normalize(user_vec, dim=-1)
        keys = self.slot_key(observed_vecs)
        values = self.slot_value(observed_vecs)

        batch_size = user_vec.shape[0]
        queries = F.normalize(self.slot_queries, dim=-1).unsqueeze(0).expand(batch_size, -1, -1)
        queries = queries + self.user_to_slot(user_vec).view(batch_size, self.num_slots, self.embedding_dim)

        scores = torch.einsum("bkd,bld->bkl", queries, keys) / math.sqrt(self.embedding_dim)
        invalid_positions = ~observed_mask.bool().unsqueeze(1)
        scores = scores.masked_fill(invalid_positions, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = attn.masked_fill(invalid_positions, 0.0)

        slot_values = torch.einsum("bkl,bld->bkd", attn, values)
        slot_features = torch.cat(
            [
                slot_values,
                queries,
                slot_values * queries,
                torch.abs(slot_values - queries),
            ],
            dim=-1,
        )
        slots = self.slot_fusion(slot_features.view(batch_size * self.num_slots, -1))
        slots = slots.view(batch_size, self.num_slots, self.embedding_dim)
        return F.normalize(slots, dim=-1)

    def _model_features(self, z_t, user_vec, timesteps):
        batch_size = z_t.shape[0]
        user_slots = user_vec.unsqueeze(1).expand(-1, self.num_slots, -1)
        time_emb = self._time_embedding(timesteps).unsqueeze(1).expand(-1, self.num_slots, -1)
        return torch.cat(
            [
                z_t,
                user_slots,
                z_t * user_slots,
                torch.abs(z_t - user_slots),
                time_emb,
            ],
            dim=-1,
        ).view(batch_size * self.num_slots, -1)

    def _predict_clean(self, z_t, user_vec, timesteps):
        features = self._model_features(z_t, user_vec, timesteps)
        pred = self.denoiser(features)
        return pred.view(z_t.shape[0], self.num_slots, self.embedding_dim)

    def _extract(self, values, timesteps, reference):
        out = values[timesteps]
        while out.ndim < reference.ndim:
            out = out.unsqueeze(-1)
        return out.expand_as(reference)

    def q_sample(self, z0, timesteps, noise=None):
        if noise is None:
            noise = torch.randn_like(z0)
        coeff1 = self._extract(self.sqrt_alphas_cumprod, timesteps, z0)
        coeff2 = self._extract(self.sqrt_one_minus_alphas_cumprod, timesteps, z0)
        return coeff1 * z0 + coeff2 * noise

    def training_loss(self, batch):
        user_indices = batch["user_indices"]
        observed_indices = batch["observed_indices"]
        observed_mask = batch["observed_mask"]

        user_vec = F.normalize(self.user_embeddings[user_indices], dim=-1)
        z0 = self._encode_observed_slots(observed_indices, observed_mask, user_vec)
        timesteps = torch.randint(0, self.num_steps, (z0.shape[0],), device=z0.device)
        noise = torch.randn_like(z0)
        z_t = self.q_sample(z0, timesteps, noise)
        z_pred = self._predict_clean(z_t, user_vec, timesteps)

        diff_loss = F.mse_loss(z_pred, z0)

        safe_indices = observed_indices.clamp(min=0)
        observed_vecs = F.normalize(self.bundle_embeddings[safe_indices], dim=-1)
        pred_norm = F.normalize(z_pred, dim=-1)
        cosine = torch.einsum("bkd,bld->bkl", pred_norm, observed_vecs)
        bundle_coverage = cosine.max(dim=1).values * observed_mask
        mean_coverage = bundle_coverage.sum(dim=1) / observed_mask.sum(dim=1).clamp(min=1.0)
        set_loss = (1.0 - mean_coverage).mean()

        diversity_loss = self._slot_diversity_loss(z_pred)
        return diff_loss + self.set_loss_weight * set_loss + self.diversity_weight * diversity_loss

    def refine_latent(self, user_indices, observed_bundle_indices, observed_mask):
        user_vec = F.normalize(self.user_embeddings[user_indices], dim=-1)
        z = self._encode_observed_slots(observed_bundle_indices, observed_mask, user_vec)
        for step in reversed(range(self.num_steps)):
            timesteps = torch.full((z.shape[0],), step, dtype=torch.long, device=z.device)
            z = self._predict_clean(z, user_vec, timesteps)
        return z

    def rebuild_topk(self, user_indices, observed_bundle_indices, rebuild_k=1):
        if rebuild_k <= 0:
            raise ValueError("rebuild_k must be positive")

        device = self.user_embeddings.device
        observed_bundle_indices = observed_bundle_indices.to(device)
        valid_mask = observed_bundle_indices.ge(0)
        observed_mask = valid_mask.float()
        safe_indices = observed_bundle_indices.clamp(min=0)

        z_star = self.refine_latent(user_indices, safe_indices, observed_mask)
        z_norm = F.normalize(z_star, dim=-1)
        observed_vecs = F.normalize(self.bundle_embeddings[safe_indices], dim=-1)
        slot_scores = torch.einsum("bkd,bld->bkl", z_norm, observed_vecs)
        if self.score_agg == "max":
            scores = slot_scores.max(dim=1).values
        elif self.score_agg == "logsumexp":
            scores = torch.logsumexp(slot_scores, dim=1)
        else:
            raise ValueError(f"Unsupported latent diffusion score aggregator {self.score_agg}")
        scores = scores.masked_fill(~valid_mask, float("-inf"))

        topk = min(rebuild_k, observed_bundle_indices.shape[1])
        _, selected_pos = torch.topk(scores, k=topk, dim=1)
        selected_bundles = torch.gather(safe_indices, 1, selected_pos)
        selected_valid = torch.gather(valid_mask, 1, selected_pos)
        return selected_bundles, selected_valid

    def _slot_diversity_loss(self, slots):
        if self.num_slots <= 1:
            return torch.zeros((), device=slots.device)

        slot_norm = F.normalize(slots, dim=-1)
        similarity = torch.einsum("bkd,bqd->bkq", slot_norm, slot_norm)
        eye = torch.eye(self.num_slots, device=slots.device, dtype=similarity.dtype).unsqueeze(0)
        off_diag = similarity.masked_select(~eye.bool())
        return off_diag.square().mean()
