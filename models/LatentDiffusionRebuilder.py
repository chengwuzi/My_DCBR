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

        model_input_dim = self.embedding_dim * 4 + self.time_dim
        self.denoiser = MLP(
            model_input_dim,
            self.hidden_dim,
            self.embedding_dim,
            num_layers=conf["latent_diffusion_denoiser_layers"],
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

    def _pool_observed(self, observed_bundle_indices, observed_mask):
        safe_indices = observed_bundle_indices.clamp(min=0)
        observed_vecs = self.bundle_embeddings[safe_indices]
        observed_mask = observed_mask.unsqueeze(-1)
        summed = (observed_vecs * observed_mask).sum(dim=1)
        counts = observed_mask.sum(dim=1).clamp(min=1.0)
        return summed / counts

    def _model_features(self, z_t, user_vec, timesteps):
        return torch.cat(
            [
                z_t,
                user_vec,
                z_t * user_vec,
                torch.abs(z_t - user_vec),
                self._time_embedding(timesteps),
            ],
            dim=-1,
        )

    def _predict_clean(self, z_t, user_vec, timesteps):
        features = self._model_features(z_t, user_vec, timesteps)
        return self.denoiser(features)

    def _extract(self, values, timesteps, reference):
        out = values[timesteps].view(-1, 1)
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

        z0 = self._pool_observed(observed_indices, observed_mask)
        user_vec = self.user_embeddings[user_indices]
        timesteps = torch.randint(0, self.num_steps, (z0.shape[0],), device=z0.device)
        noise = torch.randn_like(z0)
        z_t = self.q_sample(z0, timesteps, noise)
        z_pred = self._predict_clean(z_t, user_vec, timesteps)

        diff_loss = F.mse_loss(z_pred, z0)

        safe_indices = observed_indices.clamp(min=0)
        observed_vecs = self.bundle_embeddings[safe_indices]
        pred_norm = F.normalize(z_pred, dim=-1).unsqueeze(1)
        observed_norm = F.normalize(observed_vecs, dim=-1)
        cosine = (pred_norm * observed_norm).sum(dim=-1)
        cosine = cosine * observed_mask
        mean_cosine = cosine.sum(dim=1) / observed_mask.sum(dim=1).clamp(min=1.0)
        set_loss = (1.0 - mean_cosine).mean()

        return diff_loss + self.set_loss_weight * set_loss

    def refine_latent(self, user_indices, observed_bundle_indices, observed_mask):
        z = self._pool_observed(observed_bundle_indices, observed_mask)
        user_vec = self.user_embeddings[user_indices]
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
        z_norm = F.normalize(z_star, dim=-1).unsqueeze(1)
        observed_vecs = self.bundle_embeddings[safe_indices]
        observed_norm = F.normalize(observed_vecs, dim=-1)
        scores = (z_norm * observed_norm).sum(dim=-1)
        scores = scores.masked_fill(~valid_mask, float("-inf"))

        topk = min(rebuild_k, observed_bundle_indices.shape[1])
        _, selected_pos = torch.topk(scores, k=topk, dim=1)
        selected_bundles = torch.gather(safe_indices, 1, selected_pos)
        selected_valid = torch.gather(valid_mask, 1, selected_pos)
        return selected_bundles, selected_valid
