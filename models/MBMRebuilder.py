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


class MBMRebuilder(nn.Module):
    def __init__(self, conf, user_embeddings, bundle_embeddings):
        super().__init__()

        if user_embeddings.ndim != 2 or bundle_embeddings.ndim != 2:
            raise ValueError("External user and bundle embeddings must be 2D")
        if user_embeddings.shape[1] != bundle_embeddings.shape[1]:
            raise ValueError("User and bundle embeddings must share the same embedding dimension")

        self.embedding_dim = user_embeddings.shape[1]
        self.hidden_dim = conf["mbm_hidden_dim"]
        self.dropout = conf["mbm_dropout"]
        self.anchor_lambda = conf["mbm_anchor_lambda"]
        self.score_mode = conf["mbm_score_mode"]
        self.contribution_metric = conf["mbm_contribution_metric"]

        token_input_dim = self.embedding_dim * 4
        candidate_input_dim = self.embedding_dim * 6

        self.phi = MLP(
            token_input_dim,
            self.hidden_dim,
            self.hidden_dim,
            num_layers=conf["mbm_phi_layers"],
            dropout=self.dropout,
        )
        self.rho = MLP(
            self.hidden_dim,
            self.hidden_dim,
            self.hidden_dim,
            num_layers=conf["mbm_rho_layers"],
            dropout=self.dropout,
        )
        self.context_proj = nn.Linear(self.hidden_dim, self.embedding_dim)
        self.scorer = MLP(
            candidate_input_dim,
            self.hidden_dim,
            1,
            num_layers=conf["mbm_scorer_layers"],
            dropout=self.dropout,
        )

        # The exported LightGCN embeddings are treated as a frozen structural prior.
        self.register_buffer("user_embeddings", user_embeddings.detach().float())
        self.register_buffer("bundle_embeddings", bundle_embeddings.detach().float())

    def _build_token_input(self, user_vec, bundle_vec):
        return torch.cat(
            [user_vec, bundle_vec, user_vec * bundle_vec, torch.abs(user_vec - bundle_vec)],
            dim=-1,
        )

    def _context_from_tokens(self, token_sum, count):
        count = count.clamp(min=1.0)
        mean_token = token_sum / count.unsqueeze(-1)
        context_hidden = self.rho(mean_token)
        return self.context_proj(context_hidden)

    def _candidate_features(self, context_vec, user_vec, bundle_vec):
        return torch.cat(
            [
                context_vec,
                user_vec,
                bundle_vec,
                user_vec * bundle_vec,
                context_vec * bundle_vec,
                torch.abs(context_vec - bundle_vec),
            ],
            dim=-1,
        )

    def score_candidates(self, context_vec, user_indices, candidate_indices):
        user_vec = self.user_embeddings[user_indices]
        if candidate_indices.ndim == 1:
            bundle_vec = self.bundle_embeddings[candidate_indices]
            features = self._candidate_features(context_vec, user_vec, bundle_vec)
            return self.scorer(features).squeeze(-1)

        batch_size, candidate_count = candidate_indices.shape
        user_vec = user_vec.unsqueeze(1).expand(-1, candidate_count, -1)
        context_vec = context_vec.unsqueeze(1).expand(-1, candidate_count, -1)
        bundle_vec = self.bundle_embeddings[candidate_indices]
        features = self._candidate_features(context_vec, user_vec, bundle_vec)
        return self.scorer(features).squeeze(-1)

    def training_loss(self, batch):
        user_indices = batch["user_indices"]
        context_indices = batch["context_indices"]
        context_mask = batch["context_mask"]
        pos_indices = batch["pos_indices"]
        neg_indices = batch["neg_indices"]

        safe_context_indices = context_indices.clamp(min=0)
        user_vec = self.user_embeddings[user_indices].unsqueeze(1).expand(-1, context_indices.shape[1], -1)
        bundle_vec = self.bundle_embeddings[safe_context_indices]
        token_input = self._build_token_input(user_vec, bundle_vec)
        token_hidden = self.phi(token_input) * context_mask.unsqueeze(-1)
        token_sum = token_hidden.sum(dim=1)
        context_vec = self._context_from_tokens(token_sum, context_mask.sum(dim=1))

        pos_scores = self.score_candidates(context_vec, user_indices, pos_indices).unsqueeze(1)
        neg_scores = self.score_candidates(context_vec, user_indices, neg_indices)
        logits = torch.cat([pos_scores, neg_scores], dim=1)
        target = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, target)

    def rebuild_topk(self, user_indices, observed_bundle_indices, rebuild_k=1):
        if rebuild_k <= 0:
            raise ValueError("rebuild_k must be positive")

        device = self.user_embeddings.device
        observed_bundle_indices = observed_bundle_indices.to(device)
        batch_size, max_len = observed_bundle_indices.shape

        valid_mask = observed_bundle_indices.ge(0)
        safe_indices = observed_bundle_indices.clamp(min=0)
        user_vec = self.user_embeddings[user_indices].unsqueeze(1).expand(-1, max_len, -1)
        bundle_vec = self.bundle_embeddings[safe_indices]
        token_input = self._build_token_input(user_vec, bundle_vec)
        token_hidden = self.phi(token_input) * valid_mask.unsqueeze(-1)
        token_sum = token_hidden.sum(dim=1)
        counts = valid_mask.sum(dim=1).float()

        full_context = self._context_from_tokens(token_sum, counts)
        leave_counts = (counts.unsqueeze(1).expand(-1, max_len) - 1).clamp(min=1.0)
        leave_token_sum = token_sum.unsqueeze(1) - token_hidden
        leave_context = self._context_from_tokens(
            leave_token_sum.view(batch_size * max_len, self.hidden_dim),
            leave_counts.view(-1),
        ).view(batch_size, max_len, self.embedding_dim)

        rec_features = self._candidate_features(leave_context, user_vec, bundle_vec)
        rec_scores = self.scorer(rec_features).squeeze(-1)
        rec_scores = rec_scores.masked_fill(~valid_mask, float("-inf"))

        if self.score_mode == "rec_only":
            anchor_scores = rec_scores
        else:
            if self.contribution_metric == "context_shift":
                contribution_scores = torch.norm(
                    full_context.unsqueeze(1) - leave_context,
                    p=2,
                    dim=-1,
                )
            else:
                raise ValueError(f"Unsupported contribution metric {self.contribution_metric}")
            contribution_scores = contribution_scores.masked_fill(~valid_mask, float("-inf"))
            if self.score_mode == "contribution_only":
                anchor_scores = contribution_scores
            elif self.score_mode == "hybrid":
                rec_norm = self._rowwise_minmax(rec_scores, valid_mask)
                contribution_norm = self._rowwise_minmax(contribution_scores, valid_mask)
                anchor_scores = (
                    self.anchor_lambda * contribution_norm
                    + (1.0 - self.anchor_lambda) * rec_norm
                )
                anchor_scores = anchor_scores.masked_fill(~valid_mask, float("-inf"))
            else:
                raise ValueError(f"Unsupported MBM score mode {self.score_mode}")

        topk = min(rebuild_k, max_len)
        _, selected_pos = torch.topk(anchor_scores, k=topk, dim=1)
        selected_bundles = torch.gather(safe_indices, 1, selected_pos)
        selected_valid = torch.gather(valid_mask, 1, selected_pos)
        return selected_bundles, selected_valid

    @staticmethod
    def _rowwise_minmax(scores, valid_mask):
        safe_scores = scores.masked_fill(~valid_mask, float("inf"))
        row_min = safe_scores.min(dim=1, keepdim=True).values
        safe_scores = scores.masked_fill(~valid_mask, float("-inf"))
        row_max = safe_scores.max(dim=1, keepdim=True).values
        denom = (row_max - row_min).clamp(min=1e-8)
        normalized = (scores - row_min) / denom
        return normalized.masked_fill(~valid_mask, 0.0)
