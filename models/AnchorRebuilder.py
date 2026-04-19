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


class AnchorRebuilder(nn.Module):
    def __init__(self, conf, user_embeddings, bundle_embeddings):
        super().__init__()

        if user_embeddings.ndim != 2 or bundle_embeddings.ndim != 2:
            raise ValueError("External user and bundle embeddings must be 2D")
        if user_embeddings.shape[1] != bundle_embeddings.shape[1]:
            raise ValueError("User and bundle embeddings must share the same embedding dimension")

        self.embedding_dim = user_embeddings.shape[1]
        self.hidden_dim = conf["anchor_hidden_dim"]
        self.dropout = conf["anchor_dropout"]
        self.eval_chunk_size = conf["anchor_eval_chunk_size"]

        anchor_input_dim = self.embedding_dim * 4
        target_input_dim = self.embedding_dim * 6

        self.anchor_encoder = MLP(
            anchor_input_dim,
            self.hidden_dim,
            self.embedding_dim,
            num_layers=conf["anchor_encoder_layers"],
            dropout=self.dropout,
        )
        self.target_scorer = MLP(
            target_input_dim,
            self.hidden_dim,
            1,
            num_layers=conf["anchor_scorer_layers"],
            dropout=self.dropout,
        )

        self.register_buffer("user_embeddings", user_embeddings.detach().float())
        self.register_buffer("bundle_embeddings", bundle_embeddings.detach().float())

    def _anchor_features(self, user_vec, anchor_vec):
        return torch.cat(
            [user_vec, anchor_vec, user_vec * anchor_vec, torch.abs(user_vec - anchor_vec)],
            dim=-1,
        )

    def _target_features(self, anchor_context, user_vec, anchor_vec, target_vec):
        return torch.cat(
            [
                anchor_context,
                user_vec,
                target_vec,
                anchor_context * target_vec,
                anchor_vec * target_vec,
                torch.abs(anchor_vec - target_vec),
            ],
            dim=-1,
        )

    def _encode_anchor(self, user_indices, anchor_indices):
        user_vec = self.user_embeddings[user_indices]
        anchor_vec = self.bundle_embeddings[anchor_indices]
        anchor_input = self._anchor_features(user_vec, anchor_vec)
        anchor_context = self.anchor_encoder(anchor_input)
        return anchor_context, user_vec, anchor_vec

    def score_targets(self, anchor_context, user_vec, anchor_vec, target_indices):
        if target_indices.ndim == 1:
            target_vec = self.bundle_embeddings[target_indices]
            features = self._target_features(anchor_context, user_vec, anchor_vec, target_vec)
            return self.target_scorer(features).squeeze(-1)

        target_vec = self.bundle_embeddings[target_indices]
        batch_size, target_count = target_indices.shape
        anchor_context = anchor_context.unsqueeze(1).expand(-1, target_count, -1)
        user_vec = user_vec.unsqueeze(1).expand(-1, target_count, -1)
        anchor_vec = anchor_vec.unsqueeze(1).expand(-1, target_count, -1)
        features = self._target_features(anchor_context, user_vec, anchor_vec, target_vec)
        return self.target_scorer(features).squeeze(-1)

    def training_loss(self, batch):
        user_indices = batch["user_indices"]
        anchor_indices = batch["anchor_indices"]
        pos_indices = batch["pos_indices"]
        neg_indices = batch["neg_indices"]

        anchor_context, user_vec, anchor_vec = self._encode_anchor(user_indices, anchor_indices)
        pos_scores = self.score_targets(anchor_context, user_vec, anchor_vec, pos_indices).unsqueeze(1)
        neg_scores = self.score_targets(anchor_context, user_vec, anchor_vec, neg_indices)
        logits = torch.cat([pos_scores, neg_scores], dim=1)
        target = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, target)

    def rebuild_topk(self, user_indices, observed_bundle_indices, rebuild_k=1):
        if rebuild_k <= 0:
            raise ValueError("rebuild_k must be positive")

        device = self.user_embeddings.device
        observed_bundle_indices = observed_bundle_indices.to(device)
        valid_mask = observed_bundle_indices.ge(0)
        safe_indices = observed_bundle_indices.clamp(min=0)

        batch_size, max_len = safe_indices.shape
        user_vec = self.user_embeddings[user_indices].unsqueeze(1).expand(-1, max_len, -1)
        anchor_vec = self.bundle_embeddings[safe_indices]
        anchor_input = self._anchor_features(user_vec, anchor_vec)
        anchor_context = self.anchor_encoder(anchor_input)

        support_sum = torch.zeros(batch_size, max_len, device=device)
        support_count = torch.zeros(batch_size, max_len, device=device)

        chunk_size = max(1, self.eval_chunk_size)
        for start in range(0, max_len, chunk_size):
            end = min(start + chunk_size, max_len)
            target_indices = safe_indices[:, start:end]
            target_valid = valid_mask[:, start:end]
            target_vec = self.bundle_embeddings[target_indices]
            chunk_len = end - start

            expanded_context = anchor_context.unsqueeze(2).expand(-1, -1, chunk_len, -1)
            expanded_user = user_vec.unsqueeze(2).expand(-1, -1, chunk_len, -1)
            expanded_anchor = anchor_vec.unsqueeze(2).expand(-1, -1, chunk_len, -1)
            expanded_target = target_vec.unsqueeze(1).expand(-1, max_len, -1, -1)

            features = self._target_features(
                expanded_context,
                expanded_user,
                expanded_anchor,
                expanded_target,
            )
            scores = self.target_scorer(features).squeeze(-1)

            pair_mask = valid_mask.unsqueeze(2) & target_valid.unsqueeze(1)
            if start < max_len:
                diag_positions = torch.arange(start, end, device=device)
                for offset, col_idx in enumerate(diag_positions):
                    pair_mask[:, col_idx, offset] = False

            scores = scores.masked_fill(~pair_mask, 0.0)
            support_sum += scores.sum(dim=2)
            support_count += pair_mask.sum(dim=2)

        anchor_scores = support_sum / support_count.clamp(min=1.0)
        anchor_scores = anchor_scores.masked_fill(~valid_mask, float("-inf"))

        topk = min(rebuild_k, max_len)
        _, selected_pos = torch.topk(anchor_scores, k=topk, dim=1)
        selected_bundles = torch.gather(safe_indices, 1, selected_pos)
        selected_valid = torch.gather(valid_mask, 1, selected_pos)
        return selected_bundles, selected_valid
