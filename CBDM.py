import math
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


@dataclass
class CBDMConfig:
    """Default config follows the iFashion setting from DCBR.

    BLCC is kept but disabled by default.
    """

    # iFashion defaults
    dims: Tuple[int, ...] = (1000,)
    norm: bool = False
    time_emb_dim: int = 10
    steps: int = 140
    noise_scale: float = 0.1
    noise_min: float = 1e-4
    noise_max: float = 2e-2
    sampling_noise: bool = False
    sampling_steps: int = 0
    rebuild_k: int = 1
    lambda_0: float = 2.0

    # training
    lr: float = 1e-3
    batch_size_train: int = 2048
    epochs: int = 50
    dropout: float = 0.5
    use_blcc: bool = False
    beta_fixed: bool = True
    shuffle: bool = True
    num_workers: int = 0
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # misc
    verbose: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


DEFAULT_IFASHION_CONFIG: Dict[str, Any] = CBDMConfig().to_dict()


class DiffusionDataset(Dataset):
    def __init__(self, adjacency_matrix: torch.Tensor):
        if adjacency_matrix.dim() != 2:
            raise ValueError("adjacency_matrix must be a 2D dense tensor")
        self.adjacency_matrix = adjacency_matrix

    def __getitem__(self, user_index: int):
        return self.adjacency_matrix[user_index], user_index

    def __len__(self) -> int:
        return self.adjacency_matrix.shape[0]


class DNN(nn.Module):
    def __init__(
        self,
        in_dims: Tuple[int, ...],
        out_dims: Tuple[int, ...],
        time_emb_dim: int,
        norm: bool = False,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.in_dims = tuple(in_dims)
        self.out_dims = tuple(out_dims)
        self.time_emb_dim = int(time_emb_dim)
        self.norm = bool(norm)

        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)
        in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + list(self.in_dims[1:])
        out_dims_temp = list(self.out_dims)

        self.in_layers = nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])]
        )
        self.out_layers = nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])]
        )
        self.drop = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        for layer in list(self.in_layers) + list(self.out_layers) + [self.emb_layer]:
            fan_out, fan_in = layer.weight.size(0), layer.weight.size(1)
            std = math.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        use_dropout: bool = True,
        max_period: int = 10000,
    ) -> torch.Tensor:
        device = x.device
        half = self.time_emb_dim // 2
        if half > 0:
            freqs = torch.exp(
                -math.log(max_period)
                * torch.arange(start=0, end=half, dtype=torch.float32, device=device)
                / half
            )
            args = timesteps[:, None].float() * freqs[None]
            timestep_embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        else:
            timestep_embedding = torch.zeros((x.shape[0], 0), device=device)
        if self.time_emb_dim % 2:
            timestep_embedding = torch.cat(
                [timestep_embedding, torch.zeros_like(timestep_embedding[:, :1])], dim=-1
            )
        emb = self.emb_layer(timestep_embedding)

        if self.norm:
            x = F.normalize(x, dim=-1)
        if use_dropout:
            x = self.drop(x)

        h = torch.cat([x, emb], dim=-1)
        for layer in self.in_layers:
            h = torch.tanh(layer(h))
        for i, layer in enumerate(self.out_layers):
            h = layer(h)
            if i != len(self.out_layers) - 1:
                h = torch.tanh(h)
        return h


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        noise_scale: float,
        noise_min: float,
        noise_max: float,
        steps: int,
        device: Union[str, torch.device],
        beta_fixed: bool = True,
    ):
        super().__init__()
        self.noise_scale = float(noise_scale)
        self.noise_min = float(noise_min)
        self.noise_max = float(noise_max)
        self.steps = int(steps)
        self.device = torch.device(device)

        if self.noise_scale != 0:
            betas = torch.tensor(self.get_betas(), dtype=torch.float64, device=self.device)
            if beta_fixed:
                betas[0] = 1e-4
            self.betas = betas
            self.calculate_for_diffusion(self.device)
        else:
            self.betas = None

    def get_betas(self, max_beta: float = 0.999) -> np.ndarray:
        start = self.noise_scale * self.noise_min
        end = self.noise_scale * self.noise_max
        variance = np.linspace(start, end, self.steps, dtype=np.float64)
        alpha_bar = 1 - variance
        betas = [1 - alpha_bar[0]]
        for i in range(1, self.steps):
            betas.append(min(1 - alpha_bar[i] / alpha_bar[i - 1], max_beta))
        return np.array(betas, dtype=np.float64)

    def calculate_for_diffusion(self, device: Union[str, torch.device]):
        device = torch.device(device)
        alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, axis=0).to(device)
        self.alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]]
        ).to(device)
        self.alphas_cumprod_next = torch.cat(
            [self.alphas_cumprod[1:], torch.tensor([0.0], device=device)]
        ).to(device)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - self.alphas_cumprod)

    def mean_flat(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.mean(dim=list(range(1, len(tensor.shape))))

    def _extract_into_tensor(self, arr: torch.Tensor, timesteps: torch.Tensor, broadcast_shape, device) -> torch.Tensor:
        arr = arr.to(device)
        res = arr[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)

    def SNR(self, t: torch.Tensor) -> torch.Tensor:
        # Clamp because training_CBDM_losses uses ts-1.
        t = torch.clamp(t, min=0)
        alphas_cumprod = self.alphas_cumprod.to(t.device)
        return alphas_cumprod[t] / (1 - alphas_cumprod[t])

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape, x_start.device) * x_start
            + self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape, x_start.device) * noise
        )

    def q_posterior_mean_variance(self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        posterior_mean = (
            self._extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape, x_start.device) * x_start
            + self._extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape, x_start.device) * x_t
        )
        return posterior_mean

    def p_mean_variance(self, model: nn.Module, x: torch.Tensor, t: torch.Tensor) -> Dict[str, torch.Tensor]:
        model_output = model(x, t, use_dropout=False)
        model_log_variance = self._extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape, x.device)
        model_mean = self.q_posterior_mean_variance(x_start=model_output, x_t=x, t=t)
        return {"mean": model_mean, "log_variance": model_log_variance}

    def p_sample(self, model: nn.Module, x_start: torch.Tensor, steps: int, sampling_noise: bool = False) -> torch.Tensor:
        device = x_start.device
        if steps == 0:
            x_t = x_start
        else:
            t = torch.tensor([steps - 1] * x_start.shape[0], device=device)
            x_t = self.q_sample(x_start, t)

        indices = list(range(self.steps))[::-1]
        if self.noise_scale == 0.0:
            for i in indices:
                t = torch.tensor([i] * x_t.shape[0], device=device)
                x_t = model(x_t, t)
            return x_t

        for i in indices:
            t = torch.tensor([i] * x_t.shape[0], device=device)
            out = self.p_mean_variance(model, x_t, t)
            if sampling_noise:
                noise = torch.randn_like(x_t)
                nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
                x_t = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
            else:
                x_t = out["mean"]
        return x_t

    def training_CBDM_losses(
        self,
        model: nn.Module,
        x_start: torch.Tensor,
        user_embeds: Optional[torch.Tensor],
        bundle_embeds: Optional[torch.Tensor],
        batch_user_index: torch.Tensor,
        use_blcc: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = x_start.device
        batch_size = x_start.size(0)
        ts = torch.randint(0, self.steps, (batch_size,), device=device).long()
        noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, ts, noise) if self.noise_scale != 0 else x_start

        model_output = model(x_t, ts)
        mse = self.mean_flat((x_start - model_output) ** 2)
        weight = self.SNR(ts - 1) - self.SNR(ts)
        weight = torch.where(ts == 0, torch.ones_like(weight), weight)
        elbo_loss = weight * mse

        if use_blcc:
            if user_embeds is None or bundle_embeds is None:
                raise ValueError("user_embeds and bundle_embeds are required when use_blcc=True")
            batch_user_embeds = user_embeds[batch_user_index]
            new_bundle_embeds = torch.mm(model_output.transpose(0, 1), batch_user_embeds)
            blcc_loss = self.mean_flat((new_bundle_embeds - bundle_embeds) ** 2)
            return elbo_loss.mean(), blcc_loss.mean()

        zero = torch.zeros((), device=device)
        return elbo_loss.mean(), zero


class CBDM:
    """Standalone CBDM module extracted from DCBR.

    Default behavior = pure CBDM (BLCC off).
    Outputs both denoised_ub_mat and UB_propagation_graph.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        merged = DEFAULT_IFASHION_CONFIG.copy()
        if config:
            merged.update(config)
        self.config = CBDMConfig(**merged)
        self.device = torch.device(self.config.device)
        self._set_seed(self.config.seed)

        self.num_users: Optional[int] = None
        self.num_bundles: Optional[int] = None
        self.denoise_model: Optional[DNN] = None
        self.diffusion_model: Optional[GaussianDiffusion] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self._fitted = False
        self.train_history = []

    @staticmethod
    def _set_seed(seed: int):
        import random

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def _build_models(self, num_bundles: int):
        out_dims = tuple(self.config.dims) + (num_bundles,)
        in_dims = tuple(reversed(out_dims))
        self.denoise_model = DNN(
            in_dims=in_dims,
            out_dims=out_dims,
            time_emb_dim=self.config.time_emb_dim,
            norm=self.config.norm,
            dropout=self.config.dropout,
        ).to(self.device)
        self.diffusion_model = GaussianDiffusion(
            noise_scale=self.config.noise_scale,
            noise_min=self.config.noise_min,
            noise_max=self.config.noise_max,
            steps=self.config.steps,
            device=self.device,
            beta_fixed=self.config.beta_fixed,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.denoise_model.parameters(), lr=self.config.lr, weight_decay=0)

    @staticmethod
    def _to_dense_tensor(ub_matrix: Union[sp.spmatrix, np.ndarray, torch.Tensor]) -> torch.Tensor:
        if sp.issparse(ub_matrix):
            dense = ub_matrix.toarray().astype(np.float32, copy=False)
            return torch.from_numpy(dense)
        if isinstance(ub_matrix, np.ndarray):
            return torch.from_numpy(ub_matrix.astype(np.float32, copy=False))
        if isinstance(ub_matrix, torch.Tensor):
            return ub_matrix.float().cpu()
        raise TypeError("ub_matrix must be scipy sparse matrix, numpy array, or torch tensor")

    @staticmethod
    def build_ub_propagation_graph(
        denoised_ub_mat: sp.spmatrix,
        device: Union[str, torch.device] = "cpu",
    ) -> torch.Tensor:
        num_users, num_bundles = denoised_ub_mat.shape
        adjacency_matrix = sp.bmat(
            [
                [sp.csr_matrix((num_users, num_users)), denoised_ub_mat],
                [denoised_ub_mat.T, sp.csr_matrix((num_bundles, num_bundles))],
            ],
            format="csr",
        )
        adjacency_matrix = adjacency_matrix + sp.eye(adjacency_matrix.shape[0], format="csr")
        row_sum = np.asarray(adjacency_matrix.sum(axis=1)).reshape(-1)
        d_inv = np.power(row_sum, -0.5, where=row_sum > 0)
        d_inv[np.isinf(d_inv)] = 0.0
        d_inv[row_sum <= 0] = 0.0
        degree_matrix = sp.diags(d_inv)
        norm_adjacency = (degree_matrix @ adjacency_matrix @ degree_matrix).tocoo()
        indices = np.vstack((norm_adjacency.row, norm_adjacency.col))
        values = norm_adjacency.data.astype(np.float32, copy=False)
        return torch.sparse_coo_tensor(
            torch.LongTensor(indices),
            torch.FloatTensor(values),
            torch.Size(norm_adjacency.shape),
            device=device,
        ).coalesce()

    def fit(
        self,
        ub_matrix: Union[sp.spmatrix, np.ndarray, torch.Tensor],
        user_embeds: Optional[Union[np.ndarray, torch.Tensor]] = None,
        bundle_embeds: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ):
        dense_ub = self._to_dense_tensor(ub_matrix)
        self.num_users, self.num_bundles = dense_ub.shape

        if self.denoise_model is None or self.diffusion_model is None:
            self._build_models(self.num_bundles)

        dataset = DiffusionDataset(dense_ub)
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size_train,
            shuffle=self.config.shuffle,
            num_workers=self.config.num_workers,
        )

        u_emb = None
        b_emb = None
        if user_embeds is not None:
            u_emb = torch.as_tensor(user_embeds, dtype=torch.float32, device=self.device)
        if bundle_embeds is not None:
            b_emb = torch.as_tensor(bundle_embeds, dtype=torch.float32, device=self.device)

        self.train_history = []
        for epoch in range(self.config.epochs):
            epoch_loss = 0.0
            epoch_elbo = 0.0
            epoch_blcc = 0.0
            n_batches = 0

            self.denoise_model.train(True)
            for batch_user_bundle, batch_user_index in loader:
                batch_user_bundle = batch_user_bundle.to(self.device)
                batch_user_index = batch_user_index.to(self.device)

                self.optimizer.zero_grad()
                elbo_loss, blcc_loss = self.diffusion_model.training_CBDM_losses(
                    self.denoise_model,
                    batch_user_bundle,
                    user_embeds=u_emb,
                    bundle_embeds=b_emb,
                    batch_user_index=batch_user_index,
                    use_blcc=self.config.use_blcc,
                )
                total_loss = elbo_loss + self.config.lambda_0 * blcc_loss
                total_loss.backward()
                self.optimizer.step()

                epoch_loss += float(total_loss.detach().cpu())
                epoch_elbo += float(elbo_loss.detach().cpu())
                epoch_blcc += float(blcc_loss.detach().cpu())
                n_batches += 1

            stats = {
                "epoch": epoch + 1,
                "loss": epoch_loss / max(n_batches, 1),
                "elbo_loss": epoch_elbo / max(n_batches, 1),
                "blcc_loss": epoch_blcc / max(n_batches, 1),
            }
            self.train_history.append(stats)
            if self.config.verbose:
                print(
                    f"[CBDM] epoch {stats['epoch']:03d}/{self.config.epochs:03d} | "
                    f"loss={stats['loss']:.6f} | elbo={stats['elbo_loss']:.6f} | blcc={stats['blcc_loss']:.6f}"
                )

        self._fitted = True
        return self

    @torch.no_grad()
    def rebuild_ub_matrix(
        self,
        ub_matrix: Union[sp.spmatrix, np.ndarray, torch.Tensor],
    ) -> sp.coo_matrix:
        if not self._fitted:
            raise RuntimeError("CBDM must be fitted before rebuild_ub_matrix()")

        dense_ub = self._to_dense_tensor(ub_matrix)
        dataset = DiffusionDataset(dense_ub)
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size_train,
            shuffle=False,
            num_workers=self.config.num_workers,
        )

        self.denoise_model.train(False)
        u_list, b_list, edge_list = [], [], []

        for batch_user_bundle, batch_user_index in loader:
            batch_user_bundle = batch_user_bundle.to(self.device)
            batch_user_index = batch_user_index.to(self.device)
            denoised_batch = self.diffusion_model.p_sample(
                self.denoise_model,
                batch_user_bundle,
                self.config.sampling_steps,
                self.config.sampling_noise,
            )
            _, indices = torch.topk(denoised_batch, k=self.config.rebuild_k, dim=1)
            for i in range(batch_user_index.shape[0]):
                user_id = int(batch_user_index[i].item())
                for j in range(indices.shape[1]):
                    bundle_id = int(indices[i, j].item())
                    u_list.append(user_id)
                    b_list.append(bundle_id)
                    edge_list.append(1.0)

        denoised_ub_mat = sp.coo_matrix(
            (np.array(edge_list, dtype=np.float32), (np.array(u_list), np.array(b_list))),
            shape=(dense_ub.shape[0], dense_ub.shape[1]),
            dtype=np.float32,
        )
        return denoised_ub_mat

    @torch.no_grad()
    def rebuild_outputs(
        self,
        ub_matrix: Union[sp.spmatrix, np.ndarray, torch.Tensor],
    ) -> Dict[str, Any]:
        denoised_ub_mat = self.rebuild_ub_matrix(ub_matrix)
        ub_propagation_graph = self.build_ub_propagation_graph(denoised_ub_mat, self.device)
        return {
            "denoised_ub_mat": denoised_ub_mat,
            "UB_propagation_graph": ub_propagation_graph,
        }

    def fit_transform(
        self,
        ub_matrix: Union[sp.spmatrix, np.ndarray, torch.Tensor],
        user_embeds: Optional[Union[np.ndarray, torch.Tensor]] = None,
        bundle_embeds: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        self.fit(ub_matrix, user_embeds=user_embeds, bundle_embeds=bundle_embeds)
        return self.rebuild_outputs(ub_matrix)

    def summary(self) -> Dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "num_users": self.num_users,
            "num_bundles": self.num_bundles,
            "fitted": self._fitted,
            "history": self.train_history,
        }


__all__ = [
    "CBDM",
    "CBDMConfig",
    "DEFAULT_IFASHION_CONFIG",
    "DiffusionDataset",
    "DNN",
    "GaussianDiffusion",
]
