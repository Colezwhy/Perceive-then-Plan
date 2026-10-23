"""Geometry feature merger and gated FiLM fusion (adapted from VG-LLM, https://github.com/LaVi-Lab/VG-LLM)."""

import math

import torch
import torch.nn as nn


class GeometryFeatureMerger(nn.Module):
    """Merges 2x2 VGGT patch features and projects them to the Qwen visual width."""

    def __init__(self, output_dim: int, hidden_dim: int, context_dim: int, spatial_merge_size: int = 2):
        super().__init__()
        self.input_dim = context_dim * (spatial_merge_size**2)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.merge_size = spatial_merge_size

        self.ln_q = nn.LayerNorm(context_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [n_image, h_patch, w_patch, dim] -> [n_image, h_patch // 2, w_patch // 2, output_dim]."""
        n_image, h_patch, w_patch, dim = x.shape
        m = self.merge_size
        x = x[:, : h_patch // m * m, : w_patch // m * m, :]
        x = x.reshape(n_image, h_patch // m, m, w_patch // m, m, dim)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = self.mlp(self.ln_q(x).view(-1, self.input_dim))
        return x.reshape(n_image, h_patch // m, w_patch // m, -1)


class FeatureFusionModule(nn.Module):
    """F = g * [(1 + dgamma) * F_img + beta] + (1 - g) * F_img, with dgamma, beta, g predicted from geometry."""

    def __init__(self, hidden_size: int, fusion_method: str = "film"):
        super().__init__()
        if fusion_method != "film":
            raise ValueError(f"Only fusion_method='film' is supported, got {fusion_method!r}")
        self.fusion_method = fusion_method
        self.hidden_size = hidden_size

        self.to_gamma = nn.Linear(hidden_size, hidden_size)
        self.to_beta = nn.Linear(hidden_size, hidden_size)
        self.to_gate = nn.Conv2d(in_channels=hidden_size, out_channels=1, kernel_size=1)
        # Unused; kept for checkpoint compatibility.
        self.img_norm = nn.LayerNorm(hidden_size)
        self.geo_norm = nn.LayerNorm(hidden_size)
        self._init_film()

    def _init_film(self):
        nn.init.zeros_(self.to_gamma.weight)
        nn.init.zeros_(self.to_gamma.bias)
        nn.init.zeros_(self.to_beta.weight)
        nn.init.zeros_(self.to_beta.bias)
        nn.init.kaiming_uniform_(self.to_gate.weight, a=math.sqrt(5))
        nn.init.normal_(self.to_gate.bias, mean=0.0, std=0.01)

    def forward(self, features_2d: torch.Tensor, features_3d: torch.Tensor) -> torch.Tensor:
        """Both inputs: [n_image, h, w, hidden_size]."""
        features_3d = self.geo_norm(features_3d)
        delta_gamma = self.to_gamma(features_3d)
        beta = self.to_beta(features_3d)
        gate = torch.sigmoid(self.to_gate(features_3d.permute(0, 3, 1, 2))).permute(0, 2, 3, 1)
        modulated = (1 + delta_gamma) * features_2d + beta
        return gate * modulated + (1.0 - gate) * features_2d
