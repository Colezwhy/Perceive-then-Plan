"""Frozen VGGT geometry encoder (adapted from VG-LLM, https://github.com/LaVi-Lab/VG-LLM)."""

import torch
import torch.nn as nn

from ..third_party.vggt.models.vggt import VGGT


class VGGTEncoder(nn.Module):
    """Wraps the VGGT aggregator and returns per-patch geometry tokens.

    Only the aggregator is built (camera, depth, point and track heads are disabled). The encoder is
    always frozen and runs without gradients.
    """

    patch_size = 14
    feature_dim = 2048

    def __init__(self):
        super().__init__()
        self.vggt = VGGT(enable_camera=False, enable_point=False, enable_depth=False, enable_track=False)
        self._freeze()

    def _freeze(self):
        for param in self.vggt.parameters():
            param.requires_grad = False

    def load_model(self, model_path: str) -> None:
        """Load pretrained VGGT weights, e.g. ``facebook/VGGT-1B``."""
        self.vggt = VGGT.from_pretrained(
            model_path, enable_camera=False, enable_point=False, enable_depth=False, enable_track=False
        )
        self._freeze()

    def get_feature_dim(self) -> int:
        return self.feature_dim

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """images: [n_frames, 3, H, W] in [0, 1] -> [n_frames, (H/14) * (W/14), 2048]."""
        self.vggt.eval()
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.autocast("cuda", dtype=dtype):
            aggregated_tokens_list, patch_start_idx = self.vggt.aggregator(images[None])
            # Second-to-last aggregator layer, image patches only (camera/register tokens dropped).
            features = aggregated_tokens_list[-2][0, :, patch_start_idx:]
        return features

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.encode(images)

    @property
    def device(self):
        param = next(self.parameters(), None)
        if param is not None:
            return param.device
        buffer = next(self.buffers(), None)
        if buffer is not None:
            return buffer.device
        return torch.device("cpu")
