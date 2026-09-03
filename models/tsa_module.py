"""Temporal-spatial attention module used by LA-Net."""

import math

import torch
from torch import Tensor, nn


def _sigmoid_inverse(value: float, eps: float = 1.0e-4) -> float:
    value = min(max(float(value), eps), 1.0 - eps)
    return math.log(value / (1.0 - value))


class LearnableSigmoidScale(nn.Module):
    def __init__(self, initial_scale: float, learnable: bool = False, max_scale: float = 1.0):
        super().__init__()
        self.learnable = bool(learnable)
        self.max_scale = float(max_scale)
        initial_scale = float(initial_scale)
        clipped_ratio = initial_scale / self.max_scale
        if self.learnable:
            self.scale_logit = nn.Parameter(torch.tensor(_sigmoid_inverse(clipped_ratio), dtype=torch.float32))
        else:
            self.register_buffer("fixed_scale", torch.tensor(initial_scale, dtype=torch.float32))

    def forward(self) -> Tensor:
        if self.learnable:
            return self.max_scale * torch.sigmoid(self.scale_logit)
        return self.fixed_scale


class TSAModule(nn.Module):
    """Residual temporal-spatial attention block.

    The implementation intentionally preserves the validated public behavior of the
    historical TSA gate while exposing the public name `TSAModule`.
    """

    def __init__(
        self,
        channels: int,
        time_kernel: int = 9,
        reduction: int = 4,
        gate_scale: float = 0.5,
        learnable_gate_scale: bool = False,
        centered_gate: bool = False,
    ):
        super().__init__()
        hidden = max(1, int(channels) // int(reduction))
        self.centered_gate = bool(centered_gate)
        self.last_explain = None
        self.gate_scale = LearnableSigmoidScale(
            gate_scale,
            learnable=learnable_gate_scale,
            max_scale=2.0,
        )
        self.channel_mlp = nn.Sequential(
            nn.Linear(int(channels), hidden),
            nn.GELU(),
            nn.Linear(hidden, int(channels)),
        )
        self.temporal_conv = nn.Conv1d(
            1,
            1,
            kernel_size=int(time_kernel),
            padding=int(time_kernel) // 2,
            bias=True,
        )

    def get_last_explain(self):
        return self.last_explain

    def forward(self, x: Tensor) -> Tensor:
        batch_size, channels, _, timesteps = x.shape
        channel_context = x.mean(dim=(2, 3))
        channel_gate = torch.sigmoid(self.channel_mlp(channel_context)).view(batch_size, channels, 1, 1)

        temporal_context = x.mean(dim=(1, 2), keepdim=False).unsqueeze(1)
        temporal_gate = torch.sigmoid(self.temporal_conv(temporal_context)).view(batch_size, 1, 1, timesteps)

        gate = channel_gate * temporal_gate
        if self.centered_gate:
            gate = gate - 0.5
        scale = self.gate_scale()
        self.last_explain = {
            "channel_gate": channel_gate.detach(),
            "temporal_gate": temporal_gate.detach(),
            "scale": scale.detach(),
        }
        return x + scale * x * gate


__all__ = ["LearnableSigmoidScale", "TSAModule"]
