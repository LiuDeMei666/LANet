"""LA-Net model definition."""

from typing import List

import torch
from einops.layers.torch import Rearrange
from torch import Tensor, nn

from .dataset_meta import number_class_channel
from .heads import ClassificationHead
from .tsa_module import LearnableSigmoidScale, TSAModule


DATASET_HEMISPHERE_LAYOUTS = {
    "A": {
        "order": [
            "Fz",
            "FC3",
            "FC1",
            "FCz",
            "FC2",
            "FC4",
            "C5",
            "C3",
            "C1",
            "Cz",
            "C2",
            "C4",
            "C6",
            "CP3",
            "CP1",
            "CPz",
            "CP2",
            "CP4",
            "P1",
            "Pz",
            "P2",
            "POz",
        ],
        "left": {"FC3", "FC1", "C5", "C3", "C1", "CP3", "CP1", "P1"},
        "right": {"FC2", "FC4", "C2", "C4", "C6", "CP2", "CP4", "P2"},
        "mid": {"Fz", "FCz", "Cz", "CPz", "Pz", "POz"},
    },
    "B": {
        "order": ["C3", "Cz", "C4"],
        "left": {"C3"},
        "right": {"C4"},
        "mid": {"Cz"},
    },
}


def _indices(channel_names: List[str], selected) -> List[int]:
    return [idx for idx, name in enumerate(channel_names) if name in selected]


def _resolve_hemisphere_indices(dataset_type: str, number_channel: int):
    layout = DATASET_HEMISPHERE_LAYOUTS.get(str(dataset_type))
    if layout is None:
        supported = ", ".join(sorted(DATASET_HEMISPHERE_LAYOUTS.keys()))
        raise ValueError(
            f"LSAModule supports dataset_type in {{{supported}}}, got {dataset_type!r}."
        )
    channel_order = list(layout["order"])
    if len(channel_order) != int(number_channel):
        raise ValueError(
            f"Hemisphere layout for dataset_type={dataset_type!r} expects "
            f"{len(channel_order)} channels, got {number_channel}."
        )
    left_idx = _indices(channel_order, layout["left"])
    right_idx = _indices(channel_order, layout["right"])
    mid_idx = _indices(channel_order, layout["mid"])
    if not left_idx or not right_idx:
        raise ValueError("Hemisphere layout must contain both left and right channels.")
    return left_idx, right_idx, mid_idx


class LSAModule(nn.Module):
    """Lateralized spatial asymmetry module."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dataset_type: str,
        number_channel: int,
        asym_scale: float = 0.15,
        learnable_asym_scale: bool = False,
        asym_branch_type: str = "legacy",
    ):
        super().__init__()
        branch_type = str(asym_branch_type or "legacy")
        if branch_type not in {"legacy", "diff_only"}:
            raise ValueError(
                f"Unknown asym_branch_type={branch_type!r}. Available: legacy, diff_only"
            )
        self.asym_branch_type = branch_type
        self.out_channels = int(out_channels)
        self.last_explain = None
        self.asym_scale = LearnableSigmoidScale(asym_scale, learnable=learnable_asym_scale)
        self.left_idx, self.right_idx, self.mid_idx = _resolve_hemisphere_indices(
            dataset_type,
            number_channel,
        )
        kernel_size = (3, 1) if branch_type == "legacy" else (1, 1)
        self.spatial_asym_conv = nn.Conv2d(
            int(in_channels),
            int(out_channels),
            kernel_size=kernel_size,
            stride=(1, 1),
            groups=int(in_channels),
            bias=False,
        )

    def get_last_explain(self):
        return self.last_explain

    def forward(self, x: Tensor) -> Tensor:
        left = x[:, :, self.left_idx, :].mean(dim=2, keepdim=True)
        right = x[:, :, self.right_idx, :].mean(dim=2, keepdim=True)
        if self.mid_idx:
            mid = x[:, :, self.mid_idx, :].mean(dim=2, keepdim=True)
        else:
            mid = 0.5 * (left + right)
        diff = left - right + 0.5 * mid
        if self.asym_branch_type == "legacy":
            asym_input = torch.cat([left, right, diff], dim=2)
        else:
            asym_input = diff
        scale = self.asym_scale()
        self.last_explain = {
            "left": left.detach(),
            "right": right.detach(),
            "mid": mid.detach(),
            "asymmetry": diff.detach(),
            "scale": scale.detach(),
        }
        return scale * self.spatial_asym_conv(asym_input)


class LANetBackbone(nn.Module):
    def __init__(
        self,
        f1: int = 8,
        kernel_size: int = 64,
        d: int = 2,
        pooling_size1: int = 4,
        pooling_size2: int = 8,
        dropout_rate: float = 0.3,
        dataset_type: str = "A",
        number_channel: int = 22,
        tcr_reduction: int = 4,
        tcr_time_kernel: int = 9,
        tcr_gate_scale: float = 0.5,
        learnable_tcr_gate_scale: bool = False,
        centered_tcr_gate: bool = False,
        asym_branch_scale: float = 0.15,
        learnable_asym_branch_scale: bool = False,
        asym_branch_type: str = "legacy",
    ):
        super().__init__()
        f2 = int(d) * int(f1)
        self.temporal_conv = nn.Conv2d(1, int(f1), (1, int(kernel_size)), (1, 1), padding="same", bias=False)
        self.temporal_bn = nn.BatchNorm2d(int(f1))

        self.spatial_conv = nn.Conv2d(
            int(f1),
            f2,
            (int(number_channel), 1),
            (1, 1),
            groups=int(f1),
            padding="valid",
            bias=False,
        )
        self.lsa_module = LSAModule(
            int(f1),
            f2,
            dataset_type=dataset_type,
            number_channel=number_channel,
            asym_scale=asym_branch_scale,
            learnable_asym_scale=learnable_asym_branch_scale,
            asym_branch_type=asym_branch_type,
        )
        self.spatial_bn = nn.BatchNorm2d(f2)
        self.spatial_activation = nn.ELU()
        self.spatial_pool = nn.AvgPool2d((1, int(pooling_size1)))
        self.spatial_drop = nn.Dropout(float(dropout_rate))

        self.temporal_refine = nn.Conv2d(f2, f2, (1, 16), padding="same", bias=False)
        self.temporal_refine_bn = nn.BatchNorm2d(f2)
        self.temporal_refine_activation = nn.ELU()
        self.temporal_refine_pool = nn.AvgPool2d((1, int(pooling_size2)))
        self.temporal_refine_drop = nn.Dropout(float(dropout_rate))
        self.tsa_module = TSAModule(
            channels=f2,
            time_kernel=tcr_time_kernel,
            reduction=tcr_reduction,
            gate_scale=tcr_gate_scale,
            learnable_gate_scale=learnable_tcr_gate_scale,
            centered_gate=centered_tcr_gate,
        )
        self.projection = Rearrange("b e h w -> b (h w) e")

    def get_last_explain(self):
        return {
            "lsa_module": self.lsa_module.get_last_explain(),
            "tsa_module": self.tsa_module.get_last_explain(),
        }

    def forward(self, x: Tensor) -> Tensor:
        x = self.temporal_bn(self.temporal_conv(x))
        spatial_main = self.spatial_conv(x)
        spatial_asym = self.lsa_module(x)
        x = self.spatial_bn(spatial_main + spatial_asym)
        x = self.spatial_activation(x)
        x = self.spatial_pool(x)
        x = self.spatial_drop(x)
        x = self.temporal_refine(x)
        x = self.temporal_refine_bn(x)
        x = self.temporal_refine_activation(x)
        x = self.temporal_refine_pool(x)
        x = self.temporal_refine_drop(x)
        x = self.tsa_module(x)
        return self.projection(x)


class LANet(nn.Module):
    def __init__(
        self,
        database_type: str = "A",
        input_samples: int = 1000,
        eeg1_f1: int = 8,
        eeg1_kernel_size: int = 64,
        eeg1_d: int = 2,
        eeg1_pooling_size1: int = 4,
        eeg1_pooling_size2: int = 8,
        eeg1_dropout_rate: float = 0.3,
        flatten_eeg1=None,
        number_channel=None,
        tcr_reduction: int = 4,
        tcr_time_kernel: int = 9,
        tcr_gate_scale: float = 0.5,
        learnable_tcr_gate_scale: bool = False,
        centered_tcr_gate: bool = False,
        asym_branch_scale: float = 0.15,
        learnable_asym_branch_scale: bool = False,
        asym_branch_type: str = "legacy",
        **kwargs,
    ):
        super().__init__()
        self.database_type = str(database_type)
        self.number_class, inferred_channels = number_class_channel(self.database_type)
        self.number_channel = int(number_channel or inferred_channels)
        self.input_samples = int(input_samples)

        self.cnn = LANetBackbone(
            f1=eeg1_f1,
            kernel_size=eeg1_kernel_size,
            d=eeg1_d,
            pooling_size1=eeg1_pooling_size1,
            pooling_size2=eeg1_pooling_size2,
            dropout_rate=eeg1_dropout_rate,
            dataset_type=self.database_type,
            number_channel=self.number_channel,
            tcr_reduction=tcr_reduction,
            tcr_time_kernel=tcr_time_kernel,
            tcr_gate_scale=tcr_gate_scale,
            learnable_tcr_gate_scale=learnable_tcr_gate_scale,
            centered_tcr_gate=centered_tcr_gate,
            asym_branch_scale=asym_branch_scale,
            learnable_asym_branch_scale=learnable_asym_branch_scale,
            asym_branch_type=asym_branch_type,
        )
        self.flatten = nn.Flatten()
        self.flatten_eeg1 = int(flatten_eeg1) if flatten_eeg1 is not None else self._infer_flatten_dim()
        self.classification = ClassificationHead(self.flatten_eeg1, self.number_class)

    def _infer_flatten_dim(self) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.number_channel, self.input_samples)
            tokens = self.cnn(dummy)
            return int(self.flatten(tokens).shape[1])

    def get_last_explain(self):
        return self.cnn.get_last_explain()

    def extract_token_features(self, x: Tensor) -> Tensor:
        return self.cnn(x)

    def extract_features(self, x: Tensor):
        token_features = self.extract_token_features(x)
        penultimate = self.flatten(token_features)
        return penultimate, token_features

    def classify(self, penultimate: Tensor) -> Tensor:
        return self.classification(penultimate)

    def forward(self, x: Tensor, return_features: bool = False, return_explain: bool = False):
        penultimate, token_features = self.extract_features(x)
        logits = self.classify(penultimate)
        if return_features or return_explain:
            output = {
                "logits": logits,
                "features": penultimate,
                "token_features": token_features,
            }
            if return_explain:
                output["explain"] = self.get_last_explain()
            return output
        return token_features, logits


__all__ = ["DATASET_HEMISPHERE_LAYOUTS", "LSAModule", "LANetBackbone", "LANet"]
