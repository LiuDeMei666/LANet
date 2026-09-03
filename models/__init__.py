"""Public model registry for the LA-Net release."""

from .dataset_meta import number_class_channel


def _is_loso_mode(cfg):
    evaluate_mode = str(cfg.get("data", {}).get("evaluate_mode", ""))
    return evaluate_mode.upper().startswith("LOSO")


def _resolve_dropout_rate(cfg, default_non_loso=0.5, default_loso=0.25):
    model_cfg = cfg.get("model", {})
    value = model_cfg.get("eegnet1_dropout_rate", "auto")
    if value in {"auto", None}:
        return default_loso if _is_loso_mode(cfg) else default_non_loso
    return float(value)


def _build_lanet(dataset_type, cfg):
    from .lanet import LANet

    model_cfg = cfg.get("model", {})
    _, number_channel = number_class_channel(dataset_type)
    return LANet(
        database_type=dataset_type,
        input_samples=int(model_cfg.get("input_samples", cfg.get("data", {}).get("input_samples", 1000))),
        eeg1_f1=int(model_cfg.get("eegnet1_f1", 8)),
        eeg1_kernel_size=int(model_cfg.get("eegnet1_kernel_size", 64)),
        eeg1_d=int(model_cfg.get("eegnet1_d", 2)),
        eeg1_pooling_size1=int(model_cfg.get("eegnet1_pool_size1", model_cfg.get("tcr_pool_size1", 4) or 4)),
        eeg1_pooling_size2=int(model_cfg.get("eegnet1_pool_size2", model_cfg.get("tcr_pool_size2", 8) or 8)),
        eeg1_dropout_rate=_resolve_dropout_rate(cfg),
        flatten_eeg1=model_cfg.get("flatten_eeg1", model_cfg.get("flatten_eegnet1", None)),
        number_channel=number_channel,
        tcr_reduction=int(model_cfg.get("tcr_reduction", model_cfg.get("tsa_reduction", 4))),
        tcr_time_kernel=int(model_cfg.get("tcr_time_kernel", model_cfg.get("tsa_time_kernel_v2", 9))),
        tcr_gate_scale=float(model_cfg.get("tcr_gate_scale", model_cfg.get("tsa_gate_scale", 0.5))),
        learnable_tcr_gate_scale=bool(
            model_cfg.get(
                "learnable_tcr_gate_scale",
                model_cfg.get("learnable_tsa_gate_scale", False),
            )
        ),
        centered_tcr_gate=bool(model_cfg.get("centered_tcr_gate", model_cfg.get("centered_axis_gate", False))),
        asym_branch_scale=float(model_cfg.get("asym_branch_scale", 0.15)),
        learnable_asym_branch_scale=bool(model_cfg.get("learnable_asym_branch_scale", False)),
        asym_branch_type=str(model_cfg.get("asym_branch_type", "legacy")),
    )


MODEL_REGISTRY = {
    "LANet": _build_lanet,
}


def build_model(model_name, dataset_type, cfg):
    if model_name not in MODEL_REGISTRY:
        supported = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(f"Unknown model_name={model_name!r}. Supported models: {supported}")
    return MODEL_REGISTRY[model_name](dataset_type, cfg)


__all__ = ["MODEL_REGISTRY", "build_model", "number_class_channel"]
