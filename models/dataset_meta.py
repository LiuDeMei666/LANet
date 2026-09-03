"""Dataset metadata for the public LA-Net release."""

DATASET_META = {
    "A": {
        "name": "BCI Competition IV-2a",
        "n_classes": 4,
        "n_channels": 22,
        "n_subjects": 9,
        "input_samples": 1000,
    },
    "B": {
        "name": "BCI Competition IV-2b",
        "n_classes": 2,
        "n_channels": 3,
        "n_subjects": 9,
        "input_samples": 1000,
    },
}


def get_dataset_meta(dataset_type):
    key = str(dataset_type).strip()
    if key not in DATASET_META:
        supported = ", ".join(sorted(DATASET_META.keys()))
        raise ValueError(f"Unknown dataset_type={dataset_type!r}. Supported datasets: {supported}")
    return DATASET_META[key]


def number_class_channel(dataset_type):
    meta = get_dataset_meta(dataset_type)
    return meta["n_classes"], meta["n_channels"]


__all__ = ["DATASET_META", "get_dataset_meta", "number_class_channel"]
