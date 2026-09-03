"""Common utilities for the public LA-Net release."""

import json
import os
import random

import numpy as np
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, precision_score, recall_score


def set_random_seed(seed: int):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: str):
    requested = str(device).strip().lower()
    if requested == "cpu":
        return torch.device("cpu")
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device)
    if requested.isdigit() and torch.cuda.is_available():
        return torch.device(f"cuda:{requested}")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def cal_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
    }


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


def save_json(path: str, payload):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


__all__ = ["cal_metrics", "ensure_dir", "resolve_device", "save_json", "set_random_seed"]
