"""Evaluation entry point for public LA-Net checkpoints."""

import argparse
import os

import pandas as pd
import torch
import yaml

from data.loader import build_dataloaders
from models import build_model
from utils import cal_metrics, resolve_device


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a public LA-Net checkpoint.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config used for evaluation.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a public LA-Net checkpoint dict.")
    parser.add_argument("--subject", type=int, required=True, help="1-based subject id.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Evaluation device, e.g. cuda:0 or cpu.")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader worker count.")
    parser.add_argument("--out_csv", type=str, default="", help="Optional path to save a single-row metrics CSV.")
    return parser.parse_args()


def forward_logits(model, x):
    output = model(x)
    if isinstance(output, dict):
        return output["logits"]
    if isinstance(output, tuple):
        return output[1]
    return output


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["model_name"] = "LANet"

    device = resolve_device(args.device)
    bundle = build_dataloaders(
        data_dir=config["data"]["dir"],
        dataset_type=config["data"]["dataset_type"],
        subject_id=args.subject,
        evaluate_mode=config["data"].get("evaluate_mode", "subject_dependent"),
        n_total=int(config["data"].get("n_subject", config["data"].get("n_subjects", 9))),
        batch_size=int(config["training"].get("batch_size", 72)),
        num_workers=args.num_workers,
        validate_ratio=float(config["training"].get("validate_ratio", 0.3)),
        use_bandpass_filter=bool(config.get("preprocessing", {}).get("use_bandpass_filter", False)),
        bandpass_low_hz=config.get("preprocessing", {}).get("bandpass_low_hz", 4.0),
        bandpass_high_hz=config.get("preprocessing", {}).get("bandpass_high_hz", 40.0),
        sampling_rate=config.get("preprocessing", {}).get("sampling_rate", None),
        bandpass_order=int(config.get("preprocessing", {}).get("bandpass_order", 5)),
        bandpass_filter_type=config.get("preprocessing", {}).get("bandpass_filter_type", "butter"),
        bandpass_filt_mode=config.get("preprocessing", {}).get("bandpass_filt_mode", "filtfilt"),
        bandpass_filt_allowance=float(config.get("preprocessing", {}).get("bandpass_filt_allowance", 2.0)),
        normalization_mode=config.get("preprocessing", {}).get("normalization_mode", "global"),
        validation_split_strategy=config["data"].get("validation_split_strategy", "tail"),
        pin_memory=device.type == "cuda",
    )

    model = build_model("LANet", config["data"]["dataset_type"], config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    predictions = []
    targets = []
    with torch.no_grad():
        for batch in bundle["test_loader"]:
            inputs = batch[0].float().to(device)
            label = batch[1].long().to(device)
            logits = forward_logits(model, inputs)
            pred = logits.argmax(dim=1)
            predictions.append(pred.cpu())
            targets.append(label.cpu())

    y_pred = torch.cat(predictions).numpy()
    y_true = torch.cat(targets).numpy()
    metrics = cal_metrics(y_true, y_pred)
    row = {
        "subject_id": int(args.subject),
        "accuracy": metrics["accuracy"] * 100.0,
        "precision": metrics["precision"] * 100.0,
        "recall": metrics["recall"] * 100.0,
        "f1": metrics["f1"] * 100.0,
        "kappa": metrics["kappa"] * 100.0,
        "checkpoint": os.path.abspath(args.checkpoint),
    }

    if args.out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
        pd.DataFrame([row]).to_csv(args.out_csv, index=False)

    print(
        f"Subject {args.subject} | "
        f"acc: {row['accuracy']:.2f}% | "
        f"kappa: {row['kappa']:.2f}%"
    )


if __name__ == "__main__":
    main()
