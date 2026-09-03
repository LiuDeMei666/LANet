"""Training entry point for the public LA-Net release."""

import argparse
import datetime
import os

import pandas as pd
import yaml

from trainer import Trainer
from utils import ensure_dir, resolve_device, save_json, set_random_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train LA-Net on BCI Competition IV-2a/2b.")
    parser.add_argument("--config", type=str, required=True, help="Path to an official YAML config.")
    parser.add_argument("--subject", type=int, required=True, help="1-based subject id.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Training device, e.g. cuda:0 or cpu.")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader worker count.")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed override.")
    parser.add_argument("--out_dir", type=str, default="", help="Optional explicit output directory.")
    return parser.parse_args()


def resolve_seed(config, subject_id, seed_override=None):
    if seed_override is not None:
        return int(seed_override)
    training_cfg = config.get("training", {})
    subject_seeds = training_cfg.get("subject_seeds")
    if isinstance(subject_seeds, (list, tuple)) and len(subject_seeds) >= int(subject_id):
        return int(subject_seeds[int(subject_id) - 1])
    return int(training_cfg.get("random_seed", 966))


def resolve_output_dir(config, subject_id, explicit_out_dir=""):
    if explicit_out_dir:
        return os.path.abspath(explicit_out_dir)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_type = config["data"]["dataset_type"]
    evaluate_mode = config["data"].get("evaluate_mode", "unknown")
    model_name = config.get("model_name", "LANet")
    return os.path.abspath(
        os.path.join(
            "runs",
            dataset_type,
            model_name,
            evaluate_mode,
            timestamp,
            f"subject_{int(subject_id):02d}",
        )
    )


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    config["model_name"] = "LANet"
    seed = resolve_seed(config, args.subject, args.seed)
    set_random_seed(seed)
    device = resolve_device(args.device)
    out_dir = resolve_output_dir(config, args.subject, args.out_dir)

    ensure_dir(out_dir)
    with open(os.path.join(out_dir, "config.yaml"), "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)

    trainer = Trainer(
        config=config,
        subject_id=args.subject,
        output_dir=out_dir,
        device=device,
        num_workers=args.num_workers,
    )
    result = trainer.fit()

    summary = {
        "subject_id": int(args.subject),
        "seed": int(seed),
        "device": str(device),
        "output_dir": out_dir,
        "metrics": result["metrics"],
        "checkpoint_path": result["checkpoint_path"],
        "data_info": result["data_info"],
    }
    save_json(os.path.join(out_dir, "summary.json"), summary)
    pd.DataFrame([result["metrics"]]).to_csv(os.path.join(out_dir, "metrics.csv"), index=False)

    print(f"Training finished for subject {args.subject}.")
    print(f"Checkpoint: {result['checkpoint_path']}")
    print(
        "Metrics | "
        f"acc: {result['metrics']['accuracy']:.2f}% | "
        f"kappa: {result['metrics']['kappa']:.2f}% | "
        f"best_epoch: {result['metrics']['best_epoch']}"
    )


if __name__ == "__main__":
    main()
