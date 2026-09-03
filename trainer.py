"""Minimal trainer for the public LA-Net release."""

import os

import numpy as np
import pandas as pd
import torch
from torch import nn

from data.loader import build_dataloaders
from models import build_model, number_class_channel
from utils import cal_metrics, ensure_dir


class Trainer:
    def __init__(self, config, subject_id, output_dir, device, num_workers=0):
        self.config = config
        self.subject_id = int(subject_id)
        self.output_dir = os.path.abspath(output_dir)
        self.device = device
        self.num_workers = int(num_workers)

        self.data_cfg = config.get("data", {})
        self.model_cfg = config.get("model", {})
        self.training_cfg = config.get("training", {})
        self.preprocessing_cfg = config.get("preprocessing", {})

        self.dataset_type = str(self.data_cfg["dataset_type"])
        self.data_dir = self.data_cfg["dir"]
        self.evaluate_mode = str(self.data_cfg.get("evaluate_mode", "subject_dependent"))
        self.n_subjects = int(self.data_cfg.get("n_subject", self.data_cfg.get("n_subjects", 9)))
        self.model_name = str(config.get("model_name", "LANet"))

        self.batch_size = int(self.training_cfg.get("batch_size", 72))
        self.epochs = int(self.training_cfg.get("epochs", 500))
        self.learning_rate = float(self.training_cfg.get("learning_rate", 1e-3))
        self.validate_ratio = float(self.training_cfg.get("validate_ratio", 0.3))
        self.number_augmentation = int(self.training_cfg.get("n_aug", 0))
        self.number_seg = int(self.training_cfg.get("n_seg", 1))
        self.label_smoothing = float(self.training_cfg.get("label_smoothing", 0.0))

        self.number_class, self.number_channel = number_class_channel(self.dataset_type)

        self.model = build_model(self.model_name, self.dataset_type, config).to(self.device)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.paths = {
            "root": ensure_dir(self.output_dir),
            "checkpoints": ensure_dir(os.path.join(self.output_dir, "checkpoints")),
            "metrics": ensure_dir(os.path.join(self.output_dir, "metrics")),
            "predictions": ensure_dir(os.path.join(self.output_dir, "predictions")),
            "curves": ensure_dir(os.path.join(self.output_dir, "curves")),
            "meta": ensure_dir(os.path.join(self.output_dir, "meta")),
        }
        self.checkpoint_path = os.path.join(self.paths["checkpoints"], f"subject_{self.subject_id:02d}.pt")

    def _class_domain_groups(self, label, domain_ids=None):
        label = np.asarray(label).reshape(-1)
        if domain_ids is None:
            return {cls: [np.where(label == cls + 1)[0]] for cls in range(self.number_class)}

        domain_ids = np.asarray(domain_ids).reshape(-1)
        groups = {}
        for cls in range(self.number_class):
            cls_groups = []
            cls_mask = label == (cls + 1)
            for domain_id in np.unique(domain_ids[cls_mask]):
                idx = np.where(cls_mask & (domain_ids == domain_id))[0]
                if idx.size > 0:
                    cls_groups.append(idx)
            groups[cls] = cls_groups
        return groups

    def interaug(self, full_data, full_label, domain_ids=None):
        if self.number_augmentation <= 0:
            return None, None

        aug_data = []
        aug_label = []
        n_records = self.number_augmentation * int(self.batch_size / self.number_class)
        time_len = int(full_data.shape[-1])
        n_seg_points = time_len // self.number_seg
        if n_seg_points <= 0:
            raise ValueError(f"number_seg={self.number_seg} is too large for time_len={time_len}")

        class_groups = self._class_domain_groups(full_label, domain_ids)
        for cls in range(self.number_class):
            pools = [idx for idx in class_groups.get(cls, []) if len(idx) > 0]
            if not pools:
                continue
            tmp_aug = np.zeros((n_records, 1, self.number_channel, time_len), dtype=np.float32)
            for row in range(n_records):
                pool_idx = pools[np.random.randint(0, len(pools))]
                rand_idx = np.random.choice(pool_idx, size=self.number_seg, replace=True)
                for seg in range(self.number_seg):
                    start = seg * n_seg_points
                    end = time_len if seg == self.number_seg - 1 else (seg + 1) * n_seg_points
                    tmp_aug[row, :, :, start:end] = full_data[rand_idx[seg], :, :, start:end]
            aug_data.append(tmp_aug)
            aug_label.append(np.full((n_records,), cls + 1, dtype=np.int64))

        if not aug_data:
            return None, None
        aug_data = np.concatenate(aug_data, axis=0)
        aug_label = np.concatenate(aug_label, axis=0)
        shuffle_index = np.random.permutation(len(aug_data))
        aug_data = aug_data[shuffle_index]
        aug_label = aug_label[shuffle_index]
        return (
            torch.from_numpy(aug_data).float().to(self.device),
            torch.from_numpy(aug_label - 1).long().to(self.device),
        )

    def _forward_logits(self, x):
        output = self.model(x)
        if isinstance(output, dict):
            return output["logits"]
        if isinstance(output, tuple):
            return output[1]
        return output

    def _evaluate_loader(self, loader):
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        predictions = []
        probabilities = []
        targets_all = []

        with torch.no_grad():
            for batch in loader:
                inputs = batch[0].float().to(self.device)
                targets = batch[1].long().to(self.device)
                logits = self._forward_logits(inputs)
                loss = self.criterion(logits, targets)
                probs = torch.softmax(logits, dim=1)
                preds = probs.argmax(dim=1)

                batch_size = int(targets.size(0))
                total_loss += float(loss.item()) * batch_size
                total_correct += int((preds == targets).sum().item())
                total_samples += batch_size
                predictions.append(preds.cpu())
                probabilities.append(probs.cpu())
                targets_all.append(targets.cpu())

        mean_loss = total_loss / float(max(total_samples, 1))
        accuracy = total_correct / float(max(total_samples, 1))
        return {
            "loss": mean_loss,
            "accuracy": accuracy,
            "predictions": torch.cat(predictions) if predictions else torch.empty(0, dtype=torch.long),
            "probabilities": torch.cat(probabilities) if probabilities else torch.empty(0),
            "targets": torch.cat(targets_all) if targets_all else torch.empty(0, dtype=torch.long),
        }

    def _save_checkpoint(self, epoch, val_metrics, bundle):
        checkpoint = {
            "model_name": self.model_name,
            "dataset_type": self.dataset_type,
            "subject_id": self.subject_id,
            "evaluate_mode": self.evaluate_mode,
            "epoch": int(epoch),
            "state_dict": self.model.state_dict(),
            "config": self.config,
            "val_loss": float(val_metrics["loss"]),
            "val_accuracy": float(val_metrics["accuracy"]),
            "source_mean": bundle["source_mean"],
            "source_std": bundle["source_std"],
            "data_info": bundle["info"],
        }
        torch.save(checkpoint, self.checkpoint_path)

    def fit(self):
        bundle = build_dataloaders(
            data_dir=self.data_dir,
            dataset_type=self.dataset_type,
            subject_id=self.subject_id,
            evaluate_mode=self.evaluate_mode,
            n_total=self.n_subjects,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            validate_ratio=self.validate_ratio,
            use_bandpass_filter=bool(self.preprocessing_cfg.get("use_bandpass_filter", False)),
            bandpass_low_hz=self.preprocessing_cfg.get("bandpass_low_hz", 4.0),
            bandpass_high_hz=self.preprocessing_cfg.get("bandpass_high_hz", 40.0),
            sampling_rate=self.preprocessing_cfg.get("sampling_rate", None),
            bandpass_order=int(self.preprocessing_cfg.get("bandpass_order", 5)),
            bandpass_filter_type=self.preprocessing_cfg.get("bandpass_filter_type", "butter"),
            bandpass_filt_mode=self.preprocessing_cfg.get("bandpass_filt_mode", "filtfilt"),
            bandpass_filt_allowance=float(self.preprocessing_cfg.get("bandpass_filt_allowance", 2.0)),
            normalization_mode=self.preprocessing_cfg.get("normalization_mode", "global"),
            validation_split_strategy=self.data_cfg.get("validation_split_strategy", "tail"),
            pin_memory=self.device.type == "cuda",
        )

        history = []
        best_val_loss = float("inf")
        best_epoch = -1
        aug_data = bundle["aug_train_data"]
        aug_label = bundle["aug_train_label"]
        aug_domain_ids = bundle["aug_train_domain_ids"]

        for epoch in range(self.epochs):
            self.model.train()
            train_loss_sum = 0.0
            train_correct = 0
            train_samples = 0

            for batch in bundle["train_loader"]:
                inputs = batch[0].float().to(self.device)
                targets = batch[1].long().to(self.device)
                aug_batch, aug_targets = self.interaug(aug_data, aug_label, aug_domain_ids)
                if aug_batch is not None:
                    cls_input = torch.cat([inputs, aug_batch], dim=0)
                    cls_target = torch.cat([targets, aug_targets], dim=0)
                else:
                    cls_input = inputs
                    cls_target = targets

                logits = self._forward_logits(cls_input)
                loss = self.criterion(logits, cls_target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                preds = logits.argmax(dim=1)
                train_loss_sum += float(loss.item()) * int(cls_target.size(0))
                train_correct += int((preds == cls_target).sum().item())
                train_samples += int(cls_target.size(0))

            train_loss = train_loss_sum / float(max(train_samples, 1))
            train_acc = train_correct / float(max(train_samples, 1))
            val_metrics = self._evaluate_loader(bundle["val_loader"])

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": float(val_metrics["loss"]),
                    "val_acc": float(val_metrics["accuracy"]),
                }
            )

            if float(val_metrics["loss"]) < best_val_loss:
                best_val_loss = float(val_metrics["loss"])
                best_epoch = epoch
                self._save_checkpoint(epoch, val_metrics, bundle)

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)

        test_metrics = self._evaluate_loader(bundle["test_loader"])
        y_true = test_metrics["targets"].numpy()
        y_pred = test_metrics["predictions"].numpy()
        metrics = cal_metrics(y_true, y_pred)

        history_df = pd.DataFrame(history)
        history_df.to_csv(os.path.join(self.paths["curves"], f"process_train_subject_{self.subject_id - 1}.csv"), index=False)

        pred_df = pd.DataFrame(
            {
                "true": y_true.astype(int),
                "pred": y_pred.astype(int),
            }
        )
        if test_metrics["probabilities"].numel() > 0:
            probs = test_metrics["probabilities"].numpy()
            if probs.ndim == 2:
                for idx in range(probs.shape[1]):
                    pred_df[f"prob_{idx}"] = probs[:, idx]
        pred_df.to_csv(
            os.path.join(self.paths["predictions"], f"pred_true_subject_{self.subject_id - 1}.csv"),
            index=False,
        )

        metrics_row = {
            "subject_id": self.subject_id,
            "accuracy": metrics["accuracy"] * 100.0,
            "precision": metrics["precision"] * 100.0,
            "recall": metrics["recall"] * 100.0,
            "f1": metrics["f1"] * 100.0,
            "kappa": metrics["kappa"] * 100.0,
            "best_epoch": best_epoch,
        }
        pd.DataFrame([metrics_row]).to_csv(
            os.path.join(self.paths["metrics"], f"result_metric_subject_{self.subject_id - 1}.csv"),
            index=False,
        )

        return {
            "history": history_df,
            "metrics": metrics_row,
            "checkpoint_path": self.checkpoint_path,
            "data_info": bundle["info"],
        }


__all__ = ["Trainer"]
