"""Minimal data loading pipeline for the public LA-Net release."""

import os

import numpy as np
import scipy.io
import scipy.signal
import torch


def _get_default_sampling_rate(dataset_type):
    if dataset_type in {"A", "B"}:
        return 250.0
    raise ValueError(f"Unknown dataset type for sampling rate: {dataset_type}")


def apply_bandpass_filter_trials(
    data,
    low_hz,
    high_hz,
    sampling_rate,
    order=5,
    filter_type="butter",
    filt_mode="filtfilt",
    filt_allowance=2.0,
):
    if low_hz is None or high_hz is None:
        return data.astype(np.float32)
    low_hz = float(low_hz)
    high_hz = float(high_hz)
    sampling_rate = float(sampling_rate)
    filt_allowance = float(filt_allowance)
    if not (0 < low_hz < high_hz < sampling_rate / 2.0):
        raise ValueError(
            f"Invalid bandpass range [{low_hz}, {high_hz}] for sampling rate {sampling_rate}."
        )

    filter_type = str(filter_type).lower()
    filt_mode = str(filt_mode).lower()
    if filter_type == "butter":
        sos = scipy.signal.butter(
            order,
            [low_hz, high_hz],
            btype="bandpass",
            fs=sampling_rate,
            output="sos",
        )
    elif filter_type == "cheby2":
        nyquist = sampling_rate / 2.0
        f_pass = [low_hz / nyquist, high_hz / nyquist]
        f_stop = [(low_hz - filt_allowance) / nyquist, (high_hz + filt_allowance) / nyquist]
        if f_stop[0] <= 0 or f_stop[1] >= 1:
            raise ValueError(
                f"Invalid Chebyshev stopband {f_stop} for band [{low_hz}, {high_hz}] Hz and fs={sampling_rate}."
            )
        a_stop = 30
        a_pass = 3
        cheby_order, wn = scipy.signal.cheb2ord(f_pass, f_stop, a_pass, a_stop)
        sos = scipy.signal.cheby2(cheby_order, a_stop, wn, btype="bandpass", output="sos")
    else:
        raise ValueError(f"Unsupported bandpass filter_type: {filter_type}")

    if filt_mode == "filtfilt":
        filtered = scipy.signal.sosfiltfilt(sos, data, axis=-1)
    elif filt_mode == "filter":
        filtered = scipy.signal.sosfilt(sos, data, axis=-1)
    else:
        raise ValueError(f"Unsupported filt_mode: {filt_mode}")
    return filtered.astype(np.float32)


def load_mat_split(data_dir, dataset_type, subject_id, mode):
    suffix = "T" if str(mode).lower() in {"train", "t"} else "E"
    filename = f"{dataset_type}{int(subject_id):02d}{suffix}.mat"
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing dataset file: {path}")
    mat = scipy.io.loadmat(path)
    data = np.asarray(mat["data"], dtype=np.float32)
    label = np.asarray(mat["label"]).reshape(-1).astype(np.int64)
    return data, label


def load_subject_dependent_split(data_dir, dataset_type, subject_id):
    train_data, train_label = load_mat_split(data_dir, dataset_type, subject_id, "train")
    test_data, test_label = load_mat_split(data_dir, dataset_type, subject_id, "eval")
    return train_data, train_label, test_data, test_label


def load_subject_merged(data_dir, dataset_type, subject_id):
    train_data, train_label = load_mat_split(data_dir, dataset_type, subject_id, "train")
    test_data, test_label = load_mat_split(data_dir, dataset_type, subject_id, "eval")
    data = np.concatenate([train_data, test_data], axis=0).astype(np.float32)
    label = np.concatenate([train_label, test_label], axis=0).astype(np.int64)
    return data, label


def load_loso_subjects(data_dir, dataset_type, subject_id, n_total=9):
    source_subjects = []
    target_data, target_label = None, None
    for sid in range(1, int(n_total) + 1):
        data, label = load_subject_merged(data_dir, dataset_type, sid)
        if sid == int(subject_id):
            target_data, target_label = data, label
        else:
            source_subjects.append((sid, data, label))
    if target_data is None or target_label is None:
        raise ValueError(f"Target subject {subject_id} was not found.")
    return source_subjects, target_data, target_label


def _normalize_with_source_stats(source_data, arrays_to_normalize):
    source_mean = float(np.mean(source_data))
    source_std = float(np.std(source_data))
    if source_std == 0:
        source_std = 1.0
    normalized = [((arr - source_mean) / source_std).astype(np.float32) for arr in arrays_to_normalize]
    return normalized, source_mean, source_std


def _normalize_per_channel_with_source_stats(source_data, arrays_to_normalize):
    source_mean = np.mean(source_data, axis=(0, 1, 3), keepdims=True)
    source_std = np.std(source_data, axis=(0, 1, 3), keepdims=True)
    source_std = np.where(source_std == 0, 1.0, source_std)
    normalized = [((arr - source_mean) / source_std).astype(np.float32) for arr in arrays_to_normalize]
    return normalized, source_mean.astype(np.float32), source_std.astype(np.float32)


def _split_tail(source_data, source_label, validate_ratio):
    n_total = int(source_data.shape[0])
    n_validate = int(validate_ratio * n_total)
    n_validate = min(max(1, n_validate), n_total - 1)
    split_idx = n_total - n_validate
    return (
        source_data[:split_idx],
        source_label[:split_idx],
        source_data[split_idx:],
        source_label[split_idx:],
    )


def _split_stratified_by_class(source_data, source_label, validate_ratio):
    labels = np.asarray(source_label).reshape(-1)
    train_indices = []
    val_indices = []
    per_class = {}
    for class_label in sorted(np.unique(labels).tolist()):
        class_indices = np.where(labels == class_label)[0]
        n_class = int(class_indices.shape[0])
        if n_class < 2:
            train_indices.extend(class_indices.tolist())
            per_class[int(class_label)] = {"total": n_class, "train": n_class, "val": 0}
            continue
        n_val = int(round(float(validate_ratio) * n_class))
        n_val = min(max(1, n_val), n_class - 1)
        train_idx = class_indices[:-n_val]
        val_idx = class_indices[-n_val:]
        train_indices.extend(train_idx.tolist())
        val_indices.extend(val_idx.tolist())
        per_class[int(class_label)] = {
            "total": n_class,
            "train": int(train_idx.shape[0]),
            "val": int(val_idx.shape[0]),
        }
    if not train_indices or not val_indices:
        raise ValueError("Stratified validation split produced an empty split.")
    train_indices = np.asarray(train_indices, dtype=np.int64)
    val_indices = np.asarray(val_indices, dtype=np.int64)
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    return (
        source_data[train_indices],
        source_label[train_indices],
        source_data[val_indices],
        source_label[val_indices],
        per_class,
    )


def _scldgn_split_idx(indices, train_ratio=0.8, seed=0, dataset_id=0):
    indices = list(indices)
    if dataset_id == 100:
        idx1 = indices[:-40]
        idx2 = indices[-40:]
    elif dataset_id in {6, 7}:
        idx = list(indices)
        np.random.RandomState(seed).shuffle(idx)
        cutoff = int(len(idx) * train_ratio) + 1
        return idx[:cutoff], idx[cutoff:]
    else:
        half = len(indices) // 2
        idx1 = indices[:half]
        idx2 = indices[half:]

    np.random.RandomState(seed).shuffle(idx1)
    np.random.RandomState(seed).shuffle(idx2)

    n1 = int(len(idx1) * train_ratio)
    n2 = int(len(idx2) * train_ratio)
    idx_train = idx1[:n1] + idx2[:n2]
    idx_val = idx1[n1:] + idx2[n2:]
    return idx_train, idx_val


def _build_scldgn_loso_train_val_from_subjects(source_subjects, train_ratio=0.8, seed=0, dataset_id=0):
    train_data_list = []
    train_label_list = []
    train_domain_list = []
    val_data_list = []
    val_label_list = []
    val_domain_list = []
    per_subject = []

    for subject_id, subject_data, subject_label in source_subjects:
        labels = np.asarray(subject_label).reshape(-1)
        subject_train_idx = []
        subject_val_idx = []
        per_class = {}
        for class_label in sorted(np.unique(labels).tolist()):
            class_idx = np.where(labels == class_label)[0].tolist()
            class_train_idx, class_val_idx = _scldgn_split_idx(
                class_idx,
                train_ratio=float(train_ratio),
                seed=int(seed),
                dataset_id=int(dataset_id),
            )
            subject_train_idx.extend(class_train_idx)
            subject_val_idx.extend(class_val_idx)
            per_class[int(class_label)] = {
                "total": int(len(class_idx)),
                "train": int(len(class_train_idx)),
                "val": int(len(class_val_idx)),
            }

        train_data_list.append(subject_data[subject_train_idx])
        train_label_list.append(labels[subject_train_idx])
        train_domain_list.append(np.full((len(subject_train_idx),), int(subject_id), dtype=np.int64))

        val_data_list.append(subject_data[subject_val_idx])
        val_label_list.append(labels[subject_val_idx])
        val_domain_list.append(np.full((len(subject_val_idx),), int(subject_id), dtype=np.int64))

        per_subject.append(
            {
                "subject": f"{int(subject_id):02d}",
                "domain_id": int(subject_id),
                "train": int(len(subject_train_idx)),
                "val": int(len(subject_val_idx)),
                "per_class": per_class,
            }
        )

    train_data = np.concatenate(train_data_list, axis=0).astype(np.float32)
    train_label = np.concatenate(train_label_list, axis=0).astype(np.int64)
    train_domain_ids = np.concatenate(train_domain_list, axis=0).astype(np.int64)
    val_data = np.concatenate(val_data_list, axis=0).astype(np.float32)
    val_label = np.concatenate(val_label_list, axis=0).astype(np.int64)
    val_domain_ids = np.concatenate(val_domain_list, axis=0).astype(np.int64)
    split_info = {
        "name": "scldgn_bci2a_loso",
        "train_ratio": float(train_ratio),
        "seed": int(seed),
        "dataset_id": int(dataset_id),
        "source_subjects": per_subject,
        "train_samples": int(train_data.shape[0]),
        "val_samples": int(val_data.shape[0]),
    }
    return train_data, train_label, val_data, val_label, train_domain_ids, val_domain_ids, split_info


def build_dataloaders(
    data_dir,
    dataset_type,
    subject_id,
    evaluate_mode="LOSO",
    n_total=9,
    batch_size=72,
    num_workers=0,
    validate_ratio=0.3,
    use_bandpass_filter=False,
    bandpass_low_hz=4.0,
    bandpass_high_hz=40.0,
    sampling_rate=None,
    bandpass_order=5,
    bandpass_filter_type="butter",
    bandpass_filt_mode="filtfilt",
    bandpass_filt_allowance=2.0,
    normalization_mode="global",
    validation_split_strategy="tail",
    pin_memory=False,
):
    dataset_type = str(dataset_type)
    evaluate_mode = str(evaluate_mode)
    is_loso = evaluate_mode.upper().startswith("LOSO")
    use_scldgn_loso_split = is_loso and dataset_type == "A"
    data_dir = os.path.abspath(data_dir)

    info = {
        "dataset_type": dataset_type,
        "evaluate_mode": evaluate_mode,
        "subject_id": int(subject_id),
        "data_dir": data_dir,
        "alignment_method": "none",
        "bandpass_applied": False,
        "validation_split_strategy": None,
        "validation_split_per_class": None,
    }

    if is_loso:
        source_subjects, test_data, test_label = load_loso_subjects(
            data_dir,
            dataset_type,
            subject_id,
            n_total=n_total,
        )
        if use_bandpass_filter:
            sampling_rate = sampling_rate or _get_default_sampling_rate(dataset_type)
            source_subjects = [
                (
                    sid,
                    apply_bandpass_filter_trials(
                        subject_data,
                        bandpass_low_hz,
                        bandpass_high_hz,
                        sampling_rate,
                        order=bandpass_order,
                        filter_type=bandpass_filter_type,
                        filt_mode=bandpass_filt_mode,
                        filt_allowance=bandpass_filt_allowance,
                    ),
                    subject_label,
                )
                for sid, subject_data, subject_label in source_subjects
            ]
            test_data = apply_bandpass_filter_trials(
                test_data,
                bandpass_low_hz,
                bandpass_high_hz,
                sampling_rate,
                order=bandpass_order,
                filter_type=bandpass_filter_type,
                filt_mode=bandpass_filt_mode,
                filt_allowance=bandpass_filt_allowance,
            )
            info["bandpass_applied"] = True
        source_data = np.concatenate([subject_data for _, subject_data, _ in source_subjects], axis=0).astype(np.float32)
        source_label = np.concatenate([subject_label for _, _, subject_label in source_subjects], axis=0).astype(np.int64)
    else:
        source_data, source_label, test_data, test_label = load_subject_dependent_split(
            data_dir,
            dataset_type,
            subject_id,
        )
        if use_bandpass_filter:
            sampling_rate = sampling_rate or _get_default_sampling_rate(dataset_type)
            source_data = apply_bandpass_filter_trials(
                source_data,
                bandpass_low_hz,
                bandpass_high_hz,
                sampling_rate,
                order=bandpass_order,
                filter_type=bandpass_filter_type,
                filt_mode=bandpass_filt_mode,
                filt_allowance=bandpass_filt_allowance,
            )
            test_data = apply_bandpass_filter_trials(
                test_data,
                bandpass_low_hz,
                bandpass_high_hz,
                sampling_rate,
                order=bandpass_order,
                filter_type=bandpass_filter_type,
                filt_mode=bandpass_filt_mode,
                filt_allowance=bandpass_filt_allowance,
            )
            info["bandpass_applied"] = True

    source_data = np.expand_dims(source_data, axis=1).astype(np.float32)
    test_data = np.expand_dims(test_data, axis=1).astype(np.float32)
    info["source_shape_before_split"] = tuple(source_data.shape)
    info["test_shape"] = tuple(test_data.shape)

    normalization_mode = str(normalization_mode).lower()
    if normalization_mode in {"global", "scalar"}:
        normalized_arrays, source_mean, source_std = _normalize_with_source_stats(source_data, [source_data, test_data])
    elif normalization_mode in {"per_channel", "channel"}:
        normalized_arrays, source_mean, source_std = _normalize_per_channel_with_source_stats(
            source_data,
            [source_data, test_data],
        )
    else:
        raise ValueError(f"Unsupported normalization_mode: {normalization_mode}")
    source_data = normalized_arrays[0]
    test_data = normalized_arrays[1]
    info["normalization_mode"] = normalization_mode

    train_domain_ids = None
    val_domain_ids = None
    if use_scldgn_loso_split:
        normalized_subjects = []
        offset = 0
        for sid, _, subject_label in source_subjects:
            n_trials = int(np.asarray(subject_label).shape[0])
            normalized_subjects.append((sid, source_data[offset : offset + n_trials], np.asarray(subject_label).reshape(-1)))
            offset += n_trials
        train_data, train_label, val_data, val_label, train_domain_ids, val_domain_ids, split_info = (
            _build_scldgn_loso_train_val_from_subjects(
                normalized_subjects,
                train_ratio=0.8,
                seed=0,
                dataset_id=0,
            )
        )
        split_info["test_subject"] = f"{int(subject_id):02d}"
        split_info["test_samples"] = int(test_data.shape[0])
        info["scldgn_loso_split"] = split_info
        info["validation_split_strategy"] = "scldgn_subjectwise"
        aug_train_data = train_data
        aug_train_label = train_label
        aug_train_domain_ids = train_domain_ids
    else:
        shuffle_index = np.random.permutation(len(source_data))
        source_data = source_data[shuffle_index]
        source_label = source_label[shuffle_index]
        split_strategy = str(validation_split_strategy).strip().lower()
        if split_strategy in {"stratified", "stratified_by_class", "class_stratified"}:
            train_data, train_label, val_data, val_label, per_class = _split_stratified_by_class(
                source_data,
                source_label,
                validate_ratio,
            )
            info["validation_split_strategy"] = "stratified_by_class"
            info["validation_split_per_class"] = per_class
        else:
            train_data, train_label, val_data, val_label = _split_tail(source_data, source_label, validate_ratio)
            info["validation_split_strategy"] = "tail"
        aug_train_data = source_data
        aug_train_label = source_label
        aug_train_domain_ids = None

    train_tensors = [torch.from_numpy(train_data), torch.from_numpy(train_label - 1)]
    val_tensors = [torch.from_numpy(val_data), torch.from_numpy(val_label - 1)]
    test_tensors = [torch.from_numpy(test_data), torch.from_numpy(test_label - 1)]
    if train_domain_ids is not None:
        train_tensors.append(torch.from_numpy(train_domain_ids))
    if val_domain_ids is not None:
        val_tensors.append(torch.from_numpy(val_domain_ids))

    persistent_workers = bool(num_workers) and int(num_workers) > 0
    train_loader = torch.utils.data.DataLoader(
        dataset=torch.utils.data.TensorDataset(*train_tensors),
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        persistent_workers=persistent_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=torch.utils.data.TensorDataset(*val_tensors),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        persistent_workers=persistent_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=torch.utils.data.TensorDataset(*test_tensors),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        persistent_workers=persistent_workers,
    )

    info["train_shape"] = tuple(train_data.shape)
    info["val_shape"] = tuple(val_data.shape)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "aug_train_data": aug_train_data.astype(np.float32),
        "aug_train_label": aug_train_label.astype(np.int64),
        "aug_train_domain_ids": aug_train_domain_ids.astype(np.int64) if aug_train_domain_ids is not None else None,
        "test_label_tensor": torch.from_numpy(test_label - 1),
        "source_mean": source_mean,
        "source_std": source_std,
        "info": info,
    }


__all__ = [
    "apply_bandpass_filter_trials",
    "build_dataloaders",
    "load_mat_split",
    "load_subject_dependent_split",
    "load_subject_merged",
    "load_loso_subjects",
]
