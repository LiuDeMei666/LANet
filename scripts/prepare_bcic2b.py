"""Prepare historical BCIC IV-2b caches used by the LA-Net release.

This script is an engineering cleanup of the historical preprocessing source.
It preserves the original session-merging and epoching logic used to generate
the legacy `mymat_b` cache.
"""

import argparse
import os

import mne
import numpy as np
import scipy.io as sio
from scipy.io import savemat


EXPECTED_COUNTS = {
    1: {"T": 400, "E": 320},
    2: {"T": 400, "E": 280},
    3: {"T": 400, "E": 320},
    4: {"T": 420, "E": 320},
    5: {"T": 420, "E": 320},
    6: {"T": 400, "E": 320},
    7: {"T": 400, "E": 320},
    8: {"T": 440, "E": 320},
    9: {"T": 400, "E": 320},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare historical BCIC IV-2b .mat caches.")
    parser.add_argument("--raw_dir", type=str, required=True, help="Directory containing B0101T.gdf ... B0905E.gdf.")
    parser.add_argument("--label_dir", type=str, required=True, help="Directory containing B0101T.mat ... B0905E.mat labels.")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for mymat_b style .mat files.")
    return parser.parse_args()


def _session_file(raw_dir, label_dir, subject_id, session_id, suffix):
    stem = f"B{int(subject_id):02d}{int(session_id):02d}{suffix}"
    gdf_path = os.path.join(raw_dir, f"{stem}.gdf")
    label_path = os.path.join(label_dir, f"{stem}.mat")
    if not os.path.exists(gdf_path):
        raise FileNotFoundError(f"Missing raw GDF file: {gdf_path}")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Missing label MAT file: {label_path}")
    return gdf_path, label_path


def _load_session(raw_dir, label_dir, subject_id, session_id, suffix):
    gdf_path, label_path = _session_file(raw_dir, label_dir, subject_id, session_id, suffix)
    raw = mne.io.read_raw_gdf(gdf_path, preload=False)
    events, event_dict = mne.events_from_annotations(raw)
    if suffix == "T":
        event_id = {"Left": event_dict["769"], "Right": event_dict["770"]}
    else:
        event_id = {"Unknown": event_dict["783"]}
    selected_events = events[np.isin(events[:, 2], list(event_id.values()))]

    raw.info["bads"] += ["EOG:ch01", "EOG:ch02", "EOG:ch03"]
    picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False, exclude="bads")
    epochs = mne.Epochs(
        raw,
        selected_events,
        event_id,
        picks=picks,
        tmin=0,
        tmax=3.996,
        preload=True,
        baseline=None,
        on_missing="ignore",
    )
    data = epochs.get_data().astype(np.float32)
    labels = np.asarray(sio.loadmat(label_path)["classlabel"]).astype(np.int64)
    if data.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Trial count mismatch for subject {subject_id} session {session_id} suffix {suffix}: "
            f"{data.shape[0]} epochs vs {labels.shape[0]} labels."
        )
    return data, labels


def _merge_sessions(raw_dir, label_dir, subject_id, session_ids, suffix):
    all_data = []
    all_labels = []
    for session_id in session_ids:
        data, labels = _load_session(raw_dir, label_dir, subject_id, session_id, suffix)
        all_data.append(data)
        all_labels.append(labels)
    merged_data = np.concatenate(all_data, axis=0).astype(np.float32)
    merged_labels = np.concatenate(all_labels, axis=0).astype(np.int64)
    return merged_data, merged_labels


def main():
    args = parse_args()
    mne.set_log_level("WARNING")
    os.makedirs(args.out_dir, exist_ok=True)

    for subject_id in range(1, 10):
        train_data, train_labels = _merge_sessions(args.raw_dir, args.label_dir, subject_id, session_ids=(1, 2, 3), suffix="T")
        eval_data, eval_labels = _merge_sessions(args.raw_dir, args.label_dir, subject_id, session_ids=(4, 5), suffix="E")

        expected_train_count = EXPECTED_COUNTS[subject_id]["T"]
        expected_eval_count = EXPECTED_COUNTS[subject_id]["E"]
        expected_train_shape = (expected_train_count, 3, 1000)
        expected_eval_shape = (expected_eval_count, 3, 1000)

        if train_data.shape != expected_train_shape:
            raise ValueError(
                f"Unexpected train shape for subject {subject_id}: "
                f"got {train_data.shape}, expected {expected_train_shape}."
            )
        if eval_data.shape != expected_eval_shape:
            raise ValueError(
                f"Unexpected eval shape for subject {subject_id}: "
                f"got {eval_data.shape}, expected {expected_eval_shape}."
            )

        train_path = os.path.join(args.out_dir, f"B{subject_id:02d}T.mat")
        eval_path = os.path.join(args.out_dir, f"B{subject_id:02d}E.mat")
        savemat(train_path, {"data": train_data, "label": train_labels})
        savemat(eval_path, {"data": eval_data, "label": eval_labels})
        print(f"Saved {train_path} | data={tuple(train_data.shape)} | label={tuple(train_labels.shape)}")
        print(f"Saved {eval_path} | data={tuple(eval_data.shape)} | label={tuple(eval_labels.shape)}")


if __name__ == "__main__":
    main()
