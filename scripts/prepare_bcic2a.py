"""Prepare historical BCIC IV-2a caches used by the LA-Net release.

This script is an engineering cleanup of the historical preprocessing source.
It preserves the original epoching logic and output format used to generate the
legacy `mymat_a` cache.
"""

import argparse
import os

import mne
import numpy as np
import scipy.io as sio
from scipy.io import savemat


EXPECTED_SHAPE = (288, 22, 1000)


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare historical BCIC IV-2a .mat caches.")
    parser.add_argument("--raw_dir", type=str, required=True, help="Directory containing A01T.gdf ... A09E.gdf.")
    parser.add_argument("--label_dir", type=str, required=True, help="Directory containing A01T.mat ... A09E.mat labels.")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for mymat_a style .mat files.")
    return parser.parse_args()


def _mode_suffix(mode):
    mode = str(mode).lower()
    if mode in {"train", "t"}:
        return "T"
    if mode in {"eval", "e", "test"}:
        return "E"
    raise ValueError(f"Unsupported mode: {mode}")


def _load_epoch_data(raw_dir, label_dir, subject_id, mode):
    suffix = _mode_suffix(mode)
    gdf_path = os.path.join(raw_dir, f"A{int(subject_id):02d}{suffix}.gdf")
    label_path = os.path.join(label_dir, f"A{int(subject_id):02d}{suffix}.mat")
    if not os.path.exists(gdf_path):
        raise FileNotFoundError(f"Missing raw GDF file: {gdf_path}")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Missing label MAT file: {label_path}")

    raw = mne.io.read_raw_gdf(gdf_path, preload=False)
    events, event_dict = mne.events_from_annotations(raw)
    if suffix == "T":
        event_id = {
            "Left": event_dict["769"],
            "Right": event_dict["770"],
            "Foot": event_dict["771"],
            "Tongue": event_dict["772"],
        }
    else:
        event_id = {"Unknown": event_dict["783"]}
    selected_events = events[np.isin(events[:, 2], list(event_id.values()))]

    raw.info["bads"] += ["EOG-left", "EOG-central", "EOG-right"]
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
    )
    data = epochs.get_data().astype(np.float32)
    labels = np.asarray(sio.loadmat(label_path)["classlabel"]).astype(np.int64)

    if data.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Trial count mismatch for subject {subject_id} mode {suffix}: "
            f"{data.shape[0]} epochs vs {labels.shape[0]} labels."
        )
    if data.shape != EXPECTED_SHAPE:
        raise ValueError(
            f"Unexpected output shape for subject {subject_id} mode {suffix}: "
            f"got {data.shape}, expected {EXPECTED_SHAPE}."
        )
    return data, labels


def main():
    args = parse_args()
    mne.set_log_level("WARNING")
    os.makedirs(args.out_dir, exist_ok=True)

    for mode in ("train", "eval"):
        suffix = _mode_suffix(mode)
        for subject_id in range(1, 10):
            data, labels = _load_epoch_data(args.raw_dir, args.label_dir, subject_id, mode)
            out_path = os.path.join(args.out_dir, f"A{subject_id:02d}{suffix}.mat")
            savemat(out_path, {"data": data, "label": labels})
            print(f"Saved {out_path} | data={tuple(data.shape)} | label={tuple(labels.shape)}")


if __name__ == "__main__":
    main()
