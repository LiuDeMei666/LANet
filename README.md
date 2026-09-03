# LA-Net

Official minimal release of **LA-Net** for EEG-based motor imagery decoding on **BCI Competition IV-2a** and **BCI Competition IV-2b**.

This repository is intentionally narrow:

- one public model family: `LANet`
- one public spatial prior module: `LSAModule`
- one public temporal-spatial attention module: `TSAModule`
- four official configs
- four official weight groups
- official raw-data preparation scripts for `mymat_a` / `mymat_b`

It does **not** include the private experiment zoo, paper assets, auxiliary baselines, or unpublished branches.

## Citation

If you use this code or the released weights, please cite:

Demei Liu and Yujuan Quan, "LA-Net: A Lateralized Asymmetry Network for Motor Imagery EEG Decoding."

Please replace this entry with the final published citation when available.

## Repository layout

```text
LA-Net/
├─ models/
├─ data/
├─ configs/official/
├─ scripts/
├─ weights/
├─ datasets/
│  ├─ mymat_a/
│  └─ mymat_b/
├─ README.md
├─ DATA_PREPARATION.md
├─ LICENSE
└─ requirements.txt
```

## Environment

Validated environment:

- Python `3.11.14`
- PyTorch `2.5.1`
- CUDA build used in the validated GPU environment: `2.5.1+cu121`

`requirements.txt` now reflects the exact package versions used in the tested public release.

```bash
pip install -r requirements.txt
```

If you prefer Conda, an exact environment file is also provided:

```bash
conda env create -f environment.yml
conda activate LA-Net
```

If you need GPU training or evaluation, replace the default `torch` install with the CUDA wheel that matches your driver and platform.

## Data preparation

The public release provides the official data-preparation workflow for `mymat_a` and `mymat_b`.

Prepare **BCI Competition IV-2a**:

```bash
python scripts/prepare_bcic2a.py \
  --raw_dir /path/to/BCICIV_2a_gdf \
  --label_dir /path/to/true_labels(BCICIV_2a_gdf) \
  --out_dir datasets/mymat_a
```

Prepare **BCI Competition IV-2b**:

```bash
python scripts/prepare_bcic2b.py \
  --raw_dir /path/to/BCICIV_2b_gdf \
  --label_dir /path/to/true_labels(BCICIV_2b_gdf) \
  --out_dir datasets/mymat_b
```

Important:

- they save only `data` and `label`
- the generated caches are directly compatible with the released configs and official weights
- the IV-2b cache has **subject-dependent trial counts** and is not uniformly `400/320`

Details are documented in [DATA_PREPARATION.md](DATA_PREPARATION.md).

The original BCI Competition IV-2a and IV-2b recordings are not included in
this repository. Please obtain them from the official dataset source and
comply with the dataset terms of use. The MIT License in this repository
covers only the original LA-Net code and does not cover the datasets or
third-party dependencies.

## Official configs

- `configs/official/bcic2a_subject_dependent.yaml`
- `configs/official/bcic2a_loso.yaml`
- `configs/official/bcic2b_subject_dependent.yaml`
- `configs/official/bcic2b_loso.yaml`

The official configs include fixed per-subject random seeds to improve reproducibility of the released results. If you want to reproduce the reported `accuracy` and `kappa` as closely as possible, use the provided seeds directly. You can still override them with `--seed` when running your own experiments.

## Training

Example: subject-dependent training on IV-2a subject 1

```bash
python train.py \
  --config configs/official/bcic2a_subject_dependent.yaml \
  --subject 1 \
  --device cuda:0 \
  --num_workers 4
```

Example: LOSO training on IV-2b with target subject 1

```bash
python train.py \
  --config configs/official/bcic2b_loso.yaml \
  --subject 1 \
  --device cuda:0 \
  --num_workers 4
```

## Evaluation

```bash
python eval.py \
  --config configs/official/bcic2a_loso.yaml \
  --checkpoint weights/bcic2a_loso/subject_01.pt \
  --subject 1 \
  --device cuda:0
```

## Weights

The `weights/` directory contains four official result groups:

- `bcic2a_subject_dependent/`
- `bcic2a_loso/`
- `bcic2b_subject_dependent/`
- `bcic2b_loso/`

Each group contains:

- `subject_01.pt` ... `subject_09.pt`
- `config.yaml`
- `metrics.csv`

Each checkpoint is a plain checkpoint dict with `state_dict`, metadata, and the config snapshot used during conversion.

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
PyTorch, MNE, NumPy, SciPy, scikit-learn, pandas, PyYAML, and einops remain
under their respective licenses; consult the corresponding project
documentation for details.
