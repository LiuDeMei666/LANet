# Data Preparation

This document describes the official data-cache generation workflow used by the public LA-Net release.

## Scope

Two datasets are supported:

- **BCI Competition IV-2a**
- **BCI Competition IV-2b**

The original recordings are not redistributed here. Users must obtain the
datasets from the official BCI Competition IV source and follow its terms of
use. The scripts below generate only derived MATLAB cache files.

The output cache format is fixed:

- MATLAB `.mat` files
- key `data`
- key `label`

Output directories:

- `datasets/mymat_a/`
- `datasets/mymat_b/`

## Preparation workflow

The public preparation scripts are organized from the original source pipeline into CLI tools:

- `preprocessing_for_2a.py`
- `preprocessing_for_2b.py`

They preserve the established cache-generation behavior while exposing a cleaner command-line interface.

The scripts:

- read the official `.gdf` files
- remove the EOG channels
- epoch fixed 4-second windows
- merge IV-2b sessions into subject-level train/eval caches
- write MATLAB cache files containing only `data` and `label`

The generated `mymat_a` and `mymat_b` caches are the expected dataset format for the released configs and official weights.

## BCIC IV-2a

Command:

```bash
python scripts/prepare_bcic2a.py \
  --raw_dir /path/to/BCICIV_2a_gdf \
  --label_dir /path/to/true_labels(BCICIV_2a_gdf) \
  --out_dir datasets/mymat_a
```

Expected outputs:

- `A01T.mat` ... `A09T.mat`
- `A01E.mat` ... `A09E.mat`

Expected shape:

- `data`: `[288, 22, 1000]`
- `label`: `[288, 1]`

## BCIC IV-2b

Command:

```bash
python scripts/prepare_bcic2b.py \
  --raw_dir /path/to/BCICIV_2b_gdf \
  --label_dir /path/to/true_labels(BCICIV_2b_gdf) \
  --out_dir datasets/mymat_b
```

Expected outputs:

- `B01T.mat` ... `B09T.mat`
- `B01E.mat` ... `B09E.mat`

Expected shapes:

- channel/time dimensions are fixed at `[3, 1000]`
- trial counts vary by subject:
  - `B01T=400`, `B01E=320`
  - `B02T=400`, `B02E=280`
  - `B03T=400`, `B03E=320`
  - `B04T=420`, `B04E=320`
  - `B05T=420`, `B05E=320`
  - `B06T=400`, `B06E=320`
  - `B07T=400`, `B07E=320`
  - `B08T=440`, `B08E=320`
  - `B09T=400`, `B09E=320`

## Sanity checks

After preparation, spot-check at least one subject:

- file exists
- keys are exactly `data` and `label`
- shape matches the expected cache
- label count equals trial count

That is enough to confirm compatibility with the official configs and weights released here.
