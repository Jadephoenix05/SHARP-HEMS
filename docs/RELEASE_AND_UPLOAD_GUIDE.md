# SHARP V1 — release, upload and safe local cleanup

## 1. Build and verify the release

```bash
.venv\Scripts\python.exe scripts\build_sharp_master_release_v1.py
.venv\Scripts\python.exe scripts\verify_sharp_release_v1.py
```

Both must finish clean. `verify_sharp_release_v1.py` exits non-zero on any
failure, so it is safe to treat as a gate.

## 2. Upload to a PRIVATE Kaggle dataset

The release folder is `release/SHARP_MASTER_V1/`.

```bash
pip install kaggle
kaggle datasets init -p release/SHARP_MASTER_V1
```

Edit the generated `dataset-metadata.json`:

```json
{
  "title": "SHARP Master Dataset V1",
  "id": "<your-kaggle-username>/sharp-master-dataset-v1",
  "licenses": [{"name": "other"}]
}
```

Then create it as private:

```bash
kaggle datasets create -p release/SHARP_MASTER_V1 --dir-mode zip
```

Keep it **private**. Several upstream sources are licence-restricted, and the
release deliberately excludes their raw microdata — see `not_redistributed` in
`MANIFEST.json`. Do not make it public without checking every upstream licence.

For later versions:

```bash
kaggle datasets version -p release/SHARP_MASTER_V1 -m "V1.1 notes" --dir-mode zip
```

## 3. Verify the downloaded copy before trusting it

On Kaggle, or after downloading elsewhere:

```bash
python verify_sharp_release_v1.py --release /kaggle/input/sharp-master-dataset-v1
```

This is the teammate-reproduction gate. It re-checks hashes, feature widths,
split disjointness, physical plausibility and episode structure using only the
released files.

## 4. Train BDQ

```bash
.venv\Scripts\python.exe scripts\train_sharp_bdq_v1.py --steps 3000
```

Reads `rl_transitions.parquet`, fits normalisation on train only, evaluates on
validation, and never touches the test split. Point `--root` at the Kaggle input
directory to train there.

## 5. GitHub

Code, configs and the registry only. `.gitignore` already excludes `data/`,
`release/`, `.venv/` and the large archives.

```bash
git remote add origin https://github.com/<you>/sharp.git
git push -u origin main
```

## 6. Deleting local files — order matters

Do **not** delete anything until steps 1–3 have passed on the *downloaded*
Kaggle copy, not the local one.

**Safe to delete once verified** (re-downloadable from official sources, and the
download scripts plus recorded SHA-256 hashes are in the repo):

| Path | Size | How to get it back |
|---|---|---|
| `data/raw/refit` | 8.4 GB | Strathclyde PurePortal, CLEAN_REFIT_081116 |
| `data/raw/reside_ac` | 12 MB | Figshare, CC0 |
| `data/raw/weather` | 1.6 MB | `scripts/download_weather.py`, `scripts/download_reside_site_weather.py` |
| `data/interim/iawe` | 1.2 GB | regenerate from `data/raw/iawe` |
| `data/interim/emarc` | 170 MB | regenerate from `data/raw/emarc` |

**Keep, or re-acquiring is painful:**

- `data/raw/tus_india` (1.8 GB) — MoSPI microdata sits behind a registration
  flow, and `tus_2024` still has open review items.
- `data/raw/iawe` (2.4 GB) — mirror availability is not guaranteed.
- `data/raw/ires`, `data/raw/emarc`, `data/raw/bee_clasp`, `data/raw/grid_india`
  — small, and each still carries open semantic checks.

**Never delete:** `scripts/`, `configs/`, `data_registry/`, `reports/`, `docs/`,
and the `.git` history. These are the build definition; without them the data
cannot be rebuilt or defended.

## 7. What is still open after V1

- Appliance power values are proxies and declared assumptions, not measured
  Indian ratings.
- Thermal parameters are declared assumptions bounded by the observed RESIDE
  envelope. A fitted coefficient was attempted and rejected.
- No solar, battery, outage or human-attention model.
- `iced_hourly` was never acquired; grid context came from Grid-India.
- Several sources retain open semantic checks; see `remaining_checks` per entry
  in `data_registry/sources.yaml`.
