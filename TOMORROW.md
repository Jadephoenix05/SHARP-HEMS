# Tomorrow: upload checklist

Everything is built and verified locally. This is the whole job.

---

## 1. Kaggle — the training dataset (59 MB)

```bash
.venv\Scripts\python.exe -m kaggle datasets init -p release\SHARP_MASTER_V2
```

Edit the generated `release\SHARP_MASTER_V2\dataset-metadata.json`:

```json
{
  "title": "SHARP Master Dataset V2",
  "id": "YOUR_KAGGLE_USERNAME/sharp-master-dataset-v2",
  "licenses": [{"name": "other"}]
}
```

```bash
.venv\Scripts\python.exe -m kaggle datasets create -p release\SHARP_MASTER_V2 --dir-mode zip
```

Keep it **private**. Kaggle defaults to private; do not pass `--public`.

---

## 2. Kaggle — the raw source archives (2.79 GB)

```bash
.venv\Scripts\python.exe scripts\prepare_raw_kaggle_dataset.py --username YOUR_KAGGLE_USERNAME
```

```bash
.venv\Scripts\python.exe -m kaggle datasets create -p raw_archive --dir-mode tar
```

Also private. Four archives (`tus_india`, `emarc`, `ires`, `bee_clasp`) are
licence-restricted and are flagged in `RAW_MANIFEST.json`.

---

## 3. GitHub — the code (~15 MB, no data)

```bash
git remote add origin https://github.com/YOUR_USERNAME/sharp.git
```

```bash
git push -u origin main
```

`.gitignore` already excludes `data/`, `release/`, `raw_archive/`, `.venv/`.

---

## 4. Verify the downloaded copy before trusting it

In a Kaggle notebook, after attaching the dataset:

```bash
python verify_sharp_release_v1.py --release /kaggle/input/sharp-master-dataset-v2
```

Re-checks all 34 SHA-256 hashes, both leakage gates, feature widths, episode
structure and the honesty flags. Exits non-zero on any failure.

---

## 5. Train on Kaggle

```bash
python train_sharp_bdq_v1.py --steps 4000
```

Point `--root` at the Kaggle input directory. Loading needs **pandas and numpy
only** — no SHARP module is imported, verified.

```python
import pandas as pd, numpy as np
d = pd.read_parquet('/kaggle/input/sharp-master-dataset-v2/rl_transitions/splits/train.parquet')
X = np.stack(d.state.to_numpy()).astype('float32')      # (N, 305)
M = np.stack(d.device_present.to_numpy()).astype('float32')  # (N, 28) branch mask
```

Train on `splits/train.parquet`. Tune on `validation.parquet`. **Do not open
`test.parquet` until final reporting.**

---

## What is in the release

| | |
|---|---|
| Transitions | 400,896 |
| Episodes | 4,176 (96 steps each) |
| Households | 464 (train 323 / validation 69 / test 72) |
| Features | 305 (25 global + 28 devices x 10) |
| Override preference pairs | 47,915 |
| Seasons | all three |
| Behaviour policies | 3 |

Gates: billing reconciliation PASS · state continuity PASS · household leakage
NONE · date leakage NONE · infeasible steps outside an outage 0 · **E8 critical
load safety 0 violations in 10,000 attempts**.

Audit: `audit_sharp_rl_transitions_v2.py` passes with 0 failures, 0 warnings.

---

## Do NOT delete raw data yet

Delete only after step 4 passes on the **downloaded** copy, not the local one.
Then `data/raw/refit` (8.4 GB) is the safe first deletion: it has a download
script and a recorded SHA-256. Keep `data/raw/tus_india` — MoSPI is behind a
registration flow. Full procedure in `docs/RELEASE_AND_UPLOAD_GUIDE.md`.

---

## Known gaps, carried forward honestly

- Appliance power is measurement-grounded for 685 of 4,124 devices (16.6%);
  the rest are declared engineering assumptions. No Indian appliance-level
  metering exists at scale to fix this.
- Override and attention evidence is SYNTHETIC. A real Pi deployment logging
  user overrides is what replaces it.
- Service clock times are synthetic: surveys report daily hours, not when.
- Outage duration is reported; placement within the day is assumed.
- REFIT load SHAPES are still unused; only medians are. `refit_trace_library/`
  holds ~2.7M rows of segmented traces ready for that work.

Full accounting: `reports/dataset_provenance_v1.json`.
