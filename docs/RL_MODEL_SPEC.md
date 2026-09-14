# SHARP RL model specification

What to build, why, and what it must not claim. Lightweight and offline by
design, because the policy has to run on a Raspberry Pi and there is no live
environment to explore in.

Cross-checked against `sharp-master-dataset-v2` and the project idea book.

---

## 1. The dataset you are training on

| | |
|---|---|
| Transitions | 400,896 |
| Episodes | 4,176 × 96 steps (one simulated day each) |
| Households | 464 — train 323 / validation 69 / test 72 |
| State | **305 features** = 25 global + 28 device slots × 10 |
| Action | **28 branches × 3 levels** |
| Behaviour policies | `serve_preferred`, `peak_aware`, `random_binary` |
| Override preference pairs | 3,180 |

Splits are disjoint in **two** dimensions: household **and** calendar year
(train 2021–22, validation 2023, test 2024). Neither a household nor a date
appears in two splits.

```python
import pandas as pd, numpy as np
d = pd.read_parquet('/kaggle/input/sharp-master-dataset-v2/'
                    'rl_transitions/splits/train.parquet')
X = np.stack(d.state.to_numpy()).astype('float32')          # (N, 305)
M = np.stack(d.device_present.to_numpy()).astype('float32') # (N, 28) branch mask
```

Loading needs **pandas and numpy only** — no SHARP module is imported. Verified.

---

## 2. State: 305 features

### Global (25)

| Idx | Feature | Note |
|---|---|---|
| 0–3 | `obs_T2M`, `obs_RH2M`, `obs_ALLSKY_SFC_SW_DWN`, `obs_WS10M` | NASA POWER, Guntur |
| 4–5 | `obs_grid_percentile`, `obs_grid_peak_severity` | thresholds fitted on training years only |
| 6 | `fraction_of_day` | |
| 7 | `month_to_date_kwh_div500` | drives the telescopic slab |
| 8 | `connection_limit_kw` | |
| 9–11 | `indoor_temperature_c_div50`, `degrees_above_comfort_band`, `degrees_below_comfort_band` | |
| 12 | `household_has_air_conditioner` | |
| 13–14 | `occupancy_adult_home_fraction`, `attention_available` | TUS proxy |
| 15 | `marginal_tariff_inr_kwh_div10` | real APCPDCL slab rate |
| 16–18 | `mode_grid_import`, `mode_self_sufficient`, `mode_islanded` | one-hot |
| 19–23 | `self_sufficient_fraction`, `battery_state_of_charge`, `pv_generation_kw`, `unserved_demand_kw`, `grid_absent` | |
| 24 | `recent_override_count_div10` | |

### Per device (10 × 28 slots)

`remaining_service_hours`, `power_proxy_kw`, `current_on`, `protected_service`,
`cycle_type`, `elapsed_state_steps_divided_by_96`, `preferred_service_fraction`,
`is_air_conditioner`, `steps_since_shed_div96`, `device_override_count_div10`.

The last two are the anti-fatigue memory your idea book's `build_state` sketch
calls for.

**37 of 305 features have zero variance** (padding, plus `pv_generation_kw` and
`unserved_demand_kw` which are constant in the main dataset). Harmless — guard
the normaliser with `sd[sd < 1e-8] = 1.0` — but ~12 % of your input width
carries no information.

---

## 3. Action space

```
0 = OFF (shed)      1 = ON (full)      2 = REDUCED (dim / low speed)
```

Level 2 is legal only where `supports_reduced = True`: ceiling fan, table fan,
air cooler, LED bulb, LED tube, incandescent bulb. **2,460 of 4,124 devices.**

Necessity appliances — fans, all lighting, fridge, router, water purifier —
are **masked out of level 0 entirely**. Verified: across 401,562 cases where the
grid was up, the occupant wanted a necessity and budget remained, it was denied
**zero times**.

So the agent's real decision space is: shed the 969 discretionary devices, and
dim the 2,460 dimmable ones.

---

## 4. Architecture — Branching Dueling Double-Q

Branching is the right choice and your idea book already justifies it: a joint
action space over 28 binary devices is 2²⁸; branching is 28 × 3 = 84 outputs,
**linear rather than exponential**.

```
input 305
  └── shared trunk: Dense(128) + ReLU          ~39k params
        ├── value head:     Dense(1)
        └── advantage head: Dense(28 × 3)      ~10k params
  Q[i] = V + A[i] − mean(A[i])       duelling, per branch
```

Total ≈ **50k parameters, under 1 MB in float32.** That is the entire reason to
prefer this over a monolithic DQN: it fits in a Pi's cache and infers in
milliseconds.

Reference implementation: `scripts/train_sharp_bdq_v1.py`, pure NumPy, no
framework. It already trains: validation TD error 0.0933 over 3,000 updates.

### Why NumPy and not PyTorch

The Pi has to run inference, not training. A NumPy forward pass is three matrix
multiplies and a ReLU — no runtime, no version drift, no 300 MB wheel on an ARM
board. Train wherever you like; export the weights.

---

## 5. Offline RL — the part that needs care

You have a fixed batch. You cannot explore. That changes what is safe to do.

### The core hazard

Offline Q-learning overestimates actions that rarely appear in the data. The
max in the Bellman target picks the most *overestimated* action, not the best
one, and there is no environment to correct the error. It compounds.

**Current status: no correction is applied.** `train_sharp_bdq_v1.py` is honest
Double-Q with train-only normalisation, and its own limitations list says so.

### What was added, and what it bought

All three are implemented in `scripts/train_sharp_bdq_v2.py` and in the Kaggle
notebook `notebooks/sharp_rl_training_kaggle.ipynb`.

1. **Conservative Q-Learning (CQL).** The penalty term turns out to be exactly
   the cross-entropy of the logged action under a softmax over that branch's Q
   values, so the conservative penalty and behaviour cloning are the *same term*
   — cloning is that loss with the TD part switched off. One objective, no
   second head.
2. **Behaviour cloning warm start.** Phase 1 of the same trainer. This is E5.
3. **Level-legality masking.** Level 2 is meaningless on a device that cannot
   dim, and ~40 % cannot. Verified first: across 499,392 logged device-steps,
   the behaviour policies chose level 2 on a non-dimmable device **zero** times,
   so the mask is safe to impose. Note what is *not* masked — OFF on a necessity
   appliance, because the necessity rule is conditional and a fridge is
   legitimately off at 3 a.m.

Still outstanding: **shield-consistent targets.** Next-action selection now
applies the legality mask but still does not re-apply the joint shield, which
needs nested device state the flat release does not carry.

### ⚠ Two metrics that lie

**Pooled TD error.** Non-terminal TD *falls* while terminal TD *rises* over the
same run (0.083 → 0.072 against 1.86 → 2.10). Pooled, that reads as progress.
The v1 headline of 0.0933 was the flattering half. Report them apart.

**Raw agreement with the logged action.** 83.18 % of logged actions are OFF, so a
model that answers OFF unconditionally scores 0.83. An early conservative run
scored 0.8335 and looked excellent — until the greedy distribution showed
**99.6 % OFF and level-2 recall of exactly 0.0000.** It had collapsed to the
majority class.

The fix is inverse-frequency weighting on the CQL term, and the headline metric
is **balanced accuracy** — the mean of the three per-level recalls — which that
collapsed model scores 0.333 on, exactly chance.

### Measured ablation (validation)

From the Kaggle notebook at full budget — 2,000 behaviour-cloning updates plus
5,000 conservative updates per arm, level-legality mask applied throughout.

| Config | Balanced accuracy | off / on / reduced recall |
|---|---|---|
| Neither | 0.385 | 0.42 / 0.42 / 0.31 |
| + BC warm start | 0.473 | 0.42 / 0.43 / 0.57 |
| + CQL only | 0.494 | 0.41 / 0.39 / 0.68 |
| + both | 0.530 | 0.42 / 0.40 / 0.78 |

Both corrections earn their place, and the gain is concentrated in **REDUCED
recall** — 0.31 to 0.78. That is the level SHARP exists to choose, and the level
an unweighted objective abandons first because it is only 2.6 % of logged
actions.

Running the main configuration longer — 2,000 warm start plus **15,000**
conservative updates at α = 1.0 — reaches **0.6008**, with zero illegal action
selections. Chance is 0.333.

### ⚠ The ceiling on this metric is about 0.78, not 1.0

`random_binary` is **exactly one third of every split** and it *samples* its
action. Its choices are not a function of the state, so no model reading the
state can predict them, at any capacity. Perfect imitation of the two
deterministic policies plus chance on the random third bounds the metric at
roughly `(2/3 × 1.0) + (1/3 × 0.33) ≈ 0.78`.

The per-policy breakdown shows the mechanism directly:

| Behaviour policy | Model scores |
|---|---|
| `serve_preferred` (deterministic) | 0.597 |
| `peak_aware` (deterministic) | 0.497 |
| `random_binary` (samples) | 0.482 |

**Do not chase this number.** Balanced accuracy measures agreement with three
scripted controllers. Scoring near the ceiling would mean an excellent imitation
of a rule-based policy you already have for free — the opposite of the claim the
project makes. It is a sanity check that the network learned structure, nothing
more. The figures that belong in the results section are cost against baseline,
peak-to-average ratio and comfort hours, and those need the simulator in the
loop.

`scripts/analyse_imitation_ceiling_v1.py` measures this.

## 6. The reward, and the half that is missing

Currently implemented in `sharp_reward_billing.py`:

```
reward = −(cost_inr + peak_kwh × severity + 0.5 × discomfort
           + 0.01 × switching + 10 × unmet_service)
```

Your idea book specifies **cost, peak, comfort, switching wear, override risk
and safety**. Override risk and safety are absent, and comfort is a hand-set
band penalty rather than the learned model your method section describes:

```python
reward = −(cost + peak) + reward_model.predict(s, a)   # r_ψ
```

### The Bradley-Terry preference reward — and its defect

Implemented in `scripts/fit_preference_reward_v1.py` and in section 9 of the
notebook. Fit on the 3,180 pairs in
`rl_transitions/override_preference_pairs.parquet`.

**Read this before quoting a number.** Every one of the 3,180 pairs compares
preferred = 1 (ON) against proposed = 0 (OFF). The direction never varies,
because a pair is only recorded when the occupant overrode a shed. So the
constant rule *"the occupant always wants it on"* scores **100 %**, beating the
fitted model's 98.6 %. **The classification accuracy is not a result and must not
appear in the abstract.**

What *is* a result is the **margin**, `r_ψ(s,i,ON) − r_ψ(s,i,OFF)` — a learned,
state-dependent measure of how badly the occupant wants that appliance back,
which is exactly what a reward has to supply. On held-out households and dates:

| Evidence | Value |
|---|---|
| margin vs `override_probability` | pearson **+0.302**, spearman **+0.320** |
| mean margin by latency (0 → 3 steps) | 2.70 → 2.59 → 1.94 → **1.88** |

The monotone decay with latency is the generator's own weighting recovered from
comparisons alone. Those are the E2 numbers.

**To make the direction informative**, the pair generator must also emit
comparisons where the occupant *let a shed stand*. Those transitions exist but
produce no pair today. That is a dataset regeneration, not a script change.

Also handled: 52 pairs carry `preference_weight` exactly 0 because response
latency pushed the override past the moment everyone left the house. They are
dropped rather than trained at zero weight.

**Every pair is synthetic**, generated from the stated rule in
`sharp_human_model.py`. Recovering it shows the rule is learnable from override
comparisons — not that real occupants behave this way. Label it synthetic every
time.

## 7. Experiments

| | Experiment | Status |
|---|---|---|
| E1 | SHARP vs rule-based / MILP / flat DQN / PPO | ✅ 3 behaviour policies available as baselines |
| **E2** | **Reward recovery θ̂ vs θ\*** | ⚠ fitted; margin recovers pressure (r = +0.30), but pair direction is constant so accuracy is meaningless |
| E3 | Override rate over training rounds | ❌ needs an online loop |
| E4 | Ablations: −shield, −latency, −ranking, −censoring | ✅ latency varies {0,1,2,3} |
| E5 | Sample efficiency with/without BC warm start | ✅ ablation measured: 0.341 → 0.375 balanced accuracy |
| E6 | Generalisation to held-out households | ✅ 72 test households, disjoint |
| E7 | Indian validation on iAWE, outage mode | ✅ `reports/e7_iawe_heldout_v1.json` |
| **E8** | **10,000 attempts to shed a critical load** | ✅ **0 violations** |

E8 already passes and is reportable as-is: 10,000 attempts across four attack
routes, including 2,500 direct human-override attacks on critical loads.

---

## 8. Export for the Pi

```python
np.savez('sharp_policy_v2.npz',
         w=net.p['w'], b=net.p['b'], v=net.p['v'], vb=net.p['vb'],
         a=net.p['a'], ab=net.p['ab'],
         mean=mean, sd=sd,                 # train-only normalisation
         feature_names=schema['global_features'] + device_names,
         n_features=305, n_branches=28, n_levels=3,
         policy_version='bdq_v2', trained_at=...)
```

Ship `mean` and `sd` with the weights. If the Pi normalises with anything else,
inference is silently wrong and nothing will tell you.

Inference on the Pi is three matrix multiplies:

```python
h = np.maximum(0, (x - mean) / sd @ w + b)
q = (h @ v + vb) + (h @ a + ab).reshape(28, 3)
q -= q.mean(axis=1, keepdims=True)
action = np.argmax(np.where(legal_mask, q, -np.inf), axis=1)
```

Then **`apply_shield` runs on the result**, before actuation, unchanged.

---

## 9. What you must not claim

- A falling TD error is **not** evidence of policy quality. Offline RL without a
  distribution-shift correction can overestimate confidently and be wrong.
- Override and attention evidence is **synthetic**.
- Appliance power is a proxy: only **685 of 4,124 devices** are measurement-grounded
  (512 REFIT, 173 iAWE). The rest are declared assumptions.
- Thermal parameters are **declared assumptions** bounded by the RESIDE
  envelope. A fitted coefficient was attempted and rejected — it lost to plain
  persistence in 7 of 11 houses.
- REFIT is 20 **UK** homes. iAWE is **one** Delhi home. RESIDE is 11 **Hyderabad**
  houses over 19 days.
- **Do not touch the test split** until you report final numbers.

The dataset carries all of this in `scope_and_limits` inside
`transition_validation.json`. Copy it into the paper rather than paraphrasing.

---

## 10. Order of work

Steps 1–5 are **done**. What remains:

1. ~~Reproduce the baseline run.~~
2. ~~Split the TD metric into terminal and non-terminal.~~ It was flattering.
3. ~~Add CQL.~~ α = 1.0, level-balanced.
4. ~~Fit the Bradley-Terry reward.~~ E2 — read the defect above before quoting it.
5. ~~Add the BC warm start.~~ E5.
6. **Substitute r_ψ into the training reward** and retrain. Currently the reward
   is still the hand-weighted billing reward; the preference model is fitted but
   not applied.
7. **Regenerate the preference pairs in both directions** if E2 is to stand as
   the headline contribution.
8. Evaluate on **validation** only. Ablations → E4.
9. Export weights and the golden vector, hand to hardware.
10. Touch **test** once, at the end.

## 11. Training on Kaggle

`notebooks/sharp_rl_training_kaggle.ipynb` runs the whole thing end to end from
the published dataset — attach `sharp-master-dataset-v2`, Run All, accelerator
**None**. It loads the pre-split parquet files directly, so there is no split
filtering to get wrong.

It is verified by `scripts/verify_kaggle_training_notebook_v1.py`, which extracts
every code cell and executes it locally against the release files, so it cannot
ship broken.

Outputs to `/kaggle/working`:

| File | Use |
|---|---|
| `sharp_policy_bdq_v2.npz` | weights **plus `mean` and `sd`** |
| `golden_vector.json` | Join 1 boot assertion for the Pi |
| `training_report.json` | metrics, ablations, limitations |

The export is checked by reloading it and re-running the forward pass — an export
that does not reproduce the in-memory model fails on the Pi, not here.
