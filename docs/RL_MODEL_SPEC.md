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

### What to add, in order of value

1. **Conservative Q-Learning (CQL).** One extra loss term that pushes down Q on
   actions not in the data. Roughly ten lines, and the single highest-value
   change you can make.
2. **Behaviour cloning warm start.** Pre-train the advantage head to predict the
   logged action, then fine-tune. This is also what unblocks **E5**.
3. **Shield-consistent targets.** The current trainer masks padded branches but
   does *not* re-apply the joint shield when selecting the next action, so a
   target can be bootstrapped from an action the shield would refuse. Fixing
   this needs nested device state, which the flat release does not carry.
   Documented as a known limitation.

### Hyperparameters that work

| | Value | Why |
|---|---|---|
| Hidden | 128 | 50k params; larger overfits 279k rows |
| Batch | 256 | |
| LR | 1e-3 with gradient-norm clip 10 | |
| γ | 0.99 | 96-step episodes |
| Target sync | every 250 updates | |
| Reward scale | ÷ 10 | see the terminal-reward warning below |

### ⚠ Terminal rewards are ~77× step rewards

| | mean | sd |
|---|---|---|
| Non-terminal (99 %) | −0.23 | 0.58 |
| Terminal (step 95) | **−17.68** | **31.15** |

The unmet-service penalty (weight 10 × remaining hours) lands as one lump at
step 95. Two consequences:

- Your **validation TD error is dominated by 1 % of rows**, so the headline TD error flatters
  the policy. Report TD separately for terminal and non-terminal.
- Learning is slower, because the value function must represent a huge spike.

If you want a cleaner signal, spread the unmet-service penalty per step or drop
its weight from 10 to ~2, and regenerate. That is a config change plus an
8-minute rebuild.

---

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

### Building the Bradley-Terry preference reward

This is your headline contribution. The data is there: **3,180 preference
pairs** in `rl_transitions/override_preference_pairs.parquet`.

```python
pairs = pd.read_parquet('.../override_preference_pairs.parquet')
# proposed_action=0 (what the controller did) is DISPREFERRED
# preferred_action=1 (what the user wanted)  is PREFERRED
# preference_weight = occupancy_gating × pressure ÷ (1 + latency_steps)
```

Fit `r_ψ(s, a)` by maximising the weighted Bradley-Terry likelihood, then use
`−(cost + peak) + r_ψ` as the reward. That is **E2**, the headline figure.

Each pair carries `latency_steps` (0–3, and it varies), `occupancy_adult_home_fraction`,
`pressure_source`, `operating_mode` and `override_honoured` — everything the
weighting in your abstract needs.

**Every pair is synthetic**, generated from a stated behavioural rule in
`sharp_human_model.py`. No dataset on earth records real demand-response
overrides in an Indian home; that is what the Pi deployment produces. Label it
as synthetic in the paper, every time.

---

## 7. Experiments

| | Experiment | Status |
|---|---|---|
| E1 | SHARP vs rule-based / MILP / flat DQN / PPO | ✅ 3 behaviour policies available as baselines |
| **E2** | **Reward recovery θ̂ vs θ\*** | ✅ 3,180 preference pairs |
| E3 | Override rate over training rounds | ❌ needs an online loop |
| E4 | Ablations: −shield, −latency, −ranking, −censoring | ✅ latency varies {0,1,2,3} |
| E5 | Sample efficiency with/without BC warm start | ❌ training-procedure work |
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

1. Load the data, reproduce the baseline run (~1 min on a laptop).
2. Split the TD metric into terminal and non-terminal. You will immediately see
   the reported number was flattering.
3. Add **CQL**. Biggest single win.
4. Fit the **Bradley-Terry reward** from the preference pairs → E2.
5. Add the **BC warm start** → E5.
6. Evaluate on **validation** only. Ablations → E4.
7. Export weights, hand to hardware.
8. Touch **test** once, at the end.
