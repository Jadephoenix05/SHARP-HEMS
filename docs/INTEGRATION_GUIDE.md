# SHARP integration guide

**Purpose: let four people build at the same time without waiting for each
other.** Read Part 1 (contracts) in full. Then read only your own lane in
Part 2. Come back for Part 3 when you start joining pieces together.

The whole point is that **nobody waits for hardware and nobody waits for the
trained model.** Every lane has a mock defined here that behaves exactly like
the real thing.

---

# PART 0 — The one rule

> **Freeze the contracts in Part 1 today. Never change one silently.**

Every contract below is already fixed by the shipped dataset, so this costs you
nothing — it is documentation of what exists, not a negotiation.

If a contract must change, it changes in this file first, with a version bump,
announced to all four owners. A contract that drifts silently is how parallel
work turns into three incompatible halves the night before the deadline.

---

# PART 1 — FROZEN CONTRACTS

Everyone codes against these. They come from
`sharp-master-dataset-v2` and are already true.

## 1.1 Appliance registry

The source of truth is `simulator_inputs/device_power_models.parquet`.
Seven appliances in the prototype:

| id | `appliance_type` | Class | Necessity | Dimmable | Watts | Family |
|---|---|---|---|---|---|---|
| `fridge_01` | `refrigerator` | Critical | **yes** | no | 47.7 | thermostatic |
| `fan_01` | `ceiling_fan` | Critical | **yes** | **yes (0.50)** | 60.0 | continuous |
| `light_01` | `led_bulb` | Critical | **yes** | **yes (0.40)** | 9.0 | continuous |
| `ac_01` | `air_conditioner` | Thermostatic | no | no | 1114.6 | thermostatic |
| `washer_01` | `washing_machine` | Deferrable | no | no | 110.1 | cycle |
| `tv_01` | `television` | Interruptible | no | no | 118.5 | continuous |
| `pump_01` | `water_pump` | Interruptible | no | no | 750.0 | task |

**Watts are 15-minute mean power, not nameplate.** A real CT clamp will read
nothing like these during a compressor cycle. That is expected, not a fault.

## 1.2 Action levels

```
0 = OFF      shed the load
1 = ON       full power
2 = REDUCED  dim / low speed   (only where dimmable = yes)
```

Two hard rules, enforced in three independent places:

- **A necessity appliance can never be set to level 0.** Not by the policy, not
  by the shield, not by a human override.
- **Level 2 is illegal on a non-dimmable appliance.** Reject it.

## 1.3 State vector — 305 features

25 global, then 28 device slots × 10 features. Slots are ordered by
`device_id` **ascending** and zero-padded; `device_present` is the mask.

Global order (indices 0–24), exactly:

```
 0 obs_T2M                       13 occupancy_adult_home_fraction
 1 obs_RH2M                      14 attention_available
 2 obs_ALLSKY_SFC_SW_DWN         15 marginal_tariff_inr_kwh_div10
 3 obs_WS10M                     16 mode_grid_import
 4 obs_grid_percentile           17 mode_self_sufficient
 5 obs_grid_peak_severity        18 mode_islanded
 6 fraction_of_day               19 self_sufficient_fraction
 7 month_to_date_kwh_div500      20 battery_state_of_charge
 8 connection_limit_kw           21 pv_generation_kw
 9 indoor_temperature_c_div50    22 unserved_demand_kw
10 degrees_above_comfort_band    23 grid_absent
11 degrees_below_comfort_band    24 recent_override_count_div10
12 household_has_air_conditioner
```

Device order (10 per slot), exactly:

```
0 remaining_service_hours        5 elapsed_state_steps_divided_by_96
1 power_proxy_kw                 6 preferred_service_fraction
2 current_on                     7 is_air_conditioner
3 protected_service              8 steps_since_shed_div96
4 cycle_type                     9 device_override_count_div10
```

Authoritative copy: `rl_transitions/feature_schema.json`. **Load it, never
retype it.**

## 1.4 MQTT topics

```
home/demo/state                          Pi ─► all     retain=true   QoS 0
home/demo/health                         Pi ─► all     retain=true   QoS 0
home/demo/intent                         Pi ─► all                   QoS 0
home/demo/sensor/<id>/power_15min_mean_w Pi ─► all                   QoS 0
home/demo/sensor/<id>/measured_w         Pi ─► all                   QoS 0
home/demo/actuator/<id>/cmd              agent ─► Pi                 QoS 1
home/demo/actuator/<id>/ack              Pi ─► all                   QoS 1
home/demo/override/<id>                  API ─► Pi                   QoS 1
```

`<id>` is the appliance id from 1.1 (`fan_01`, `ac_01`, …).

## 1.5 Message schemas

### `state` — published every 15 min, retained

```json
{
  "schema_version": "sharp_state_v2",
  "house_id": "demo",
  "timestamp_ist": "2026-09-14T19:15:00+05:30",
  "step_id": 77,
  "aggregate_power_kw": 0.412,
  "background_load_kw": 0.084,
  "sanctioned_load_kw": 0.52,
  "indoor_temperature_c": 31.2,
  "outdoor_temperature_c": 34.8,
  "occupancy_adult_home_fraction": 0.83,
  "attention_available": true,
  "marginal_tariff_inr_kwh": 4.50,
  "month_to_date_kwh": 96.4,
  "operating_mode": "grid_import",
  "grid_absent": false,
  "self_sufficient_fraction": 0.0,
  "battery_state_of_charge": 0.5,
  "grid_peak_severity": 0.62,
  "appliances": [
    {
      "appliance_id": "fan_01",
      "appliance_type": "ceiling_fan",
      "is_necessity": true,
      "supports_reduced": true,
      "level": 2,
      "power_15min_mean_w": 30.0,
      "measured_w": 0.019,
      "remaining_service_hours": 3.5
    }
  ],
  "state_vector": [0.0, 0.0],
  "data_age_seconds": 4
}
```

`state_vector` is the full 305 floats for the policy. The dashboard ignores it;
the Pi and any evaluator use it.

### `intent` — what the agent wants, and what the shield allowed

```json
{
  "schema_version": "sharp_intent_v2",
  "command_id": "9f1c...",
  "timestamp_ist": "2026-09-14T19:15:00+05:30",
  "proposed": {"fan_01": 1, "tv_01": 1, "pump_01": 1},
  "executed": {"fan_01": 2, "tv_01": 0, "pump_01": 0},
  "shield_reasons": {
    "tv_01": ["import_capacity_shedding"],
    "fan_01": ["protected_on"]
  },
  "policy_source": "bdq_v2",
  "decision_latency_ms": 34
}
```

### `cmd` — agent to Pi

```json
{
  "schema_version": "sharp_cmd_v2",
  "command_id": "9f1c...",
  "appliance_id": "fan_01",
  "level": 2,
  "issued_at": "2026-09-14T19:15:00+05:30",
  "expires_at": "2026-09-14T19:15:10+05:30"
}
```

### `ack` — Pi to everyone. **Required for every cmd, including rejections.**

```json
{
  "schema_version": "sharp_ack_v2",
  "command_id": "9f1c...",
  "appliance_id": "fan_01",
  "accepted": true,
  "applied_level": 2,
  "rejected_reason": null,
  "gpio_state": "pwm_50",
  "measured_w": 0.019,
  "verification": "MATCH",
  "acked_at": "2026-09-14T19:15:00.240+05:30",
  "latency_ms": 240
}
```

`rejected_reason` is one of: `NECESSITY_MASK`, `COMPRESSOR_LOCKOUT`,
`CYCLE_ACTIVE`, `MIN_ON_TIME`, `MIN_OFF_TIME`, `COMMAND_EXPIRED`,
`LEVEL_UNSUPPORTED`, `GPIO_FAILURE`.

`verification` is `MATCH`, `MISMATCH_STILL_DRAWING`, `MISMATCH_NOT_DRAWING`,
`MISMATCH_WRONG_LEVEL`, or `NO_METER`.

### `override` — API to Pi

```json
{
  "schema_version": "sharp_override_v2",
  "override_id": "3a7b...",
  "appliance_id": "tv_01",
  "requested_level": 1,
  "issued_at": "2026-09-14T19:16:12+05:30",
  "user_id": "resident_01",
  "client_latency_ms": 8400
}
```

`client_latency_ms` is the time from the intent appearing on screen to the user
tapping. **It becomes the weight in the preference-learning reward.** If the
dashboard does not measure it, real deployment data is weaker than the
synthetic data the model trained on.

## 1.6 Two fields that must never be merged

`power_15min_mean_w` (simulated or modelled) and `measured_w` (what the meter or
LED rig actually read) stay **separate, always**.

It is tempting to publish the simulated wattage as "measured" so the numbers
look right. Do that and a stuck relay, a failed GPIO write and a wiring fault
all become invisible, because the fake number says the appliance is running
regardless. Keeping them apart gives you a free check:

```
measured_w > threshold   must agree with   applied_level > 0
```

That disagreement rate is a reportable result.

---

# PART 2 — YOUR LANE, STARTING TODAY

Each lane below can start **now**, with no dependency on the others.

---

## LANE A — Vaishnavi: dashboard and API

### Day 1, no hardware, no model

Build the **replay publisher** first. It makes every other lane testable.

```python
# replay.py - publishes recorded transitions as if a Pi were live
import pandas as pd, json, time, paho.mqtt.client as mqtt

d = pd.read_parquet('sharp-master-dataset-v2/rl_transitions/splits/validation.parquet')
episode = d[d.episode_id == d.episode_id.iloc[0]].sort_values('step_id')

c = mqtt.Client(transport='websockets')
c.username_pw_set('pi_rw', '...')
c.tls_set()
c.connect('xxxxx.s1.eu.hivemq.cloud', 8884)

for _, row in episode.iterrows():
    c.publish('home/demo/state', json.dumps(to_state_message(row)), retain=True)
    time.sleep(1)      # 96 steps = 96 seconds per simulated day
```

`to_state_message(row)` maps dataset columns to the schema in 1.5. Every field
exists in the parquet — nothing needs inventing.

### Your build order

1. Replay publisher (above) — **unblocks Lanes B and C**
2. HiveMQ cluster + three users with ACLs from `HOSTING_AND_CONNECTIVITY_PLAN.md`
3. Next.js on Vercel, browser MQTT, panels 1–6
4. FastAPI on Fly.io: `/api/override`, `/api/history`, `/api/metrics`
5. Failure injection: stale data, duplicates, disconnect, rejected acks

### Definition of done

- [ ] Replay publishes a full 96-step episode on the real broker
- [ ] Dashboard renders **three** appliance states: off / on / **dim**
- [ ] Safety-block panel shows `shield_reasons` in plain words
- [ ] Override POST is **idempotent** on `override_id`
- [ ] `client_latency_ms` is measured and sent
- [ ] Data older than 2 intervals is visibly greyed
- [ ] Refused overrides appear as an explanation, not an error toast

### Mock for what you lack

| Missing | Use instead |
|---|---|
| Pi | replay publisher |
| Policy | `intent.proposed` from the dataset's `requested_action` |
| PZEM | `measured_w = power_15min_mean_w × uniform(0.9, 1.1)` |

---

## LANE B — Harini: Pi actuation

### Day 1, no model, no dashboard

You need **only** the contracts in Part 1 and the shield file from the repo.

```bash
# On the Pi
git clone https://github.com/supriya-07G/SHARP-HEMS.git
cp SHARP-HEMS/scripts/sharp_action_shield.py .
```

**Copy the shield. Do not rewrite it.** Two implementations drift, and the one
that drifts is the one enforcing safety. It is pure Python with no dependencies.

### Your build order

1. GPIO map from `HARDWARE_PROTOTYPE_SPEC.md` §4. **`fan_01` and `light_01` need
   hardware PWM on GPIO12/13** — order parts accordingly
2. Actuator service: subscribe `cmd`, validate, drive GPIO, publish `ack`
3. All 8 local interlocks (spec §6)
4. Watchdog: no valid cmd for 3 intervals → **hold last safe state**
5. Run `test_e8_critical_load_safety.py` **on the Pi**

### Definition of done

- [ ] Every `cmd` produces an `ack`, including every rejection with a reason
- [ ] `test_e8_critical_load_safety.py` on the Pi: **0 violations / 10,000**
- [ ] `fan_01` at level 2 → PWM 50 %; `light_01` at level 2 → PWM 40 %
- [ ] A level-0 command to `fridge_01` is **rejected**, reason `NECESSITY_MASK`
- [ ] A command past `expires_at` is rejected
- [ ] A repeated `command_id` is ignored, not re-applied
- [ ] MQTT killed → relays hold, do **not** all switch off or on
- [ ] Pi timezone is `Asia/Kolkata`, asserted at boot

### Mock for what you lack

Publish commands by hand while you build:

```bash
mosquitto_pub -h ... -t home/demo/actuator/fan_01/cmd -m \
  '{"command_id":"t1","appliance_id":"fan_01","level":2,
    "issued_at":"...","expires_at":"..."}'
```

---

## LANE C — Supriya: state builder and policy

### Day 1

Train on Kaggle against the uploaded dataset. Lane B and C meet at the **golden
vector** (Part 3), so define that early.

### Your build order

1. Train BDQ; report TD **split by terminal and non-terminal**
2. Export `sharp_policy_v2.npz` with `mean`, `sd`, **and the golden vector**
3. Pi-side state builder producing the 305 vector in the exact order of 1.3
4. Fit the Bradley-Terry reward from the 3,180 preference pairs → **E2**
5. Add CQL — the highest-value offline-RL fix

### Definition of done

- [ ] Policy trains; test split **untouched**
- [ ] Export contains `mean`, `sd`, golden state, golden Q, schema hash
- [ ] State builder reproduces a dataset row **bit-for-bit** from its inputs
- [ ] `apply_shield` runs on every action before it is published
- [ ] TD reported separately for terminal and non-terminal steps

### Mock for what you lack

| Missing | Use instead |
|---|---|
| Pi | write `state` to MQTT from a script |
| Real sensors | dataset columns |
| Dashboard | log `intent` to stdout |

---

## LANE D — Charu: sensing

### Day 1, no mains

Develop entirely against recorded data. Your handbook already says live PZEM
work waits for supervision.

### Your build order

1. Meter service reading Modbus RTU, with mocked serial first
2. Sampling schedule, timestamping in **IST**, unit validation, quality flags
3. Publish `measured_w` and `health`
4. Calibration against a known load — **supervised only**

### Definition of done

- [ ] `measured_w` published per appliance per interval
- [ ] Every reading carries a quality flag and an age
- [ ] `health` publishes per-component status, never a single aggregated "OK"
- [ ] A stale reading is flagged stale, not silently republished

---

# PART 3 — WHERE THE LANES MEET

Three joins, in this order. Each has an owner pair and a pass condition.

## Join 1 — Policy ↔ Pi: the golden vector

**Owners: Supriya + Harini. This is the most dangerous join in the project.**

The policy expects 305 features in an exact order, normalised with an exact
`mean` and `sd`. If the Pi assembles them differently, inference **does not
crash** — it returns confident nonsense. Relays click. The dashboard looks fine.
Nothing tells you.

Supriya exports, at training time:

```python
golden = {
    'state': X_validation[0].tolist(),     # 305 floats
    'expected_q': q.tolist(),              # 28 x 3
    'expected_action': action.tolist(),    # 28 ints
    'schema_sha256': sha256_of(feature_schema.json),
}
```

Harini runs, at every Pi boot, **before accepting any command**:

```python
q = infer(golden['state'])
assert np.allclose(q, golden['expected_q'], atol=1e-6), 'POLICY MISMATCH'
assert pi_schema['global_features'] == training_schema['global_features']
assert pi_schema['device_features'] == training_schema['device_features']
```

Compare feature **names**, not just the count. Two schemas can both be 305 long
and disagree.

**Pass:** Q matches to 1e-6 and both name lists are identical. On failure the Pi
refuses to actuate and publishes to `health`.

## Join 2 — Pi ↔ Dashboard: the ack loop

**Owners: Harini + Vaishnavi.**

**Pass:** every `cmd` produces an `ack` within 500 ms (p99), rejections included,
and the dashboard renders the rejection reason in words.

## Join 3 — Dashboard ↔ Pi: the override round trip

**Owners: Vaishnavi + Harini.**

```
tap → POST /api/override → MQTT override → Pi re-runs apply_shield
    → honoured or refused → ack → dashboard shows outcome
```

Only **8 %** of overrides are honoured in the dataset — the shield keeps a
safety or capacity constraint in the other 92 %. That is correct. A refused
override is **data, not an error**, and must reach the screen as an explanation.

**Pass:** honoured and refused overrides both round-trip and both display
correctly.

---

# PART 4 — INTEGRATION TESTS

Run in order. Each gates the next.

| # | Test | Owner | Pass |
|---|---|---|---|
| I1 | Golden vector on the Pi | Supriya + Harini | Q matches to 1e-6 |
| I2 | Schema names match | Supriya + Harini | exact list equality |
| I3 | Pi timezone | Harini | `Asia/Kolkata` |
| I4 | E8 on the Pi | Harini | **0 / 10,000** |
| I5 | cmd → ack latency | Harini + Vaishnavi | < 500 ms p99 |
| I6 | Actuation verification | Harini + Charu | mismatch < 1 % |
| I7 | **Necessity mask end to end** | all | fridge-shed refused at API, shield **and** Pi |
| I8 | Dim path end to end | all | level 2 → PWM 50 % → measured ≈ 50 % |
| I9 | Outage | all | AC/TV dead; fan/light continue on battery |
| I10 | Watchdog | Harini | MQTT killed → holds state, does not fail open |
| I11 | Replay determinism | Supriya | same episode twice → identical actions |
| I12 | 24 h soak | all | no memory growth, no drift |

**I7 is the headline.** Demonstrate all three layers refusing, separately.
**I10 is the one teams forget**, and it is a safety property.

---

# PART 5 — DEMO SCRIPT

1. **Normal** — evening, everything on.
2. **Peak** — severity crosses 0.5. Fan drops to **dim**, TV sheds, fridge and
   lights stay on. *This is the money shot.*
3. **Override** — user restores the TV. Honoured. Preference pair recorded.
4. **Blocked override** — user tries to shed the fridge. **Refused**, reason on
   screen.
5. **Outage** — grid drops. AC and TV go dead; fan and lights continue on the
   inverter. Objective flips to battery runway.
6. **Recovery** — grid returns, battery recharges from mains.

Step 2 shows what a rule-based shedder cannot do. Step 5 is what separates this
from a UK-style demo — and the dataset backs it with 28,116 outage steps derived
from IRES-reported supply hours.

---

# PART 6 — HONESTY RULES THAT SURVIVE INTO THE DEMO

- Simulated and measured power stay **separate fields**, always.
- Simulator-sourced values on the Pi are flagged in `health`.
- Overrides captured live are **real**; overrides in training are **synthetic**.
  Never pool them without a source column.
- Appliance wattages are **15-minute means**, not nameplate ratings.
- A working demo is **not** evidence of policy quality. Report the E-series
  experiments, not the demo.
- Nothing here is approved for mains control without supervision.
