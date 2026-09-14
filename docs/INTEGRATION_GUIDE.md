# SHARP integration guide

How the RL model, the Raspberry Pi and the dashboard fit together, and the
handoff contract between the three owners.

Read `HARDWARE_PROTOTYPE_SPEC.md`, `RL_MODEL_SPEC.md` and `DASHBOARD_GUIDE.md`
first. This document is only about the seams between them.

---

## 1. Who owns what

| Owner | Owns | Explicitly does not own |
|---|---|---|
| **Supriya** | State builder, BDQ policy, shield, appliance registry | GPIO, web UI |
| **Harini** | Actuation, GPIO, local interlocks, acknowledgements | Training, UI |
| **Charu** | PZEM telemetry, calibration, sensing health | Decisions, actuation |
| **Vaishnavi** | FastAPI, dashboard, override capture, metrics | Decisions, GPIO |

The seams are where this project will break, not the components. All four
handbooks say the same thing in different words: *publishing a command is not
proof that a relay moved.*

---

## 2. The 15-minute loop

```
t=0s    GATHER    read MQTT retained topics → build the 305-dim state
t=0.1s  SHIELD    compute legal levels per device (apply_shield)
t=0.2s  DECIDE    Q = policy(state); argmax over legal levels per branch
t=0.3s  PROJECT   apply_shield again on the chosen action
t=0.4s  PUBLISH   intent → MQTT (dashboard renders it)
t=0.5s  COMMAND   per-appliance cmd with command_id and expires_at
t=0.5–2s ACTUATE  Pi validates locally, drives GPIO/PWM, publishes ack
t=2–900s WAIT     accept overrides; re-shield and re-actuate if one arrives
t=900s  RECORD    write the transition; next step
```

The shield runs **twice**: once to build the legal set before the policy
chooses, once on the chosen action before it leaves the Pi. Then the Pi's own
interlocks check it a third time. That redundancy is deliberate.

---

## 3. The handoff artifacts

Four things move from the RL side to the hardware side. Version all four
together; a mismatch is silent.

| Artifact | From | Used by |
|---|---|---|
| `sharp_policy_v2.npz` | `train_sharp_bdq_v1.py` | Pi inference |
| `feature_schema.json` | the dataset release | Pi state builder |
| `device_power_models.parquet` | `configure_sharp_appliance_models.py` | Pi registry, dashboard |
| `sharp_action_shield.py` | the repo, **unchanged** | Pi, verbatim |

### Ship the shield, do not reimplement it

The single most dangerous thing this project could do is rewrite the shield in
another language for the Pi. Two implementations drift, and the one that drifts
is the one enforcing safety. `sharp_action_shield.py` is pure Python with no
dependencies — copy the file.

`test_e8_critical_load_safety.py` then runs **on the Pi**, unmodified. If it
does not return 0 violations there, the Pi is not ready.

---

## 4. The failure this project is most likely to have

**A state-builder mismatch between training and the Pi.**

The policy expects 305 features in an exact order, normalised with an exact
`mean` and `sd`. If the Pi assembles them in a different order, or normalises
differently, inference does not crash — it returns confident nonsense. Relays
click. The dashboard looks fine. Nothing tells you.

### The guard

Ship a golden test vector with the weights:

```python
# built once, at export time, from a known validation row
golden = {
    'state': X_validation[0].tolist(),          # 305 floats
    'expected_q': q.tolist(),                   # 28 x 3
    'expected_action': action.tolist(),         # 28 ints
    'policy_version': 'bdq_v2',
    'schema_sha256': sha256(feature_schema.json),
}
```

The Pi runs this **at every boot**, before accepting any command:

```python
q = infer(golden['state'])
assert np.allclose(q, golden['expected_q'], atol=1e-6), 'POLICY MISMATCH'
```

If it fails, the Pi refuses to actuate and publishes to `health`. Ten lines,
and it catches the one bug that is otherwise invisible.

### Also verify the feature order by name

```python
assert pi_schema['global_features'] == training_schema['global_features']
assert pi_schema['device_features'] == training_schema['device_features']
```

Compare names, not just the count. Two schemas can both be 305 long and disagree.

---

## 5. Building the state on the Pi

The Pi must produce the same vector the simulator produced. Where a value is not
measurable, it comes from the simulator on the same 15-minute tick — the idea
book is explicit that the Pi does not care whether a number arrived from a sensor
or a simulator, because it reads the same MQTT topic either way.

| Feature group | Source on the Pi |
|---|---|
| Weather (4) | Simulator, or a weather API with the same units |
| Grid percentile, severity (2) | Simulator, from historical labels |
| `fraction_of_day` | Pi clock, **IST** |
| `month_to_date_kwh` | Pi billing ledger, persisted across reboots |
| `connection_limit_kw` | Config |
| Indoor temperature, comfort band (3) | Sensor if present, else thermal model |
| Occupancy, attention (2) | Simulator, or a PIR sensor |
| Marginal tariff | `sharp_apcpdcl_tariff.next_unit_rate(month_kwh)` |
| Operating mode (3) | Measured: grid present? PV? battery? |
| ρ, SoC, PV, unserved, grid_absent (5) | Measured, or 0 with a flag |
| Override count | Pi's own counter |
| Per device (10 × 28) | Registry + Pi timers + PZEM |

### Two things that will silently break it

**Timezone.** `fraction_of_day` is IST. A Pi in UTC shifts every time-dependent
feature by 5.5 hours, and the policy will behave as though it is the middle of
the night. Set the Pi to `Asia/Kolkata` and assert it at boot.

**Device order.** Devices are ordered by `device_id` **ascending**, and padded to
28 slots with zeros. `device_present` is the mask. If the Pi enumerates in
discovery order instead, every device feature lands in the wrong slot.

---

## 6. Actuation verification

This is what turns a light show into a cyber-physical result.

```
commanded_level  →  GPIO write  →  ack  →  measured_w
```

The Pi must check the last arrow:

| Commanded | Expected measured | Verdict if not |
|---|---|---|
| 0 (off) | ≈ 0 W | `ACTUATION_MISMATCH_STILL_DRAWING` |
| 1 (on) | ≈ rated | `ACTUATION_MISMATCH_NOT_DRAWING` |
| 2 (dim) | ≈ rated × fraction | `ACTUATION_MISMATCH_WRONG_LEVEL` |

Publish the verdict on `ack`. Count mismatches in the metrics. **Never repair a
mismatch by overwriting `measured_w` with the expected value** — the whole point
of keeping the two fields separate is that the disagreement is the finding.

A mismatch rate is a reportable result. In a paper it is the difference between
"we commanded" and "we verified".

---

## 7. Override round trip

```
dashboard tap
  → POST /api/override  (override_id, level, client_latency_ms)
  → FastAPI publishes home/<id>/override/<appliance>
  → Pi re-runs apply_shield WITH human_actions
  → honoured or refused
  → ack carries human_override_honored
  → dashboard shows the outcome, including refusals
  → transition records the preference pair
```

### Refused overrides are data, not errors

`apply_shield` already returns `human_override_honored` per device. In the
dataset only **8 %** of overrides are honoured — the shield keeps a safety or
capacity constraint the user tried to breach in the other 92 %. That is correct behaviour and it
must reach the dashboard as an explanation, not disappear.

### Capture latency

`client_latency_ms` — intent shown to user tapping — becomes `latency_steps` in
the preference pair, and the weight is
`occupancy × pressure ÷ (1 + latency_steps)`. Without it, real deployment data
is weaker than the synthetic data the model was trained on.

---

## 8. Integration test plan

Run in order. Each gates the next.

| # | Test | Pass condition |
|---|---|---|
| I1 | Golden vector on the Pi | Q matches training to 1e-6 |
| I2 | Schema names match | Exact list equality |
| I3 | Timezone | Pi clock is IST |
| I4 | E8 on the Pi | 0 violations / 10,000 |
| I5 | Command → ack | < 500 ms, p99 |
| I6 | Actuation verification | Mismatch rate < 1 % on the LED rig |
| I7 | Necessity mask end to end | Fridge shed command refused at API, shield **and** Pi |
| I8 | Dim path end to end | Level 2 → PWM 50 % → measured ≈ 50 % |
| I9 | Outage | Grid off → AC/TV dead, fan/light continue on battery |
| I10 | Watchdog | MQTT killed → Pi holds last safe state, does not fail open |
| I11 | Replay determinism | Same recorded episode twice → identical actions |
| I12 | 24 h soak | No memory growth, no SD wear alarm, no drift |

**I7 is the headline.** A fridge-shed command must be refused at three
independent layers. Demonstrate all three refusing, separately.

**I10 is the one teams forget.** A controller that fails open during a network
drop is more dangerous than no controller.

---

## 9. Milestone order

1. **Contracts frozen** — schema, topics, command/ack. Everyone unblocks.
2. **Replay harness** — Vaishnavi publishes recorded transitions. Dashboard and
   Pi both develop against real data with no hardware.
3. **LED rig** — 7 LEDs, 2 on PWM, all interlocks, E8 on the Pi.
4. **Policy export** — golden vector, I1–I3 pass.
5. **Closed loop on LEDs** — I4–I8.
6. **PZEM, supervised** — calibration evidence.
7. **Dashboard live** — Vercel + FastAPI against the real Pi.
8. **Demo rehearsal** — the six-step script, twice, end to end.

Steps 1 and 2 unblock everyone and need no hardware. Do them this week.

---

## 10. Honesty rules that survive into deployment

These are not paperwork. They are what stops the demo from claiming more than it
shows.

- The dashboard shows **simulated** and **measured** power as separate fields.
- Any simulator-sourced value on the Pi is flagged as such in `health`.
- Overrides captured in deployment are **real**; overrides in the training data
  are **synthetic**. Never pool them without a source column.
- Appliance wattages are 15-minute means, not nameplate ratings. Label the topic
  `power_15min_mean_w`. A real CT clamp will read nothing like these, and that
  is expected.
- A working demo is not evidence of policy quality. Report the E-series
  experiments, not the demo.
- Nothing in this stack is approved for hardware control decisions on real
  mains without supervision. Every module says so in its own validation report,
  and that stays true until a qualified electrician signs it off.
