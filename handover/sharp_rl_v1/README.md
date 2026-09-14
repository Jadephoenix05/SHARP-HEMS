# SHARP RL policy — hardware handover

Everything the Raspberry Pi needs. No SHARP module, no framework, numpy only.

| File | What it is |
|---|---|
| `sharp_policy.npz` | weights **plus the normalisation vectors** |
| `golden_vector.json` | boot assertion — run this before accepting any command |
| `training_report.json` | how it was trained, and its limits |
| `run_policy.py` | reference inference, ~40 lines |

---

## 1. The contract

| | |
|---|---|
| Input | **305 floats** — 25 global + 28 device slots × 10 |
| Output | **28 branches × 3 levels** |
| Levels | `0 = OFF (shed)` · `1 = ON` · `2 = REDUCED` |
| Size | 50,133 parameters, **196 KB** float32 |
| Inference | three matrix multiplies, well under 1 ms |

**Level 2 is never selected in this build.** A critical load is never shed and
never dimmed while the occupant wants it, and the only non-critical dimmable
appliance in these homes is the air cooler. Two states reach the relays: ON and
SHED.

---

## 2. Run it at boot, before anything else

```python
import json, numpy as np

w = np.load('sharp_policy.npz')
g = json.load(open('golden_vector.json'))

x = np.asarray(g['state'], float)               # already normalised in the file
h = np.maximum(0, x @ w['w'] + w['b'])
q = (h @ w['v'] + w['vb']).reshape(1, 1) + (h @ w['a'] + w['ab']).reshape(28, 3)
q -= q.mean(axis=1, keepdims=True)

assert np.allclose(q, np.asarray(g['expected_q'], float), atol=1e-8)

allowed = np.asarray(g['device_present'], bool)[:, None] & np.asarray(g['legal_levels'], bool)
action = np.argmax(np.where(allowed, q, -np.inf), axis=1)
assert (action[np.asarray(g['device_present'], bool)]
        == np.asarray(g['expected_greedy_action'])[np.asarray(g['device_present'], bool)]).all()
print('golden vector OK')
```

**If this fails, do not actuate anything.** It means the Pi is building a
different state vector from the one the model was trained on. That failure does
not announce itself any other way — the relays click, the dashboard looks
correct, and the decisions are confident nonsense.

---

## 3. Live inference

```python
x = (state_305 - w['mean']) / w['sd']           # normalise with the SHIPPED vectors
h = np.maximum(0, x @ w['w'] + w['b'])
q = (h @ w['v'] + w['vb']).reshape(1, 1) + (h @ w['a'] + w['ab']).reshape(28, 3)
q -= q.mean(axis=1, keepdims=True)

legal = np.ones((28, 3), bool)
legal[:, 2] = supports_reduced & ~is_necessity   # nothing critical is ever dimmed
legal[:, 0] &= ~(is_necessity & occupant_wants)  # nothing critical in use is shed
action = np.argmax(np.where(device_present[:, None] & legal, q, -np.inf), axis=1)

action = apply_shield(devices, action)           # the shield runs LAST and may refuse
```

Three rules, in order of how badly they bite:

1. **Normalise with the shipped `mean` and `sd`.** Anything else is silently
   wrong — no exception, no warning, just bad commands.
2. **The shield runs after the model, never before, and it may refuse.** The
   model proposes; the shield decides; the relay acts.
3. **Protection is conditional.** A fridge nobody is asking for at 3 a.m. is
   correctly OFF. Protecting essential service does not mean running it around
   the clock.

---

## 4. The four load classes

From the idea book: *Critical loads are never interrupted. Thermostatic loads
can be adjusted inside comfort bands. Deferrable loads can move to another time.
Interruptible loads can pause briefly.*

| Class | Rule | `appliance_type` values |
|---|---|---|
| **Critical** | never shed, never dimmed, when in use | `ceiling_fan` `table_fan` `led_bulb` `led_tube` `cfl_bulb` `cfl_tube` `incandescent_bulb` `refrigerator` `water_purifier` `modem_router` |
| **Thermostatic** | on/off + **advisory** setpoint | `air_conditioner` `geyser` |
| **Deferrable** | shed, but never mid-cycle | `washing_machine` `electric_rice_cooker` |
| **Interruptible** | shed freely | `television` `mixer_grinder` `water_pump` `air_cooler` `electric_iron` `laptop_tablet` `desktop` `electric_kettle` |

`Critical` is a *permission*; the other three are *dynamics*. A refrigerator is
both Critical and thermostatic, and that is not a contradiction: SHARP leaves it
alone, and compressor protection still applies.

**The model reads each device's features, not its name.** You can change which
seven appliances are in the rig without retraining — just keep the
`appliance_type` string matching this list so the Pi looks up the right flags.

---

## 5. Prototype rig — 7 LEDs, no PWM

| # | Appliance | Class | BCM | Commandable |
|---|---|---|---|---|
| 1 | Ceiling fan | Critical | 17 | **No** |
| 2 | LED light | Critical | 27 | **No** |
| 3 | Refrigerator | Critical + thermostatic | 22 | **No** |
| 4 | Air conditioner | Thermostatic | 16 | 0 / 1 |
| 5 | Washing machine | Deferrable | 20 | 0 / 1, min-on 4 steps |
| 6 | Television | Interruptible | 21 | 0 / 1 |
| 7 | Mixer grinder | Interruptible | 26 | 0 / 1 |

Seven LEDs, seven 220 Ω resistors, plain digital outputs. **GPIO12 and 13 are
left free** — they are the hardware-PWM channels, so dimming can be added later
without rewiring.

---

## 6. What this model is, and is not

**Is:** a validated offline policy that beats the rule-based baseline on peak
demand and peak-to-average ratio, and has never degraded a protected load.

| | |
|---|---|
| Balanced accuracy | vs a 0.333 chance floor |
| Critical loads degraded | **0** of 276,226 protected device-steps |
| Peak | 0.3895 kW against 0.4085 uncontrolled |
| Peak-to-average | 2.558 against 2.593 uncontrolled |
| Validation gates | **17 / 17** |

Peak reduction of 4.65 per cent is the relief a rolling blackout would get by
cutting supply to **4.7 homes in 100** — delivered without cutting anyone off.

**Is not:** field-proven. Every training signal is simulated, the occupant model
is synthetic, and 3,439 of 4,124 device power values are declared assumptions.
`approved_for_deployment` is `false` in every report, deliberately.

Run it in **shadow mode** first: publish `intent`, let the rule-based controller
actuate. Real state vectors, real timing, zero risk — and the only honest route
to preference data from real people.
