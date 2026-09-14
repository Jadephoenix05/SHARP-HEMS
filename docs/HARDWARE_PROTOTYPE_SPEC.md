# SHARP hardware prototype specification

For the team meeting. Every appliance, pin, topic and safety rule the prototype
needs, cross-checked against the shipped dataset and the project handbooks.

**Safety boundary, from Harini's handbook, unchanged:** do not work on exposed
230 V mains. Stage 1 is LEDs and low-voltage loads only. PZEM or mains work
requires a supervised lab or a qualified electrician, with enclosure, protection
and isolation.

---

## 1. What the prototype must demonstrate

The dataset defines four control classes. The prototype has to make all four
physically visible, or the demo proves less than the dataset supports.

| Class | Dataset field | Behaviour to show |
|---|---|---|
| Necessity | `is_necessity = True` | **Never shed.** Masked out of the action space. |
| Thermostatic | `dynamics_family = thermostatic` | Setpoint control; compressor duty is an output |
| Deferrable | `dynamics_family = cycle` | Cannot be interrupted once started |
| Interruptible | the rest | Shed or dim per step |

And the three action levels the shield now supports:

```
0 = OFF       shed
1 = ON        full power
2 = REDUCED   dimmed or low speed
```

Level 2 is the point of the whole design: under grid stress SHARP **dims a fan
or light rather than switching it off**. A demo that only shows on/off has not
shown SHARP.

---

## 2. Appliance list — 7 devices

Six controllable plus one critical indicator, matching the idea book's
prototype table.

| # | Appliance | Class | Dataset `appliance_type` | Levels | Sim power | Rig |
|---|---|---|---|---|---|---|
| — | Refrigerator | **Necessity** | `refrigerator` | *not in action space* | 47.7 W | LED, always lit |
| 1 | Ceiling fan | **Necessity, dimmable** | `ceiling_fan` | 0 blocked, 1, 2 | 60 W | LED on **PWM** |
| 2 | LED light | **Necessity, dimmable** | `led_bulb` | 0 blocked, 1, 2 | 9 W | LED on **PWM** |
| 3 | Air conditioner | Thermostatic | `air_conditioner` | setpoint | 1114.6 W | LED + relay, **never cut mid-compressor** |
| 4 | Washing machine | Deferrable | `washing_machine` | 0, 1 | 110.1 W | LED + relay, min-on 4 steps |
| 5 | Television | Interruptible | `television` | 0, 1 | 118.5 W | LED + relay |
| 6 | Water pump | Interruptible | `water_pump` | 0, 1 | 750 W | LED + relay (largest sheddable) |

**Two of the seven need PWM, not a relay.** The fan and the LED light must dim.
Tell whoever orders parts today — discovering this after the relay board arrives
costs a week.

### Why these seven

- Every control class appears exactly once.
- Fridge and fan prove the necessity mask: the shield refuses to shed them even
  under a direct human override. That is **E8**, and it passes 10,000/10,000.
- Fan and light prove dimming, which is where SHARP differs from a rule-based
  shedder.
- Pump is 750 W, large enough that capacity shedding is visible.
- AC proves setpoint control with duty cycle as an output.

### What NOT to include

Geyser (2 kW) is tempting because the idea book calls it the ideal first real
load — purely resistive, genuinely deferrable. **But only 13 of 498 households
in the dataset own one.** Including it in a demo billed as a typical AP home
would quietly undo the geyser-ownership finding. Keep it for a later mains
stage, clearly labelled.

---

## 3. Bill of materials

### Stage 1 — LED rig, no mains (build this first)

| Item | Qty | Notes |
|---|---|---|
| Raspberry Pi 4, 2–4 GB | 1 | Pi Zero 2 W is tight for inference; Pi 4 gives headroom |
| **Passive aluminium heatsink case** | 1 | The *official* case throttles to 428 MHz. Use FLIRC-style. |
| microSD 32 GB A2, or USB SSD | 1 | SSD strongly preferred — see SD-card wear below |
| 5 V 3 A USB-C supply | 1 | Official supply; brownouts corrupt cards |
| 5 mm LEDs | 7 | 2 must be on PWM-capable pins |
| Resistors 220 Ω | 7 | |
| Breadboard + jumpers | 1 set | |
| 8-channel opto-isolated relay board, 5 V | 1 | Stage 2 only; opto-isolation is not optional |

### Stage 3+ — sensing and mains (supervised only)

| Item | Qty | Notes |
|---|---|---|
| PZEM-004T v3.0 + CT clamp | 1 | Modbus RTU over UART |
| USB–TTL adapter (CP2102/CH340) | 1 | Keep PZEM off the Pi's own UART |
| Enclosure, DIN rail, MCB, ferrules | 1 set | Required before any mains work |

---

## 4. GPIO map

BCM numbering, 3.3 V logic. Relay boards are typically **active-low** — verify
yours before wiring, because an inverted board energises every load at boot.

| Appliance | BCM pin | Mode | Note |
|---|---|---|---|
| Refrigerator indicator | 17 | digital out | Always on. Never commandable. |
| Ceiling fan | **12** | **PWM (hardware)** | GPIO12 is hardware-PWM capable |
| LED light | **13** | **PWM (hardware)** | GPIO13 is hardware-PWM capable |
| Air conditioner | 16 | digital out | |
| Washing machine | 20 | digital out | |
| Television | 21 | digital out | |
| Water pump | 26 | digital out | |
| Override button (optional) | 6 | input, pull-up | Physical override capture |

GPIO12 and GPIO13 are the hardware-PWM channels. Software PWM on other pins
flickers visibly under CPU load, which looks like a fault during a demo.

PWM duty for level 2 comes from the dataset's `reduced_power_fraction`:

| Appliance | Fraction | PWM duty |
|---|---|---|
| ceiling_fan | 0.50 | 50 % |
| table_fan | 0.50 | 50 % |
| air_cooler | 0.55 | 55 % |
| led_bulb / led_tube | 0.40 | 40 % |
| incandescent_bulb | 0.50 | 50 % |

CFL is deliberately absent — it dims poorly, and the dataset excludes it rather
than pretending otherwise.

---

## 5. MQTT contract

Matching the idea book's topic layout.

```
home/<house_id>/sensor/<appliance_id>/power_15min_mean_w   simulated or measured
home/<house_id>/sensor/<appliance_id>/measured_w           real LED/PZEM reading
home/<house_id>/sensor/source/grid
home/<house_id>/sensor/source/pv
home/<house_id>/sensor/source/battery_soc
home/<house_id>/state                                      full 305-dim state
home/<house_id>/intent                                     proposed action + reason
home/<house_id>/actuator/<appliance_id>/cmd                agent -> device
home/<house_id>/actuator/<appliance_id>/ack                device -> agent
home/<house_id>/override/<appliance_id>                    dashboard -> agent
home/<house_id>/health
```

### Keep simulated and measured on separate topics

It is tempting to publish the simulated appliance wattage in place of the LED's
real reading, so the numbers "look right". **Do not.** Harini's handbook is
explicit that publishing a command is not proof a relay moved. If you overwrite
the LED's real 0.02 W with a fake 1,115 W, a stuck relay, a failed GPIO write or
a wiring fault all become invisible, because the fake number says the AC is
running regardless.

Two topics gives you a free and genuinely valuable check:

```
measured_w > threshold   must agree with   commanded_state
```

That is an actuation-verification result you can report.

### Command and acknowledgement schema

```json
{
  "command_id": "uuid4",
  "house_id": "demo",
  "appliance_id": "ceiling_fan_01",
  "level": 2,
  "issued_at": "2026-09-14T19:15:00+05:30",
  "policy_source": "bdq_v2",
  "shield_reasons": ["protected_on"],
  "expires_at": "2026-09-14T19:15:10+05:30"
}
```

```json
{
  "command_id": "uuid4",
  "appliance_id": "ceiling_fan_01",
  "accepted": true,
  "applied_level": 2,
  "rejected_reason": null,
  "gpio_state": "pwm_50",
  "measured_w": 0.019,
  "acked_at": "2026-09-14T19:15:00.240+05:30",
  "latency_ms": 240
}
```

`expires_at` matters: a command that arrives late after an MQTT reconnect must
not be applied to a world that has moved on.

---

## 6. Local safety interlocks

These run **on the Pi**, independently of the RL model. The Pi must refuse an
unsafe command even if the policy, the network and the dashboard all say
otherwise. This is the second half of the shield; the first half runs in
`sharp_action_shield.py`.

| # | Interlock | Rule |
|---|---|---|
| 1 | Necessity mask | Reject any command setting a necessity appliance to level 0 |
| 2 | Compressor protection | AC and fridge: enforce **minimum 3 min off** before restart |
| 3 | Cycle protection | Reject OFF while `noninterruptible_cycle_active` |
| 4 | Minimum on/off | Enforce `min_on_steps` / `min_off_steps` locally |
| 5 | Command expiry | Reject a command past `expires_at` |
| 6 | Level validity | Reject level 2 on an appliance with `supports_reduced = False` |
| 7 | Watchdog | No valid command for 3 intervals → **hold last safe state**, do not fail open |
| 8 | Boot state | All relays de-energised at boot, before any command is accepted |

Interlock 2 is the one that damages hardware if skipped. A compressor restarted
against head pressure can stall; on a real AC that is an expensive failure. The
dataset's 15-minute step already prevents this by construction, but the Pi must
enforce it independently, because a replayed or malformed command can arrive at
any time.

Every rejection is published on `ack` with a reason and logged. Per the idea
book: **violations become evidence and never become executable actions.**

---

## 7. Build stages

| Stage | Scope | Gate to proceed |
|---|---|---|
| 0 | Design review, GPIO map, command schema signed off | Supriya approves appliance registry and critical-load list |
| 1 | 7 LEDs on breadboard, no mains | All 8 interlocks pass; E8-style attack test on the Pi |
| 2 | Pi services, MQTT, replay telemetry | Command → ack round trip under 500 ms |
| 3 | PZEM under supervision | Calibration evidence against a known load |
| 4 | One controlled mains circuit | Enclosure, protection and acceptance checks complete |

**Do not skip Stage 1.** Every interlock, the watchdog and the whole MQTT
contract can be proven on LEDs, at zero risk, in a day.

---

## 8. Two things that will bite you

**SD-card wear.** The replay buffer writing every 15 minutes will kill a
microSD. Keep the buffer in a RAM ring buffer, checkpoint every 15 min, and boot
from a USB SSD if you can.

**Thermal throttling.** The official Pi case throttles to 428 MHz, a 71 % cut.
A passive aluminium case does not throttle at all. Budget ₹400–600 and avoid
debugging phantom latency for a week.

---

## 9. What the hardware team needs from the RL side

Bring these to the meeting:

1. **The appliance registry** — `device_power_models.parquet` with
   `is_necessity`, `supports_reduced`, `reduced_power_fraction`, `dynamics_family`.
2. **The exported policy** — `models/sharp_bdq_v2/checkpoint.npz` plus the
   normalisation vectors.
3. **The feature schema** — `feature_schema.json`, 305 features, because the Pi
   must build a byte-identical state vector.
4. **The shield module** — `sharp_action_shield.py` runs unchanged on the Pi.
   Do not reimplement it; a reimplementation is a second thing that can diverge.

See `INTEGRATION_GUIDE.md` for the handoff contract.
