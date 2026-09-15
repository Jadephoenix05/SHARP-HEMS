# SHARP hardware prototype specification

For the team meeting. Every appliance, pin, topic and safety rule the prototype
needs, cross-checked against the shipped dataset and the project handbooks.

**Safety boundary, from Harini's handbook, unchanged:** do not work on exposed
230 V mains. Stage 1 is LEDs and low-voltage loads only. PZEM or mains work
requires a supervised lab or a qualified electrician, with enclosure, protection
and isolation.

---

## 1. What the prototype must demonstrate

The idea book's demand-response vocabulary defines four load classes. All four
must be physically visible, or the demo proves less than the dataset supports.

| Class | Rule | Devices in the 431 target homes |
|---|---|---|
| **Critical** | Never interrupted | **2,586** |
| **Thermostatic** | Adjusted inside comfort bands | 30 |
| **Deferrable** | Moved to another time | 63 |
| **Interruptible** | Paused briefly | 674 |

Critical is a *permission* — may SHARP touch it? The other three are *dynamics* —
how does it respond? A refrigerator is both Critical and thermostatic, and that
is not a contradiction: SHARP leaves it alone, and its compressor protection
still applies.

### Action levels, and the decision that fans and lights are never dimmed

```
0 = OFF       shed
1 = ON        full power
2 = REDUCED   dim / low speed  - NOT used on any critical load
```

A fan, a light or a fridge the occupant is using is **never shed and never
dimmed**. In these homes the ceiling fan is the only cooling 94 per cent of
families have, and degrading it is not what "restricting luxury appliances"
means.

The measured consequence, stated rather than discovered: level 2 then applies
only to discretionary dimmables — 37 air coolers out of 3,353 devices — so
SHARP is a **binary shedder for 98.9 per cent of appliances**. It gives up about
half the achievable peak reduction. That is the deliberate price of the
guarantee, and the guarantee is the product.

**So the Stage 1 rig needs no PWM at all.** Seven plain GPIO pins.

---

## 2. Appliance list — 10 devices

The median household in the dataset owns **8 devices** (mean 8.3, max 28), so a
rig of 10 is a typical home rather than a showcase. Ownership is measured across
the 431 target households: no inverter, no solar.

| # | Appliance | Class | `appliance_type` | Part | BCM | Owns it |
|---|---|---|---|---|---|---|
| 1 | Ceiling fan | **Critical** | `ceiling_fan` | **DC motor + driver** | 17 | **97.0 %** |
| 2 | Table fan | **Critical** | `table_fan` | **USB fan, 5 V** | 27 | 12.3 % |
| 3 | Light 1 | **Critical** | `led_bulb` | LED lamp | 22 | 68.9 % |
| 4 | Light 2 | **Critical** | `led_tube` | LED | 19 | 33.2 % |
| 5 | Refrigerator | **Critical + thermostatic** | `refrigerator` | LED | 23 | 30.4 % |
| 6 | Air conditioner | Thermostatic | `air_conditioner` | LED + relay | 16 | 5.6 % |
| 7 | Washing machine | Deferrable | `washing_machine` | LED | 20 | 8.6 % |
| 8 | **EV charger** | **Deferrable** | see below | **toy car** | 6 | — |
| 9 | Television | Interruptible | `television` | LED | 21 | **78.9 %** |
| 10 | Mixer grinder | Interruptible | `mixer_grinder` | LED | 26 | **51.7 %** |

Five Critical, one Thermostatic, two Deferrable, two Interruptible. All four
classes covered, and the Critical group is large enough that "the protected row
never moves" is visible rather than anecdotal.

**The water pump was dropped in favour of the mixer grinder.** The pump is in
11.1 per cent of homes; the mixer is in 51.7 per cent and is the second most
common sheddable appliance in Andhra Pradesh. Both are Interruptible `task`
loads, so the class coverage is identical and the mixer is more representative.

**No PWM anywhere.** Nothing dims, so every load is a plain digital output.
GPIO12 and GPIO13 are deliberately left free - they are the hardware-PWM
channels, so dimming could be reintroduced later without rewiring.

### The EV charger

There were no electric vehicles in IRES 2020, so no EV appears anywhere in
training. But a charger is a textbook Deferrable load - large, flexible, and
nobody minds when it charges so long as it is done by morning - and the policy
handles it through the same class as the washing machine.

The model reads each device's FEATURES, never its name: necessity flag, class,
power, remaining service hours. Give the charger Deferrable features and a
deadline and it is controlled correctly without retraining.

State it plainly: *EV charging is handled as a Deferrable load. No EV appeared in
the 2020 survey data, so this is a forward-looking extension rather than a
validated result.*

### The appliance list is not fixed

Because the model reads features rather than names, **the rig's appliances can be
changed without retraining**, as long as each `appliance_type` string matches one
of the 22 the dataset defines so the Pi looks up the right flags. Twelve loads is
the hard ceiling: 26 usable GPIO, minus two for the OLED on I2C, at two pins each
(one output, one switch).

### Is water a luxury?

The water pump is classed Interruptible and a reviewer should be invited to
question that. Many Andhra Pradesh homes pump to an overhead tank; shed the pump
at the wrong time and there is no water in the morning, which is not a luxury.
The honest position is that the classification comes from the dataset's
`control_permission` field, and that a tank-level constraint - shed only while
the tank is above a threshold - is the correct fix and is not yet implemented.

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
| 5 mm LEDs | 7 | any GPIO; no PWM needed |
| Resistors 220 Ω | 7 | one per LED |
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

| Appliance | BCM pin | Class | Commandable? |
|---|---|---|---|
| Ceiling fan | 17 | Critical | **No** — on whenever the occupant wants it |
| LED light | 27 | Critical | **No** — same |
| Refrigerator | 22 | Critical | **No** — always on, never commandable |
| Air conditioner | 16 | Thermostatic | 0 / 1 |
| Washing machine | 20 | Deferrable | 0 / 1, min-on 4 steps |
| Television | 21 | Interruptible | 0 / 1 |
| Mixer grinder | 26 | Interruptible | 0 / 1 |
| **Override switch, ceiling fan only** | **5** | input, pull-up | the one physical switch |
| OLED (I2C) | 2, 3 | SDA / SCL | status display |
| Buzzer | 18 | digital out | peak onset only |

### Two overrides, in series

```
appliance runs = resident permits  AND  power available
```

A wall switch and the supply are physically in series, so both must be closed.
The resident may switch anything off, including a fan - it is their house, and
the shield protects essential service from the CONTROLLER, not from the person
living there. Neither path can FORCE power on, which is why a peak lockout
cannot be defeated from either side.

### One physical switch, the rest from the phone

Ten switches is ten inputs, ten wires and debounce code for a rule the phone
already exercises: the shield honours a human OFF from either source, because it
branches on *who asked*, not on *how they asked*.

So fit **one** switch, on the ceiling fan, and drive the other nine from the
dashboard.

| | Cost | Buys |
|---|---|---|
| Ten switches | 10 inputs, 10 wires, debounce | completeness nobody asked for |
| **One, on the fan** | 1 input, 1 wire, ~10 lines | the demo moment and the offline path |
| None | nothing | phone-only, and nothing works if the network drops |

The reason to keep one is not tidiness. A phone override needs the dashboard, the
broker and the Pi all reachable. **A switch works with the network dead**, and the
thing it demonstrates - that the resident, not the controller, has the final say -
is the claim the project most wants to make. Losing the ability to show it
because the campus Wi-Fi dropped would be an avoidable way to lose the argument.

The ceiling fan is the right one: 97 per cent of homes own one, and it is the
load the whole protection rule is built around.

Wire that switch in series with its relay output and the AND is free in hardware;
the Pi reads the pin only so it can report and log the override.

### The OLED sequence

`handover/sharp_rl_v1/oled_display.py` implements it and runs with no hardware
attached, printing the panel to the terminal so wording and timing can be
checked before anything is wired.

```
BOOT ........ welcome, then the self-test result
NORMAL ...... rotates every 4 s: live load -> supply -> appliances
PEAK_ALERT .. BUZZER, inverted full-screen banner, 3 s
PEAK_ACTIVE . protected and shed lists, held for the event
RECOVER ..... "peak over", 2 s, back to NORMAL
```

The buzzer sounds on the TRANSITION into a peak, three short beeps, never a
continuous tone. A buzzer that runs for forty minutes gets disconnected, and
then it is not there for the one event it existed to announce.

All plain digital outputs. **GPIO12 and GPIO13 are left free** — they are the
hardware-PWM channels, and keeping them unused means dimming can be added later
without rewiring.

`reduced_power_fraction` stays in the dataset (fan 0.50, LED 0.40, cooler 0.55)
so the capability survives the decision not to use it.

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

### No PZEM in this build - so `measured_w` is null, not faked

There is no power meter on this rig. The LEDs stand in for appliances and the
wattages shown are simulated, which is the point of a simulation rig.

What matters is WHERE that simulated number goes. Publishing it in a field named
`measured_w` would make a stuck relay, a failed GPIO write and a broken wire all
invisible at once, because the fake number says the appliance is running
whatever the hardware is doing. So:

| Field | Value | Why it is honest |
|---|---|---|
| `power_15min_mean_w` | simulated appliance wattage | that is what it is |
| `measured_w` | **`null`** | no meter is fitted |
| `gpio_state` | **real pin readback** | a genuine measurement |
| `actuation_verified` | `gpio_state == commanded` | a real check |

### GPIO readback is a free actuation check

An output pin can be read back, and what comes back is what the pin is really
driving:

```python
GPIO.output(pin, GPIO.HIGH)
actual = GPIO.input(pin)          # not what we asked for - what is there
```

That catches a failed write, a mis-numbered pin, and a pin held by another
process. It will not catch a physically stuck relay - nothing without a sensor
will - and the reports should say so rather than imply full verification.

Publish `actuation_verified` per appliance. A mismatch is the single most
valuable fault signal the rig can produce, and it costs nothing.

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
