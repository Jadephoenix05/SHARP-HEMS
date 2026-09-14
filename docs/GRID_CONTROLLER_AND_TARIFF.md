# Grid controller, peak events and the tariff

The third screen, the mechanism behind it, and the regulation that permits it.

Companion to `DASHBOARD_SPECIFICATION.md` (the resident's view) and
`HARDWARE_PROTOTYPE_SPEC.md` (the rig).

---

## 1. Why a second dashboard

A real peak is not a curve known in advance. It varies daily and the grid
operator is the one who knows. So SHARP has two authorities, and the idea book
already separates them:

> *Local power determines household stress, while historical grid curves or
> **authenticated events** determine grid stress.*

| Source | Used when | Authority |
|---|---|---|
| **Declared event** | the controller raises one | authoritative |
| Historical curve | no event is active | fallback |

The state feature is the same either way, so **no retraining is needed** to add
this. The model reads `obs_grid_peak_severity`; it does not care who set it.

### Three screens

| Screen | Audience | Shows |
|---|---|---|
| Resident | the citizen | appliances, protected vs shed, overrides |
| **Grid controller** | the DISCOM | declare event, severity, homes responding, MW relieved |
| Rig OLED | the room | what the hardware itself believes |

The OLED is not redundant. It reads local GPIO; the dashboard reads MQTT. If
they disagree you have found a real fault — a stuck relay, a dropped message, a
failed write — in front of the audience. Showing both is a confidence move.

---

## 2. The measured peak is midday, not evening

From the shipped context data, averaged over the validation period:

| Hour | Grid severity | Household kW |
|---|---|---|
| 03–07 | **0.000** | 0.16 |
| **08–13** | **0.42 – 0.67** | 0.17 |
| 14–17 | 0.47 – 0.59 | 0.15 |
| 18–23 | 0.000 | **0.24 at 19:00** |

**Grid peak 08:00–17:00. Household peak 19:00.** They are different events, which
is the classic Indian pattern: agricultural pumping and commercial cooling by
day, homes after work.

This is good news. The grid needs relief when houses are **empty**, so the
shedding costs residents least. Any demo script that says "evening, severity
crosses 0.5" is wrong against this data — severity is exactly zero in the
evening. **Run the demo peak at midday.**

---

## 3. The peak event

```
topic:  grid/<region>/peak_event      QoS 1, retained
```

```json
{
  "event_id": "apcpdcl-2026-09-15-001",
  "sequence": 47,
  "region": "guntur",
  "severity": 0.67,
  "declared_at": "2026-09-15T14:05:00+05:30",
  "expires_at":  "2026-09-15T16:00:00+05:30",
  "reason": "system_peak",
  "signature": "<base64 over every field above>"
}
```

### Five rules, and each exists because of a specific failure

**1. Signed, and verified on every Pi.** This broadcast sheds load in every home
that hears it. Broker ACLs are not enough — a leaked credential must not be
sufficient to switch off every television in Guntur, or to declare a false
all-clear during a real peak.

**2. `expires_at` is mandatory.** A stuck or replayed event must not hold luxury
off indefinitely. On expiry the Pi reverts to the historical curve by itself.

**3. Monotonic `sequence`.** Rejects replay of an old "peak on" after it ended.

**4. A local cap.** No event may keep luxury off for more than **8 consecutive
intervals** (two hours) without renewal. The Pi protects the resident even from a
malfunctioning control centre.

**5. A grid event can never touch a critical load.** `apply_shield` raises if a
peak lockout names a necessity appliance:

> *Critical loads cannot be locked out at peak. A peak restricts luxury; it never
> cuts essential service.*

Rule 5 is the one that makes this acceptable to a citizen. **Even the DISCOM
cannot take your fan.**

---

## 4. Two overrides, in series

There are two ways a resident acts, and they compose the way a real house does:

```
appliance runs  =  physical switch ON   AND   power available
                   ───────────────────         ────────────────
                   the resident's own          SHARP + shield +
                   wall switch                 peak lockout
```

This is not a convention. A wall switch and the DISCOM's supply are physically
in series, and every case falls out of it:

| Switch | SHARP allows | Result |
|---|---|---|
| ON | yes | runs |
| **OFF** | yes | **off** — the resident opened the circuit |
| ON | **no, peak** | **off** — no supply on that circuit |
| OFF | no | off |

**Neither path can force power on.** The dashboard sends a *request*; the switch
*permits*. That is why the peak lockout cannot be defeated from either side.

### Who is being constrained

| Actor | Action | Allowed |
|---|---|---|
| **SHARP** | shed a critical load in use | **never** |
| **SHARP** | dim a critical load | **never, unconditionally** |
| **Resident** | switch off their own fan | **yes — it is their house** |
| **Resident** | switch on a locked-out luxury | **no — no power there** |

The shield protects essential service **from the controller**, not from the
person who lives there. It could not do the latter anyway: a fan has a physical
switch on the wall.

---

## 5. The tariff — what the law actually says

The **Electricity (Rights of Consumers) Amendment Rules, 2023**, notified
14 June 2023, mandate Time-of-Day tariffs:

| Rule | Value |
|---|---|
| Peak-period tariff | **not less than 1.10 × normal** |
| Solar-hours tariff | **at least 20 % below normal** |
| Domestic consumers, effective | **1 April 2025** |

Two honest caveats, both of which must be stated:

1. A draft update pushes this to **April 2027** (C&I) and **April 2028** (others),
   tied to smart-meter rollout.
2. **APCPDCL's own FY2025-26 order still carries `time_of_day_rates: false`.**
   Mandated nationally; not yet implemented in this DISCOM.

So the correct framing is: *the Rules require ToD for domestic consumers,
APCPDCL has not implemented it yet, and SHARP is built for the tariff that is
coming.* Cite the 1.10× floor rather than inventing a number.

### The solar-hours conflict

The Rules make solar hours **cheaper**, but the measured grid peak here is
08:00–17:00, which *is* solar hours. These are different signals and must stay
separate:

| Signal | When | Set by |
|---|---|---|
| Grid stress | when the operator says | declared event |
| ToD price | fixed windows | the tariff order |

SHARP responds to the first. The bill reflects the second.

---

## 6. Paying to keep a luxury load during a peak

What this is, in the literature, is **Critical Peak Pricing** with opt-out.
Established practice, and three of its design rules matter here.

**Cap the events at 10–15 a year, published in advance.** Unlimited peak events
would let a DISCOM suppress luxury indefinitely, which breaks the promise that
supply is for everyone.

**Compensate off-peak.** CPP is not a pure surcharge: participants get cheaper
electricity the rest of the year. Without that it is a penalty, and a regulator
will reject it. A household that never opts out should end the year **paying
less**.

**Default to shed, allow opt-out.** Measured participation is **85 % for opt-out
against 28 % for opt-in**, so the default carries the programme.

### The flow

```
peak declared  ->  luxury sheds  ->  resident notified
                                        |
                    "Keep the AC on for this event?  about Rs 12"
                                        |
                       [ Keep it on ]        [ No thanks ]
                              |                     |
                    surcharge recorded      relief credited
```

### The statement

```
SHARP monthly statement — September 2026

  Energy, normal hours        118 kWh  x Rs 4.50      Rs 531.00
  Energy, peak hours            4 kWh  x Rs 4.95      Rs  19.80
    (1.10x normal, the floor set by the 2023 Rules)

  Peak events this month: 3
    14 Sep 14:05-16:00   opted out (air conditioner)  Rs 12.40
    22 Sep 11:00-13:00   participated                       -
    29 Sep 12:30-14:30   participated                       -

  Peak relief credit                                  -Rs 35.00
  ------------------------------------------------------------
  Total                                               Rs 515.80
```

The credit line is what makes it fair, and what a regulator would require. A
statement showing only the surcharge is not this mechanism.

---

## 7. The grid controller screen

| Panel | Shows |
|---|---|
| Declare | severity slider, duration, reason, **Declare** button |
| Active event | countdown to `expires_at`, renew, cancel |
| Response | homes reached / responding / opted out |
| Relief | kW avoided, and **"equivalent to N homes in 100 not blacked out"** |
| Audit | every event this month, who declared it, how long |

The relief line is the project's whole argument as a live number. On held-out
data SHARP cut household peak by **4.65 %** — the relief a rolling blackout would
get by cutting supply to **4.7 homes in 100**, delivered without cutting anyone.

The audit panel matters for the same reason the events are capped and signed:
a mechanism that can switch off appliances in many homes must be accountable to
somebody.

---

## 8. What must not be claimed

- Do not present the surcharge as operating under current APCPDCL tariffs. It is
  not in their order. It is what the 2023 Rules require and this DISCOM has yet
  to implement.
- Do not show a peak event as able to touch a fan, a light or a fridge. It
  cannot, and the shield raises rather than obeying.
- Do not imply the opt-out charge funds the project. It is a tariff mechanism,
  set by a regulator, not a fee.
- Do not run the demo peak in the evening. The data says the grid peak is
  08:00–17:00 and severity is zero after 18:00.
