# SHARP dashboard specification

What to build, what it must show, and what it must never claim. Written so the
dashboard can be built **before any hardware exists** — every contract below is
already frozen by the shipped dataset and the validated model.

Companion to `DASHBOARD_GUIDE.md` (architecture and hosting) and
`INTEGRATION_GUIDE.md` (the joins between lanes). Where they disagree, this file
wins, because it was written against the validated model.

---

## 0. Three screens, not one

| Screen | Audience | Shows |
|---|---|---|
| **Resident** | the citizen | appliances, protected vs shed, overrides |
| **Grid controller** | the DISCOM | declare peak, severity, homes responding, MW relieved |
| Rig OLED | the room | what the hardware itself believes |

The grid controller view and the peak-event mechanism are specified in
`GRID_CONTROLLER_AND_TARIFF.md`, along with the tariff basis and the opt-out
billing. Read that before building Panel 2 or Panel 7.

**Run the demo peak at MIDDAY.** The measured grid peak in the shipped data is
08:00-17:00; severity is exactly 0.000 after 18:00. Household load peaks at
19:00, which is a different event. Any script that says "evening, severity
crosses 0.5" contradicts the data.

---

## 1. What this dashboard is for

It makes a cyber-physical control loop **legible**. It does not make the control
decision and it does not drive GPIO.

Three questions a reviewer will ask, which the UI has to answer on screen:

1. What did the agent *want* to do, and why?
2. What did the safety shield *refuse*, and why?
3. What happened when a human disagreed?

A dashboard that only plots power answers none of them.

---

## 2. The ten appliances

Ownership is measured across the 431 target households — no inverter, no solar.
These ids are frozen: the Pi, the dashboard and the rig must all spell them the
same way.

The idea book's four classes: **Critical** never interrupted, **Thermostatic**
adjusted inside comfort bands, **Deferrable** moved in time, **Interruptible**
paused briefly.

| id | Appliance | Class | Legal levels | Watts | Owns it |
|---|---|---|---|---|---|
| `ceiling_fan_01` | Ceiling fan | **Critical** | **1 only** | 60.0 | 97.0 % |
| `table_fan_01` | Table fan | **Critical** | **1 only** | 40.0 | 12.3 % |
| `led_bulb_01` | Light 1 | **Critical** | **1 only** | 9.0 | 68.9 % |
| `led_tube_01` | Light 2 | **Critical** | **1 only** | 18.0 | 33.2 % |
| `refrigerator_01` | Refrigerator | **Critical + thermostatic** | **1 only** | 43.2 | 30.4 % |
| `air_conditioner_01` | Air conditioner | Thermostatic | 0, 1 + advisory setpoint | 1328.4 | 5.6 % |
| `washing_machine_01` | Washing machine | Deferrable | 0, 1 | 113.7 | 8.6 % |
| `ev_charger_01` | **EV charger** | Deferrable | 0, 1 + deadline | — | — |
| `television_01` | Television | Interruptible | 0, 1 | 104.6 | 78.9 % |
| `mixer_grinder_01` | Mixer grinder | Interruptible | 0, 1 | 500.0 | 51.7 % |

**Five Critical.** That matters for Panel 4: with five protected loads, "the top
row never moves during a peak" is visible rather than anecdotal.

**The EV charger is a forward-looking extension.** No electric vehicle appeared
in IRES 2020, so it is handled through the Deferrable class alongside the washing
machine and must be labelled as such wherever it is shown — not presented as a
validated result.

**Level 0 is not offered on a necessity appliance.** Not greyed out after the
fact — never rendered as available. The fan, the light and the fridge cannot be
shed, by anyone, including the resident.

**Level 2 is not offered at all in this build.** A critical load is never shed
*and never dimmed* while the occupant is using it, and the only non-critical
dimmable appliance in these homes is the air cooler — 37 devices out of 3,353.
The dashboard therefore renders **two states, ON and SHED**, and a critical load
renders as ON with no control at all.

The protection is conditional, and the UI must reflect that: a fridge nobody is
asking for at 3 a.m. is legitimately off. Protecting essential service does not
mean running it around the clock.

### The two overrides are in series

```
appliance runs = physical switch ON  AND  power available
```

A resident may switch anything off, including their own fan. Neither the switch
nor the dashboard can force power ON. So the override button is a **request**,
and the UI must never imply it is a command - during a peak, tapping "keep the
TV on" returns a refusal with a reason, not a state change.

| Actor | Action | Allowed |
|---|---|---|
| SHARP | shed or dim a critical load in use | **never** |
| Resident | switch off their own fan | **yes** |
| Resident | switch on a locked-out luxury | **no - no power there** |

### The air conditioner is a special case

The agent may switch it **on or off**. It may **not** set the temperature. The
dashboard shows a *recommended* setpoint as advice the resident can act on:

```
Suggested: 26 °C   (grid peak severity 0.7, you are in the ₹6.00 slab)
```

Advisory only, this cycle. Automating it is future work and must not be
implied by the UI.

---

## 3. The levels, and what the resident sees

```
0 = OFF       shed      - luxury loads only
1 = ON        full power
2 = REDUCED   reserved; not used on any critical load
```

| State | Treatment | Applies to |
|---|---|---|
| `PROTECTED` | solid, **no control shown** | fan, light, fridge in use |
| `ON` | solid, control shown | luxury currently served |
| `SHED` | outline only, reason on hover | luxury paused by SHARP |

**A critical appliance must render with no off control at all** — not a greyed
toggle. The resident should never see a button that would cut their fan, because
no such action exists anywhere in the system.

Level 2 stays in the contract and in `reduced_power_fraction` so dimming can be
reintroduced without a schema change. It is simply never selected today.

---

## 4. Panels

Ten panels. Panels 4, 5 and 6 are the ones that carry the argument.

| # | Panel | Shows | Source |
|---|---|---|---|
| 1 | Live demand | aggregate kW, 96-step sparkline, sanctioned-load line | `sensor/*` |
| 2 | Grid condition | peak severity, percentile, peak badge | `state` |
| 3 | Tariff | marginal ₹/kWh, slab position, month-to-date kWh | `state` |
| 4 | **Appliances** | per device: **three-state** control, class badge, watts | `state` + `ack` |
| 5 | **Proposed action** | what the agent wants **and why** | `intent` |
| 6 | **Safety blocks** | what the shield refused, with reasons | `intent.shield_reasons` |
| 7 | Override | per-appliance request buttons | → FastAPI |
| 8 | Power flow | grid → house, animated | `sensor/source/*` |
| 9 | Outage mode | battery runway, objective switch | `state.mode_islanded` |
| 10 | Results | **peak, PAR**, comfort hours, override rate | FastAPI metrics |

### The headline is equity, not savings

The line that carries the project:

> **We do not choose which village goes dark. We take the television from
> everyone so that nobody goes dark.**

Measured on 40 held-out household-days, SHARP cuts household peak by **4.65 %** —
the same relief a rolling blackout would get by cutting supply to **4.7 homes in
100**, delivered without cutting anyone. The shed-luxury-only ceiling is 10.96 %,
equivalent to blacking out 11 homes in 100.

Panel 10 should say that in those terms. "Peak avoided, equivalent to N homes
not blacked out" is the number a DISCOM and a citizen both understand.

### Panel 10 must lead with peak and PAR, not cost

This is the most important design decision in the document.

APCPDCL's domestic tariff is **telescopic on monthly units with no time-of-day
rate** (`time_of_day_rates: false`). Shifting load in time therefore saves the
household **nothing**. Measured on 40 held-out household-days:

| Metric | No control | Rule-based | **SHARP** |
|---|---|---|---|
| Peak kW | 0.408 | 0.409 | **0.379** |
| Peak-to-average | 2.593 | 2.803 | **2.558** |
| Cost ₹/day | 19.18 | **18.06** | 19.00 |

SHARP wins peak and PAR; the rule-based controller makes **both worse than
doing nothing**. On cost the rule-based controller wins, and the dashboard must
not imply otherwise. Lead with peak reduction and PAR. If cost is shown at all,
show it honestly.

### Panel 6 is what a reviewer will ask about

When a resident tries to shed the fridge and the system refuses, that refusal
appears on screen with its reason. Violations become evidence. In validation the
shield refused **0 of 4,103** entitled necessity requests — it never had to,
because the agent never asked — but the panel must still render a refusal
correctly when one occurs, and the demo script deliberately triggers one.

---

## 5. Data contracts

Keep in `lib/contracts.ts`. These are frozen by the dataset; changing them
silently breaks the Pi.

```ts
export type ActionLevel = 0 | 1 | 2;                 // off | on | reduced
export type OperatingMode = 'grid_import' | 'self_sufficient' | 'islanded_outage';
export type ServiceClass = 'necessity' | 'thermostatic' | 'deferrable' | 'interruptible';

export interface ApplianceState {
  appliance_id: string;            // 'ceiling_fan_01'
  appliance_type: string;          // matches the dataset's appliance_type
  service_class: ServiceClass;
  is_necessity: boolean;           // true -> level 0 is NEVER offered
  supports_reduced: boolean;       // true -> the dim control exists
  level: ActionLevel;
  power_15min_mean_w: number;      // SIMULATED appliance wattage
  measured_w: number | null;       // null in this build - no meter fitted
  gpio_state: 0 | 1;               // real pin readback - a true measurement
  actuation_verified: boolean;     // gpio_state agrees with the command
  remaining_service_hours: number;
}

export interface HomeState {
  house_id: string;
  timestamp_ist: string;
  aggregate_power_kw: number;
  background_load_kw: number;      // unmodelled, NOT controllable - see below
  sanctioned_load_kw: number;
  indoor_temperature_c: number;
  outdoor_temperature_c: number;
  occupancy_adult_home_fraction: number;
  attention_available: boolean;
  marginal_tariff_inr_kwh: number;
  month_to_date_kwh: number;
  grid_peak_severity: number;      // 0..1, drives the peak badge
  operating_mode: OperatingMode;
  battery_state_of_charge: number;
  grid_absent: boolean;
  appliances: ApplianceState[];
}

export interface Intent {
  command_id: string;
  proposed: Record<string, ActionLevel>;
  executed: Record<string, ActionLevel>;
  shield_reasons: Record<string, string[]>;   // why a level was refused
  policy_source: string;                      // 'bdq_v2_cql'
  decision_latency_ms: number;
}
```

`rejected_reason` is a closed set. Do not invent strings:

```
necessity_mask | compressor_protection | cycle_active | min_on_steps
min_off_steps | command_expired | level_not_supported | watchdog_hold
```

### Two fields that are load-bearing

**There is no PZEM in this build, so `measured_w` is `null` — never a simulated
number.** Show the simulated wattage everywhere; that is what the rig exists for.
Just never put it in a field named *measured*, or a stuck relay, a failed GPIO
write and a wiring fault all become invisible behind a number that says the
appliance is running regardless.

`gpio_state` is the real signal available without a meter. An output pin read
back reports what it is actually driving, which catches a failed write, a
mis-numbered pin, and a pin held by another process. It will not catch a
physically stuck relay — nothing without a sensor will — so say that rather than
imply full verification.

Render **`actuation_verified`** per appliance. A mismatch between commanded and
actual is the most valuable fault the rig can surface, and it costs nothing.

**`background_load_kw` is 61 % of household load and is not controllable.** It is
unmodelled draw calibrated against these households' own reported bills. Show it
as a distinct, uncontrollable band in Panel 1. If the UI implies the agent can
act on it, every result looks inexplicably small.

---

## 6. Build order — all of this needs no hardware

| # | Step | Blocked on |
|---|---|---|
| 1 | Replay publisher: dataset transitions → MQTT at 1 step/sec | nothing |
| 2 | Panels 1–4 against replayed state | 1 |
| 3 | Panels 5–6 against `intent` | 1 |
| 4 | Override → FastAPI → MQTT | broker credentials |
| 5 | Panel 10 metrics | FastAPI |
| 6 | Swap the replay publisher for the real Pi | Harini |

**Step 1 is the critical path for the whole team.** It unblocks the dashboard and
gives Harini realistic `state` messages to test against. It needs no model and no
Pi:

```python
d = pd.read_parquet('.../splits/validation.parquet')
episode = d[d.episode_id == d.episode_id.iloc[0]].sort_values('step_id')
for _, row in episode.iterrows():
    client.publish('home/demo/state', json.dumps(to_home_state(row)))
    time.sleep(1)                      # 96 steps = one simulated day in 96 s
```

---

## 7. Failure states the UI must render

A dashboard that shows stale data as live is worse than one that shows nothing.

| Condition | Required behaviour |
|---|---|
| No state for 2 intervals | Grey the panel, show data age in minutes |
| `ack.accepted = false` | Show the appliance as **refused**, with the reason |
| Command past `expires_at` | Show as expired, not as applied |
| Duplicate `command_id` | Ignore the second; do not toggle twice |
| MQTT disconnected | Banner, reconnect with backoff, re-read retained state |
| `grid_absent = true` | Outage mode: only inverter-circuit loads live |

---

## 8. Credentials

`NEXT_PUBLIC_*` variables are **visible to anyone who opens the page**. Not
obscured — visible.

| Variable | Value | Exposed |
|---|---|---|
| `NEXT_PUBLIC_MQTT_URL` | `wss://...:8884/mqtt` | yes |
| `NEXT_PUBLIC_MQTT_USER` | **subscribe-only** user | yes |
| `NEXT_PUBLIC_MQTT_PASS` | that user's password | yes |
| `API_BASE_URL`, `API_TOKEN` | server side only | no |

The browser gets a read-only broker user. If it leaks, someone reads demo
telemetry. If you put the publishing credential there instead, anyone on the
internet can command your relays.

Overrides go browser → FastAPI (authenticated server-side) → broker. That is the
main reason the backend exists.

---

## 9. Demo script

1. **Normal** — evening, moderate load, everything on.
2. **Peak** — severity crosses 0.5. The fan drops to **DIM**, the TV sheds.
   Fridge and lights stay. *This is the money shot.*
3. **Override** — resident restores the TV. Honoured. Preference pair recorded.
4. **Blocked override** — resident tries to shed the fridge. **Refused**, reason
   on screen, Panel 6.
5. **Outage** — grid drops. AC and TV go dead; fan and lights continue on the
   inverter circuit.
6. **Recovery** — grid returns.

Step 5 separates this from a UK-style demo. The dataset carries 28,116 outage
steps from IRES-reported supply hours.

---

## 10. What the dashboard must not claim

- Do not label simulated power as measured.
- Do not show a cost saving as the headline. Under this tariff there is almost
  none available, and the rule-based baseline beats SHARP on it.
- Do not imply the AC setpoint is automated. It is advisory.
- Do not show the model as "approved for deployment". Every validation report
  says `approved_for_deployment: false`, and that is deliberate.
- Override and occupancy data in the replay are **synthetic**, generated from a
  stated behavioural rule. Label them so in any screenshot that reaches a report.

The validated model: 17 of 17 gates passed, balanced accuracy 0.7478 against a
0.3333 floor, dim recall 0.6345, and **0 of 4,103** necessity loads shed while
the occupant was entitled to them.
