# SHARP dashboard guide

React + TypeScript on Vercel, FastAPI behind it. Built from Vaishnavi's
integration handbook and the idea book's Layer 4 sketch.

The dashboard makes the cyber-physical loop understandable. It does **not** make
the RL decision and it does **not** drive GPIO. It receives trusted events,
preserves their meaning, and captures overrides.

---

## 1. Architecture

```
Raspberry Pi                     Cloud
┌──────────────────┐            ┌────────────────────────┐
│ PZEM / LED rig   │            │ Vercel                 │
│ state builder    │            │  Next.js dashboard     │
│ BDQ inference    │  MQTT/TLS  │                        │
│ safety shield    │◄──────────►│  browser subscribes    │
│ actuator service │            │  directly over WSS     │
└──────────────────┘            └───────────┬────────────┘
         │                                  │ REST
         │ MQTT                             ▼
         └──────────────────────►┌────────────────────────┐
                                 │ FastAPI (Fly/Render)   │
                                 │  MQTT consumer         │
                                 │  history, metrics      │
                                 │  override endpoint     │
                                 └────────────────────────┘
```

### Why the browser talks MQTT directly

Vercel serverless functions cannot hold a long-lived MQTT connection — they are
request-scoped and freeze between invocations. The idea book already anticipates
this: the MQTT client **runs in the browser, not on Vercel's server**.

Live state goes browser ← MQTT over WSS. History, metrics and override writes go
browser → FastAPI → MQTT. Trying to proxy live telemetry through Vercel will
produce intermittent, hard-to-debug dropouts.

**FastAPI must be hosted somewhere that allows long-lived connections** — Fly.io,
Render or Railway. Not Vercel.

---

## 2. Panels

Ten panels, following the idea book's list.

| # | Panel | Shows | Source |
|---|---|---|---|
| 1 | Live demand | Aggregate kW, 96-step sparkline, sanctioned-load line | `sensor/.../power` |
| 2 | Grid condition | `obs_grid_peak_severity`, percentile, peak badge | `state` |
| 3 | Tariff | Current **marginal** ₹/kWh, slab position, month-to-date kWh | `state` |
| 4 | Appliances | Per device: level (off/on/**dim**), class badge, watts | `state` + `ack` |
| 5 | Proposed action | What the agent wants, **and why** | `intent` |
| 6 | Safety blocks | Actions the shield removed, with reasons | `intent.shield_reasons` |
| 7 | Override | Per-appliance buttons | → FastAPI |
| 8 | Power flow | Grid / PV / battery → house, animated | `sensor/source/*` |
| 9 | Outage mode | Battery runway, objective switch | `state.mode_islanded` |
| 10 | Savings | Cost vs baseline, PAR, comfort hours | FastAPI metrics |

### Panel 4 must show three states, not two

`off` · `on` · **`dim`**. The dim state is SHARP's distinguishing behaviour — if
the UI only renders a toggle, the demo cannot show the thing that makes the
project novel. Use a three-position control, and colour the dim state distinctly.

### Panel 6 is the one reviewers will ask about

Showing *blocked* actions is what makes the shield visible. When a user tries to
shed the fridge and the system refuses, that refusal has to appear on screen with
its reason. Per the idea book: violations become evidence.

### Panel 10 — appliance priority editor

Drag appliances between Necessity / Flexible / Deferrable. Necessity items grey
out and become un-sheddable: **the user configuring the shield themselves.** The
idea book flags that the necessity boundary is household-specific, so this is a
real feature, not decoration.

---

## 3. Data contracts

TypeScript types mirroring the dataset schema. Keep these in
`lib/contracts.ts` and generate the Python Pydantic models from the same source
of truth, or they will drift.

```ts
export type ActionLevel = 0 | 1 | 2;          // off | on | reduced
export type OperatingMode = 'grid_import' | 'self_sufficient' | 'islanded_outage';
export type ServiceClass = 'necessity' | 'thermostatic' | 'deferrable' | 'interruptible';

export interface ApplianceState {
  appliance_id: string;
  appliance_type: string;          // matches the dataset's appliance_type
  service_class: ServiceClass;
  is_necessity: boolean;           // true -> cannot be shed, ever
  supports_reduced: boolean;       // true -> the dim control is available
  level: ActionLevel;
  power_15min_mean_w: number;      // simulated or PZEM
  measured_w: number | null;       // the rig's real reading, never overwritten
  remaining_service_hours: number;
}

export interface HomeState {
  house_id: string;
  timestamp_ist: string;
  aggregate_power_kw: number;
  background_load_kw: number;      // unmodelled, not controllable
  sanctioned_load_kw: number;
  indoor_temperature_c: number;
  outdoor_temperature_c: number;
  occupancy_adult_home_fraction: number;
  attention_available: boolean;
  marginal_tariff_inr_kwh: number;
  month_to_date_kwh: number;
  operating_mode: OperatingMode;
  self_sufficient_fraction: number;
  battery_state_of_charge: number;
  grid_absent: boolean;
  appliances: ApplianceState[];
}

export interface Intent {
  command_id: string;
  proposed: Record<string, ActionLevel>;
  executed: Record<string, ActionLevel>;
  shield_reasons: Record<string, string[]>;   // why a level was refused
  policy_source: string;                      // 'bdq_v2'
  decision_latency_ms: number;
}
```

**`measured_w` is nullable and separate from `power_15min_mean_w` on purpose.**
Never render the simulated figure in a field labelled "measured". See the
hardware spec for why that distinction is load-bearing.

---

## 4. Project layout

```
dashboard/
├── app/
│   ├── page.tsx                 live view
│   ├── history/page.tsx
│   └── api/override/route.ts    thin proxy to FastAPI
├── components/
│   ├── LiveDemand.tsx
│   ├── ApplianceGrid.tsx        3-state control
│   ├── SafetyBlockPanel.tsx
│   ├── PowerFlow.tsx
│   ├── OutageBanner.tsx
│   └── PriorityEditor.tsx
├── lib/
│   ├── contracts.ts
│   ├── mqtt.ts                  browser-side client
│   └── useHomeState.ts
└── package.json
```

### MQTT in the browser

```ts
// lib/mqtt.ts — runs in the browser, not on Vercel's server
import mqtt from 'mqtt';

export const connect = (onState: (s: HomeState) => void) => {
  const c = mqtt.connect(process.env.NEXT_PUBLIC_MQTT_URL!, {
    username: process.env.NEXT_PUBLIC_MQTT_USER,
    password: process.env.NEXT_PUBLIC_MQTT_PASS,   // READ-ONLY credentials
  });
  c.on('connect', () => c.subscribe(['home/demo/state', 'home/demo/intent']));
  c.on('message', (topic, payload) => {
    if (topic.endsWith('/state')) onState(JSON.parse(payload.toString()));
  });
  return () => c.end();
};
```

`NEXT_PUBLIC_*` variables are **visible to anyone who opens the page.** Use a
read-only MQTT user with subscribe-only ACLs on `home/demo/#`. Never put the
publishing credential there — overrides go through FastAPI, which authenticates
server-side.

---

## 5. FastAPI service

```python
@app.get('/api/state')                  # latest snapshot
@app.get('/api/history')                # 96-step window
@app.get('/api/metrics')                # cost, PAR, comfort, override rate
@app.post('/api/override')              # authenticated, idempotent
@app.get('/api/health')                 # per-component, never aggregated away
@app.websocket('/ws/live')              # fallback if browser MQTT is blocked
```

### The override endpoint

```python
@app.post('/api/override')
async def override(req: OverrideRequest, user=Depends(require_role('resident'))):
    # Idempotent: the same override_id twice must not double-publish.
    if await seen(req.override_id):
        return {'status': 'duplicate', 'override_id': req.override_id}
    # The API records intent. It does NOT decide whether the override is safe -
    # the Pi's shield does, and it may refuse.
    await mqtt_publish(f'home/{req.house_id}/override/{req.appliance_id}', {
        'override_id': req.override_id,
        'requested_level': req.level,
        'issued_at': now_ist(),
        'user_id': user.id,
        'latency_ms': req.client_latency_ms,   # feeds the preference weight
    })
    return {'status': 'published', 'override_id': req.override_id}
```

Two things this must get right:

**Idempotency.** A user on a flaky connection will double-tap. Without an
`override_id` check you publish twice and corrupt the preference-pair record.

**The API never adjudicates safety.** It publishes intent. The Pi's shield
decides, and it is allowed to refuse — a user cannot override a safety
constraint, by design. Show the refusal in Panel 6.

### Capture override latency

`latency_ms` — the time between the intent appearing and the user tapping — is
the weight in your Bradley-Terry reward model. The dataset's
`preference_weight` is `occupancy × pressure ÷ (1 + latency_steps)`. If the
dashboard does not measure it, the real-deployment preference data will be
weaker than the synthetic data you trained on.

---

## 6. Deploy

```bash
npx create-next-app@latest dashboard --typescript --app --tailwind
cd dashboard && npm i mqtt recharts
```

```bash
npx vercel --prod
```

Environment variables in the Vercel dashboard:

| Variable | Value | Exposed? |
|---|---|---|
| `NEXT_PUBLIC_MQTT_URL` | `wss://xxxxx.s1.eu.hivemq.cloud:8884/mqtt` | **yes** |
| `NEXT_PUBLIC_MQTT_USER` | read-only user | **yes** |
| `NEXT_PUBLIC_MQTT_PASS` | read-only password | **yes** |
| `API_BASE_URL` | your FastAPI URL | no |
| `API_TOKEN` | server-side only | no |

HiveMQ Cloud's free tier (100 connections) is enough. Enable TLS and set ACLs so
the public user can only subscribe.

FastAPI goes on Fly.io or Render — anywhere that permits a long-lived MQTT
consumer.

---

## 7. Develop without hardware

Nobody should be blocked waiting for the Pi. Vaishnavi's handbook makes this her
first deliverable, and it is the right call.

Publish recorded transitions to MQTT at 1 step/second:

```python
d = pd.read_parquet('.../splits/validation.parquet')
episode = d[d.episode_id == d.episode_id.iloc[0]].sort_values('step_id')
for _, row in episode.iterrows():
    client.publish('home/demo/state', json.dumps(to_home_state(row)))
    time.sleep(1)            # 96 steps = 96 s per simulated day
```

This gives the dashboard real state vectors, real tariff slabs, real outages and
real override events before any hardware exists.

### Failure injection

Test these deliberately — the handbook lists them and they are where demos fail:

- stale PZEM readings · missing readings · invalid units
- duplicate and out-of-order messages
- MQTT disconnect mid-episode
- relay rejection (`ack.accepted = false`)
- delayed acknowledgement past `expires_at`

**A stale reading must look different from a fresh one.** Show data age; grey
the panel past 2 intervals. A dashboard that renders stale data as live is worse
than one that shows nothing.

---

## 8. Demo script

1. **Normal** — evening, moderate load, everything on.
2. **Peak** — `obs_grid_peak_severity` crosses 0.5. The fan drops to **dim**,
   the TV sheds. Fridge and lights stay on. *This is the money shot.*
3. **Override** — user restores the TV. The shield honours it. Preference pair
   recorded.
4. **Blocked override** — user tries to shed the fridge. **Refused**, with the
   reason on screen.
5. **Outage** — grid drops. Objective flips to battery runway. AC and TV go
   dead; fan and lights continue on the inverter.
6. **Recovery** — grid returns, battery recharges from mains.

Step 5 is the one that separates this from a UK-style demo. Your idea book is
explicit: *"During an outage, users want the luxury loads off so the fans and
lights last until the grid returns."* The dataset models exactly that — 28,116
outage steps from IRES-reported supply hours.
