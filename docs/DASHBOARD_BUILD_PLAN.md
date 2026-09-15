# Dashboard build plan — Supriya and Vaishnavi

How two people build this without waiting on each other or on hardware.

`DASHBOARD_SPECIFICATION.md` says *what* to build. This says *in what order*,
*who*, and *what unblocks what*.

---

## 1. The seam

Split by layer, not by panel. If you split by panel you both touch the MQTT
client, the state shape and the styling, and you spend the week merging.

| | Supriya — data | Vaishnavi — presentation |
|---|---|---|
| Owns | `lib/` — contracts, MQTT client, state store, replay publisher | `components/` — panels, layout, styling |
| Publishes | a `HomeState` object that updates | components that take `HomeState` as a prop |
| Never touches | component internals | MQTT, parsing, topic names |

**They meet at one file: `lib/contracts.ts`.** Write it first, together, in one
sitting. After that neither of you needs the other to make progress, because
Vaishnavi builds against a mock that satisfies the same types.

### The mock is what makes parallel work possible

```ts
// lib/mockState.ts — Vaishnavi imports this until the real feed exists
export function mockHomeState(t: number): HomeState { ... }
```

One function, advancing a fake day, covering every state the UI must render:
normal, peak active, a shed appliance, a refused override, an outage. Build it in
the first hour. It is also what you demo from if the network fails on the day.

---

## 2. Order of work

### Stage 0 — together, one sitting (about 2 hours)

- [ ] `npx create-next-app@latest dashboard --typescript --app --tailwind`
- [ ] `lib/contracts.ts` — copy the types straight out of the specification
- [ ] `lib/mockState.ts` — a fake day covering all five states
- [ ] Agree the appliance ids: `ceiling_fan_01`, `led_bulb_01`, … exactly as the
      hardware spec spells them
- [ ] `git push`, deploy once to Vercel so the pipeline is proven empty

Do not skip the deploy. A first deployment that fails is much easier to fix on
an empty project than on a finished one.

### Stage 1 — parallel, the bulk of it

**Supriya (data)**

1. **The replay publisher.** A Python script that reads validation episodes and
   publishes `state` to MQTT at one step per second. **This is the critical path
   for the whole team** - it gives Vaishnavi a real feed and Harini realistic
   messages to test the Pi against, and needs no model and no hardware.
2. HiveMQ Cloud cluster, three users, ACLs (`HOSTING_AND_CONNECTIVITY_PLAN.md`)
3. `lib/mqtt.ts` — browser client over **WSS on 443**, subscribe, parse, reconnect
4. `lib/useHomeState.ts` — a hook exposing `HomeState`, `Intent`, connection
   status and **data age in seconds**

**Vaishnavi (presentation)**

1. Layout shell and the four appliance groups
2. Panel 4 — appliances, grouped `PROTECTED / RUNNING / SHED / DEFERRED`
3. Panel 1 — live demand, sparkline, sanctioned-load line
4. Panel 2 — grid condition and the **peak badge**
5. Panel 6 — safety blocks with reasons

### Stage 2 — join

- [ ] Swap `mockHomeState` for `useHomeState`. **One line.** If it is more than
      one line, the seam was wrong.
- [ ] Data-age greying at 2 intervals
- [ ] Reconnect banner

### Stage 3 — the parts that need a backend

Only now, and only these:

- [ ] Panel 7 override → FastAPI → MQTT (authenticated server-side)
- [ ] Panel 10 metrics: peak, PAR, homes-not-blacked-out
- [ ] History beyond the session

Live state does **not** need the backend. Do not let Stage 3 block Stage 2.

---

## 3. Build Panel 4 first, and get it right

It carries the whole argument, and it is the one a reviewer will stare at.

```
PROTECTED   fan · fan · light · light · fridge      (no control shown at all)
RUNNING     AC · iron                                (luxury, allowed)
SHED        TV · mixer · pump          peak active   (reason on hover)
DEFERRED    washer → 22:00 · EV → 23:00
```

Three rules that are easy to get wrong:

**A critical appliance renders with no off control.** Not greyed, not disabled —
absent. The resident must never see a button that would cut their fan, because
no such action exists anywhere in the system.

**Two states, not three.** `ON` and `SHED`. Nothing dims in this build.

**The top row never moves.** A reviewer should be able to watch a peak event
start and see that the protected row is unchanged. That is the claim, rendered.

---

## 4. Five things that will break, and what to do

| Problem | What to do |
|---|---|
| **Port 1883 blocked on campus Wi-Fi** | Use **WSS on 443** from day one. Test on campus, not at home. |
| **Stale data rendered as live** | Show data age; grey past 2 intervals. Worse than showing nothing. |
| Browser tab sleeps, MQTT drops | Reconnect with backoff; re-read the retained `state` |
| Credentials in `NEXT_PUBLIC_*` are public | Browser gets a **subscribe-only** user. Overrides go via FastAPI. |
| Demo network fails | Keep the replay publisher and the mock. Demo from the mock if you must. |

---

## 5. The demo runs at MIDDAY

The measured grid peak in the dataset is **08:00–17:00**. Severity is exactly
**0.000** after 18:00, while household load peaks at 19:00 — a different event.

An evening peak badge will never light. Every older draft of the demo script said
"evening"; they were wrong against the data.

---

## 6. What the resident sees during the money shot

```
14:05   GRID PEAK 0.67                    [buzzer on the rig]

PROTECTED    fan  fan  light  light  fridge          unchanged
SHED         TV   mixer  pump                        "grid peak"
RUNNING      AC                          Keep on for this event? ~Rs 12
DEFERRED     washer → 22:00   EV → 23:00

peak avoided: equivalent to 4.2 homes in 100 not blacked out
```

Two things happen at once and both must be visible: **the protected row does not
move**, and the resident is **offered a choice** rather than simply cut off.

---

## 7. Honest rendering

- `measured_w` is **`null`** — there is no meter. Never render a simulated number
  in a field labelled measured.
- `power_15min_mean_w` is the **simulated** wattage. Label it as simulated.
- `gpio_state` is real. Render `actuation_verified` per appliance.
- Lead Panel 10 with **peak and PAR**, not cost. Under a telescopic tariff with
  no time-of-day rate there is almost no cost saving available, and the
  rule-based baseline beats SHARP on it.
- Never show the model as approved for deployment.

---

## 8. Rough effort

| Stage | Supriya | Vaishnavi |
|---|---|---|
| 0 together | 2 h | 2 h |
| 1 | 8–10 h (replay publisher is half) | 10–12 h |
| 2 join | 2 h | 2 h |
| 3 backend | 6–8 h | 3 h |

Stages 0–2 give a complete live demo. **Stage 3 is optional for demo day** and
should be treated that way if time runs short.

---

## 9. Start here

```bash
npx create-next-app@latest dashboard --typescript --app --tailwind
cd dashboard && npm i mqtt recharts
```

Then `lib/contracts.ts`, then `lib/mockState.ts`, then split.

The single highest-value thing you can build today is the **replay publisher** —
it unblocks Vaishnavi, unblocks Harini, needs no hardware, no model and no
network, and it is the fallback that saves the demo if the Wi-Fi fails.
