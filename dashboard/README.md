# SHARP resident dashboard

Next.js 16 (App Router) + Tailwind. Runs standalone on mock data — **no broker,
no Raspberry Pi and no network are required** to develop or demonstrate it.

```bash
npm install
npm run dev      # http://localhost:3000
```

## Going live

`app/page.tsx` swaps between the mock and the real feed in **one line**. If that
swap ever needs more than one line, the seam between data and presentation was
drawn in the wrong place.

```ts
// import { connectFeed } from '@/lib/mqtt';   // <- uncomment
```

Then set, in `.env.local` (never committed):

| Variable | Notes |
|---|---|
| `NEXT_PUBLIC_USE_MOCK` | `false` to use the live feed |
| `NEXT_PUBLIC_MQTT_URL`  | `wss://…:443` — port 443, not 1883 |
| `NEXT_PUBLIC_MQTT_USER` | **subscribe-only** credential |
| `NEXT_PUBLIC_MQTT_PASS` | see the warning below |

> Anything in `NEXT_PUBLIC_*` is visible to anyone who opens the page. This user
> must be subscribe-only on `home/<house>/#`. If a subscribe-only credential
> leaks, someone reads demo telemetry; if the *publishing* credential leaks,
> they can command your relays.

MQTT runs over WebSockets Secure on port 443 because campus and hostel Wi-Fi
routinely block plain MQTT. Test this on campus before demo day.

## Layout

| Path | Role |
|---|---|
| `lib/contracts.ts` | **FROZEN.** The Pi, the rig and this app must agree. Where this file and any document disagree, this file wins. |
| `lib/mockState.ts` | A fake day covering every state the UI must render: normal, peak, shed load, refused override, outage. |
| `lib/mqtt.ts` | Browser MQTT client. Runs in the browser, never on a serverless function. |
| `components/ApplianceGrid.tsx` | Panel 4 — the panel that carries the argument. |

## Two rules that are easy to break

1. A **critical appliance in use renders with no off control** — absent, not
   greyed. The resident must never see a button that would cut their fan,
   because no such action exists anywhere in the system.
2. **Stale data must not look live.** The age indicator is driven by message
   timestamps, not by socket state.

Appliance wattages are **simulated**; no meter is fitted in this build. Label
them as simulated wherever they are shown.

Full panel-by-panel specification: [`../docs/DASHBOARD_SPECIFICATION.md`](../docs/DASHBOARD_SPECIFICATION.md).
