# SHARP hosting and connectivity plan

How the Raspberry Pi, sitting behind a home router, reaches a dashboard hosted
on the public internet — and where each piece lives.

---

## 1. The constraint that decides the architecture

Your Pi is on home or campus Wi-Fi. It has a **private IP** (192.168.x.x) behind
NAT. That means:

- It has **no public address**. Nothing on the internet can dial it.
- Port forwarding needs router admin access, a static IP or DDNS, and on campus
  Wi-Fi it is simply not available.
- Even at home, most Indian ISPs use **CG-NAT**, so port forwarding often does
  not work at all.

So the naive design — dashboard calls the Pi's API — **cannot work**. The Pi has
to start every connection itself.

## 2. The answer: a broker as rendezvous

Both sides dial **outbound** to a cloud broker. Neither needs a public address.

```
   HOME / CAMPUS                  PUBLIC INTERNET                 ANY BROWSER
┌──────────────────┐         ┌──────────────────────┐         ┌──────────────┐
│  Raspberry Pi    │         │   HiveMQ Cloud       │         │  Dashboard   │
│                  │         │   (MQTT broker)      │         │  on Vercel   │
│  publishes ──────┼────────►│                      │◄────────┼──── WSS      │
│  state, ack      │ outbound│   topic router       │ outbound│   subscribes │
│  subscribes ◄────┼─────────┤                      ├─────────┼──► publishes │
│  cmd, override   │  :443   │                      │   :443  │   overrides  │
└──────────────────┘         └──────────────────────┘         └──────────────┘
         NAT: fine                                                 NAT: fine
```

Outbound connections traverse NAT without configuration. This is the standard
IoT pattern and it is why the idea book's topic layout is built around MQTT.

### Use port 443, not 1883

Plain MQTT is port 1883. **Campus and hostel Wi-Fi frequently block it.** MQTT
over WebSockets Secure runs on **443**, the same port as HTTPS, so it passes
through essentially any network that allows web browsing.

Configure both the Pi and the browser for `wss://...:8884/mqtt` or the provider's
443 endpoint. Test this on **campus Wi-Fi before demo day**, not on the morning.

---

## 3. Where each piece lives

| Piece | Host | Free tier | Dials |
|---|---|---|---|
| Dashboard (Next.js) | **Vercel** | 100 GB bandwidth | — served to browser |
| MQTT broker | **HiveMQ Cloud** | 100 connections, 10 GB/mo | — accepts |
| Backend API (FastAPI) | **Fly.io** (Mumbai) | 3 × 256 MB | outbound to broker |
| Pi runtime | your desk | — | outbound to broker |

### Why FastAPI cannot go on Vercel

Vercel functions are request-scoped: they wake, answer, freeze. Your backend must
hold a **long-lived MQTT subscription**. A frozen function drops it, and you get
intermittent missing telemetry that is painful to diagnose. Same reason rules out
Netlify Functions and Cloudflare Workers.

Fly.io runs a persistent process and has a **Mumbai region (`bom`)**, which keeps
the round trip to your Pi short.

---

## 4. Start without the backend

For the demo you may not need FastAPI at all. The browser can subscribe to MQTT
directly:

```
Pi ──MQTT/WSS──► HiveMQ ──WSS──► browser (page served by Vercel)
```

That delivers the entire live view — demand, appliance levels including dim,
safety blocks, outage mode — with **two services, both free**.

Add the backend only for what a browser genuinely cannot do:

| Need | Requires backend? |
|---|---|
| Live state, intents, acks | No — browser MQTT |
| History beyond this session | Yes |
| Metrics (cost vs baseline, PAR) | Yes |
| **Authenticated overrides** | **Yes** — see credentials below |

---

## 5. Credentials, and the mistake to avoid

Anything in `NEXT_PUBLIC_*` is **visible to anyone who opens the page**. Not
obscured — visible.

So create **two broker users** in HiveMQ when you set up the cluster:

| User | Permission | Used by |
|---|---|---|
| `dashboard_ro` | **subscribe only** on `home/demo/#` | browser |
| `pi_rw` | publish + subscribe on `home/demo/#` | Raspberry Pi |
| `api_rw` | publish on `home/demo/+/override` | FastAPI |

The browser gets `dashboard_ro`. If that credential leaks, the worst case is
somebody reads your demo telemetry. If you put `pi_rw` in the browser instead,
anyone could command your relays.

Overrides go browser → FastAPI (authenticated server-side) → broker. That is the
main reason the backend exists.

---

## 6. Topics

```
home/<house_id>/state                         Pi ──► everyone   (retained)
home/<house_id>/intent                        Pi ──► everyone
home/<house_id>/sensor/<appliance>/power_15min_mean_w
home/<house_id>/sensor/<appliance>/measured_w
home/<house_id>/actuator/<appliance>/cmd      agent ──► Pi
home/<house_id>/actuator/<appliance>/ack      Pi ──► everyone
home/<house_id>/override/<appliance>          API ──► Pi
home/<house_id>/health                        Pi ──► everyone   (retained)
```

### Use retained messages for state and health

A browser opening the page mid-episode should see the current state immediately,
not wait up to 15 minutes for the next publish. Publish `state` and `health` with
`retain=true`; the broker replays the last one to every new subscriber.

### QoS

| Topic | QoS | Why |
|---|---|---|
| `state`, `sensor/*` | 0 | A dropped reading is replaced in 15 min |
| `cmd`, `ack`, `override` | **1** | These must arrive; duplicates are handled by `command_id` |

QoS 1 means at-least-once, so a command can arrive twice. That is exactly why
every command carries a `command_id` and the Pi ignores one it has already
applied.

---

## 7. Failure modes to plan for now

| Failure | What happens | What to build |
|---|---|---|
| Pi loses Wi-Fi | State stops updating | Dashboard greys out past 2 intervals and shows data age |
| Broker unreachable from Pi | No commands arrive | **Watchdog: hold last safe state.** Never fail open |
| Browser tab sleeps | MQTT connection drops | Reconnect with backoff; re-read retained state |
| Command arrives late | World has moved on | `expires_at` on every command; Pi rejects stale ones |
| Duplicate command (QoS 1) | Relay toggles twice | Ignore a `command_id` already applied |
| Free tier connection limit | New clients refused | One client ID per session; clean up on unmount |

**The watchdog is the one that matters for safety.** If the Pi has heard nothing
valid for three intervals, it holds the last safe state — it does not switch
everything off and it does not switch everything on. A controller that fails open
during a network drop is more dangerous than no controller.

---

## 8. Latency budget

The control loop is 15 minutes, so network latency is almost irrelevant to
control. It matters only for how responsive the demo feels.

| Hop | Typical | Notes |
|---|---|---|
| Pi → broker | 30–80 ms | Choose a broker region near you |
| Broker → browser | 30–80 ms | |
| Override: tap → Pi | under 500 ms | Target for I5 in the integration tests |
| Decision (inference) | under 50 ms | 50k-parameter model |

Pick the **Mumbai or Singapore** broker region. A US or EU region adds 200–300 ms
round trip for no reason.

---

## 9. Demo-day contingency

Public Wi-Fi is the single most likely thing to break a live demo.

1. **Phone hotspot as backup.** Test the Pi on it beforehand and have the
   hotspot already configured in `wpa_supplicant`.
2. **Local fallback broker.** Install Mosquitto on the Pi. If the cloud is
   unreachable, run broker, controller and a local dashboard build entirely on
   the Pi, served over the hotspot. Everything keeps working; only remote
   viewing is lost.
3. **Recorded replay.** Keep a recorded episode that the dashboard can replay
   with no Pi at all. If the hardware fails on stage you still show the full
   six-step story.

Contingency 3 costs almost nothing — it is the same replay harness the team
needs anyway for development.

---

## 10. Build order

| # | Step | Blocks on |
|---|---|---|
| 1 | HiveMQ Cloud cluster, 3 users, ACLs | nothing |
| 2 | Replay harness publishing dataset transitions | nothing |
| 3 | Next.js on Vercel, browser MQTT, live panels | 1 + 2 |
| 4 | Pi publishes real state to the same topics | 1 |
| 5 | Fly.io FastAPI: history, metrics, override endpoint | 1 |
| 6 | Override round trip end to end | 4 + 5 |
| 7 | Test on campus Wi-Fi, port 443 | 3 + 4 |

Steps 1–3 need **no hardware** and no Pi. They can be done today, and they
unblock the dashboard work entirely — which is what Vaishnavi's handbook asks
for as her first deliverable.

---

## 11. What this costs

Nothing, at your scale.

| Service | Free tier | Your usage |
|---|---|---|
| Vercel | 100 GB/month | a few MB |
| HiveMQ Cloud | 100 connections, 10 GB/month | 1 Pi + a few browsers |
| Fly.io | 3 × 256 MB VMs | 1 small API |
| GitHub | unlimited private | 17 MB |
| Kaggle | 100 GB | 2.9 GB |

Fly.io asks for a card on file even inside the free allowance. If that is a
blocker, **Render** works but sleeps after 15 minutes idle and takes ~30 s to
wake — which will happen exactly when a reviewer opens the page cold. Given the
backend is optional for the live view, the simplest answer is to defer it.
