# SHARP — how to finish the project

The other nine documents each describe **one** subsystem. This one describes the
**whole**: what is finished, what is not, the order to do the rest in, and the
gate each piece must pass before you call it done.

Read this first. Then read the document for the lane you are working in.

| If you are working on | Read |
|---|---|
| The model | [RL_MODEL_SPEC.md](RL_MODEL_SPEC.md) |
| The dashboard | [DASHBOARD_BUILD_PLAN.md](DASHBOARD_BUILD_PLAN.md), then [DASHBOARD_SPECIFICATION.md](DASHBOARD_SPECIFICATION.md) |
| The Pi and the rig | [HARDWARE_PROTOTYPE_SPEC.md](HARDWARE_PROTOTYPE_SPEC.md) |
| Anything that crosses a boundary | [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) |
| Broker, hosting, demo day | [HOSTING_AND_CONNECTIVITY_PLAN.md](HOSTING_AND_CONNECTIVITY_PLAN.md) |
| The grid side and the tariff | [GRID_CONTROLLER_AND_TARIFF.md](GRID_CONTROLLER_AND_TARIFF.md) |

---

## 1. Status

| Subsystem | State | Evidence |
|---|---|---|
| Dataset and simulator | **Done** | `reports/` validation suite |
| RL model — trained, validated, exported | **Done** | 17 gates, `MODEL_VALIDATED` |
| Kaggle training notebook | **Done** | 14/14 cells execute, verifier passes |
| Handover bundle for the Pi | **Done** | 21 checks, `READY FOR HARDWARE` |
| Documentation | **Done** | 174 checks, `DOCS_CONSISTENT` |
| Hardware specification | **Done** (spec only) | 17 checks, `HARDWARE_SPEC_BUILDABLE` |
| Dashboard scaffold + Panel 4 | **Done** | `next build` exit 0, runs on mock data |
| **Replay publisher** | **Not started** | — |
| Dashboard panels 1/2/3/5/6/8/9/10 | **Not started** | — |
| Grid controller screen | **Not started** | — |
| FastAPI backend | **Not started** | — |
| Rig wired and actuating | **Not started** | — |

Everything in the first block is reproducible from this repository today.
Everything in the second block is the remaining work, and **none of it needs the
model to change**.

---

## 2. The model is finished. Stop tuning it.

This is the most important instruction in the document, because tuning is
comfortable and the remaining work is not.

The shipped policy is `handover/sharp_rl_v1/sharp_policy.npz`: 50,133
parameters, 196 KB in float32, 305 features into 28 branches of 3 levels.

**Held-out behaviour**

| Measure | Value | Note |
|---|---|---|
| Balanced accuracy | **0.6232** | chance is 0.3333 |
| TD error, non-terminal | 0.6034 | do not quote the pooled figure |
| Illegal greedy picks | **0** | |
| Critical load shed while in use | **0** | |
| Critical load dimmed, ever | **0** | |

**Simulator, 30 held-out household-days, seed 7** — every row below was measured
through the same harness, so the rows are comparable to each other and to
nothing else:

| Arm | Peak kW | PAR | Cost ₹/day |
|---|---|---|---|
| No demand response | 0.315 | 2.805 | 11.53 |
| Rule-based | 0.312 | 2.967 | 11.08 |
| **SHARP, shipped** | **0.281** | **2.650** | 11.39 |
| CQL 16.5× (`scale 2.5`) | 0.295 | 2.735 | 11.45 |
| CQL 33.0× (`scale 5.0`) | 0.295 | 2.736 | 11.45 |

Confirmed on 90 household-days at seed 99: peak 0.453 → **0.434**, PAR 2.594 →
**2.467**, and **0 necessity loads shed while entitled out of 9,304**
opportunities.

### Two caveats that must survive into the report

1. **The median per-household-day change is 0.0 % on every metric.** On most
   days SHARP changes nothing; the gain is concentrated in a minority of days.
   A mean-only figure overstates how often the controller acts. Quote both.
2. **The rule-based controller makes PAR *worse* than doing nothing** (2.967
   against 2.805) while winning on cost. SHARP wins peak and PAR and loses
   slightly on cost. Lead with peak and PAR, and say why: APCPDCL's domestic
   tariff is telescopic on monthly units with **no time-of-day rate**, so
   shifting load in time cannot save the household money. Cost was never a
   winnable axis here.

### The knob that is already at its optimum

`LEVEL_WEIGHT_SCALE` multiplies the conservative (CQL) term. It controls
**control quality**, not the accuracy metric, which barely moves with it. It is
**not monotone**:

```
 1.0x  (mean-normalised)  ->  0.302 kW
 6.6x  (min-normalised)   ->  0.281 kW   <- shipped
16.5x                     ->  0.295 kW
33.0x                     ->  0.295 kW
```

Past roughly 6.6× the imitation term swamps the reward and the controller
regresses towards the logged behaviour. **Do not raise it.**

### Checkpoints that score better on peak, and why they are not shipped

`models/reward_driven.npz` reaches 0.257 kW and
`models/candidate_pranitha_config.npz` 0.264 kW — both better than the shipped
0.281 on the same harness. They are kept, and they are **not** shipped, because
they buy that peak by shedding harder: `reward_driven` scores **0.4715**
balanced accuracy against the shipped model's 0.6232. A controller that
disagrees with the resident far more often is not a better controller for this
project, whatever the peak column says. If you revisit this, argue it on both
axes — and note that neither candidate ships a golden vector or a training
report, so neither can currently pass the full 17 gates.

---

## 3. Order of the remaining work

### 3.1 The replay publisher — do this first

`scripts/replay_publisher.py`: read validation episodes, publish `state` and
`intent` to MQTT at one step per second.

It is the critical path for the whole team. It gives the dashboard a real feed
and the Pi realistic messages, and it needs **no model, no hardware and no
network**. It is also the fallback that saves demo day if the campus Wi-Fi drops
— which the hosting plan flags as the likeliest failure.

Skeleton is in [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) Lane A.

**Gate:** `mosquitto_sub` on `home/demo/#` shows a state message every second
whose JSON parses against `dashboard/lib/contracts.ts` with no field missing.

### 3.2 Dashboard, remaining panels

Order matters — this is the order in which each panel becomes demonstrable:

1. Panel 1 — live demand, sparkline, sanctioned-load line
2. Panel 2 — grid condition and the peak badge
3. Panel 6 — safety blocks, with the reason rendered
4. Panels 3, 5, 8, 9
5. Panel 10 — results, **after** the backend exists

**Gate:** with the replay publisher running, a reviewer can watch a peak begin
and see the PROTECTED row visibly not move. That is the project's claim,
rendered. If that is not obvious on screen, the panel is not finished.

### 3.3 Grid controller screen

The second dashboard: declare a peak event, set severity, see which homes
responded. See [GRID_CONTROLLER_AND_TARIFF.md](GRID_CONTROLLER_AND_TARIFF.md) §7.

**Gate:** declaring an event from this screen changes the resident screen within
one control interval, and the resident's fan does not move.

### 3.4 FastAPI backend

Only three things need it: the override path (Panel 7), the metrics (Panel 10),
and history beyond the session. **Live state does not need the backend.**

**Gate:** an override posted from the phone is either applied or refused with a
reason from the closed `RejectedReason` set — never silently dropped.

### 3.5 The rig

Follow [HARDWARE_PROTOTYPE_SPEC.md](HARDWARE_PROTOTYPE_SPEC.md) build stages.

**Gate:** `handover/sharp_rl_v1/verify_integration.py` passes on the Pi itself,
and the golden vector asserts at boot. If the golden vector fails, the Pi must
refuse to run the policy — a wrong policy is worse than no policy.

---

## 4. Rough effort

| Piece | Effort | Blocks |
|---|---|---|
| Replay publisher | 4–5 h | everything else |
| Remaining panels | 10–12 h | demo |
| Grid controller screen | 4 h | the equity story |
| FastAPI backend | 6–8 h | overrides, metrics |
| Rig | hardware team | the physical demo |

**The replay publisher plus the remaining panels is a complete live demo.** The
backend and the rig are both optional for demo day and should be treated that
way if time runs short.

---

## 5. Reproducing every number in this document

Use the project interpreter. Git Bash's `python` is 3.14 and has no numpy; the
one with the dependencies is `./.venv` (3.12.12). The trap here is a run that
appears to pass because its exit code was swallowed by a pipe — check `$?`, not
the last line of output.

```bash
./.venv/Scripts/python.exe scripts/validate_sharp_model_end_to_end_v1.py \
    --checkpoint handover/sharp_rl_v1/sharp_policy.npz
./.venv/Scripts/python.exe scripts/evaluate_sharp_policy_v1.py --episodes 30 --seed 7
./.venv/Scripts/python.exe scripts/verify_kaggle_training_notebook_v1.py
./.venv/Scripts/python.exe scripts/validate_docs_against_decisions_v1.py
./.venv/Scripts/python.exe scripts/validate_hardware_spec_v1.py
```

```bash
cd handover/sharp_rl_v1 && python verify_integration.py
```

```bash
cd dashboard && npm install && npm run build
```

`evaluate_sharp_policy_v1.py` **overwrites** `reports/policy_evaluation_v1.json`.
The committed copy is the shipped policy at 30 days, seed 7. If you evaluate
anything else, restore it afterwards with
`git checkout -- reports/policy_evaluation_v1.json`.

---

## 6. What must not be claimed

Carried from the individual specs and collected here, because these are the
sentences a reviewer will test:

- **No cost saving.** APCPDCL has no time-of-day domestic tariff. Peak and PAR
  are the winnable metrics; say so before anyone asks.
- **Wattages are simulated.** No meter is fitted. Label them as simulated
  wherever they appear.
- **Not approved for deployment.** The model is validated in simulation and on
  held-out data. That is not the same thing.
- **The preference reward is fitted but inert.** It is not in the training
  reward, and every one of the 3,180 preference pairs points the same way
  (ON over OFF), so its accuracy figure is meaningless. Do not put it in the
  abstract.
- **Dimming does not happen in this build.** Level 2 exists in the action space
  and is never selected. Fans and lights are never dimmed, by design.
- **The imitation ceiling is not what a naive i.i.d. calculation gives.** Logged
  actions are highly persistent — median 0.833, with 77.9 % of devices at or
  above 80 % persistence. Correction due to Pranitha.
