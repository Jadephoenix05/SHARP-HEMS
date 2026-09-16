# SHARP — Shielded Human-override Adaptive Reward Personalisation

A 15-minute smart-home demand-response simulator and reinforcement-learning
system for **Guntur / Vijayawada, Andhra Pradesh**, under APCPDCL.

This repository holds the **build definition**: scripts, configs, the source
registry and validation reports. It holds no datasets. The approved data release
lives in a private Kaggle dataset; see
[docs/RELEASE_AND_UPLOAD_GUIDE.md](docs/RELEASE_AND_UPLOAD_GUIDE.md).

## Pipeline

```
official sources and configuration
  -> GitHub-controlled build and validation scripts
  -> canonical processed inputs
  -> household simulator (shield, thermal, appliance dynamics)
  -> tariff, billing and reward
  -> RL transitions (state, action, reward, next_state, done)
  -> Branching Dueling Q-learning
  -> Raspberry Pi and dashboard deployment
```

**Public datasets do not train the policy.** REFIT, IRES, BEE/CLASP, eMARC,
iAWE, RESIDE-AC, TUS, NASA POWER and Grid-India define, calibrate and validate
the simulator. BDQ trains on the generated transition layer.

## Where to start

**[docs/PROJECT_COMPLETION_PLAN.md](docs/PROJECT_COMPLETION_PLAN.md)** — what is
finished, what is left, the order to build it in, and the gate each piece must
pass. Every other document covers one subsystem; that one covers the whole
project. Read it first.

| Document | Covers |
|---|---|
| [PROJECT_COMPLETION_PLAN.md](docs/PROJECT_COMPLETION_PLAN.md) | Status, remaining work, acceptance gates |
| [RL_MODEL_SPEC.md](docs/RL_MODEL_SPEC.md) | State, action space, BDQ, offline RL, export |
| [INTEGRATION_GUIDE.md](docs/INTEGRATION_GUIDE.md) | Frozen contracts across model, Pi and dashboard |
| [DASHBOARD_BUILD_PLAN.md](docs/DASHBOARD_BUILD_PLAN.md) | Who builds what, in what order |
| [DASHBOARD_SPECIFICATION.md](docs/DASHBOARD_SPECIFICATION.md) | Panels, contracts, what must not be claimed |
| [DASHBOARD_GUIDE.md](docs/DASHBOARD_GUIDE.md) | Architecture, FastAPI service, deploy |
| [HARDWARE_PROTOTYPE_SPEC.md](docs/HARDWARE_PROTOTYPE_SPEC.md) | Ten appliances, GPIO map, interlocks |
| [GRID_CONTROLLER_AND_TARIFF.md](docs/GRID_CONTROLLER_AND_TARIFF.md) | Peak events, the tariff, the law |
| [HOSTING_AND_CONNECTIVITY_PLAN.md](docs/HOSTING_AND_CONNECTIVITY_PLAN.md) | Broker, topics, demo-day contingency |
| [RELEASE_AND_UPLOAD_GUIDE.md](docs/RELEASE_AND_UPLOAD_GUIDE.md) | Building and uploading the private release |

## Key modules

| Path | Role |
|---|---|
| `scripts/sharp_thermal_rc.py` | Lumped-RC room with a thermostat. AC setpoint is the action; compressor duty is an output. |
| `scripts/sharp_action_shield.py` | Hard safety shield and legal-action masks. |
| `scripts/sharp_apcpdcl_tariff.py` | APCPDCL domestic LT tariff. |
| `scripts/sharp_reward_billing.py` | Billing ledger and reward components. |
| `scripts/sharp_transition_core.py` | Atomic shield to dynamics to billing to reward step. |
| `scripts/generate_sharp_rl_transitions_v1.py` | Builds the RL transition layer. |
| `scripts/train_sharp_bdq_v1.py` | Branching Dueling Double-Q training. |
| `scripts/build_sharp_master_release_v1.py` | Assembles the Kaggle release. |
| `scripts/verify_sharp_release_v1.py` | Verifies a downloaded release. |

## Rebuild order

```bash
.venv\Scripts\python.exe scripts\build_reside_thermal_envelope_v1.py
.venv\Scripts\python.exe scripts\validate_sharp_thermal_rc.py
.venv\Scripts\python.exe scripts\generate_sharp_rl_transitions_v1.py
.venv\Scripts\python.exe scripts\finalize_master_registry_v1.py
.venv\Scripts\python.exe scripts\build_sharp_master_release_v1.py
.venv\Scripts\python.exe scripts\verify_sharp_release_v1.py
.venv\Scripts\python.exe scripts\train_sharp_bdq_v1.py
```

## Scientific rules this repository enforces

These are checked in code and recorded in the generated reports, not just
asserted in prose:

- REFIT is **UK** evidence. It is never presented as Indian household data.
- RESIDE-AC is **11 Hyderabad houses over 19 days**, never an Indian population,
  and never Andhra Pradesh.
- iAWE is **one New Delhi home**, never a population distribution.
- No **causal AC cooling effect** is established. A weather-conditioned fit was
  attempted and rejected: it lost to plain persistence in 7 of 11 houses and
  implied envelope time constants of 9 hours to 9.5 days. Thermal parameters are
  declared assumptions bounded by the observed RESIDE envelope.
- Proxy wattages are **not** measured appliance ratings.
- Experimental time-of-use multipliers are **not** official APCPDCL ToD tariffs.
- Zero REFIT power is **not** a confirmed physical OFF event.
- A successful training run is **not** evidence of convergence, generalisation
  or deployability. Nothing here is approved for hardware control.
- Splits are household-disjoint **and** date-disjoint; grid peak thresholds are
  fitted on training years only.
- Generated Parquet outputs are never hand-edited. Fix the script or config and
  rebuild.

## One resolved question worth noting

RESIDE-AC's CSV timestamps read May 2019 while its description says May 2021.
A pre-registered test correlated each house's daily indoor temperature against
Hyderabad outdoor temperature for both candidate years. **2019 is supported**:
mean r 0.524 versus 0.126, favoured by 10 of 11 houses on AC-off intervals and
9 of 11 on all intervals. This is statistical evidence about the clock, not a
correction from the dataset authors, and they have not been asked.
