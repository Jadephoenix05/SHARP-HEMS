"""Run the trained policy through the simulator and measure whether it helps.

Why this exists. Everything up to now measured the policy against the logged
actions - how often it agrees with three scripted controllers. That is a sanity
check, not a result, and it cannot tell you whether the controller saves money,
cuts the peak, or leaves people uncomfortable.

This closes the loop. The same households, the same days, the same weather and
the same outages are replayed three times: once serving whatever the occupant
asked for, once under the rule-based peak-aware controller, and once under the
learned policy. The only thing that differs is the decision, so the difference
in outcome is attributable to the decision.

What is measured, per household-day:

    cost               rupees billed, on the real APCPDCL telescopic slab
    peak               highest 15-minute demand
    peak-to-average    the ratio the demand-response literature reports
    unserved service   hours the occupant asked for and did not get
    overrides          times the occupant countermanded the controller
    comfort            degree-steps outside the comfort band
    safety             necessity loads shed while the occupant wanted them

The last one must be zero. A controller that saves money by cutting the fridge
has not solved the problem.

Held-out only: validation households and validation dates, both disjoint from
training. The test split is not touched.
"""
from pathlib import Path
import argparse
import json
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

N_BRANCHES = 28
N_LEVELS = 3
STEP_HOURS = 0.25


def check(condition, message):
    if not condition:
        raise ValueError(message)


def load_policy(checkpoint):
    weights = np.load(checkpoint, allow_pickle=False)
    p = {k: weights[k] for k in ['w', 'b', 'v', 'vb', 'a', 'ab']}
    mean, sd = weights['mean'], weights['sd']
    check(len(mean) == p['w'].shape[0], 'Normalisation length does not match the trunk')

    def act(*, state, device_present, supports_reduced, is_air_conditioner,
            occupant_wants, budget):
        x = ((state - mean) / sd)[None, :]
        h = np.maximum(0, x @ p['w'] + p['b'])
        advantage = (h @ p['a'] + p['ab']).reshape(N_BRANCHES, N_LEVELS)
        q = (h @ p['v'] + p['vb']).reshape(1, 1) + advantage - advantage.mean(
            1, keepdims=True)

        n = len(supports_reduced)
        legal = np.ones((n, N_LEVELS), bool)
        legal[:, 2] = supports_reduced          # a device that cannot dim, cannot dim
        level = np.argmax(np.where(legal, q[:n], -np.inf), axis=1)

        # SHARP decides how fully to serve what the occupant asked for. It does
        # not switch on appliances nobody wants - that is not demand response,
        # and it is not what the behaviour policies did either, so allowing it
        # would make the comparison meaningless.
        level = np.where(occupant_wants, level, 0)
        return level

    return act


def serve_preferred_reference(*, state, device_present, supports_reduced,
                              is_air_conditioner, occupant_wants, budget):
    """The no-demand-response baseline: give the occupant exactly what they asked."""
    return occupant_wants.astype(int)


def summarise(frame, label):
    """Per household-day outcomes, then averaged. Ratios are averaged per day,
    never computed from pooled sums, which would let big households dominate."""
    rows = []
    for episode, g in frame.groupby('episode_id'):
        g = g.sort_values('step_id')
        power = g.aggregate_power_kw.to_numpy(float)
        mean_power = power.mean()
        rows.append({
            'episode_id': episode,
            # The episode id ends in the policy name, so it differs between arms.
            # Pair on the household-day instead: that is what is actually held
            # constant across the three replays.
            'household_day': ':'.join(str(episode).split(':')[:3]),
            'household_id': g.household_id.iloc[0],
            'cost_inr': float(g.reward_cost_inr.sum()),
            'import_kwh': float(g.grid_import_kwh.sum()),
            'peak_kw': float(power.max()),
            'mean_kw': float(mean_power),
            'peak_to_average': float(power.max() / mean_power) if mean_power > 0 else np.nan,
            'overrides': int(g.override_count.iloc[-1]),
            'discomfort': float(g.reward_discomfort.sum()),
            'unserved_kwh': float(g.unserved_demand_kw.sum() * STEP_HOURS),
            'infeasible_steps': int((~g.constraint_feasible.astype(bool)).sum()),
            'outage_steps': int(g.grid_absent.astype(bool).sum()),
        })
    per_day = pd.DataFrame(rows)
    per_day['policy'] = label
    return per_day


def compare(baseline, treatment, column, lower_is_better=True):
    """Paired comparison on the same household-days."""
    joined = baseline[['household_day', column]].merge(
        treatment[['household_day', column]], on='household_day',
        suffixes=('_base', '_new'))
    check(len(joined) == len(baseline), 'Episodes did not pair up')
    base = joined[f'{column}_base'].to_numpy(float)
    new = joined[f'{column}_new'].to_numpy(float)
    difference = new - base
    with np.errstate(divide='ignore', invalid='ignore'):
        relative = np.where(np.abs(base) > 1e-9, difference / base, np.nan)
    improved = difference < 0 if lower_is_better else difference > 0
    pooled = (float(100 * (np.nanmean(new) - np.nanmean(base)) / np.nanmean(base))
              if abs(np.nanmean(base)) > 1e-9 else float('nan'))
    return {
        'baseline_mean': float(np.nanmean(base)),
        'policy_mean': float(np.nanmean(new)),
        'absolute_change': float(np.nanmean(difference)),
        # Three views, because they disagree and each can mislead alone. The
        # mean of per-day ratios is dominated by low-consumption days; the
        # pooled change is dominated by large households; the median is robust
        # to both and is the one to quote.
        'percent_change_mean_of_ratios': float(100 * np.nanmean(relative)),
        'percent_change_median': float(100 * np.nanmedian(relative)),
        'percent_change_pooled': pooled,
        'household_days_improved': int(improved.sum()),
        'household_days_total': int(len(difference)),
    }


GLOBAL_FEATURES = 25
DEVICE_FEATURES = 10


def safety_audit(frame, models):
    """Was a necessity load shed while the occupant wanted it AND could have it?

    The necessity guarantee is conditional, and the condition matters. A fridge
    at level 0 is not a violation if the occupant was not asking for it, if its
    service budget for the day is already met, or if the grid is down and the
    appliance is not on the inverter circuit. Counting those as violations
    inflates the number with cases where OFF was the only correct answer.

    Remaining service budget is read out of the state vector itself - device
    feature 0 is remaining_service_hours - so this uses exactly the quantity the
    shield used when it decided.
    """
    necessity = models.set_index('device_id').is_necessity.to_dict()
    order = (models.sort_values(['template_id', 'device_id'])
             .groupby('template_id').device_id.apply(list).to_dict())
    violations, checked = 0, 0
    for household, g in frame.groupby('household_id'):
        ids = order.get(household)
        if not ids:
            continue
        flags = np.array([bool(necessity.get(d, False)) for d in ids])
        for action, requested, state, absent in zip(
                g.action, g.requested_action, g.state, g.grid_absent):
            if bool(absent):
                continue            # during an outage only the inverter circuit is live
            action = np.asarray(action, int)
            requested = np.asarray(requested, int)
            state = np.asarray(state, float)
            span = min(len(action), len(flags), len(requested))
            budget = np.array([state[GLOBAL_FEATURES + j * DEVICE_FEATURES] * 4
                               for j in range(span)])
            entitled = flags[:span] & (requested[:span] > 0) & (budget > 1e-9)
            checked += int(entitled.sum())
            violations += int((entitled & (action[:span] == 0)).sum())
    return {'necessity_service_opportunities': checked,
            'necessity_shed_while_entitled': violations,
            'condition': ('necessity appliance, occupant requesting it, service '
                          'budget remaining, grid up')}


def evaluate(root, checkpoint, episodes, seed):
    import generate_sharp_rl_transitions_v2 as gen

    shipped = pd.read_parquet(
        root / 'data/processed/sharp_rl_transitions_v2/rl_transitions.parquet',
        columns=['episode_id', 'split', 'policy'])
    pool = sorted(shipped.loc[shipped.split.eq('validation')
                              & shipped.policy.eq('serve_preferred')].episode_id.unique())
    check(len(pool) > 0, 'No validation episodes')
    rng = np.random.default_rng(seed)
    chosen = list(rng.choice(pool, size=min(episodes, len(pool)), replace=False))
    print(f'{len(chosen)} held-out household-days, from {len(pool)} available')

    inputs = gen.load_inputs(root)
    learned = load_policy(root / checkpoint)

    arms = {
        'serve_preferred (no demand response)': ('serve_preferred', None),
        'peak_aware (rule-based)': ('peak_aware', None),
        'SHARP learned policy': ('learned', learned),
    }
    per_day, transitions = {}, {}
    for label, (policy_name, policy_fn) in arms.items():
        started = time.time()
        frame, _, pairs = gen.replay_episodes(
            root, chosen, policy_fn=policy_fn, policy_override=policy_name,
            inputs=inputs)
        transitions[label] = frame
        per_day[label] = summarise(frame, label)
        print(f'  {label:38s} {len(frame):6d} steps  '
              f'{len(pairs):4d} override pairs  {time.time() - started:5.1f}s',
              flush=True)

    models = pd.read_parquet(
        root / 'data/processed/simulator_devices_v1/unknown_quantity_one'
               '/baseline_power_v1/device_power_models.parquet',
        columns=['template_id', 'device_id', 'is_necessity'])

    baseline_label = 'serve_preferred (no demand response)'
    rule_label = 'peak_aware (rule-based)'
    learned_label = 'SHARP learned policy'

    results = {}
    for label in [rule_label, learned_label]:
        results[label] = {
            metric: compare(per_day[baseline_label], per_day[label], metric,
                            lower_is_better=lower)
            for metric, lower in [('cost_inr', True), ('peak_kw', True),
                                  ('peak_to_average', True), ('import_kwh', True),
                                  ('overrides', True), ('discomfort', True),
                                  ('unserved_kwh', True)]}
        results[label]['safety'] = safety_audit(transitions[label], models)

    report = {
        'status': 'POLICY_EVALUATED_IN_SIMULATOR',
        'checkpoint': str(checkpoint),
        'split': 'validation',
        'test_split_used': False,
        'household_days': len(chosen),
        'baseline': baseline_label,
        'arms': list(arms),
        'results_versus_baseline': results,
        'absolute_means': {label: {
            metric: float(frame[metric].mean()) for metric in
            ['cost_inr', 'import_kwh', 'peak_kw', 'peak_to_average', 'overrides',
             'discomfort', 'unserved_kwh', 'infeasible_steps', 'outage_steps']}
            for label, frame in per_day.items()},
        'metric_notes': {
            'cost_inr': 'Billed on the real APCPDCL FY2025-26 telescopic slab.',
            'peak_to_average': 'Averaged per household-day, not pooled.',
            'overrides': 'Synthetic occupant model; counts are not real behaviour.',
            'unserved_kwh': 'Demand the supply could not meet, mostly during outages.',
        },
        'limitations': [
            'The occupant, including every override, is synthetic and generated '
            'from a stated rule. Override counts compare controllers under the '
            'same simulated occupant; they are not evidence about real people.',
            'Appliance power values are proxies for 3,439 of 4,124 devices.',
            'The learned policy chooses a service level for appliances the '
            'occupant asked for; it never switches on an unwanted appliance, '
            'which is also what the scripted policies do.',
            'This is simulator evaluation, not field evidence.',
        ],
        'approved_for_deployment': False,
    }
    out = root / 'reports/policy_evaluation_v1.json'
    out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    pd.concat(per_day.values(), ignore_index=True).to_csv(
        root / 'reports/policy_evaluation_per_day_v1.csv', index=False)

    print(f'\nPOLICY EVALUATION ({len(chosen)} held-out household-days)')
    print(f"{'metric':18s} {'no DR':>10s} {'rule-based':>12s} {'SHARP':>10s}"
          f" {'SHARP vs no DR':>16s}")
    print('(last column: median per-household-day change)')
    for metric in ['cost_inr', 'peak_kw', 'peak_to_average', 'import_kwh',
                   'overrides', 'discomfort', 'unserved_kwh']:
        base = report['absolute_means'][baseline_label][metric]
        rule = report['absolute_means'][rule_label][metric]
        new = report['absolute_means'][learned_label][metric]
        change = results[learned_label][metric]['percent_change_median']
        print(f'{metric:18s} {base:10.3f} {rule:12.3f} {new:10.3f} {change:+15.1f}%')
    print()
    for label in [rule_label, learned_label]:
        safety = results[label]['safety']
        print(f"{label:38s} necessity shed while entitled: "
              f"{safety['necessity_shed_while_entitled']} of "
              f"{safety['necessity_service_opportunities']}")
    print('\nOutput:', out)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--checkpoint', type=Path,
                   default=Path('models/sharp_bdq_v2_cql1.0_balanced_bc_long/checkpoint.npz'))
    p.add_argument('--episodes', type=int, default=60)
    p.add_argument('--seed', type=int, default=7)
    a = p.parse_args()
    evaluate(a.root.resolve(), a.checkpoint, a.episodes, a.seed)
