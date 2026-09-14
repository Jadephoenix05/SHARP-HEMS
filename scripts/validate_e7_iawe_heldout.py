"""E7: Indian validation against iAWE, including outage mode.

The project's acceptance criterion E7 asks for Indian validation on iAWE with
mode-3 behaviour against real outage data. This compares SHARP's simulated
household load with iAWE's measured whole-home load, and its simulated outages
with the supply interruptions visible in that meter.

WHAT IS GENUINELY HELD OUT, AND WHAT IS NOT. This matters more than the numbers.
iAWE channels 3, 4, 5, 7 and 10 were used as appliance power proxies in
configure_sharp_appliance_models.py, so those appliances are NOT held out: the
simulator was partly fitted to them. The MAINS channels 1 and 2 were never used
for anything, so whole-home load and supply interruptions are a real held-out
comparison. The project's own notes are blunt about the risk: "Train and test on
the same single home and you have leakage, not a result."

SCOPE. iAWE is ONE New Delhi household over about 73 days in 2013. It is a
plausibility check on a single real Indian home, not a population test and not
Andhra Pradesh. A disagreement here is informative; an agreement proves much
less than it appears to.
"""
from pathlib import Path
import argparse
import glob
import json
import numpy as np
import pandas as pd

STEP_HOURS = 0.25
MAINS_CHANNELS = (1, 2)
CALIBRATION_CHANNELS = (3, 4, 5, 7, 10)
OUTAGE_THRESHOLD_W = 1.0   # a mains meter reading essentially nothing


def check(condition, message):
    if not condition:
        raise ValueError(message)


def load_mains(root):
    frames = []
    for path in sorted(glob.glob(str(root / 'data/interim/iawe/15min/*.parquet'))):
        frame = pd.read_parquet(path, columns=['channel_id', 'appliance_name',
                                               'interval_start_utc',
                                               'power_sample_mean_w'])
        if int(frame.channel_id.iloc[0]) in MAINS_CHANNELS:
            frames.append(frame)
    check(frames, 'No iAWE mains channels found')
    mains = pd.concat(frames, ignore_index=True)
    # The two mains channels are separate phases of the same supply, so the
    # household draw is their sum at each interval, not either one alone.
    total = (mains.groupby('interval_start_utc').power_sample_mean_w
             .sum().rename('household_w').reset_index())
    stamp = pd.to_datetime(total.interval_start_utc, utc=True).dt.tz_convert('Asia/Kolkata')
    total['step_of_day'] = stamp.dt.hour * 4 + stamp.dt.minute // 15
    total['date'] = stamp.dt.strftime('%Y-%m-%d')
    return total


def build(root):
    mains = load_mains(root)
    transitions = pd.read_parquet(
        root / 'data/processed/sharp_rl_transitions_v2/rl_transitions.parquet',
        columns=['episode_id', 'step_id', 'policy', 'aggregate_power_kw',
                 'grid_absent'])
    sharp = transitions[transitions.policy.eq('serve_preferred')]
    check(len(sharp) > 0, 'No serve_preferred rows')

    # Whole-home daily energy.
    daily = mains.groupby('date').agg(kwh=('household_w', lambda s: float(s.sum())
                                           * STEP_HOURS / 1000.0),
                                      readings=('household_w', 'size')).reset_index()
    complete = daily[daily.readings.eq(96)]
    check(len(complete) > 10, 'Too few complete iAWE days')
    sharp_daily = (sharp.groupby('episode_id').aggregate_power_kw.sum()
                   .mul(STEP_HOURS))

    # Diurnal shape, normalised so scale does not dominate.
    iawe_profile = mains.groupby('step_of_day').household_w.mean() / 1000.0
    sharp_profile = sharp.groupby('step_id').aggregate_power_kw.mean()
    iawe_shape = (iawe_profile / iawe_profile.mean()).to_numpy()
    sharp_shape = (sharp_profile / sharp_profile.mean()).to_numpy()
    correlation = float(np.corrcoef(iawe_shape, sharp_shape)[0, 1])

    # Mode 3. An interval where the mains meter reads essentially nothing is a
    # supply interruption, which is what the project cites iAWE for.
    mains['is_outage'] = mains.household_w < OUTAGE_THRESHOLD_W
    outage_share = float(mains.is_outage.mean())
    per_day = mains.groupby('date').is_outage.sum().mul(STEP_HOURS)
    sharp_outage_share = float(sharp.grid_absent.mean())

    def clock(step):
        return f'{step // 4:02d}:{15 * (step % 4):02d}'

    report = {
        'status': 'E7_IAWE_HELDOUT_COMPARED',
        'held_out': {
            'mains_channels': list(MAINS_CHANNELS),
            'note': 'Never used to fit anything, so this comparison is held out.',
        },
        'not_held_out': {
            'calibration_channels': list(CALIBRATION_CHANNELS),
            'note': ('Used as appliance power proxies for the refrigerator, air '
                     'conditioner, laptop and television, so those appliances '
                     'were partly fitted to this same household.'),
        },
        'scope': 'ONE New Delhi household, about 73 days in 2013. Not a population.',
        'daily_energy_kwh': {
            'iawe_median': float(complete.kwh.median()),
            'iawe_p25': float(complete.kwh.quantile(0.25)),
            'iawe_p75': float(complete.kwh.quantile(0.75)),
            'iawe_complete_days': int(len(complete)),
            'sharp_median': float(sharp_daily.median()),
            'ratio_sharp_over_iawe': float(sharp_daily.median() / complete.kwh.median()),
        },
        'diurnal_shape': {
            'correlation': correlation,
            'iawe_peak_clock': clock(int(iawe_profile.idxmax())),
            'sharp_peak_clock': clock(int(sharp_profile.idxmax())),
            'iawe_trough_clock': clock(int(iawe_profile.idxmin())),
            'sharp_trough_clock': clock(int(sharp_profile.idxmin())),
        },
        'outage_mode': {
            'iawe_interval_share_below_threshold': outage_share,
            'iawe_hours_per_day_median': float(per_day.median()),
            'iawe_hours_per_day_max': float(per_day.max()),
            'sharp_interval_share': sharp_outage_share,
            'sharp_source': 'IRES-reported grid_supply_hours_daily for AP households',
            'threshold_w': OUTAGE_THRESHOLD_W,
            'caveat': ('A mains reading near zero can be a genuine interruption or '
                       'a metering dropout; the two cannot be separated here.'),
        },
        'interpretation': [
            'iAWE is New Delhi in 2013; SHARP is Andhra Pradesh in 2021-2024. '
            'Differences in level and in outage frequency are expected.',
            'Appliance power for the fridge, air conditioner, laptop and television '
            'came from this same household, so agreement on those is not evidence.',
            'The mains comparison is the part that carries weight, and one home '
            'cannot establish population behaviour either way.',
        ],
        'is_population_validation': False,
    }
    (root / 'reports/e7_iawe_heldout_v1.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8')

    print('E7 IAWE HELD-OUT COMPARISON')
    print(f"  iAWE complete days: {len(complete)}")
    print(f"  daily kWh   iAWE median {report['daily_energy_kwh']['iawe_median']:.2f}"
          f"  SHARP {report['daily_energy_kwh']['sharp_median']:.2f}"
          f"  ratio {report['daily_energy_kwh']['ratio_sharp_over_iawe']:.2f}x")
    print(f"  shape correlation: {correlation:+.3f}")
    print(f"  peak    iAWE {report['diurnal_shape']['iawe_peak_clock']}"
          f"   SHARP {report['diurnal_shape']['sharp_peak_clock']}")
    print(f"  outage  iAWE {report['outage_mode']['iawe_hours_per_day_median']:.2f} h/day median"
          f" (max {report['outage_mode']['iawe_hours_per_day_max']:.2f})")
    print(f"          SHARP interval share {sharp_outage_share:.4f}"
          f" vs iAWE {outage_share:.4f}")
    print('  held out: mains only. Channels 3/4/5/7/10 calibrated the simulator.')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    a = p.parse_args()
    build(a.root.resolve())
