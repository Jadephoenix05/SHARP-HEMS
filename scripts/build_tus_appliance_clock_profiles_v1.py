"""Derive WHEN appliances are used from real Indian time-use diaries.

The gap this closes. Comparing SHARP against eMARC mainline meters showed the
simulator reproducing almost none of the Indian daily rhythm: shape correlation
+0.205, peak at 19:00 against a real 07:45, and no morning peak at all. The
cause was that appliance clock times were synthetic. Surveys report how many
HOURS an appliance runs, never WHEN, so the preferred-slot model had been
filling that in by assumption.

MoSPI Time Use Survey 2024 does record when. It has 370,205 Andhra Pradesh
activity records with explicit time_from and time_to, so the clock can come from
measured behaviour instead of a guess.

Activity to appliance mapping, deliberately conservative - an appliance is only
mapped where the activity plainly requires it:

  Preparing meals/snacks, Cleaning up after food  -> mixer grinder, rice cooker,
                                                     electric kettle
  Watching/listening to television and video      -> television
  Personal hygiene and care                       -> geyser
  Ironing/pressing/folding                        -> electric iron
  Hand/machine-washing                            -> washing machine

WHAT THIS IS. Measured timing of the ACTIVITY, used as the clock for the
appliance that activity needs. It is not metered appliance timing: TUS records
that someone prepared a meal at 07:15, not that a mixer grinder drew power then.
The distinction matters and is carried in the output.

Appliances with no matching activity keep their existing schedule: fans and
lighting follow occupancy, the refrigerator is thermostatic, the air conditioner
is driven by temperature, and the water pump has no diary equivalent.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd

STEPS = 96

ACTIVITY_TO_APPLIANCE = {
    'mixer_grinder': ['Preparing meals/snacks',
                      'Cleaning up after food preparation/meals/snacks'],
    'electric_rice_cooker': ['Preparing meals/snacks'],
    'electric_kettle': ['Preparing meals/snacks',
                        'Drinking other than with meal or snack'],
    'television': ['Watching/listening to television and video'],
    'geyser': ['Personal hygiene and care'],
    'electric_iron': ['Ironing/pressing/folding'],
    'washing_machine': ['Hand/machine-washing'],
}

UNMAPPED_REASON = {
    'ceiling_fan': 'Follows occupancy, not a discrete diary activity.',
    'table_fan': 'Follows occupancy.',
    'air_cooler': 'Follows occupancy.',
    'led_bulb': 'Lighting follows presence and darkness, not one activity.',
    'led_tube': 'Lighting follows presence and darkness.',
    'cfl_bulb': 'Lighting follows presence and darkness.',
    'cfl_tube': 'Lighting follows presence and darkness.',
    'incandescent_bulb': 'Lighting follows presence and darkness.',
    'refrigerator': 'Thermostatic and always powered.',
    'air_conditioner': 'Driven by indoor temperature, not a diary activity.',
    'modem_router': 'Always on.',
    'water_pump': 'No time-use diary equivalent.',
    'water_purifier': 'No time-use diary equivalent.',
    'laptop_tablet': 'Spans many activities; no single diary match.',
    'desktop': 'Spans many activities; no single diary match.',
}

# Sleep is not an appliance, but it says when a household is awake, which is
# what lighting actually follows. The previous lighting heuristic kept lights on
# from 18:00 right through to 06:00, so every simulated home burned lights all
# night and the daily load trough landed at 23:45 instead of mid-afternoon.
SLEEP_ACTIVITY = 'Night sleep/essential sleep'

MINIMUM_RECORDS = 500


def check(condition, message):
    if not condition:
        raise ValueError(message)


def sleep_profile(frame):
    """Share asleep by slot. Sleep spans midnight, so the wrap is handled."""
    sub = frame[frame.activity_code.eq(SLEEP_ACTIVITY)]
    check(len(sub) > MINIMUM_RECORDS, 'Too few sleep diary records')
    start = pd.to_datetime(sub.time_from, format='%H:%M', errors='coerce')
    end = pd.to_datetime(sub.time_to, format='%H:%M', errors='coerce')
    valid = start.notna() & end.notna()
    a = (start.dt.hour * 4 + start.dt.minute // 15)[valid].to_numpy(int)
    b = (end.dt.hour * 4 + end.dt.minute // 15)[valid].to_numpy(int)
    profile = np.zeros(STEPS)
    for s, e in zip(a, b):
        if e <= s:
            e = e + STEPS      # the entry crosses midnight
        for slot in range(s, e):
            profile[slot % STEPS] += 1
    return profile / profile.max()


def activity_profile(frame, names):
    """Share of reported activity minutes falling in each 15-minute slot."""
    sub = frame[frame.activity_code.isin(names)]
    if len(sub) < MINIMUM_RECORDS:
        return None, len(sub)
    start = pd.to_datetime(sub.time_from, format='%H:%M', errors='coerce')
    end = pd.to_datetime(sub.time_to, format='%H:%M', errors='coerce')
    valid = start.notna() & end.notna()
    a = (start.dt.hour * 4 + start.dt.minute // 15)[valid].to_numpy(int)
    b = (end.dt.hour * 4 + end.dt.minute // 15)[valid].to_numpy(int)
    profile = np.zeros(STEPS)
    for s, e in zip(a, b):
        if e <= s:
            e = s + 1          # a diary entry that ends on its start slot still occupied it
        if e - s > STEPS:
            continue           # malformed span; drop rather than wrap
        for slot in range(s, min(e, STEPS)):
            profile[slot] += 1
    total = profile.sum()
    check(total > 0, f'Empty profile for {names}')
    return profile / total, int(valid.sum())


def build(root):
    source = root / 'data/interim/occupancy_profiles/tus2024_ap_activities.parquet'
    check(source.exists(), f'Missing {source}')
    frame = pd.read_parquet(source, columns=['activity_code', 'time_from',
                                             'time_to', 'day_type'])
    check(len(frame) > 100_000, 'Unexpectedly small TUS activity table')

    profiles, records, skipped = {}, {}, {}
    for appliance, names in ACTIVITY_TO_APPLIANCE.items():
        profile, count = activity_profile(frame, names)
        if profile is None:
            skipped[appliance] = f'only {count} diary records, below {MINIMUM_RECORDS}'
            continue
        profiles[appliance] = profile
        records[appliance] = count

    check(profiles, 'No appliance clock profiles could be built')

    asleep = sleep_profile(frame)
    awake = 1.0 - asleep
    # Lighting is wanted when it is dark AND somebody is awake. Sunrise and
    # sunset in coastal Andhra sit near 06:15 and 18:15 year round.
    dark = np.array([1.0 if (slot < 25 or slot >= 73) else 0.0 for slot in range(STEPS)])
    lighting = dark * awake
    if lighting.sum() > 0:
        profiles['lighting'] = lighting / lighting.sum()
        records['lighting'] = int(len(frame[frame.activity_code.eq(SLEEP_ACTIVITY)]))

    out = root / 'data/processed/tus_appliance_clock_v1'
    out.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame({'step_of_day': range(STEPS),
                          'clock': [f'{s // 4:02d}:{15 * (s % 4):02d}'
                                    for s in range(STEPS)]})
    for appliance, profile in profiles.items():
        table[appliance] = profile
    table.to_csv(out / 'appliance_clock_profiles.csv', index=False)

    summary = {}
    print(f"{'appliance':22s} {'records':>8s} {'peak':>7s} {'morning':>8s} {'evening':>8s}")
    for appliance, profile in profiles.items():
        peak = int(profile.argmax())
        morning = float(profile[20:36].sum())   # 05:00-09:00
        evening = float(profile[68:88].sum())   # 17:00-22:00
        summary[appliance] = {
            'diary_records': records[appliance],
            'peak_step': peak,
            'peak_clock': f'{peak // 4:02d}:{15 * (peak % 4):02d}',
            'morning_share_05_09': morning,
            'evening_share_17_22': evening,
            'activities': ACTIVITY_TO_APPLIANCE.get(
                appliance, [f'derived from {SLEEP_ACTIVITY} and local darkness']),
        }
        print(f'{appliance:22s} {records[appliance]:8d} '
              f'{summary[appliance]["peak_clock"]:>7s} {morning:8.3f} {evening:8.3f}')

    report = {
        'status': 'TUS_APPLIANCE_CLOCK_PROFILES_BUILT',
        'source': 'MoSPI Time Use Survey 2024, Andhra Pradesh activity diaries',
        'activity_records': int(len(frame)),
        'mapped_appliances': summary,
        'unmapped_appliances': {**UNMAPPED_REASON, **skipped},
        'basis': 'MEASURED_ACTIVITY_TIMING_NOT_METERED_APPLIANCE_TIMING',
        'limitations': [
            'TUS records when an activity happened, not when an appliance drew power.',
            'A household may prepare a meal without using a mixer grinder.',
            'Diaries are per person; simultaneous activity is recorded separately.',
            'Profiles are pooled across Andhra Pradesh, not per household.',
            'Appliances with no diary equivalent keep their previous schedule.',
        ],
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    (out / 'clock_profile_report.json').write_text(json.dumps(report, indent=2),
                                                   encoding='utf-8')
    print(f'\nTUS APPLIANCE CLOCK PROFILES BUILT')
    print(f'  mapped: {len(profiles)} appliances from {len(frame):,} diary records')
    print(f'  unmapped: {len(UNMAPPED_REASON) + len(skipped)} (reason recorded for each)')
    print('  output:', out)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    a = p.parse_args()
    build(a.root.resolve())
