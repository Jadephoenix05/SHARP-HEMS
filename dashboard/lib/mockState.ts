/**
 * A fake day, so the UI can be built and demonstrated with no broker, no Pi and
 * no network.
 *
 * Two jobs. It unblocks the presentation work while the data layer is being
 * written, and it is the fallback that saves the demo if the campus Wi-Fi drops
 * - which the hosting plan flags as the single most likely thing to go wrong.
 *
 * It must therefore cover every state the UI has to render, not just the happy
 * one: normal, peak active, a shed luxury load, a refused override, and an
 * outage.
 */

import {
  ApplianceState,
  ConnectionStatus,
  Feed,
  HomeState,
  Intent,
} from './contracts';

/** Wattages and flags come from the shipped dataset, not invented. */
const CATALOGUE: Array<
  Pick<
    ApplianceState,
    'appliance_id' | 'appliance_type' | 'service_class' | 'is_necessity'
    | 'supports_reduced' | 'power_15min_mean_w'
  >
> = [
  { appliance_id: 'ceiling_fan_01', appliance_type: 'ceiling_fan', service_class: 'critical', is_necessity: true, supports_reduced: true, power_15min_mean_w: 60.0 },
  { appliance_id: 'table_fan_01', appliance_type: 'table_fan', service_class: 'critical', is_necessity: true, supports_reduced: true, power_15min_mean_w: 40.0 },
  { appliance_id: 'led_bulb_01', appliance_type: 'led_bulb', service_class: 'critical', is_necessity: true, supports_reduced: true, power_15min_mean_w: 9.0 },
  { appliance_id: 'led_tube_01', appliance_type: 'led_tube', service_class: 'critical', is_necessity: true, supports_reduced: true, power_15min_mean_w: 18.0 },
  { appliance_id: 'refrigerator_01', appliance_type: 'refrigerator', service_class: 'critical', is_necessity: true, supports_reduced: false, power_15min_mean_w: 43.2 },
  { appliance_id: 'air_conditioner_01', appliance_type: 'air_conditioner', service_class: 'thermostatic', is_necessity: false, supports_reduced: false, power_15min_mean_w: 1328.4 },
  { appliance_id: 'washing_machine_01', appliance_type: 'washing_machine', service_class: 'deferrable', is_necessity: false, supports_reduced: false, power_15min_mean_w: 113.7 },
  { appliance_id: 'ev_charger_01', appliance_type: 'ev_charger', service_class: 'deferrable', is_necessity: false, supports_reduced: false, power_15min_mean_w: 1500.0 },
  { appliance_id: 'television_01', appliance_type: 'television', service_class: 'interruptible', is_necessity: false, supports_reduced: false, power_15min_mean_w: 104.6 },
  { appliance_id: 'mixer_grinder_01', appliance_type: 'mixer_grinder', service_class: 'interruptible', is_necessity: false, supports_reduced: false, power_15min_mean_w: 500.0 },
];

/**
 * Grid severity by hour, from the shipped dataset.
 *
 * The peak is 08:00-17:00 - agricultural pumping and commercial cooling - and
 * severity is EXACTLY ZERO after 18:00, while household load peaks at 19:00.
 * Any demo built around an evening peak will never light the badge.
 */
const SEVERITY_BY_HOUR = [
  0, 0, 0, 0, 0, 0, 0, 0,
  0.42, 0.66, 0.67, 0.67, 0.67, 0.67, 0.59, 0.57,
  0.55, 0.47, 0, 0, 0, 0, 0, 0,
];

/** Household demand by hour, also from the dataset. Peaks at 19:00. */
const HOUSEHOLD_KW_BY_HOUR = [
  0.235, 0.195, 0.167, 0.165, 0.167, 0.158, 0.175, 0.186,
  0.166, 0.168, 0.150, 0.169, 0.168, 0.180, 0.179, 0.155,
  0.151, 0.144, 0.196, 0.245, 0.223, 0.212, 0.124, 0.116,
];

/** A plausible evening/daytime usage pattern per appliance. */
function occupantWants(id: string, hour: number): boolean {
  switch (id) {
    case 'refrigerator_01': return true;
    case 'ceiling_fan_01': return hour >= 10 || hour <= 6;
    case 'table_fan_01': return hour >= 12 && hour <= 23;
    case 'led_bulb_01': return hour >= 18 || hour <= 6;
    case 'led_tube_01': return hour >= 19 || hour <= 5;
    case 'television_01': return hour >= 19 && hour <= 22;
    case 'mixer_grinder_01': return hour === 7 || hour === 18;
    case 'air_conditioner_01': return hour >= 13 && hour <= 16;
    case 'washing_machine_01': return hour >= 10 && hour <= 12;
    case 'ev_charger_01': return hour >= 20 || hour <= 6;
    default: return false;
  }
}

export interface MockOptions {
  /** Force an outage, to exercise the islanded view. */
  outage?: boolean;
  /** Force a peak even outside 08:00-17:00, for rehearsing. */
  forcePeak?: boolean;
}

/** step: 0..95, one simulated day at 15-minute resolution. */
export function mockHomeState(step: number, options: MockOptions = {}): HomeState {
  const t = ((step % 96) + 96) % 96;
  const hour = Math.floor((t * 15) / 60);
  const severity = options.forcePeak ? 0.67 : SEVERITY_BY_HOUR[hour];
  const peakActive = severity >= 0.5;
  const outage = options.outage ?? false;

  const appliances: ApplianceState[] = CATALOGUE.map((base) => {
    const wants = occupantWants(base.appliance_id, hour);
    const protectedNow = base.is_necessity && wants;

    // Luxury is locked out during a peak. A critical load never is - the shield
    // raises rather than obeying if anyone tries.
    const lockedOut = peakActive && !base.is_necessity && !outage;

    let level: 0 | 1 | 2 = wants ? 1 : 0;
    if (lockedOut) level = 0;
    if (outage && !base.is_necessity) level = 0;
    if (protectedNow) level = 1;

    // The washer and the EV are deferred rather than simply refused.
    const deferred =
      base.service_class === 'deferrable' && (lockedOut || (peakActive && wants));
    if (deferred) level = 0;

    return {
      ...base,
      level,
      occupant_wants: wants,
      measured_w: null,               // no meter fitted in this build
      gpio_state: level > 0 ? 1 : 0,  // readback agrees, as it should
      actuation_verified: true,
      remaining_service_hours: wants ? 2.5 : 0,
      peak_locked_out: lockedOut,
    };
  });

  const controllable =
    appliances.reduce((sum, a) => sum + (a.level > 0 ? a.power_15min_mean_w : 0), 0) / 1000;
  const background = 0.106;           // 61 % of load, and not controllable

  return {
    house_id: 'demo',
    timestamp_ist: new Date(Date.UTC(2026, 8, 15, hour - 5, (t * 15) % 60)).toISOString(),
    aggregate_power_kw: outage ? 0 : Number((controllable + background).toFixed(4)),
    background_load_kw: outage ? 0 : background,
    sanctioned_load_kw: 2.0,
    indoor_temperature_c: 28 + Math.sin((t / 96) * Math.PI * 2) * 4,
    outdoor_temperature_c: 30 + Math.sin(((t - 20) / 96) * Math.PI * 2) * 8,
    occupancy_adult_home_fraction: hour >= 18 || hour <= 7 ? 1 : 0.3,
    attention_available: hour >= 18 || hour <= 7,
    marginal_tariff_inr_kwh: 4.5,
    month_to_date_kwh: 96 + t * 0.04,
    grid_peak_severity: severity,
    peak_event_active: peakActive,
    peak_event_expires_at: peakActive
      ? new Date(Date.UTC(2026, 8, 15, 11, 30)).toISOString()
      : null,
    operating_mode: outage ? 'islanded_outage' : 'grid_import',
    battery_state_of_charge: outage ? 0.62 : 1,
    grid_absent: outage,
    appliances,
  };
}

/** An intent that includes a REFUSAL, because Panel 6 has to render one. */
export function mockIntent(state: HomeState): Intent {
  const proposed: Intent['proposed'] = {};
  const executed: Intent['executed'] = {};
  const shield_reasons: Intent['shield_reasons'] = {};

  for (const a of state.appliances) {
    executed[a.appliance_id] = a.level;
    if (a.is_necessity && a.occupant_wants && state.peak_event_active) {
      // The agent asked to shed an essential load; the shield refused it.
      proposed[a.appliance_id] = 0;
      shield_reasons[a.appliance_id] = ['necessity_mask'];
    } else if (a.peak_locked_out) {
      proposed[a.appliance_id] = 0;
      shield_reasons[a.appliance_id] = ['peak_lockout'];
    } else {
      proposed[a.appliance_id] = a.level;
    }
  }

  return {
    command_id: `mock-${state.timestamp_ist}`,
    proposed,
    executed,
    shield_reasons,
    policy_source: 'bdq_v2_cql',
    decision_latency_ms: 1.2,
  };
}

/**
 * A complete feed, advancing on its own. Drop-in replacement for the live hook:
 * swapping useMockFeed for useHomeState should be ONE line. If it is more, the
 * seam between the data and presentation layers was drawn in the wrong place.
 */
export function mockFeed(step: number, options: MockOptions = {}): Feed {
  const state = mockHomeState(step, options);
  return {
    status: 'live' as ConnectionStatus,
    dataAgeSeconds: 0,
    state,
    intent: mockIntent(state),
  };
}

/** Step index for a wall-clock hour, for jumping straight to the demo moment. */
export function stepForHour(hour: number): number {
  return Math.floor((hour * 60) / 15);
}

/** The demo runs at MIDDAY. Severity is zero after 18:00. */
export const DEMO_PEAK_STEP = stepForHour(12);
