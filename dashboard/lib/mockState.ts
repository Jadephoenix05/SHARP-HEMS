import { HomeState, PeakEvent, SystemConnectionStatus } from './contracts';

export const INITIAL_PEAK_EVENT: PeakEvent = {
  event_id: 'apcpdcl-2026-09-15-001',
  sequence: 47,
  region: 'APCPDCL / Guntur Urban Sub-12',
  severity: 0.67,
  declared_at: '14:05 IST',
  expires_at: '16:00 IST',
  reason: 'system_peak',
  is_active: true,
  homes_responding: 431,
  mw_relieved: 0.28,
};

export const MOCK_SYSTEM_STATUS: SystemConnectionStatus = {
  status: 'connected',
  data_age_seconds: 1,
  broker_endpoint: 'wss://hivemq.sharp-mesh.internal:443/mqtt (WSS)',
  last_verified_at: '14:05:02 IST',
  is_fallback: false,
};

/**
 * Normal Operating State (Midday, normal grid condition)
 */
export const NORMAL_HOME_STATE: HomeState = {
  house_id: 'AP-GNT-H042',
  timestamp_ist: '2026-09-15T12:30:00+05:30',
  aggregate_power_kw: 0.412,
  background_load_kw: 0.084, // 61% baseline unmodelled draw
  sanctioned_load_kw: 1.0,
  indoor_temperature_c: 30.8,
  outdoor_temperature_c: 35.2,
  occupancy_adult_home_fraction: 0.83,
  attention_available: true,
  marginal_tariff_inr_kwh: 4.50,
  month_to_date_kwh: 96.4,
  grid_peak_severity: 0.15,
  operating_mode: 'grid_import',
  battery_state_of_charge: 0.85,
  grid_absent: false,
  data_age_seconds: 1,
  step_id: 50,
  appliances: [
    // PROTECTED NECESSITY LOADS (Must never offer OFF controls)
    {
      appliance_id: 'fridge_01',
      display_name: 'Refrigerator (Inverter Double Door)',
      appliance_type: 'refrigerator',
      service_class: 'necessity',
      is_necessity: true,
      supports_reduced: false,
      level: 1,
      power_15min_mean_w: 47.7,
      measured_w: null,
      gpio_state: 1,
      actuation_verified: true,
      remaining_service_hours: 24.0,
      shed_reason: null,
    },
    {
      appliance_id: 'fan_01',
      display_name: 'Ceiling Fan (Master Bed)',
      appliance_type: 'ceiling_fan',
      service_class: 'necessity',
      is_necessity: true,
      supports_reduced: true,
      level: 1,
      power_15min_mean_w: 60.0,
      measured_w: null,
      gpio_state: 1,
      actuation_verified: true,
      remaining_service_hours: 8.5,
      shed_reason: null,
    },
    {
      appliance_id: 'light_01',
      display_name: 'Main Room LED Light',
      appliance_type: 'led_bulb',
      service_class: 'necessity',
      is_necessity: true,
      supports_reduced: true,
      level: 1,
      power_15min_mean_w: 9.0,
      measured_w: null,
      gpio_state: 1,
      actuation_verified: true,
      remaining_service_hours: 4.0,
      shed_reason: null,
    },
    // FLEXIBLE / LUXURY LOADS
    {
      appliance_id: 'ac_01',
      display_name: 'Split Air Conditioner (1.5 Ton)',
      appliance_type: 'air_conditioner',
      service_class: 'thermostatic',
      is_necessity: false,
      supports_reduced: false,
      level: 1,
      power_15min_mean_w: 1114.6,
      measured_w: null,
      gpio_state: 1,
      actuation_verified: true,
      remaining_service_hours: 3.5,
      shed_reason: null,
    },
    {
      appliance_id: 'washer_01',
      display_name: 'Washing Machine (Front Load)',
      appliance_type: 'washing_machine',
      service_class: 'deferrable',
      is_necessity: false,
      supports_reduced: false,
      level: 1,
      power_15min_mean_w: 110.1,
      measured_w: null,
      gpio_state: 1,
      actuation_verified: true,
      remaining_service_hours: 1.25,
      shed_reason: null,
    },
    {
      appliance_id: 'tv_01',
      display_name: 'Living Room Smart TV',
      appliance_type: 'television',
      service_class: 'interruptible',
      is_necessity: false,
      supports_reduced: false,
      level: 1,
      power_15min_mean_w: 118.5,
      measured_w: null,
      gpio_state: 1,
      actuation_verified: true,
      remaining_service_hours: 2.0,
      shed_reason: null,
    },
    {
      appliance_id: 'pump_01',
      display_name: 'Overhead Water Pump (0.75 kW)',
      appliance_type: 'water_pump',
      service_class: 'interruptible',
      is_necessity: false,
      supports_reduced: false,
      level: 1,
      power_15min_mean_w: 750.0,
      measured_w: null,
      gpio_state: 1,
      actuation_verified: true,
      remaining_service_hours: 0.5,
      shed_reason: null,
    },
  ],
};

/**
 * Midday Peak Active State (14:05 IST, Severity 0.67)
 * Protected loads remain protected, luxury loads are shed or deferred.
 */
export const PEAK_ACTIVE_HOME_STATE: HomeState = {
  house_id: 'AP-GNT-H042',
  timestamp_ist: '2026-09-15T14:05:00+05:30',
  aggregate_power_kw: 0.145, // Dropped from ~1.4kW down to protected + background
  background_load_kw: 0.084,
  sanctioned_load_kw: 1.0,
  indoor_temperature_c: 31.4,
  outdoor_temperature_c: 36.8,
  occupancy_adult_home_fraction: 0.83,
  attention_available: true,
  marginal_tariff_inr_kwh: 4.50,
  month_to_date_kwh: 96.9,
  grid_peak_severity: 0.67, // Midday Peak active!
  operating_mode: 'grid_import',
  battery_state_of_charge: 0.78,
  grid_absent: false,
  data_age_seconds: 1,
  step_id: 56,
  appliances: [
    // PROTECTED NECESSITY LOADS (UNCHANGED, NEVER SHED)
    {
      appliance_id: 'fridge_01',
      display_name: 'Refrigerator (Inverter Double Door)',
      appliance_type: 'refrigerator',
      service_class: 'necessity',
      is_necessity: true,
      supports_reduced: false,
      level: 1,
      power_15min_mean_w: 47.7,
      measured_w: null,
      gpio_state: 1,
      actuation_verified: true,
      remaining_service_hours: 24.0,
      shed_reason: null,
    },
    {
      appliance_id: 'fan_01',
      display_name: 'Ceiling Fan (Master Bed)',
      appliance_type: 'ceiling_fan',
      service_class: 'necessity',
      is_necessity: true,
      supports_reduced: true,
      level: 1,
      power_15min_mean_w: 60.0,
      measured_w: null,
      gpio_state: 1,
      actuation_verified: true,
      remaining_service_hours: 7.0,
      shed_reason: null,
    },
    {
      appliance_id: 'light_01',
      display_name: 'Main Room LED Light',
      appliance_type: 'led_bulb',
      service_class: 'necessity',
      is_necessity: true,
      supports_reduced: true,
      level: 1,
      power_15min_mean_w: 9.0,
      measured_w: null,
      gpio_state: 1,
      actuation_verified: true,
      remaining_service_hours: 3.0,
      shed_reason: null,
    },
    // FLEXIBLE LOADS - SHED OR DEFERRED BY SHARP SAFETY POLICY
    {
      appliance_id: 'ac_01',
      display_name: 'Split Air Conditioner (1.5 Ton)',
      appliance_type: 'air_conditioner',
      service_class: 'thermostatic',
      is_necessity: false,
      supports_reduced: false,
      level: 0,
      power_15min_mean_w: 0.0,
      measured_w: null,
      gpio_state: 0,
      actuation_verified: true,
      remaining_service_hours: 3.5,
      shed_reason: 'GRID_PEAK',
    },
    {
      appliance_id: 'washer_01',
      display_name: 'Washing Machine (Front Load)',
      appliance_type: 'washing_machine',
      service_class: 'deferrable',
      is_necessity: false,
      supports_reduced: false,
      level: 0,
      power_15min_mean_w: 0.0,
      measured_w: null,
      gpio_state: 0,
      actuation_verified: true,
      remaining_service_hours: 1.25,
      shed_reason: 'PEAK_EVENT',
      deferred_until: '22:00 IST',
    },
    {
      appliance_id: 'tv_01',
      display_name: 'Living Room Smart TV',
      appliance_type: 'television',
      service_class: 'interruptible',
      is_necessity: false,
      supports_reduced: false,
      level: 0,
      power_15min_mean_w: 0.0,
      measured_w: null,
      gpio_state: 0,
      actuation_verified: true,
      remaining_service_hours: 2.0,
      shed_reason: 'GRID_PEAK',
    },
    {
      appliance_id: 'pump_01',
      display_name: 'Overhead Water Pump (0.75 kW)',
      appliance_type: 'water_pump',
      service_class: 'interruptible',
      is_necessity: false,
      supports_reduced: false,
      level: 0,
      power_15min_mean_w: 0.0,
      measured_w: null,
      gpio_state: 0,
      actuation_verified: true,
      remaining_service_hours: 0.5,
      shed_reason: 'GRID_PEAK',
    },
  ],
};

/**
 * 96-step (15-min intervals) simulated daily power curve for chart visualization
 */
export interface LoadProfilePoint {
  time: string;
  hour: number;
  baselineKw: number;
  sharpKw: number;
  gridPeakSeverity: number;
  sanctionedLimitKw: number;
  backgroundKw: number;
}

export function generateDailyLoadProfile(): LoadProfilePoint[] {
  const points: LoadProfilePoint[] = [];
  for (let i = 0; i < 96; i++) {
    const totalMinutes = i * 15;
    const hour = Math.floor(totalMinutes / 60);
    const minute = totalMinutes % 60;
    const timeStr = `${hour.toString().padStart(2, '0')}:${minute.toString().padStart(2, '0')}`;

    // Grid severity is high 08:00 - 17:00 (midday peak in AP dataset)
    let gridPeak = 0.0;
    if (hour >= 8 && hour < 17) {
      // bell curve peaked around 13:00 - 14:00
      const peakDist = Math.abs(totalMinutes - 13.5 * 60) / 180;
      gridPeak = Math.max(0.0, Number((0.68 - peakDist * 0.4).toFixed(2)));
    }

    // Baseline household load without control
    const isMorning = hour >= 6 && hour <= 9;
    const isMidday = hour >= 12 && hour <= 15;
    const isEvening = hour >= 18 && hour <= 22;

    let basePower = 0.12 + Math.sin((i / 96) * Math.PI * 2) * 0.04;
    if (isMorning) basePower += 0.22;
    if (isMidday) basePower += 0.38; // AC + appliances
    if (isEvening) basePower += 0.48; // Lighting, TV, cooking

    // SHARP active load reduction during midday grid peak
    let sharpPower = basePower;
    if (gridPeak > 0.4) {
      // SHARP sheds discretionary cooling and deferrable loads
      sharpPower = basePower - 0.28;
    }

    points.push({
      time: timeStr,
      hour,
      baselineKw: Number(Math.max(0.08, basePower).toFixed(3)),
      sharpKw: Number(Math.max(0.08, sharpPower).toFixed(3)),
      gridPeakSeverity: gridPeak,
      sanctionedLimitKw: 1.0,
      backgroundKw: 0.084,
    });
  }
  return points;
}
