/**
 * SHARP Data Contracts (Frozen v2)
 *
 * Source of truth: docs/DASHBOARD_SPECIFICATION.md §5, docs/INTEGRATION_GUIDE.md §1.5,
 * and docs/GRID_CONTROLLER_AND_TARIFF.md.
 *
 * Action levels:
 *  0 = OFF       (shed luxury loads only)
 *  1 = ON        (full power)
 *  2 = REDUCED   (reserved/dimmed, not offered in current binary build)
 */

export type ActionLevel = 0 | 1 | 2;

export type OperatingMode = 'grid_import' | 'self_sufficient' | 'islanded_outage';

export type ServiceClass = 'necessity' | 'thermostatic' | 'deferrable' | 'interruptible';

export interface ApplianceState {
  appliance_id: string;            // e.g. 'fridge_01', 'fan_01', 'ac_01'
  appliance_type: string;          // e.g. 'refrigerator', 'ceiling_fan', 'air_conditioner'
  service_class: ServiceClass;
  is_necessity: boolean;           // true -> level 0 is NEVER offered / rendered
  supports_reduced: boolean;       // true -> dim/eco capable
  level: ActionLevel;              // 0: SHED, 1: ON, 2: REDUCED
  power_15min_mean_w: number;      // SIMULATED appliance wattage (must be labelled simulated)
  measured_w: number | null;       // null in this build - no meter fitted. NEVER fake this!
  gpio_state: 0 | 1;               // real pin readback - true measurement
  actuation_verified: boolean;     // gpio_state agrees with commanded action
  remaining_service_hours: number;
  display_name?: string;
  shed_reason?: string | null;     // reason if level === 0 (e.g. 'GRID_PEAK', 'CAPACITY_SHED')
  deferred_until?: string | null;  // e.g. '22:00' for washing machine
}

export interface HomeState {
  house_id: string;
  timestamp_ist: string;
  aggregate_power_kw: number;
  background_load_kw: number;      // unmodelled, NOT controllable (~61% of household load)
  sanctioned_load_kw: number;
  indoor_temperature_c: number;
  outdoor_temperature_c: number;
  occupancy_adult_home_fraction: number;
  attention_available: boolean;
  marginal_tariff_inr_kwh: number;
  month_to_date_kwh: number;
  grid_peak_severity: number;      // 0..1, drives peak badge & grid response
  operating_mode: OperatingMode;
  battery_state_of_charge: number; // 0..1 fraction
  grid_absent: boolean;            // true in outage mode
  appliances: ApplianceState[];
  data_age_seconds?: number;
  step_id?: number;
}

export interface PeakEvent {
  event_id: string;
  sequence: number;
  region: string;
  severity: number;                // 0.00 .. 1.00
  declared_at: string;             // ISO-8601 IST string
  expires_at: string;              // ISO-8601 IST string
  reason: string;                  // 'system_peak' | 'transmission_congestion' | 'manual_test'
  is_active: boolean;
  homes_responding?: number;
  mw_relieved?: number;
}

export interface Intent {
  command_id: string;
  proposed: Record<string, ActionLevel>;
  executed: Record<string, ActionLevel>;
  shield_reasons: Record<string, string[]>;
  policy_source: string;          // e.g. 'bdq_v2_cql'
  decision_latency_ms: number;
}

export interface CommandAck {
  command_id: string;
  appliance_id: string;
  accepted: boolean;
  applied_level: ActionLevel;
  rejected_reason: string | null; // e.g. 'NECESSITY_MASK', 'PEAK_LOCKOUT'
  gpio_state: 0 | 1;
  measured_w: number | null;
  verification: 'MATCH' | 'MISMATCH_STILL_DRAWING' | 'MISMATCH_NOT_DRAWING' | 'MISMATCH_WRONG_LEVEL' | 'NO_METER';
  acked_at: string;
  latency_ms: number;
}

export interface OverrideRequest {
  override_id: string;
  appliance_id: string;
  requested_level: ActionLevel;
  issued_at: string;
  user_id: string;
  client_latency_ms: number;
}

export interface SystemConnectionStatus {
  status: 'connected' | 'disconnected' | 'reconnecting';
  data_age_seconds: number;
  broker_endpoint: string;
  last_verified_at: string;
  is_fallback: boolean;
}
