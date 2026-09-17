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

export type OperatingMode =
  | 'grid_import'
  | 'self_sufficient'
  | 'islanded_outage';

export type ServiceClass =
  | 'necessity'
  | 'thermostatic'
  | 'deferrable'
  | 'interruptible';

export interface ApplianceState {
  appliance_id: string;
  appliance_type: string;
  service_class: ServiceClass;
  is_necessity: boolean;
  supports_reduced: boolean;
  level: ActionLevel;
  power_15min_mean_w: number;
  measured_w: number | null;
  gpio_state: 0 | 1;
  actuation_verified: boolean;
  remaining_service_hours: number;
  display_name?: string;
  shed_reason?: string | null;
  deferred_until?: string | null;
}

export interface HomeState {
  house_id: string;
  timestamp_ist: string;
  aggregate_power_kw: number;
  background_load_kw: number;
  sanctioned_load_kw: number;
  indoor_temperature_c: number;
  outdoor_temperature_c: number;
  occupancy_adult_home_fraction: number;
  attention_available: boolean;
  marginal_tariff_inr_kwh: number;
  month_to_date_kwh: number;
  grid_peak_severity: number;
  operating_mode: OperatingMode;
  battery_state_of_charge: number;
  grid_absent: boolean;
  appliances: ApplianceState[];
  data_age_seconds?: number;
  step_id?: number;
}

export interface PeakEvent {
  event_id: string;
  sequence: number;
  region: string;
  severity: number;
  declared_at: string;
  expires_at: string;
  reason: string;
  is_active: boolean;
  homes_responding?: number;
  mw_relieved?: number;
}

export interface Intent {
  command_id: string;
  proposed: Record<string, ActionLevel>;
  executed: Record<string, ActionLevel>;
  shield_reasons: Record<string, string[]>;
  policy_source: string;
  decision_latency_ms: number;
}

export interface CommandAck {
  command_id: string;
  appliance_id: string;
  accepted: boolean;
  applied_level: ActionLevel;
  rejected_reason: string | null;
  gpio_state: 0 | 1;
  measured_w: number | null;
  verification:
    | 'MATCH'
    | 'MISMATCH_STILL_DRAWING'
    | 'MISMATCH_NOT_DRAWING'
    | 'MISMATCH_WRONG_LEVEL'
    | 'NO_METER';
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

/**
 * MQTT feed status used by the browser dashboard.
 *
 * This is the connection/data layer status and is separate from
 * SystemConnectionStatus, which is the dashboard-facing status contract.
 */
export interface Feed {
  status: 'live' | 'stale' | 'connecting' | 'offline';
  dataAgeSeconds: number;
  state: HomeState | null;
  intent: Intent | null;
}