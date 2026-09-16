/**
 * SHARP data contracts. FROZEN.
 *
 * The Pi, the dashboard and the rig must all agree on these names and values.
 * Changing a field here without changing the Pi breaks the system silently -
 * nothing throws, the UI just renders stale or wrong state.
 *
 * Generated from the shipped dataset and the validated model. Where this file
 * and any document disagree, this file wins.
 */

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------

/** 0 shed, 1 full power, 2 reduced. Level 2 is never selected in this build. */
export type ActionLevel = 0 | 1 | 2;

export const LEVEL_NAME: Record<ActionLevel, string> = {
  0: 'SHED',
  1: 'ON',
  2: 'REDUCED',
};

/**
 * The idea book's four classes.
 *   critical      never interrupted
 *   thermostatic  adjusted inside comfort bands
 *   deferrable    moved to another time
 *   interruptible paused briefly
 *
 * `critical` is a PERMISSION; the other three are DYNAMICS. A refrigerator is
 * both critical and thermostatic, which is not a contradiction.
 */
export type ServiceClass =
  | 'critical'
  | 'thermostatic'
  | 'deferrable'
  | 'interruptible';

export type OperatingMode = 'grid_import' | 'self_sufficient' | 'islanded_outage';

/** Closed set. Do not invent strings - the Pi will not send anything else. */
export type RejectedReason =
  | 'necessity_mask'
  | 'compressor_protection'
  | 'cycle_active'
  | 'min_on_steps'
  | 'min_off_steps'
  | 'command_expired'
  | 'level_not_supported'
  | 'peak_lockout'
  | 'watchdog_hold';

export const REJECTION_TEXT: Record<RejectedReason, string> = {
  necessity_mask: 'Essential load — cannot be shed',
  compressor_protection: 'Compressor needs 3 minutes before restarting',
  cycle_active: 'Mid-cycle — cannot be interrupted',
  min_on_steps: 'Minimum run time not yet reached',
  min_off_steps: 'Minimum off time not yet reached',
  command_expired: 'Command arrived too late',
  level_not_supported: 'This appliance has no such level',
  peak_lockout: 'Unavailable during grid peak',
  watchdog_hold: 'No valid command — holding last safe state',
};

// ---------------------------------------------------------------------------
// Appliances — the ten ids are frozen
// ---------------------------------------------------------------------------

export const APPLIANCE_IDS = [
  'ceiling_fan_01',
  'table_fan_01',
  'led_bulb_01',
  'led_tube_01',
  'refrigerator_01',
  'air_conditioner_01',
  'washing_machine_01',
  'ev_charger_01',
  'television_01',
  'mixer_grinder_01',
] as const;

export type ApplianceId = (typeof APPLIANCE_IDS)[number];

export interface ApplianceState {
  appliance_id: ApplianceId;
  /** Must match one of the dataset's 22 types, so the Pi finds the right flags. */
  appliance_type: string;
  service_class: ServiceClass;

  /** true -> level 0 is NEVER offered while the occupant wants it. */
  is_necessity: boolean;
  /** true -> the appliance can physically dim. Never true for a critical load. */
  supports_reduced: boolean;

  level: ActionLevel;
  /** The occupant is asking for it now. Protection is conditional on this. */
  occupant_wants: boolean;

  /** SIMULATED appliance wattage. Label it as simulated wherever it is shown. */
  power_15min_mean_w: number;
  /** null in this build - there is no meter fitted. Never put a simulated
   *  number here, or a stuck relay becomes invisible. */
  measured_w: number | null;
  /** Real GPIO pin readback. This one IS a measurement. */
  gpio_state: 0 | 1;
  /** gpio_state agrees with what was commanded. */
  actuation_verified: boolean;

  remaining_service_hours: number;
  /** Set while a grid peak forbids energising this circuit. */
  peak_locked_out: boolean;
}

// ---------------------------------------------------------------------------
// House state
// ---------------------------------------------------------------------------

export interface HomeState {
  house_id: string;
  timestamp_ist: string;

  aggregate_power_kw: number;
  /** 61 % of household load, and the agent cannot touch it. Render it as a
   *  distinct uncontrollable band or every result looks inexplicably small. */
  background_load_kw: number;
  sanctioned_load_kw: number;

  indoor_temperature_c: number;
  outdoor_temperature_c: number;

  occupancy_adult_home_fraction: number;
  attention_available: boolean;

  marginal_tariff_inr_kwh: number;
  month_to_date_kwh: number;

  /** 0..1. A declared grid event overrides the historical curve. */
  grid_peak_severity: number;
  peak_event_active: boolean;
  peak_event_expires_at: string | null;

  operating_mode: OperatingMode;
  battery_state_of_charge: number;
  grid_absent: boolean;

  appliances: ApplianceState[];
}

/** What the agent proposed, and what the shield allowed. */
export interface Intent {
  command_id: string;
  proposed: Partial<Record<ApplianceId, ActionLevel>>;
  executed: Partial<Record<ApplianceId, ActionLevel>>;
  shield_reasons: Partial<Record<ApplianceId, RejectedReason[]>>;
  policy_source: string;
  decision_latency_ms: number;
}

export interface OverrideRequest {
  override_id: string;
  house_id: string;
  appliance_id: ApplianceId;
  requested_level: ActionLevel;
  /** Time between the intent appearing and the tap. Feeds the preference
   *  weight in the reward model, so it is worth measuring properly. */
  client_latency_ms: number;
}

export interface OverrideResult {
  override_id: string;
  accepted: boolean;
  applied_level: ActionLevel | null;
  rejected_reason: RejectedReason | null;
}

// ---------------------------------------------------------------------------
// Connection — the UI must be able to tell live from stale
// ---------------------------------------------------------------------------

export type ConnectionStatus = 'connecting' | 'live' | 'stale' | 'offline';

export interface Feed {
  status: ConnectionStatus;
  /** Seconds since the last state message. Grey the UI past two intervals. */
  dataAgeSeconds: number;
  state: HomeState | null;
  intent: Intent | null;
}

// ---------------------------------------------------------------------------
// Derived helpers. Keep the rules here, not scattered through components.
// ---------------------------------------------------------------------------

export type DisplayGroup = 'PROTECTED' | 'RUNNING' | 'SHED' | 'DEFERRED';

/**
 * Which group an appliance belongs in.
 *
 * PROTECTED is the important one: it must be visibly unchanged when a peak
 * starts. That is the project's whole claim, rendered.
 */
export function displayGroup(a: ApplianceState): DisplayGroup {
  if (a.is_necessity && a.occupant_wants) return 'PROTECTED';
  if (a.service_class === 'deferrable' && a.level === 0) return 'DEFERRED';
  if (a.level === 0) return 'SHED';
  return 'RUNNING';
}

/**
 * Whether to show an off control at all.
 *
 * A critical appliance in use renders with NO off control - not greyed, absent.
 * The resident must never see a button that would cut their fan, because no
 * such action exists anywhere in the system.
 */
export function canShowOffControl(a: ApplianceState): boolean {
  return !(a.is_necessity && a.occupant_wants);
}

/**
 * Whether an override can even be attempted right now.
 *
 * A phone override needs the broker and the Pi reachable, and can fail
 * silently in a way a wall switch cannot. Disable the control and say why
 * rather than accepting a tap that will never arrive.
 */
export function overrideAvailable(
  a: ApplianceState,
  status: ConnectionStatus,
): { enabled: boolean; reason?: string } {
  if (status !== 'live') return { enabled: false, reason: 'No connection' };
  if (a.peak_locked_out) return { enabled: false, reason: 'Unavailable during grid peak' };
  if (!canShowOffControl(a)) return { enabled: false, reason: 'Essential load' };
  return { enabled: true };
}

/** Peak avoided, expressed the way a DISCOM and a citizen both understand. */
export function homesNotBlackedOut(peakCutPercent: number): string {
  return `equivalent to ${peakCutPercent.toFixed(1)} homes in 100 not blacked out`;
}
