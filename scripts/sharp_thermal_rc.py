"""SHARP simulator thermal component: lumped RC room with a thermostat.

Every parameter is a DECLARED ASSUMPTION from configs/thermal/sharp_thermal_rc_v1.json,
bounded by the observed RESIDE envelope. Nothing here is fitted to RESIDE and
nothing here establishes a causal cooling effect. See the config's
`why_not_fitted` block for why the fitted route was rejected.

Action semantics, per the SHARP idea book: the agent sets an AC SETPOINT, it
does not switch a compressor. Compressor duty is an OUTPUT of the thermostat
and is what the AC power model should consume:

    ac_energy_kwh = rated_power_kw * compressor_duty_fraction * 0.25

One 15-minute step:
    passive change = a * (T_out - T_in) + internal_gain,   a = 0.25h / tau
    with the AC enabled, the thermostat removes only as much heat as is needed
    to approach the setpoint, capped by full-compressor capacity.

Not approved for hardware control decisions.
"""
from pathlib import Path
import json
import math

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / 'configs/thermal/sharp_thermal_rc_v1.json'
STEP_MINUTES = 15
STEP_HOURS = STEP_MINUTES / 60

# Refuse to simulate outside any temperature a habitable room could plausibly hold.
ABSOLUTE_MIN_TEMPERATURE_C = 0.0
ABSOLUTE_MAX_TEMPERATURE_C = 60.0


class ThermalConfigError(ValueError):
    """Raised when the thermal configuration is unusable."""


def load_config(path=CONFIG_PATH, scenario='baseline'):
    """Load declared parameters, optionally overridden by a named scenario."""
    config = json.loads(Path(path).read_text(encoding='utf-8'))
    if config.get('step_minutes') != STEP_MINUTES:
        raise ThermalConfigError('Config step is not 15 minutes.')
    scenarios = config.get('sensitivity_scenarios', {})
    if scenario not in scenarios:
        raise ThermalConfigError(
            f'Unknown scenario {scenario!r}; available: {sorted(scenarios)}')

    params = {}
    for name, entry in config['parameters'].items():
        value = entry['value']
        low, high = entry['plausible_range']
        if isinstance(value, list):
            if not all(low <= v <= high for v in value):
                raise ThermalConfigError(f'{name} outside its declared range.')
        elif not low <= value <= high:
            raise ThermalConfigError(f'{name} outside its declared range.')
        params[name] = value

    for name, value in scenarios[scenario].items():
        if name not in params:
            raise ThermalConfigError(f'Scenario {scenario!r} sets unknown {name!r}.')
        low, high = config['parameters'][name]['plausible_range']
        if not low <= value <= high:
            raise ThermalConfigError(
                f'Scenario {scenario!r} puts {name} outside its declared range.')
        params[name] = value

    tau = params['envelope_time_constant_hours']
    if tau <= 0:
        raise ThermalConfigError('Time constant must be positive.')
    coupling = STEP_HOURS / tau
    if not 0 < coupling < 1:
        # A coupling at or above one makes the discrete step overshoot or oscillate.
        raise ThermalConfigError(
            f'Time constant {tau} h gives an unstable per-step coupling {coupling}.')
    params['envelope_coupling_per_step'] = coupling
    params['scenario'] = scenario
    params['config_id'] = config['config_id']
    return params


def _finite(name, value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f'{name} must be finite.')
    return value


def thermal_step(indoor_temperature_c, outdoor_temperature_c, params,
                 *, ac_enabled=False, setpoint_c=None):
    """Advance indoor temperature by exactly 15 minutes.

    ac_enabled says whether the AC is permitted to run this interval.
    setpoint_c is the commanded setpoint; the thermostat, not the agent,
    decides how much of the interval the compressor actually runs.
    """
    indoor = _finite('indoor_temperature_c', indoor_temperature_c)
    outdoor = _finite('outdoor_temperature_c', outdoor_temperature_c)
    for name, value in [('indoor_temperature_c', indoor),
                        ('outdoor_temperature_c', outdoor)]:
        if not ABSOLUTE_MIN_TEMPERATURE_C <= value <= ABSOLUTE_MAX_TEMPERATURE_C:
            raise ValueError(f'{name} {value} is outside the simulated range.')

    if ac_enabled:
        setpoint = _finite(
            'setpoint_c',
            params['default_setpoint_c'] if setpoint_c is None else setpoint_c)
        low, high = params['comfort_band_c']
        if not ABSOLUTE_MIN_TEMPERATURE_C <= setpoint <= ABSOLUTE_MAX_TEMPERATURE_C:
            raise ValueError(f'setpoint_c {setpoint} is outside the simulated range.')
    else:
        setpoint = None

    coupling = params['envelope_coupling_per_step']
    passive_change = coupling * (outdoor - indoor) + params['internal_gain_c_per_15min']
    passive_temperature = indoor + passive_change

    full_cooling = params['ac_full_cooling_c_per_15min']
    if not ac_enabled:
        cooling_applied = 0.0
    elif passive_temperature <= setpoint:
        # Already at or below setpoint; the thermostat calls for no cooling.
        cooling_applied = 0.0
    else:
        # Remove only what is needed to reach setpoint, capped by capacity.
        cooling_applied = min(full_cooling, passive_temperature - setpoint)

    next_temperature = passive_temperature - cooling_applied
    if not math.isfinite(next_temperature):
        raise ValueError('Nonfinite predicted temperature.')

    duty = cooling_applied / full_cooling if full_cooling > 0 else 0.0
    duty = min(1.0, max(0.0, duty))

    band_low, band_high = params['comfort_band_c']
    return {
        'next_temperature_c': next_temperature,
        'temperature_change_c': next_temperature - indoor,
        'passive_change_c': passive_change,
        'cooling_applied_c': cooling_applied,
        'compressor_duty_fraction': duty,
        'setpoint_c': setpoint,
        'ac_enabled': bool(ac_enabled),
        'below_comfort_band': next_temperature < band_low,
        'above_comfort_band': next_temperature > band_high,
        'interval_minutes': STEP_MINUTES,
        'scenario': params['scenario'],
        'config_id': params['config_id'],
        'parameter_basis': 'DECLARED_ENGINEERING_ASSUMPTIONS_NOT_FITTED',
        'causal_cooling_effect_established': False,
        'fitted_to_reside': False,
        'approved_for_hardware_control': False,
    }


def equilibrium_temperature_c(outdoor_temperature_c, params, *, ac_enabled=False):
    """Steady-state indoor temperature the model converges to, for sanity checks."""
    coupling = params['envelope_coupling_per_step']
    passive = outdoor_temperature_c + params['internal_gain_c_per_15min'] / coupling
    if not ac_enabled:
        return passive
    full_cooling = params['ac_full_cooling_c_per_15min']
    floor = (outdoor_temperature_c
             + (params['internal_gain_c_per_15min'] - full_cooling) / coupling)
    return max(params['default_setpoint_c'], floor)
