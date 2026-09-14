"""Grid outages, inverter battery, solar and operating mode for SHARP.

This module supplies the state the idea book calls source-awareness and the
third operating mode, and the sink-aware shedding value that the project treats
as its distinguishing contribution.

WHAT IS SOURCE-DERIVED (IRES 2020, Andhra Pradesh, 498 households):
  * grid_supply_hours_daily     outage DURATION, reported, 358 of 498 households
                                experience under 24 h supply, minimum 12 h
  * evening_supply_hours        how much of the evening actually has supply
  * inverter_battery_available  35 of 498 households own an inverter battery
  * solar_home_system_available 3 of 498, at 20-25 W

WHAT IS A DECLARED ASSUMPTION:
  * WHEN within the day an outage falls. IRES reports a power-cut pattern code
    (q304_grid_patternpowercut) but its 1-5 value labels are not decoded in the
    published codebook, so timing is assigned deterministically and flagged
    rather than inferred from a code whose meaning is unknown. Evening outage
    length is still pinned by the reported evening supply hours.
  * Inverter battery capacity and efficiency.
  * Rooftop PV. IRES shows essentially no rooftop solar in this population: 3
    households at 20-25 W are solar lanterns, not rooftop systems. Any rooftop
    PV is therefore a clearly labelled SCENARIO overlay, never presented as an
    Andhra Pradesh ownership rate.

SINK-AWARE SHEDDING. Shedding a load running on own solar only has value if the
freed generation goes somewhere useful. Per the project spec:

    V_shed = P_shed * [ (1 - rho) * T_import + rho * s_t ]

    s_t = T_import * round_trip_efficiency   if the battery has headroom
        = T_export                           if full and export is allowed
        = 0                                  if full and export is blocked

When s_t is zero the shed action is worthless, so the SHIELD masks it rather
than applying a soft penalty. That is the spec's explicit instruction.
"""
from dataclasses import dataclass
import hashlib
import math

STEPS_PER_DAY = 96
STEP_HOURS = 0.25
EVENING_START_STEP = 72   # 18:00 IST
EVENING_END_STEP = 96     # 24:00 IST

# Declared assumptions. None of these is a measured value.
INVERTER_USABLE_KWH = 0.9          # typical 12 V / 150 Ah unit at ~50% depth
INVERTER_MAX_DISCHARGE_KW = 0.6
INVERTER_MAX_CHARGE_KW = 0.3
ROUND_TRIP_EFFICIENCY = 0.85
SELF_SUFFICIENT_RHO = 0.5          # at or above this the home counts as self-sufficient
PV_PERFORMANCE_RATIO = 0.75        # module, inverter and soiling losses combined

# Which appliances an Indian inverter circuit actually carries. In practice the
# inverter is wired to fans, lights and the router: low-power loads that make a
# blackout liveable and that a ~0.9 kWh battery can sustain for hours. High-power
# and surge loads are not on that circuit, so during an outage they are simply
# UNPOWERED - not "shed" by the agent, but physically dead.
#
# This matters for the override model too. The idea book is explicit: "During an
# outage, users want the luxury loads off so the fans and lights last until the
# grid returns." Preference inverts in islanded mode, and a policy trained with
# the air conditioner running on battery would learn the opposite.
#
# Declared assumption about typical wiring, not a measured circuit survey.
INVERTER_CIRCUIT_APPLIANCES = frozenset({
    'ceiling_fan', 'table_fan',
    'led_bulb', 'led_tube', 'cfl_bulb', 'cfl_tube', 'incandescent_bulb',
    'modem_router',
})


def on_inverter_circuit(appliance_type):
    """Whether this appliance can run at all while the grid is absent."""
    return str(appliance_type) in INVERTER_CIRCUIT_APPLIANCES


MODE_GRID_IMPORT = 0
MODE_SELF_SUFFICIENT = 1
MODE_ISLANDED = 2
MODE_NAMES = {MODE_GRID_IMPORT: 'grid_import',
              MODE_SELF_SUFFICIENT: 'self_sufficient',
              MODE_ISLANDED: 'islanded_outage'}


def check(condition, message):
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class HouseholdPower:
    template_id: str
    supply_hours_daily: float
    evening_supply_hours: float
    has_inverter: bool
    battery_capacity_kwh: float
    pv_capacity_kw: float
    pv_is_scenario_overlay: bool
    export_allowed: bool


def build_household_power(row, *, pv_scenario_kw=0.0, export_allowed=False):
    """Read the source-reported fields and attach declared assumptions."""
    supply = float(row['grid_supply_hours_daily'])
    check(math.isfinite(supply) and 0 <= supply <= 24,
          f'Reported supply hours out of range: {supply}')
    evening = float(row['evening_supply_hours'])
    check(math.isfinite(evening) and 0 <= evening <= 6,
          f'Reported evening supply hours out of range: {evening}')
    has_inverter = bool(row['inverter_battery_available'])
    reported_pv_w = row.get('solar_home_system_capacity_w')
    reported_pv_kw = (float(reported_pv_w) / 1000.0
                      if reported_pv_w is not None and not _isnan(reported_pv_w) else 0.0)
    pv_kw = reported_pv_kw + float(pv_scenario_kw)
    return HouseholdPower(
        template_id=str(row['template_id']), supply_hours_daily=supply,
        evening_supply_hours=evening, has_inverter=has_inverter,
        battery_capacity_kwh=INVERTER_USABLE_KWH if has_inverter else 0.0,
        pv_capacity_kw=pv_kw, pv_is_scenario_overlay=bool(pv_scenario_kw > 0),
        export_allowed=bool(export_allowed))


def _isnan(value):
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return True


def outage_schedule(household, date_key):
    """96 booleans: True where the grid is ABSENT.

    Duration is source-reported. Placement is a declared assumption, except in
    the evening, where the reported evening supply hours pin how many evening
    steps may be out.
    """
    total_out_steps = int(round((24.0 - household.supply_hours_daily) / STEP_HOURS))
    total_out_steps = max(0, min(STEPS_PER_DAY, total_out_steps))
    evening_out_steps = int(round((6.0 - household.evening_supply_hours) / STEP_HOURS))
    evening_out_steps = max(0, min(EVENING_END_STEP - EVENING_START_STEP,
                                   evening_out_steps, total_out_steps))
    schedule = [False] * STEPS_PER_DAY
    if total_out_steps == 0:
        return schedule

    seed = int(hashlib.sha256(
        f'{household.template_id}|{date_key}'.encode()).hexdigest()[:8], 16)

    # Evening outage sits as one contiguous block inside the evening window.
    if evening_out_steps:
        span = EVENING_END_STEP - EVENING_START_STEP - evening_out_steps
        start = EVENING_START_STEP + (seed % (span + 1) if span > 0 else 0)
        for step in range(start, start + evening_out_steps):
            schedule[step] = True

    # The remainder sits as one contiguous daytime block, never overlapping it.
    remaining = total_out_steps - evening_out_steps
    if remaining > 0:
        span = EVENING_START_STEP - remaining
        start = (seed // 97) % (span + 1) if span > 0 else 0
        for step in range(start, min(start + remaining, EVENING_START_STEP)):
            schedule[step] = True
    return schedule


def pv_generation_kw(household, irradiance_w_m2):
    """Instantaneous PV output from measured NASA POWER surface irradiance."""
    if household.pv_capacity_kw <= 0:
        return 0.0
    value = float(irradiance_w_m2)
    if not math.isfinite(value) or value < 0:
        raise ValueError('Irradiance must be finite and nonnegative')
    # Rated at 1000 W/m2 standard test conditions.
    return household.pv_capacity_kw * min(1.0, value / 1000.0) * PV_PERFORMANCE_RATIO


def dispatch(*, demand_kw, pv_kw, battery_kwh, household, grid_absent):
    """Resolve one 15-minute interval into grid, PV and battery flows."""
    for name, value in [('demand_kw', demand_kw), ('pv_kw', pv_kw),
                        ('battery_kwh', battery_kwh)]:
        check(math.isfinite(float(value)) and float(value) >= 0,
              f'{name} must be finite and nonnegative')
    demand = float(demand_kw)
    pv = float(pv_kw)
    capacity = household.battery_capacity_kwh
    stored = min(float(battery_kwh), capacity)

    direct_pv = min(pv, demand)
    surplus_pv = pv - direct_pv
    unmet = demand - direct_pv

    # Battery discharges to cover what PV cannot, and must cover everything
    # during an outage because the grid is simply not there.
    max_discharge = min(INVERTER_MAX_DISCHARGE_KW * STEP_HOURS, stored)
    discharge_kwh = min(max_discharge, unmet * STEP_HOURS) if capacity > 0 else 0.0
    discharge_kw = discharge_kwh / STEP_HOURS
    unmet -= discharge_kw
    stored -= discharge_kwh

    headroom_kwh = capacity - stored
    charge_kwh = 0.0
    grid_charge_kwh = 0.0
    if capacity > 0 and surplus_pv > 0:
        charge_kwh = min(INVERTER_MAX_CHARGE_KW * STEP_HOURS, headroom_kwh,
                         surplus_pv * STEP_HOURS * ROUND_TRIP_EFFICIENCY)
        stored += charge_kwh
        surplus_pv -= charge_kwh / STEP_HOURS / ROUND_TRIP_EFFICIENCY
        headroom_kwh = capacity - stored
    # An inverter battery charges from the MAINS, which is the whole point: it
    # refills while the grid is up so it can carry the house through the next
    # cut. Charging only from solar surplus left every battery flat, because
    # these households have no solar. The charge draws real grid energy, so it
    # is added to demand and billed.
    # Only after solar has had its turn, and only up to the charger's remaining
    # rate. A home with surplus PV must never import to charge a battery that
    # its own generation could fill.
    if (capacity > 0 and not grid_absent and headroom_kwh > 1e-9
            and surplus_pv <= 1e-9):
        grid_charge_kwh = min(INVERTER_MAX_CHARGE_KW * STEP_HOURS - charge_kwh,
                              headroom_kwh)
        grid_charge_kwh = max(0.0, grid_charge_kwh)
        stored += grid_charge_kwh
        headroom_kwh = capacity - stored

    if grid_absent:
        # Islanded: nothing can be imported. Any residual demand is unserved.
        grid_kw = 0.0
        unserved_kw = max(0.0, unmet)
        export_kw = 0.0
    else:
        # Battery charging is real grid draw on top of appliance demand.
        grid_kw = max(0.0, unmet) + grid_charge_kwh / STEP_HOURS / ROUND_TRIP_EFFICIENCY
        unserved_kw = 0.0
        export_kw = surplus_pv if household.export_allowed else 0.0

    served = demand - unserved_kw
    rho = ((direct_pv + discharge_kw) / served) if served > 1e-9 else 0.0
    rho = min(1.0, max(0.0, rho))

    if grid_absent:
        mode = MODE_ISLANDED
    elif rho >= SELF_SUFFICIENT_RHO:
        mode = MODE_SELF_SUFFICIENT
    else:
        mode = MODE_GRID_IMPORT

    return {
        'grid_import_kw': grid_kw, 'pv_direct_kw': direct_pv,
        'battery_discharge_kw': discharge_kw,
        'battery_charge_kwh': charge_kwh, 'battery_grid_charge_kwh': grid_charge_kwh,
        'battery_kwh': stored,
        'battery_headroom_kwh': headroom_kwh,
        'export_kw': export_kw, 'unserved_kw': unserved_kw,
        'self_sufficient_fraction': rho, 'operating_mode': mode,
        'operating_mode_name': MODE_NAMES[mode], 'grid_absent': bool(grid_absent),
    }


def shed_value_per_kw(*, flows, import_rate_inr_kwh, export_rate_inr_kwh=0.0):
    """Value of shedding one kW for this interval, and whether it is worthless.

    Implements V_shed = (1 - rho) * T_import + rho * s_t from the project spec.
    """
    rho = float(flows['self_sufficient_fraction'])
    import_rate = float(import_rate_inr_kwh)
    check(math.isfinite(import_rate) and import_rate >= 0, 'Bad import rate')
    if flows['battery_headroom_kwh'] > 1e-9:
        sink = import_rate * ROUND_TRIP_EFFICIENCY
        sink_reason = 'BATTERY_HEADROOM_STORES_IT'
    elif flows['export_kw'] > 0 or export_rate_inr_kwh > 0:
        sink = float(export_rate_inr_kwh)
        sink_reason = 'EXPORTED_AT_EXPORT_RATE'
    else:
        sink = 0.0
        sink_reason = 'BATTERY_FULL_AND_EXPORT_BLOCKED_SO_CURTAILED'
    value = (1.0 - rho) * import_rate + rho * sink
    return {
        'shed_value_inr_per_kwh': value, 'sink_value_inr_per_kwh': sink,
        'sink_reason': sink_reason,
        # A shed that saves nothing must be removed by the shield, not merely
        # discouraged by a penalty.
        'shed_is_worthless': bool(value <= 1e-12),
    }


def self_test():
    row = {'template_id': 'h1', 'grid_supply_hours_daily': 20.0,
           'evening_supply_hours': 4.0, 'inverter_battery_available': 1,
           'solar_home_system_capacity_w': None}
    household = build_household_power(row, pv_scenario_kw=1.0)
    assert household.battery_capacity_kwh == INVERTER_USABLE_KWH
    assert household.pv_is_scenario_overlay

    schedule = outage_schedule(household, '2024-03-01')
    assert sum(schedule) == int((24 - 20) / STEP_HOURS) == 16
    evening_out = sum(schedule[EVENING_START_STEP:EVENING_END_STEP])
    assert evening_out == int((6 - 4) / STEP_HOURS) == 8, evening_out
    assert outage_schedule(household, '2024-03-01') == schedule, 'Not deterministic'

    full_supply = build_household_power(
        {**row, 'grid_supply_hours_daily': 24.0, 'evening_supply_hours': 6.0})
    assert not any(outage_schedule(full_supply, '2024-03-01'))

    assert pv_generation_kw(household, 1000.0) == 1.0 * PV_PERFORMANCE_RATIO
    assert pv_generation_kw(household, 0.0) == 0.0

    # Grid present, no PV: appliance demand imports, and an empty battery also
    # refills from the mains.
    flows = dispatch(demand_kw=1.0, pv_kw=0.0, battery_kwh=0.0,
                     household=household, grid_absent=False)
    assert flows['grid_import_kw'] > 1.0, flows
    assert flows['battery_grid_charge_kwh'] > 0, 'Mains must recharge the inverter'
    assert flows['operating_mode'] == MODE_GRID_IMPORT
    # A full battery with nothing to serve draws no charging current. Demand is
    # zero here on purpose: with load present the battery discharges first and
    # is no longer full, so it would legitimately recharge.
    full_flows = dispatch(demand_kw=0.0, pv_kw=0.0,
                          household=household, grid_absent=False,
                          battery_kwh=household.battery_capacity_kwh)
    assert full_flows['battery_grid_charge_kwh'] == 0.0, full_flows
    assert full_flows['grid_import_kw'] == 0.0
    # Nothing charges from a grid that is not there.
    assert dispatch(demand_kw=0.0, pv_kw=0.0, battery_kwh=0.0,
                    household=household, grid_absent=True)['battery_grid_charge_kwh'] == 0.0

    # Islanded with an empty battery: demand goes unserved, never imported.
    flows = dispatch(demand_kw=1.0, pv_kw=0.0, battery_kwh=0.0,
                     household=household, grid_absent=True)
    assert flows['grid_import_kw'] == 0.0 and flows['unserved_kw'] == 1.0
    assert flows['operating_mode'] == MODE_ISLANDED

    # Plenty of PV: the home is self-sufficient and imports nothing.
    flows = dispatch(demand_kw=0.5, pv_kw=2.0, battery_kwh=0.0,
                     household=household, grid_absent=False)
    assert flows['grid_import_kw'] == 0.0
    assert flows['self_sufficient_fraction'] == 1.0
    assert flows['operating_mode'] == MODE_SELF_SUFFICIENT
    assert flows['battery_charge_kwh'] > 0, 'Surplus should charge a battery with headroom'

    # Sink value: headroom stores it, so shedding still has value.
    value = shed_value_per_kw(flows=flows, import_rate_inr_kwh=8.75)
    assert value['sink_reason'] == 'BATTERY_HEADROOM_STORES_IT'
    assert not value['shed_is_worthless']

    # Battery full and export blocked: shedding is worthless and must be masked.
    full = dict(flows, battery_headroom_kwh=0.0, export_kw=0.0,
                self_sufficient_fraction=1.0)
    value = shed_value_per_kw(flows=full, import_rate_inr_kwh=8.75)
    assert value['shed_is_worthless'], value
    assert value['sink_reason'] == 'BATTERY_FULL_AND_EXPORT_BLOCKED_SO_CURTAILED'

    # Partly self-sufficient with a blocked sink: value falls but is not zero.
    partial = dict(full, self_sufficient_fraction=0.4)
    value = shed_value_per_kw(flows=partial, import_rate_inr_kwh=10.0)
    assert abs(value['shed_value_inr_per_kwh'] - 6.0) < 1e-9, value
    # The inverter circuit carries what makes a blackout liveable, nothing more.
    assert on_inverter_circuit('ceiling_fan') and on_inverter_circuit('led_bulb')
    assert on_inverter_circuit('modem_router')
    for luxury in ['air_conditioner', 'geyser', 'water_pump', 'washing_machine',
                   'electric_iron', 'television', 'refrigerator']:
        assert not on_inverter_circuit(luxury), luxury
    print('SHARP POWER SYSTEM SELF-TEST PASSED')
    print('  outage duration is source-reported; placement is a declared assumption')
    print('  rooftop PV is a scenario overlay, not an Andhra Pradesh ownership rate')


if __name__ == '__main__':
    self_test()
