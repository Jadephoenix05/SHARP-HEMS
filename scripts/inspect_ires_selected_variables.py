import pandas as pd
from pathlib import Path

path = Path(r"data\raw\ires\ires_2020_variable_dictionary.xlsx")

selected = [
    "hhid",
    "s_name",
    "state_abbv",
    "q103_survey_type",
    "q213_no_members",
    "q216_house_pucca_kachha",
    "q217_house_type",
    "q223_house_no_bedrooms",
    "q234_month_exp",
    "q236_income_category",

    "q301_grid_yn",
    "q302_grid_hrs_no",
    "q303_grid_hrs_even_no",
    "q304_grid_patternpowercut",
    "q305_grid_knownpattern_yn",
    "q306_grid_rural_powercut_days_20",
    "q307_grid_urban_powercut_days_10",
    "q308_grid_voltage_low_app",
    "q309_grid_voltage_low_app_fail",
    "q310_volt_stab_yn",
    "avg_monthly_bill",
    "q316_invertor_battery_yn",
    "q317_genset_yn",
    "q319_shs_use_yn",
    "q319_b_shs_capacity_watts",
    "q324_prim_source_electricity",

    "q405_c_led_bulb_no",
    "q405_d_led_tube_light_no",
    "q409_ceiling_fan_yn",
    "q410_ceiling_fan_no",
    "q411_celing_fan_use_months_no",
    "q415_table_fan_yn",
    "q416_table_fan_no",
    "q421_air_coolers_yn",
    "q422_air_coolers_no",
    "q427_ac_yn",
    "q428_ac_no",
    "q431_ac_most_capacity_tons",
    "q436_ac_most_hrs_mar_jun",
    "q436_ac_most_hrs_jul_oct",
    "q436_ac_most_hrs_nov_feb",

    "q453_geyser_no",
    "q459_tv_yn",
    "q460_tv_no",
    "q465_desktop_yn",
    "q465_laptop_yn",
    "q465_modem_yn",
    "q466_fridge_yn",
    "q467_fridge_no",
    "q471_elec_water_purifier_1_yn",
    "q471_mixer_grinder_2_yn",
    "q471_elec_kettle_3_yn",
    "q472_wash_mach_yn",
    "q473_wash_mach_week_use_no",
    "q476_iron_yn",
    "q477_water_pump_yn",
    "q480_water_pump_cap_hp",
    "q481_water_pump_hrs",
    "q481_water_pump_min",
    "q526_a_ecoil_yn",
    "q526_b_induc_cookstove_yn",
    "q526_c_oven_yn",
    "q526_d_grill_toast_yn",
    "q526_e_elec_rice_cooker_yn",

    "q610_d_sanctioned_load",
    "sw_dist",
    "sw_state",
    "asset_decile_1",
    "asset_decile_2",
]

labels = pd.read_excel(path, sheet_name="Variable names & labels ")
labels.columns = ["variable", "label"]

result = labels[labels["variable"].isin(selected)].copy()
result["requested_order"] = result["variable"].map(
    {name: i for i, name in enumerate(selected)}
)
result = result.sort_values("requested_order").drop(
    columns="requested_order"
)

missing = sorted(set(selected) - set(result["variable"]))

output = Path(r"reports\ires_selected_variable_labels.csv")
output.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(output, index=False)

print("IRES SELECTED VARIABLE LABELS")
print("-" * 80)
print("Requested variables:", len(selected))
print("Labels found:", len(result))
print("Missing:", missing)
print()
print(result.to_string(index=False))
print()
print("Saved:", output)