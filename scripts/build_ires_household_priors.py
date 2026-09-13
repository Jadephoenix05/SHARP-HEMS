from pathlib import Path
import hashlib

import pandas as pd
import yaml


INPUT = Path("data/raw/ires/ires_2020_data.tab")
OUTPUT = Path(
    "data/interim/indian_priors/ires_household_priors_v1.parquet"
)
AP_OUTPUT = Path(
    "data/interim/indian_priors/ires_andhra_pradesh_priors_v1.parquet"
)
REPORT = Path("reports/ires_household_priors_validation_v1.csv")
SOURCES = Path("data_registry/sources.yaml")


# Source column -> SHARP column.
# Undecoded categories and unverified units remain explicitly marked.
columns = {
    "q103_survey_type": "urban_rural_code",
    "q213_no_members": "household_size",
    "q216_house_pucca_kachha": "house_construction_code",
    "q217_house_type": "dwelling_type_code",
    "q223_house_no_bedrooms": "bedroom_count",
    "q234_month_exp": "monthly_expenditure_inr",
    "q236_income_category": "income_category_code",

    "q301_grid_yn": "grid_supply_status_code",
    "q302_grid_hrs_no": "grid_supply_hours_daily",
    "q303_grid_hrs_even_no": "evening_supply_hours",
    "q304_grid_patternpowercut": "power_cut_pattern_code",
    "q305_grid_knownpattern_yn": "power_cut_known",
    "q306_grid_rural_powercut_days_20": "severe_power_cut_days_rural",
    "q307_grid_urban_powercut_days_10": "severe_power_cut_days_urban",
    "q308_grid_voltage_low_app": "low_voltage_days",
    "q309_grid_voltage_low_app_fail": "voltage_damage_days",
    "q310_volt_stab_yn": "voltage_stabilizer_available",
    "avg_monthly_bill": "average_monthly_bill_inr",
    "q316_invertor_battery_yn": "inverter_battery_available",
    "q317_genset_yn": "generator_available",
    "q319_shs_use_yn": "solar_home_system_available",
    "q319_b_shs_capacity_watts": "solar_home_system_capacity_w",
    "q324_prim_source_electricity": "primary_electricity_source_code",

    "q405_c_led_bulb_no": "led_bulb_count",
    "q405_d_led_tube_light_no": "led_tube_count",
    "q409_ceiling_fan_yn": "ceiling_fan_available",
    "q410_ceiling_fan_no": "ceiling_fan_count",
    "q411_celing_fan_use_months_no": "ceiling_fan_months_per_year",
    "q415_table_fan_yn": "table_fan_available",
    "q416_table_fan_no": "table_fan_count",
    "q421_air_coolers_yn": "air_cooler_available",
    "q422_air_coolers_no": "air_cooler_count",
    "q427_ac_yn": "air_conditioner_available",
    "q428_ac_no": "air_conditioner_count",
    "q431_ac_most_capacity_tons": "primary_ac_capacity_ton",
    "q436_ac_most_hrs_mar_jun": "ac_hours_daily_mar_jun",
    "q436_ac_most_hrs_jul_oct": "ac_hours_daily_jul_oct",
    "q436_ac_most_hrs_nov_feb": "ac_hours_daily_nov_feb",

    "q453_geyser_no": "geyser_count",
    "q459_tv_yn": "television_available",
    "q460_tv_no": "television_count",
    "q465_desktop_yn": "desktop_available",
    "q465_laptop_yn": "laptop_tablet_available",
    "q465_modem_yn": "modem_router_available",
    "q466_fridge_yn": "refrigerator_available",
    "q467_fridge_no": "refrigerator_count",
    "q471_elec_water_purifier_1_yn": "water_purifier_available",
    "q471_mixer_grinder_2_yn": "mixer_grinder_available",
    "q471_elec_kettle_3_yn": "electric_kettle_available",
    "q472_wash_mach_yn": "washing_machine_available",
    "q473_wash_mach_week_use_no": "washing_machine_uses_weekly",
    "q476_iron_yn": "electric_iron_available",
    "q477_water_pump_yn": "water_pump_available",
    "q480_water_pump_cap_hp": "water_pump_capacity_hp",
    "q481_water_pump_hrs": "water_pump_daily_hours",
    "q481_water_pump_min": "water_pump_daily_minutes_source",
    "q526_a_ecoil_yn": "electric_coil_available",
    "q526_b_induc_cookstove_yn": "induction_cooktop_available",
    "q526_c_oven_yn": "microwave_available",
    "q526_d_grill_toast_yn": "electric_grill_toaster_available",
    "q526_e_elec_rice_cooker_yn": "electric_rice_cooker_available",

    "q610_d_sanctioned_load": "sanctioned_load_source_value",
    "sw_dist": "district_survey_weight",
    "sw_state": "state_survey_weight",
    "asset_decile_1": "wealth_decile",
}


# Q301 is intentionally excluded: it has three valid response codes.
dummy_sources = [
    "q305_grid_knownpattern_yn",
    "q310_volt_stab_yn",
    "q316_invertor_battery_yn",
    "q317_genset_yn",
    "q319_shs_use_yn",
    "q409_ceiling_fan_yn",
    "q415_table_fan_yn",
    "q421_air_coolers_yn",
    "q427_ac_yn",
    "q459_tv_yn",
    "q465_desktop_yn",
    "q465_laptop_yn",
    "q465_modem_yn",
    "q466_fridge_yn",
    "q471_elec_water_purifier_1_yn",
    "q471_mixer_grinder_2_yn",
    "q471_elec_kettle_3_yn",
    "q472_wash_mach_yn",
    "q476_iron_yn",
    "q477_water_pump_yn",
    "q526_a_ecoil_yn",
    "q526_b_induc_cookstove_yn",
    "q526_c_oven_yn",
    "q526_d_grill_toast_yn",
    "q526_e_elec_rice_cooker_yn",
]


def main():
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    if not SOURCES.exists():
        raise FileNotFoundError(SOURCES)

    required = ["hhid", "s_name", "state_abbv"] + list(columns)

    raw = pd.read_csv(
        INPUT,
        sep="\t",
        usecols=required,
        dtype={"hhid": "string"},
        low_memory=False,
    )

    result = pd.DataFrame(index=raw.index)

    # Pseudonymous identifier, not a guarantee of anonymisation.
    # Missing source IDs remain missing and cause validation to fail.
    result["profile_id"] = raw["hhid"].map(
        lambda value: (
            pd.NA
            if pd.isna(value)
            else "ires_" + hashlib.sha256(
                f"IRES2020:{value}".encode("utf-8")
            ).hexdigest()[:16]
        )
    ).astype("string")

    result["state"] = raw["s_name"].astype("string").str.strip()
    result["state_abbreviation"] = (
        raw["state_abbv"].astype("string").str.strip()
    )

    unexpected_dummy_codes = 0
    unexpected_grid_codes = 0
    numeric_parse_failures = 0

    for source, target in columns.items():
        values = pd.to_numeric(raw[source], errors="coerce")

        numeric_parse_failures += int(
            (raw[source].notna() & values.isna()).sum()
        )

        if source == "q301_grid_yn":
            unexpected_grid_codes += int(
                (values.notna() & ~values.isin([0, 1, 2])).sum()
            )

            # Preserve original numeric status.
            result[target] = values

            result["grid_supply_status"] = values.map({
                0: "no_grid_supply",
                1: "grid_supply_available",
                2: "connection_without_electricity_yet",
            }).astype("string")

            result["grid_connected"] = values.map({
                0: 0,
                1: 1,
                2: 1,
            }).astype("Int8")

            result["grid_supply_available"] = values.map({
                0: 0,
                1: 1,
                2: 0,
            }).astype("Int8")

            result["grid_services_questions_applicable"] = values.map({
                0: 0,
                1: 1,
                2: 0,
            }).astype("Int8")

        elif source in dummy_sources:
            unexpected_dummy_codes += int(
                (values.notna() & ~values.isin([0, 1, 99])).sum()
            )

            # Retain source coding, including 99, for traceability.
            result[f"{target}_source_code"] = values

            # Unknown responses are not treated as "no".
            result[target] = values.where(
                values.isin([0, 1])
            ).astype("Int8")

        else:
            # Preserve values until variable-specific coding QA.
            # Do not globally replace 99: it can be a real numeric value.
            result[target] = values

    result["urban_rural"] = result["urban_rural_code"].map({
        1: "rural",
        2: "urban",
    }).astype("string")

    # Selected semantic corrections verified against the supplied questionnaire.
    # Raw numeric columns remain available alongside derived profile fields.
    semantic_metrics = {}
    for field, sentinel, allowed in [
        ("income_category_code", 88, range(1, 11)),
        ("power_cut_pattern_code", 99, range(1, 6)),
    ]:
        original = result[field].copy()
        result[f"{field}_original"] = original
        result[f"{field}_nonresponse"] = original.eq(sentinel)
        invalid = original.notna() & ~original.isin(list(allowed) + [sentinel])
        semantic_metrics[f"{field}_unexpected"] = int(invalid.sum())
        semantic_metrics[f"{field}_nonresponse_rows"] = int(original.eq(sentinel).sum())
        result[field] = original.where(original.isin(allowed))

    count_pairs = {
        "ceiling_fan_count": "ceiling_fan_available",
        "table_fan_count": "table_fan_available",
        "air_cooler_count": "air_cooler_available",
        "air_conditioner_count": "air_conditioner_available",
        "television_count": "television_available",
        "refrigerator_count": "refrigerator_available",
    }
    count_conflicts = 0
    for count_field, ownership_field in count_pairs.items():
        count = result[count_field]
        ownership = result[ownership_field]
        invalid = count.notna() & ((count < 0) | (count % 1 != 0))
        conflict = ((ownership.eq(0) & count.gt(0)) |
                    (ownership.eq(1) & count.eq(0))).fillna(False) | invalid
        count_conflicts += int(conflict.sum())
        fill_zero = ownership.eq(0).fillna(False) & count.isna()
        result[f"{count_field}_for_profile"] = count.mask(fill_zero, 0).mask(conflict)
        result[f"{count_field}_conflict"] = conflict
        semantic_metrics[f"{count_field}_conflicts"] = int(conflict.sum())
        semantic_metrics[f"{count_field}_structural_zeros"] = int(fill_zero.sum())
    semantic_metrics["count_conflicts"] = count_conflicts
    result["total_fan_count"] = result[
        ["ceiling_fan_count_for_profile", "table_fan_count_for_profile"]
    ].sum(axis=1, min_count=2)
    result["total_led_light_count"] = result[
        ["led_bulb_count", "led_tube_count"]
    ].sum(axis=1, min_count=2)
    semantic_metrics["usable_total_fan_count_rows"] = int(result["total_fan_count"].notna().sum())

    hours = result["water_pump_daily_hours"]
    minutes = result["water_pump_daily_minutes_source"]
    total = hours * 60 + minutes
    complete = hours.notna() & minutes.notna()
    invalid_components = ((hours.notna() & ~hours.between(0, 24)) |
                          (minutes.notna() & ~minutes.between(0, 59)))
    invalid_time = invalid_components | (complete & total.gt(1440))
    pump_conflict = (result["water_pump_available"].eq(0) & total.gt(0)).fillna(False)
    result["water_pump_daily_minutes"] = total.where(complete & ~invalid_time & ~pump_conflict)
    result["water_pump_time_conflict"] = invalid_time | pump_conflict
    semantic_metrics["pump_partial_time_rows"] = int((hours.notna() ^ minutes.notna()).sum())
    semantic_metrics["pump_time_conflicts"] = int(result["water_pump_time_conflict"].sum())

    load = result["sanctioned_load_source_value"]
    result["sanctioned_load_kw"] = load.where(load.gt(0) & load.ne(9999))
    semantic_metrics["sanctioned_load_nonpositive_rows"] = int((load.notna() & load.le(0)).sum())
    semantic_metrics["sanctioned_load_not_visible_rows"] = int(load.eq(9999).sum())

    for weight in ["state_survey_weight", "district_survey_weight"]:
        values = result[weight]
        semantic_metrics[f"{weight}_invalid_or_missing"] = int(
            (values.isna() | ~values.gt(0) | values.isin([float("inf"), -float("inf")])).sum()
        )
    # These are selected checks, not complete semantic or release approval.
    issues = (count_conflicts + semantic_metrics["pump_time_conflicts"] +
              sum(v for k, v in semantic_metrics.items()
                  if k.endswith("_unexpected") or k.endswith("_invalid_or_missing")))
    selected_semantic_status = "PASS" if issues == 0 else "REVIEW_REQUIRED"

    result["source_dataset"] = "IRES_2020"
    result["source_version"] = "Harvard_Dataverse_v1"
    result["survey_period"] = "2019-2020"
    result["is_synthetic"] = False

    ap_mask = result["state"].str.casefold().eq(
        "andhra pradesh"
    ).fillna(False)
    ap = result.loc[ap_mask].copy()

    unexpected_survey_type_codes = int(
        (
            result["urban_rural_code"].notna()
            & ~result["urban_rural_code"].isin([1, 2])
        ).sum()
    )

    grid_counts = result["grid_supply_status_code"].value_counts()

    structural_pass = (
        len(result) == len(raw)
        and len(result) > 0
        and raw["hhid"].notna().all()
        and not raw["hhid"].duplicated().any()
        and result["profile_id"].notna().all()
        and not result["profile_id"].duplicated().any()
        and result["state"].notna().all()
        and len(ap) > 0
        and unexpected_dummy_codes == 0
        and unexpected_grid_codes == 0
        and unexpected_survey_type_codes == 0
        and numeric_parse_failures == 0
    )

    status = "PASS" if structural_pass else "FAIL"

    checks = {
        "status": status,
        "validation_scope": "structural_codes_and_selected_semantic_corrections",
        "selected_semantic_checks": selected_semantic_status,
        "correction_version": "ires_semantic_patch_v1",
        "semantic_qa_status": "PENDING",
        "raw_rows": len(raw),
        "processed_rows": len(result),
        "unique_profile_ids": result["profile_id"].nunique(),
        "duplicate_profile_ids": int(
            result["profile_id"].duplicated().sum()
        ),
        "missing_profile_ids": int(
            result["profile_id"].isna().sum()
        ),
        "states_found": result["state"].nunique(),
        "andhra_pradesh_rows": len(ap),
        "unexpected_dummy_codes": unexpected_dummy_codes,
        "unexpected_grid_codes": unexpected_grid_codes,
        "unexpected_survey_type_codes": unexpected_survey_type_codes,
        "numeric_parse_failures": numeric_parse_failures,
        "grid_code_0_rows": int(grid_counts.get(0, 0)),
        "grid_code_1_rows": int(grid_counts.get(1, 0)),
        "grid_code_2_rows": int(grid_counts.get(2, 0)),
        "readme_documented_rows": 14850,
        "observed_unique_rows": raw["hhid"].nunique(),
        "documentation_row_difference": (
            raw["hhid"].nunique() - 14850
        ),
    }

    checks.update(semantic_metrics)

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"metric": key, "value": value}
         for key, value in checks.items()]
    ).to_csv(REPORT, index=False)

    with SOURCES.open("r", encoding="utf-8") as f:
        sources = yaml.safe_load(f)

    entry = sources["ires_2020"]

    if not structural_pass:
        entry["status"] = "VALIDATION_FAILED"

        with SOURCES.open("w", encoding="utf-8") as f:
            yaml.safe_dump(
                sources, f, sort_keys=False, allow_unicode=True
            )

        print("IRES HOUSEHOLD PRIORS BUILD")
        print("-" * 50)
        print("Status: FAIL")
        print("Report:", REPORT)
        print("Output Parquet files were NOT updated.")
        print("Any previous outputs must not be used.")
        raise SystemExit(1)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUTPUT, index=False)
    ap.to_parquet(AP_OUTPUT, index=False)

    entry["status"] = "COMPACT_TABLE_BUILT_PENDING_SEMANTIC_QA"
    entry["processed_path"] = OUTPUT.as_posix()
    entry["andhra_pradesh_processed_path"] = AP_OUTPUT.as_posix()
    entry["notes"] = [
        f"{len(result)} unique household records observed",
        "README states 14850 households",
        "Original household ID replaced by deterministic pseudonymous ID",
        "District village and enumerator fields excluded",
        "Q301: 0=no supply; 1=supply available; 2=connection without electricity yet",
        "Q302-Q316 skipped unless Q301 equals 1; missing values preserved",
        "Dummy source codes retained; 99 becomes missing in derived dummy fields",
        "Fan totals use explicit nonownership structural zeros; source counts retained",
        "LED totals still require both components; no inferred geyser absence",
        "Pump duration derived as hours*60+minutes only with valid complete components",
        "Q236=88 and Q304=99 masked in derived categories; originals retained",
        "Q610.d sanctioned load in kW; nonpositive and 9999 excluded in derived field",
        "State and district weights retained separately; never multiply them",
        "Q307 urban severe power cuts use questionnaire threshold over 12 hours",
        "Variable-specific missing codes units ranges and survey weights need semantic QA",
        "Structural PASS is not final simulator or release approval",
    ]

    with SOURCES.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            sources, f, sort_keys=False, allow_unicode=True
        )

    print("IRES HOUSEHOLD PRIORS BUILD")
    print("-" * 50)
    print("Status:", status)
    print("Semantic QA: PENDING (full review)")
    print("Selected semantic checks:", selected_semantic_status)
    print("Count conflicts:", count_conflicts)
    print("Pump time conflicts:", semantic_metrics["pump_time_conflicts"])
    print("National rows:", len(result))
    print("Andhra Pradesh rows:", len(ap))
    print("States:", result["state"].nunique())
    print("Unexpected dummy codes:", unexpected_dummy_codes)
    print("Unexpected grid codes:", unexpected_grid_codes)
    print("Connection without electricity:", int(grid_counts.get(2, 0)))
    print("Output:", OUTPUT)
    print("AP output:", AP_OUTPUT)
    print("Report:", REPORT)


if __name__ == "__main__":
    main()