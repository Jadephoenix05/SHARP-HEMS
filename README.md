# SHARP verified tariff component

This completes the energy, fixed and customer-charge calculation extracted from the official APCPDCL FY2025-26 order. It is one component of the master build, not a finished master dataset or a complete utility-bill calculator.

Extract this ZIP into C:\Users\SUPRIYA\SHARP_Master_Dataset. It adds a new script and config; it does not replace the existing tariff scenarios or data.

Run from the project root:

    python scripts\sharp_apcpdcl_tariff.py --validate

Example, 200 kWh and a 2 kW connected load:

    python scripts\sharp_apcpdcl_tariff.py --kwh 200 --connected-kw 2

Expected energy Rs867, customer Rs50, fixed Rs20; subtotal Rs937. Duty, FPPCA, true-up/down and other consumer-specific adjustments are outside this subtotal.

## Simulator integration contract

- Use Tariff.incremental_components(consumption_before, step_import_kwh).
- Keep cumulative billing-period consumption in the observation and checkpoint. Do not reset it to zero at every midnight or arbitrary daily episode boundary.
- Book the fixed charge and opening customer charge once per billing period. The incremental method includes changes in customer charge when a slab boundary is crossed.
- Use connected load from the applicable connection specification. Do not replace a missing value with zero or infer it from the appliance peak.
- No domestic ToD tariff is specified. Experimental TOU scenarios stay explicitly separate. Shifting identical imported kWh within the billing period does not create tariff savings; grid peak reduction is a separate objective.
- The effective period is 1 April 2025 to 31 March 2026. Applying these rates to another year's weather is a frozen-tariff simulation, not a reconstructed historical bill.
- These rates do not establish a complete utility bill. The order itself excludes duty, FPPCA, true-up/down and other recoveries. Their applicable inputs must be resolved separately for a full-bill claim.

## Evidence and validation

The official source PDFs and SHA256 hashes are included. Domestic energy and fixed charges: printed page150 (PDF164); customer charges: printed page169 (PDF183); exclusions and dates: printed page149 (PDF163). The domestic table was visually checked. Seven unit tests passed, including independently calculated boundary examples, fractional load conversion, customer-charge jumps, invalid inputs and telescoping reward accounting.

The source data and earlier generated outputs are preserved. This package does not publish, delete files, or set the master dataset to release-ready.
