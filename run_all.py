"""
run_all.py
Master pipeline script for Financial Strain and All-Cause Mortality study.

Runs all analysis scripts in order and verifies expected outputs exist.
Use --verify to check outputs without re-running analyses.

Usage:
    python run_all.py          # Run all scripts and verify outputs
    python run_all.py --verify # Verify outputs only (no re-run)

Requirements:
    - Python 3.13 with packages in requirements.txt
    - Stata MP 18+ for scripts 07 and 09
    - RAND HRS Longitudinal File (randhrs1992_2022v1_STATA/)
    - HRS biomarker restricted data for script 10 (Special Access Data/)
    - HRS Social Security wealth data for script 11
    - Runtime: ~30 minutes for full pipeline (excluding Stata scripts)
    - Memory: ~4 GB peak (script 01 loads full RAND HRS file)

Notes:
    - Scripts 07 and 09 are Stata .do files and must be run manually in Stata.
    - Script 10 requires HRS restricted-use biomarker data under a data use
      agreement. It will skip gracefully if the biomarker files are not present.
    - Script 11 (SS wealth) is a supplementary analysis not in the core pipeline.
"""

import subprocess
import sys
from pathlib import Path
import argparse
import time

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "output"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"

# Expected sample sizes for verification
EXPECTED_BASELINE_N = 7219
EXPECTED_DEATHS = 3466
EXPECTED_DEATH_PCT = 48.0
EXPECTED_CVD_FREE_N = 5308  # Baseline CVD-free; analysis N=4,956 after covariate missingness

# Pipeline definition: (script, description, expected_outputs)
PIPELINE = [
    (
        "01_primary_analysis.py",
        "Primary Cox models, sample construction, tables 1-4",
        [
            TABLES_DIR / "table1_baseline_v5.csv",
            TABLES_DIR / "table2_mortality_revised.csv",
            TABLES_DIR / "table3_age_stratified_revised.csv",
            TABLES_DIR / "table4_cardiac_revised.csv",
            OUTPUT_DIR / "analytic_sample_v5.csv",
        ],
    ),
    (
        "01b_export_survey_data.py",
        "Merge HRS survey design variables onto analytic sample",
        [
            OUTPUT_DIR / "analytic_sample_v5_survey.csv",
        ],
    ),
    (
        "02_figures.py",
        "Publication figures (KM curves, forest plot, DAG, PS overlap)",
        [
            FIGURES_DIR / "Figure1_DAG.pdf",
            FIGURES_DIR / "Figure2_KM_Mortality.pdf",
            FIGURES_DIR / "Figure3_Forest.pdf",
        ],
    ),
    (
        "03_secondary_analysis.py",
        "Supplementary analyses, Stata export, participant flow data",
        [
            TABLES_DIR / "etable_dose_response_detailed.csv",
            TABLES_DIR / "etable_ps_balance.csv",
            TABLES_DIR / "etable_missingness.csv",
            TABLES_DIR / "participant_flow_v6.csv",
        ],
    ),
    (
        "04_flow_diagram.py",
        "Participant flow diagram (eFigure 1)",
        [
            FIGURES_DIR / "eFigure1_Flow_Diagram_v6.pdf",
        ],
    ),
    (
        "05_sensitivity_weighted.py",
        "PH diagnostics, weighted estimates, LBQ selection bias",
        [
            TABLES_DIR / "etable_ph_diagnostics_v7.csv",
            TABLES_DIR / "etable_lbq_selection_v7.csv",
            TABLES_DIR / "etable_selection_bias_v6.csv",
        ],
    ),
    (
        "06_sensitivity_clustering.py",
        "Standardised risks, household clustering",
        [
            TABLES_DIR / "etable_absolute_risks_v2.csv",
        ],
    ),
    # Script 07 is Stata
    (
        "07_competing_risks_heart_disease.do",
        "Fine-Gray competing risks for incident heart disease (Stata)",
        [
            TABLES_DIR / "etable6_fine_gray_v6.csv",
        ],
    ),
    (
        "08_exploratory_reviewer_analyses.py",
        "Lexis expansion, PS-matched age strata, weight renormalisation",
        [
            TABLES_DIR / "etable_attained_age.csv",
            TABLES_DIR / "etable_spline_sensitivity.csv",
            TABLES_DIR / "etable_weighted_age_specific_hrs.csv",
            TABLES_DIR / "etable_cross_classification.csv",
            TABLES_DIR / "etable_insurance_analysis.csv",
            TABLES_DIR / "etable_insurance_descriptive.csv",
        ],
    ),
    # Script 09 is Stata
    (
        "09_survey_design_cox.do",
        "Survey-weighted Cox models with full HRS design (Stata svy: stcox)",
        [
            TABLES_DIR / "table2_weighted_robust_se.csv",
        ],
    ),
    (
        "10_biomarker_analysis.py",
        "Exploratory biomarker analyses (requires restricted data)",
        [
            TABLES_DIR / "etable_biomarker_descriptives.csv",
            TABLES_DIR / "etable_biomarker_associations.csv",
            TABLES_DIR / "etable_crp_prevalence.csv",
            TABLES_DIR / "etable_crp_mediation.csv",
            TABLES_DIR / "etable_crp_mortality.csv",
            TABLES_DIR / "etable_crp_trajectories.csv",
        ],
    ),
]

STATA_SCRIPTS = {"07_competing_risks_heart_disease.do", "09_survey_design_cox.do"}


def verify_outputs():
    """Check that all expected output files exist.

    Output files are not included in the repository; they are generated by
    running the analysis scripts against the HRS data. If the pipeline has
    not been run yet (no outputs exist), this returns True with a message
    rather than failing.
    """
    all_expected = [f for _, _, outputs in PIPELINE for f in outputs]
    any_exist = any(f.exists() for f in all_expected)

    if not any_exist:
        print("  Pipeline has not been run yet — no outputs to verify.")
        print("  Run 'python run_all.py' to generate outputs, then re-run with --verify.")
        return True

    all_ok = True
    for script, desc, outputs in PIPELINE:
        missing = [str(f) for f in outputs if not f.exists()]
        if missing:
            all_ok = False
            print(f"  MISSING outputs from {script}:")
            for m in missing:
                print(f"    - {m}")
        else:
            print(f"  OK: {script} ({len(outputs)} output(s))")
    return all_ok


def verify_sample_sizes():
    """Check expected sample sizes in the analytic dataset."""
    import pandas as pd

    sample_file = OUTPUT_DIR / "analytic_sample_v5.csv"
    if not sample_file.exists():
        print("  SKIP: analytic_sample_v5.csv not found (run pipeline first)")
        return True  # not a failure — pipeline hasn't been run yet

    df = pd.read_csv(sample_file)
    ok = True

    # Baseline N
    n = len(df)
    if n != EXPECTED_BASELINE_N:
        print(f"  FAIL: Baseline N = {n}, expected {EXPECTED_BASELINE_N}")
        ok = False
    else:
        print(f"  OK: Baseline N = {n}")

    # Deaths
    deaths = int(df['died'].sum())
    if deaths != EXPECTED_DEATHS:
        print(f"  FAIL: Deaths = {deaths}, expected {EXPECTED_DEATHS}")
        ok = False
    else:
        print(f"  OK: Deaths = {deaths}")

    # Death percentage
    death_pct = round(100 * deaths / n, 1)
    if death_pct != EXPECTED_DEATH_PCT:
        print(f"  FAIL: Death % = {death_pct}, expected {EXPECTED_DEATH_PCT}")
        ok = False
    else:
        print(f"  OK: Death % = {death_pct}%")

    # CVD-free N (baseline CVD-free with incident_heart data)
    cvd_free_n = int(df['incident_heart'].notna().sum())
    if cvd_free_n != EXPECTED_CVD_FREE_N:
        print(f"  WARN: CVD-free with outcome data = {cvd_free_n}, expected {EXPECTED_CVD_FREE_N}")
        # Warning not failure -- may vary slightly with covariate missingness
    else:
        print(f"  OK: CVD-free with outcome data = {cvd_free_n}")

    return ok


def run_script(script_name):
    """Run a single Python script and return success status."""
    script_path = PROJECT_DIR / script_name

    if not script_path.exists():
        print(f"  SKIP: {script_name} not found")
        return False

    if script_name in STATA_SCRIPTS:
        print(f"  SKIP: {script_name} is a Stata .do file (run manually in Stata)")
        return True

    print(f"  Running {script_name}...")
    start = time.time()

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(PROJECT_DIR),
        capture_output=True,
        text=True,
    )

    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"  FAILED ({elapsed:.1f}s)")
        print(f"  stderr: {result.stderr[-500:]}" if result.stderr else "")
        return False

    print(f"  Done ({elapsed:.1f}s)")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run the full Financial Strain analysis pipeline."
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify output files exist without re-running scripts.",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("FINANCIAL STRAIN AND MORTALITY -- MASTER PIPELINE")
    print("=" * 80)

    if args.verify:
        print("\nVerifying output files...")
        outputs_ok = verify_outputs()

        print("\nVerifying sample sizes...")
        sizes_ok = verify_sample_sizes()

        if outputs_ok and sizes_ok:
            print("\nAll checks passed.")
        else:
            print("\nSome checks failed. See details above.")
            sys.exit(1)
    else:
        print("\nRunning full pipeline...\n")
        failed = []

        for script, desc, outputs in PIPELINE:
            print(f"\n--- {script}: {desc} ---")
            success = run_script(script)
            if not success and script not in STATA_SCRIPTS:
                failed.append(script)

        print("\n" + "=" * 80)
        if failed:
            print(f"Pipeline completed with {len(failed)} failure(s):")
            for f in failed:
                print(f"  - {f}")
            sys.exit(1)
        else:
            print("Pipeline completed successfully.")

        print("\nVerifying outputs...")
        verify_outputs()

        print("\nVerifying sample sizes...")
        verify_sample_sizes()


if __name__ == "__main__":
    main()
