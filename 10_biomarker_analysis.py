"""
10_biomarker_analysis.py
Exploratory biomarker analyses for biological plausibility assessment.

Examines whether financial strain is associated with inflammatory and metabolic
biomarkers (dried blood spot assays from HRS 2006), stratified by age (<65 vs
65+). These cross-sectional associations cannot establish temporal ordering and
should NOT be interpreted as mediation. The biomarker and exposure were measured
in the same wave.

Analyses performed:
1. Merge NHANES-calibrated biomarker data onto analytic sample
2. Descriptive biomarker means by strain status and age group
3. Age-stratified OLS regressions for each biomarker
4. High CRP prevalence (>3 mg/L) by strain and age group
5. CRP overlap characterisation: Model 3b with/without CRP adjustment
6. CRP as mortality predictor (confirmatory)
7. Longitudinal CRP trajectories (2006, 2010, 2014)
8. Strain x age interaction on log CRP

Biomarker data require an HRS restricted-data use agreement.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from lifelines import CoxPHFitter
from scipy import stats
import warnings
import pyreadstat
from model_specs import MODEL3B_COVARS, EXPOSURE_VAR, PENALIZER

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "output"
TABLES_DIR = OUTPUT_DIR / "tables"
TABLES_DIR.mkdir(exist_ok=True, parents=True)

# Biomarker file paths (restricted data)
BIOMK06_FILE = PROJECT_DIR / "Special Access Data" / "biomkr06" / "biomk06bl_r.sav"
CRP_XWAVE_FILE = PROJECT_DIR / "Special Access Data" / "dbs_crp" / "HRS_CRP_XWAVE.dta"

# NHANES-calibrated biomarker variable names from the 2006 DBS file
BIOMARKER_VARS = {
    "KCRP_ADJ": "CRP (mg/L)",
    "KA1C_ADJ": "HbA1c (%)",
    "KTC_ADJ": "Total cholesterol (mg/dL)",
    "KHDL_ADJ": "HDL cholesterol (mg/dL)",
    "KCYSC_ADJ": "Cystatin C (mg/L)",
}

CRP_HIGH_THRESHOLD = 3.0  # mg/L, AHA/CDC clinical threshold

print("=" * 80)
print("EXPLORATORY BIOMARKER ANALYSES")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD AND MERGE BIOMARKER DATA
# ============================================================================

print("\nStep 1: Loading and merging biomarker data...")

df = pd.read_csv(OUTPUT_DIR / "analytic_sample_v5.csv")
print(f"  Analytic sample: N={len(df)}")

# Load 2006 DBS biomarkers (.sav format)
if not BIOMK06_FILE.exists():
    print(f"  ERROR: Biomarker file not found: {BIOMK06_FILE}")
    print("  This script requires HRS restricted-use biomarker data.")
    raise SystemExit(1)

biomk_cols = ["HHID", "PN"] + list(BIOMARKER_VARS.keys())
biomk06, _ = pyreadstat.read_sav(str(BIOMK06_FILE), usecols=biomk_cols)

# Create merge key: hhidpn = HHID + PN as integer
biomk06["hhidpn"] = (
    biomk06["HHID"].astype(str).str.strip()
    + biomk06["PN"].astype(str).str.strip()
).astype(int)
biomk06 = biomk06.drop(columns=["HHID", "PN"])

n_before = len(df)
df = df.merge(biomk06, on="hhidpn", how="left")
assert len(df) == n_before, "Merge changed row count"

# Load cross-wave CRP data (.dta format)
if not CRP_XWAVE_FILE.exists():
    print(f"  WARNING: Cross-wave CRP file not found: {CRP_XWAVE_FILE}")
    print("  Longitudinal CRP trajectory analysis will be skipped.")
    crp_xwave = None
else:
    crp_xwave, _ = pyreadstat.read_dta(str(CRP_XWAVE_FILE))
    crp_xwave["hhidpn"] = (
        crp_xwave["HHID"].astype(str).str.strip()
        + crp_xwave["PN"].astype(str).str.strip()
    ).astype(int)

# Convert biomarker columns to numeric
for var in BIOMARKER_VARS:
    if var in df.columns:
        df[var] = pd.to_numeric(df[var], errors="coerce")

# Log-transform CRP and create high-CRP indicator
df["log_crp"] = np.log(df["KCRP_ADJ"])
df["crp_high"] = np.where(
    df["KCRP_ADJ"].notna(),
    np.where(df["KCRP_ADJ"] > CRP_HIGH_THRESHOLD, 1, 0),
    np.nan,
)

n_with_biomarkers = df["KCRP_ADJ"].notna().sum()
pct_with = 100 * n_with_biomarkers / len(df)
print(f"  With DBS biomarker data: {n_with_biomarkers} ({pct_with:.1f}%)")

# ============================================================================
# STEP 2: DESCRIPTIVE BIOMARKER MEANS BY STRAIN AND AGE GROUP
# ============================================================================

print("\nStep 2: Descriptive biomarker statistics...")

desc_rows = []
for age_grp in ["<65 years", ">=65 years", "All"]:
    if age_grp == "All":
        subset = df
    else:
        subset = df[df["age_group"] == age_grp]

    for strain_val, strain_label in [(0, "No/low strain"), (1, "High strain")]:
        s = subset[subset[EXPOSURE_VAR] == strain_val]

        row = {"Age group": age_grp, "Financial strain": strain_label, "N": len(s)}

        for var, label in BIOMARKER_VARS.items():
            vals = s[var].dropna()
            row[f"{label} N"] = len(vals)
            row[f"{label} mean"] = f"{vals.mean():.2f}" if len(vals) > 0 else ""
            row[f"{label} SD"] = f"{vals.std():.2f}" if len(vals) > 0 else ""

        # Log CRP
        log_vals = s["log_crp"].dropna()
        row["Log CRP mean"] = f"{log_vals.mean():.3f}" if len(log_vals) > 0 else ""
        row["Log CRP SD"] = f"{log_vals.std():.3f}" if len(log_vals) > 0 else ""

        desc_rows.append(row)

    # P-value for strain difference within age group (t-test on each biomarker)
    no_strain = subset[subset[EXPOSURE_VAR] == 0]
    hi_strain = subset[subset[EXPOSURE_VAR] == 1]
    p_row = {"Age group": age_grp, "Financial strain": "P (t-test)", "N": ""}
    for var, label in BIOMARKER_VARS.items():
        v0 = no_strain[var].dropna()
        v1 = hi_strain[var].dropna()
        if len(v0) > 1 and len(v1) > 1:
            _, p = stats.ttest_ind(v0, v1, equal_var=False)
            p_row[f"{label} mean"] = f"{p:.3f}"
        else:
            p_row[f"{label} mean"] = ""
        p_row[f"{label} N"] = ""
        p_row[f"{label} SD"] = ""

    # Log CRP p-value
    l0 = no_strain["log_crp"].dropna()
    l1 = hi_strain["log_crp"].dropna()
    if len(l0) > 1 and len(l1) > 1:
        _, p = stats.ttest_ind(l0, l1, equal_var=False)
        p_row["Log CRP mean"] = f"{p:.3f}"
    else:
        p_row["Log CRP mean"] = ""
    p_row["Log CRP SD"] = ""
    desc_rows.append(p_row)

desc_df = pd.DataFrame(desc_rows)
desc_df.to_csv(TABLES_DIR / "etable_biomarker_descriptives.csv", index=False)
print("  Saved: etable_biomarker_descriptives.csv")

# ============================================================================
# STEP 3: AGE-STRATIFIED OLS FOR EACH BIOMARKER
# ============================================================================

print("\nStep 3: Age-stratified OLS biomarker associations...")


def ols_regression(y, X):
    """OLS via numpy.linalg.lstsq. Returns coefficients, SEs, P-values."""
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    n, k = X_arr.shape

    beta, residuals, rank, sv = np.linalg.lstsq(X_arr, y_arr, rcond=None)

    y_hat = X_arr @ beta
    resid = y_arr - y_hat
    sigma2 = np.sum(resid ** 2) / (n - k)

    # Covariance matrix of coefficients
    try:
        cov_beta = sigma2 * np.linalg.inv(X_arr.T @ X_arr)
    except np.linalg.LinAlgError:
        cov_beta = sigma2 * np.linalg.pinv(X_arr.T @ X_arr)

    se = np.sqrt(np.diag(cov_beta))
    t_stat = beta / se
    p_values = 2 * stats.t.sf(np.abs(t_stat), df=n - k)

    return beta, se, p_values, n


# Covariates for biomarker regressions (Model 3b set)
bio_covars = MODEL3B_COVARS.copy()

assoc_rows = []
biomarker_outcomes = list(BIOMARKER_VARS.keys()) + ["log_crp"]
biomarker_labels = dict(BIOMARKER_VARS)
biomarker_labels["log_crp"] = "Log CRP (mg/L)"

for age_grp in ["<65 years", ">=65 years", "All"]:
    if age_grp == "All":
        subset = df.copy()
    else:
        subset = df[df["age_group"] == age_grp].copy()

    for outcome_var in biomarker_outcomes:
        label = biomarker_labels[outcome_var]

        # Build design matrix: exposure + covariates + intercept
        model_vars = [EXPOSURE_VAR] + bio_covars
        reg_data = subset[[outcome_var] + model_vars].dropna()

        if len(reg_data) < 30:
            continue

        y = reg_data[outcome_var].values
        X = reg_data[model_vars].values
        X = np.column_stack([np.ones(len(X)), X])  # Add intercept

        beta, se, pvals, n = ols_regression(y, X)

        # Exposure coefficient is index 1 (index 0 is intercept)
        coef = beta[1]
        se_exp = se[1]
        p_exp = pvals[1]
        ci_lo = coef - 1.96 * se_exp
        ci_hi = coef + 1.96 * se_exp

        # For log CRP, express as percentage change
        if outcome_var == "log_crp":
            pct_change = (np.exp(coef) - 1) * 100
            pct_note = f"{pct_change:+.1f}%"
        else:
            pct_note = ""

        assoc_rows.append({
            "Age group": age_grp,
            "Biomarker": label,
            "N": n,
            "Coefficient": f"{coef:.3f}",
            "SE": f"{se_exp:.3f}",
            "95% CI": f"{ci_lo:.3f} to {ci_hi:.3f}",
            "P": f"{p_exp:.3f}",
            "% change (if log)": pct_note,
        })

assoc_df = pd.DataFrame(assoc_rows)
assoc_df.to_csv(TABLES_DIR / "etable_biomarker_associations.csv", index=False)
print("  Saved: etable_biomarker_associations.csv")

# ============================================================================
# STEP 4: HIGH CRP PREVALENCE BY STRAIN AND AGE GROUP
# ============================================================================

print("\nStep 4: High CRP prevalence (>3 mg/L)...")

prev_rows = []
for age_grp in ["<65 years", ">=65 years", "All"]:
    if age_grp == "All":
        subset = df
    else:
        subset = df[df["age_group"] == age_grp]

    for strain_val, strain_label in [(0, "No/low strain"), (1, "High strain")]:
        s = subset[subset[EXPOSURE_VAR] == strain_val]
        crp_valid = s["crp_high"].dropna()
        n_valid = len(crp_valid)
        n_high = int(crp_valid.sum())
        pct_high = 100 * n_high / n_valid if n_valid > 0 else np.nan

        prev_rows.append({
            "Age group": age_grp,
            "Financial strain": strain_label,
            "N with CRP": n_valid,
            "N CRP >3 mg/L": n_high,
            "% CRP >3 mg/L": f"{pct_high:.1f}" if pd.notna(pct_high) else "",
        })

    # Chi-square test for strain difference
    ct = pd.crosstab(
        subset.loc[subset["crp_high"].notna(), EXPOSURE_VAR],
        subset.loc[subset["crp_high"].notna(), "crp_high"],
    )
    if ct.shape == (2, 2):
        chi2, p_chi, _, _ = stats.chi2_contingency(ct)
        prev_rows.append({
            "Age group": age_grp,
            "Financial strain": "P (chi-square)",
            "N with CRP": "",
            "N CRP >3 mg/L": "",
            "% CRP >3 mg/L": f"{p_chi:.3f}",
        })

prev_df = pd.DataFrame(prev_rows)
prev_df.to_csv(TABLES_DIR / "etable_crp_prevalence.csv", index=False)
print("  Saved: etable_crp_prevalence.csv")

# ============================================================================
# STEP 5: CRP OVERLAP CHARACTERISATION (MODEL 3b WITH/WITHOUT CRP)
# ============================================================================

print("\nStep 5: CRP overlap with mortality signal...")

med_rows = []
cox_covars_base = [EXPOSURE_VAR] + MODEL3B_COVARS

for age_grp in ["<65 years", ">=65 years", "All"]:
    if age_grp == "All":
        subset = df.copy()
    else:
        subset = df[df["age_group"] == age_grp].copy()

    # Restrict to those with CRP data
    subset_crp = subset[subset["log_crp"].notna()].copy()

    for model_label, covars in [
        ("Model 3b (biomarker subsample)", cox_covars_base),
        ("Model 3b + log CRP", cox_covars_base + ["log_crp"]),
    ]:
        available = [c for c in covars if c in subset_crp.columns]
        cox_data = subset_crp[available + ["followup_years", "died"]].dropna()

        if len(cox_data) < 50 or cox_data["died"].sum() < 20:
            continue

        cph = CoxPHFitter(penalizer=PENALIZER)
        cph.fit(cox_data, duration_col="followup_years", event_col="died")

        hr = np.exp(cph.params_[EXPOSURE_VAR])
        ci = np.exp(cph.confidence_intervals_.loc[EXPOSURE_VAR].values)
        p = cph.summary.loc[EXPOSURE_VAR, "p"]

        row = {
            "Age group": age_grp,
            "Model": model_label,
            "N": len(cox_data),
            "Deaths": int(cox_data["died"].sum()),
            "Strain HR": f"{hr:.2f}",
            "95% CI": f"{ci[0]:.2f} to {ci[1]:.2f}",
            "P": f"{p:.3f}",
        }

        # Report CRP HR if in model
        if "log_crp" in available and "log_crp" in cph.params_.index:
            hr_crp = np.exp(cph.params_["log_crp"])
            ci_crp = np.exp(cph.confidence_intervals_.loc["log_crp"].values)
            p_crp = cph.summary.loc["log_crp", "p"]
            row["CRP HR (per log-unit)"] = f"{hr_crp:.2f}"
            row["CRP 95% CI"] = f"{ci_crp[0]:.2f} to {ci_crp[1]:.2f}"
            row["CRP P"] = f"{p_crp:.3f}"

        med_rows.append(row)

med_df = pd.DataFrame(med_rows)
med_df.to_csv(TABLES_DIR / "etable_crp_mediation.csv", index=False)
print("  Saved: etable_crp_mediation.csv")

# ============================================================================
# STEP 6: CRP -> MORTALITY CONFIRMATORY
# ============================================================================

print("\nStep 6: CRP as mortality predictor...")

mort_rows = []
df_crp = df[df["log_crp"].notna()].copy()

for age_grp in ["<65 years", ">=65 years", "All"]:
    if age_grp == "All":
        subset = df_crp.copy()
    else:
        subset = df_crp[df_crp["age_group"] == age_grp].copy()

    for crp_var, crp_label in [("log_crp", "Log CRP (continuous)"),
                                ("crp_high", "CRP >3 mg/L (binary)")]:
        covars = [crp_var] + MODEL3B_COVARS
        available = [c for c in covars if c in subset.columns]
        cox_data = subset[available + ["followup_years", "died"]].dropna()

        if len(cox_data) < 50 or cox_data["died"].sum() < 20:
            continue

        cph = CoxPHFitter(penalizer=PENALIZER)
        cph.fit(cox_data, duration_col="followup_years", event_col="died")

        hr = np.exp(cph.params_[crp_var])
        ci = np.exp(cph.confidence_intervals_.loc[crp_var].values)
        p = cph.summary.loc[crp_var, "p"]

        mort_rows.append({
            "Age group": age_grp,
            "CRP measure": crp_label,
            "N": len(cox_data),
            "Deaths": int(cox_data["died"].sum()),
            "HR": f"{hr:.2f}",
            "95% CI": f"{ci[0]:.2f} to {ci[1]:.2f}",
            "P": f"{p:.3f}",
        })

mort_df = pd.DataFrame(mort_rows)
mort_df.to_csv(TABLES_DIR / "etable_crp_mortality.csv", index=False)
print("  Saved: etable_crp_mortality.csv")

# ============================================================================
# STEP 7: LONGITUDINAL CRP TRAJECTORIES (2006, 2010, 2014)
# ============================================================================

print("\nStep 7: Longitudinal CRP trajectories...")

if crp_xwave is not None:
    # Cross-wave CRP file contains CRP measures across waves
    # Identify wave-specific CRP columns
    crp_wave_cols = {}
    for col in crp_xwave.columns:
        col_upper = col.upper()
        # Wave 8 = 2006, Wave 10 = 2010, Wave 12 = 2014
        if "8" in col_upper and "CRP" in col_upper and "ADJ" in col_upper:
            crp_wave_cols["2006"] = col
        elif "10" in col_upper and "CRP" in col_upper and "ADJ" in col_upper:
            crp_wave_cols["2010"] = col
        elif "12" in col_upper and "CRP" in col_upper and "ADJ" in col_upper:
            crp_wave_cols["2014"] = col

    if crp_wave_cols:
        keep_cols = ["hhidpn"] + list(crp_wave_cols.values())
        crp_long = crp_xwave[keep_cols].copy()

        # Merge onto analytic sample
        df_traj = df[["hhidpn", EXPOSURE_VAR, "age_group"]].merge(
            crp_long, on="hhidpn", how="inner"
        )

        traj_rows = []
        for age_grp in ["<65 years", ">=65 years", "All"]:
            if age_grp == "All":
                subset = df_traj
            else:
                subset = df_traj[df_traj["age_group"] == age_grp]

            for strain_val, strain_label in [(0, "No/low strain"), (1, "High strain")]:
                s = subset[subset[EXPOSURE_VAR] == strain_val]

                row = {
                    "Age group": age_grp,
                    "Financial strain": strain_label,
                    "N": len(s),
                }

                for year, col in sorted(crp_wave_cols.items()):
                    vals = pd.to_numeric(s[col], errors="coerce").dropna()
                    if len(vals) > 0:
                        log_vals = np.log(vals[vals > 0])
                        row[f"CRP {year} mean (mg/L)"] = f"{vals.mean():.2f}"
                        row[f"CRP {year} N"] = len(vals)
                        row[f"Log CRP {year} mean"] = (
                            f"{log_vals.mean():.3f}" if len(log_vals) > 0 else ""
                        )
                    else:
                        row[f"CRP {year} mean (mg/L)"] = ""
                        row[f"CRP {year} N"] = 0
                        row[f"Log CRP {year} mean"] = ""

                traj_rows.append(row)

        traj_df = pd.DataFrame(traj_rows)
        traj_df.to_csv(TABLES_DIR / "etable_crp_trajectories.csv", index=False)
        print("  Saved: etable_crp_trajectories.csv")
    else:
        print("  WARNING: Could not identify wave-specific CRP columns in cross-wave file.")
        print(f"  Available columns: {list(crp_xwave.columns[:20])}")
        # Write empty placeholder
        pd.DataFrame().to_csv(TABLES_DIR / "etable_crp_trajectories.csv", index=False)
        print("  Saved: etable_crp_trajectories.csv (empty -- columns not found)")
else:
    print("  SKIP: Cross-wave CRP file not available.")
    pd.DataFrame().to_csv(TABLES_DIR / "etable_crp_trajectories.csv", index=False)
    print("  Saved: etable_crp_trajectories.csv (empty -- file not found)")

# ============================================================================
# STEP 8: STRAIN x AGE INTERACTION ON LOG CRP
# ============================================================================

print("\nStep 8: Strain x age interaction on log CRP...")

df_crp_int = df[df["log_crp"].notna()].copy()
df_crp_int["age_under65"] = np.where(df_crp_int["age"] < 65, 1, 0)
df_crp_int["strain_x_age_under65"] = (
    df_crp_int[EXPOSURE_VAR] * df_crp_int["age_under65"]
)

# Drop 'age' from covariates to avoid collinearity with age_under65
interaction_covars = [c for c in MODEL3B_COVARS if c != "age"]
model_vars = [EXPOSURE_VAR, "age_under65", "strain_x_age_under65"] + interaction_covars
reg_data = df_crp_int[["log_crp"] + model_vars].dropna()

if len(reg_data) >= 30:
    y = reg_data["log_crp"].values
    X = reg_data[model_vars].values
    X = np.column_stack([np.ones(len(X)), X])  # Add intercept

    beta, se, pvals, n = ols_regression(y, X)

    # Variable indices: 0=intercept, 1=exposure, 2=age_under65, 3=interaction
    var_names = ["intercept"] + model_vars
    print(f"  N = {n}")
    print(f"  Strain coefficient: {beta[1]:.3f} (P={pvals[1]:.3f})")
    print(f"  Age <65 coefficient: {beta[2]:.3f} (P={pvals[2]:.3f})")
    print(f"  Strain x Age <65 interaction: {beta[3]:.3f} (P={pvals[3]:.3f})")

    # Also report the implied strain effect within each age group
    strain_effect_under65 = beta[1] + beta[3]  # Main + interaction
    strain_effect_65plus = beta[1]  # Main effect only

    print(f"\n  Implied strain effect, age <65: {strain_effect_under65:.3f} "
          f"({(np.exp(strain_effect_under65)-1)*100:+.1f}%)")
    print(f"  Implied strain effect, age 65+: {strain_effect_65plus:.3f} "
          f"({(np.exp(strain_effect_65plus)-1)*100:+.1f}%)")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("BIOMARKER ANALYSIS COMPLETE")
print("=" * 80)
print(f"\nParticipants with biomarker data: {n_with_biomarkers} ({pct_with:.1f}%)")
print(f"\nOutput files:")
print(f"  etable_biomarker_descriptives.csv  -- Biomarker means by strain and age")
print(f"  etable_biomarker_associations.csv  -- OLS regressions (age-stratified)")
print(f"  etable_crp_prevalence.csv          -- High CRP (>3 mg/L) prevalence")
print(f"  etable_crp_mediation.csv           -- Model 3b with/without CRP")
print(f"  etable_crp_mortality.csv           -- CRP as mortality predictor")
print(f"  etable_crp_trajectories.csv        -- Longitudinal CRP (2006-2014)")
