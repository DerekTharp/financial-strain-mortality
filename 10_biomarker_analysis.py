"""
Exploratory biomarker analyses (tables S12 to S14), using the 2006 dried blood spot assays.
Table S12 compares baseline measures. Tables S13 and S14 use subsequent mortality.

Needs the HRS 2006 biomarker file, which is released under a separate data use agreement.
Writes biomarker_associations.csv, crp_prevalence.csv, crp_interaction.csv, crp_mortality.csv
and crp_adjustment.csv to output/tables/.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from lifelines import CoxPHFitter
from scipy import stats
import pyreadstat
from model_specs import MODEL3B_COVARS, EXPOSURE_VAR, PENALIZER

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "output"
TABLES_DIR = OUTPUT_DIR / "tables"
TABLES_DIR.mkdir(exist_ok=True, parents=True)

BIOMK06_FILE = PROJECT_DIR / "Special Access Data" / "biomkr06" / "biomk06bl_r.sav"

# NHANES-calibrated biomarker variable names from the 2006 DBS file
BIOMARKER_VARS = {
    "KCRP_ADJ": "CRP (mg/L)",
    "KA1C_ADJ": "HbA1c (%)",
    "KTC_ADJ": "Total cholesterol (mg/dL)",
    "KHDL_ADJ": "HDL cholesterol (mg/dL)",
    "KCYSC_ADJ": "Cystatin C (mg/L)",
}

CRP_HIGH_THRESHOLD = 3.0  # mg/L, AHA/CDC clinical threshold

if not BIOMK06_FILE.exists():
    print(f"Biomarker file not found: {BIOMK06_FILE}")
    print("Tables S12 to S14 are not produced without it.")
    raise SystemExit(0)

df = pd.read_csv(OUTPUT_DIR / "analytic_sample.csv")

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

# Age-stratified OLS for each biomarker (table S12)


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

for age_grp in ["<65 years", ">=65 years"]:
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
assoc_df.to_csv(TABLES_DIR / "biomarker_associations.csv", index=False)
print("Saved: biomarker_associations.csv")

# Prevalence of CRP above 3 mg/L by strain and age group (table S12)

prev_rows = []
for age_grp in ["<65 years", ">=65 years"]:
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

prev_df = pd.DataFrame(prev_rows)
prev_df.to_csv(TABLES_DIR / "crp_prevalence.csv", index=False)
print("Saved: crp_prevalence.csv")

# Strain x age interaction on log CRP (table S12)

df_crp_int = df[df["log_crp"].notna()].copy()
df_crp_int["age_under65"] = np.where(df_crp_int["age"] < 65, 1, 0)
df_crp_int["strain_x_age_under65"] = (
    df_crp_int[EXPOSURE_VAR] * df_crp_int["age_under65"]
)

# The pooled interaction uses the under-65 indicator in place of continuous age.
interaction_covars = [c for c in MODEL3B_COVARS if c != "age"]
model_vars = [EXPOSURE_VAR, "age_under65", "strain_x_age_under65"] + interaction_covars
reg_data = df_crp_int[["log_crp"] + model_vars].dropna()

if len(reg_data) >= 30:
    y = reg_data["log_crp"].values
    X = reg_data[model_vars].values
    X = np.column_stack([np.ones(len(X)), X])  # Add intercept

    beta, se, pvals, n = ols_regression(y, X)

    # Variable indices: 0=intercept, 1=exposure, 2=age_under65, 3=interaction
    pd.DataFrame([{
        "Term": "Strain x age <65 (log CRP)",
        "N": n,
        "Coefficient": beta[3],
        "CI_lower": beta[3] - 1.96 * se[3],
        "CI_upper": beta[3] + 1.96 * se[3],
        "P": pvals[3],
    }]).to_csv(TABLES_DIR / "crp_interaction.csv", index=False)
    print("Saved: crp_interaction.csv")

# CRP as a predictor of mortality (table S13)

mort_rows = []
df_crp = df[df["log_crp"].notna()].copy()

for crp_var, crp_label in [("log_crp", "Log CRP (continuous)"),
                            ("crp_high", "CRP >3 mg/L (binary)")]:
    covars = [crp_var] + MODEL3B_COVARS
    available = [c for c in covars if c in df_crp.columns]
    cox_data = df_crp[available + ["followup_years", "died"]].dropna()

    if len(cox_data) < 50 or cox_data["died"].sum() < 20:
        continue

    cph = CoxPHFitter(penalizer=PENALIZER)
    cph.fit(cox_data, duration_col="followup_years", event_col="died")

    hr = np.exp(cph.params_[crp_var])
    ci = np.exp(cph.confidence_intervals_.loc[crp_var].values)
    p = cph.summary.loc[crp_var, "p"]

    mort_rows.append({
        "CRP measure": crp_label,
        "N": len(cox_data),
        "Deaths": int(cox_data["died"].sum()),
        "HR": f"{hr:.2f}",
        "95% CI": f"{ci[0]:.2f} to {ci[1]:.2f}",
        "P": f"{p:.3f}",
    })

mort_df = pd.DataFrame(mort_rows)
mort_df.to_csv(TABLES_DIR / "crp_mortality.csv", index=False)
print("Saved: crp_mortality.csv")

# Strain and mortality with and without adjustment for CRP (table S14)

adj_rows = []
cox_covars_base = [EXPOSURE_VAR] + MODEL3B_COVARS

for age_grp in ["<65 years", ">=65 years"]:
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

        adj_rows.append({
            "Age group": age_grp,
            "Model": model_label,
            "N": len(cox_data),
            "Deaths": int(cox_data["died"].sum()),
            "Strain HR": f"{hr:.2f}",
            "95% CI": f"{ci[0]:.2f} to {ci[1]:.2f}",
            "P": f"{p:.4f}",
        })

adj_df = pd.DataFrame(adj_rows)
adj_df.to_csv(TABLES_DIR / "crp_adjustment.csv", index=False)
print("Saved: crp_adjustment.csv")
