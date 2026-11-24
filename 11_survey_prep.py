"""
Prepares the file for the Stata survey models and fits the propensity score analyses.

Merges Wave 7 (2004) covariates from the RAND HRS Longitudinal File, runs 1:1 caliper matching
on the Model 3b covariates (and a variant adding self-rated health), builds stabilised
inverse-probability-of-treatment weights under two propensity models, and computes E-values
for the unweighted and matched estimates.

Writes output/analytic_sample_svy.csv (input to 12_survey_models.do) and, in output/tables/:
w7_attrition.csv, matched_refit.csv, matched_balance.csv, iptw_diagnostics.csv,
iptw_overlap.csv and evalues_unweighted.csv.
"""

import numpy as np
import pandas as pd
import pyreadstat
from pathlib import Path
from lifelines import CoxPHFitter
from scipy.special import logit as scipy_logit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from model_specs import MODEL3B_COVARS, EXPOSURE_VAR, PENALIZER

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "output"
TABLES_DIR = OUTPUT_DIR / "tables"
HRS_FILE = PROJECT_DIR / "randhrs1992_2022v1_STATA" / "randhrs1992_2022v1.dta"

SEED = 42
W7_RAW_VARS = ["hhidpn", "r7smoken", "r7bmi", "r7hibpe", "r7diabe", "r7hearte",
               "r7stroke", "h7itot", "h7atotb"]

# 1. Analytic sample with Wave 7 covariates

df = pd.read_csv(OUTPUT_DIR / "analytic_sample_survey.csv", low_memory=False)
assert len(df) == 7219, f"Expected N=7219 in analytic sample, got {len(df)}"

w7, _ = pyreadstat.read_dta(str(HRS_FILE), usecols=W7_RAW_VARS)
w7["hhidpn"] = w7["hhidpn"].astype("int64")
df["hhidpn"] = df["hhidpn"].astype("int64")
df = df.merge(w7, on="hhidpn", how="left", validate="one_to_one")

# Wave 7 versions of the behaviour/condition covariates (same definitions as Wave 8:
# binary conditions, current smoking, continuous BMI, IHS-transformed income and wealth).
# pyreadstat converts Stata extended missings to NaN, so no numeric sentinel codes reach
# these recodes.
df["current_smoker_w7"] = np.where(df["r7smoken"].isin([0, 1]), df["r7smoken"], np.nan)
df["bmi_w7"] = df["r7bmi"]
for cond, raw in [("baseline_hypertension_w7", "r7hibpe"), ("baseline_diabetes_w7", "r7diabe"),
                  ("baseline_heart_w7", "r7hearte"), ("baseline_stroke_w7", "r7stroke")]:
    df[cond] = np.where(df[raw].isin([0, 1]), df[raw], np.nan)
df["asinh_income_w7"] = np.arcsinh(df["h7itot"])
df["asinh_wealth_w7"] = np.arcsinh(df["h7atotb"])

W7_LAG_VARS = ["current_smoker_w7", "bmi_w7", "baseline_hypertension_w7",
               "baseline_diabetes_w7", "baseline_heart_w7", "baseline_stroke_w7"]

# Characterise participants lost to the Wave 7 covariate requirement (table S5 footnote)
w7_missing = df[W7_LAG_VARS].isna().any(axis=1)
attr_rows = []
for var in ["age", "female", EXPOSURE_VAR, "asinh_wealth"]:
    attr_rows.append({
        "Variable": var,
        "W7_complete_mean": round(df.loc[~w7_missing, var].mean(), 3),
        "W7_missing_mean": round(df.loc[w7_missing, var].mean(), 3),
        "N_missing": int(w7_missing.sum()),
    })
pd.DataFrame(attr_rows).to_csv(TABLES_DIR / "w7_attrition.csv", index=False)
print("Saved: w7_attrition.csv")

# 2. Propensity score matching (tables S17 and S18)


def caliper_match_ps(data, treat_col, ps_col, caliper_sd=0.2, seed=SEED):
    """Greedy 1:1 nearest-neighbour matching without replacement within a caliper of
    0.2 SD of the logit propensity score; treated order shuffled with a fixed seed."""
    np.random.seed(seed)
    data = data.copy()
    eps = 1e-6
    ps_clipped = np.clip(data[ps_col], eps, 1 - eps)
    data["logit_ps"] = scipy_logit(ps_clipped)
    caliper = caliper_sd * data["logit_ps"].std()

    treated = data[data[treat_col] == 1].copy()
    control = data[data[treat_col] == 0].copy()
    treated = treated.sample(frac=1, random_state=seed)

    matched_rows = []
    used_control_idx = set()
    pair_id = 0
    for t_idx, t_row in treated.iterrows():
        available = control.loc[~control.index.isin(used_control_idx)]
        if available.empty:
            break
        distances = (available["logit_ps"] - t_row["logit_ps"]).abs()
        within = distances[distances <= caliper]
        if within.empty:
            continue
        c_idx = within.idxmin()
        used_control_idx.add(c_idx)
        pair_id += 1
        t_copy = t_row.copy()
        t_copy["pair_id"] = pair_id
        c_copy = control.loc[c_idx].copy()
        c_copy["pair_id"] = pair_id
        matched_rows.append(pd.DataFrame([t_copy]))
        matched_rows.append(pd.DataFrame([c_copy]))
    if not matched_rows:
        return None, 0
    return pd.concat(matched_rows, ignore_index=True), pair_id


def smd(var, g1, g0, weights1=None, weights0=None):
    """Standardised mean difference; weighted means/variances when weights given."""
    if weights1 is None:
        m1, v1 = g1[var].mean(), g1[var].var()
        m0, v0 = g0[var].mean(), g0[var].var()
    else:
        m1 = np.average(g1[var], weights=weights1)
        v1 = np.average((g1[var] - m1) ** 2, weights=weights1)
        m0 = np.average(g0[var], weights=weights0)
        v0 = np.average((g0[var] - m0) ** 2, weights=weights0)
    pooled = np.sqrt((v1 + v0) / 2)
    return (m1 - m0) / pooled if pooled > 0 else np.nan


PS_SPECS = {
    "model3b": list(MODEL3B_COVARS),
    "model3b_plus_srh": list(MODEL3B_COVARS) + ["srh_fairpoor"],
}

matched_results = []
balance_rows = []

for spec_name, ps_vars in PS_SPECS.items():
    cols = [EXPOSURE_VAR, "followup_years", "died"] + ps_vars
    d = df[cols].dropna().copy()
    X = StandardScaler().fit_transform(d[ps_vars].values)
    ps_model = LogisticRegression(max_iter=1000, random_state=SEED)
    ps_model.fit(X, d[EXPOSURE_VAR].values)
    d["ps"] = ps_model.predict_proba(X)[:, 1]

    matched, n_pairs = caliper_match_ps(d, EXPOSURE_VAR, "ps")
    assert matched is not None and n_pairs > 0, f"No matches for spec {spec_name}"
    print(f"  {spec_name}: {n_pairs} pairs, {int(matched['died'].sum())} deaths")

    mt = matched[matched[EXPOSURE_VAR] == 1]
    mc = matched[matched[EXPOSURE_VAR] == 0]
    for var in ps_vars:
        row = {"Spec": spec_name, "Covariate": var}
        # Balance before matching is reported for the primary specification only (table S17)
        if spec_name == "model3b":
            row["SMD_before"] = round(smd(var, d[d[EXPOSURE_VAR] == 1], d[d[EXPOSURE_VAR] == 0]), 4)
        row["SMD_after"] = round(smd(var, mt, mc), 4)
        balance_rows.append(row)

    fit_df = matched[["followup_years", "died", EXPOSURE_VAR, "pair_id"]].astype(float).dropna()

    # Primary: marginal Cox, SEs clustered on matched pair.
    cph_marginal = CoxPHFitter(penalizer=PENALIZER)
    cph_marginal.fit(fit_df, duration_col="followup_years", event_col="died",
                     formula=EXPOSURE_VAR, cluster_col="pair_id")
    # Sensitivity: pair-stratified Cox, model-based variance (conditional HR).
    cph_conditional = CoxPHFitter(penalizer=PENALIZER)
    cph_conditional.fit(fit_df, duration_col="followup_years", event_col="died",
                        formula=EXPOSURE_VAR, strata=["pair_id"])

    for fit_label, cph in [("marginal_pair_clustered", cph_marginal),
                           ("pair_stratified_conditional", cph_conditional)]:
        s = cph.summary.loc[EXPOSURE_VAR]
        matched_results.append({
            "Spec": spec_name, "Fit": fit_label,
            "N": len(fit_df), "Pairs": n_pairs, "Deaths": int(fit_df["died"].sum()),
            "HR": float(np.exp(s["coef"])),
            "CI_lower": float(np.exp(s["coef lower 95%"])),
            "CI_upper": float(np.exp(s["coef upper 95%"])),
            "P": float(s["p"]),
        })

matched_df = pd.DataFrame(matched_results)
matched_df.round(4).to_csv(TABLES_DIR / "matched_refit.csv", index=False)
pd.DataFrame(balance_rows).to_csv(TABLES_DIR / "matched_balance.csv", index=False)
print("Saved: matched_refit.csv, matched_balance.csv")

# 3. E-values for the unweighted and matched estimates (table S19)


def evalue(hr):
    """E-value for a hazard ratio with a common outcome (VanderWeele and Ding 2017):
    RR = (1 - 0.5^sqrt(HR)) / (1 - 0.5^sqrt(1/HR)), then E = RR + sqrt(RR(RR - 1))."""
    rr = (1 - 0.5 ** np.sqrt(hr)) / (1 - 0.5 ** np.sqrt(1 / hr))
    if rr < 1:
        rr = 1 / rr
    return rr + np.sqrt(rr * (rr - 1))


# Computed from the unrounded estimates. The E-value for the confidence limit is 1 when the
# interval includes 1.
unweighted = pd.read_csv(TABLES_DIR / "mortality_unweighted.csv").set_index("Model")
age_strata = pd.read_csv(TABLES_DIR / "age_stratified_unweighted.csv").set_index("Analysis")
matched_3b = matched_df[(matched_df["Spec"] == "model3b")
                        & (matched_df["Fit"] == "marginal_pair_clustered")].iloc[0]
estimates = [
    ("Model 3 (unweighted)", unweighted.loc["Model 3", "HR_unrounded"],
     unweighted.loc["Model 3", "CI_lower_unrounded"]),
    ("Model 3b (unweighted)", unweighted.loc["Model 3b", "HR_unrounded"],
     unweighted.loc["Model 3b", "CI_lower_unrounded"]),
    ("Age <65 (unweighted)", age_strata.loc["<65 years_model3b", "HR_unrounded"],
     age_strata.loc["<65 years_model3b", "CI_lower_unrounded"]),
    ("Propensity score matched (marginal, Model 3b covariates)",
     matched_3b["HR"], matched_3b["CI_lower"]),
]
evalue_rows = [{"Analysis": label, "HR": hr, "CI_lower": lo,
                "E_value": f"{evalue(hr):.2f}",
                "E_value_CI": f"{evalue(lo):.2f}" if lo > 1 else "1.00"}
               for label, hr, lo in estimates]
pd.DataFrame(evalue_rows).to_csv(TABLES_DIR / "evalues_unweighted.csv", index=False)
print("Saved: evalues_unweighted.csv")

# 4. Inverse-probability-of-treatment weights (table S7)

design_cols = [EXPOSURE_VAR, "followup_years", "died", "lb_weight", "raestrat",
               "raehsamp"] + list(MODEL3B_COVARS)
mask = df[design_cols].notna().all(axis=1) & (df["lb_weight"] > 0)
dw = df.loc[mask].copy()
assert len(dw) == 6808, f"Expected weighted-design N=6808, got {len(dw)}"

diag_rows = []

# Stabilised weights: the numerator is the marginal exposure prevalence (unweighted for
# approach A, LBQ-weighted for approach B); the denominator is the fitted propensity of the
# observed exposure state.

# Approach A (primary, DuGoff et al 2014): unweighted logistic fit, survey weight
# included as a predictor; survey weights enter at the outcome stage.
vars_a = list(MODEL3B_COVARS) + ["lb_weight"]
Xa = StandardScaler().fit_transform(dw[vars_a].values)
ps_a = LogisticRegression(max_iter=1000, random_state=SEED).fit(
    Xa, dw[EXPOSURE_VAR].values).predict_proba(Xa)[:, 1]
dw["ps_dugoff"] = ps_a
prev_a = dw[EXPOSURE_VAR].mean()
dw["sw_a"] = np.where(dw[EXPOSURE_VAR] == 1, prev_a / dw["ps_dugoff"],
                      (1 - prev_a) / (1 - dw["ps_dugoff"]))

# Approach B (sensitivity, Ridgeway et al 2015): PS model estimated with the survey
# weight as a sampling weight.
Xb = StandardScaler().fit_transform(dw[list(MODEL3B_COVARS)].values)
ps_b = LogisticRegression(max_iter=1000, random_state=SEED).fit(
    Xb, dw[EXPOSURE_VAR].values, sample_weight=dw["lb_weight"].values).predict_proba(Xb)[:, 1]
dw["ps_svywt"] = ps_b
prev_b = np.average(dw[EXPOSURE_VAR], weights=dw["lb_weight"])
dw["sw_b"] = np.where(dw[EXPOSURE_VAR] == 1, prev_b / dw["ps_svywt"],
                      (1 - prev_b) / (1 - dw["ps_svywt"]))

# Propensity-score overlap by exposure group, both approaches
overlap_rows = []
for label, pscol in [("A_dugoff", "ps_dugoff"), ("B_svyweighted", "ps_svywt")]:
    for grp, gname in [(1, "exposed"), (0, "unexposed")]:
        g = dw.loc[dw[EXPOSURE_VAR] == grp, pscol]
        overlap_rows.append({
            "Approach": label, "Group": gname, "N": len(g),
            "PS_min": round(g.min(), 4), "PS_max": round(g.max(), 4),
            "N_below_0.01": int((g < 0.01).sum()), "N_above_0.99": int((g > 0.99).sum()),
        })

for label, col in [("A_dugoff", "sw_a"), ("B_svyweighted", "sw_b")]:
    lo, hi = dw[col].quantile([0.01, 0.99])
    dw[f"{col}_trunc"] = dw[col].clip(lo, hi)
    for variant, wcol in [("untruncated", col), ("truncated_1_99", f"{col}_trunc")]:
        composite = dw[wcol] * dw["lb_weight"]
        ess = composite.sum() ** 2 / (composite ** 2).sum()
        g1, g0 = dw[dw[EXPOSURE_VAR] == 1], dw[dw[EXPOSURE_VAR] == 0]
        w1 = (g1[wcol] * g1["lb_weight"]).values
        w0 = (g0[wcol] * g0["lb_weight"]).values
        ess1 = w1.sum() ** 2 / (w1 ** 2).sum()
        ess0 = w0.sum() ** 2 / (w0 ** 2).sum()
        smds = {v: smd(v, g1, g0, w1, w0) for v in MODEL3B_COVARS}
        max_abs_smd = max(abs(s) for s in smds.values())
        diag_rows.append({
            "Approach": label, "Variant": variant,
            "StabWeight_median": round(dw[wcol].median(), 4),
            "StabWeight_max": round(dw[wcol].max(), 4),
            "ESS_composite": round(ess, 2),
            "ESS_exposed": round(ess1, 2),
            "ESS_unexposed": round(ess0, 2),
            "Max_abs_weighted_SMD": round(max_abs_smd, 4),
        })

pd.DataFrame(diag_rows).to_csv(TABLES_DIR / "iptw_diagnostics.csv", index=False)
pd.DataFrame(overlap_rows).to_csv(TABLES_DIR / "iptw_overlap.csv", index=False)
print("Saved: iptw_diagnostics.csv, iptw_overlap.csv")

# 5. Analytic file for the Stata survey models

iptw_cols = ["hhidpn", "sw_a", "sw_a_trunc", "sw_b", "sw_b_trunc", "ps_dugoff", "ps_svywt"]
out = df.merge(dw[iptw_cols], on="hhidpn", how="left", validate="one_to_one")
out.to_csv(OUTPUT_DIR / "analytic_sample_svy.csv", index=False)
assert len(out) == 7219
print(f"Saved: analytic_sample_svy.csv (N={len(out)}, "
      f"{out['sw_a'].notna().sum()} with IPT weights)")
