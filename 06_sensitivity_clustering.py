"""
06_sensitivity_clustering.py
Supplementary sensitivity analyses: standardized risks and household clustering.

1. Covariate-standardized absolute risks within age strata (eTable 12)
2. Household clustering sensitivity analysis with robust SEs (eTable 13)

Date: February 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path
from lifelines import CoxPHFitter

# Set paths
PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "output"
TABLES_DIR = OUTPUT_DIR / "tables"
TABLES_DIR.mkdir(exist_ok=True, parents=True)

print("="*80)
print("06_sensitivity_clustering: Standardized risks and household clustering")
print("="*80)

# Load analytic sample
df = pd.read_csv(OUTPUT_DIR / "analytic_sample_v5.csv")
print(f"Loaded {len(df)} observations")

# Model 3b covariates (used throughout)
covariates = ['fin_strain_binary', 'age', 'female', 'race_nh_black', 'race_hispanic',
              'race_nh_other', 'education_yrs', 'married_partnered', 'current_smoker',
              'bmi', 'baseline_hypertension', 'baseline_diabetes', 'baseline_heart',
              'baseline_stroke', 'asinh_income', 'asinh_wealth']

# ============================================================================
# COVARIATE-STANDARDIZED ABSOLUTE RISKS WITHIN AGE STRATA
# ============================================================================

print("\n" + "="*80)
print("Covariate-standardized absolute risks by age strata")
print("="*80)

# Complete cases for Model 3b
model_vars = covariates + ['followup_years', 'died']

df_mort = df[model_vars].dropna().copy()
df_mort = df_mort[df_mort['followup_years'] > 0]
print(f"Mortality analysis sample: N = {len(df_mort)}")

# Create age strata
df_mort['age_under65'] = (df_mort['age'] < 65).astype(int)

def compute_standardized_survival(df_stratum, time_points=[5, 10]):
    """
    Compute covariate-standardized survival via marginal standardization (g-computation).
    Fits Cox within stratum, predicts under counterfactual exposure, averages across individuals.
    Age is retained as a covariate for within-stratum adjustment.
    """
    cph = CoxPHFitter(penalizer=1e-5)

    try:
        cph.fit(df_stratum[covariates + ['followup_years', 'died']],
                duration_col='followup_years', event_col='died')
    except Exception as e:
        print(f"  Model fitting error: {e}")
        return None

    results = {}

    for t in time_points:
        for exposure in [0, 1]:
            df_cf = df_stratum[covariates].copy()
            df_cf['fin_strain_binary'] = exposure

            try:
                surv_funcs = cph.predict_survival_function(df_cf, times=[t])
                # surv_funcs: index=times, columns=individuals (lifelines 0.30)
                mean_surv = float(surv_funcs.loc[t].mean())
                mean_mort = 1 - mean_surv

                exposure_label = 'high_strain' if exposure == 1 else 'low_strain'
                results[f'mortality_{t}yr_{exposure_label}'] = mean_mort * 100
            except Exception as e:
                print(f"  Prediction error for t={t}, exposure={exposure}: {e}")
                continue

    # Calculate risk differences
    for t in time_points:
        high_key = f'mortality_{t}yr_high_strain'
        low_key = f'mortality_{t}yr_low_strain'
        if high_key in results and low_key in results:
            results[f'risk_diff_{t}yr'] = results[high_key] - results[low_key]

    return results

# Compute for each age stratum
print("\n--- Age < 65 Years ---")
df_under65 = df_mort[df_mort['age_under65'] == 1].copy()
print(f"  N = {len(df_under65)}, Deaths = {df_under65['died'].sum()}")
results_under65 = compute_standardized_survival(df_under65, time_points=[5, 10])
if results_under65:
    for key, val in results_under65.items():
        print(f"  {key}: {val:.1f}%")

print("\n--- Age >= 65 Years ---")
df_65plus = df_mort[df_mort['age_under65'] == 0].copy()
print(f"  N = {len(df_65plus)}, Deaths = {df_65plus['died'].sum()}")
results_65plus = compute_standardized_survival(df_65plus, time_points=[5, 10])
if results_65plus:
    for key, val in results_65plus.items():
        print(f"  {key}: {val:.1f}%")

# Create output table
standardized_risks = []
for stratum, results, n, deaths in [
    ('Age < 65 years', results_under65, len(df_under65), df_under65['died'].sum()),
    ('Age >= 65 years', results_65plus, len(df_65plus), df_65plus['died'].sum())
]:
    if results:
        row = {
            'Age Stratum': stratum,
            'N': n,
            'Deaths': deaths,
            '5-Year Mortality, Low/No Strain (%)': f"{results.get('mortality_5yr_low_strain', np.nan):.1f}",
            '5-Year Mortality, High Strain (%)': f"{results.get('mortality_5yr_high_strain', np.nan):.1f}",
            '5-Year Risk Difference (pp)': f"{results.get('risk_diff_5yr', np.nan):.1f}",
            '10-Year Mortality, Low/No Strain (%)': f"{results.get('mortality_10yr_low_strain', np.nan):.1f}",
            '10-Year Mortality, High Strain (%)': f"{results.get('mortality_10yr_high_strain', np.nan):.1f}",
            '10-Year Risk Difference (pp)': f"{results.get('risk_diff_10yr', np.nan):.1f}",
        }
        standardized_risks.append(row)

df_std_risks = pd.DataFrame(standardized_risks)
df_std_risks.to_csv(TABLES_DIR / "etable_standardized_risks_v8.csv", index=False)
print(f"\nSaved: {TABLES_DIR / 'etable_standardized_risks_v8.csv'}")

# ============================================================================
# HOUSEHOLD CLUSTERING SENSITIVITY ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("Household clustering sensitivity")
print("="*80)

# HRS HHIDPN = HHID * 1000 + PN; extract household ID via integer division
df_mort_cluster = df[model_vars + ['hhidpn']].dropna().copy()
df_mort_cluster = df_mort_cluster[df_mort_cluster['followup_years'] > 0]

# Ensure hhidpn is integer before extracting household ID
df_mort_cluster['hhidpn'] = pd.to_numeric(df_mort_cluster['hhidpn'], errors='coerce')
n_bad = df_mort_cluster['hhidpn'].isna().sum()
if n_bad > 0:
    print(f"  Warning: {n_bad} non-numeric hhidpn values dropped")
    df_mort_cluster = df_mort_cluster.dropna(subset=['hhidpn'])

df_mort_cluster['household_id'] = (df_mort_cluster['hhidpn'] // 1000).astype(int)

# Sanity check: person number should be 10-99 (2-digit)
pn = (df_mort_cluster['hhidpn'] % 1000).astype(int)
if (pn < 10).any() or (pn > 99).any():
    n_odd = ((pn < 10) | (pn > 99)).sum()
    print(f"  Note: {n_odd} observations with unexpected person number (outside 10-99)")

n_households = df_mort_cluster['household_id'].nunique()
n_individuals = len(df_mort_cluster)
print(f"N individuals: {n_individuals}")
print(f"N households: {n_households}")
print(f"Mean individuals per household: {n_individuals/n_households:.2f}")

# Count households with multiple individuals
household_sizes = df_mort_cluster.groupby('household_id').size()
multi_person_hh = (household_sizes > 1).sum()
print(f"Households with >1 person: {multi_person_hh} ({100*multi_person_hh/n_households:.1f}%)")

# Fit standard Cox model (Model 3b)
print("\n--- Standard Model (Model 3b) ---")
cph_standard = CoxPHFitter(penalizer=1e-5)
cph_standard.fit(df_mort_cluster[covariates + ['followup_years', 'died']],
                 duration_col='followup_years', event_col='died')

hr_standard = np.exp(cph_standard.params_['fin_strain_binary'])
ci_low_std = np.exp(cph_standard.confidence_intervals_.loc['fin_strain_binary', '95% lower-bound'])
ci_high_std = np.exp(cph_standard.confidence_intervals_.loc['fin_strain_binary', '95% upper-bound'])
p_standard = cph_standard.summary.loc['fin_strain_binary', 'p']
print(f"  HR = {hr_standard:.2f} (95% CI: {ci_low_std:.2f}-{ci_high_std:.2f}), P = {p_standard:.3f}")

# Fit clustered model using robust variance estimator
print("\n--- Clustered Model (Robust SE by Household) ---")
cph_clustered = CoxPHFitter(penalizer=1e-5)
cph_clustered.fit(df_mort_cluster[covariates + ['followup_years', 'died', 'household_id']],
                  duration_col='followup_years', event_col='died',
                  cluster_col='household_id',
                  robust=True)

hr_clustered = np.exp(cph_clustered.params_['fin_strain_binary'])
ci_low_cl = np.exp(cph_clustered.confidence_intervals_.loc['fin_strain_binary', '95% lower-bound'])
ci_high_cl = np.exp(cph_clustered.confidence_intervals_.loc['fin_strain_binary', '95% upper-bound'])
p_clustered = cph_clustered.summary.loc['fin_strain_binary', 'p']
print(f"  HR = {hr_clustered:.2f} (95% CI: {ci_low_cl:.2f}-{ci_high_cl:.2f}), P = {p_clustered:.3f}")

# Compare standard errors
se_standard = cph_standard.summary.loc['fin_strain_binary', 'se(coef)']
se_clustered = cph_clustered.summary.loc['fin_strain_binary', 'se(coef)']
print(f"\n  SE comparison: Standard = {se_standard:.4f}, Clustered = {se_clustered:.4f}")
print(f"  SE ratio (clustered/standard): {se_clustered/se_standard:.3f}")

# Create output table
cluster_results = pd.DataFrame([
    {
        'Model': 'Standard (Model 3b)',
        'N': n_individuals,
        'Clusters': 'None',
        'HR': f"{hr_standard:.2f}",
        '95% CI': f"{ci_low_std:.2f}-{ci_high_std:.2f}",
        'P': f"{p_standard:.3f}",
        'SE(coef)': f"{se_standard:.4f}"
    },
    {
        'Model': 'Clustered by Household',
        'N': n_individuals,
        'Clusters': str(n_households),
        'HR': f"{hr_clustered:.2f}",
        '95% CI': f"{ci_low_cl:.2f}-{ci_high_cl:.2f}",
        'P': f"{p_clustered:.3f}",
        'SE(coef)': f"{se_clustered:.4f}"
    }
])

cluster_results.to_csv(TABLES_DIR / "etable_household_clustering_v8.csv", index=False)
print(f"\nSaved: {TABLES_DIR / 'etable_household_clustering_v8.csv'}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("Analysis complete")
print("="*80)

print(f"""
Standardized risks (eTable 12):
  Complete-case covariate-adjusted mortality at 5 and 10 years, stratified by age

Household clustering (eTable 13):
  {n_households} households, {multi_person_hh} with >1 person
  Standard HR: {hr_standard:.2f} ({ci_low_std:.2f}-{ci_high_std:.2f})
  Clustered HR: {hr_clustered:.2f} ({ci_low_cl:.2f}-{ci_high_cl:.2f})
  SE ratio: {se_clustered/se_standard:.3f}
""")
