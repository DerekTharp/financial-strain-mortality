"""
05_sensitivity_weighted.py
Supplementary sensitivity analyses.

Analyses performed:
1. Proportional hazards diagnostics (Schoenfeld residuals)
2. Age-specific HRs with 95% CIs (eTable 5)
3. Survey-weighted estimates with HRS psychosocial weights (eTable 11)
4. LBQ responder vs nonresponder comparison for selection bias (eTable 9)

Date: February 2026
"""

import pandas as pd
import numpy as np
import pyreadstat
from pathlib import Path
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
import json

# Set paths
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "randhrs1992_2022v1_STATA"
OUTPUT_DIR = PROJECT_DIR / "output"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR.mkdir(exist_ok=True, parents=True)
FIGURES_DIR.mkdir(exist_ok=True, parents=True)

print("="*80)
print("05_sensitivity_weighted: Supplementary sensitivity analyses")
print("="*80)

# ============================================================================
# LOAD ANALYTIC SAMPLE FROM V5/V6
# ============================================================================

print("\nLoading analytic sample...")
df = pd.read_csv(OUTPUT_DIR / 'analytic_sample_v5.csv')
print(f"Loaded N = {len(df)}")

# Model 3b covariates (primary model)
model3b_vars = ['age', 'female', 'race_nh_black', 'race_hispanic', 'race_nh_other',
                'education_yrs', 'married_partnered', 'current_smoker', 'bmi',
                'baseline_hypertension', 'baseline_diabetes', 'baseline_heart', 'baseline_stroke',
                'asinh_income', 'asinh_wealth']

# ============================================================================
# PROPORTIONAL HAZARDS DIAGNOSTICS
# ============================================================================

print("\n" + "="*80)
print("Proportional hazards diagnostics (Schoenfeld residuals)")
print("="*80)

# Prepare data for primary model
model_vars = ['fin_strain_binary', 'followup_years', 'died'] + model3b_vars
df_model = df[model_vars].dropna().copy()

print(f"Complete cases for primary model: N = {len(df_model)}")

# Fit primary model
cph = CoxPHFitter(penalizer=1e-5)
cph.fit(df_model, duration_col='followup_years', event_col='died',
        formula='fin_strain_binary + ' + ' + '.join(model3b_vars))

# Proportional hazards test using lifelines
print("\nProportional Hazards Test (Schoenfeld Residuals):")
print("-" * 60)

try:
    ph_test = proportional_hazard_test(cph, df_model, time_transform='rank')
    ph_results = ph_test.summary

    # Save full results
    ph_results.to_csv(TABLES_DIR / 'etable_ph_diagnostics_v7.csv')
    print("Full results saved to etable_ph_diagnostics_v7.csv")

    # Print key results
    print(f"\n{'Covariate':<25} {'Test Statistic':>15} {'P-value':>12}")
    print("-" * 55)
    for idx in ph_results.index:
        test_stat = ph_results.loc[idx, 'test_statistic']
        p_val = ph_results.loc[idx, 'p']
        flag = " *" if p_val < 0.05 else ""
        print(f"{str(idx):<25} {test_stat:>15.3f} {p_val:>12.4f}{flag}")

    print("\n* P < 0.05 indicates potential violation of proportional hazards")

    # Extract exposure row directly
    if 'fin_strain_binary' not in ph_results.index:
        raise KeyError("fin_strain_binary not found in PH test results")

    ph_summary = {
        'exposure_test_stat': float(ph_results.loc['fin_strain_binary', 'test_statistic']),
        'exposure_p': float(ph_results.loc['fin_strain_binary', 'p']),
        'n_violations_p05': int((ph_results['p'] < 0.05).sum()),
        'note': 'Based on Schoenfeld residuals with rank time transform'
    }

except Exception as e:
    print(f"Warning: Could not run standard PH test: {e}")
    print("Attempting alternative approach...")

    # Alternative: test time-varying effect manually
    df_model['log_time'] = np.log(df_model['followup_years'] + 0.01)
    df_model['strain_x_logtime'] = df_model['fin_strain_binary'] * df_model['log_time']

    cph_time = CoxPHFitter(penalizer=1e-5)
    time_formula = 'fin_strain_binary + strain_x_logtime + ' + ' + '.join(model3b_vars)
    cph_time.fit(df_model, duration_col='followup_years', event_col='died', formula=time_formula)

    time_int_p = cph_time.summary.loc['strain_x_logtime', 'p']
    print(f"\nTime-varying effect test (strain × log(time)):")
    print(f"  Interaction P = {time_int_p:.4f}")

    ph_summary = {
        'exposure_time_interaction_p': float(time_int_p),
        'interpretation': 'Non-significant suggests PH assumption reasonable for exposure',
        'note': 'Based on strain × log(time) interaction test'
    }

# ============================================================================
# AGE-SPECIFIC HRs WITH CONFIDENCE INTERVALS
# ============================================================================

print("\n" + "="*80)
print("Age-specific HRs with 95% confidence intervals")
print("="*80)

# Fit continuous age interaction model
df_int = df[['fin_strain_binary', 'followup_years', 'died', 'age'] +
            [v for v in model3b_vars if v != 'age']].dropna().copy()

# Center age at 65
df_int['age_centered'] = df_int['age'] - 65
df_int['strain_x_age_centered'] = df_int['fin_strain_binary'] * df_int['age_centered']

cph_int = CoxPHFitter(penalizer=1e-5)
int_formula = 'fin_strain_binary + age_centered + strain_x_age_centered + ' + \
              ' + '.join([v for v in model3b_vars if v != 'age'])
cph_int.fit(df_int, duration_col='followup_years', event_col='died', formula=int_formula)

# Get coefficients and variance-covariance matrix
beta_strain = cph_int.params_['fin_strain_binary']
beta_int = cph_int.params_['strain_x_age_centered']

# Get variance-covariance matrix
vcov = cph_int.variance_matrix_

# Find indices for strain and interaction terms
param_names = list(cph_int.params_.index)
idx_strain = param_names.index('fin_strain_binary')
idx_int = param_names.index('strain_x_age_centered')

var_strain = vcov.iloc[idx_strain, idx_strain]
var_int = vcov.iloc[idx_int, idx_int]
cov_strain_int = vcov.iloc[idx_strain, idx_int]

# Calculate age-specific HRs and CIs
ages = [55, 60, 65, 70, 75, 80]
age_specific_results = []

print(f"\n{'Age':>6} {'HR':>8} {'95% CI':>18} {'SE(log HR)':>12}")
print("-" * 50)

for age in ages:
    age_c = age - 65  # centered

    # Log HR at this age
    log_hr = beta_strain + beta_int * age_c

    # Variance of log HR: Var(b1 + b2*age) = Var(b1) + age^2*Var(b2) + 2*age*Cov(b1,b2)
    var_log_hr = var_strain + (age_c**2) * var_int + 2 * age_c * cov_strain_int
    se_log_hr = np.sqrt(var_log_hr)

    # HR and CI
    hr = np.exp(log_hr)
    ci_lower = np.exp(log_hr - 1.96 * se_log_hr)
    ci_upper = np.exp(log_hr + 1.96 * se_log_hr)

    age_specific_results.append({
        'Age': age,
        'HR': round(hr, 2),
        'CI_Lower': round(ci_lower, 2),
        'CI_Upper': round(ci_upper, 2),
        'SE_logHR': round(se_log_hr, 4)
    })

    print(f"{age:>6} {hr:>8.2f} {ci_lower:>7.2f} - {ci_upper:<7.2f} {se_log_hr:>12.4f}")

# Get interaction P-value
int_p = cph_int.summary.loc['strain_x_age_centered', 'p']
print(f"\nContinuous age interaction P = {int_p:.4f}")

# Save results
age_hr_df = pd.DataFrame(age_specific_results)
age_hr_df['95% CI'] = age_hr_df.apply(lambda x: f"{x['CI_Lower']:.2f}-{x['CI_Upper']:.2f}", axis=1)
age_hr_df.to_csv(TABLES_DIR / 'etable5_age_specific_hrs_v7.csv', index=False)
print("\nSaved: etable5_age_specific_hrs_v7.csv")

# ============================================================================
# LBQ-WEIGHTED ANALYSES
# ============================================================================

print("\n" + "="*80)
print("LBQ-weighted analyses (probability weights only; PSU/strata not available)")
print("="*80)

# Check weight availability
weight_col = 'lb_weight'
df_weighted = df[model_vars + [weight_col]].dropna().copy()
df_weighted = df_weighted[df_weighted[weight_col] > 0]

print(f"Sample with valid LBQ weights: N = {len(df_weighted)}")
print(f"Weight range: {df_weighted[weight_col].min():.2f} to {df_weighted[weight_col].max():.2f}")
print(f"Weight mean: {df_weighted[weight_col].mean():.2f}")

# Fit weighted model
cph_weighted = CoxPHFitter(penalizer=1e-5)
cph_weighted.fit(df_weighted, duration_col='followup_years', event_col='died',
                 formula='fin_strain_binary + ' + ' + '.join(model3b_vars),
                 weights_col=weight_col, robust=True)

hr_weighted = np.exp(cph_weighted.params_['fin_strain_binary'])
ci_weighted = np.exp(cph_weighted.confidence_intervals_.loc['fin_strain_binary'])
p_weighted = cph_weighted.summary.loc['fin_strain_binary', 'p']

print(f"\nWeighted Model 3b Results:")
print(f"  HR = {hr_weighted:.2f} (95% CI: {ci_weighted['95% lower-bound']:.2f}-{ci_weighted['95% upper-bound']:.2f})")
print(f"  P = {p_weighted:.4f}")

# Compare to unweighted
cph_unweighted = CoxPHFitter(penalizer=1e-5)
cph_unweighted.fit(df_weighted, duration_col='followup_years', event_col='died',
                   formula='fin_strain_binary + ' + ' + '.join(model3b_vars))

hr_unweighted = np.exp(cph_unweighted.params_['fin_strain_binary'])
ci_unweighted = np.exp(cph_unweighted.confidence_intervals_.loc['fin_strain_binary'])
p_unweighted = cph_unweighted.summary.loc['fin_strain_binary', 'p']

print(f"\nUnweighted (same sample):")
print(f"  HR = {hr_unweighted:.2f} (95% CI: {ci_unweighted['95% lower-bound']:.2f}-{ci_unweighted['95% upper-bound']:.2f})")
print(f"  P = {p_unweighted:.4f}")

# Age-stratified weighted
print("\nAge-stratified weighted estimates:")
weighted_age_results = []

for age_grp, age_label in [('<65', '<65 years'), ('>=65', '>=65 years')]:
    if age_grp == '<65':
        df_age = df_weighted[df_weighted['age'] < 65].copy()
    else:
        df_age = df_weighted[df_weighted['age'] >= 65].copy()

    if len(df_age) > 100 and df_age['fin_strain_binary'].sum() > 20:
        cph_age = CoxPHFitter(penalizer=1e-5)
        cph_age.fit(df_age, duration_col='followup_years', event_col='died',
                    formula='fin_strain_binary + ' + ' + '.join(model3b_vars),
                    weights_col=weight_col, robust=True)

        hr = np.exp(cph_age.params_['fin_strain_binary'])
        ci = np.exp(cph_age.confidence_intervals_.loc['fin_strain_binary'])
        p = cph_age.summary.loc['fin_strain_binary', 'p']

        weighted_age_results.append({
            'Age_Group': age_label,
            'N': len(df_age),
            'Deaths': int(df_age['died'].sum()),
            'HR': round(hr, 2),
            'CI_Lower': round(ci['95% lower-bound'], 2),
            'CI_Upper': round(ci['95% upper-bound'], 2),
            'P': round(p, 4)
        })

        print(f"  {age_label}: HR = {hr:.2f} ({ci['95% lower-bound']:.2f}-{ci['95% upper-bound']:.2f}), P = {p:.4f}")

# Save weighted results
weighted_results_df = pd.DataFrame([
    {'Model': 'Model 3b (LBQ-weighted)', 'N': len(df_weighted), 'Deaths': int(df_weighted['died'].sum()),
     'HR': round(hr_weighted, 2), 'CI_Lower': round(ci_weighted['95% lower-bound'], 2),
     'CI_Upper': round(ci_weighted['95% upper-bound'], 2), 'P': round(p_weighted, 4)},
    {'Model': 'Model 3b (unweighted, same sample)', 'N': len(df_weighted), 'Deaths': int(df_weighted['died'].sum()),
     'HR': round(hr_unweighted, 2), 'CI_Lower': round(ci_unweighted['95% lower-bound'], 2),
     'CI_Upper': round(ci_unweighted['95% upper-bound'], 2), 'P': round(p_unweighted, 4)}
] + weighted_age_results)

weighted_results_df.to_csv(TABLES_DIR / 'etable_weighted_estimates_v7.csv', index=False)
print("\nSaved: etable_weighted_estimates_v7.csv")

# ============================================================================
# SELECTION BIAS ASSESSMENT
# ============================================================================

print("\n" + "="*80)
print("LBQ responder vs nonresponder comparison")
print("="*80)

HRS_FILE = DATA_DIR / "randhrs1992_2022v1.dta"

print("\nLoading full HRS data for selection comparison...")
hrs_full, _ = pyreadstat.read_dta(str(HRS_FILE),
                                  usecols=[
                                      'hhidpn',
                                      'r8agey_e', 'ragender', 'raracem', 'rahispan',
                                      'raedyrs', 'h8itot', 'h8atotb',
                                      'r8hibpe', 'r8diabe', 'r8hearte', 'r8stroke',
                                      'r8lbfinprb', 'r8lbwgtr', 'r8iwmid'
                                  ])

# Filter to Wave 8 respondents
hrs_full = hrs_full[hrs_full['r8iwmid'].notna()].copy()
print(f"Wave 8 respondents: N = {len(hrs_full)}")

# Identify LBQ responders (valid financial strain response)
# Note: r8lbwgtr is defined only for LBQ responders, not all assigned.
# We compare LBQ responders vs other Wave 8 respondents.
hrs_full['lbq_responded'] = hrs_full['r8lbfinprb'].notna() & hrs_full['r8lbfinprb'].isin([1,2,3,4])
responders = hrs_full[hrs_full['lbq_responded']].copy()
nonresponders = hrs_full[~hrs_full['lbq_responded']].copy()

print(f"  LBQ responders: {len(responders)}")
print(f"  Other Wave 8 respondents: {len(nonresponders)}")

# Calculate SMDs for key variables
def calculate_smd(var, df1, df2):
    """Calculate standardized mean difference."""
    v1 = df1[var].dropna()
    v2 = df2[var].dropna()
    if len(v1) == 0 or len(v2) == 0:
        return np.nan
    m1, m2 = v1.mean(), v2.mean()
    s1, s2 = v1.std(), v2.std()
    if s1 == 0 and s2 == 0:
        return 0
    pooled_s = np.sqrt((s1**2 + s2**2) / 2)
    if pooled_s == 0:
        return 0
    return (m1 - m2) / pooled_s

# Recode variables for meaningful comparison
for grp in [responders, nonresponders]:
    grp['female'] = (grp['ragender'] == 2).astype(float)
    grp['asinh_income'] = np.arcsinh(grp['h8itot'].clip(lower=0))
    grp['asinh_wealth'] = np.arcsinh(grp['h8atotb'])

selection_comparison = []
comparison_vars = [
    ('r8agey_e', 'Age, years'),
    ('female', 'Female, %'),
    ('raedyrs', 'Education, years'),
    ('asinh_income', 'Household income (asinh)'),
    ('asinh_wealth', 'Total wealth (asinh)'),
    ('r8hibpe', 'Hypertension, %'),
    ('r8diabe', 'Diabetes, %'),
    ('r8hearte', 'Heart disease, %'),
    ('r8stroke', 'Stroke, %')
]

print(f"\n{'Characteristic':<30} {'LBQ Resp.':>12} {'Other W8':>12} {'SMD':>8}")
print("-" * 65)

for var, label in comparison_vars:
    resp_mean = responders[var].mean()
    nonresp_mean = nonresponders[var].mean()
    smd = calculate_smd(var, responders, nonresponders)

    selection_comparison.append({
        'Characteristic': label,
        'LBQ_Responders': round(resp_mean, 2),
        'Other_Wave8': round(nonresp_mean, 2),
        'SMD': round(smd, 3)
    })

    print(f"{label:<30} {resp_mean:>12.2f} {nonresp_mean:>12.2f} {smd:>8.3f}")

selection_df = pd.DataFrame(selection_comparison)
selection_df['N_LBQ_Responders'] = len(responders)
selection_df['N_Other_Wave8'] = len(nonresponders)
selection_df.to_csv(TABLES_DIR / 'etable_lbq_selection_v7.csv', index=False)
print("\nSaved: etable_lbq_selection_v7.csv")

# ============================================================================
# SAVE RESULTS MANIFEST
# ============================================================================

print("\n" + "="*80)
print("Saving results manifest")
print("="*80)

v7_manifest = {
    'version': 7,
    'date': '2026-02',
    'new_analyses': [
        'Proportional hazards diagnostics (Schoenfeld residuals)',
        'Age-specific HRs with 95% CIs',
        'Survey-weighted estimates',
        'LBQ responder vs nonresponder comparison'
    ],
    'proportional_hazards': ph_summary,
    'age_specific_hrs': age_specific_results,
    'continuous_age_interaction_p': float(int_p),
    'weighted_estimates': {
        'model3b': {
            'hr': float(hr_weighted),
            'ci_lower': float(ci_weighted['95% lower-bound']),
            'ci_upper': float(ci_weighted['95% upper-bound']),
            'p': float(p_weighted)
        },
        'n_with_weights': int(len(df_weighted))
    },
    'selection_bias': {
        'lbq_responded': int(len(responders)),
        'other_wave8': int(len(nonresponders)),
        'max_smd': float(max(abs(x['SMD']) for x in selection_comparison if not np.isnan(x['SMD'])))
    }
}

with open(OUTPUT_DIR / 'results_manifest_v7.json', 'w') as f:
    json.dump(v7_manifest, f, indent=2, default=str)

print("Saved: results_manifest_v7.json")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("Analysis complete")
print("="*80)

max_smd = max(abs(x['SMD']) for x in selection_comparison if not np.isnan(x['SMD']))

print(f"""
Tables created:
  - etable_ph_diagnostics_v7.csv (proportional hazards tests)
  - etable5_age_specific_hrs_v7.csv (age-specific HRs with CIs)
  - etable_weighted_estimates_v7.csv (LBQ-weighted results)
  - etable_lbq_selection_v7.csv (LBQ responder comparison)

Key findings:
  PH test: exposure P = {ph_summary.get('exposure_p', ph_summary.get('exposure_time_interaction_p', 'N/A'))}
  Age-specific HRs: age 55 HR={age_specific_results[0]['HR']}, age 65 HR={age_specific_results[2]['HR']}, age 80 HR={age_specific_results[5]['HR']}
  Continuous interaction P = {int_p:.4f}
  Weighted HR = {hr_weighted:.2f} vs unweighted HR = {hr_unweighted:.2f}
  LBQ selection max SMD = {max_smd:.3f}
""")
