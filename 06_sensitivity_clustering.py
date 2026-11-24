"""
Covariate-standardised mortality risks within age strata (table S9) and household-clustered
standard errors (table S10).

Writes standardized_risks.csv and household_clustering.csv to output/tables/.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from lifelines import CoxPHFitter

from model_specs import EXPOSURE_VAR, MODEL3B_COVARS, PENALIZER

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "output"
TABLES_DIR = OUTPUT_DIR / "tables"
TABLES_DIR.mkdir(exist_ok=True, parents=True)

df = pd.read_csv(OUTPUT_DIR / "analytic_sample.csv")

covariates = [EXPOSURE_VAR] + MODEL3B_COVARS

# Covariate-standardised risks by age stratum

model_vars = covariates + ['followup_years', 'died']

df_mort = df[model_vars].dropna().copy()
df_mort = df_mort[df_mort['followup_years'] > 0]

df_mort['age_under65'] = (df_mort['age'] < 65).astype(int)

def compute_standardized_survival(df_stratum, time_points=[5, 10]):
    """
    Compute covariate-standardized survival via marginal standardization (g-computation).
    Fits Cox within stratum, predicts under counterfactual exposure, averages across individuals.
    Age is retained as a covariate for within-stratum adjustment.
    """
    cph = CoxPHFitter(penalizer=PENALIZER)
    cph.fit(df_stratum[covariates + ['followup_years', 'died']],
            duration_col='followup_years', event_col='died')

    results = {}

    for t in time_points:
        for exposure in [0, 1]:
            df_cf = df_stratum[covariates].copy()
            df_cf['fin_strain_binary'] = exposure

            surv_funcs = cph.predict_survival_function(df_cf, times=[t])
            # surv_funcs: index=times, columns=individuals (lifelines 0.30)
            mean_surv = float(surv_funcs.loc[t].mean())
            mean_mort = 1 - mean_surv

            exposure_label = 'high_strain' if exposure == 1 else 'low_strain'
            results[f'mortality_{t}yr_{exposure_label}'] = mean_mort * 100

    # Calculate risk differences
    for t in time_points:
        high_key = f'mortality_{t}yr_high_strain'
        low_key = f'mortality_{t}yr_low_strain'
        if high_key in results and low_key in results:
            results[f'risk_diff_{t}yr'] = results[high_key] - results[low_key]

    return results

df_under65 = df_mort[df_mort['age_under65'] == 1].copy()
results_under65 = compute_standardized_survival(df_under65, time_points=[5, 10])

df_65plus = df_mort[df_mort['age_under65'] == 0].copy()
results_65plus = compute_standardized_survival(df_65plus, time_points=[5, 10])

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
df_std_risks.to_csv(TABLES_DIR / "standardized_risks.csv", index=False)
print("Saved: standardized_risks.csv")

# Household clustering

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

n_individuals = len(df_mort_cluster)

# Model 3b with model-based standard errors
cph_standard = CoxPHFitter(penalizer=PENALIZER)
cph_standard.fit(df_mort_cluster[covariates + ['followup_years', 'died']],
                 duration_col='followup_years', event_col='died')

hr_standard = np.exp(cph_standard.params_['fin_strain_binary'])
ci_low_std = np.exp(cph_standard.confidence_intervals_.loc['fin_strain_binary', '95% lower-bound'])
ci_high_std = np.exp(cph_standard.confidence_intervals_.loc['fin_strain_binary', '95% upper-bound'])
p_standard = cph_standard.summary.loc['fin_strain_binary', 'p']

# Model 3b with standard errors clustered on household
cph_clustered = CoxPHFitter(penalizer=PENALIZER)
cph_clustered.fit(df_mort_cluster[covariates + ['followup_years', 'died', 'household_id']],
                  duration_col='followup_years', event_col='died',
                  cluster_col='household_id',
                  robust=True)

hr_clustered = np.exp(cph_clustered.params_['fin_strain_binary'])
ci_low_cl = np.exp(cph_clustered.confidence_intervals_.loc['fin_strain_binary', '95% lower-bound'])
ci_high_cl = np.exp(cph_clustered.confidence_intervals_.loc['fin_strain_binary', '95% upper-bound'])
p_clustered = cph_clustered.summary.loc['fin_strain_binary', 'p']

cluster_results = pd.DataFrame([
    {
        'Model': 'Standard (Model 3b)',
        'N': n_individuals,
        'HR': f"{hr_standard:.2f}",
        '95% CI': f"{ci_low_std:.2f}-{ci_high_std:.2f}",
        'P': f"{p_standard:.3f}",
    },
    {
        'Model': 'Clustered by Household',
        'N': n_individuals,
        'HR': f"{hr_clustered:.2f}",
        '95% CI': f"{ci_low_cl:.2f}-{ci_high_cl:.2f}",
        'P': f"{p_clustered:.3f}",
    }
])

cluster_results.to_csv(TABLES_DIR / "household_clustering.csv", index=False)
print("Saved: household_clustering.csv")
