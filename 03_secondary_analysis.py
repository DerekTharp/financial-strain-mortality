"""
Selection comparison (table S1, panel A), distribution of the financial strain item
(table S2), participant flow counts (figure S1), and the data file for the Fine-Gray models.

Writes selection_bias.csv, strain_item_distribution.csv and participant_flow.csv to
output/tables/, and output/cardiac_competing_risks.dta for
07_competing_risks_heart_disease.do.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from model_specs import MODEL3B_COVARS, CARDIAC_COVARS

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "randhrs1992_2022v1_STATA"
OUTPUT_DIR = PROJECT_DIR / "output"
TABLES_DIR = OUTPUT_DIR / "tables"
TABLES_DIR.mkdir(exist_ok=True, parents=True)

df = pd.read_csv(OUTPUT_DIR / 'analytic_sample.csv')

# Included versus excluded Wave 8 respondents

rand_file = DATA_DIR / "randhrs1992_2022v1.dta"
wave8_vars = ['hhidpn', 'r8iwstat', 'ragender', 'r8agey_b', 'raedyrs',
              'h8itot', 'r8hibpe', 'r8diabe', 'r8hearte']
df_full = pd.read_stata(rand_file, convert_categoricals=False,
                        columns=wave8_vars + ['r8lbfinprb'])

df_wave8 = df_full[wave8_vars].copy()
df_wave8 = df_wave8[df_wave8['r8iwstat'] == 1]  # Responded in wave 8

df_wave8['in_analytic'] = df_wave8['hhidpn'].isin(df['hhidpn']).astype(int)

selection_comparison = []

def compare_groups(var_name, label, df_comp):
    included = df_comp[df_comp['in_analytic'] == 1][var_name].dropna()
    excluded = df_comp[df_comp['in_analytic'] == 0][var_name].dropna()

    inc_mean = included.mean()
    exc_mean = excluded.mean()

    # Pooled SD for SMD
    pooled_sd = np.sqrt(((len(included)-1)*included.std()**2 + (len(excluded)-1)*excluded.std()**2) /
                        (len(included) + len(excluded) - 2))
    smd = (inc_mean - exc_mean) / pooled_sd if pooled_sd > 0 else 0

    return {
        'Variable': label,
        'Included_N': len(included),
        'Included_Mean': inc_mean,
        'Excluded_N': len(excluded),
        'Excluded_Mean': exc_mean,
        'SMD': smd
    }

selection_comparison.append(compare_groups('r8agey_b', 'Age, years', df_wave8))
df_wave8['female'] = (df_wave8['ragender'] == 2).astype(float)
selection_comparison.append(compare_groups('female', 'Female, %', df_wave8))
selection_comparison.append(compare_groups('raedyrs', 'Education, years', df_wave8))
df_wave8['log_income'] = np.log1p(df_wave8['h8itot'].clip(lower=0))
selection_comparison.append(compare_groups('log_income', 'Log household income', df_wave8))
selection_comparison.append(compare_groups('r8hibpe', 'Hypertension, %', df_wave8))
selection_comparison.append(compare_groups('r8diabe', 'Diabetes, %', df_wave8))
selection_comparison.append(compare_groups('r8hearte', 'Heart disease, %', df_wave8))

selection_df = pd.DataFrame(selection_comparison)
selection_df.to_csv(TABLES_DIR / 'selection_bias.csv', index=False)

# Participant flow

flow_data = {
    'step': [],
    'description': [],
    'n': [],
    'excluded': [],
    'exclusion_reason': [],
    'excluded_heart': [],
    'excluded_stroke_only': [],
    'excluded_other': [],
}

# Step 1: Wave 8 respondents
n_wave8 = len(df_wave8)
flow_data['step'].append(1)
flow_data['description'].append('HRS 2006 (Wave 8) respondents')
flow_data['n'].append(n_wave8)
flow_data['excluded'].append(0)
flow_data['exclusion_reason'].append('')
flow_data['excluded_heart'].append('')
flow_data['excluded_stroke_only'].append('')
flow_data['excluded_other'].append('')

# Step 2: valid financial strain response, with the filter used in 01_primary_analysis.py
wave8_valid_strain = (df_full['r8iwstat'] == 1) & df_full['r8lbfinprb'].notna() & df_full['r8lbfinprb'].isin([1, 2, 3, 4])
n_valid_exposure = wave8_valid_strain.sum()
n_age_filtered = len(df)  # Analytic sample is already filtered to age >= 50

# Distribution of the item among respondents with a valid response (table S2)
item_counts = df_full.loc[wave8_valid_strain, 'r8lbfinprb'].astype(int).value_counts().sort_index()
pd.DataFrame({'r8lbfinprb': item_counts.index, 'n': item_counts.values}).to_csv(
    TABLES_DIR / 'strain_item_distribution.csv', index=False)

flow_data['step'].append(2)
flow_data['description'].append('Returned LBQ with valid financial strain response')
flow_data['n'].append(n_valid_exposure)
flow_data['excluded'].append(n_wave8 - n_valid_exposure)
flow_data['exclusion_reason'].append('No valid LBQ financial strain response')
flow_data['excluded_heart'].append('')
flow_data['excluded_stroke_only'].append('')
flow_data['excluded_other'].append('')

flow_data['step'].append(3)
flow_data['description'].append('Age ≥50 years at baseline')
flow_data['n'].append(n_age_filtered)
flow_data['excluded'].append(n_valid_exposure - n_age_filtered)
flow_data['exclusion_reason'].append('Age <50 years')
flow_data['excluded_heart'].append('')
flow_data['excluded_stroke_only'].append('')
flow_data['excluded_other'].append('')

# Step 4: complete covariates for Model 3b
n_complete = df[['fin_strain_binary', 'followup_years', 'died'] + MODEL3B_COVARS].dropna().shape[0]

flow_data['step'].append(4)
flow_data['description'].append('Complete covariate data (mortality analysis)')
flow_data['n'].append(n_complete)
flow_data['excluded'].append(n_age_filtered - n_complete)
flow_data['exclusion_reason'].append('Missing covariates')
flow_data['excluded_heart'].append('')
flow_data['excluded_stroke_only'].append('')
flow_data['excluded_other'].append('')

# Step 5: cardiac analytic sample, as in 01_primary_analysis.py: free of heart disease and
# stroke, positive cardiac follow-up, complete cardiac covariates
cvd_free_mask = (df['baseline_heart'] == 0) & (df['baseline_stroke'] == 0)
df_cardiac_flow = df[cvd_free_mask].copy()
df_cardiac_flow = df_cardiac_flow[df_cardiac_flow['cardiac_followup_years'] > 0]
cardiac_cols = ['cardiac_followup_years', 'incident_heart', 'fin_strain_binary'] + CARDIAC_COVARS
n_cardiac_complete = df_cardiac_flow[cardiac_cols].dropna().shape[0]

# Compute heart disease / stroke breakdown from complete-covariate sample (Step 4)
df_step4 = df[['fin_strain_binary', 'followup_years', 'died'] + MODEL3B_COVARS].dropna()
df_step4 = df.loc[df_step4.index]  # align to full df columns
n_heart_in_step4 = int((df_step4['baseline_heart'] == 1).sum())
n_stroke_only_in_step4 = int(((df_step4['baseline_stroke'] == 1) & (df_step4['baseline_heart'] == 0)).sum())
n_cvd_total = n_heart_in_step4 + n_stroke_only_in_step4
n_other_cardiac_excl = (n_complete - n_cardiac_complete) - n_cvd_total

flow_data['step'].append(5)
flow_data['description'].append('Free of heart disease or stroke at baseline (cardiac analysis)')
flow_data['n'].append(n_cardiac_complete)
flow_data['excluded'].append(n_complete - n_cardiac_complete)
flow_data['exclusion_reason'].append('Baseline heart disease or stroke')
flow_data['excluded_heart'].append(n_heart_in_step4)
flow_data['excluded_stroke_only'].append(n_stroke_only_in_step4)
flow_data['excluded_other'].append(n_other_cardiac_excl)

flow_df = pd.DataFrame(flow_data)
flow_df.to_csv(TABLES_DIR / 'participant_flow.csv', index=False)
print(flow_df[['step', 'description', 'n', 'excluded']].to_string(index=False))

# Data for the Fine-Gray models: participants free of heart disease and stroke at baseline

cardiac_stata_vars = ['hhidpn', 'cardiac_followup_years', 'cardiac_fu_fg', 'incident_heart', 'died',
                      'fin_strain_binary', 'cardiac_event_type'] + CARDIAC_COVARS
df_cardiac_full = df[(df['baseline_heart'] == 0) & (df['baseline_stroke'] == 0)].copy()
df_cardiac_stata = df_cardiac_full[cardiac_stata_vars].dropna()

# cardiac_event_type: 0=censored, 1=heart disease, 2=death without HD
df_cardiac_stata.to_stata(OUTPUT_DIR / 'cardiac_competing_risks.dta', write_index=False)
print(f"Saved: cardiac_competing_risks.dta (N={len(df_cardiac_stata)})")
