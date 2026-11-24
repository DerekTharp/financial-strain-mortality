"""
Proportional hazards diagnostics (table S8), unweighted age-specific hazard ratios from the
continuous age interaction (table S15), and the comparison of Leave-Behind Questionnaire
responders with other Wave 8 respondents (table S1, panel B).

Writes ph_diagnostics.csv, age_specific_hrs.csv and lbq_selection.csv to output/tables/.
"""

import pandas as pd
import numpy as np
import pyreadstat
from pathlib import Path
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test

from model_specs import MODEL3B_COVARS, PENALIZER

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "randhrs1992_2022v1_STATA"
OUTPUT_DIR = PROJECT_DIR / "output"
TABLES_DIR = OUTPUT_DIR / "tables"
TABLES_DIR.mkdir(exist_ok=True, parents=True)

df = pd.read_csv(OUTPUT_DIR / 'analytic_sample.csv')

# Proportional hazards test (Schoenfeld residuals, rank time transform), unweighted Model 3b

model_vars = ['fin_strain_binary', 'followup_years', 'died'] + MODEL3B_COVARS
df_model = df[model_vars].dropna().copy()

cph = CoxPHFitter(penalizer=PENALIZER)
cph.fit(df_model, duration_col='followup_years', event_col='died',
        formula='fin_strain_binary + ' + ' + '.join(MODEL3B_COVARS))

ph_test = proportional_hazard_test(cph, df_model, time_transform='rank')
ph_results = ph_test.summary
ph_results.to_csv(TABLES_DIR / 'ph_diagnostics.csv')
print("Saved: ph_diagnostics.csv")

# Age-specific hazard ratios from the continuous age interaction

df_int = df[['fin_strain_binary', 'followup_years', 'died', 'age'] +
            [v for v in MODEL3B_COVARS if v != 'age']].dropna().copy()

# Center age at 65
df_int['age_centered'] = df_int['age'] - 65
df_int['strain_x_age_centered'] = df_int['fin_strain_binary'] * df_int['age_centered']

cph_int = CoxPHFitter(penalizer=PENALIZER)
int_formula = 'fin_strain_binary + age_centered + strain_x_age_centered + ' + \
              ' + '.join([v for v in MODEL3B_COVARS if v != 'age'])
cph_int.fit(df_int, duration_col='followup_years', event_col='died', formula=int_formula)

beta_strain = cph_int.params_['fin_strain_binary']
beta_int = cph_int.params_['strain_x_age_centered']

vcov = cph_int.variance_matrix_

param_names = list(cph_int.params_.index)
idx_strain = param_names.index('fin_strain_binary')
idx_int = param_names.index('strain_x_age_centered')

var_strain = vcov.iloc[idx_strain, idx_strain]
var_int = vcov.iloc[idx_int, idx_int]
cov_strain_int = vcov.iloc[idx_strain, idx_int]

ages = [55, 60, 65, 70, 75, 80]
age_specific_results = []

for age in ages:
    age_c = age - 65  # centered

    log_hr = beta_strain + beta_int * age_c

    # Variance of log HR: Var(b1 + b2*age) = Var(b1) + age^2*Var(b2) + 2*age*Cov(b1,b2)
    var_log_hr = var_strain + (age_c**2) * var_int + 2 * age_c * cov_strain_int
    se_log_hr = np.sqrt(var_log_hr)

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

age_hr_df = pd.DataFrame(age_specific_results)
age_hr_df['95% CI'] = age_hr_df.apply(lambda x: f"{x['CI_Lower']:.2f}-{x['CI_Upper']:.2f}", axis=1)
age_hr_df.to_csv(TABLES_DIR / 'age_specific_hrs.csv', index=False)
print("Saved: age_specific_hrs.csv")

# Leave-Behind Questionnaire responders versus other Wave 8 respondents

HRS_FILE = DATA_DIR / "randhrs1992_2022v1.dta"

hrs_full, _ = pyreadstat.read_dta(str(HRS_FILE),
                                  usecols=[
                                      'hhidpn',
                                      'r8agey_e', 'ragender',
                                      'raedyrs', 'h8itot', 'h8atotb',
                                      'r8hibpe', 'r8diabe', 'r8hearte', 'r8stroke',
                                      'r8lbfinprb', 'r8iwmid'
                                  ])

# Filter to Wave 8 respondents
hrs_full = hrs_full[hrs_full['r8iwmid'].notna()].copy()

# Responders are Wave 8 respondents with a valid financial strain response
hrs_full['lbq_responded'] = hrs_full['r8lbfinprb'].notna() & hrs_full['r8lbfinprb'].isin([1,2,3,4])
responders = hrs_full[hrs_full['lbq_responded']].copy()
nonresponders = hrs_full[~hrs_full['lbq_responded']].copy()

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

for var, label in comparison_vars:
    resp_mean = responders[var].mean()
    nonresp_mean = nonresponders[var].mean()
    smd = calculate_smd(var, responders, nonresponders)

    selection_comparison.append({
        'Characteristic': label,
        'LBQ_Responders': round(resp_mean, 4),
        'Other_Wave8': round(nonresp_mean, 4),
        'SMD': round(smd, 3)
    })

selection_df = pd.DataFrame(selection_comparison)
selection_df['N_LBQ_Responders'] = len(responders)
selection_df['N_Other_Wave8'] = len(nonresponders)
selection_df.to_csv(TABLES_DIR / 'lbq_selection.csv', index=False)
print("Saved: lbq_selection.csv")
