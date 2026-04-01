"""
01_primary_analysis.py
Primary analysis script for Financial Strain and All-Cause Mortality study.

Analyses performed:
1. Data construction and analytic sample definition
2. Table 1: Baseline characteristics by financial strain status
3. Sequential Cox proportional hazards models (Models 1-4b)
4. Propensity score matching (1:1 nearest-neighbor, caliper 0.2 SD)
5. Dose-response analysis (4-level ordinal exposure)
6. E-value sensitivity analysis for unmeasured confounding
7. Age interaction modeling (binary and continuous)
8. Incident heart disease (cause-specific hazard)
9. Absolute risk estimation (Kaplan-Meier)
10. Cross-classification of financial strain by income quartile

Date: February 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path
from lifelines import CoxPHFitter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.special import logit as scipy_logit
import json
import pyreadstat

# Set paths
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "randhrs1992_2022v1_STATA"
OUTPUT_DIR = PROJECT_DIR / "output"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR.mkdir(exist_ok=True, parents=True)
FIGURES_DIR.mkdir(exist_ok=True, parents=True)

print("="*80)
print("FINANCIAL STRAIN AND ALL-CAUSE MORTALITY - PRIMARY ANALYSIS")
print("="*80)

# ============================================================================
# STEP 1: LOAD AND PREPARE DATA WITH ALL FIXES
# ============================================================================

print("\n" + "="*80)
print("STEP 1: Creating analytic sample")
print("="*80)

HRS_FILE = DATA_DIR / "randhrs1992_2022v1.dta"

print("\nLoading HRS data...")
hrs_data, _ = pyreadstat.read_dta(str(HRS_FILE),
                                      usecols=[
                                          'hhidpn',
                                          # Demographics
                                          'r8agey_e', 'ragender', 'raracem', 'rahispan',
                                          # SES
                                          'raedyrs', 'r8mstat', 'h8itot', 'h8atotb',
                                          # Health behaviors
                                          'r8smoken', 'r8smokev', 'r8bmi',
                                          # Baseline conditions
                                          'r8hibpe', 'r8diabe', 'r8hearte', 'r8stroke',
                                          # Psychosocial
                                          'r8cesd', 'r8shlt',
                                          # Financial strain (Leave-Behind)
                                          'r8lbfinprb',
                                          # Insurance and employment
                                          'r8covr', 'r8govmr', 'r8govmd',
                                          'r8henum', 'r8work', 'r8lbrf',
                                          'r8sayret',
                                          # Leave-behind weight
                                          'r8lbwgtr',
                                          # Mortality
                                          'raddate', 'r8iwmid',
                                          # Follow-up interviews and heart disease
                                          'r9hearte', 'r10hearte', 'r11hearte',
                                          'r12hearte', 'r13hearte', 'r14hearte',
                                          'r15hearte', 'r16hearte',
                                          # Interview dates for proper censoring
                                          'r9iwmid', 'r10iwmid', 'r11iwmid',
                                          'r12iwmid', 'r13iwmid', 'r14iwmid',
                                          'r15iwmid', 'r16iwmid',
                                      ])

print(f"Loaded {len(hrs_data)} observations")

# Helper functions that preserve NaN during recoding
def recode_binary_preserve_na(series, yes_value=1):
    """Recode binary variable preserving NaN."""
    result = pd.Series(np.nan, index=series.index)
    result[series == yes_value] = 1
    result[series.notna() & (series != yes_value)] = 0
    return result

def recode_condition_preserve_na(series):
    """Recode health condition preserving NaN."""
    result = pd.Series(np.nan, index=series.index)
    result[series == 0] = 0
    result[series == 1] = 1
    return result

df = hrs_data.copy()

# Filter to those with financial strain data
has_fin_strain = df['r8lbfinprb'].notna() & df['r8lbfinprb'].isin([1, 2, 3, 4])
print(f"With valid financial strain response: {has_fin_strain.sum()}")
df = df[has_fin_strain].copy()

# Restrict to age >= 50
df['age'] = df['r8agey_e']
age_eligible = df['age'] >= 50
print(f"Age >= 50 at baseline: {age_eligible.sum()}")
df = df[age_eligible].copy()

# ============================================================================
# CREATE ALL VARIABLES
# ============================================================================

# Financial strain variables
df['fin_strain_raw'] = df['r8lbfinprb']
df['fin_strain_binary'] = np.where(df['fin_strain_raw'].isin([3, 4]), 1, 0)
strain_map = {1: 'No strain', 2: 'Yes, not upsetting', 3: 'Somewhat upsetting', 4: 'Very upsetting'}
df['fin_strain_cat'] = df['fin_strain_raw'].map(strain_map)
df['fin_strain_ordinal'] = df['fin_strain_raw'] - 1  # 0, 1, 2, 3

# Demographics
df['female'] = np.where(df['ragender'] == 2, 1, 0)

# Race/ethnicity coding (mutually exclusive categories)
conditions = [
    df['rahispan'] == 1,
    (df['rahispan'] == 0) & (df['raracem'] == 1),
    (df['rahispan'] == 0) & (df['raracem'] == 2),
    (df['rahispan'] == 0) & (df['raracem'] == 3),
]
choices = ['Hispanic', 'NH White', 'NH Black', 'NH Other']
df['race_ethnicity'] = np.select(conditions, choices, default='Missing')
df.loc[df['race_ethnicity'] == 'Missing', 'race_ethnicity'] = np.nan

df['race_nh_white'] = np.where(df['race_ethnicity'] == 'NH White', 1,
                               np.where(pd.isna(df['race_ethnicity']), np.nan, 0))
df['race_nh_black'] = np.where(df['race_ethnicity'] == 'NH Black', 1,
                               np.where(pd.isna(df['race_ethnicity']), np.nan, 0))
df['race_hispanic'] = np.where(df['race_ethnicity'] == 'Hispanic', 1,
                               np.where(pd.isna(df['race_ethnicity']), np.nan, 0))
df['race_nh_other'] = np.where(df['race_ethnicity'] == 'NH Other', 1,
                               np.where(pd.isna(df['race_ethnicity']), np.nan, 0))

# Education
df['education_yrs'] = df['raedyrs']

# Marital status (separated classified as not partnered)
# RAND codes: 1=married, 2=married spouse absent, 3=partnered, 4=separated, 5=divorced, 6=divorced/not sep, 7=widowed, 8=never married
df['married_partnered'] = np.where(df['r8mstat'].isin([1, 2, 3]), 1,  # married, married spouse absent, partnered
                                   np.where(df['r8mstat'].isin([4, 5, 6, 7, 8]), 0,  # separated through never married
                                            np.nan))

# SES: inverse hyperbolic sine transform for income and wealth
df['income'] = df['h8itot']
df['wealth'] = df['h8atotb']
df['asinh_income'] = np.arcsinh(df['income'])
df['asinh_wealth'] = np.arcsinh(df['wealth'])

# Health behaviors - preserve NaN
df['current_smoker'] = recode_binary_preserve_na(df['r8smoken'], yes_value=1)
df['bmi'] = df['r8bmi']

# Baseline conditions - preserve NaN
df['baseline_hypertension'] = recode_condition_preserve_na(df['r8hibpe'])
df['baseline_diabetes'] = recode_condition_preserve_na(df['r8diabe'])
df['baseline_heart'] = recode_condition_preserve_na(df['r8hearte'])
df['baseline_stroke'] = recode_condition_preserve_na(df['r8stroke'])

# Psychosocial
df['cesd_score'] = df['r8cesd']
df['depressed'] = np.where(df['cesd_score'].isna(), np.nan,
                           np.where(df['cesd_score'] >= 3, 1, 0))
df['srh'] = df['r8shlt']
df['srh_fairpoor'] = np.where(df['srh'].isna(), np.nan,
                              np.where(df['srh'].isin([4, 5]), 1, 0))

# Survey weight
df['lb_weight'] = pd.to_numeric(df['r8lbwgtr'], errors='coerce')
df.loc[df['lb_weight'] <= 0, 'lb_weight'] = np.nan

# Insurance and employment variables
df['has_medicare'] = np.where(df['r8govmr'].isin([1]), 1,
                              np.where(df['r8govmr'].notna(), 0, np.nan))
df['has_medicaid'] = np.where(df['r8govmd'].isin([1]), 1,
                              np.where(df['r8govmd'].notna(), 0, np.nan))
df['has_employer_plan'] = np.where(df['r8covr'].isin([1]), 1,
                                   np.where(df['r8covr'].notna(), 0, np.nan))
# Any insurance: Medicare, Medicaid, or employer coverage
df['any_insurance'] = np.where(
    (df['has_medicare'] == 1) | (df['has_medicaid'] == 1) | (df['has_employer_plan'] == 1), 1,
    np.where(df['has_medicare'].notna() | df['has_medicaid'].notna() | df['has_employer_plan'].notna(), 0, np.nan)
)
# Employment
df['working'] = np.where(df['r8work'].isin([1]), 1,
                         np.where(df['r8work'].notna(), 0, np.nan))

# ============================================================================
# MORTALITY OUTCOME
# ============================================================================

df['death_date'] = df['raddate']
df['baseline_date'] = df['r8iwmid']

# End of study as Stata date
end_of_study = pd.to_datetime('2022-12-31').toordinal() - pd.to_datetime('1960-01-01').toordinal()

df['died'] = np.where(df['death_date'].notna(), 1, 0)

# Find last interview date for censoring
def get_last_interview_date(row):
    """Get last interview date for proper censoring."""
    for wave_col in ['r16iwmid', 'r15iwmid', 'r14iwmid', 'r13iwmid',
                     'r12iwmid', 'r11iwmid', 'r10iwmid', 'r9iwmid']:
        if pd.notna(row.get(wave_col)):
            return row[wave_col]
    return end_of_study

df['last_interview_date'] = df.apply(get_last_interview_date, axis=1)

df['end_date'] = np.where(df['died'] == 1, df['death_date'], df['last_interview_date'])
df['followup_years'] = (df['end_date'] - df['baseline_date']) / 365.25

# Remove invalid follow-up
valid_followup = df['followup_years'] > 0
df = df[valid_followup].copy()

# Ensure numeric dtypes (pyreadstat can produce object dtype from Stata)
numeric_cols = ['age', 'education_yrs', 'bmi', 'followup_years',
                'asinh_income', 'asinh_wealth', 'cesd_score']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Age at entry/exit for age-as-time-scale
df['age_at_entry'] = df['age']
df['age_at_exit'] = df['age'] + df['followup_years']

# Age group
df['age_group'] = np.where(df['age'] < 65, '<65 years', '>=65 years')
df['age_under65'] = np.where(df['age'] < 65, 1, 0)

# ============================================================================
# INCIDENT HEART DISEASE CENSORING
# ============================================================================

print("\nCreating incident heart disease with proper biennial censoring...")

df['baseline_cvd_free'] = (df['baseline_heart'] == 0) & (df['baseline_stroke'] == 0)

heart_waves = ['r9hearte', 'r10hearte', 'r11hearte', 'r12hearte',
               'r13hearte', 'r14hearte', 'r15hearte', 'r16hearte']
date_waves = ['r9iwmid', 'r10iwmid', 'r11iwmid', 'r12iwmid',
              'r13iwmid', 'r14iwmid', 'r15iwmid', 'r16iwmid']

def find_incident_heart_proper_censoring(row):
    """
    Find incident heart disease with proper censoring at last interview
    with outcome data; censored at last interview with heart disease assessment.
    """
    if not row['baseline_cvd_free']:
        return pd.Series({'incident_heart': np.nan,
                         'incident_heart_date': np.nan,
                         'last_heart_obs_date': np.nan})

    last_obs_date = row['baseline_date']  # Start with baseline

    for heart_var, date_var in zip(heart_waves, date_waves):
        wave_date = row.get(date_var)
        heart_status = row.get(heart_var)

        if pd.notna(wave_date) and pd.notna(heart_status):
            last_obs_date = wave_date  # Update last observation date

            if heart_status == 1:
                return pd.Series({'incident_heart': 1,
                                 'incident_heart_date': wave_date,
                                 'last_heart_obs_date': wave_date})

    return pd.Series({'incident_heart': 0,
                     'incident_heart_date': np.nan,
                     'last_heart_obs_date': last_obs_date})

print("  Finding incident heart disease cases with proper censoring...")
incident_data = df.apply(find_incident_heart_proper_censoring, axis=1)
df['incident_heart'] = incident_data['incident_heart']
df['incident_heart_date'] = incident_data['incident_heart_date']
df['last_heart_obs_date'] = incident_data['last_heart_obs_date']

# Cardiac follow-up time: to event OR to last interview with heart data OR death
# Censor at death if death occurs before last heart observation
df['cardiac_end_date'] = np.where(
    df['incident_heart'] == 1,
    df['incident_heart_date'],
    np.where(
        (df['died'] == 1) & (df['death_date'] < df['last_heart_obs_date']),
        df['death_date'],  # Censor at death if before last obs
        df['last_heart_obs_date']
    )
)

df['cardiac_followup_years'] = (df['cardiac_end_date'] - df['baseline_date']) / 365.25

# Competing event indicator for Fine-Gray
# 0 = censored alive, 1 = heart disease, 2 = death without heart disease
df['cardiac_event_type'] = np.where(
    df['incident_heart'] == 1, 1,
    np.where((df['died'] == 1) & (df['incident_heart'] == 0), 2, 0)
)

# ============================================================================
# MISSINGNESS DOCUMENTATION
# ============================================================================

print("\n" + "="*80)
print("STEP 2: Documenting missingness patterns")
print("="*80)

all_analysis_vars = ['fin_strain_binary', 'age', 'female',
                     'race_nh_black', 'race_hispanic', 'race_nh_other',
                     'education_yrs', 'married_partnered',
                     'current_smoker', 'bmi',
                     'baseline_hypertension', 'baseline_diabetes',
                     'baseline_heart', 'baseline_stroke',
                     'asinh_income', 'asinh_wealth',
                     'depressed', 'srh_fairpoor',
                     'followup_years', 'died']

missingness_table = []
for var in all_analysis_vars:
    n_missing = df[var].isna().sum()
    pct_missing = n_missing / len(df) * 100
    missingness_table.append({
        'Variable': var,
        'N_Missing': n_missing,
        'Pct_Missing': f"{pct_missing:.1f}%"
    })

missingness_df = pd.DataFrame(missingness_table)
missingness_df.to_csv(TABLES_DIR / 'etable_missingness.csv', index=False)
print("  Saved: etable_missingness.csv")

# Save analytic sample for use by other scripts
df.to_csv(OUTPUT_DIR / 'analytic_sample_v5.csv', index=False)
print("  Saved: analytic_sample_v5.csv")
print(f"  Total N: {len(df)}")
print(f"  Variables with missingness:")
for _, row in missingness_df[missingness_df['N_Missing'] > 0].iterrows():
    print(f"    {row['Variable']}: {row['N_Missing']} ({row['Pct_Missing']})")

# ============================================================================
# TABLE 1: BASELINE CHARACTERISTICS
# ============================================================================

print("\n" + "="*80)
print("TABLE 1: Baseline Characteristics by Financial Strain Status")
print("="*80)

def calculate_smd(var, df1, df2):
    """Calculate standardized mean difference."""
    if df1[var].isna().all() or df2[var].isna().all():
        return np.nan
    m1, m2 = df1[var].dropna().mean(), df2[var].dropna().mean()
    s1, s2 = df1[var].dropna().std(), df2[var].dropna().std()
    if s1 == 0 and s2 == 0:
        return 0
    pooled_s = np.sqrt((s1**2 + s2**2) / 2)
    if pooled_s == 0:
        return 0
    return (m1 - m2) / pooled_s

df_low = df[df['fin_strain_binary'] == 0]
df_high = df[df['fin_strain_binary'] == 1]

print(f"  No/Low strain: N = {len(df_low)}")
print(f"  High strain:   N = {len(df_high)}")
print(f"  Total:         N = {len(df)}")

table1_rows = []

# N row
table1_rows.append({
    'Variable': 'N',
    'No/Low Financial Strain': f'{len(df_low):,}',
    'High Financial Strain': f'{len(df_high):,}',
    'Overall': f'{len(df):,}',
    'SMD': ''
})

# Continuous: mean (SD)
for var, label in [('age', 'Age, mean (SD), y'),
                   ('education_yrs', 'Education, mean (SD), y'),
                   ('bmi', 'BMI, mean (SD), kg/m²')]:
    smd = calculate_smd(var, df_high, df_low)
    table1_rows.append({
        'Variable': label,
        'No/Low Financial Strain': f"{df_low[var].mean():.1f} ({df_low[var].std():.1f})",
        'High Financial Strain': f"{df_high[var].mean():.1f} ({df_high[var].std():.1f})",
        'Overall': f"{df[var].mean():.1f} ({df[var].std():.1f})",
        'SMD': f"{smd:.2f}"
    })

# Continuous: median (IQR) for income/wealth
for var, label in [('income', 'Household income, median (IQR), $'),
                   ('wealth', 'Total wealth, median (IQR), $')]:
    smd = calculate_smd(var, df_high, df_low)
    for grp, grp_label in [(df_low, 'No/Low Financial Strain'),
                            (df_high, 'High Financial Strain'),
                            (df, 'Overall')]:
        med = grp[var].median()
        q25 = grp[var].quantile(0.25)
        q75 = grp[var].quantile(0.75)
        if grp_label == 'No/Low Financial Strain':
            low_val = f"{med:,.0f} ({q25:,.0f}-{q75:,.0f})"
        elif grp_label == 'High Financial Strain':
            high_val = f"{med:,.0f} ({q25:,.0f}-{q75:,.0f})"
        else:
            overall_val = f"{med:,.0f} ({q25:,.0f}-{q75:,.0f})"
    table1_rows.append({
        'Variable': label,
        'No/Low Financial Strain': low_val,
        'High Financial Strain': high_val,
        'Overall': overall_val,
        'SMD': f"{smd:.2f}"
    })

# Binary: Female
smd = calculate_smd('female', df_high, df_low)
table1_rows.append({
    'Variable': 'Female, No. (%)',
    'No/Low Financial Strain': f"{int(df_low['female'].sum())} ({df_low['female'].mean()*100:.1f}%)",
    'High Financial Strain': f"{int(df_high['female'].sum())} ({df_high['female'].mean()*100:.1f}%)",
    'Overall': f"{int(df['female'].sum())} ({df['female'].mean()*100:.1f}%)",
    'SMD': f"{smd:.2f}"
})

# Race/ethnicity (compute per-indicator SMDs using binary indicator columns)
race_indicator_map = {'NH White': 'race_nh_white', 'NH Black': 'race_nh_black',
                      'Hispanic': 'race_hispanic', 'NH Other': 'race_nh_other'}
for race, label in [('NH White', 'Non-Hispanic White, No. (%)'),
                    ('NH Black', 'Non-Hispanic Black, No. (%)'),
                    ('Hispanic', 'Hispanic, No. (%)'),
                    ('NH Other', 'Non-Hispanic Other, No. (%)')]:
    low_n = (df_low['race_ethnicity'] == race).sum()
    low_pct = low_n / len(df_low) * 100
    high_n = (df_high['race_ethnicity'] == race).sum()
    high_pct = high_n / len(df_high) * 100
    all_n = (df['race_ethnicity'] == race).sum()
    all_pct = all_n / len(df) * 100
    race_smd = calculate_smd(race_indicator_map[race], df_high, df_low)
    table1_rows.append({
        'Variable': f'  {label}',
        'No/Low Financial Strain': f"{low_n} ({low_pct:.1f}%)",
        'High Financial Strain': f"{high_n} ({high_pct:.1f}%)",
        'Overall': f"{all_n} ({all_pct:.1f}%)",
        'SMD': f"{race_smd:.2f}"
    })

# Other binary variables
for var, label in [('married_partnered', 'Married/partnered, No. (%)'),
                   ('current_smoker', 'Current smoker, No. (%)'),
                   ('baseline_hypertension', 'Hypertension, No. (%)'),
                   ('baseline_diabetes', 'Diabetes, No. (%)'),
                   ('baseline_heart', 'Heart disease, No. (%)'),
                   ('baseline_stroke', 'Stroke, No. (%)'),
                   ('depressed', 'Depressive symptoms (CES-D ≥3), No. (%)'),
                   ('srh_fairpoor', 'Fair/poor self-rated health, No. (%)')]:
    smd = calculate_smd(var, df_high, df_low)
    table1_rows.append({
        'Variable': label,
        'No/Low Financial Strain': f"{int(df_low[var].sum())} ({df_low[var].mean()*100:.1f}%)",
        'High Financial Strain': f"{int(df_high[var].sum())} ({df_high[var].mean()*100:.1f}%)",
        'Overall': f"{int(df[var].sum())} ({df[var].mean()*100:.1f}%)",
        'SMD': f"{smd:.2f}"
    })

# Outcome variables
smd = calculate_smd('died', df_high, df_low)
table1_rows.append({
    'Variable': 'Died during follow-up, No. (%)',
    'No/Low Financial Strain': f"{int(df_low['died'].sum())} ({df_low['died'].mean()*100:.1f}%)",
    'High Financial Strain': f"{int(df_high['died'].sum())} ({df_high['died'].mean()*100:.1f}%)",
    'Overall': f"{int(df['died'].sum())} ({df['died'].mean()*100:.1f}%)",
    'SMD': f"{smd:.2f}"
})

smd = calculate_smd('followup_years', df_high, df_low)
table1_rows.append({
    'Variable': 'Follow-up, mean (SD), y',
    'No/Low Financial Strain': f"{df_low['followup_years'].mean():.1f} ({df_low['followup_years'].std():.1f})",
    'High Financial Strain': f"{df_high['followup_years'].mean():.1f} ({df_high['followup_years'].std():.1f})",
    'Overall': f"{df['followup_years'].mean():.1f} ({df['followup_years'].std():.1f})",
    'SMD': f"{smd:.2f}"
})

table1_df = pd.DataFrame(table1_rows)
table1_df.to_csv(TABLES_DIR / 'table1_baseline_v5.csv', index=False)
print("  Saved: table1_baseline_v5.csv")
print(table1_df.to_string(index=False))

# ============================================================================
# DEFINE MODEL SPECIFICATIONS
# ============================================================================

print("\n" + "="*80)
print("STEP 3: Defining model specifications")
print("="*80)

# Core covariate sets
demo_vars = ['age', 'female']
race_vars = ['race_nh_black', 'race_hispanic', 'race_nh_other']
ses_basic_vars = ['education_yrs', 'married_partnered']
behavior_vars = ['current_smoker', 'bmi']
health_vars = ['baseline_hypertension', 'baseline_diabetes', 'baseline_heart', 'baseline_stroke']
ses_objective_vars = ['asinh_income', 'asinh_wealth']
psych_vars = ['depressed', 'srh_fairpoor']

# Model specifications
model2_vars = demo_vars
model3_vars = demo_vars + race_vars + ses_basic_vars + behavior_vars + health_vars
model3b_vars = model3_vars + ses_objective_vars  # PRIMARY MODEL

# Model 4: CES-D as explicit confounder
model4_confounder_vars = model3b_vars + ['depressed']

# Model 4b: Full with SRH (potential mediator/confounder)
model4b_vars = model3b_vars + psych_vars

print("Model specifications:")
print(f"  Model 2: {len(model2_vars)} vars (age, sex)")
print(f"  Model 3: {len(model3_vars)} vars (+ demographics, health)")
print(f"  Model 3b: {len(model3b_vars)} vars (+ SES) - PRIMARY")
print(f"  Model 4 (CES-D confounder): {len(model4_confounder_vars)} vars")
print(f"  Model 4b (all psych): {len(model4b_vars)} vars")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def fit_cox_model(df, covariates, exposure='fin_strain_binary',
                  duration='followup_years', event='died', weights=None,
                  strata=None, penalizer=1e-5):
    """Fit Cox model and return comprehensive results."""
    model_vars = [exposure, duration, event] + covariates
    if weights:
        model_vars.append(weights)
    if strata:
        model_vars.append(strata)

    df_model = df[model_vars].dropna().copy()
    # Ensure all columns are numeric (pyreadstat can produce object dtypes)
    for col in df_model.columns:
        df_model[col] = pd.to_numeric(df_model[col], errors='coerce')
    df_model = df_model.dropna()

    if df_model[exposure].nunique() < 2 or len(df_model) < 100:
        return None

    cph = CoxPHFitter(penalizer=penalizer)

    try:
        if strata:
            cph.fit(df_model, duration_col=duration, event_col=event,
                   formula=f"{exposure} + " + " + ".join(covariates),
                   strata=[strata], robust=True)
        elif weights:
            cph.fit(df_model, duration_col=duration, event_col=event,
                   formula=f"{exposure} + " + " + ".join(covariates),
                   weights_col=weights, robust=True)
        else:
            cph.fit(df_model, duration_col=duration, event_col=event,
                   formula=f"{exposure} + " + " + ".join(covariates))

        hr = np.exp(cph.params_[exposure])
        ci = np.exp(cph.confidence_intervals_.loc[exposure])
        p = cph.summary.loc[exposure, 'p']

        return {
            'n': len(df_model),
            'events': int(df_model[event].sum()),
            'person_years': float(df_model[duration].sum()),
            'hr': float(hr),
            'ci_lower': float(ci['95% lower-bound']),
            'ci_upper': float(ci['95% upper-bound']),
            'p_value': float(p),
            'model': cph
        }
    except Exception as e:
        print(f"    Model failed: {e}")
        return None


def calculate_evalue(hr, ci_lower=None):
    """Calculate E-value for unmeasured confounding."""
    if hr >= 1:
        evalue = hr + np.sqrt(hr * (hr - 1))
    else:
        hr_inv = 1 / hr
        evalue = hr_inv + np.sqrt(hr_inv * (hr_inv - 1))

    evalue_ci = None
    if ci_lower is not None:
        if ci_lower >= 1:
            evalue_ci = ci_lower + np.sqrt(ci_lower * (ci_lower - 1))
        else:
            evalue_ci = 1

    return evalue, evalue_ci

# ============================================================================
# STEP 4: PRIMARY MORTALITY ANALYSES
# ============================================================================

print("\n" + "="*80)
print("STEP 4: Primary mortality analyses")
print("="*80)

results_mortality = {}

# Model 1: Unadjusted
print("\nModel 1: Unadjusted")
df_m1 = df[['fin_strain_binary', 'followup_years', 'died']].dropna()
cph1 = CoxPHFitter(penalizer=1e-5)
cph1.fit(df_m1, duration_col='followup_years', event_col='died', formula='fin_strain_binary')
results_mortality['model1'] = {
    'n': len(df_m1),
    'events': int(df_m1['died'].sum()),
    'hr': float(np.exp(cph1.params_['fin_strain_binary'])),
    'ci_lower': float(np.exp(cph1.confidence_intervals_.loc['fin_strain_binary', '95% lower-bound'])),
    'ci_upper': float(np.exp(cph1.confidence_intervals_.loc['fin_strain_binary', '95% upper-bound'])),
    'p_value': float(cph1.summary.loc['fin_strain_binary', 'p'])
}
print(f"  N={results_mortality['model1']['n']}, HR={results_mortality['model1']['hr']:.2f} "
      f"({results_mortality['model1']['ci_lower']:.2f}-{results_mortality['model1']['ci_upper']:.2f}), "
      f"P={results_mortality['model1']['p_value']:.4f}")

# Model 2: Age + Sex
print("\nModel 2: Age + Sex")
results_mortality['model2'] = fit_cox_model(df, model2_vars)
if results_mortality['model2']:
    print(f"  N={results_mortality['model2']['n']}, HR={results_mortality['model2']['hr']:.2f}, "
          f"P={results_mortality['model2']['p_value']:.4f}")

# Model 3: Full adjustment (no SES)
print("\nModel 3: Full adjustment (no SES)")
results_mortality['model3'] = fit_cox_model(df, model3_vars)
if results_mortality['model3']:
    print(f"  N={results_mortality['model3']['n']}, HR={results_mortality['model3']['hr']:.2f}, "
          f"P={results_mortality['model3']['p_value']:.4f}")

# Model 3b: + SES (PRIMARY MODEL)
print("\nModel 3b: + Income + Wealth (PRIMARY)")
results_mortality['model3b'] = fit_cox_model(df, model3b_vars)
if results_mortality['model3b']:
    print(f"  N={results_mortality['model3b']['n']}, HR={results_mortality['model3b']['hr']:.2f} "
          f"({results_mortality['model3b']['ci_lower']:.2f}-{results_mortality['model3b']['ci_upper']:.2f}), "
          f"P={results_mortality['model3b']['p_value']:.4f}")

    # Calculate attenuation
    if results_mortality['model3']:
        excess_m3 = results_mortality['model3']['hr'] - 1
        excess_m3b = results_mortality['model3b']['hr'] - 1
        attenuation = (1 - excess_m3b / excess_m3) * 100 if excess_m3 > 0 else 0
        print(f"  Attenuation from Model 3: {attenuation:.1f}%")

# Model 4: CES-D as explicit confounder
print("\nModel 4: + CES-D as explicit confounder")
results_mortality['model4_cesd_confounder'] = fit_cox_model(df, model4_confounder_vars)
if results_mortality['model4_cesd_confounder']:
    print(f"  N={results_mortality['model4_cesd_confounder']['n']}, HR={results_mortality['model4_cesd_confounder']['hr']:.2f} "
          f"({results_mortality['model4_cesd_confounder']['ci_lower']:.2f}-{results_mortality['model4_cesd_confounder']['ci_upper']:.2f}), "
          f"P={results_mortality['model4_cesd_confounder']['p_value']:.4f}")

# Model 4b: All psychological
print("\nModel 4b: + All psychological (CES-D + SRH)")
results_mortality['model4b_all_psych'] = fit_cox_model(df, model4b_vars)
if results_mortality['model4b_all_psych']:
    print(f"  N={results_mortality['model4b_all_psych']['n']}, HR={results_mortality['model4b_all_psych']['hr']:.2f} "
          f"({results_mortality['model4b_all_psych']['ci_lower']:.2f}-{results_mortality['model4b_all_psych']['ci_upper']:.2f}), "
          f"P={results_mortality['model4b_all_psych']['p_value']:.4f}")

# ============================================================================
# STEP 5: AGE INTERACTION ANALYSES
# ============================================================================

print("\n" + "="*80)
print("STEP 5: Age interaction with robust modeling")
print("="*80)

results_age_strat = {}

# Age-stratified analyses with within-stratum age adjustment
print("\nAge-stratified (Model 3b with within-stratum age adjustment):")
for age_grp in ['<65 years', '>=65 years']:
    df_grp = df[df['age_group'] == age_grp].copy()
    # Note: age is still included for within-stratum adjustment
    result = fit_cox_model(df_grp, model3b_vars)
    if result:
        results_age_strat[f'{age_grp}_model3b'] = result
        print(f"  {age_grp}: N={result['n']}, Deaths={result['events']}, "
              f"HR={result['hr']:.2f} ({result['ci_lower']:.2f}-{result['ci_upper']:.2f}), "
              f"P={result['p_value']:.4f}")

# Interaction test using binary age indicator
print("\nInteraction test (binary age <65):")
int_vars = list(set(['fin_strain_binary', 'followup_years', 'died', 'age_under65'] + model3b_vars))
df_int = df[int_vars].dropna().copy()
for col in df_int.columns:
    df_int[col] = pd.to_numeric(df_int[col], errors='coerce')
df_int = df_int.dropna()
df_int['strain_x_age65'] = df_int['fin_strain_binary'] * df_int['age_under65']

cph_int = CoxPHFitter(penalizer=1e-5)
int_covars = [v for v in model3b_vars if v != 'age']
int_formula = 'fin_strain_binary + age_under65 + strain_x_age65 + ' + ' + '.join(int_covars)
cph_int.fit(df_int, duration_col='followup_years', event_col='died', formula=int_formula)

interaction_hr = np.exp(cph_int.params_['strain_x_age65'])
interaction_ci = np.exp(cph_int.confidence_intervals_.loc['strain_x_age65'])
interaction_p = cph_int.summary.loc['strain_x_age65', 'p']

results_age_strat['interaction_term'] = {
    'hr': float(interaction_hr),  # Ratio of HRs
    'ci_lower': float(interaction_ci['95% lower-bound']),
    'ci_upper': float(interaction_ci['95% upper-bound']),
    'p_value': float(interaction_p)
}
print(f"  Interaction HR (ratio of HRs): {interaction_hr:.2f} ({interaction_ci['95% lower-bound']:.2f}-{interaction_ci['95% upper-bound']:.2f})")
print(f"  P for interaction: {interaction_p:.4f}")

# Continuous age interaction (age × strain)
print("\nContinuous age interaction (age × strain):")
df_cont_int = df[['fin_strain_binary', 'followup_years', 'died', 'age'] +
                 [v for v in model3b_vars if v != 'age']].dropna().copy()
for col in df_cont_int.columns:
    df_cont_int[col] = pd.to_numeric(df_cont_int[col], errors='coerce')
df_cont_int = df_cont_int.dropna()
df_cont_int['strain_x_age'] = df_cont_int['fin_strain_binary'] * df_cont_int['age']

cph_cont_int = CoxPHFitter(penalizer=1e-5)
cont_int_formula = 'fin_strain_binary + age + strain_x_age + ' + ' + '.join([v for v in model3b_vars if v != 'age'])
cph_cont_int.fit(df_cont_int, duration_col='followup_years', event_col='died', formula=cont_int_formula)

cont_interaction_p = cph_cont_int.summary.loc['strain_x_age', 'p']
results_age_strat['continuous_interaction_p'] = float(cont_interaction_p)
print(f"  P for continuous age interaction: {cont_interaction_p:.4f}")

# ============================================================================
# STEP 6: PROPENSITY SCORE MATCHING WITH MODEL 3b COVARIATES
# ============================================================================

print("\n" + "="*80)
print("STEP 6: PS matching with Model 3b covariates")
print("="*80)

# PS covariates: Model 3b covariates plus SRH (richer covariate set for balance)
ps_vars = ['age', 'female', 'race_nh_black', 'race_hispanic', 'race_nh_other',
           'education_yrs', 'married_partnered', 'current_smoker', 'bmi',
           'baseline_hypertension', 'baseline_diabetes', 'baseline_heart', 'baseline_stroke',
           'asinh_income', 'asinh_wealth', 'srh_fairpoor']

ps_model_vars = ['fin_strain_binary', 'followup_years', 'died'] + ps_vars
df_ps = df[ps_model_vars].dropna().copy()
print(f"  Sample for PS: N={len(df_ps)}")

# Fit PS model
X = df_ps[ps_vars].values
y = df_ps['fin_strain_binary'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

ps_model = LogisticRegression(max_iter=1000, random_state=42)
ps_model.fit(X_scaled, y)
df_ps['ps'] = ps_model.predict_proba(X_scaled)[:, 1]

print(f"  PS distribution - Control: {df_ps[df_ps['fin_strain_binary']==0]['ps'].mean():.3f}, "
      f"Treated: {df_ps[df_ps['fin_strain_binary']==1]['ps'].mean():.3f}")

# Caliper matching (0.2 SD of logit PS)
def caliper_match_ps(df, treat_col, ps_col, caliper_sd=0.2, seed=42):
    """Caliper matching with 0.2 SD of logit PS."""
    np.random.seed(seed)
    df = df.copy()

    eps = 1e-6
    ps_clipped = np.clip(df[ps_col], eps, 1 - eps)
    df['logit_ps'] = scipy_logit(ps_clipped)

    caliper = caliper_sd * df['logit_ps'].std()

    treated = df[df[treat_col] == 1].copy()
    control = df[df[treat_col] == 0].copy()
    treated = treated.sample(frac=1, random_state=seed)

    matched_pairs = []
    used_control_idx = set()
    pair_id = 0

    for idx, t_row in treated.iterrows():
        available = control[~control.index.isin(used_control_idx)]
        if available.empty:
            continue

        distances = np.abs(available['logit_ps'] - t_row['logit_ps'])
        within_caliper = distances[distances <= caliper]

        if within_caliper.empty:
            continue

        closest_idx = within_caliper.idxmin()
        used_control_idx.add(closest_idx)
        pair_id += 1

        t_row_copy = t_row.copy()
        c_row_copy = control.loc[closest_idx].copy()
        t_row_copy['pair_id'] = pair_id
        c_row_copy['pair_id'] = pair_id

        matched_pairs.append(pd.DataFrame([t_row_copy]))
        matched_pairs.append(pd.DataFrame([c_row_copy]))

    if not matched_pairs:
        return None, 0, len(treated)

    matched_df = pd.concat(matched_pairs, ignore_index=True)
    return matched_df, pair_id, len(treated) - pair_id

matched_df, n_pairs, n_unmatched = caliper_match_ps(df_ps, 'fin_strain_binary', 'ps', caliper_sd=0.2)

if matched_df is not None:
    print(f"  Matched pairs: {n_pairs}")
    print(f"  Unmatched treated: {n_unmatched}")

    # Balance table
    matched_treat = matched_df[matched_df['fin_strain_binary'] == 1]
    matched_ctrl = matched_df[matched_df['fin_strain_binary'] == 0]

    balance_rows = []
    print("\n  Covariate balance (SMD):")
    for var in ps_vars:
        smd_before = calculate_smd(var, df_ps[df_ps['fin_strain_binary']==1],
                                   df_ps[df_ps['fin_strain_binary']==0])
        smd_after = calculate_smd(var, matched_treat, matched_ctrl)
        balance_rows.append({
            'Variable': var,
            'SMD_Before': round(smd_before, 3) if not np.isnan(smd_before) else 'NA',
            'SMD_After': round(smd_after, 3) if not np.isnan(smd_after) else 'NA'
        })
        if abs(smd_before) > 0.1 or (not np.isnan(smd_after) and abs(smd_after) > 0.1):
            print(f"    {var}: Before={smd_before:.3f}, After={smd_after:.3f}")

    balance_df = pd.DataFrame(balance_rows)
    balance_df.to_csv(TABLES_DIR / 'etable_ps_balance.csv', index=False)
    print("  All SMDs after matching < 0.10: ",
          all(abs(float(r['SMD_After'])) < 0.1 for r in balance_rows if r['SMD_After'] != 'NA'))

    # Cox on matched sample (stratified by pair)
    matched_fit_df = matched_df[['followup_years', 'died', 'fin_strain_binary', 'pair_id']].copy()
    for col in ['followup_years', 'died', 'fin_strain_binary', 'pair_id']:
        matched_fit_df[col] = pd.to_numeric(matched_fit_df[col], errors='coerce')
    matched_fit_df = matched_fit_df.dropna()
    cph_matched = CoxPHFitter(penalizer=1e-5)
    cph_matched.fit(matched_fit_df,
                    duration_col='followup_years', event_col='died',
                    formula='fin_strain_binary', strata=['pair_id'],
                    robust=True)

    hr_matched = np.exp(cph_matched.params_['fin_strain_binary'])
    ci_matched = np.exp(cph_matched.confidence_intervals_.loc['fin_strain_binary'])
    p_matched = cph_matched.summary.loc['fin_strain_binary', 'p']

    results_mortality['ps_matched'] = {
        'n': len(matched_df),
        'n_pairs': n_pairs,
        'events': int(matched_df['died'].sum()),
        'hr': float(hr_matched),
        'ci_lower': float(ci_matched['95% lower-bound']),
        'ci_upper': float(ci_matched['95% upper-bound']),
        'p_value': float(p_matched),
        'covariates': 'Model 3b covariates plus SRH'
    }

    print(f"\n  PS-matched: HR={hr_matched:.2f} ({ci_matched['95% lower-bound']:.2f}-"
          f"{ci_matched['95% upper-bound']:.2f}), P={p_matched:.4f}")

# ============================================================================
# STEP 7: REVERSE CAUSATION SENSITIVITY (1 and 2 year exclusions)
# ============================================================================

print("\n" + "="*80)
print("STEP 7: Reverse causation sensitivity analyses")
print("="*80)

for exclude_years in [1, 2]:
    df_rev = df.copy()
    early_deaths = (df_rev['died'] == 1) & (df_rev['followup_years'] < exclude_years)
    df_rev = df_rev[~early_deaths].copy()
    df_rev['followup_years_adj'] = df_rev['followup_years'] - exclude_years
    df_rev = df_rev[df_rev['followup_years_adj'] > 0].copy()

    result = fit_cox_model(df_rev, model3b_vars, duration='followup_years_adj')
    if result:
        results_mortality[f'exclude_{exclude_years}yr'] = result
        print(f"  Exclude {exclude_years} yr: N={result['n']}, Deaths={result['events']}, "
              f"HR={result['hr']:.2f} ({result['ci_lower']:.2f}-{result['ci_upper']:.2f}), "
              f"P={result['p_value']:.4f}")

# ============================================================================
# STEP 8: INCIDENT HEART DISEASE WITH COMPETING RISKS NOTE
# ============================================================================

print("\n" + "="*80)
print("STEP 8: Incident heart disease analysis")
print("="*80)

df_cardiac = df[df['baseline_cvd_free'] == True].copy()
df_cardiac = df_cardiac[df_cardiac['cardiac_followup_years'] > 0].copy()

print(f"  CVD-free at baseline: N={len(df_cardiac)}")
print(f"  Incident heart disease: {df_cardiac['incident_heart'].sum():.0f}")
print(f"  Deaths without heart disease (competing): {((df_cardiac['died']==1) & (df_cardiac['incident_heart']==0)).sum()}")

results_cardiac = {}

# Cardiac model vars (exclude baseline heart/stroke since restricted to CVD-free)
cardiac_model3b_vars = [v for v in model3b_vars if v not in ['baseline_heart', 'baseline_stroke']]

# Cause-specific Cox
print("\nCause-specific Cox (death as censoring):")
results_cardiac['cause_specific'] = fit_cox_model(
    df_cardiac, cardiac_model3b_vars,
    duration='cardiac_followup_years', event='incident_heart'
)
if results_cardiac['cause_specific']:
    print(f"  HR={results_cardiac['cause_specific']['hr']:.2f} "
          f"({results_cardiac['cause_specific']['ci_lower']:.2f}-{results_cardiac['cause_specific']['ci_upper']:.2f}), "
          f"P={results_cardiac['cause_specific']['p_value']:.4f}")

# Note about Fine-Gray (requires R/cmprsk)
print("\n  NOTE: Fine-Gray subdistribution hazard estimated via Stata 18 stcrreg.")
print("  Abstract should note: cause-specific HR significant, Fine-Gray attenuated (see eTable)")

# Age-stratified cardiac
print("\nAge-stratified cardiac (Model 3b):")
for age_grp in ['<65 years', '>=65 years']:
    df_grp = df_cardiac[df_cardiac['age_group'] == age_grp]
    if df_grp['incident_heart'].sum() >= 20:
        result = fit_cox_model(df_grp, cardiac_model3b_vars,
                               duration='cardiac_followup_years', event='incident_heart')
        if result:
            results_cardiac[f'{age_grp}'] = result
            print(f"  {age_grp}: HR={result['hr']:.2f} ({result['ci_lower']:.2f}-{result['ci_upper']:.2f}), "
                  f"P={result['p_value']:.4f}")

# ============================================================================
# STEP 9: ABSOLUTE RISK METRICS
# ============================================================================

print("\n" + "="*80)
print("STEP 9: Absolute risk metrics")
print("="*80)

# Get complete case data for survival predictions
df_surv = df[['fin_strain_binary', 'followup_years', 'died'] + model3b_vars].dropna()
for col in df_surv.columns:
    df_surv[col] = pd.to_numeric(df_surv[col], errors='coerce')
df_surv = df_surv.dropna()

cph_surv = CoxPHFitter(penalizer=1e-5)
cph_surv.fit(df_surv, duration_col='followup_years', event_col='died',
             formula='fin_strain_binary + ' + ' + '.join(model3b_vars))

# Create average covariate profile - need all columns including exposure
all_model_vars = ['fin_strain_binary'] + model3b_vars
avg_profile = df_surv[all_model_vars].mean().to_dict()

# Survival at 5, 10, 15 years
surv_times = [5, 10, 15]
survival_results = {'times': surv_times, 'high_strain': [], 'low_strain': []}

for strain_val in [0, 1]:
    profile = avg_profile.copy()
    profile['fin_strain_binary'] = strain_val
    profile_df = pd.DataFrame([profile])

    try:
        surv_func = cph_surv.predict_survival_function(profile_df)

        for t in surv_times:
            if t <= surv_func.index.max():
                time_diffs = np.abs(surv_func.index.values - t)
                idx = np.argmin(time_diffs)
                surv_prob = surv_func.iloc[idx, 0]
            else:
                surv_prob = np.nan

            if strain_val == 1:
                survival_results['high_strain'].append(float(surv_prob))
            else:
                survival_results['low_strain'].append(float(surv_prob))
    except Exception as e:
        print(f"  Warning: Could not predict survival function: {e}")
        for t in surv_times:
            if strain_val == 1:
                survival_results['high_strain'].append(np.nan)
            else:
                survival_results['low_strain'].append(np.nan)

print("\nAdjusted survival probabilities (at mean covariate values):")
print(f"  {'Time':>6} {'No/Low Strain':>15} {'High Strain':>15} {'ARD':>12}")
for i, t in enumerate(surv_times):
    low = survival_results['low_strain'][i]
    high = survival_results['high_strain'][i]
    ard = (1 - high) - (1 - low)  # Absolute risk difference
    print(f"  {t:>4} yr {low*100:>14.1f}% {high*100:>14.1f}% {ard*100:>11.1f} pp")

# Save
survival_df = pd.DataFrame({
    'Time_years': surv_times,
    'Survival_NoLowStrain': [f"{x*100:.1f}%" for x in survival_results['low_strain']],
    'Survival_HighStrain': [f"{x*100:.1f}%" for x in survival_results['high_strain']],
    'AbsoluteRiskDiff_pp': [f"{((1-survival_results['high_strain'][i])-(1-survival_results['low_strain'][i]))*100:.1f}"
                            for i in range(len(surv_times))]
})
survival_df.to_csv(TABLES_DIR / 'etable_absolute_risks_v2.csv', index=False)

# ============================================================================
# STEP 10: DOSE-RESPONSE WITH CATEGORY-SPECIFIC EMPHASIS
# ============================================================================

print("\n" + "="*80)
print("STEP 10: Dose-response analysis (category-specific emphasis)")
print("="*80)

df['strain_not_upsetting'] = np.where(df['fin_strain_ordinal'] == 1, 1, 0)
df['strain_somewhat'] = np.where(df['fin_strain_ordinal'] == 2, 1, 0)
df['strain_very'] = np.where(df['fin_strain_ordinal'] == 3, 1, 0)

dose_vars = model3b_vars + ['strain_not_upsetting', 'strain_somewhat', 'strain_very']
df_dose = df[['followup_years', 'died'] + dose_vars].dropna()
for col in df_dose.columns:
    df_dose[col] = pd.to_numeric(df_dose[col], errors='coerce')
df_dose = df_dose.dropna()

cph_dose = CoxPHFitter(penalizer=1e-5)
cph_dose.fit(df_dose, duration_col='followup_years', event_col='died',
             formula=' + '.join(dose_vars))

# Category counts and results
print("\nCategory distribution and results:")
dose_results = []
for cat, label, dummy_var in [(1, 'No strain (ref)', None),
                               (2, 'Yes, not upsetting', 'strain_not_upsetting'),
                               (3, 'Somewhat upsetting', 'strain_somewhat'),
                               (4, 'Very upsetting', 'strain_very')]:
    n = (df['fin_strain_raw'] == cat).sum()
    deaths = df[df['fin_strain_raw'] == cat]['died'].sum()

    if dummy_var:
        hr = np.exp(cph_dose.params_[dummy_var])
        ci = np.exp(cph_dose.confidence_intervals_.loc[dummy_var])
        p = cph_dose.summary.loc[dummy_var, 'p']
        hr_str = f"{hr:.2f} ({ci['95% lower-bound']:.2f}-{ci['95% upper-bound']:.2f})"
        p_str = f"{p:.3f}"
    else:
        hr_str = "1.00 (ref)"
        p_str = "-"

    dose_results.append({
        'Category': label,
        'N': n,
        'Deaths': int(deaths),
        'HR_95CI': hr_str,
        'P': p_str
    })
    print(f"  {label}: N={n}, Deaths={deaths}, {hr_str}, P={p_str}")

# Trend test
df_trend = df[['followup_years', 'died', 'fin_strain_ordinal'] + model3b_vars].dropna()
for col in df_trend.columns:
    df_trend[col] = pd.to_numeric(df_trend[col], errors='coerce')
df_trend = df_trend.dropna()
cph_trend = CoxPHFitter(penalizer=1e-5)
cph_trend.fit(df_trend, duration_col='followup_years', event_col='died',
              formula='fin_strain_ordinal + ' + ' + '.join(model3b_vars))

# Find the ordinal variable in summary
trend_summary = cph_trend.summary
trend_p = None
for idx in trend_summary.index:
    if 'ordinal' in str(idx).lower() or 'strain' in str(idx).lower() and 'strain_' not in str(idx):
        if 'strain_not' not in str(idx) and 'strain_somewhat' not in str(idx) and 'strain_very' not in str(idx):
            trend_p = trend_summary.loc[idx, 'p']
            break

if trend_p is not None:
    print(f"\n  P for linear trend: {trend_p:.4f}")
else:
    print("\n  Warning: Could not find trend term. Available:", list(trend_summary.index)[:5])
    trend_p = np.nan
print("  NOTE: Pattern is NON-MONOTONIC ('very upsetting' not significantly elevated)")

pd.DataFrame(dose_results).to_csv(TABLES_DIR / 'etable_dose_response_detailed.csv', index=False)

# ============================================================================
# STEP 11: CROSS-CLASSIFICATION STRAIN x SES
# ============================================================================

print("\n" + "="*80)
print("STEP 11: Cross-classification strain x income quartiles")
print("="*80)

# Create income quartiles
df['income_quartile'] = pd.qcut(df['income'].rank(method='first'), 4, labels=['Q1 (lowest)', 'Q2', 'Q3', 'Q4 (highest)'])

cross_class_results = []
for iq in ['Q1 (lowest)', 'Q2', 'Q3', 'Q4 (highest)']:
    df_iq = df[df['income_quartile'] == iq].copy()

    # Fit model within income quartile
    model_vars_no_income = [v for v in model3b_vars if v != 'asinh_income']
    result = fit_cox_model(df_iq, model_vars_no_income)

    if result:
        cross_class_results.append({
            'Income_Quartile': iq,
            'N': result['n'],
            'N_HighStrain': int(df_iq['fin_strain_binary'].sum()),
            'Deaths': result['events'],
            'HR': f"{result['hr']:.2f}",
            'CI': f"{result['ci_lower']:.2f}-{result['ci_upper']:.2f}",
            'P': f"{result['p_value']:.3f}"
        })
        print(f"  {iq}: N={result['n']}, HR={result['hr']:.2f} ({result['ci_lower']:.2f}-{result['ci_upper']:.2f}), P={result['p_value']:.4f}")

pd.DataFrame(cross_class_results).to_csv(TABLES_DIR / 'etable_cross_classification.csv', index=False)

# ============================================================================
# STEP 12: E-VALUES
# ============================================================================

print("\n" + "="*80)
print("STEP 12: E-value calculations")
print("="*80)

evalues = {}
for name, result in [
    ('Model 3', results_mortality.get('model3')),
    ('Model 3b (Primary)', results_mortality.get('model3b')),
    ('Model 4 (CES-D confounder)', results_mortality.get('model4_cesd_confounder')),
    ('Age <65', results_age_strat.get('<65 years_model3b')),
    ('Age >=65', results_age_strat.get('>=65 years_model3b')),
    ('PS-matched', results_mortality.get('ps_matched')),
]:
    if result and 'hr' in result:
        ev, ev_ci = calculate_evalue(result['hr'], result.get('ci_lower'))
        evalues[name] = {'hr': result['hr'], 'evalue': ev, 'evalue_ci': ev_ci}
        ev_ci_str = f"{ev_ci:.2f}" if ev_ci else "N/A"
        print(f"  {name}: HR={result['hr']:.2f}, E-value={ev:.2f}, E-value(CI)={ev_ci_str}")

print("\n  NOTE: E-values address unmeasured confounding only, not selection bias or misclassification")

# ============================================================================
# STEP 13: SURVEY-WEIGHTED MODELS (PRIMARY)
# ============================================================================

print("\n" + "="*80)
print("STEP 13: Survey-weighted models (PRIMARY for revision)")
print("="*80)

# Weighted subsample: those with valid LBQ weight
df_wt = df[df['lb_weight'].notna()].copy()
print(f"  Weighted subsample N = {len(df_wt)}")

results_weighted = {}

# Weighted Model 1: Unadjusted
print("\nWeighted Model 1: Unadjusted")
df_wm1 = df_wt[['fin_strain_binary', 'followup_years', 'died', 'lb_weight']].dropna()
for col in df_wm1.columns:
    df_wm1[col] = pd.to_numeric(df_wm1[col], errors='coerce')
df_wm1 = df_wm1.dropna()
cph_wm1 = CoxPHFitter(penalizer=1e-5)
cph_wm1.fit(df_wm1, duration_col='followup_years', event_col='died',
            formula='fin_strain_binary', weights_col='lb_weight', robust=True)
results_weighted['model1'] = {
    'n': len(df_wm1), 'events': int(df_wm1['died'].sum()),
    'hr': float(np.exp(cph_wm1.params_['fin_strain_binary'])),
    'ci_lower': float(np.exp(cph_wm1.confidence_intervals_.loc['fin_strain_binary', '95% lower-bound'])),
    'ci_upper': float(np.exp(cph_wm1.confidence_intervals_.loc['fin_strain_binary', '95% upper-bound'])),
    'p_value': float(cph_wm1.summary.loc['fin_strain_binary', 'p'])
}
print(f"  N={results_weighted['model1']['n']}, HR={results_weighted['model1']['hr']:.2f} "
      f"({results_weighted['model1']['ci_lower']:.2f}-{results_weighted['model1']['ci_upper']:.2f}), "
      f"P={results_weighted['model1']['p_value']:.4f}")

# Weighted Models 2, 3, 3b, 4, 4b
for model_name, covars, label in [
    ('model2', model2_vars, 'Weighted Model 2: Age + Sex'),
    ('model3', model3_vars, 'Weighted Model 3: Full adj (no SES)'),
    ('model3b', model3b_vars, 'Weighted Model 3b: + Income + Wealth (PRIMARY)'),
    ('model4', model4_confounder_vars, 'Weighted Model 4: + CES-D'),
    ('model4b', model4b_vars, 'Weighted Model 4b: + All psych'),
]:
    print(f"\n{label}")
    result = fit_cox_model(df_wt, covars, weights='lb_weight')
    if result:
        results_weighted[model_name] = result
        print(f"  N={result['n']}, HR={result['hr']:.2f} "
              f"({result['ci_lower']:.2f}-{result['ci_upper']:.2f}), P={result['p_value']:.4f}")

# Weighted attenuation
if results_weighted.get('model3') and results_weighted.get('model3b'):
    excess_m3_wt = results_weighted['model3']['hr'] - 1
    excess_m3b_wt = results_weighted['model3b']['hr'] - 1
    attenuation_wt = (1 - excess_m3b_wt / excess_m3_wt) * 100 if excess_m3_wt > 0 else 0
    print(f"\n  Weighted attenuation from Model 3 to 3b: {attenuation_wt:.1f}%")
    results_weighted['attenuation_pct'] = float(attenuation_wt)

# Weighted age-stratified
print("\nWeighted age-stratified (Model 3b):")
results_weighted_age = {}
for age_grp in ['<65 years', '>=65 years']:
    df_grp = df_wt[df_wt['age_group'] == age_grp].copy()
    result = fit_cox_model(df_grp, model3b_vars, weights='lb_weight')
    if result:
        results_weighted_age[age_grp] = result
        print(f"  {age_grp}: N={result['n']}, Deaths={result['events']}, "
              f"HR={result['hr']:.2f} ({result['ci_lower']:.2f}-{result['ci_upper']:.2f}), "
              f"P={result['p_value']:.4f}")

# Weighted interaction test
print("\nWeighted interaction test (binary age <65):")
int_vars_wt = list(set(['fin_strain_binary', 'followup_years', 'died', 'age_under65', 'lb_weight'] + model3b_vars))
df_int_wt = df_wt[int_vars_wt].dropna().copy()
for col in df_int_wt.columns:
    df_int_wt[col] = pd.to_numeric(df_int_wt[col], errors='coerce')
df_int_wt = df_int_wt.dropna()
df_int_wt['strain_x_age65'] = df_int_wt['fin_strain_binary'] * df_int_wt['age_under65']

cph_int_wt = CoxPHFitter(penalizer=1e-5)
int_covars_wt = [v for v in model3b_vars if v != 'age']
int_formula_wt = 'fin_strain_binary + age_under65 + strain_x_age65 + ' + ' + '.join(int_covars_wt)
cph_int_wt.fit(df_int_wt, duration_col='followup_years', event_col='died',
               formula=int_formula_wt, weights_col='lb_weight', robust=True)

wt_interaction_p = float(cph_int_wt.summary.loc['strain_x_age65', 'p'])
results_weighted_age['interaction_p'] = wt_interaction_p
print(f"  Weighted interaction P: {wt_interaction_p:.4f}")

# Weighted continuous age interaction (center age at 65 for numerical stability)
print("\nWeighted continuous age interaction:")
df_cont_wt = df_wt[['fin_strain_binary', 'followup_years', 'died', 'age', 'lb_weight'] +
                    [v for v in model3b_vars if v != 'age']].dropna().copy()
for col in df_cont_wt.columns:
    df_cont_wt[col] = pd.to_numeric(df_cont_wt[col], errors='coerce')
df_cont_wt = df_cont_wt.dropna()
df_cont_wt['age_centered'] = df_cont_wt['age'] - 65
df_cont_wt['strain_x_age_centered'] = df_cont_wt['fin_strain_binary'] * df_cont_wt['age_centered']

# Normalize weights to sum to N for this model. Lifelines treats weights_col as
# frequency weights, so raw probability weights (sum ~74M) inflate effective N and
# collapse model-based variance in variance_matrix_. Normalizing to sum=N keeps
# point estimates identical and gives variance_matrix_ values close to the robust
# sandwich SEs, which we need for linear-combination CIs at specific ages.
n_cont_wt = len(df_cont_wt)
df_cont_wt['lb_weight_norm'] = df_cont_wt['lb_weight'] * n_cont_wt / df_cont_wt['lb_weight'].sum()

cph_cont_wt = CoxPHFitter(penalizer=1e-5)
cont_formula_wt = 'fin_strain_binary + age_centered + strain_x_age_centered + ' + ' + '.join([v for v in model3b_vars if v != 'age'])
cph_cont_wt.fit(df_cont_wt, duration_col='followup_years', event_col='died',
                formula=cont_formula_wt, weights_col='lb_weight_norm', robust=True)
wt_cont_int_p = float(cph_cont_wt.summary.loc['strain_x_age_centered', 'p'])
results_weighted_age['continuous_interaction_p'] = wt_cont_int_p
print(f"  Weighted continuous interaction P: {wt_cont_int_p:.4f}")

# Age-specific HRs from weighted continuous interaction model
# Use variance_matrix_ (model-based) for linear combination CIs — with normalized
# weights this approximates the robust sandwich variance.
print("\nWeighted age-specific HRs from continuous interaction:")
wt_strain_coef = cph_cont_wt.params_['fin_strain_binary']
wt_interact_coef = cph_cont_wt.params_['strain_x_age_centered']
wt_strain_var = cph_cont_wt.variance_matrix_.loc['fin_strain_binary', 'fin_strain_binary']
wt_interact_var = cph_cont_wt.variance_matrix_.loc['strain_x_age_centered', 'strain_x_age_centered']
wt_cov = cph_cont_wt.variance_matrix_.loc['fin_strain_binary', 'strain_x_age_centered']

wt_age_specific_hrs = []
for target_age in [55, 60, 65, 70, 75, 80]:
    age_c = target_age - 65  # centered
    log_hr = wt_strain_coef + wt_interact_coef * age_c
    se_log_hr = np.sqrt(wt_strain_var + age_c**2 * wt_interact_var + 2 * age_c * wt_cov)
    hr_val = np.exp(log_hr)
    ci_lo = np.exp(log_hr - 1.96 * se_log_hr)
    ci_hi = np.exp(log_hr + 1.96 * se_log_hr)
    wt_age_specific_hrs.append({'Age': target_age, 'HR': round(hr_val, 2),
                                 'CI_lower': round(ci_lo, 2), 'CI_upper': round(ci_hi, 2)})
    print(f"  Age {target_age}: HR = {hr_val:.2f} ({ci_lo:.2f}, {ci_hi:.2f})")

pd.DataFrame(wt_age_specific_hrs).to_csv(TABLES_DIR / 'etable_weighted_age_specific_hrs.csv', index=False)

# Save weighted results table
wt_table = []
for model, label in [
    ('model1', 'Model 1: Unadjusted (weighted)'),
    ('model2', 'Model 2: Age + Sex (weighted)'),
    ('model3', 'Model 3: Full adj (weighted)'),
    ('model3b', 'Model 3b: + Income + Wealth (weighted, PRIMARY)'),
    ('model4', 'Model 4: + CES-D (weighted)'),
    ('model4b', 'Model 4b: + All psych (weighted)'),
]:
    if model in results_weighted and results_weighted[model]:
        r = results_weighted[model]
        wt_table.append({
            'Model': label,
            'N': r['n'],
            'Deaths': r['events'],
            'HR': f"{r['hr']:.2f}",
            'CI': f"{r['ci_lower']:.2f}-{r['ci_upper']:.2f}",
            'P': f"{r['p_value']:.4f}" if r['p_value'] >= 0.0005 else '<.001'
        })
pd.DataFrame(wt_table).to_csv(TABLES_DIR / 'table2_weighted_primary.csv', index=False)
print("  Saved: table2_weighted_primary.csv")

# Weighted E-values
print("\nWeighted E-values:")
evalues_weighted = {}
for name, result_dict in [
    ('Model 3 (weighted)', results_weighted.get('model3')),
    ('Model 3b (weighted, primary)', results_weighted.get('model3b')),
    ('Model 4 (weighted)', results_weighted.get('model4')),
    ('Age <65 (weighted)', results_weighted_age.get('<65 years')),
    ('Age >=65 (weighted)', results_weighted_age.get('>=65 years')),
]:
    if result_dict and 'hr' in result_dict:
        ev, ev_ci = calculate_evalue(result_dict['hr'], result_dict.get('ci_lower'))
        evalues_weighted[name] = {'hr': result_dict['hr'], 'evalue': ev, 'evalue_ci': ev_ci}
        ev_ci_str = f"{ev_ci:.2f}" if ev_ci else "N/A"
        print(f"  {name}: HR={result_dict['hr']:.2f}, E-value={ev:.2f}, E-value(CI)={ev_ci_str}")

# ============================================================================
# STEP 14: ATTAINED-AGE SENSITIVITY ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("STEP 14: Attained-age (age-as-time-scale) sensitivity analysis")
print("="*80)

# Attained-age model: use age_at_entry and age_at_exit as time scale
# Remove age from covariates (implicit in time scale)
attained_covars = [v for v in model3b_vars if v != 'age']

results_attained = {}

# Unweighted attained-age
print("\nUnweighted attained-age model:")
aa_vars = ['fin_strain_binary', 'age_at_entry', 'age_at_exit', 'died'] + attained_covars
df_aa = df[aa_vars].dropna().copy()
for col in df_aa.columns:
    df_aa[col] = pd.to_numeric(df_aa[col], errors='coerce')
df_aa = df_aa.dropna()
df_aa = df_aa[df_aa['age_at_exit'] > df_aa['age_at_entry']].copy()

cph_aa = CoxPHFitter(penalizer=1e-5)
cph_aa.fit(df_aa, duration_col='age_at_exit', event_col='died',
           entry_col='age_at_entry',
           formula='fin_strain_binary + ' + ' + '.join(attained_covars))

results_attained['unweighted'] = {
    'n': len(df_aa), 'events': int(df_aa['died'].sum()),
    'hr': float(np.exp(cph_aa.params_['fin_strain_binary'])),
    'ci_lower': float(np.exp(cph_aa.confidence_intervals_.loc['fin_strain_binary', '95% lower-bound'])),
    'ci_upper': float(np.exp(cph_aa.confidence_intervals_.loc['fin_strain_binary', '95% upper-bound'])),
    'p_value': float(cph_aa.summary.loc['fin_strain_binary', 'p'])
}
print(f"  N={results_attained['unweighted']['n']}, "
      f"HR={results_attained['unweighted']['hr']:.2f} "
      f"({results_attained['unweighted']['ci_lower']:.2f}-{results_attained['unweighted']['ci_upper']:.2f}), "
      f"P={results_attained['unweighted']['p_value']:.4f}")

# Weighted attained-age
print("\nWeighted attained-age model:")
aa_vars_wt = aa_vars + ['lb_weight']
df_aa_wt = df_wt[aa_vars_wt].dropna().copy()
for col in df_aa_wt.columns:
    df_aa_wt[col] = pd.to_numeric(df_aa_wt[col], errors='coerce')
df_aa_wt = df_aa_wt.dropna()
df_aa_wt = df_aa_wt[df_aa_wt['age_at_exit'] > df_aa_wt['age_at_entry']].copy()

cph_aa_wt = CoxPHFitter(penalizer=1e-5)
cph_aa_wt.fit(df_aa_wt, duration_col='age_at_exit', event_col='died',
              entry_col='age_at_entry',
              formula='fin_strain_binary + ' + ' + '.join(attained_covars),
              weights_col='lb_weight', robust=True)

results_attained['weighted'] = {
    'n': len(df_aa_wt), 'events': int(df_aa_wt['died'].sum()),
    'hr': float(np.exp(cph_aa_wt.params_['fin_strain_binary'])),
    'ci_lower': float(np.exp(cph_aa_wt.confidence_intervals_.loc['fin_strain_binary', '95% lower-bound'])),
    'ci_upper': float(np.exp(cph_aa_wt.confidence_intervals_.loc['fin_strain_binary', '95% upper-bound'])),
    'p_value': float(cph_aa_wt.summary.loc['fin_strain_binary', 'p'])
}
print(f"  N={results_attained['weighted']['n']}, "
      f"HR={results_attained['weighted']['hr']:.2f} "
      f"({results_attained['weighted']['ci_lower']:.2f}-{results_attained['weighted']['ci_upper']:.2f}), "
      f"P={results_attained['weighted']['p_value']:.4f}")

# Attained-age: age-stratified
print("\nAttained-age age-stratified:")
for age_grp in ['<65 years', '>=65 years']:
    df_grp = df[df['age_group'] == age_grp].copy()
    df_grp_aa = df_grp[aa_vars].dropna().copy()
    for col in df_grp_aa.columns:
        df_grp_aa[col] = pd.to_numeric(df_grp_aa[col], errors='coerce')
    df_grp_aa = df_grp_aa.dropna()
    df_grp_aa = df_grp_aa[df_grp_aa['age_at_exit'] > df_grp_aa['age_at_entry']].copy()

    if len(df_grp_aa) > 100:
        cph_grp = CoxPHFitter(penalizer=1e-5)
        cph_grp.fit(df_grp_aa, duration_col='age_at_exit', event_col='died',
                    entry_col='age_at_entry',
                    formula='fin_strain_binary + ' + ' + '.join(attained_covars))
        results_attained[f'{age_grp}'] = {
            'n': len(df_grp_aa), 'events': int(df_grp_aa['died'].sum()),
            'hr': float(np.exp(cph_grp.params_['fin_strain_binary'])),
            'ci_lower': float(np.exp(cph_grp.confidence_intervals_.loc['fin_strain_binary', '95% lower-bound'])),
            'ci_upper': float(np.exp(cph_grp.confidence_intervals_.loc['fin_strain_binary', '95% upper-bound'])),
            'p_value': float(cph_grp.summary.loc['fin_strain_binary', 'p'])
        }
        print(f"  {age_grp}: HR={results_attained[f'{age_grp}']['hr']:.2f} "
              f"({results_attained[f'{age_grp}']['ci_lower']:.2f}-{results_attained[f'{age_grp}']['ci_upper']:.2f}), "
              f"P={results_attained[f'{age_grp}']['p_value']:.4f}")

# Save attained-age results
aa_table = []
for key, result in results_attained.items():
    aa_table.append({
        'Analysis': f'Attained-age ({key})',
        'N': result['n'], 'Deaths': result['events'],
        'HR': f"{result['hr']:.2f}",
        'CI': f"{result['ci_lower']:.2f}-{result['ci_upper']:.2f}",
        'P': f"{result['p_value']:.4f}" if result['p_value'] >= 0.0005 else '<.001'
    })
pd.DataFrame(aa_table).to_csv(TABLES_DIR / 'etable_attained_age.csv', index=False)
print("  Saved: etable_attained_age.csv")

# ============================================================================
# STEP 15: INCOME/WEALTH SPLINE SENSITIVITY
# ============================================================================

print("\n" + "="*80)
print("STEP 15: Restricted cubic spline sensitivity for income/wealth")
print("="*80)

def restricted_cubic_spline_basis(x, knots):
    """Create restricted cubic spline basis (Harrell method).
    Returns DataFrame with k-2 spline columns (for k knots)."""
    k = len(knots)
    bases = {}
    for j in range(1, k - 1):
        h_j = ((np.maximum(0, x - knots[j])**3 -
                 np.maximum(0, x - knots[-2])**3 *
                 (knots[-1] - knots[j]) / (knots[-1] - knots[-2]) +
                 np.maximum(0, x - knots[-1])**3 *
                 (knots[-2] - knots[j]) / (knots[-1] - knots[-2])) /
                (knots[-1] - knots[0])**2)
        bases[f'rcs_{j}'] = h_j
    return pd.DataFrame(bases, index=x.index)

# Get complete cases for spline model
spline_vars = ['fin_strain_binary', 'followup_years', 'died'] + model3b_vars
df_spline = df[spline_vars].dropna().copy()
for col in df_spline.columns:
    df_spline[col] = pd.to_numeric(df_spline[col], errors='coerce')
df_spline = df_spline.dropna()

# 5 knots at Harrell percentiles (5, 27.5, 50, 72.5, 95)
for var_name in ['asinh_income', 'asinh_wealth']:
    pcts = [5, 27.5, 50, 72.5, 95]
    knots = [np.percentile(df_spline[var_name], p) for p in pcts]
    rcs_basis = restricted_cubic_spline_basis(df_spline[var_name], knots)
    rcs_basis.columns = [f'{var_name}_{c}' for c in rcs_basis.columns]
    df_spline = pd.concat([df_spline, rcs_basis], axis=1)

# Replace linear income/wealth with spline terms
spline_covars = [v for v in model3b_vars if v not in ['asinh_income', 'asinh_wealth']]
spline_extra = [c for c in df_spline.columns if c.startswith('asinh_income_rcs') or c.startswith('asinh_wealth_rcs')]
spline_covars_full = spline_covars + ['asinh_income', 'asinh_wealth'] + spline_extra

cph_spline = CoxPHFitter(penalizer=1e-5)
cph_spline.fit(df_spline, duration_col='followup_years', event_col='died',
               formula='fin_strain_binary + ' + ' + '.join(spline_covars_full))

results_spline = {
    'n': len(df_spline),
    'events': int(df_spline['died'].sum()),
    'hr': float(np.exp(cph_spline.params_['fin_strain_binary'])),
    'ci_lower': float(np.exp(cph_spline.confidence_intervals_.loc['fin_strain_binary', '95% lower-bound'])),
    'ci_upper': float(np.exp(cph_spline.confidence_intervals_.loc['fin_strain_binary', '95% upper-bound'])),
    'p_value': float(cph_spline.summary.loc['fin_strain_binary', 'p'])
}

# Compare with linear model
print(f"\n  Linear Model 3b: HR = {results_mortality['model3b']['hr']:.2f} "
      f"({results_mortality['model3b']['ci_lower']:.2f}-{results_mortality['model3b']['ci_upper']:.2f})")
print(f"  Spline Model 3b: HR = {results_spline['hr']:.2f} "
      f"({results_spline['ci_lower']:.2f}-{results_spline['ci_upper']:.2f}), P = {results_spline['p_value']:.4f}")

spline_df = pd.DataFrame([{
    'Model': 'Linear income/wealth (Model 3b)',
    'N': results_mortality['model3b']['n'],
    'HR': f"{results_mortality['model3b']['hr']:.2f}",
    'CI': f"{results_mortality['model3b']['ci_lower']:.2f}-{results_mortality['model3b']['ci_upper']:.2f}",
    'P': f"{results_mortality['model3b']['p_value']:.4f}"
}, {
    'Model': 'RCS income/wealth (5 knots)',
    'N': results_spline['n'],
    'HR': f"{results_spline['hr']:.2f}",
    'CI': f"{results_spline['ci_lower']:.2f}-{results_spline['ci_upper']:.2f}",
    'P': f"{results_spline['p_value']:.4f}"
}])
spline_df.to_csv(TABLES_DIR / 'etable_spline_sensitivity.csv', index=False)
print("  Saved: etable_spline_sensitivity.csv")

# ============================================================================
# STEP 16: INSURANCE/EMPLOYMENT ANALYSIS (MEDICARE HYPOTHESIS)
# ============================================================================

print("\n" + "="*80)
print("STEP 16: Insurance and employment analysis (Medicare hypothesis)")
print("="*80)

# Descriptive: insurance coverage by age group × financial strain
print("\nInsurance coverage and employment by age group × financial strain:")
for age_grp in ['<65 years', '>=65 years']:
    df_grp = df[df['age_group'] == age_grp]
    for strain_val, strain_label in [(0, 'No/Low'), (1, 'High')]:
        df_sub = df_grp[df_grp['fin_strain_binary'] == strain_val]
        n = len(df_sub)
        medicare_pct = df_sub['has_medicare'].mean() * 100 if df_sub['has_medicare'].notna().sum() > 0 else np.nan
        medicaid_pct = df_sub['has_medicaid'].mean() * 100 if df_sub['has_medicaid'].notna().sum() > 0 else np.nan
        employer_pct = df_sub['has_employer_plan'].mean() * 100 if df_sub['has_employer_plan'].notna().sum() > 0 else np.nan
        any_ins_pct = df_sub['any_insurance'].mean() * 100 if df_sub['any_insurance'].notna().sum() > 0 else np.nan
        work_pct = df_sub['working'].mean() * 100 if df_sub['working'].notna().sum() > 0 else np.nan
        print(f"  {age_grp}, {strain_label} strain (N={n}): "
              f"Medicare={medicare_pct:.1f}%, Medicaid={medicaid_pct:.1f}%, "
              f"Employer={employer_pct:.1f}%, Any={any_ins_pct:.1f}%, Working={work_pct:.1f}%")

# Save descriptive table
insurance_desc = []
for age_grp in ['<65 years', '>=65 years']:
    df_grp = df[df['age_group'] == age_grp]
    for strain_val, strain_label in [(0, 'No/Low'), (1, 'High')]:
        df_sub = df_grp[df_grp['fin_strain_binary'] == strain_val]
        row = {
            'Age_Group': age_grp, 'Financial_Strain': strain_label, 'N': len(df_sub),
            'Medicare_pct': f"{df_sub['has_medicare'].mean()*100:.1f}" if df_sub['has_medicare'].notna().sum() > 0 else 'NA',
            'Medicaid_pct': f"{df_sub['has_medicaid'].mean()*100:.1f}" if df_sub['has_medicaid'].notna().sum() > 0 else 'NA',
            'Employer_pct': f"{df_sub['has_employer_plan'].mean()*100:.1f}" if df_sub['has_employer_plan'].notna().sum() > 0 else 'NA',
            'Any_Insurance_pct': f"{df_sub['any_insurance'].mean()*100:.1f}" if df_sub['any_insurance'].notna().sum() > 0 else 'NA',
            'Working_pct': f"{df_sub['working'].mean()*100:.1f}" if df_sub['working'].notna().sum() > 0 else 'NA',
        }
        insurance_desc.append(row)

pd.DataFrame(insurance_desc).to_csv(TABLES_DIR / 'etable_insurance_descriptive.csv', index=False)
print("  Saved: etable_insurance_descriptive.csv")

# Within <65: stratify by insurance status
print("\nWithin <65: financial strain × insurance status:")
df_under65 = df[df['age_group'] == '<65 years'].copy()

results_insurance = {}
for ins_val, ins_label in [(0, 'Uninsured'), (1, 'Insured')]:
    df_ins = df_under65[df_under65['any_insurance'] == ins_val].copy()
    if df_ins['fin_strain_binary'].sum() >= 20 and len(df_ins) >= 100:
        result = fit_cox_model(df_ins, model3b_vars)
        if result:
            results_insurance[ins_label] = result
            print(f"  {ins_label}: N={result['n']}, Deaths={result['events']}, "
                  f"HR={result['hr']:.2f} ({result['ci_lower']:.2f}-{result['ci_upper']:.2f}), "
                  f"P={result['p_value']:.4f}")
    else:
        print(f"  {ins_label}: insufficient sample (N={len(df_ins)}, events={df_ins['died'].sum():.0f})")

# Interaction model: fin_strain_binary × any_insurance within <65
print("\nInteraction: financial strain × insurance within <65:")
ins_int_vars = list(set(['fin_strain_binary', 'followup_years', 'died', 'any_insurance'] + model3b_vars))
df_ins_int = df_under65[ins_int_vars].dropna().copy()
for col in df_ins_int.columns:
    df_ins_int[col] = pd.to_numeric(df_ins_int[col], errors='coerce')
df_ins_int = df_ins_int.dropna()

if len(df_ins_int) >= 100:
    df_ins_int['strain_x_insurance'] = df_ins_int['fin_strain_binary'] * df_ins_int['any_insurance']
    cph_ins_int = CoxPHFitter(penalizer=1e-5)
    ins_int_formula = ('fin_strain_binary + any_insurance + strain_x_insurance + ' +
                       ' + '.join(model3b_vars))
    cph_ins_int.fit(df_ins_int, duration_col='followup_years', event_col='died',
                    formula=ins_int_formula)
    ins_int_p = float(cph_ins_int.summary.loc['strain_x_insurance', 'p'])
    ins_int_hr = float(np.exp(cph_ins_int.params_['strain_x_insurance']))
    results_insurance['interaction_p'] = ins_int_p
    results_insurance['interaction_hr'] = ins_int_hr
    print(f"  Interaction HR: {ins_int_hr:.2f}, P: {ins_int_p:.4f}")

# Save insurance analysis results
ins_table = []
for key, result in results_insurance.items():
    if isinstance(result, dict) and 'hr' in result:
        ins_table.append({
            'Analysis': f'<65, {key}',
            'N': result['n'], 'Deaths': result['events'],
            'HR': f"{result['hr']:.2f}",
            'CI': f"{result['ci_lower']:.2f}-{result['ci_upper']:.2f}",
            'P': f"{result['p_value']:.4f}" if result['p_value'] >= 0.0005 else '<.001'
        })
pd.DataFrame(ins_table).to_csv(TABLES_DIR / 'etable_insurance_analysis.csv', index=False)
print("  Saved: etable_insurance_analysis.csv")

# ============================================================================
# STEP 17: SAVE COMPREHENSIVE RESULTS
# ============================================================================

print("\n" + "="*80)
print("STEP 17: Saving all results")
print("="*80)

# Main results table
main_table = []
for model, label in [
    ('model1', 'Model 1: Unadjusted'),
    ('model2', 'Model 2: Age + Sex'),
    ('model3', 'Model 3: Fully adjusted'),
    ('model3b', 'Model 3b: + Income + Wealth (PRIMARY)'),
    ('model4_cesd_confounder', 'Model 4: + CES-D (confounder)'),
    ('model4b_all_psych', 'Model 4b: + All psychological'),
    ('ps_matched', 'PS-matched'),
    ('exclude_1yr', 'Excluding 1-year deaths'),
    ('exclude_2yr', 'Excluding 2-year deaths'),
]:
    if model in results_mortality and results_mortality[model]:
        r = results_mortality[model]
        main_table.append({
            'Model': label,
            'N': r['n'],
            'Deaths': r['events'],
            'HR': f"{r['hr']:.2f}",
            'CI': f"{r['ci_lower']:.2f}-{r['ci_upper']:.2f}",
            'P': f"{r['p_value']:.4f}" if r['p_value'] >= 0.0005 else '<.001'
        })

pd.DataFrame(main_table).to_csv(TABLES_DIR / 'table2_mortality_revised.csv', index=False)
print("  Saved: table2_mortality_revised.csv")

# Age-stratified table
age_table = []
for key, result in results_age_strat.items():
    if isinstance(result, dict) and 'hr' in result:
        age_table.append({
            'Analysis': key,
            'N': result.get('n', 'N/A'),
            'Deaths': result.get('events', 'N/A'),
            'HR': f"{result['hr']:.2f}",
            'CI': f"{result['ci_lower']:.2f}-{result['ci_upper']:.2f}",
            'P': f"{result['p_value']:.4f}" if result['p_value'] >= 0.0005 else '<.001'
        })

pd.DataFrame(age_table).to_csv(TABLES_DIR / 'table3_age_stratified_revised.csv', index=False)
print("  Saved: table3_age_stratified_revised.csv")

# Cardiac results table
cardiac_table = []
for key, result in results_cardiac.items():
    if result:
        cardiac_table.append({
            'Analysis': key,
            'N': result['n'],
            'Events': result['events'],
            'HR': f"{result['hr']:.2f}",
            'CI': f"{result['ci_lower']:.2f}-{result['ci_upper']:.2f}",
            'P': f"{result['p_value']:.4f}" if result['p_value'] >= 0.0005 else '<.001'
        })

pd.DataFrame(cardiac_table).to_csv(TABLES_DIR / 'table4_cardiac_revised.csv', index=False)
print("  Saved: table4_cardiac_revised.csv")

# Save comprehensive manifest
manifest = {
    'analysis_version': 'v5',
    'date': '2026-02',
    'analyses_implemented': [
        'Complete-case analysis with <2% missingness',
        'PS matching with Model 3b covariates',
        'Incident HD censoring for biennial ascertainment',
        'CES-D as explicit confounder in Model 4b',
        'Age interaction with binary and continuous tests',
        'Cross-classification strain x SES',
        'Absolute risk metrics (KM and standardized)',
        'Category-specific dose-response'
    ],
    'sample': {
        'n_total': len(df),
        'n_high_strain': int(df['fin_strain_binary'].sum()),
        'pct_high_strain': float(df['fin_strain_binary'].mean() * 100),
        'n_deaths': int(df['died'].sum()),
        'mean_followup_years': float(df['followup_years'].mean()),
        'n_complete_case_model3b': results_mortality['model3b']['n'] if results_mortality.get('model3b') else None
    },
    'model1_unadjusted': results_mortality.get('model1'),
    'model2_age_sex': results_mortality.get('model2'),
    'primary_result_model3b': results_mortality.get('model3b'),
    'age_interaction': {
        'binary_p': results_age_strat.get('interaction_term', {}).get('p_value'),
        'continuous_p': results_age_strat.get('continuous_interaction_p'),
        'under65_hr': results_age_strat.get('<65 years_model3b', {}).get('hr'),
        'under65_ci_lower': results_age_strat.get('<65 years_model3b', {}).get('ci_lower'),
        'under65_ci_upper': results_age_strat.get('<65 years_model3b', {}).get('ci_upper'),
        'under65_p': results_age_strat.get('<65 years_model3b', {}).get('p_value'),
        'over65_hr': results_age_strat.get('>=65 years_model3b', {}).get('hr'),
        'over65_ci_lower': results_age_strat.get('>=65 years_model3b', {}).get('ci_lower'),
        'over65_ci_upper': results_age_strat.get('>=65 years_model3b', {}).get('ci_upper'),
        'over65_p': results_age_strat.get('>=65 years_model3b', {}).get('p_value')
    },
    'sensitivity_analyses': {
        'model3_no_ses': results_mortality.get('model3'),
        'ps_matched': results_mortality.get('ps_matched'),
        'exclude_1yr': results_mortality.get('exclude_1yr'),
        'exclude_2yr': results_mortality.get('exclude_2yr'),
        'cesd_confounder': results_mortality.get('model4_cesd_confounder')
    },
    'cardiac': results_cardiac,
    'evalues': evalues,
    'evalues_weighted': evalues_weighted,
    'survival_probabilities': survival_results,
    'dose_response_trend_p': float(trend_p),
    'weighted_primary': results_weighted,
    'weighted_age_interaction': results_weighted_age,
    'weighted_age_specific_hrs': wt_age_specific_hrs,
    'attained_age': results_attained,
    'spline_sensitivity': results_spline,
    'insurance_analysis': {
        'descriptive': insurance_desc,
        'stratified': {k: v for k, v in results_insurance.items() if isinstance(v, dict) and 'hr' in v},
        'interaction_p': results_insurance.get('interaction_p'),
        'interaction_hr': results_insurance.get('interaction_hr')
    }
}

with open(OUTPUT_DIR / 'results_manifest_v5.json', 'w') as f:
    json.dump(manifest, f, indent=2, default=str)
print("  Saved: results_manifest_v5.json")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("ANALYSIS COMPLETE - KEY FINDINGS SUMMARY")
print("="*80)

print(f"""
SAMPLE:
  Total N = {len(df)}
  High financial strain: {df['fin_strain_binary'].sum()} ({df['fin_strain_binary'].mean()*100:.1f}%)
  Deaths: {df['died'].sum()}
  Person-years: {df['followup_years'].sum():.0f}
  Mean follow-up: {df['followup_years'].mean():.1f} years

PRIMARY RESULTS (Model 3b - with income and wealth):
  HR = {results_mortality['model3b']['hr']:.2f} (95% CI: {results_mortality['model3b']['ci_lower']:.2f}-{results_mortality['model3b']['ci_upper']:.2f})
  P = {results_mortality['model3b']['p_value']:.4f}
  N complete cases = {results_mortality['model3b']['n']}

CES-D AS CONFOUNDER (sensitivity analysis):
  HR = {results_mortality['model4_cesd_confounder']['hr']:.2f} (95% CI: {results_mortality['model4_cesd_confounder']['ci_lower']:.2f}-{results_mortality['model4_cesd_confounder']['ci_upper']:.2f})
  P = {results_mortality['model4_cesd_confounder']['p_value']:.4f}

AGE INTERACTION:
  <65 years: HR = {results_age_strat['<65 years_model3b']['hr']:.2f} ({results_age_strat['<65 years_model3b']['ci_lower']:.2f}-{results_age_strat['<65 years_model3b']['ci_upper']:.2f})
  >=65 years: HR = {results_age_strat['>=65 years_model3b']['hr']:.2f} ({results_age_strat['>=65 years_model3b']['ci_lower']:.2f}-{results_age_strat['>=65 years_model3b']['ci_upper']:.2f})
  Binary interaction P = {results_age_strat['interaction_term']['p_value']:.4f}
  Continuous interaction P = {results_age_strat['continuous_interaction_p']:.4f}

SENSITIVITY ANALYSES:
  PS-matched: HR = {results_mortality['ps_matched']['hr']:.2f}, P = {results_mortality['ps_matched']['p_value']:.4f}
  Exclude 1 yr: HR = {results_mortality['exclude_1yr']['hr']:.2f}, P = {results_mortality['exclude_1yr']['p_value']:.4f}
  Exclude 2 yr: HR = {results_mortality['exclude_2yr']['hr']:.2f}, P = {results_mortality['exclude_2yr']['p_value']:.4f}

E-VALUES:
  Model 3b: E-value = {evalues['Model 3b (Primary)']['evalue']:.2f}

INCIDENT HEART DISEASE (cause-specific):
  HR = {results_cardiac['cause_specific']['hr']:.2f} ({results_cardiac['cause_specific']['ci_lower']:.2f}-{results_cardiac['cause_specific']['ci_upper']:.2f})
  P = {results_cardiac['cause_specific']['p_value']:.4f}
  NOTE: Fine-Gray competing risks model run separately in Stata 18 stcrreg
""")

print("\nAll tables saved to:", TABLES_DIR)
print("Results manifest saved to:", OUTPUT_DIR / 'results_manifest_v5.json')
print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
