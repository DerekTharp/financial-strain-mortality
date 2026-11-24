"""
Builds the analytic sample from the RAND HRS Longitudinal File and the HRS Cross-Wave Tracker,
then fits the unweighted Cox models.

Writes output/analytic_sample.csv and, in output/tables/: cohort_summary.csv, missingness.csv,
table1_baseline.csv, mortality_unweighted.csv, age_stratified_unweighted.csv,
cardiac_unweighted.csv, dose_response.csv, attained_age_unweighted.csv and
spline_sensitivity.csv.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from lifelines import CoxPHFitter
import pyreadstat

from model_specs import MODEL3B_COVARS, MODEL3_COVARS, CARDIAC_COVARS, PENALIZER

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "randhrs1992_2022v1_STATA"
OUTPUT_DIR = PROJECT_DIR / "output"
TABLES_DIR = OUTPUT_DIR / "tables"
TABLES_DIR.mkdir(exist_ok=True, parents=True)

HRS_FILE = DATA_DIR / "randhrs1992_2022v1.dta"

print("Loading HRS data...")
hrs_data, _ = pyreadstat.read_dta(str(HRS_FILE),
                                      usecols=[
                                          'hhidpn',
                                          # Demographics
                                          'r8agey_e', 'ragender', 'raracem', 'rahispan',
                                          # SES
                                          'raedyrs', 'r8mstat', 'h8itot', 'h8atotb',
                                          # Health behaviors
                                          'r8smoken', 'r8bmi',
                                          # Baseline conditions
                                          'r8hibpe', 'r8diabe', 'r8hearte', 'r8stroke',
                                          # Psychosocial
                                          'r8cesd', 'r8shlt',
                                          # Financial strain (Leave-Behind)
                                          'r8lbfinprb',
                                          # Leave-behind weight
                                          'r8lbwgtr',
                                          # Mortality
                                          'raddate', 'r8iwmid',
                                          # Follow-up interviews and heart disease
                                          'r9hearte', 'r10hearte', 'r11hearte',
                                          'r12hearte', 'r13hearte', 'r14hearte',
                                          'r15hearte', 'r16hearte',
                                          'r9iwmid', 'r10iwmid', 'r11iwmid',
                                          'r12iwmid', 'r13iwmid', 'r14iwmid',
                                          'r15iwmid', 'r16iwmid',
                                      ])


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

# Valid financial strain response
has_fin_strain = df['r8lbfinprb'].notna() & df['r8lbfinprb'].isin([1, 2, 3, 4])
df = df[has_fin_strain].copy()

# Age 50 or older at baseline
df['age'] = df['r8agey_e']
age_eligible = df['age'] >= 50
df = df[age_eligible].copy()

# Financial strain: 1 no, 2 yes but not upsetting, 3 somewhat upsetting, 4 very upsetting
df['fin_strain_raw'] = df['r8lbfinprb']
df['fin_strain_binary'] = np.where(df['fin_strain_raw'].isin([3, 4]), 1, 0)
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

# Marital status: married, married with spouse absent, or partnered (codes 1 to 3) versus
# separated, divorced, widowed, or never married (codes 4 to 8)
df['married_partnered'] = np.where(df['r8mstat'].isin([1, 2, 3]), 1,
                                   np.where(df['r8mstat'].isin([4, 5, 6, 7, 8]), 0,
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

# Mortality

# Vital status from the HRS Cross-Wave Tracker (2022 Final V1.0), merged on HHID+PN = hhidpn.
# Death dates follow a fixed hierarchy: the Tracker's known-deceased year and month (exit
# interview, spouse report, field report, or HRS imputation) placed at the 15th; the RAND date
# where the Tracker month is unknown; otherwise the verified National Death Index year and
# quarter placed at mid-quarter. Survivors are followed to 31 December 2022.
TRACKER_FILE = PROJECT_DIR / "trk2022v1" / "trk2022tr_r.dta"
tracker_cols = ['HHID', 'PN', 'KNOWNDECEASEDMO', 'KNOWNDECEASEDYR',
                'LASTALIVEMO', 'LASTALIVEYR', 'LASTALIVESOURCE', 'NYEAR', 'NMONTH']
tracker, _ = pyreadstat.read_dta(str(TRACKER_FILE), usecols=tracker_cols)
tracker['hhidpn'] = (tracker['HHID'].astype(str).str.zfill(6)
                     + tracker['PN'].astype(str).str.zfill(3)).astype(int)
for col in tracker_cols[2:]:
    tracker[col] = pd.to_numeric(tracker[col], errors='coerce')
df = df.merge(tracker.drop(columns=['HHID', 'PN']), on='hhidpn', how='left', validate='one_to_one')

STATA_EPOCH = pd.Timestamp('1960-01-01')
end_of_study = (pd.Timestamp('2022-12-31') - STATA_EPOCH).days


def stata_days(year, month, day=15):
    """Days since 1960-01-01 for a year/month series; NaT-safe."""
    return (pd.to_datetime(dict(year=year, month=month, day=day), errors='coerce') - STATA_EPOCH).dt.days


df['baseline_date'] = df['r8iwmid']
tracker_month_valid = df['KNOWNDECEASEDMO'].between(1, 12)
known_date = stata_days(df['KNOWNDECEASEDYR'], df['KNOWNDECEASEDMO'].where(tracker_month_valid))
# A death recorded in the month of the baseline interview cannot precede the interview:
# use RAND's day-level date for those few cases.
known_date = known_date.where(~(known_date <= df['baseline_date']), df['raddate'])
# NDI records carry a year and a calendar quarter; some carry the year alone. Quarter-dated
# deaths sit at mid-quarter and year-only deaths at 1 July, so that every verified death year
# within follow-up counts as a death (timing variants for year-only deaths feed table S3).
ndi_quarter_valid = df['NMONTH'].between(1, 4)
ndi_mid_quarter = stata_days(df['NYEAR'], df['NMONTH'] * 3 - 1)
ndi_mid_year = stata_days(df['NYEAR'], 7, day=1)
ndi_date = ndi_mid_quarter.where(ndi_quarter_valid, ndi_mid_year)
ndi_date = ndi_date.where(~(ndi_date <= df['baseline_date']), df['baseline_date'] + 15)
df['ndi_date_precision'] = np.select([df['NYEAR'].isna(), ndi_quarter_valid], ['none', 'quarter'], default='year')
df['death_date'] = np.where(known_date.notna(), known_date,
                            np.where(df['KNOWNDECEASEDYR'].notna(), df['raddate'], ndi_date))
df['death_source'] = np.select(
    [known_date.notna(), df['KNOWNDECEASEDYR'].notna(), ndi_date.notna()],
    ['tracker', 'tracker_month_unknown_rand_date', 'ndi'], default='none')
df['ndi_only_death'] = (df['death_source'] == 'ndi').astype(int)

# Last core interview and the Tracker's last documented-alive month
iw_cols = [f'r{w}iwmid' for w in range(9, 17)]
df['last_interview_date'] = df[iw_cols].max(axis=1).fillna(df['baseline_date'])
df['last_alive_date'] = stata_days(df['LASTALIVEYR'], df['LASTALIVEMO'])
ndi_quarter_end = (pd.to_datetime(dict(year=df['NYEAR'], month=df['NMONTH'] * 3, day=1), errors='coerce')
                   + pd.offsets.MonthEnd(0) - STATA_EPOCH).dt.days
df['ndi_conflict_informant'] = ((df['ndi_only_death'] == 1)
                                & (ndi_quarter_end < df['last_alive_date'] - 16)).astype(int)

# Primary rule: administrative censoring at 31 December 2022
df['died'] = np.where(df['death_date'].notna() & (df['death_date'] <= end_of_study), 1, 0)
df['end_date'] = np.where(df['died'] == 1, df['death_date'], end_of_study)
df['followup_years'] = (df['end_date'] - df['baseline_date']) / 365.25

known_year = df['KNOWNDECEASEDYR'].fillna(df['NYEAR'])
assert not ((known_year <= 2022) & (df['died'] == 0)).any(), \
    "a verified death year within follow-up is coded as a censored survivor"
year_only_ndi = (df['death_source'] == 'ndi') & (df['ndi_date_precision'] == 'year')

# Timing variants for year-only NDI deaths (table S3): 1 January and 31 December
for tag, month, day in (('jan', 1, 1), ('dec', 12, 31)):
    alt = np.maximum(stata_days(df['NYEAR'], month, day=day), df['baseline_date'] + 15)
    dd = np.where(year_only_ndi, alt, df['death_date'])
    df[f'died_ndi_{tag}'] = np.where(pd.notna(dd) & (dd <= end_of_study), 1, 0)
    df[f'followup_ndi_{tag}'] = (np.where(df[f'died_ndi_{tag}'] == 1, dd, end_of_study) - df['baseline_date']) / 365.25

# Alternative ascertainment assumptions (table S3): survivors censored when last documented
# alive; RAND death dates alone with survivors censored at their last interview; RAND death
# dates alone with administrative censoring
df['died_lastalive'] = df['died']
df['end_lastalive'] = np.where(df['died'] == 1, df['death_date'],
                               np.minimum(df['last_alive_date'], end_of_study))
df['followup_lastalive'] = (df['end_lastalive'] - df['baseline_date']) / 365.25
rand_died = df['raddate'].notna()
never_reint = df[iw_cols].isna().all(axis=1) & ~rand_died
df['died_randlastiw'] = rand_died.astype(int)
df['end_randlastiw'] = np.where(rand_died, df['raddate'],
                                np.where(never_reint, end_of_study, df['last_interview_date']))
df['followup_randlastiw'] = (df['end_randlastiw'] - df['baseline_date']) / 365.25
df['died_randadmin'] = (rand_died & (df['raddate'] <= end_of_study)).astype(int)
df['end_randadmin'] = np.where(df['died_randadmin'] == 1, df['raddate'], end_of_study)
df['followup_randadmin'] = (df['end_randadmin'] - df['baseline_date']) / 365.25

# Remove invalid follow-up
valid_followup = df['followup_years'] > 0
df = df[valid_followup].copy()

assert len(df) == 7219, f"expected 7219 participants, found {len(df)}"
assert int(df['died'].sum()) == 3635, f"expected 3635 deaths, found {int(df['died'].sum())}"

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

# Cohort and death-dating counts quoted in the text and in the footnote to table S3.
# LASTALIVESOURCE: 1 respondent's interview, 2 spouse's or partner's interview, 3 field activity.
within = df['died'] == 1
conflict = df['ndi_conflict_informant'] == 1
conflict_gap = (df['last_alive_date'] - ndi_quarter_end.reindex(df.index))[conflict] / 365.25
cohort_summary = pd.DataFrame([
    ('Participants', len(df)),
    ('Deaths within follow-up', int(within.sum())),
    ('Person-years of follow-up', round(df['followup_years'].sum())),
    ('Mean follow-up, years', round(df['followup_years'].mean(), 1)),
    ('Deaths dated from RAND where the Tracker month is unknown',
     int((within & (df['death_source'] == 'tracker_month_unknown_rand_date')).sum())),
    ('Deaths dated from the NDI', int((within & (df['death_source'] == 'ndi')).sum())),
    ('Deaths dated from the NDI year and quarter',
     int((within & (df['death_source'] == 'ndi') & (df['ndi_date_precision'] == 'quarter')).sum())),
    ('Deaths dated from the NDI year alone',
     int((within & (df['death_source'] == 'ndi') & (df['ndi_date_precision'] == 'year')).sum())),
    ('NDI records for participants with no HRS-recorded death date',
     int((df['death_source'] == 'ndi').sum())),
    ('NDI deaths dated before the last-alive report', int(conflict.sum())),
    ('Conflicting last-alive reports from respondent or spouse interviews',
     int((conflict & df['LASTALIVESOURCE'].isin([1, 2])).sum())),
    ('Years from the NDI quarter to the last-alive report, minimum', round(conflict_gap.min(), 1)),
    ('Years from the NDI quarter to the last-alive report, maximum', round(conflict_gap.max(), 1)),
], columns=['Quantity', 'Value'], dtype=object)
cohort_summary.to_csv(TABLES_DIR / 'cohort_summary.csv', index=False)

# Incident heart disease

df['baseline_cvd_free'] = (df['baseline_heart'] == 0) & (df['baseline_stroke'] == 0)

heart_waves = ['r9hearte', 'r10hearte', 'r11hearte', 'r12hearte',
               'r13hearte', 'r14hearte', 'r15hearte', 'r16hearte']
date_waves = ['r9iwmid', 'r10iwmid', 'r11iwmid', 'r12iwmid',
              'r13iwmid', 'r14iwmid', 'r15iwmid', 'r16iwmid']

def find_incident_heart(row):
    """
    First follow-up interview reporting heart disease; participants without an event are
    censored at their last interview with heart disease data.
    """
    if not row['baseline_cvd_free']:
        return pd.Series({'incident_heart': np.nan,
                         'incident_heart_date': np.nan,
                         'last_heart_obs_date': np.nan})

    last_obs_date = row['baseline_date']

    for heart_var, date_var in zip(heart_waves, date_waves):
        wave_date = row.get(date_var)
        heart_status = row.get(heart_var)

        if pd.notna(wave_date) and pd.notna(heart_status):
            last_obs_date = wave_date

            if heart_status == 1:
                return pd.Series({'incident_heart': 1,
                                 'incident_heart_date': wave_date,
                                 'last_heart_obs_date': wave_date})

    return pd.Series({'incident_heart': 0,
                     'incident_heart_date': np.nan,
                     'last_heart_obs_date': last_obs_date})

incident_data = df.apply(find_incident_heart, axis=1)
df['incident_heart'] = incident_data['incident_heart']
df['incident_heart_date'] = incident_data['incident_heart_date']
df['last_heart_obs_date'] = incident_data['last_heart_obs_date']

# Cardiac follow-up ends at the event, at the last interview with heart disease data, or at
# death if death comes before that interview
df['cardiac_end_date'] = np.where(
    df['incident_heart'] == 1,
    df['incident_heart_date'],
    np.where(
        (df['died'] == 1) & (df['death_date'] < df['last_heart_obs_date']),
        df['death_date'],
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

# Fine-Gray time: competing deaths at their death dates (cause-specific models keep
# censoring at the last interview with heart-disease data)
df['cardiac_fu_fg'] = np.where(
    df['cardiac_event_type'] == 2,
    (df['death_date'] - df['baseline_date']) / 365.25,
    df['cardiac_followup_years']
)

# Missing values, overall and by exposure group (table 1 footnote)

all_analysis_vars = (['fin_strain_binary'] + MODEL3B_COVARS
                     + ['depressed', 'srh_fairpoor', 'followup_years', 'died'])

missingness_table = []
for var in all_analysis_vars:
    missingness_table.append({
        'Variable': var,
        'N_Missing': df[var].isna().sum(),
        'N_Missing_NoLowStrain': (df[var].isna() & (df['fin_strain_binary'] == 0)).sum(),
        'N_Missing_HighStrain': (df[var].isna() & (df['fin_strain_binary'] == 1)).sum(),
    })

missingness_df = pd.DataFrame(missingness_table)
missingness_df.to_csv(TABLES_DIR / 'missingness.csv', index=False)

df.to_csv(OUTPUT_DIR / 'analytic_sample.csv', index=False)
print(f"Analytic sample: N={len(df)}, deaths={int(df['died'].sum())}")

# Table 1: baseline characteristics

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

table1_rows = []

# N row
table1_rows.append({
    'Variable': 'N',
    'No/Low Financial Strain': f'{len(df_low):,}',
    'High Financial Strain': f'{len(df_high):,}',
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
        'SMD': f"{smd:.2f}"
    })

# Continuous: median (IQR) for income/wealth
for var, label in [('income', 'Household income, median (IQR), $'),
                   ('wealth', 'Total wealth, median (IQR), $')]:
    smd = calculate_smd(var, df_high, df_low)
    row = {'Variable': label}
    for grp, grp_label in [(df_low, 'No/Low Financial Strain'),
                            (df_high, 'High Financial Strain')]:
        med = grp[var].median()
        q25 = grp[var].quantile(0.25)
        q75 = grp[var].quantile(0.75)
        row[grp_label] = f"{med:,.0f} ({q25:,.0f}-{q75:,.0f})"
    row['SMD'] = f"{smd:.2f}"
    table1_rows.append(row)

# Binary: Female
smd = calculate_smd('female', df_high, df_low)
table1_rows.append({
    'Variable': 'Female, No. (%)',
    'No/Low Financial Strain': f"{int(df_low['female'].sum())} ({df_low['female'].mean()*100:.1f}%)",
    'High Financial Strain': f"{int(df_high['female'].sum())} ({df_high['female'].mean()*100:.1f}%)",
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
    race_smd = calculate_smd(race_indicator_map[race], df_high, df_low)
    table1_rows.append({
        'Variable': f'  {label}',
        'No/Low Financial Strain': f"{low_n} ({low_pct:.1f}%)",
        'High Financial Strain': f"{high_n} ({high_pct:.1f}%)",
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
        'SMD': f"{smd:.2f}"
    })

# Outcome variables
smd = calculate_smd('died', df_high, df_low)
table1_rows.append({
    'Variable': 'Died during follow-up, No. (%)',
    'No/Low Financial Strain': f"{int(df_low['died'].sum())} ({df_low['died'].mean()*100:.1f}%)",
    'High Financial Strain': f"{int(df_high['died'].sum())} ({df_high['died'].mean()*100:.1f}%)",
    'SMD': f"{smd:.2f}"
})

smd = calculate_smd('followup_years', df_high, df_low)
table1_rows.append({
    'Variable': 'Follow-up, mean (SD), y',
    'No/Low Financial Strain': f"{df_low['followup_years'].mean():.1f} ({df_low['followup_years'].std():.1f})",
    'High Financial Strain': f"{df_high['followup_years'].mean():.1f} ({df_high['followup_years'].std():.1f})",
    'SMD': f"{smd:.2f}"
})

table1_df = pd.DataFrame(table1_rows)
table1_df.to_csv(TABLES_DIR / 'table1_baseline.csv', index=False)


def fit_cox_model(df, covariates, exposure='fin_strain_binary',
                  duration='followup_years', event='died'):
    """Fit a Cox model on complete cases and return the estimate for the exposure."""
    model_vars = [exposure, duration, event] + covariates

    df_model = df[model_vars].dropna().copy()
    # Ensure all columns are numeric (pyreadstat can produce object dtypes)
    for col in df_model.columns:
        df_model[col] = pd.to_numeric(df_model[col], errors='coerce')
    df_model = df_model.dropna()

    if df_model[exposure].nunique() < 2 or len(df_model) < 100:
        return None

    cph = CoxPHFitter(penalizer=PENALIZER)
    cph.fit(df_model, duration_col=duration, event_col=event,
            formula=f"{exposure} + " + " + ".join(covariates))

    hr = np.exp(cph.params_[exposure])
    ci = np.exp(cph.confidence_intervals_.loc[exposure])
    p = cph.summary.loc[exposure, 'p']

    return {
        'n': len(df_model),
        'events': int(df_model[event].sum()),
        'hr': float(hr),
        'ci_lower': float(ci['95% lower-bound']),
        'ci_upper': float(ci['95% upper-bound']),
        'p_value': float(p),
    }


# Unweighted mortality models
print("Fitting unweighted models...")

results_mortality = {}

# Model 3: demographics, behaviours, health conditions
results_mortality['model3'] = fit_cox_model(df, MODEL3_COVARS)

# Model 3b: adds income and wealth
results_mortality['model3b'] = fit_cox_model(df, MODEL3B_COVARS)

# Model 3b among participants younger than 65 (age stays in the model)
results_age_strat = {}
df_grp = df[df['age_group'] == '<65 years'].copy()
result = fit_cox_model(df_grp, MODEL3B_COVARS)
if result:
    results_age_strat['<65 years_model3b'] = result

# Excluding deaths in the first 1 and 2 years of follow-up
for exclude_years in [1, 2]:
    df_rev = df.copy()
    early_deaths = (df_rev['died'] == 1) & (df_rev['followup_years'] < exclude_years)
    df_rev = df_rev[~early_deaths].copy()
    df_rev['followup_years_adj'] = df_rev['followup_years'] - exclude_years
    df_rev = df_rev[df_rev['followup_years_adj'] > 0].copy()

    result = fit_cox_model(df_rev, MODEL3B_COVARS, duration='followup_years_adj')
    if result:
        results_mortality[f'exclude_{exclude_years}yr'] = result

# Incident heart disease: cause-specific Cox models, overall and by age group

df_cardiac = df[df['baseline_cvd_free'] == True].copy()
df_cardiac = df_cardiac[df_cardiac['cardiac_followup_years'] > 0].copy()

results_cardiac = {}

results_cardiac['cause_specific'] = fit_cox_model(
    df_cardiac, CARDIAC_COVARS,
    duration='cardiac_followup_years', event='incident_heart'
)

for age_grp in ['<65 years', '>=65 years']:
    df_grp = df_cardiac[df_cardiac['age_group'] == age_grp]
    if df_grp['incident_heart'].sum() >= 20:
        result = fit_cox_model(df_grp, CARDIAC_COVARS,
                               duration='cardiac_followup_years', event='incident_heart')
        if result:
            results_cardiac[f'{age_grp}'] = result

# Age interaction (table S4 footnote): an under-65 indicator and its product with strain
# added to the cause-specific model. The row holds the ratio of the two age-specific HRs.
df_cardiac['strain_x_under65'] = df_cardiac['fin_strain_binary'] * df_cardiac['age_under65']
results_cardiac['strain x age <65 interaction (ratio of HRs)'] = fit_cox_model(
    df_cardiac, ['fin_strain_binary', 'age_under65'] + CARDIAC_COVARS,
    exposure='strain_x_under65',
    duration='cardiac_followup_years', event='incident_heart'
)

# Dose-response across the four responses

df['strain_not_upsetting'] = np.where(df['fin_strain_ordinal'] == 1, 1, 0)
df['strain_somewhat'] = np.where(df['fin_strain_ordinal'] == 2, 1, 0)
df['strain_very'] = np.where(df['fin_strain_ordinal'] == 3, 1, 0)

dose_vars = MODEL3B_COVARS + ['strain_not_upsetting', 'strain_somewhat', 'strain_very']
df_dose = df[['followup_years', 'died'] + dose_vars].dropna()
for col in df_dose.columns:
    df_dose[col] = pd.to_numeric(df_dose[col], errors='coerce')
df_dose = df_dose.dropna()

cph_dose = CoxPHFitter(penalizer=PENALIZER)
cph_dose.fit(df_dose, duration_col='followup_years', event_col='died',
             formula=' + '.join(dose_vars))

dose_results = []
for cat, label, dummy_var in [(1, 'No strain (ref)', None),
                               (2, 'Yes, not upsetting', 'strain_not_upsetting'),
                               (3, 'Somewhat upsetting', 'strain_somewhat'),
                               (4, 'Very upsetting', 'strain_very')]:
    n = (df['fin_strain_raw'] == cat).sum()

    if dummy_var:
        hr = np.exp(cph_dose.params_[dummy_var])
        ci = np.exp(cph_dose.confidence_intervals_.loc[dummy_var])
        hr_str = f"{hr:.2f} ({ci['95% lower-bound']:.2f}-{ci['95% upper-bound']:.2f})"
    else:
        hr_str = "1.00 (ref)"

    dose_results.append({'Category': label, 'N': n, 'HR_95CI': hr_str, 'P': ''})

# Linear trend across the four responses
df_trend = df[['followup_years', 'died', 'fin_strain_ordinal'] + MODEL3B_COVARS].dropna()
for col in df_trend.columns:
    df_trend[col] = pd.to_numeric(df_trend[col], errors='coerce')
df_trend = df_trend.dropna()
cph_trend = CoxPHFitter(penalizer=PENALIZER)
cph_trend.fit(df_trend, duration_col='followup_years', event_col='died',
              formula='fin_strain_ordinal + ' + ' + '.join(MODEL3B_COVARS))
trend_p = cph_trend.summary.loc['fin_strain_ordinal', 'p']
dose_results.append({'Category': 'Linear trend', 'N': '', 'HR_95CI': '', 'P': f"{trend_p:.3f}"})

pd.DataFrame(dose_results).to_csv(TABLES_DIR / 'dose_response.csv', index=False)

# Attained age as the time scale, with delayed entry at baseline age

# Age leaves the covariates because it is the time scale
attained_covars = [v for v in MODEL3B_COVARS if v != 'age']

results_attained = {}

aa_vars = ['fin_strain_binary', 'age_at_entry', 'age_at_exit', 'died'] + attained_covars
df_aa = df[aa_vars].dropna().copy()
for col in df_aa.columns:
    df_aa[col] = pd.to_numeric(df_aa[col], errors='coerce')
df_aa = df_aa.dropna()
df_aa = df_aa[df_aa['age_at_exit'] > df_aa['age_at_entry']].copy()

cph_aa = CoxPHFitter(penalizer=PENALIZER)
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

for age_grp in ['<65 years', '>=65 years']:
    df_grp = df[df['age_group'] == age_grp].copy()
    df_grp_aa = df_grp[aa_vars].dropna().copy()
    for col in df_grp_aa.columns:
        df_grp_aa[col] = pd.to_numeric(df_grp_aa[col], errors='coerce')
    df_grp_aa = df_grp_aa.dropna()
    df_grp_aa = df_grp_aa[df_grp_aa['age_at_exit'] > df_grp_aa['age_at_entry']].copy()

    if len(df_grp_aa) > 100:
        cph_grp = CoxPHFitter(penalizer=PENALIZER)
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

aa_table = []
for key, result in results_attained.items():
    aa_table.append({
        'Analysis': f'Attained-age ({key})',
        'N': result['n'], 'Deaths': result['events'],
        'HR': f"{result['hr']:.2f}",
        'CI': f"{result['ci_lower']:.2f}-{result['ci_upper']:.2f}",
        'P': f"{result['p_value']:.4f}" if result['p_value'] >= 0.0005 else '<.001'
    })
pd.DataFrame(aa_table).to_csv(TABLES_DIR / 'attained_age_unweighted.csv', index=False)

# Restricted cubic splines for income and wealth

def restricted_cubic_spline_basis(x, knots):
    """Create restricted cubic spline basis (Harrell method).
    Returns DataFrame with k-2 spline columns (for k knots)."""
    k = len(knots)
    bases = {}
    for j in range(k - 2):   # k-2 nonlinear terms use knots 0..k-3 against the last two knots
        h_j = ((np.maximum(0, x - knots[j])**3 -
                 np.maximum(0, x - knots[-2])**3 *
                 (knots[-1] - knots[j]) / (knots[-1] - knots[-2]) +
                 np.maximum(0, x - knots[-1])**3 *
                 (knots[-2] - knots[j]) / (knots[-1] - knots[-2])) /
                (knots[-1] - knots[0])**2)
        bases[f'rcs_{j}'] = h_j
    return pd.DataFrame(bases, index=x.index)

spline_vars = ['fin_strain_binary', 'followup_years', 'died'] + MODEL3B_COVARS
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

# Spline terms enter alongside the linear income and wealth terms
spline_covars = [v for v in MODEL3B_COVARS if v not in ['asinh_income', 'asinh_wealth']]
spline_extra = [c for c in df_spline.columns if c.startswith('asinh_income_rcs') or c.startswith('asinh_wealth_rcs')]
spline_covars_full = spline_covars + ['asinh_income', 'asinh_wealth'] + spline_extra

cph_spline = CoxPHFitter(penalizer=PENALIZER)
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
spline_df.to_csv(TABLES_DIR / 'spline_sensitivity.csv', index=False)

# Save the model tables. The unrounded estimate and lower limit are kept for the E-values
# that 11_survey_prep.py computes.

main_table = []
for model, label in [
    ('model3', 'Model 3'),
    ('model3b', 'Model 3b'),
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
            'P': f"{r['p_value']:.4f}" if r['p_value'] >= 0.0005 else '<.001',
            'HR_unrounded': r['hr'],
            'CI_lower_unrounded': r['ci_lower'],
        })

pd.DataFrame(main_table).to_csv(TABLES_DIR / 'mortality_unweighted.csv', index=False)

age_table = []
for key, result in results_age_strat.items():
    if isinstance(result, dict) and 'hr' in result:
        age_table.append({
            'Analysis': key,
            'N': result.get('n', 'N/A'),
            'Deaths': result.get('events', 'N/A'),
            'HR': f"{result['hr']:.2f}",
            'CI': f"{result['ci_lower']:.2f}-{result['ci_upper']:.2f}",
            'P': f"{result['p_value']:.4f}" if result['p_value'] >= 0.0005 else '<.001',
            'HR_unrounded': result['hr'],
            'CI_lower_unrounded': result['ci_lower'],
        })

pd.DataFrame(age_table).to_csv(TABLES_DIR / 'age_stratified_unweighted.csv', index=False)

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

pd.DataFrame(cardiac_table).to_csv(TABLES_DIR / 'cardiac_unweighted.csv', index=False)

print(f"Saved tables to {TABLES_DIR}")
