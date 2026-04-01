"""Export analytic sample with HRS survey design variables (strata, PSU) for Stata.

Run after 01_primary_analysis.py. Merges raestrat and raehsamp from the RAND HRS
file onto the analytic sample and saves as analytic_sample_v5_survey.csv for use
by 09_survey_design_cox.do.
"""

import pandas as pd
import pyreadstat
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
HRS_FILE = PROJECT_DIR / 'randhrs1992_2022v1_STATA' / 'randhrs1992_2022v1.dta'
OUTPUT_DIR = PROJECT_DIR / 'output'

# Load analytic sample
df = pd.read_csv(OUTPUT_DIR / 'analytic_sample_v5.csv')
print(f"Analytic sample: {len(df)} observations")

# Load survey design variables from RAND HRS
survey_vars, _ = pyreadstat.read_dta(str(HRS_FILE),
                                      usecols=['hhidpn', 'raestrat', 'raehsamp'])
print(f"RAND HRS: {len(survey_vars)} observations")

# Merge
df_merged = df.merge(survey_vars, on='hhidpn', how='left')
print(f"After merge: {len(df_merged)} observations")

# Report coverage
n_valid = df_merged['raestrat'].notna().sum()
print(f"Valid strata/PSU: {n_valid} ({100*n_valid/len(df_merged):.1f}%)")
print(f"Strata: {int(df_merged['raestrat'].nunique())} unique values")
print(f"PSUs per stratum: {df_merged.groupby('raestrat')['raehsamp'].nunique().unique()}")

# Save
outpath = OUTPUT_DIR / 'analytic_sample_v5_survey.csv'
df_merged.to_csv(outpath, index=False)
print(f"Saved: {outpath}")
