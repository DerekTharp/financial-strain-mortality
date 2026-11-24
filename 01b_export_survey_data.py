"""
Adds the HRS survey design variables to the analytic sample: RAESTRAT (standard-error
stratum) and RAEHSAMP (half-sample code, used as the primary sampling unit).

Writes output/analytic_sample_survey.csv, the input to 04_flow_diagram.py and
11_survey_prep.py.
"""

import pandas as pd
import pyreadstat
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
HRS_FILE = PROJECT_DIR / 'randhrs1992_2022v1_STATA' / 'randhrs1992_2022v1.dta'
OUTPUT_DIR = PROJECT_DIR / 'output'

df = pd.read_csv(OUTPUT_DIR / 'analytic_sample.csv')

survey_vars, _ = pyreadstat.read_dta(str(HRS_FILE),
                                      usecols=['hhidpn', 'raestrat', 'raehsamp'])

df_merged = df.merge(survey_vars, on='hhidpn', how='left')
assert len(df_merged) == len(df), "merge changed the number of rows"

n_valid = df_merged['raestrat'].notna().sum()
print(f"Valid strata/PSU: {n_valid} of {len(df_merged)}")

outpath = OUTPUT_DIR / 'analytic_sample_survey.csv'
df_merged.to_csv(outpath, index=False)
print(f"Saved: {outpath}")
