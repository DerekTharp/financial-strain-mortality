"""
model_specs.py
Centralised covariate definitions for the Financial Strain and Mortality project.

All scripts import variable lists and constants from this module to ensure
consistency across primary, secondary, and sensitivity analyses.
"""

# Exposure variable
EXPOSURE_VAR = 'fin_strain_binary'

# Model 3b covariates: demographics + behaviours + health conditions + SES
# Reference category for race/ethnicity is NH White (omitted)
MODEL3B_COVARS = [
    'age',
    'female',
    'race_nh_black',
    'race_hispanic',
    'race_nh_other',
    'education_yrs',
    'married_partnered',
    'current_smoker',
    'bmi',
    'baseline_hypertension',
    'baseline_diabetes',
    'baseline_heart',
    'baseline_stroke',
    'asinh_income',
    'asinh_wealth',
]

# Penalizer for lifelines CoxPHFitter (ridge penalty for convergence stability)
PENALIZER = 1e-5
