"""Covariate lists and constants shared by the analysis scripts."""

EXPOSURE_VAR = 'fin_strain_binary'

# Model 3b covariates. The reference category for race and ethnicity is non-Hispanic White.
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

# Model 3 omits income and wealth
MODEL3_COVARS = [v for v in MODEL3B_COVARS if v not in ('asinh_income', 'asinh_wealth')]

# The incident heart disease cohort is free of heart disease and stroke at baseline
CARDIAC_COVARS = [v for v in MODEL3B_COVARS if v not in ('baseline_heart', 'baseline_stroke')]

# Ridge penalty for lifelines CoxPHFitter
PENALIZER = 1e-5
