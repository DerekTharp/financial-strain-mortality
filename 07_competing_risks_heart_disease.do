* Fine-Gray competing-risks analysis: financial strain and incident heart disease (table S4)
* Input: output/cardiac_competing_risks.dta (written by 03_secondary_analysis.py)
* Heart disease is ascertained only at interviews. The cause-specific model
* (01_primary_analysis.py) censors at the last interview with heart-disease data
* (cardiac_followup_years). The Fine-Gray model places competing deaths at their death dates
* (cardiac_fu_fg) and is an approximate sensitivity analysis that treats the interval between
* the last assessment and death as free of a reported heart-disease event. The primary
* Fine-Gray fit uses the cause-specific analysis sample; a variant adds decedents with no
* post-baseline heart-disease assessment.

clear all
set more off

use "output/cardiac_competing_risks.dta", clear

* Covariates mirror CARDIAC_COVARS in model_specs.py (Model 3b without heart disease and stroke)
local covars age female race_nh_black race_hispanic race_nh_other ///
      education_yrs married_partnered current_smoker bmi ///
      baseline_hypertension baseline_diabetes ///
      asinh_income asinh_wealth

capture program drop shr_line
program define shr_line
    args label
    local beta = _b[fin_strain_binary]
    local se = _se[fin_strain_binary]
    local hr = exp(`beta')
    local ci_lo = exp(`beta' - 1.96*`se')
    local ci_hi = exp(`beta' + 1.96*`se')
    local p = 2*(1 - normal(abs(`beta'/`se')))
    qui count if e(sample) & _st
    local n = r(N)
    qui count if e(sample) & _st & _d
    local ev = r(N)
    display "`label': SHR = " %6.4f `hr' " (95% CI " %6.4f `ci_lo' " to " %6.4f `ci_hi' "), P = " %6.4f `p' ", N = `n', events = `ev'"
    file write results "`label'" _n
    file write results "  SHR = " %6.4f (`hr') _n
    file write results "  95% CI: " %6.4f (`ci_lo') " - " %6.4f (`ci_hi') _n
    file write results "  P = " %6.4f (`p') _n
    file write results "  N = " (`n') ", events = " (`ev') _n
    file write results "  Competing deaths = " (e(N_compete)) _n _n
end

file open results using "output/tables/fine_gray_results.txt", write replace
file write results "Incident heart disease: Fine-Gray models (unweighted)" _n
file write results "cardiac_event_type: 0 censored, 1 heart disease, 2 death without heart disease" _n _n

* Competing deaths at their death dates, cause-specific analysis sample
stset cardiac_fu_fg if cardiac_followup_years > 0, failure(cardiac_event_type==1)
stcrreg fin_strain_binary `covars', compete(cardiac_event_type==2)
shr_line "Fine-Gray SHR (competing deaths at death dates; cause-specific sample)"

* Variant: adding decedents with no post-baseline heart-disease assessment
stset cardiac_fu_fg, failure(cardiac_event_type==1)
stcrreg fin_strain_binary `covars', compete(cardiac_event_type==2)
shr_line "Fine-Gray SHR (adding decedents without post-baseline assessment)"

file close results
