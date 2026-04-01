
* Survey-Design Cox Proportional Hazards Models
* Tests whether proper HRS survey design (PSU/strata) changes weighted estimates
* Compares: robust SE only vs. full svyset (strata + PSU + weight)

clear all
set more off

* Load data
import delimited "output/analytic_sample_v5_survey.csv", clear

describe raestrat raehsamp lb_weight
summarize raestrat raehsamp lb_weight

* Check for missing values in key variables
count if missing(raestrat)
count if missing(raehsamp)
count if missing(lb_weight)

* Define covariates (Model 3b - primary)
local covars_3b age female race_nh_black race_hispanic race_nh_other ///
      education_yrs married_partnered current_smoker bmi ///
      baseline_hypertension baseline_diabetes baseline_heart baseline_stroke ///
      asinh_income asinh_wealth

* Model 3 covariates (without SES)
local covars_3 age female race_nh_black race_hispanic race_nh_other ///
      education_yrs married_partnered current_smoker bmi ///
      baseline_hypertension baseline_diabetes baseline_heart baseline_stroke

* ============================================================================
* PART 1: FULL SAMPLE MODELS
* ============================================================================

display ""
display "========================================================================"
display "PART 1: FULL SAMPLE - COMPARING SE APPROACHES"
display "========================================================================"

* --- 1A: Unweighted (no survey design) ---
display ""
display "--- MODEL 3b: UNWEIGHTED (reference) ---"
stset followup_years, failure(died)
stcox fin_strain_binary `covars_3b', efron
estimates store m3b_unweighted

* --- 1B: Weighted with robust SE only (current approach / lifelines) ---
display ""
display "--- MODEL 3b: WEIGHTED, ROBUST SE ONLY (current approach) ---"
stset followup_years [pweight=lb_weight], failure(died)
stcox fin_strain_binary `covars_3b', efron vce(robust)
estimates store m3b_weighted_robust

* --- 1C: Full survey design (strata + PSU + weight) ---
display ""
display "--- MODEL 3b: FULL SURVEY DESIGN (svyset) ---"
svyset raehsamp [pweight=lb_weight], strata(raestrat)
stset followup_years, failure(died)
svy: stcox fin_strain_binary `covars_3b', breslow
estimates store m3b_svy

* --- 1D: Model 3 (without SES) for comparison ---
display ""
display "--- MODEL 3: FULL SURVEY DESIGN ---"
svy: stcox fin_strain_binary `covars_3', breslow
estimates store m3_svy

* --- 1E: Model 4 (+ CES-D as confounder) ---
display ""
display "--- MODEL 4 (+ CES-D): FULL SURVEY DESIGN ---"
svy: stcox fin_strain_binary `covars_3b' depressed, breslow
estimates store m4_svy

* Compare all full-sample models
display ""
display "========================================================================"
display "COMPARISON: FULL SAMPLE MODEL 3b"
display "========================================================================"
estimates table m3b_unweighted m3b_weighted_robust m3b_svy, ///
    keep(fin_strain_binary) b(%9.4f) se(%9.4f) stats(N)

* Show HRs and CIs side by side
display ""
display "Model 3b HR comparisons:"
foreach m in m3b_unweighted m3b_weighted_robust m3b_svy {
    estimates restore `m'
    local hr = exp(_b[fin_strain_binary])
    local se = _se[fin_strain_binary]
    local ci_lo = exp(_b[fin_strain_binary] - 1.96*`se')
    local ci_hi = exp(_b[fin_strain_binary] + 1.96*`se')
    local p = 2*(1 - normal(abs(_b[fin_strain_binary]/`se')))
    display "  `m': HR = " %5.3f `hr' " (" %5.3f `ci_lo' ", " %5.3f `ci_hi' "), P = " %6.4f `p' ", SE(coef) = " %6.4f `se'
}

* ============================================================================
* PART 2: AGE-STRATIFIED MODELS WITH SURVEY DESIGN
* ============================================================================

display ""
display "========================================================================"
display "PART 2: AGE-STRATIFIED MODELS WITH FULL SURVEY DESIGN"
display "========================================================================"

* Under 65
display ""
display "--- AGE <65: FULL SURVEY DESIGN ---"
svyset raehsamp [pweight=lb_weight], strata(raestrat)
stset followup_years if age_under65==1, failure(died)
svy, subpop(age_under65): stcox fin_strain_binary `covars_3b', breslow
estimates store svy_under65

* 65 and over
display ""
display "--- AGE >=65: FULL SURVEY DESIGN ---"
gen age_over65 = (age_under65 == 0)
stset followup_years if age_over65==1, failure(died)
svy, subpop(age_over65): stcox fin_strain_binary `covars_3b', breslow
estimates store svy_over65

* Age-stratified comparison
display ""
display "Age-stratified survey-design HR comparisons:"
foreach m in svy_under65 svy_over65 {
    estimates restore `m'
    local hr = exp(_b[fin_strain_binary])
    local se = _se[fin_strain_binary]
    local ci_lo = exp(_b[fin_strain_binary] - 1.96*`se')
    local ci_hi = exp(_b[fin_strain_binary] + 1.96*`se')
    local p = 2*(1 - normal(abs(_b[fin_strain_binary]/`se')))
    display "  `m': HR = " %5.3f `hr' " (" %5.3f `ci_lo' ", " %5.3f `ci_hi' "), P = " %6.4f `p' ", SE(coef) = " %6.4f `se'
}

* ============================================================================
* PART 3: INTERACTION TEST WITH SURVEY DESIGN
* ============================================================================

display ""
display "========================================================================"
display "PART 3: AGE INTERACTION TEST WITH FULL SURVEY DESIGN"
display "========================================================================"

gen strain_x_under65 = fin_strain_binary * age_under65

svyset raehsamp [pweight=lb_weight], strata(raestrat)
stset followup_years, failure(died)
svy: stcox fin_strain_binary age_under65 strain_x_under65 `covars_3b', breslow
estimates store svy_interaction

display ""
display "Interaction term (strain x under65):"
estimates restore svy_interaction
local hr_int = exp(_b[strain_x_under65])
local se_int = _se[strain_x_under65]
local p_int = 2*(1 - normal(abs(_b[strain_x_under65]/`se_int')))
display "  HR = " %5.3f `hr_int' ", P = " %6.4f `p_int' ", SE = " %6.4f `se_int'

* ============================================================================
* PART 4: CARDIAC OUTCOME WITH SURVEY DESIGN
* ============================================================================

display ""
display "========================================================================"
display "PART 4: INCIDENT HEART DISEASE WITH FULL SURVEY DESIGN"
display "========================================================================"

* Cardiac covariates (exclude baseline heart/stroke - CVD-free sample)
local covars_cardiac age female race_nh_black race_hispanic race_nh_other ///
      education_yrs married_partnered current_smoker bmi ///
      baseline_hypertension baseline_diabetes ///
      asinh_income asinh_wealth

gen cvd_free = (baseline_heart == 0 & baseline_stroke == 0) if !missing(baseline_heart) & !missing(baseline_stroke)

svyset raehsamp [pweight=lb_weight], strata(raestrat)
stset cardiac_followup_years if cvd_free==1, failure(incident_heart)
svy, subpop(cvd_free): stcox fin_strain_binary `covars_cardiac', breslow
estimates store svy_cardiac

estimates restore svy_cardiac
local hr = exp(_b[fin_strain_binary])
local se = _se[fin_strain_binary]
local ci_lo = exp(_b[fin_strain_binary] - 1.96*`se')
local ci_hi = exp(_b[fin_strain_binary] + 1.96*`se')
local p = 2*(1 - normal(abs(_b[fin_strain_binary]/`se')))
display ""
display "Incident heart disease (survey design): HR = " %5.3f `hr' " (" %5.3f `ci_lo' ", " %5.3f `ci_hi' "), P = " %6.4f `p'

* ============================================================================
* SUMMARY TABLE
* ============================================================================

display ""
display "========================================================================"
display "SUMMARY: KEY COMPARISONS"
display "========================================================================"
display ""

display "Model                              |   HR   |  95% CI       |  P-value | SE(coef)"
display "-----------------------------------+--------+---------------+----------+---------"

foreach m in m3b_unweighted m3b_weighted_robust m3b_svy m3_svy m4_svy svy_under65 svy_over65 svy_cardiac {
    capture estimates restore `m'
    if _rc == 0 {
        local hr = exp(_b[fin_strain_binary])
        local se = _se[fin_strain_binary]
        local ci_lo = exp(_b[fin_strain_binary] - 1.96*`se')
        local ci_hi = exp(_b[fin_strain_binary] + 1.96*`se')
        local p = 2*(1 - normal(abs(_b[fin_strain_binary]/`se')))
        local mname = "`m'"
        display %-35s "`mname'" " | " %5.3f `hr' "  | " %5.3f `ci_lo' "-" %5.3f `ci_hi' " | " %7.4f `p' "  | " %6.4f `se'
    }
}

display ""
display "KEY QUESTION: Does SE increase from m3b_weighted_robust to m3b_svy?"
display "If so, by how much? And does the CI now cross 1.00?"

display ""
display "DONE"
