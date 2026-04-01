
* Fine-Gray Competing Risks Analysis for Financial Strain and Incident Heart Disease
* Run this in Stata after loading cardiac_competing_risks_v6.dta

clear all
set more off

* Load data
use "output/cardiac_competing_risks_v6.dta", clear

describe
summarize

* Set up survival data
* cardiac_event_type: 0=censored, 1=heart disease (event of interest), 2=death (competing)
stset cardiac_followup_years, failure(cardiac_event_type==1)

* Cause-specific Cox model (for comparison - death censored)
* Note: baseline_heart and baseline_stroke excluded because sample is CVD-free at baseline
local covars age female race_nh_black race_hispanic race_nh_other ///
      education_yrs married_partnered current_smoker bmi ///
      baseline_hypertension baseline_diabetes ///
      asinh_income asinh_wealth

stcox fin_strain_binary `covars', efron nohr
estimates store cause_specific

* Fine-Gray subdistribution hazard model
stcrreg fin_strain_binary `covars', compete(cardiac_event_type==2)
estimates store fine_gray

* Compare models
estimates table cause_specific fine_gray, stats(N ll) b(%9.3f) se(%9.3f)

* Extract results using coefficient name (not positional indexing)
local beta = _b[fin_strain_binary]
local se = _se[fin_strain_binary]
local hr = exp(`beta')
local ci_lo = exp(`beta' - 1.96*`se')
local ci_hi = exp(`beta' + 1.96*`se')
local z = `beta'/`se'
local p = 2*(1 - normal(abs(`z')))

display "Fine-Gray Results for Financial Strain:"
display "  SHR = " %5.3f `hr' " (95% CI: " %5.3f `ci_lo' " - " %5.3f `ci_hi' ")"
display "  P = " %6.4f `p'

* Save results to file
file open results using "output/tables/fine_gray_results_v6.txt", write replace
file write results "Fine-Gray Subdistribution Hazard Model Results" _n
file write results "Outcome: Incident Heart Disease" _n
file write results "Competing Event: Death without Heart Disease" _n _n
file write results "Financial Strain (high vs low/none):" _n
file write results "  SHR = " %5.3f (`hr') _n
file write results "  95% CI: " %5.3f (`ci_lo') " - " %5.3f (`ci_hi') _n
file write results "  P = " %6.4f (`p') _n
file close results

display "Results saved to output/tables/fine_gray_results_v6.txt"
