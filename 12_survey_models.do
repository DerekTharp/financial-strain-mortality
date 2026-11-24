* Survey-design Cox models (svy: stcox) under the full HRS design: LBQ respondent weight,
* strata, primary sampling units, linearised variance.
*
* Input:  output/analytic_sample_svy.csv (from 11_survey_prep.py)
* Output: output/tables/svy_models.csv, svy_joint_tests.csv, svy_evalues.csv,
*         svy_race_cells.csv, svy_cardiac.csv, svy_age_specific.csv

clear all
set more off
capture log close
log using "12_survey_models.log", replace text

import delimited "output/analytic_sample_svy.csv", clear

* The covariate lists in this file mirror model_specs.py (covars_3b is MODEL3B_COVARS)
local covars_3b age female race_nh_black race_hispanic race_nh_other ///
    education_yrs married_partnered current_smoker bmi ///
    baseline_hypertension baseline_diabetes baseline_heart baseline_stroke ///
    asinh_income asinh_wealth
local covars_3 age female race_nh_black race_hispanic race_nh_other ///
    education_yrs married_partnered current_smoker bmi ///
    baseline_hypertension baseline_diabetes baseline_heart baseline_stroke
local covars_noage female race_nh_black race_hispanic race_nh_other ///
    education_yrs married_partnered current_smoker bmi ///
    baseline_hypertension baseline_diabetes baseline_heart baseline_stroke ///
    asinh_income asinh_wealth
local covars_norace age female education_yrs married_partnered current_smoker ///
    bmi baseline_hypertension baseline_diabetes baseline_heart baseline_stroke ///
    asinh_income asinh_wealth
local covars_nofem age race_nh_black race_hispanic race_nh_other education_yrs ///
    married_partnered current_smoker bmi baseline_hypertension ///
    baseline_diabetes baseline_heart baseline_stroke asinh_income asinh_wealth

gen byte age_over65 = age_under65 == 0
gen byte male = female == 0

* Results collectors
tempname R T E
tempfile fR fT fE
postfile `R' str24 part str120 label N events double(hr lo hi se p) using `fR', replace
postfile `T' str24 part str60 test df1 df2 double(F p) using `fT', replace
postfile `E' str40 label double(hr lo hi evalue evalue_ci) using `fE', replace

* Post the strain coefficient of the last svy: stcox fit.
* args: part label subpop_expr ("." for none)
capture program drop post_strain
program define post_strain
    args R part label subpopexpr
    local hr = exp(_b[fin_strain_binary])
    local se = _se[fin_strain_binary]
    local df = e(df_r)
    if `df' >= . {
        local tc = 1.96
        local p = 2*(1 - normal(abs(_b[fin_strain_binary]/`se')))
    }
    else {
        local tc = invttail(`df', 0.025)
        local p = 2*ttail(`df', abs(_b[fin_strain_binary]/`se'))
    }
    local lo = exp(_b[fin_strain_binary] - `tc'*`se')
    local hi = exp(_b[fin_strain_binary] + `tc'*`se')
    tempvar es
    gen byte `es' = e(sample)
    if "`subpopexpr'" == "." {
        qui count if `es' & _st == 1
        local nn = r(N)
        qui count if `es' & _st == 1 & _d == 1
    }
    else {
        qui count if `es' & _st == 1 & (`subpopexpr')
        local nn = r(N)
        qui count if `es' & _st == 1 & _d == 1 & (`subpopexpr')
    }
    local ev = r(N)
    post `R' ("`part'") ("`label'") (`nn') (`ev') (`hr') (`lo') (`hi') (`se') (`p')
    display "  [`part'] `label': HR=" %6.3f `hr' " (" %5.3f `lo' "-" %5.3f `hi' "), P=" %6.4f `p' ", N=`nn', events=`ev'"
end

* E-value from the last fit. The outcome is common, so hazard ratios are
* first converted with the VanderWeele-Ding common-outcome approximation
* RR = (1 - 0.5^sqrt(HR)) / (1 - 0.5^sqrt(1/HR)) before applying the E-value formula.
* CI-bound E-value uses the lower limit; 1 when the CI crosses the null.
capture program drop post_evalue
program define post_evalue
    args E label
    local hr = exp(_b[fin_strain_binary])
    local se = _se[fin_strain_binary]
    local df = e(df_r)
    local tc = invttail(`df', 0.025)
    local lo = exp(_b[fin_strain_binary] - `tc'*`se')
    local hi = exp(_b[fin_strain_binary] + `tc'*`se')
    local rr = (1 - 0.5^sqrt(`hr')) / (1 - 0.5^sqrt(1/`hr'))
    if `rr' >= 1 local ev = `rr' + sqrt(`rr'*(`rr'-1))
    else {
        local rri = 1/`rr'
        local ev = `rri' + sqrt(`rri'*(`rri'-1))
    }
    if `lo' > 1 {
        local rrlo = (1 - 0.5^sqrt(`lo')) / (1 - 0.5^sqrt(1/`lo'))
        local evci = `rrlo' + sqrt(`rrlo'*(`rrlo'-1))
    }
    else local evci = 1
    post `E' ("`label'") (`hr') (`lo') (`hi') (`ev') (`evci')
    display "  E-value `label': E=" %5.3f `ev' ", E(CI)=" %5.3f `evci'
end

* Post an interaction term as a ratio of hazard ratios (exp(coefficient)) with its
* design-df t interval and P, into the same results file as post_strain.
* args: R part label term
capture program drop post_ratio
program define post_ratio
    args R part label term
    local b = _b[`term']
    local se = _se[`term']
    local df = e(df_r)
    local tc = invttail(`df', 0.025)
    local p = 2*ttail(`df', abs(`b'/`se'))
    tempvar es
    gen byte `es' = e(sample)
    qui count if `es' & _st == 1
    local nn = r(N)
    qui count if `es' & _st == 1 & _d == 1
    local ev = r(N)
    post `R' ("`part'") ("`label'") (`nn') (`ev') (exp(`b')) (exp(`b' - `tc'*`se')) (exp(`b' + `tc'*`se')) (`se') (`p')
    display "  [`part'] `label': ratio=" %6.3f exp(`b') " (" %5.3f exp(`b' - `tc'*`se') "-" %5.3f exp(`b' + `tc'*`se') "), P=" %6.4f `p'
end

* Post a linear combination of the last svy: stcox fit as a hazard ratio with its design-df
* t interval (used for age-specific estimates from a single interaction model).
* args: R part label expression
capture program drop post_lincom
program define post_lincom
    args R part label expr
    qui lincom `expr'
    local b = r(estimate)
    local s = r(se)
    local df = e(df_r)
    local tc = invttail(`df', 0.025)
    local p = 2*ttail(`df', abs(`b'/`s'))
    tempvar es
    gen byte `es' = e(sample)
    qui count if `es' & _st == 1
    local nn = r(N)
    qui count if `es' & _st == 1 & _d == 1
    local ev = r(N)
    post `R' ("`part'") ("`label'") (`nn') (`ev') (exp(`b')) (exp(`b' - `tc'*`s')) (exp(`b' + `tc'*`s')) (`s') (`p')
    display "  [`part'] `label': HR=" %6.3f exp(`b') " (" %5.3f exp(`b' - `tc'*`s') "-" %5.3f exp(`b' + `tc'*`s') ")"
end

* Survey design and follow-up time
svyset raehsamp [pweight=lb_weight], strata(raestrat)
stset followup_years, failure(died)

display _n "Part 1: Models 1 and 2"
svy: stcox fin_strain_binary, breslow
post_strain `R' "part1" "Model 1 unadjusted (svy)" "."
svy: stcox fin_strain_binary age female, breslow
post_strain `R' "part1" "Model 2 age+sex (svy)" "."

display _n "Part 2: Models 3, 3b and 4, age strata, E-values"
svy: stcox fin_strain_binary `covars_3', breslow
local b_model3 = _b[fin_strain_binary]
post_strain `R' "part2" "Model 3 (svy)" "."
post_evalue `E' "Model 3 (weighted)"

svy: stcox fin_strain_binary `covars_3b', breslow
post_strain `R' "part2" "Model 3b (svy, primary)" "."
post_evalue `E' "Model 3b (weighted)"
display "  Attenuation of the log hazard ratio from Model 3 to Model 3b: " %5.2f 100*(1 - _b[fin_strain_binary]/`b_model3') "%"

svy: stcox fin_strain_binary `covars_3b' depressed, breslow
post_strain `R' "part2" "Model 4 +depression (svy)" "."
post_evalue `E' "Model 4 (weighted)"

svy, subpop(age_under65): stcox fin_strain_binary `covars_3b', breslow
post_strain `R' "part2" "Age <65 (svy, Model 3b)" "age_under65==1"
post_evalue `E' "Age <65 (weighted)"

svy, subpop(age_over65): stcox fin_strain_binary `covars_3b', breslow
post_strain `R' "part2" "Age >=65 (svy, Model 3b)" "age_over65==1"
post_evalue `E' "Age >=65 (weighted)"

display _n "Part 3: Joint economic interactions with age"
gen double age_c = age - 65
gen double strain_x_agec = fin_strain_binary * age_c
gen double income_x_agec = asinh_income * age_c
gen double wealth_x_agec = asinh_wealth * age_c

svy: stcox fin_strain_binary age_c strain_x_agec income_x_agec wealth_x_agec ///
    `covars_noage', breslow
post_strain `R' "part3" "Joint continuous-age model (strain main effect at 65)" "."
local dfj = e(df_r)
local tcj = invttail(`dfj', 0.025)
foreach term in strain_x_agec income_x_agec wealth_x_agec {
    local b = _b[`term']
    local s = _se[`term']
    local p = 2*ttail(`dfj', abs(`b'/`s'))
    post `R' ("part3") ("Joint cont-age: `term' (per year, log-HR)") (e(N)) (.) (exp(`b')) (exp(`b'-`tcj'*`s')) (exp(`b'+`tcj'*`s')) (`s') (`p')
}
test strain_x_agec income_x_agec wealth_x_agec
post `T' ("part3") ("Joint Wald: all three x age_c") (r(df)) (r(df_r)) (r(F)) (r(p))
test income_x_agec wealth_x_agec
post `T' ("part3") ("2 df: income + wealth x age_c") (r(df)) (r(df_r)) (r(F)) (r(p))

gen double strain_x_u65 = fin_strain_binary * age_under65
gen double income_x_u65 = asinh_income * age_under65
gen double wealth_x_u65 = asinh_wealth * age_under65
svy: stcox fin_strain_binary asinh_income asinh_wealth age_under65 ///
    strain_x_u65 income_x_u65 wealth_x_u65 `covars_3', breslow
post_strain `R' "part3" "Joint binary-65 companion (strain main effect, >=65)" "."
test strain_x_u65 income_x_u65 wealth_x_u65
post `T' ("part3") ("Joint Wald: all three x under65") (r(df)) (r(df_r)) (r(F)) (r(p))
test strain_x_u65
post `T' ("part3") ("strain x under65 alone") (r(df)) (r(df_r)) (r(F)) (r(p))

* Age-specific hazard ratios for financial strain from the strain x continuous-age model
* (Model 3b covariates, age centred at 65) under the full survey design (table S15)
svy: stcox fin_strain_binary age_c strain_x_agec `covars_noage', breslow
tempname A
tempfile fA
postfile `A' age double(hr lo hi) using `fA', replace
local dfa = e(df_r)
local tca = invttail(`dfa', 0.025)
foreach a in 55 60 65 70 75 80 {
    qui lincom fin_strain_binary + (`a' - 65)*strain_x_agec
    post `A' (`a') (exp(r(estimate))) (exp(r(estimate) - `tca'*r(se))) (exp(r(estimate) + `tca'*r(se)))
    display "  [part3b] age `a': HR=" %6.3f exp(r(estimate))
}
postclose `A'

display _n "Part 4: Sex interaction and strata"
gen double strain_x_fem = fin_strain_binary * female
svy: stcox fin_strain_binary female strain_x_fem `covars_nofem', breslow
local b = _b[strain_x_fem]
local s = _se[strain_x_fem]
local dfs = e(df_r)
local tcs = invttail(`dfs', 0.025)
local p = 2*ttail(`dfs', abs(`b'/`s'))
post `R' ("part4") ("strain x female interaction (ratio of HRs)") (e(N)) (.) (exp(`b')) (exp(`b'-`tcs'*`s')) (exp(`b'+`tcs'*`s')) (`s') (`p')

svy, subpop(female): stcox fin_strain_binary `covars_nofem', breslow
post_strain `R' "part4" "Women (svy, Model 3b covars)" "female==1"
svy, subpop(male): stcox fin_strain_binary `covars_nofem', breslow
post_strain `R' "part4" "Men (svy, Model 3b covars)" "male==1"

display _n "Part 5: Race/ethnicity omnibus and strata"
gen double strain_x_black = fin_strain_binary * race_nh_black
gen double strain_x_hisp  = fin_strain_binary * race_hispanic
gen double strain_x_other = fin_strain_binary * race_nh_other
svy: stcox fin_strain_binary strain_x_black strain_x_hisp strain_x_other `covars_3b', breslow
test strain_x_black strain_x_hisp strain_x_other
post `T' ("part5") ("Race omnibus (3 df)") (r(df)) (r(df_r)) (r(F)) (r(p))

* Cell counts in the estimation sample (suppression rule: no stratum HR where
* exposed deaths < 10; the non-Hispanic other stratum is reported as counts only)
tempvar es5
gen byte `es5' = e(sample)
tempname C
tempfile fC
postfile `C' str24 group exposed_n exposed_deaths using `fC', replace
foreach g in race_nh_black race_hispanic race_nh_other {
    qui count if `es5' & `g'==1 & fin_strain_binary==1
    local xn = r(N)
    qui count if `es5' & `g'==1 & fin_strain_binary==1 & died==1
    post `C' ("`g'") (`xn') (r(N))
}
qui count if `es5' & race_nh_black==0 & race_hispanic==0 & race_nh_other==0 & fin_strain_binary==1
local xn = r(N)
qui count if `es5' & race_nh_black==0 & race_hispanic==0 & race_nh_other==0 & fin_strain_binary==1 & died==1
post `C' ("nh_white") (`xn') (r(N))
postclose `C'

gen byte nh_white = race_nh_black==0 & race_hispanic==0 & race_nh_other==0
svy, subpop(nh_white): stcox fin_strain_binary `covars_norace', breslow
post_strain `R' "part5" "Non-Hispanic White stratum" "nh_white==1"
svy, subpop(race_nh_black): stcox fin_strain_binary `covars_norace', breslow
post_strain `R' "part5" "Non-Hispanic Black stratum" "race_nh_black==1"
svy, subpop(race_hispanic): stcox fin_strain_binary `covars_norace', breslow
post_strain `R' "part5" "Hispanic stratum" "race_hispanic==1"

display _n "Part 6: IPTW under composite weights"
foreach spec in a a_trunc b b_trunc {
    capture drop w_comp
    gen double w_comp = sw_`spec' * lb_weight
    svyset raehsamp [pweight=w_comp], strata(raestrat)
    svy: stcox fin_strain_binary, breslow
    post_strain `R' "part6" "IPTW `spec' (composite weights)" "."
    * Covariate-adjusted outcome model for the primary propensity model only
    if inlist("`spec'", "a", "a_trunc") {
        svy: stcox fin_strain_binary `covars_3b', breslow
        post_strain `R' "part6" "IPTW `spec' + covariate adjustment" "."
    }
}
svyset raehsamp [pweight=lb_weight], strata(raestrat)

display _n "Part 7: Wave 7 lagged covariates"
local covars_w7A age female race_nh_black race_hispanic race_nh_other ///
    education_yrs married_partnered current_smoker_w7 bmi_w7 ///
    baseline_hypertension_w7 baseline_diabetes_w7 baseline_heart_w7 ///
    baseline_stroke_w7 asinh_income asinh_wealth
local covars_w7B age female race_nh_black race_hispanic race_nh_other ///
    education_yrs married_partnered current_smoker_w7 bmi_w7 ///
    baseline_hypertension_w7 baseline_diabetes_w7 baseline_heart_w7 ///
    baseline_stroke_w7 asinh_income_w7 asinh_wealth_w7

svy: stcox fin_strain_binary `covars_w7A', breslow
post_strain `R' "part7" "W7 lag variant A (W7 behaviours/conditions)" "."
post_evalue `E' "W7 lag variant A"
svy: stcox fin_strain_binary `covars_w7B', breslow
post_strain `R' "part7" "W7 lag variant B (+W7 income/wealth)" "."

capture drop strain_x_u65b
gen double strain_x_u65b = fin_strain_binary * age_under65
svy: stcox fin_strain_binary age_under65 strain_x_u65b `covars_w7A', breslow
test strain_x_u65b
post `T' ("part7") ("strain x under65, W7 variant A") (r(df)) (r(df_r)) (r(F)) (r(p))
svy, subpop(age_under65): stcox fin_strain_binary `covars_w7A', breslow
post_strain `R' "part7" "W7 variant A, age <65" "age_under65==1"
svy, subpop(age_over65): stcox fin_strain_binary `covars_w7A', breslow
post_strain `R' "part7" "W7 variant A, age >=65" "age_over65==1"

* Unweighted complete-case versions of the lagged models
stcox fin_strain_binary `covars_w7A', breslow
post_strain `R' "part7" "W7 variant A, unweighted" "."
stcox fin_strain_binary `covars_w7B', breslow
post_strain `R' "part7" "W7 variant B, unweighted" "."

* Same-sample Wave 8 comparator: both the lagged and the Wave 8 covariate models
* estimated on the common complete-case intersection of the two covariate sets, so the
* lag comparison is not confounded by sample composition
gen byte w8w7cc = !missing(current_smoker_w7) & !missing(bmi_w7) & ///
    !missing(baseline_hypertension_w7) & !missing(baseline_diabetes_w7) & ///
    !missing(baseline_heart_w7) & !missing(baseline_stroke_w7) & ///
    !missing(current_smoker) & !missing(bmi) & !missing(baseline_hypertension) & ///
    !missing(baseline_diabetes) & !missing(baseline_heart) & ///
    !missing(baseline_stroke) & !missing(asinh_income) & !missing(asinh_wealth) & ///
    !missing(asinh_income_w7) & !missing(asinh_wealth_w7)
svy, subpop(w8w7cc): stcox fin_strain_binary `covars_w7A', breslow
post_strain `R' "part7" "W7 variant A, common sample" "w8w7cc==1"
svy, subpop(w8w7cc): stcox fin_strain_binary `covars_3b', breslow
post_strain `R' "part7" "W8 covariates, common sample" "w8w7cc==1"

display _n "Part 7B: Alternative mortality-ascertainment assumptions"
* Primary rule (all other parts): administrative censoring at 31 Dec 2022 with deaths from the
* Tracker hierarchy (exit/spouse-reported, else NDI). Alternatives constructed in
* 01_primary_analysis.py. Each rule: overall estimate, age-specific estimates from the
* interaction model, and the ratio of HRs (<65 vs >=65).
capture drop sx_u65
gen double sx_u65 = fin_strain_binary * age_under65
foreach rule in primary lastalive randlastiw randadmin ndijan ndidec {
    if "`rule'" == "ndijan" {
        stset followup_ndi_jan, failure(died_ndi_jan)
        local lab "Year-only NDI deaths dated 1 January"
    }
    else if "`rule'" == "ndidec" {
        stset followup_ndi_dec, failure(died_ndi_dec)
        local lab "Year-only NDI deaths dated 31 December"
    }
    else if "`rule'" == "primary" {
        stset followup_years, failure(died)
        local lab "Primary rule (admin. censoring 2022, Tracker+NDI deaths)"
    }
    else if "`rule'" == "lastalive" {
        stset followup_lastalive, failure(died_lastalive)
        local lab "Last documented alive (Tracker LASTALIVE)"
    }
    else if "`rule'" == "randlastiw" {
        stset followup_randlastiw, failure(died_randlastiw)
        local lab "RAND deaths only, censoring at last interview"
    }
    else {
        stset followup_randadmin, failure(died_randadmin)
        local lab "Admin. censoring 2022, RAND deaths only"
    }
    svy: stcox fin_strain_binary `covars_3b', breslow
    post_strain `R' "part7b" "`lab' (Model 3b)" "."
    svy: stcox fin_strain_binary age_under65 sx_u65 `covars_3b', breslow
    post_ratio `R' "part7b" "`lab', ratio of HRs" sx_u65
    post_lincom `R' "part7b" "`lab', age <65 (interaction)" "fin_strain_binary + sx_u65"
    post_lincom `R' "part7b" "`lab', age >=65 (interaction)" "fin_strain_binary"
}
stset followup_years, failure(died)

display _n "Part 8: Attained-age time scale"
stset age_at_exit if age_at_exit > age_at_entry, enter(age_at_entry) failure(died)
svy: stcox fin_strain_binary `covars_noage', breslow
post_strain `R' "part8" "Attained age (svy, full design)" "."
svy, subpop(age_under65): stcox fin_strain_binary `covars_noage', breslow
post_strain `R' "part8" "Attained age, baseline <65" "age_under65==1"
svy, subpop(age_over65): stcox fin_strain_binary `covars_noage', breslow
post_strain `R' "part8" "Attained age, baseline >=65" "age_over65==1"

* Attained-age interaction test (baseline-age 65 threshold)
capture drop strain_x_u65aa
gen double strain_x_u65aa = fin_strain_binary * age_under65
svy: stcox fin_strain_binary age_under65 strain_x_u65aa `covars_noage', breslow
test strain_x_u65aa
post `T' ("part8") ("strain x under65, attained-age scale") (r(df)) (r(df_r)) (r(F)) (r(p))

display _n "Part 9: Incident heart disease"
local covars_cardiac age female race_nh_black race_hispanic race_nh_other ///
    education_yrs married_partnered current_smoker bmi ///
    baseline_hypertension baseline_diabetes asinh_income asinh_wealth
gen byte cvd_free = (baseline_heart == 0 & baseline_stroke == 0) ///
    if !missing(baseline_heart) & !missing(baseline_stroke)
stset cardiac_followup_years if cvd_free==1, failure(incident_heart)
svy, subpop(cvd_free): stcox fin_strain_binary `covars_cardiac', breslow
local hr = exp(_b[fin_strain_binary])
local se = _se[fin_strain_binary]
local df9 = e(df_r)
local tc9 = invttail(`df9', 0.025)
local lo = exp(_b[fin_strain_binary] - `tc9'*`se')
local hi = exp(_b[fin_strain_binary] + `tc9'*`se')
local p = 2*ttail(`df9', abs(_b[fin_strain_binary]/`se'))
tempvar es9
gen byte `es9' = e(sample)
local nsub = e(N_sub)
qui count if `es9' & cvd_free==1 & incident_heart==1
local evsub = r(N)
display "  Cardiac svy subpop: N=`nsub', in-model events=`evsub', HR=" %5.3f `hr'
tempname C9
tempfile fC9
postfile `C9' str40 label N events double(hr lo hi p) using `fC9', replace
post `C9' ("Cause-specific HR, weighted") (`nsub') (`evsub') (`hr') (`lo') (`hi') (`p')
postclose `C9'

display _n "Part 10: Robustness of the age interaction"
stset followup_years, failure(died)
* Each specification posts the overall estimate, the age-specific estimates from the
* interaction model, and the ratio of HRs with its interval. Landmark analyses restart
* follow-up at 1 and 2 years among participants still under observation.
capture drop bmi_cat
gen byte bmi_cat = cond(bmi < 25, 1, cond(bmi < 30, 2, 3)) if bmi < .
foreach v in inc_s1 inc_s2 wlt_s1 wlt_s2 {
    capture drop `v'
}
mkspline inc_s = asinh_income, cubic nknots(3)
mkspline wlt_s = asinh_wealth, cubic nknots(3)
global rcov_3b `covars_3b'
global rcov_rcs age female race_nh_black race_hispanic race_nh_other ///
    education_yrs married_partnered current_smoker bmi ///
    baseline_hypertension baseline_diabetes baseline_heart baseline_stroke ///
    inc_s1 inc_s2 wlt_s1 wlt_s2

capture program drop robust_block
program define robust_block
    args R label
    svy: stcox fin_strain_binary $rcov $rextra, breslow $ropts
    post_strain `R' "part10" "`label' (overall)" "."
    svy: stcox fin_strain_binary age_under65 sx_u65 $rcov $rextra, breslow $ropts
    post_ratio `R' "part10" "`label', ratio of HRs" sx_u65
    post_lincom `R' "part10" "`label', age <65 (interaction)" "fin_strain_binary + sx_u65"
    post_lincom `R' "part10" "`label', age >=65 (interaction)" "fin_strain_binary"
end

global rcov $rcov_3b
global rextra ""
global ropts ""
robust_block `R' "Primary (Model 3b)"
global rextra "depressed"
robust_block `R' "+ depressive symptoms (Model 4)"
global rextra "srh_fairpoor"
robust_block `R' "+ fair/poor self-rated health"
global rextra "c.age#c.age"
robust_block `R' "+ age squared"
global rextra ""
global ropts "strata(baseline_heart bmi_cat)"
robust_block `R' "Stratified by heart disease x BMI category"
global ropts ""
global rcov $rcov_rcs
robust_block `R' "RCS income and wealth"
global rcov $rcov_3b
stset followup_years if followup_years > 1, failure(died) enter(time 1)
robust_block `R' "Landmark at 1 year"
stset followup_years if followup_years > 2, failure(died) enter(time 2)
robust_block `R' "Landmark at 2 years"
stset followup_years, failure(died)

* Strain x restricted-cubic-spline(age): nonlinearity and joint tests
foreach v in age_c age_s1 age_s2 sxa1 sxa2 {
    capture drop `v'
}
gen double age_c = age - 65
mkspline age_s = age_c, cubic nknots(3)
gen double sxa1 = fin_strain_binary * age_s1
gen double sxa2 = fin_strain_binary * age_s2
svy: stcox fin_strain_binary age_s1 age_s2 sxa1 sxa2 `covars_noage', breslow
test sxa2
post `T' ("part10") ("strain x age spline: nonlinearity (sxa2=0)") (r(df)) (r(df_r)) (r(F)) (r(p))
test sxa1 sxa2
post `T' ("part10") ("strain x age spline: joint interaction") (r(df)) (r(df_r)) (r(F)) (r(p))

* Export collected results
postclose `R'
postclose `T'
postclose `E'
preserve
use `fR', clear
export delimited "output/tables/svy_models.csv", replace
use `fT', clear
export delimited "output/tables/svy_joint_tests.csv", replace
use `fE', clear
export delimited "output/tables/svy_evalues.csv", replace
use `fC', clear
export delimited "output/tables/svy_race_cells.csv", replace
use `fC9', clear
export delimited "output/tables/svy_cardiac.csv", replace
use `fA', clear
export delimited "output/tables/svy_age_specific.csv", replace
restore

log close
