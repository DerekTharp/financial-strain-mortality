"""
08_exploratory_reviewer_analyses.py
Exploratory supplementary analyses.

Analyses:
1. Proportion of <65 baseline deaths that occurred before vs after reaching age 65
2. Lexis expansion — person-time split at age 65 (time-varying pre/post-Medicare)
3. Weight re-normalization within age strata
4. PS-matched age-stratified estimates

Note on Lexis expansion (Analysis 2): lifelines does not support counting-process
(start-stop) notation, so the model uses segment duration as the time variable with
a post65 indicator. Standard errors do not account for within-person correlation
across split segments. Results should be interpreted as approximate.

"""

import pandas as pd
import numpy as np
from pathlib import Path
from lifelines import CoxPHFitter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import json

# ============================================================================
# SETUP
# ============================================================================

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "output"
df = pd.read_csv(OUTPUT_DIR / "analytic_sample_v5.csv")

# Convert all to numeric
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Model 3b covariates (matching primary analysis exactly)
model3b_covars = [
    'age', 'female',
    'race_nh_black', 'race_hispanic', 'race_nh_other',
    'education_yrs', 'married_partnered',
    'current_smoker', 'bmi',
    'baseline_hypertension', 'baseline_diabetes', 'baseline_heart', 'baseline_stroke',
    'asinh_income', 'asinh_wealth'
]

def fit_cox(df_in, covars, duration='followup_years', event='died', weights=None):
    """Fit Cox model and return results dict."""
    cols = ['fin_strain_binary', duration, event] + covars
    if weights:
        cols.append(weights)
    df_fit = df_in[cols].dropna().copy()
    for c in df_fit.columns:
        df_fit[c] = pd.to_numeric(df_fit[c], errors='coerce')
    df_fit = df_fit.dropna()

    cph = CoxPHFitter(penalizer=1e-5)
    formula = 'fin_strain_binary + ' + ' + '.join(covars)

    kwargs = dict(duration_col=duration, event_col=event, formula=formula)
    if weights:
        # Normalize weights to sum to N
        n = len(df_fit)
        df_fit[f'{weights}_norm'] = df_fit[weights] * n / df_fit[weights].sum()
        kwargs['weights_col'] = f'{weights}_norm'
        kwargs['robust'] = True

    cph.fit(df_fit, **kwargs)

    hr = float(np.exp(cph.params_['fin_strain_binary']))
    ci = cph.confidence_intervals_.loc['fin_strain_binary']
    return {
        'hr': hr,
        'ci_lower': float(np.exp(ci.iloc[0])),
        'ci_upper': float(np.exp(ci.iloc[1])),
        'p_value': float(cph.summary.loc['fin_strain_binary', 'p']),
        'n': len(df_fit),
        'events': int(df_fit[event].sum())
    }


print("=" * 80)
print("EXPLORATORY ANALYSES — REVIEWER SUGGESTIONS")
print("=" * 80)

results = {}

# ============================================================================
# ANALYSIS 1: Proportion of <65 baseline deaths before vs after age 65
# ============================================================================
print("\n" + "=" * 80)
print("ANALYSIS 1: Deaths among <65 baseline — before vs after reaching 65")
print("=" * 80)

df_under65 = df[df['age_under65'] == 1].copy()
n_under65 = len(df_under65)
n_deaths_under65 = int(df_under65['died'].sum())

# Age at exit = age at entry + followup years
df_under65['age_at_exit_calc'] = df_under65['age'] + df_under65['followup_years']
df_died_under65 = df_under65[df_under65['died'] == 1].copy()

died_before_65 = int((df_died_under65['age_at_exit_calc'] < 65).sum())
died_at_or_after_65 = int((df_died_under65['age_at_exit_calc'] >= 65).sum())

print(f"  Total <65 at baseline: {n_under65}")
print(f"  Deaths among <65 baseline: {n_deaths_under65}")
print(f"  Died BEFORE reaching 65: {died_before_65} ({100*died_before_65/n_deaths_under65:.1f}%)")
print(f"  Died AT/AFTER reaching 65: {died_at_or_after_65} ({100*died_at_or_after_65/n_deaths_under65:.1f}%)")

# By strain status
for strain_val, strain_label in [(0, 'Low/No strain'), (1, 'High strain')]:
    sub = df_died_under65[df_died_under65['fin_strain_binary'] == strain_val]
    n_sub = len(sub)
    before = int((sub['age_at_exit_calc'] < 65).sum())
    after = int((sub['age_at_exit_calc'] >= 65).sum())
    print(f"  {strain_label}: {n_sub} deaths — {before} before 65 ({100*before/n_sub:.1f}%), "
          f"{after} at/after 65 ({100*after/n_sub:.1f}%)")

results['analysis1_death_timing'] = {
    'n_under65_baseline': n_under65,
    'n_deaths': n_deaths_under65,
    'died_before_65': died_before_65,
    'died_at_or_after_65': died_at_or_after_65,
    'pct_died_before_65': round(100 * died_before_65 / n_deaths_under65, 1)
}

# ============================================================================
# ANALYSIS 2: Lexis expansion — time-varying pre/post-Medicare at age 65
# ============================================================================
print("\n" + "=" * 80)
print("ANALYSIS 2: Lexis expansion — person-time split at age 65")
print("=" * 80)

# Create person-period records split at age 65
# Each person gets up to 2 records: pre-65 person-time and post-65 person-time

rows = []
for _, row in df.iterrows():
    entry_age = row['age']  # baseline age
    exit_age = entry_age + row['followup_years']
    died = row['died']

    if entry_age >= 65:
        # Entire follow-up is post-65
        rows.append({
            **{c: row[c] for c in model3b_covars + ['fin_strain_binary', 'lb_weight']},
            'duration': row['followup_years'],
            'event': died,
            'post65': 1
        })
    elif exit_age < 65:
        # Entire follow-up is pre-65
        rows.append({
            **{c: row[c] for c in model3b_covars + ['fin_strain_binary', 'lb_weight']},
            'duration': row['followup_years'],
            'event': died,
            'post65': 0
        })
    else:
        # Split at age 65
        time_to_65 = 65 - entry_age
        time_after_65 = row['followup_years'] - time_to_65

        # Pre-65 segment: no event (survived to 65)
        rows.append({
            **{c: row[c] for c in model3b_covars + ['fin_strain_binary', 'lb_weight']},
            'duration': time_to_65,
            'event': 0,
            'post65': 0
        })
        # Post-65 segment: event if died
        rows.append({
            **{c: row[c] for c in model3b_covars + ['fin_strain_binary', 'lb_weight']},
            'duration': time_after_65,
            'event': died,
            'post65': 1
        })

df_lexis = pd.DataFrame(rows)
df_lexis = df_lexis[df_lexis['duration'] > 0].copy()

print(f"  Total person-period records: {len(df_lexis)}")
print(f"  Pre-65 records: {(df_lexis['post65']==0).sum()}, events: {df_lexis.loc[df_lexis['post65']==0, 'event'].sum()}")
print(f"  Post-65 records: {(df_lexis['post65']==1).sum()}, events: {df_lexis.loc[df_lexis['post65']==1, 'event'].sum()}")

# Model with time-varying post65 indicator and interaction
df_lexis['strain_x_post65'] = df_lexis['fin_strain_binary'] * df_lexis['post65']

# Unweighted Lexis model
covars_no_age = [v for v in model3b_covars if v != 'age']
lexis_formula = 'fin_strain_binary + post65 + strain_x_post65 + ' + ' + '.join(covars_no_age)

# Drop NAs from all relevant columns before fitting
lexis_cols = ['fin_strain_binary', 'duration', 'event', 'post65', 'strain_x_post65', 'lb_weight'] + covars_no_age
df_lexis = df_lexis[lexis_cols].dropna().copy()
for c in df_lexis.columns:
    df_lexis[c] = pd.to_numeric(df_lexis[c], errors='coerce')
df_lexis = df_lexis.dropna()
df_lexis = df_lexis[df_lexis['duration'] > 0].copy()

print(f"  After dropping NAs: {len(df_lexis)} records")

# Note: lifelines does not support counting-process (start-stop) notation.
# This model uses segment duration, which is an approximation. Standard errors
# do not account for within-person correlation across split segments.
cph_lexis = CoxPHFitter(penalizer=1e-5)
cph_lexis.fit(df_lexis, duration_col='duration', event_col='event', formula=lexis_formula)

strain_coef = float(cph_lexis.params_['fin_strain_binary'])
interaction_coef = float(cph_lexis.params_['strain_x_post65'])
strain_hr_pre65 = float(np.exp(strain_coef))
strain_hr_post65 = float(np.exp(strain_coef + interaction_coef))
interaction_p = float(cph_lexis.summary.loc['strain_x_post65', 'p'])

# CIs for pre-65 effect
ci_pre = cph_lexis.confidence_intervals_.loc['fin_strain_binary']
hr_pre_lower = float(np.exp(ci_pre.iloc[0]))
hr_pre_upper = float(np.exp(ci_pre.iloc[1]))
p_pre = float(cph_lexis.summary.loc['fin_strain_binary', 'p'])

# For post-65 HR, need to compute from sum of coefficients
from scipy import stats
vcov = cph_lexis.variance_matrix_
strain_idx = list(cph_lexis.params_.index).index('fin_strain_binary')
inter_idx = list(cph_lexis.params_.index).index('strain_x_post65')
combined_se = np.sqrt(vcov.iloc[strain_idx, strain_idx] + vcov.iloc[inter_idx, inter_idx] +
                      2 * vcov.iloc[strain_idx, inter_idx])
combined_coef = strain_coef + interaction_coef
hr_post_lower = float(np.exp(combined_coef - 1.96 * combined_se))
hr_post_upper = float(np.exp(combined_coef + 1.96 * combined_se))
z_post = combined_coef / combined_se
p_post = float(2 * (1 - stats.norm.cdf(abs(z_post))))

print(f"\n  Unweighted Lexis expansion results:")
print(f"    Strain HR (pre-65 person-time): {strain_hr_pre65:.2f} ({hr_pre_lower:.2f}-{hr_pre_upper:.2f}), P={p_pre:.4f}")
print(f"    Strain HR (post-65 person-time): {strain_hr_post65:.2f} ({hr_post_lower:.2f}-{hr_post_upper:.2f}), P={p_post:.4f}")
print(f"    Interaction P (strain × post65): {interaction_p:.4f}")

# Weighted Lexis model
df_lexis_wt = df_lexis.dropna(subset=['lb_weight']).copy()
n_lw = len(df_lexis_wt)
df_lexis_wt['lb_weight_norm'] = df_lexis_wt['lb_weight'] * n_lw / df_lexis_wt['lb_weight'].sum()

cph_lexis_wt = CoxPHFitter(penalizer=1e-5)
cph_lexis_wt.fit(df_lexis_wt, duration_col='duration', event_col='event',
                 formula=lexis_formula, weights_col='lb_weight_norm', robust=True)

strain_coef_wt = float(cph_lexis_wt.params_['fin_strain_binary'])
interaction_coef_wt = float(cph_lexis_wt.params_['strain_x_post65'])
strain_hr_pre65_wt = float(np.exp(strain_coef_wt))
strain_hr_post65_wt = float(np.exp(strain_coef_wt + interaction_coef_wt))
interaction_p_wt = float(cph_lexis_wt.summary.loc['strain_x_post65', 'p'])

ci_pre_wt = cph_lexis_wt.confidence_intervals_.loc['fin_strain_binary']
hr_pre_lower_wt = float(np.exp(ci_pre_wt.iloc[0]))
hr_pre_upper_wt = float(np.exp(ci_pre_wt.iloc[1]))
p_pre_wt = float(cph_lexis_wt.summary.loc['fin_strain_binary', 'p'])

vcov_wt = cph_lexis_wt.variance_matrix_
strain_idx_wt = list(cph_lexis_wt.params_.index).index('fin_strain_binary')
inter_idx_wt = list(cph_lexis_wt.params_.index).index('strain_x_post65')
combined_se_wt = np.sqrt(vcov_wt.iloc[strain_idx_wt, strain_idx_wt] + vcov_wt.iloc[inter_idx_wt, inter_idx_wt] +
                         2 * vcov_wt.iloc[strain_idx_wt, inter_idx_wt])
combined_coef_wt = strain_coef_wt + interaction_coef_wt
hr_post_lower_wt = float(np.exp(combined_coef_wt - 1.96 * combined_se_wt))
hr_post_upper_wt = float(np.exp(combined_coef_wt + 1.96 * combined_se_wt))
z_post_wt = combined_coef_wt / combined_se_wt
p_post_wt = float(2 * (1 - stats.norm.cdf(abs(z_post_wt))))

print(f"\n  Weighted Lexis expansion results:")
print(f"    Strain HR (pre-65 person-time): {strain_hr_pre65_wt:.2f} ({hr_pre_lower_wt:.2f}-{hr_pre_upper_wt:.2f}), P={p_pre_wt:.4f}")
print(f"    Strain HR (post-65 person-time): {strain_hr_post65_wt:.2f} ({hr_post_lower_wt:.2f}-{hr_post_upper_wt:.2f}), P={p_post_wt:.4f}")
print(f"    Interaction P (strain × post65): {interaction_p_wt:.4f}")

results['analysis2_lexis'] = {
    'unweighted': {
        'n_records': len(df_lexis),
        'pre65_hr': strain_hr_pre65, 'pre65_ci': (hr_pre_lower, hr_pre_upper), 'pre65_p': p_pre,
        'post65_hr': strain_hr_post65, 'post65_ci': (hr_post_lower, hr_post_upper), 'post65_p': p_post,
        'interaction_p': interaction_p
    },
    'weighted': {
        'n_records': len(df_lexis_wt),
        'pre65_hr': strain_hr_pre65_wt, 'pre65_ci': (hr_pre_lower_wt, hr_pre_upper_wt), 'pre65_p': p_pre_wt,
        'post65_hr': strain_hr_post65_wt, 'post65_ci': (hr_post_lower_wt, hr_post_upper_wt), 'post65_p': p_post_wt,
        'interaction_p': interaction_p_wt
    }
}

# ============================================================================
# ANALYSIS 3: Weight re-normalization within age strata
# ============================================================================
print("\n" + "=" * 80)
print("ANALYSIS 3: Weight re-normalization within age strata")
print("=" * 80)

df_wt = df.dropna(subset=['lb_weight']).copy()

for age_grp, mask in [('<65 years', df_wt['age_under65'] == 1),
                       ('>=65 years', df_wt['age_under65'] == 0)]:
    df_grp = df_wt[mask].copy()
    n_grp = len(df_grp)

    # Method A: Global weight normalization — weights normalized using
    # full-sample mean weight (preserves relative weight of stratum in full sample)
    full_n = len(df_wt)
    df_grp['wt_global_norm'] = df_grp['lb_weight'] * full_n / df_wt['lb_weight'].sum()

    cph_g = CoxPHFitter(penalizer=1e-5)
    formula = 'fin_strain_binary + ' + ' + '.join(model3b_covars)
    cols_needed = ['fin_strain_binary', 'followup_years', 'died', 'wt_global_norm'] + model3b_covars
    df_g = df_grp[cols_needed].dropna()
    cph_g.fit(df_g, duration_col='followup_years', event_col='died', formula=formula,
              weights_col='wt_global_norm', robust=True)
    hr_g = float(np.exp(cph_g.params_['fin_strain_binary']))
    ci_g = cph_g.confidence_intervals_.loc['fin_strain_binary']
    p_g = float(cph_g.summary.loc['fin_strain_binary', 'p'])

    # Method B: Stratum-specific weight normalization — weights normalized
    # within stratum so they sum to stratum N (standard approach)
    df_grp['wt_stratum_norm'] = df_grp['lb_weight'] * n_grp / df_grp['lb_weight'].sum()

    cols_needed_s = ['fin_strain_binary', 'followup_years', 'died', 'wt_stratum_norm'] + model3b_covars
    df_s = df_grp[cols_needed_s].dropna()
    cph_s = CoxPHFitter(penalizer=1e-5)
    cph_s.fit(df_s, duration_col='followup_years', event_col='died', formula=formula,
              weights_col='wt_stratum_norm', robust=True)
    hr_s = float(np.exp(cph_s.params_['fin_strain_binary']))
    ci_s = cph_s.confidence_intervals_.loc['fin_strain_binary']
    p_s = float(cph_s.summary.loc['fin_strain_binary', 'p'])

    print(f"\n  {age_grp} (N={n_grp}):")
    print(f"    Stratum-normalized weights: HR={hr_s:.3f} ({float(np.exp(ci_s.iloc[0])):.3f}-{float(np.exp(ci_s.iloc[1])):.3f}), P={p_s:.4f}")
    print(f"    Global-normalized weights:  HR={hr_g:.3f} ({float(np.exp(ci_g.iloc[0])):.3f}-{float(np.exp(ci_g.iloc[1])):.3f}), P={p_g:.4f}")

    results[f'analysis3_weight_renorm_{age_grp}'] = {
        'stratum_norm': {'hr': hr_s, 'ci_lower': float(np.exp(ci_s.iloc[0])), 'ci_upper': float(np.exp(ci_s.iloc[1])), 'p': p_s},
        'global_norm': {'hr': hr_g, 'ci_lower': float(np.exp(ci_g.iloc[0])), 'ci_upper': float(np.exp(ci_g.iloc[1])), 'p': p_g}
    }

# ============================================================================
# ANALYSIS 4: PS-matched age-stratified estimates
# ============================================================================
print("\n" + "=" * 80)
print("ANALYSIS 4: PS-matched age-stratified estimates")
print("=" * 80)

ps_covars = [v for v in model3b_covars]  # same covariates for PS model

for age_grp, mask in [('<65 years', df['age_under65'] == 1),
                       ('>=65 years', df['age_under65'] == 0)]:
    df_grp = df[mask].copy()

    # Build PS model
    ps_cols = ['fin_strain_binary'] + ps_covars
    df_ps = df_grp[ps_cols].dropna()

    X = df_ps[ps_covars].values
    y = df_ps['fin_strain_binary'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lr = LogisticRegression(max_iter=1000, C=1.0)
    lr.fit(X_scaled, y)
    ps = lr.predict_proba(X_scaled)[:, 1]
    df_ps = df_ps.copy()
    df_ps['ps'] = ps

    # Add outcome columns
    df_ps['followup_years'] = df_grp.loc[df_ps.index, 'followup_years']
    df_ps['died'] = df_grp.loc[df_ps.index, 'died']

    # 1:1 nearest-neighbor matching on logit PS within caliper (0.2 SD of logit PS)
    df_ps['logit_ps'] = np.log(df_ps['ps'] / (1 - df_ps['ps']))
    logit_ps_sd = df_ps['logit_ps'].std()
    caliper = 0.2 * logit_ps_sd

    treated = df_ps[df_ps['fin_strain_binary'] == 1].copy()
    control = df_ps[df_ps['fin_strain_binary'] == 0].copy()

    matched_pairs = []
    control_available = control.copy()

    for idx, row in treated.iterrows():
        if len(control_available) == 0:
            break
        distances = np.abs(control_available['logit_ps'] - row['logit_ps'])
        min_dist = distances.min()
        if min_dist <= caliper:
            match_idx = distances.idxmin()
            matched_pairs.append((idx, match_idx))
            control_available = control_available.drop(match_idx)

    if len(matched_pairs) > 0:
        treated_idx = [p[0] for p in matched_pairs]
        control_idx = [p[1] for p in matched_pairs]
        df_t = df_ps.loc[treated_idx].copy()
        df_c = df_ps.loc[control_idx].copy()
        df_t['pair_id'] = range(len(matched_pairs))
        df_c['pair_id'] = range(len(matched_pairs))
        df_matched = pd.concat([df_t, df_c])

        # Pair-stratified Cox (consistent with primary PS analysis)
        cph_m = CoxPHFitter(penalizer=1e-5)
        cph_m.fit(df_matched[['fin_strain_binary', 'followup_years', 'died', 'pair_id']],
                  duration_col='followup_years', event_col='died',
                  formula='fin_strain_binary', strata=['pair_id'], robust=True)
        hr_m = float(np.exp(cph_m.params_['fin_strain_binary']))
        ci_m = cph_m.confidence_intervals_.loc['fin_strain_binary']
        p_m = float(cph_m.summary.loc['fin_strain_binary', 'p'])

        print(f"\n  {age_grp}:")
        print(f"    N matched pairs: {len(matched_pairs)}")
        print(f"    PS-matched HR: {hr_m:.2f} ({float(np.exp(ci_m.iloc[0])):.2f}-{float(np.exp(ci_m.iloc[1])):.2f}), P={p_m:.4f}")

        results[f'analysis4_ps_matched_{age_grp}'] = {
            'n_pairs': len(matched_pairs),
            'hr': hr_m,
            'ci_lower': float(np.exp(ci_m.iloc[0])),
            'ci_upper': float(np.exp(ci_m.iloc[1])),
            'p': p_m
        }
    else:
        print(f"\n  {age_grp}: No matched pairs found within caliper")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY OF EXPLORATORY RESULTS")
print("=" * 80)

print("\n1. Death timing among <65 baseline group:")
r1 = results['analysis1_death_timing']
print(f"   {r1['pct_died_before_65']}% of deaths occurred before reaching age 65")
print(f"   → {'Most' if r1['pct_died_before_65'] > 50 else 'Less than half of'} deaths in the <65 group occurred during genuinely pre-Medicare person-time")

print("\n2. Lexis expansion (time-varying age 65 split):")
r2u = results['analysis2_lexis']['unweighted']
r2w = results['analysis2_lexis']['weighted']
print(f"   Unweighted: Pre-65 HR={r2u['pre65_hr']:.2f}, Post-65 HR={r2u['post65_hr']:.2f}, Interaction P={r2u['interaction_p']:.4f}")
print(f"   Weighted:   Pre-65 HR={r2w['pre65_hr']:.2f}, Post-65 HR={r2w['post65_hr']:.2f}, Interaction P={r2w['interaction_p']:.4f}")
if r2w['interaction_p'] < 0.05:
    print("   → Significant time-varying interaction supports the age-dependent attenuation finding")
else:
    print("   → Non-significant time-varying interaction — original baseline-age approach may overstate the interaction")

print("\n3. Weight re-normalization within strata:")
for age_grp in ['<65 years', '>=65 years']:
    r3 = results.get(f'analysis3_weight_renorm_{age_grp}')
    if r3:
        diff = abs(r3['stratum_norm']['hr'] - r3['global_norm']['hr'])
        print(f"   {age_grp}: Stratum HR={r3['stratum_norm']['hr']:.3f} vs Global HR={r3['global_norm']['hr']:.3f} (diff={diff:.3f})")
print("   → ", end="")
diff_under = abs(results.get('analysis3_weight_renorm_<65 years', {}).get('stratum_norm', {}).get('hr', 0) -
                 results.get('analysis3_weight_renorm_<65 years', {}).get('global_norm', {}).get('hr', 0))
if diff_under < 0.02:
    print("Minimal difference — weight normalization approach does not substantially affect results")
else:
    print("Notable difference — should consider reporting stratum-normalized results")

print("\n4. PS-matched age-stratified estimates:")
for age_grp in ['<65 years', '>=65 years']:
    r4 = results.get(f'analysis4_ps_matched_{age_grp}')
    if r4:
        print(f"   {age_grp}: HR={r4['hr']:.2f} ({r4['ci_lower']:.2f}-{r4['ci_upper']:.2f}), P={r4['p']:.4f}, N pairs={r4['n_pairs']}")

# Save results
with open(OUTPUT_DIR / 'exploratory_reviewer_analyses.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved to {OUTPUT_DIR / 'exploratory_reviewer_analyses.json'}")

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
