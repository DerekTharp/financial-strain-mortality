#!/usr/bin/env python3
"""
02_figures.py
Figure generation for Financial Strain and All-Cause Mortality study.

Current JECH package figures:
1. Age-stratified Kaplan-Meier survival curves (Figure 1)
2. Forest plot of hazard ratios across models and subgroups (Figure 2)
Figure S2. Propensity score overlap
Figure S3. Directed acyclic graph (DAG)

Date: February 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import json

# Paths
PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / 'output'
FIGURES_DIR = OUTPUT_DIR / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def format_p(p, prefix=True):
    """Format P-value with leading zero and proper rounding."""
    if p < 0.001:
        val = '<0.001'
    elif p < 0.02:
        val = f'{p:.3f}'
    elif 0.045 < p < 0.055:
        # Report borderline P values to 3 decimals for clarity
        val = f'{p:.3f}'
    else:
        val = f'{round(p, 2):.2f}'
    if prefix:
        return f'P {"" if val.startswith("<") else "= "}{val}'
    return val

# Load the analytic sample
print("Loading analytic sample...")
df = pd.read_csv(OUTPUT_DIR / 'analytic_sample_v5.csv')
print(f"Loaded {len(df)} observations")

# Load results manifest
with open(OUTPUT_DIR / 'results_manifest_v5.json', 'r') as f:
    results = json.load(f)

# ============================================================================
# FIGURE 1: Age-Stratified KM Curves
# ============================================================================

print("\n" + "="*80)
print("FIGURE 1: Age-Stratified Mortality Curves")
print("="*80)

# Use complete cases for KM
km_vars = ['followup_years', 'died', 'fin_strain_binary']
df_km = df[km_vars].dropna()

fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

age_in_km = df['age'].loc[df_km.index]

for idx, age_label in enumerate(['<65 years', '≥65 years']):
    ax = axes[idx]
    df_age = df_km[age_in_km < 65].copy() if idx == 0 else df_km[age_in_km >= 65].copy()

    df_age_no = df_age[df_age['fin_strain_binary'] == 0]
    df_age_yes = df_age[df_age['fin_strain_binary'] == 1]

    kmf_no = KaplanMeierFitter()
    kmf_yes = KaplanMeierFitter()

    kmf_no.fit(df_age_no['followup_years'], df_age_no['died'], label='No/Low Financial Strain')
    kmf_yes.fit(df_age_yes['followup_years'], df_age_yes['died'], label='High Financial Strain')

    kmf_no.plot_survival_function(ax=ax, color='#2166AC', linewidth=2, ci_show=True, ci_alpha=0.15)
    kmf_yes.plot_survival_function(ax=ax, color='#B2182B', linewidth=2, ci_show=True, ci_alpha=0.15)

    # Log-rank
    lr_age = logrank_test(df_age_no['followup_years'], df_age_yes['followup_years'],
                          df_age_no['died'], df_age_yes['died'])

    p_age = format_p(lr_age.p_value, prefix=True)

    ax.set_title(f'Age {age_label}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Years Since Baseline', fontsize=11)
    ax.set_ylabel('Survival Probability', fontsize=11)
    ax.set_xlim(0, 17)
    ax.set_ylim(0.2 if idx == 1 else 0.6, 1.0)
    ax.legend(loc='lower left', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add N and log-rank P in subtitle
    ax.text(0.5, 0.02, f'N = {len(df_age)}, Log-rank {p_age}',
            transform=ax.transAxes, fontsize=9, ha='center')

plt.tight_layout()

for fmt in ['pdf', 'png']:
    plt.savefig(FIGURES_DIR / f'Figure1_Age_Stratified.{fmt}',
                dpi=300 if fmt == 'png' else None, bbox_inches='tight')
print("  Saved Figure 1")
plt.close()

# ============================================================================
# FIGURE 2: Forest Plot of All Results
# ============================================================================

print("\n" + "="*80)
print("FIGURE 2: Forest Plot (Comprehensive)")
print("="*80)

# Build forest plot data from results manifest (not hard-coded)
def _get(d, *keys, default=np.nan):
    """Safely traverse nested dict, returning default if any key missing."""
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return default
    return d

r = results
forest_data = [
    # (Label, HR, CI_lower, CI_upper, P, Group)
    # --- Survey-Weighted Primary Analyses ---
    ('Survey-Weighted (Primary)', np.nan, np.nan, np.nan, np.nan, 'Header'),
    ('  Model 3b (weighted, primary)',
     _get(r, 'weighted_primary', 'model3b', 'hr'),
     _get(r, 'weighted_primary', 'model3b', 'ci_lower'),
     _get(r, 'weighted_primary', 'model3b', 'ci_upper'),
     _get(r, 'weighted_primary', 'model3b', 'p_value'),
     'Primary'),
    ('  Model 3 (weighted, no SES)',
     _get(r, 'weighted_primary', 'model3', 'hr'),
     _get(r, 'weighted_primary', 'model3', 'ci_lower'),
     _get(r, 'weighted_primary', 'model3', 'ci_upper'),
     _get(r, 'weighted_primary', 'model3', 'p_value'),
     'Weighted'),
    ('  Model 4 (weighted, + CES-D)',
     _get(r, 'weighted_primary', 'model4', 'hr'),
     _get(r, 'weighted_primary', 'model4', 'ci_lower'),
     _get(r, 'weighted_primary', 'model4', 'ci_upper'),
     _get(r, 'weighted_primary', 'model4', 'p_value'),
     'Weighted'),
    ('', np.nan, np.nan, np.nan, np.nan, ''),
    # --- Weighted Age-Stratified ---
    ('Age-Stratified (Weighted)', np.nan, np.nan, np.nan, np.nan, 'Header'),
    ('  Age <65 years (weighted)',
     _get(r, 'weighted_age_interaction', '<65 years', 'hr'),
     _get(r, 'weighted_age_interaction', '<65 years', 'ci_lower'),
     _get(r, 'weighted_age_interaction', '<65 years', 'ci_upper'),
     _get(r, 'weighted_age_interaction', '<65 years', 'p_value'),
     'Age'),
    ('  Age ≥65 years (weighted)',
     _get(r, 'weighted_age_interaction', '>=65 years', 'hr'),
     _get(r, 'weighted_age_interaction', '>=65 years', 'ci_lower'),
     _get(r, 'weighted_age_interaction', '>=65 years', 'ci_upper'),
     _get(r, 'weighted_age_interaction', '>=65 years', 'p_value'),
     'Age'),
    ('', np.nan, np.nan, np.nan, np.nan, ''),
    # --- Unweighted & Sensitivity ---
    ('Unweighted & Sensitivity', np.nan, np.nan, np.nan, np.nan, 'Header'),
    ('  Model 3b (unweighted)',
     _get(r, 'primary_result_model3b', 'hr'),
     _get(r, 'primary_result_model3b', 'ci_lower'),
     _get(r, 'primary_result_model3b', 'ci_upper'),
     _get(r, 'primary_result_model3b', 'p_value'),
     'Sensitivity'),
    ('  PS-matched (unweighted)',
     _get(r, 'sensitivity_analyses', 'ps_matched', 'hr'),
     _get(r, 'sensitivity_analyses', 'ps_matched', 'ci_lower'),
     _get(r, 'sensitivity_analyses', 'ps_matched', 'ci_upper'),
     _get(r, 'sensitivity_analyses', 'ps_matched', 'p_value'),
     'Sensitivity'),
    ('  Exclude 1-year deaths',
     _get(r, 'sensitivity_analyses', 'exclude_1yr', 'hr'),
     _get(r, 'sensitivity_analyses', 'exclude_1yr', 'ci_lower'),
     _get(r, 'sensitivity_analyses', 'exclude_1yr', 'ci_upper'),
     _get(r, 'sensitivity_analyses', 'exclude_1yr', 'p_value'),
     'Sensitivity'),
    ('  Exclude 2-year deaths',
     _get(r, 'sensitivity_analyses', 'exclude_2yr', 'hr'),
     _get(r, 'sensitivity_analyses', 'exclude_2yr', 'ci_lower'),
     _get(r, 'sensitivity_analyses', 'exclude_2yr', 'ci_upper'),
     _get(r, 'sensitivity_analyses', 'exclude_2yr', 'p_value'),
     'Sensitivity'),
    ('', np.nan, np.nan, np.nan, np.nan, ''),
    ('Incident Heart Disease', np.nan, np.nan, np.nan, np.nan, 'Header'),
    ('  Cause-specific HR',
     _get(r, 'cardiac', 'cause_specific', 'hr'),
     _get(r, 'cardiac', 'cause_specific', 'ci_lower'),
     _get(r, 'cardiac', 'cause_specific', 'ci_upper'),
     _get(r, 'cardiac', 'cause_specific', 'p_value'),
     'Cardiac'),
]

# Warn about any missing manifest entries
for label, hr, ci_lo, ci_hi, p, group in forest_data:
    if group and group not in ('', 'Header') and pd.isna(hr):
        print(f"  WARNING: Missing manifest data for '{label}' — update manifest after reanalysis")

fig, ax = plt.subplots(figsize=(10, 8), dpi=150)

y_positions = []
y_pos = len(forest_data)

for i, (label, hr, ci_lo, ci_hi, p, group) in enumerate(forest_data):
    y = y_pos - i
    y_positions.append(y)

    if pd.isna(hr):
        if group == 'Header':
            ax.text(0.05, y, label, fontsize=10, fontweight='bold', va='center', transform=ax.get_yaxis_transform())
        continue

    # Plot point estimate
    if group == 'Primary':
        color, marker_size = '#2166AC', 100
    elif group in ('Weighted', 'Age'):
        color, marker_size = '#4393C3', 70
    else:
        color, marker_size = '#888888', 55

    ax.scatter(hr, y, s=marker_size, color=color, zorder=3, marker='s')

    # Plot CI
    ax.plot([ci_lo, ci_hi], [y, y], color=color, linewidth=2, zorder=2)

    # Add label
    ax.text(0.05, y, label, fontsize=9, va='center', transform=ax.get_yaxis_transform())

    # Add HR (CI) and P on right
    hr_text = f'{hr:.2f} ({ci_lo:.2f}-{ci_hi:.2f})'
    p_text = format_p(p, prefix=False)

    ax.text(
        1.8, y, hr_text, fontsize=9, va='center', ha='right',
        bbox=dict(facecolor='white', edgecolor='none', pad=0.15)
    )
    ax.text(2.0, y, p_text, fontsize=9, va='center', ha='center')

# Reference line at HR=1
ax.axvline(x=1, color='gray', linestyle='--', linewidth=1, zorder=1)

# Styling
ax.set_xlim(0.5, 2.2)
ax.set_ylim(0, len(forest_data) + 1)
ax.set_xlabel('Hazard Ratio (95% CI)', fontsize=11)
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

# Add column headers
ax.text(1.8, len(forest_data) + 0.5, 'HR (95% CI)', fontsize=10, fontweight='bold', ha='right')
ax.text(2.0, len(forest_data) + 0.5, 'P', fontsize=10, fontweight='bold', ha='center')

plt.tight_layout()

for fmt in ['pdf', 'png']:
    plt.savefig(FIGURES_DIR / f'Figure2_Forest.{fmt}',
                dpi=300 if fmt == 'png' else None, bbox_inches='tight')
print("  Saved Figure 2")
plt.close()

# ============================================================================
# SUPPLEMENTARY FIGURE S2: Propensity Score Overlap
# ============================================================================

print("\n" + "="*80)
print("SUPPLEMENTARY FIGURE S2: Propensity Score Overlap")
print("="*80)

# Re-fit PS model (same specification as 01_primary_analysis.py STEP 6) for plotting
ps_vars = ['age', 'female', 'race_nh_black', 'race_hispanic', 'race_nh_other',
           'education_yrs', 'married_partnered', 'current_smoker', 'bmi',
           'baseline_hypertension', 'baseline_diabetes', 'baseline_heart', 'baseline_stroke',
           'asinh_income', 'asinh_wealth', 'srh_fairpoor']

ps_model_vars = ['fin_strain_binary'] + ps_vars
df_ps = df[ps_model_vars].dropna().copy()

X = df_ps[ps_vars].values
y = df_ps['fin_strain_binary'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

ps_model = LogisticRegression(max_iter=1000, random_state=42)
ps_model.fit(X_scaled, y)
df_ps['ps'] = ps_model.predict_proba(X_scaled)[:, 1]

ps_treated = df_ps[df_ps['fin_strain_binary'] == 1]['ps']
ps_control = df_ps[df_ps['fin_strain_binary'] == 0]['ps']

fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

bins = np.linspace(0, 1, 51)
ax.hist(ps_control, bins=bins, alpha=0.5, label=f'No/Low Financial Strain (n = {len(ps_control)})',
        color='#2166ac', edgecolor='white', linewidth=0.5, density=True)
ax.hist(ps_treated, bins=bins, alpha=0.5, label=f'High Financial Strain (n = {len(ps_treated)})',
        color='#b2182b', edgecolor='white', linewidth=0.5, density=True)

ax.set_xlabel('Propensity Score', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('', fontsize=13)
ax.legend(fontsize=10, frameon=True, loc='upper right')
ax.set_xlim(0, 0.7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

for fmt in ['pdf', 'png']:
    plt.savefig(FIGURES_DIR / f'eFigure_S2_PS_Overlap.{fmt}',
                dpi=300 if fmt == 'png' else None, bbox_inches='tight')
print(f"  Saved supplementary figure S2 (PS overlap)")
print(f"  PS range - Control: [{ps_control.min():.3f}, {ps_control.max():.3f}], "
      f"Treated: [{ps_treated.min():.3f}, {ps_treated.max():.3f}]")
plt.close()

# ============================================================================
# SUPPLEMENTARY FIGURE S3: Directed Acyclic Graph (DAG)
# ============================================================================

print("\n" + "="*80)
print("SUPPLEMENTARY FIGURE S3: Directed Acyclic Graph")
print("="*80)

fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
ax.set_xlim(-0.5, 10.5)
ax.set_ylim(-1.5, 8.5)
ax.axis('off')

# Node positions
nodes = {
    'Financial\nStrain': (2, 4),
    'All-Cause\nMortality': (8, 4),
    'Income/\nWealth': (5, 6.5),
    'Age': (0, 7),
    'Sex': (1, 7.5),
    'Race/\nEthnicity': (2, 7.5),
    'Education': (3, 7.5),
    'Marital\nStatus': (4, 7.5),
    'Smoking': (5, 1),
    'BMI': (6, 1),
    'Hypertension': (7, 1.5),
    'Diabetes': (8, 1.5),
    'Heart\nDisease': (9, 2),
    'Childhood\nSES': (0, 2),
    'Personality/\nAffectivity': (0, 5.5),
}

# Draw nodes
for name, (x, y) in nodes.items():
    if name in ['Childhood\nSES', 'Personality/\nAffectivity']:
        bbox = dict(boxstyle='round,pad=0.4', facecolor='#f0f0f0',
                    edgecolor='gray', linestyle='dashed', linewidth=1.5)
    elif name == 'Income/\nWealth':
        bbox = dict(boxstyle='round,pad=0.4', facecolor='#E8D5B7',
                    edgecolor='#8B7355', linewidth=1.5)
    elif name in ['Financial\nStrain', 'All-Cause\nMortality']:
        bbox = dict(boxstyle='round,pad=0.5', facecolor='#D4E6F1',
                    edgecolor='#2C3E50', linewidth=2)
    else:
        bbox = dict(boxstyle='round,pad=0.3', facecolor='white',
                    edgecolor='#555555', linewidth=1)

    fontsize = 10 if name in ['Financial\nStrain', 'All-Cause\nMortality'] else 8
    ax.text(x, y, name, ha='center', va='center', fontsize=fontsize,
            fontweight='bold' if name in ['Financial\nStrain', 'All-Cause\nMortality'] else 'normal',
            bbox=bbox, zorder=3)

# Arrow helper
def draw_arrow(ax, start, end, color='#333333', style='->', linestyle='-', lw=1.2):
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle=style, color=color,
                                linestyle=linestyle, lw=lw),
                zorder=2)

# Main causal arrow: Financial Strain → Mortality
draw_arrow(ax, (3.2, 4), (6.8, 4), color='#2C3E50', lw=2.5)

# Income/Wealth: dual role arrows (dashed to indicate ambiguity)
draw_arrow(ax, (5, 6.0), (3.0, 4.5), color='#8B7355', linestyle='--', lw=1.5)
draw_arrow(ax, (5, 6.0), (7.0, 4.5), color='#8B7355', linestyle='--', lw=1.5)

# Confounders → Exposure
for name in ['Age', 'Sex', 'Race/\nEthnicity', 'Education', 'Marital\nStatus']:
    x, y = nodes[name]
    draw_arrow(ax, (x, y - 0.5), (2, 4.5), color='#888888', lw=0.8)

# Confounders → Outcome
for name in ['Age', 'Sex', 'Race/\nEthnicity', 'Education', 'Marital\nStatus']:
    x, y = nodes[name]
    draw_arrow(ax, (x, y - 0.5), (8, 4.5), color='#888888', lw=0.8)

# Behavioral/health mediators
for name in ['Smoking', 'BMI', 'Hypertension', 'Diabetes', 'Heart\nDisease']:
    x, y = nodes[name]
    draw_arrow(ax, (2, 3.5), (x, y + 0.5), color='#888888', lw=0.8, linestyle='--')
    draw_arrow(ax, (x, y + 0.5), (8, 3.5), color='#888888', lw=0.8)

# Unmeasured confounders (dashed arrows)
draw_arrow(ax, (0.8, 2.5), (1.8, 3.5), color='gray', linestyle=':', lw=1)
draw_arrow(ax, (0.8, 2.5), (4.8, 6.0), color='gray', linestyle=':', lw=1)
draw_arrow(ax, (0.8, 5.5), (1.8, 4.5), color='gray', linestyle=':', lw=1)
draw_arrow(ax, (0.8, 5.5), (7.5, 4.5), color='gray', linestyle=':', lw=1)

# Age effect modification annotation
ax.annotate('Effect modifier', xy=(1.5, 5.0), fontsize=9, fontstyle='italic',
            color='#2166AC', ha='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#EBF5FB', edgecolor='#2166AC'))
draw_arrow(ax, (0, 6.5), (1.5, 5.3), color='#2166AC', lw=1.5, linestyle='-.')

# Legend
legend_y = -0.5
ax.plot([0.5, 1.5], [legend_y, legend_y], color='#2C3E50', lw=2.5, zorder=3)
ax.annotate('', xy=(1.5, legend_y), xytext=(1.4, legend_y),
            arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=2.5))
ax.text(1.8, legend_y, 'Primary association', fontsize=8, va='center')

ax.plot([3.5, 4.5], [legend_y, legend_y], color='#8B7355', lw=1.5, linestyle='--', zorder=3)
ax.text(4.8, legend_y, 'Confounder and/or mediator', fontsize=8, va='center')

ax.plot([7, 8], [legend_y, legend_y], color='gray', lw=1, linestyle=':', zorder=3)
ax.text(8.3, legend_y, 'Unmeasured', fontsize=8, va='center')

plt.tight_layout()
for fmt in ['pdf', 'png']:
    plt.savefig(FIGURES_DIR / f'eFigure_S3_DAG.{fmt}',
                dpi=300 if fmt == 'png' else None, bbox_inches='tight')
print("  Saved supplementary figure S3 (DAG)")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("FIGURES COMPLETE")
print("="*80)

print(f"""
Figures Generated:
  Figure 1: Age-stratified mortality curves — Figure1_Age_Stratified.pdf/.png
  Figure 2: Forest plot - comprehensive results — Figure2_Forest.pdf/.png
  Figure S2: Propensity score overlap — eFigure_S2_PS_Overlap.pdf/.png
  Figure S3: DAG — eFigure_S3_DAG.pdf/.png

All figures saved to: {FIGURES_DIR}
""")
