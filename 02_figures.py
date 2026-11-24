"""
Figure 1 (Kaplan-Meier curves by financial strain and age group), figure 2 (forest plot of
hazard ratios across models), figure S2 (directed acyclic graph) and figure S3 (propensity
score overlap).

Reads output/analytic_sample.csv and the tables written by 01_primary_analysis.py,
11_survey_prep.py and 12_survey_models.do. Writes PDF and PNG files to output/figures/.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from diagram_geometry import Box, arrow, canvas, link, save
from model_specs import MODEL3B_COVARS

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / 'output'
TABLES_DIR = OUTPUT_DIR / 'tables'
FIGURES_DIR = OUTPUT_DIR / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def format_p(p, prefix=True):
    """Format P-value with leading zero."""
    if p < 0.001:
        val = '<0.001'
    elif p < 0.01:
        val = f'{p:.3f}'
    elif 0.045 < p < 0.055:
        # Report borderline P values to 3 decimals for clarity
        val = f'{p:.3f}'
    else:
        val = f'{round(p, 2):.2f}'
    if prefix:
        return f'P {"" if val.startswith("<") else "= "}{val}'
    return val

df = pd.read_csv(OUTPUT_DIR / 'analytic_sample.csv')

# Figure 1: Kaplan-Meier curves by age group

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

    kmf_no.fit(df_age_no['followup_years'], df_age_no['died'], label='No/not upsetting strain')
    kmf_yes.fit(df_age_yes['followup_years'], df_age_yes['died'], label='Upsetting financial strain')

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
print("Saved Figure 1")
plt.close()

# Figure 2: forest plot

# Rows come from the survey-design models (svy_models.csv), the matched refits
# (matched_refit.csv), the weighted cardiac model (svy_cardiac.csv),
# and the unweighted rows in mortality_unweighted.csv.
svy_models = pd.read_csv(TABLES_DIR / 'svy_models.csv').set_index('label')
matched_refit = pd.read_csv(TABLES_DIR / 'matched_refit.csv')
cardiac_svy = pd.read_csv(TABLES_DIR / 'svy_cardiac.csv').iloc[0]
unweighted = pd.read_csv(TABLES_DIR / 'mortality_unweighted.csv').set_index('Model')


def svy_row(label):
    s = svy_models.loc[label]
    return float(s['hr']), float(s['lo']), float(s['hi']), float(s['p'])


def matched_row(spec):
    s = matched_refit[(matched_refit['Spec'] == spec) &
                      (matched_refit['Fit'] == 'marginal_pair_clustered')].iloc[0]
    return float(s['HR']), float(s['CI_lower']), float(s['CI_upper']), float(s['P'])


def unweighted_row(model):
    s = unweighted.loc[model]
    lo, hi = (float(x) for x in str(s['CI']).split('-'))
    p = str(s['P'])
    p_val = 0.0005 if p.startswith('<') else float(p)
    return float(s['HR']), lo, hi, p_val


forest_data = [
    # (Label, HR, CI_lower, CI_upper, P, Group)
    ('Survey-Weighted (Primary)', np.nan, np.nan, np.nan, np.nan, 'Header'),
    ('  Model 3b (weighted, primary)', *svy_row('Model 3b (svy, primary)'), 'Primary'),
    ('  Model 3 (weighted, no income/wealth)', *svy_row('Model 3 (svy)'), 'Weighted'),
    ('  Model 4 (weighted, + CES-D)', *svy_row('Model 4 +depression (svy)'), 'Weighted'),
    ('', np.nan, np.nan, np.nan, np.nan, ''),
    ('Age-Stratified (Weighted)', np.nan, np.nan, np.nan, np.nan, 'Header'),
    ('  Age <65 years (interaction)', *svy_row('Primary rule (admin. censoring 2022, Tracker+NDI deaths), age <65 (interaction)'), 'Age'),
    ('  Age ≥65 years (interaction)', *svy_row('Primary rule (admin. censoring 2022, Tracker+NDI deaths), age >=65 (interaction)'), 'Age'),
    ('', np.nan, np.nan, np.nan, np.nan, ''),
    ('Population Weighting & Matching', np.nan, np.nan, np.nan, np.nan, 'Header'),
    ('  IPTW (population-average)', *svy_row('IPTW a (composite weights)'), 'Sensitivity'),
    ('  IPTW (weights truncated 1st/99th)', *svy_row('IPTW a_trunc (composite weights)'),
     'Sensitivity'),
    ('  PS-matched (Model 3b covariates)', *matched_row('model3b'), 'Sensitivity'),
    ('  PS-matched (+ self-rated health)', *matched_row('model3b_plus_srh'), 'Sensitivity'),
    ('', np.nan, np.nan, np.nan, np.nan, ''),
    ('Unweighted & Reverse Causation', np.nan, np.nan, np.nan, np.nan, 'Header'),
    ('  Model 3b (unweighted)', *unweighted_row('Model 3b'), 'Sensitivity'),
    ('  Exclude 1-year deaths', *unweighted_row('Excluding 1-year deaths'), 'Sensitivity'),
    ('  Exclude 2-year deaths', *unweighted_row('Excluding 2-year deaths'), 'Sensitivity'),
    ('', np.nan, np.nan, np.nan, np.nan, ''),
    ('Incident Heart Disease', np.nan, np.nan, np.nan, np.nan, 'Header'),
    ('  Cause-specific HR (weighted)',
     float(cardiac_svy['hr']), float(cardiac_svy['lo']), float(cardiac_svy['hi']),
     float(cardiac_svy['p']), 'Cardiac'),
]

# Guard against silently missing source rows
for label, hr, ci_lo, ci_hi, p, group in forest_data:
    if group and group not in ('', 'Header') and pd.isna(hr):
        raise ValueError(f"Missing source data for forest row '{label}'")

# Three-panel layout: labels | forest | numeric columns, so the reference line and
# confidence intervals never cross text.
fig = plt.figure(figsize=(12, 8), dpi=150)
gs = fig.add_gridspec(1, 3, width_ratios=[2.5, 3.4, 1.6], wspace=0.03)
ax_lab = fig.add_subplot(gs[0])
ax_plot = fig.add_subplot(gs[1])
ax_num = fig.add_subplot(gs[2])

n_rows = len(forest_data)
for ax in (ax_lab, ax_num):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, n_rows + 1)
    ax.axis('off')
ax_plot.set_ylim(0, n_rows + 1)

for i, (label, hr, ci_lo, ci_hi, p, group) in enumerate(forest_data):
    y = n_rows - i

    if pd.isna(hr):
        if group == 'Header':
            ax_lab.text(0.0, y, label, fontsize=10, fontweight='bold', va='center')
        continue

    if group == 'Primary':
        color, marker_size = '#2166AC', 100
    elif group in ('Weighted', 'Age'):
        color, marker_size = '#4393C3', 70
    else:
        color, marker_size = '#888888', 55

    ax_lab.text(0.05, y, label.strip(), fontsize=9, va='center')
    ax_plot.scatter(hr, y, s=marker_size, color=color, zorder=3, marker='s')
    ax_plot.plot([ci_lo, ci_hi], [y, y], color=color, linewidth=2, zorder=2)
    ax_num.text(0.62, y, f'{hr:.2f} ({ci_lo:.2f}-{ci_hi:.2f})', fontsize=9,
                va='center', ha='right')
    ax_num.text(0.88, y, format_p(p, prefix=False), fontsize=9, va='center', ha='center')

# Reference line at HR=1 (confined to the plot panel)
ax_plot.axvline(x=1, color='gray', linestyle='--', linewidth=1, zorder=1)

ax_plot.set_xlim(0.55, 1.8)
ax_plot.set_xlabel('Hazard Ratio (95% CI)', fontsize=11)
ax_plot.set_yticks([])
for side in ('top', 'right', 'left'):
    ax_plot.spines[side].set_visible(False)

ax_num.text(0.62, n_rows + 0.7, 'HR (95% CI)', fontsize=10, fontweight='bold', ha='right')
ax_num.text(0.88, n_rows + 0.7, 'P', fontsize=10, fontweight='bold', ha='center')

plt.tight_layout()

for fmt in ['pdf', 'png']:
    plt.savefig(FIGURES_DIR / f'Figure2_Forest.{fmt}',
                dpi=300 if fmt == 'png' else None, bbox_inches='tight')
print("Saved Figure 2")
plt.close()

# Figure S3: propensity score overlap

# The propensity model of the primary matching specification (Model 3b covariates,
# 11_survey_prep.py), refitted for plotting
ps_vars = list(MODEL3B_COVARS)

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
ax.hist(ps_control, bins=bins, alpha=0.5, label=f'No/not upsetting strain (n = {len(ps_control)})',
        color='#2166ac', edgecolor='white', linewidth=0.5, density=True)
ax.hist(ps_treated, bins=bins, alpha=0.5, label=f'Upsetting financial strain (n = {len(ps_treated)})',
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
    plt.savefig(FIGURES_DIR / f'FigureS3_PS_Overlap.{fmt}',
                dpi=300 if fmt == 'png' else None, bbox_inches='tight')
print("Saved Figure S3")
plt.close()

# Figure S2: directed acyclic graph

fig, ax = canvas(504, 411)
primary = dict(facecolor='#D4E6F1', edgecolor='#2C3E50', linewidth=1.35)
adjusted = dict(facecolor='white', edgecolor='#555555', linewidth=1)
unmeasured = dict(facecolor='#f0f0f0', edgecolor='gray', linestyle='dashed', linewidth=1)
mediator = dict(facecolor='#FDF2E3', edgecolor='#B9770E', linewidth=1)
modifier = dict(facecolor='#EBF5FB', edgecolor='#2166AC', linewidth=1)

# Exposure on the left and outcome on the right, joined by the main arrow. The confounders are in a row along
# the top, the effect modifier is above the middle of the main arrow, and the mediators are below it.
row, top = 178, 378
strain, mortality = Box(57, row, 104, 37), Box(447, row, 104, 37)
strain.draw(ax, 'Upsetting\nfinancial strain', size=10.3, weight='bold', **primary)
mortality.draw(ax, 'All-cause\nmortality', size=10.3, weight='bold', **primary)
link(ax, strain, mortality, primary['edgecolor'], 1.8)

mid_x = (strain.x + mortality.x) / 2
confounders = [
    (Box(strain.x, top, 104, 42), 'Childhood\nsocioeconomic\ncircumstances', unmeasured),
    (Box(mid_x, top, 248, 53),
     'Adjusted confounders\nAge, sex, race/ethnicity, education, marital status,\n'
     'household income, total wealth, smoking, BMI,\nhypertension, diabetes, heart disease, stroke', adjusted),
    (Box(mortality.x, top, 104, 42), 'Personality /\nnegative\naffectivity', unmeasured),
]
for box, label, border in confounders:
    box.draw(ax, label, size=9, **border)
    for end in (strain, mortality):
        link(ax, box, end, border['edgecolor'], style=':' if border is unmeasured else '-')

age = Box(mid_x, row + 36, 90, 30)
age.draw(ax, 'Age\n(effect modifier)', size=8.9, italic=True, ink=modifier['edgecolor'], **modifier)
arrow(ax, [age.edge('bottom'), (mid_x, row)], modifier['edgecolor'], style='-.')

mediators = [
    (Box(mid_x, row - 40, 88, 30), 'Delayed or\nforgone care'),
    (Box(mid_x, row - 86, 88, 30), 'Physiological\nstress response'),
]
for box, label in mediators:
    box.draw(ax, label, size=9, **mediator)
    link(ax, strain, box, mediator['edgecolor'])
    link(ax, box, mortality, mediator['edgecolor'])

legend_y = 48
ax.text(5, legend_y, 'Node classes:', ha='left', va='center', fontsize=8.4,
        fontweight='bold', fontfamily='DejaVu Sans')
legend = [('exposure/outcome', 87, primary), ('adjusted confounders', 96, adjusted),
          ('unmeasured confounders', 115, unmeasured), ('unadjusted mediators', 104, mediator)]
left = 79
for label, width, border in legend:
    Box(left + width / 2, legend_y, width, 18).draw(ax, label, size=8, **border)
    left += width + 6

edge_legend = ('Edges: arrows between nodes are assumed causal effects (dotted = from unmeasured confounders); the arrow from\n'
               'age to the strain-mortality edge denotes effect modification. Mediator pathways are deliberately not adjusted for.')
ax.text(5, 26, edge_legend, ha='left', va='top', fontsize=8.4,
        linespacing=1.22, fontfamily='DejaVu Sans')

save(fig, FIGURES_DIR, 'FigureS2_DAG')
print("Saved Figure S2")
