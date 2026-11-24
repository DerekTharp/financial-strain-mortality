"""
Participant flow diagram (figure S1).

Reads output/tables/participant_flow.csv (03_secondary_analysis.py) and
output/analytic_sample_survey.csv (01b_export_survey_data.py).
"""

import pandas as pd
from pathlib import Path

from diagram_geometry import Box, arrow, canvas, line, save
from model_specs import MODEL3B_COVARS, EXPOSURE_VAR

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / 'output'
FIGURES_DIR = OUTPUT_DIR / 'figures'
TABLES_DIR = OUTPUT_DIR / 'tables'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

flow_df = pd.read_csv(TABLES_DIR / 'participant_flow.csv')

required_steps = {1, 2, 3, 4, 5}
steps = {}
for _, row in flow_df.iterrows():
    steps[int(row['step'])] = {
        'n': int(row['n']),
        'excluded': int(row['excluded']),
        'excluded_heart': row.get('excluded_heart'),
        'excluded_stroke_only': row.get('excluded_stroke_only'),
        'excluded_other': row.get('excluded_other'),
    }
missing = required_steps - set(steps.keys())
if missing:
    raise ValueError(f"participant_flow.csv missing steps: {sorted(missing)}")

# Step 5 breakdown (baseline heart disease, stroke only, other) from the flow CSV
step5_excluded = steps[5]['excluded']
n_heart = int(steps[5]['excluded_heart'])
n_stroke = int(steps[5]['excluded_stroke_only'])
n_other = int(steps[5]['excluded_other'])
assert n_heart + n_stroke + n_other == step5_excluded, 'step-5 exclusion breakdown does not sum'

# Primary survey-weighted sample: complete covariates plus valid weight and design data
svy_df = pd.read_csv(OUTPUT_DIR / 'analytic_sample_survey.csv', low_memory=False)
need = list(MODEL3B_COVARS) + [EXPOSURE_VAR, 'followup_years', 'died',
                               'lb_weight', 'raestrat', 'raehsamp']
n_weighted = int((svy_df[need].notna().all(axis=1) & (svy_df['lb_weight'] > 0)).sum())

def fmt(n):
    return f'{n:,}'

fig, ax = canvas(504, 542)
blue, red = '#2166AC', '#B2182B'
cohort = dict(facecolor='white', edgecolor=blue, linewidth=1.15)
excluded = dict(facecolor='#fff5f5', edgecolor=red, linewidth=0.85)

# Four cohort boxes in one column, each joined to the next by a vertical connector.
height, gap = 55, 29
pitch = height + gap
main = [Box(209, 490 - pitch * i, 201, height) for i in range(4)]

# The two analysis cohorts hang from a fork under the last cohort box, one below each of its side edges.
# The branches are one pitch long, which leaves room for an exclusion box beside the cardiac branch.
stem = main[-1].edge('bottom')
fork = (stem[0], stem[1] - gap / 2)
offset = main[-1].width / 2
children = [Box(fork[0] + sign * offset, fork[1] - pitch - 28, 2 * offset - 12, 56) for sign in (-1, 1)]
cardiac = children[1]

# Each exclusion leaves the midpoint of the connector into the cohort it defines.
joints = [(upper.x, (upper.edge('bottom')[1] + lower.edge('top')[1]) / 2) for upper, lower in zip(main, main[1:])]
joints.append((cardiac.x, (fork[1] + cardiac.edge('top')[1]) / 2))
side = [Box(412, y, 174, height) for _, y in joints]

labels = [
    f'HRS 2006 (Wave 8) Respondents\nN = {fmt(steps[1]["n"])}',
    f'Returned Leave-Behind Questionnaire\nwith valid financial strain response\nN = {fmt(steps[2]["n"])}',
    f'Age ≥50 years at baseline\nN = {fmt(steps[3]["n"])}',
    f'Complete covariate data\n(Mortality analysis cohort)\nN = {fmt(steps[4]["n"])}',
]
last_exclusion = [
    f'Excluded (n = {fmt(step5_excluded)}):',
    f'• Baseline heart disease (n = {fmt(n_heart)})',
    f'• Baseline stroke only (n = {fmt(n_stroke)})',
]
if n_other > 0:
    last_exclusion.append(f'• Other exclusions (n = {fmt(n_other)})')
exclusion_labels = [
    f'Excluded (n = {fmt(steps[2]["excluded"])}):\n• No valid LBQ financial\nstrain response',
    f'Excluded (n = {fmt(steps[3]["excluded"])}):\n• Age <50 years',
    f'Excluded (n = {fmt(steps[4]["excluded"])}):\n• Missing covariates',
    '\n'.join(last_exclusion),
]
child_labels = [
    f'Valid survey weights\nand design data\n(Primary weighted analysis)\nN = {fmt(n_weighted)}',
    f'Free of heart disease or stroke\nat baseline\n(Cardiac analysis cohort)\nN = {fmt(steps[5]["n"])}',
]

for box, label in zip(main, labels):
    box.draw(ax, label, size=8.7, weight='bold', **cohort)
for box, label in zip(side, exclusion_labels):
    box.draw(ax, label, size=8.7, **excluded)
for box, label in zip(children, child_labels):
    box.draw(ax, label, size=8.7, weight='bold', **cohort)

# The exclusion arrows go on first, so that each connector is drawn across the start of its arrow.
for joint, box in zip(joints, side):
    arrow(ax, [joint, box.edge('left')], red)
for upper, lower in zip(main, main[1:]):
    arrow(ax, [upper.edge('bottom'), lower.edge('top')], blue)
line(ax, [stem, fork], blue)
for child in children:
    arrow(ax, [fork, (child.x, fork[1]), child.edge('top')], blue)

ax.text(252, 539, 'Figure S1. Participant Flow Diagram', ha='center', va='top',
        fontsize=12, fontweight='bold', fontfamily='DejaVu Sans')
note = ('Note: The Leave-Behind Questionnaire (LBQ) containing the financial strain measure was administered to a random\n'
        'half-sample of HRS respondents beginning in 2006. The mortality analysis includes participants with complete\n'
        'exposure and covariate data; the cardiac analysis further excludes those with prevalent heart disease or stroke.')
ax.text(5, 40, note, ha='left', va='top', fontsize=8.4, linespacing=1.15,
        fontstyle='italic', fontfamily='DejaVu Sans')

save(fig, FIGURES_DIR, 'FigureS1_Flow_Diagram')
print("Saved Figure S1")
