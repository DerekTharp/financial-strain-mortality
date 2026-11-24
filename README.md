# Financial strain and all-cause mortality by age

Replication code for:

> Tharp D. Does the association between financial strain and all-cause mortality vary by age? A prospective cohort study of US adults aged 50 years and older in the Health and Retirement Study, 2006–2022. *BMJ Public Health* (forthcoming).

The code builds the analytic sample from the Health and Retirement Study (HRS) files, then produces the figures and the estimates and counts reported in the article and its supplement.

## Software

- Python 3.14 with the packages pinned in `requirements.txt`
- StataNow 19.5 MP for the two `.do` scripts

## Data

The data are not included. HRS distributes them to registered users at https://hrsdata.isr.umich.edu/. The article used these releases:

| File | Release | Place it at |
|---|---|---|
| RAND HRS Longitudinal File 2022, Stata format | V1, May 2025 (listed with the [RAND HRS archived data products](https://hrsdata.isr.umich.edu/data-products/rand-hrs-archived-data-products)) | `randhrs1992_2022v1_STATA/randhrs1992_2022v1.dta` |
| HRS Cross-Wave Tracker File 2022, Stata format | Final V1.0, December 2025 (https://doi.org/10.7826/OZGE2932) | `trk2022v1/trk2022tr_r.dta` |
| HRS 2006 Biomarker Data, SPSS format | Final V1.0, April 2013 (https://hrsdata.isr.umich.edu/data-products/2006-biomarker-data) | `Special Access Data/biomkr06/biomk06bl_r.sav` |

The first two are public files. The biomarker file is sensitive health data and needs a separate data use agreement with HRS. Its product page has the order form. Only the exploratory biomarker tables (S12 to S14) use it. Without it, `10_biomarker_analysis.py` writes nothing and the other scripts still run.

HRS has since released later versions of the RAND file and the Tracker. Use the listed releases. If Tracker Final V1.0 is no longer offered after sign-in, ask the HRS help desk (hrsquestions@umich.edu) for it.

## Running

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_all.py
```

`run_all.py` runs the scripts in the order of the table below and stops if one fails. It takes the Stata executable from the `STATA_BIN` environment variable, then from `stata-mp` on the PATH, then from the default macOS location of StataNow. A full run takes four to five minutes.

The estimates and counts are written to `output/tables/` as CSV files (the Fine-Gray estimates as a text file), the figures to `output/figures/`, and the two Stata logs to the top folder.

## Scripts

`model_specs.py` holds the covariate lists that the Python scripts share. `diagram_geometry.py` holds the box and arrow functions that draw figures S1 and S2.

| Script | Produces |
|---|---|
| `01_primary_analysis.py` | Analytic sample; table 1; unweighted estimates in table 2, figure 2 and tables S4 (with the age-interaction test in its footnote), S10 and S11; estimates behind the unweighted E-values in table S19; dose-response estimates; cohort and death-dating counts quoted in the text and in the footnote to table S3 |
| `01b_export_survey_data.py` | Analytic sample with the survey design variables |
| `03_secondary_analysis.py` | Table S1 (panel A); item distribution in table S2; counts for figure S1; data file for script 07 |
| `04_flow_diagram.py` | Figure S1 |
| `05_sensitivity_diagnostics.py` | Table S8; unweighted column of table S15; table S1 (panel B) |
| `06_sensitivity_clustering.py` | Table S9; household clustering in table S10 |
| `07_competing_risks_heart_disease.do` | Fine-Gray estimates in table S4 |
| `10_biomarker_analysis.py` | Tables S12 to S14 |
| `11_survey_prep.py` | Matched estimates in table 2 and tables S17 and S18; weighting diagnostics in table S7; unweighted E-values in table S19; attrition figures in the footnote to table S5; analysis file for script 12 |
| `12_survey_models.do` | Survey-weighted estimates in table 2 and tables S3 to S7, S11, S15, S16, S19 and S20; unweighted lagged-covariate models in table S5; the attenuation from Model 3 to Model 3b quoted in the Discussion (printed in the log) |
| `02_figures.py` | Figures 1 and 2; figures S2 and S3 |

## License

MIT (see `LICENSE`).
