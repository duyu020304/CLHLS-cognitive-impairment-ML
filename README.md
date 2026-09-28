# CLHLS cognitive impairment analysis code

This repository contains the analysis scripts used for the manuscript on explainable stacked-ensemble prediction of concurrent cognitive impairment among older adults with anxiety symptoms using CLHLS 2018 data.

## Script order

1. `01_data_preparation.R`
   - Constructs the 70 candidate predictors.
   - Applies the participant-selection criteria.
   - Creates the final analysis dataset and quality-control outputs.

2. `02_table1_generation.R`
   - Generates participant characteristics for Table 1.

3. `03_machine_learning_analysis.R`
   - Performs the development/test split.
   - Conducts Elastic Net and RF-RFE feature selection.
   - Trains the nine base models and the stacked ensemble.
   - Evaluates held-out performance.
   - Produces SHAP-based model interpretation and related outputs.

4. `04_indirect_pathway_analysis.R`
   - Performs the nine theory-informed exploratory indirect-pathway analyses.
   - Applies bootstrap estimation and FDR adjustment.

## Data

The original CLHLS data are third-party data and are not included in this repository. Researchers should obtain the required CLHLS 2018 data from the official Peking University Open Research Data Platform under the applicable data-use requirements.

Before running the scripts, set `data_dir` in each script to the local project directory containing the required CLHLS data and generated intermediate files.

## Public-sharing edits

The public-sharing version was prepared from the authors' final analysis code. Chinese code comments were translated into English, and the local absolute project path was replaced with `path/to/project`. The analytical logic was not intentionally changed.

Some output filenames and console messages may still use the original Chinese text because they are part of the existing workflow rather than code comments. They can be translated separately if a fully English public repository is desired.

## Recommended repository additions before publication

- Add the exact R version used for the final analysis.
- Record package versions (for example with `sessionInfo()` or `renv`).
- Add an open-source license appropriate for the project.
- Archive a release in Zenodo if a persistent DOI is desired.
