# ML Model Comparison for Stock Index Performance Classification

Data Science Institute - Cohort 7 - Team ML 16 Project

## Short description
This project compares several machine learning models to classify short-term stock index price trends (e.g., whether the next day's price moves up or down) using historical open/close price data. The goal is to identify which model(s) perform best for the selected index(es) and provide recommendations for practical use.

## Table of contents
- [Project objective](#project-objective)
- [Dataset](#dataset)
- [Scope](#scope)
- [Methodology](#methodology)
- [Models compared](#models-compared)
- [Evaluation metrics](#evaluation-metrics)
- [Repository structure](#repository-structure)
- [Getting started](#getting-started)
- [Members](#members--roles)
- [Results & recommendations](#results--recommendations)
- [License](#license)
- [Contact](#contact)

## Project objective
- Predict short-term (next-day or consecutive-day) stock index direction from historical open/close prices.
- Compare multiple models and recommend the best-performing model(s) with their caveats.
- Produce reproducible experiments and clear documentation for model selection.

## Dataset
- Dataset used: "Stock Exchange Data" (https://www.kaggle.com/datasets/mattiuzc/stock-exchange-data).
- Raw data is included in this repo.

## Scope
### Realistic scenario
- Focus on one index to determine which ML model best forecasts consecutive-day price trends using only open/close features.

### Optimistic scenario
- Extend experiments to all 14 indices and compare performance differences across indices.

## Methodology
- Data ingestion and cleaning: handle missing values, create standardized fields (per index).
- Feature engineering examples: percentage change, rolling means, day-of-week, lag features (previous open/close returns), volatility measures.
- Train / validation / test: use time-series-aware splitting (no random shuffling); consider walk-forward validation for robust estimates.
- Reproducibility: set random seeds, record package versions, log experiments.

## Models compared
- Baseline: Logistic Regression.
- Advanced models: XGBoost, Decision Tree, KNN.
- Rationale: baseline vs tree-based vs distance-based methods to cover a range of model families.

## Evaluation metrics
- For classification: accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix.
- For probabilistic models: calibration and log loss can be informative.
- Reports: per-index metrics and aggregated comparisons (tables and plots).

## Repository structure
- data/                 # raw and processed datasets
- notebooks/            # exploratory data analysis and experiments (one notebook per index)
- src/                  # scripts for processing, training, evaluation
- models/               # saved model artifacts
- results/              # evaluation outputs, figures, and tables
 - src/                  # scripts and notebooks
 - data/                 # raw and processed datasets
 - model/                # saved model artifacts 
 - reports/              # evaluation outputs, figures, and tables
 - README.md

## Getting started
1. Clone the repository:
   - git clone https://github.com/JesusSolisR/Stock-Index-ML-Model-Comparison.git

2. Create environment and install dependencies:
   - Python 3.11 recommended.
   - Use environment.yml for environment setup.

3. Data:
   - indexInfo.csv - exploratory information about the RAW dataset.
   - indexData.csv - RAW dataset
   - indexProcessed.csv - Processed dataset

4. Quick Start Guide:
   - Create and activate the Python environment from `environment.yml` (example using conda):
      - conda env create -f environment.yml
      - conda activate <env-name>

## Members & roles
### Jesus Solis 
- GitHub: https://github.com/JesusSolisR
### Abeer Khetrapal
- GitHub: https://github.com/abeerkhe
### Mark Kuriy 
- GitHub: https://github.com/xqzv
### Mingxia Zeng
- GitHub: https://github.com/luoyaqifei  
- 🎧 [Audio Reflection (mingxia-reflection.m4a)](/reflections/mingxia-reflection.m4a)

## Results & recommendations
- PLACEHOLDER (TBD): summarize final comparisons here (best model(s), sample metrics, limitations, next steps).

## Contact
For questions contact the team members listed above.
