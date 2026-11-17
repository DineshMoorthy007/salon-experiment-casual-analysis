# Salon Experiment & Causal Analysis – Measuring the True Impact of Business Changes

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests Passing](https://img.shields.io/badge/tests-19%20passed-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()
[![Status: Phase 1 & 2 Completed](https://img.shields.io/badge/Phase%201%20%26%202-Complete-success.svg)]()

> **Project Developed for:** Data Science with Python Analyst Internship  
> **Author & Role:** Senior Data Scientist, Python Data Engineer & Data Analytics Architect  
> **Enterprise Analyzed:** *Aura Luxe Salon & Spa Chain*  
> **Location:** Coimbatore, Tamil Nadu, India (22 Outlets across 5 Geographic Regions)  
> **Data Scope:** ~300,000 Appointment Records (2022–2024), 18,000 A/B Trial Units, 48-Month Regional Panel

---

## 📌 Synthetic Data Disclosure & Ethical Notice

> [!IMPORTANT]
> **SIMULATION DISCLOSURE:**  
> All data files, customer demographics, booking logs, employee profiles, revenues, and experimental assignments included in this repository are **100% synthetically generated** using Python simulation scripts (Seed: 42). The enterprise entity, outlet performance, and experimental results are modeled for academic, analytical, and portfolio demonstration purposes and must **not** be represented as actual proprietary corporate records of any existing salon or spa business.

---

## 🏢 Business Context & Problem Framing

*Aura Luxe Salon & Spa Chain* operates **22 retail outlets** strategically distributed across Coimbatore, Tamil Nadu. The network encompasses flagship luxury salons in high-income neighborhoods (Race Course, RS Puram), high-throughput salons in student and IT corridors (Peelamedu, Saravanampatti), and family-oriented salons in residential suburbs (Vadavalli, Saibaba Colony).

Management piloted two major strategic interventions:
1. **Prospective Randomized A/B Experiment:** An **"Automated WhatsApp Appointment Confirmation & Smart Rescheduling System"** tested against standard 24h SMS reminders across 18,000 active clients (stratified across the 22 outlets).
2. **Quasi-Experimental Regional Policy:** A **"Centralized Regional Quality & Automated Service Guarantee Policy"** rolled out across Central and North Coimbatore regions for Difference-in-Differences (DiD) econometric evaluation.

---

## 🛠️ Technology Stack

- **Core Analytics & Data Engineering:** Python 3.10+, Pandas, NumPy
- **Statistical Inference & Causal Econometrics:** Statsmodels, SciPy, Scikit-learn
- **Data Visualization & Plotting:** Matplotlib, Seaborn
- **Interactive Exploration:** Jupyter Notebook
- **Automated Verification:** Pytest

---

## 📂 Project Repository Structure

```
Salon-Experiment-Causal-Analysis/
│
├── data/
│   ├── raw/                              # Raw synthetic datasets with realistic POS defects
│   │   ├── outlets.csv                   # 22 Coimbatore salon outlet profiles
│   │   ├── customers.csv                 # 42,000 synthetic customer demographic records
│   │   ├── services.csv                  # Standardized service catalogue (24 services)
│   │   ├── stylists.csv                  # 119 stylist profiles with seniority & specialty
│   │   ├── appointments.csv              # 303,900 raw appointments (includes duplicates & noise)
│   │   ├── product_sales.csv             # 48,000 retail product transaction records
│   │   ├── experiment_data.csv           # 18,000 prospective A/B experiment records
│   │   └── historical_regional_data.csv  # 240 monthly panel records for DiD analysis
│   ├── processed/                        # Cleaned, standardized, and validated datasets
│   │   ├── outlets_clean.csv
│   │   ├── customers_clean.csv
│   │   ├── services_clean.csv
│   │   ├── stylists_clean.csv
│   │   ├── appointments_clean.csv        # Exactly 300,000 de-duplicated records
│   │   ├── product_sales_clean.csv
│   │   ├── experiment_data_clean.csv
│   │   ├── historical_regional_data_clean.csv
│   │   └── customer_features.csv         # Engineered customer behavioral & retention features
│   └── README.md                         # Data catalog, schemas, and dictionary
│
├── notebooks/
│   ├── 01_data_exploration.ipynb         # 13-Section exploratory data analysis
│   ├── 02_ab_test_analysis.ipynb         # Prospective A/B testing & hypothesis testing
│   ├── 03_difference_in_differences.ipynb# Econometric DiD & dynamic event-study regression
│   ├── 04_customer_segmentation.ipynb    # Subgroup heterogeneity across personas & outlets
│   └── 05_cuped_analysis.ipynb           # Variance reduction via pre-experiment covariates
│
├── src/
│   ├── data_generation.py                # Synthetic dataset generation engine (Seed: 42)
│   ├── data_cleaning.py                  # Deduplication, imputation, standardization & audit
│   ├── feature_engineering.py            # Sequential, customer-level, and outlet features
│   ├── statistical_inference.py          # Power analysis, SRM check, z-test, t-test, Bootstrap CI
│   ├── causal_inference.py               # DiD regression, Event-Study, CUPED, Subgroup estimation
│   └── visualization.py                  # Complete visualization suite (15 figures, 300 DPI)
│
├── reports/
│   ├── data_quality_report.md            # Phase 1: Data engineering & quality audit report
│   └── statistical_experimentation_report.md # Phase 2: Statistical inference & causal report
│
├── outputs/
│   ├── figures/                          # 15 Publication-grade figures (300 DPI)
│   │   ├── 01_appointment_status_distribution.png
│   │   ├── 02_outlet_performance_rankings.png
│   │   ├── 03_revenue_distributions.png
│   │   ├── 04_customer_segmentation.png
│   │   ├── 05_service_category_analysis.png
│   │   ├── 06_monthly_revenue_trends.png
│   │   ├── 07_noshow_and_cancellation_analysis.png
│   │   ├── 08_experiment_baseline_and_outcomes.png
│   │   ├── 09_power_analysis_sensitivity.png
│   │   ├── 10_ab_test_treatment_vs_control.png
│   │   ├── 11_ab_test_confidence_intervals_forest.png
│   │   ├── 12_did_parallel_trends_inspection.png
│   │   ├── 13_did_event_study_dynamics.png
│   │   ├── 14_segmentation_treatment_effects.png
│   │   └── 15_cuped_variance_reduction_comparison.png
│   └── tables/                           # Summary audit logs & experimental results CSVs
│       ├── cleaning_summary_audit.csv
│       ├── outlet_performance_metrics.csv
│       ├── consolidated_experimental_results.csv
│       ├── did_regression_results.csv
│       └── cuped_variance_reduction_summary.csv
│
├── tests/
│   ├── test_pipeline.py                  # Phase 1: Data engineering & pipeline tests (9 tests)
│   └── test_statistical_inference.py    # Phase 2: Statistical, DiD & CUPED tests (10 tests)
│
├── requirements.txt                      # Project package dependencies
├── README.md                             # Project overview and reproduction guide
└── .gitignore                            # Git exclusion rules
```

---

## ⚡ Step-by-Step Reproduction Guide

### 1. Environment Setup
Clone the repository and install the verified dependencies:
```bash
pip install -r requirements.txt
```

### 2. Generate Synthetic Datasets
Generate all 8 raw synthetic datasets (~300,000 appointments, 42,000 customers, 22 outlets, 119 stylists):
```bash
python src/data_generation.py
```

### 3. Execute Data Cleaning Pipeline
Run the data quality and cleaning suite to de-duplicate, impute missing values, standardize messy service names, recover misclassified no-shows, and rectify revenue anomalies:
```bash
python src/data_cleaning.py
```

### 4. Execute Feature Engineering Pipeline
Engineer appointment-level sequential metrics, customer lifetime behavioral profiles, 60-day/90-day retention windows, and outlet performance metrics:
```bash
python src/feature_engineering.py
```

### 5. Generate Phase 2 Analytical Tables
Compute and export consolidated experiment results, DiD regression summaries, and CUPED variance reduction statistics:
```bash
python src/generate_phase2_tables.py
```

### 6. Generate Complete Visualizations
Generate all 15 publication-grade analytical plots (300 DPI):
```bash
python src/visualization.py
```

### 7. Run Automated Test Suite
Execute the comprehensive 19-test automated test suite:
```bash
python -m pytest tests/ -v
```
*(Expected: 19 passed in < 4.0s)*

### 8. Run Jupyter Notebooks
Launch and inspect the analytical notebooks:
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
jupyter notebook notebooks/02_ab_test_analysis.ipynb
jupyter notebook notebooks/03_difference_in_differences.ipynb
jupyter notebook notebooks/04_customer_segmentation.ipynb
jupyter notebook notebooks/05_cuped_analysis.ipynb
```

---

## 📊 Summary of Phase 2 Analytical Deliverables

| Deliverable Area | Method / Tool | Core Finding / Metric Value |
| :--- | :--- | :--- |
| **Statistical Power** | Statsmodels `NormalIndPower` | $N = 4,218$ per arm required for 3.0 pp MDE; actual $N = 9,000$ achieves $>98\%$ power. |
| **Sample Ratio Mismatch** | $\chi^2$ Goodness-of-Fit | Observed 9,000 vs 9,000 ($\chi^2 = 0.000$, $p = 1.000$), confirming zero allocation bias. |
| **Baseline Balance** | Cohen's $d$ (SMD), Welch's $t$, $\chi^2$ | All continuous SMDs $< 0.015$ and categorical $V < 0.01$, confirming balance. |
| **Primary Metric (Retention)** | Two-Proportion Z-Test & Bootstrap | Control: 57.84%, Treatment: 63.79%, **Lift: +5.94 pp (+10.28% relative)**, $p < 0.001$, 95% CI: $[+4.52\text{ pp}, +7.37\text{ pp}]$. |
| **Secondary Metric (No-Show)** | Two-Proportion Z-Test & Bootstrap | Control: 13.11%, Treatment: 8.41%, **Reduction: -4.70 pp (-35.86% relative)**, $p < 0.001$, 95% CI: $[-5.63\text{ pp}, -3.77\text{ pp}]$. |
| **Secondary Metric (Revenue)** | Welch's $t$-Test & Bootstrap | Control: ₹1,247.50, Treatment: ₹1,355.00, **Lift: +₹107.50 (+8.62%)**, $p < 0.001$, 95% CI: $[+₹79.80, +₹135.20]$. |
| **Difference-in-Differences** | Two-Way OLS with Cluster-Robust SE | Regional policy drove **+₹175.16 revenue lift**, **+4.61 pp retention lift**, and **-3.21 pp no-show drop** ($p < 0.001$). |
| **Parallel Trends** | Event-Study Dynamic DiD | Pre-intervention dynamic leads showed no pre-trend divergence ($p \ge 0.05$). |
| **Customer Segmentation** | Subgroup Treatment Effects | Highest retention lift in Regular/Budget clients (+6.5 pp to +7.1 pp); lowest no-shows in student hubs. |
| **Variance Reduction** | CUPED ($\theta = 0.0712$, $\rho = 0.363$) | **13.17% variance reduction** in conditional spend; **13.17% equivalent sample size / duration savings**. |
| **Verification & Tests** | Pytest Verification Suite | **19 of 19 tests passed** across data engineering, stats, DiD, and CUPED. |
