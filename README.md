# Salon Experiment and Causal Analysis: Measuring the True Impact of Business Changes

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Automated Tests](https://img.shields.io/badge/Pytest-19%20Passed-brightgreen.svg)](tests/)
[![Dashboard](https://img.shields.io/badge/Streamlit-Analytical%20Dashboard-orange.svg)](app.py)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)
[![Submission Status](https://img.shields.io/badge/Internship%20Submission-Phase%201%2C%202%20%26%203%20Complete-green.svg)]()

> **Project Developed for:** Data Science with Python Analyst Internship (Yuvaintern)  
> **Author & Role:** Senior Data Scientist, Python Data Engineer, and Experimentation Architect  
> **Enterprise Analyzed:** Aura Luxe Salon and Spa Chain  
> **Geographic Scope:** Coimbatore, Tamil Nadu, India (22 Outlets across 5 Geographic Regions)  
> **Operational Scope:** ~300,000 Appointment Records (2022-2024), 18,000 Randomized Trial Units, 48-Month Regional Panel  

---

## 1. Executive Summary

This repository presents an end-to-end data science, experimentation, and econometric evaluation framework developed for a multi-outlet salon and spa chain operating in Coimbatore, Tamil Nadu. The project establishes a production-grade data engineering pipeline, formulates prospective randomized controlled trials (A/B testing), implements quasi-experimental Difference-in-Differences (DiD) panel econometric models, and applies Controlled-Experiment Using Pre-Experiment Data (CUPED) algorithmic variance reduction.

The business evaluation measures the causal impact of two enterprise initiatives:
1. **Prospective Randomized A/B Experiment:** Deployment of an Automated WhatsApp Appointment Confirmation and Rescheduling System tested against standard 24-hour SMS notifications across 18,000 active clients.
2. **Historical Regional Policy Overhaul:** Implementation of a Centralized Regional Service Quality and Guarantee Policy across Central and North Coimbatore outlets, evaluated via a 48-month panel dataset (2021-2024).

---

## 2. Synthetic Data Disclosure and Ethical Notice

All records, client demographics, appointment transactions, stylist profiles, retail product purchases, and experimental group assignments in this repository are **100% synthetically simulated** in Python (Random Seed: `42`). 

Parametric distributions (Poisson, Negative Binomial, Gamma, and Pareto) were calibrated to mirror operational patterns of real-world retail salon chains across urban and suburban corridors in Coimbatore. The datasets and business outcomes are modeled strictly for educational, analytical, and portfolio assessment purposes and must not be represented as real proprietary company data.

---

## 3. Business Problem and Context

Aura Luxe Salon operates **22 retail outlets** across five administrative territories in Coimbatore:
- **Central:** RS Puram, Race Course (High-revenue flagship luxury lounges)
- **North:** Gandhipuram, Ram Nagar, Saibaba Colony, Ganapathy, Saravanampatti (High-traffic commercial and IT corridors)
- **South:** Kuniyamuthur, Sundarapuram, Podanur, Kovaipudur (Residential family hubs)
- **East:** Peelamedu, Singanallur, Ramanathapuram, Ondipudur, Sulur (Student and educational belt)
- **West:** Vadavalli, Thondamuthur, Perur, Telungupalayam, Selvapuram (Suburban family salons)

### Core Operational Bottlenecks
Prior to intervention, operational diagnostics revealed two primary threats to profitability:
- **Excessive Unnotified No-Shows (12.1% baseline):** Clients failing to attend booked slots without advance cancellation caused more than 420 hours of unrecoverable stylist chair vacancy each week across the network.
- **Customer Retention Plateau (58.2% baseline):** Over 41% of active clients failed to return for a repeat service within a standard 60-day observation window, driving up customer acquisition costs (CAC) and depressing customer lifetime value (LTV).

---

## 4. Methodological Frameworks and Statistical Rigor

```
+-------------------------------------------------------------------------------------------------+
|                                    METHODOLOGICAL ARCHITECTURE                                  |
+--------------------------------+-------------------------------+--------------------------------+
|      PROSPECTIVE A/B TEST      |      HISTORICAL DID PANEL     |        CUPED ACCELERATION      |
+--------------------------------+-------------------------------+--------------------------------+
| - Unit: Individual Client      | - Unit: Outlet-Month Panel    | - Covariate: Pre-Trial Spend   |
| - Stratification: Outlet ID    | - Treatment: Central & North  | - Optimal Theta: Cov(Y,X)/Var  |
| - N: 18,000 (1:1 Allocation)   | - Pre-Period: 24 Months       | - Variance Reduction: 13.17%   |
| - Primary: 60-Day Retention    | - Post-Period: 24 Months      | - Duration Savings: 13.17%     |
| - Secondary: Spend & No-Show   | - Standard Errors: Regional   | - Zero Point-Estimate Bias     |
| - Diagnostics: SRM & SMD       | - Dynamic Event Study Leads   | - Analytical & Bootstrap CIs   |
+--------------------------------+-------------------------------+--------------------------------+
```

### A. Statistical Power Analysis and Sizing
Power calculations were implemented using Statsmodels `NormalIndPower` with Cohen's $h$ metric transformation:
- **Baseline Retention ($p_1$):** 57.5%
- **Minimum Detectable Effect (MDE):** 3.0 percentage points absolute lift ($p_2 = 60.5\%$)
- **Significance Level ($\alpha$):** 0.05 (two-tailed)
- **Target Statistical Power ($1 - \beta$):** 0.80
- **Sample Allocation Ratio ($k$):** 1:1
- **Required Minimum Sample Size:** $n = 4,218$ per arm ($N_{\text{total}} = 8,436$)
- **Enrolled Trial Cohort:** $N = 18,000$ active clients (9,000 Treatment vs. 9,000 Control), achieving $> 98\%$ statistical power.

### B. Sample Ratio Mismatch (SRM) Diagnostic
To verify that the assignment mechanism remained uncompromised by technical bugs or upstream dropouts:
- **Observed Counts:** Control = 9,000; Treatment = 9,000
- **Expected Counts:** Control = 9,000; Treatment = 9,000
- **Pearson Chi-Square ($\chi^2$):** $0.0000$ ($p = 1.0000$)
- **Conclusion:** No Sample Ratio Mismatch detected. Allocation integrity is verified.

### C. Randomization Baseline Balance
Standardized Mean Differences (SMD) were calculated across all observed pre-experiment covariates:
- Age: $\text{SMD} = +0.004$
- Pre-experiment visit frequency: $\text{SMD} = -0.008$
- Pre-experiment total expenditure: $\text{SMD} = +0.012$
- Pre-experiment no-show frequency: $\text{SMD} = -0.006$
All SMDs fell strictly below the conservative threshold of $0.05$, confirming baseline covariate balance.

### D. Difference-in-Differences (DiD) Econometric Model
The impact of the regional quality policy was estimated using a two-way fixed effects panel regression:

$$Y_{rt} = \beta_0 + \beta_1(\text{Treatment}_r) + \beta_2(\text{Post}_t) + \beta_3(\text{Treatment}_r \times \text{Post}_t) + \epsilon_{rt}$$

Where:
- $\text{Treatment}_r = 1$ for outlets in Central and North regions, $0$ otherwise.
- $\text{Post}_t = 1$ for calendar months from January 2023 onward ($t \ge 0$), $0$ otherwise.
- $\beta_3$ captures the Average Treatment Effect on the Treated (ATT).
- Standard errors were clustered at the administrative regional level ($G = 5$).
- Dynamic event-study leads and lags ($t = -24$ to $t = +23$) validated the parallel trends assumption; pre-treatment leads were statistically indistinguishable from zero ($p \ge 0.05$).

### E. CUPED Algorithmic Variance Reduction
To accelerate trial velocity and increase statistical precision without introducing estimation bias:

$$Y_{i, \text{adj}} = Y_i - \hat{\theta} (X_i - \bar{X}) \quad \text{where} \quad \hat{\theta} = \frac{\widehat{\text{Cov}}(Y, X)}{\widehat{\text{Var}}(X)}$$

- **Covariate ($X$):** Total spending during the pre-experiment period.
- **Observed Correlation ($\rho_{XY}$):** $0.3629$ among active transacting clients.
- **Empirical Variance Reduction:** $13.17\%$.
- **Standard Error Shrinkage:** $7.03\%$ (from Rs. 6.10 down to Rs. 5.67).
- **Equivalent Runtime Savings:** $13.17\%$ reduction in sample size or test duration required.

---

## 5. Consolidated Experimental and Causal Results

| Metric Dimension | Counterfactual Baseline | Treatment Group | Absolute Effect (Lift) | Relative Change | 95% Confidence Interval | p-value | Statistical Conclusion | Practical Business Interpretation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **60-Day Retention (Primary)** | 64.08% | 69.38% | **+5.30 pp** | **+8.27%** | [+3.92 pp, +6.68 pp] | 4.51e-14 | Statistically Significant (p < 0.001) | Substantial repeat visit lift; drives ~477 additional retained clients per 9,000 cohort. |
| **No-Show Rate (Secondary)** | 9.31% | 4.40% | **-4.91 pp** | **-52.74%** | [-5.64 pp, -4.19 pp] | 7.51e-39 | Statistically Significant (p < 0.001) | Cut unnotified no-shows by more than half, recovering stylist utilization hours across salons. |
| **Revenue per Visit (Secondary)** | Rs. 800.16 | Rs. 945.43 | **+Rs. 145.27** | **+18.15%** | [+Rs. 125.64, +Rs. 164.89] | 1.95e-47 | Statistically Significant (p < 0.001) | Incremental spend lift driven by automated appointment completion and re-engagement links. |
| **DiD Regional Revenue / Visit** | Rs. 1409.87 | Rs. 1585.03 | **+Rs. 175.16** | **+12.42%** | [+Rs. 161.80, +Rs. 188.52] | < 1.00e-100 | Statistically Significant (p < 0.001) | Regional quality overhaul produced sustained revenue gains across Central and North outlets. |
| **DiD 60-Day Retention Rate** | 56.19% | 60.80% | **+4.61 pp** | **+8.20%** | [+4.21 pp, +5.02 pp] | < 1.00e-100 | Statistically Significant (p < 0.001) | Long-term customer stickiness sustained after regional operational restructuring. |
| **DiD No-Show Rate** | 11.85% | 8.49% | **-3.36 pp** | **-28.35%** | [-3.52 pp, -3.19 pp] | < 1.00e-100 | Statistically Significant (p < 0.001) | Parallel pre-trends verified; no-shows permanently suppressed in policy regions. |

---

## 6. Repository Architecture

```
salon-experiment-causal-analysis/
│
├── data/
│   ├── raw/                              # Synthetically generated POS transaction records
│   │   ├── outlets.csv                   # 22 Coimbatore salon outlet profiles
│   │   ├── customers.csv                 # 42,000 synthetic customer demographic records
│   │   ├── services.csv                  # Standardized service catalogue (24 services)
│   │   ├── stylists.csv                  # 119 stylist profiles with seniority & specialty
│   │   ├── appointments.csv              # 303,900 appointments (includes duplicates & noise)
│   │   ├── product_sales.csv             # 48,000 retail product sales transactions
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
│   │   └── customer_features.csv         # Customer behavioral and retention features
│   └── README.md                         # Data catalog, schemas, and metadata dictionary
│
├── notebooks/
│   ├── 01_data_exploration.ipynb         # Exploratory data analysis and POS diagnostics
│   ├── 02_ab_test_analysis.ipynb         # Prospective A/B testing and hypothesis evaluation
│   ├── 03_difference_in_differences.ipynb# Econometric DiD panel & event-study regressions
│   ├── 04_customer_segmentation.ipynb    # Subgroup treatment effect heterogeneity
│   └── 05_cuped_analysis.ipynb           # Algorithmic variance reduction implementation
│
├── src/
│   ├── data_generation.py                # Synthetic dataset generation engine (Seed: 42)
│   ├── data_cleaning.py                  # Deduplication, imputation, standardization & validation
│   ├── feature_engineering.py            # Sequential, client-level, and outlet metrics
│   ├── statistical_inference.py          # Power analysis, SRM check, z-test, t-test, Bootstrap
│   ├── causal_inference.py               # DiD regression, Event Study, CUPED, Subgroup estimation
│   ├── visualization.py                  # Full visualization suite (15 figures, 300 DPI)
│   ├── generate_phase2_tables.py         # Tabular metrics generation script
│   └── generate_docx_reports.py          # Professional DOCX report generator
│
├── reports/
│   ├── experiment_design.docx            # Mandatory Deliverable 1: 15-Section Experiment Protocol
│   ├── decision_memo.docx                # Mandatory Deliverable 3: 1-Page A4 Executive Decision Memo
│   ├── data_quality_report.md            # Supporting: Data engineering and audit documentation
│   └── statistical_experimentation_report.md # Supporting: Statistical and causal analysis report
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
│   └── tables/                           # Summary results and regression tables (CSVs)
│       ├── cleaning_summary_audit.csv
│       ├── outlet_performance_metrics.csv
│       ├── consolidated_experimental_results.csv
│       ├── did_regression_results.csv
│       └── cuped_variance_reduction_summary.csv
│
├── tests/
│   ├── test_pipeline.py                  # Data engineering and pipeline verification (9 tests)
│   └── test_statistical_inference.py    # Statistical inference, DiD, and CUPED tests (10 tests)
│
├── app.py                                # Interactive Streamlit analytical dashboard
├── requirements.txt                      # Project package dependencies
├── README.md                             # Project overview and reproduction guide
└── .gitignore                            # Git exclusion rules
```

---

## 7. Installation and Execution Guide

### Step 1: Environment Setup
Clone the repository and install dependencies:
```bash
git clone https://github.com/DineshMoorthy007/salon-experiment-causal-analysis.git
cd salon-experiment-causal-analysis
pip install -r requirements.txt
```

### Step 2: Run Data Engineering and Cleaning Pipelines
Execute modular scripts sequentially:
```bash
# Generate raw synthetic datasets (Seed: 42)
python src/data_generation.py

# Clean datasets, resolve inconsistencies, and de-duplicate records
python src/data_cleaning.py

# Engineer sequential, customer-level, and outlet features
python src/feature_engineering.py
```

### Step 3: Generate Summary Tables, Visualizations, and Reports
```bash
# Export summary metrics tables
python src/generate_phase2_tables.py

# Generate all 15 publication-grade figures (300 DPI)
python src/visualization.py

# Compile official DOCX report deliverables
python src/generate_docx_reports.py
```

### Step 4: Run Automated Verification Tests
Execute the 19 automated Pytest test suites:
```bash
python -m pytest tests/ -v
```
*Expected output: `19 passed` in under 10 seconds.*

### Step 5: Launch the Interactive Streamlit Dashboard
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser. The dashboard gracefully adapts to available plotting libraries and includes robust fallbacks if optional dependencies are not present.

---

## 8. Interactive Analytical Dashboard (`app.py`)

The Streamlit web application includes five dedicated analytical modules:
1. **Module 1: Business Overview:** Network-wide KPIs (Appointments, Outlets, Revenue in Rs. Crores, 60-Day Retention, No-Show Rate), regional filtering, monthly revenue seasonality (Diwali and Pongal surges), and appointment completion donuts.
2. **Module 2: A/B Experiment Analysis:** Core metric delta cards, Sample Ratio Mismatch (SRM) verification, consolidated hypothesis testing tables with scientific notation, treatment bar comparisons, and forest plots with 95% confidence intervals.
3. **Module 3: Historical Causal Analysis (DiD):** Single-line econometric specification, regression results table with cluster-robust standard errors, parallel trends inspection, and dynamic event-study leads and lags.
4. **Module 4: Customer Segmentation:** Interactive subgroup exploration across Customer Personas, Outlet Tiers, and Geographic Regions with treatment lift bar charts.
5. **Module 5: CUPED Variance Reduction:** Single-line covariance adjustment formulas, variance reduction tables, distribution variance shrinkage figures, and an interactive trial duration acceleration calculator.

---

## 9. Limitations and Methodological Roadmap

1. **Simulation Model Assumptions:** All data is simulated using parameterized stochastic processes. While distributions were calibrated to retail salon benchmarks in Tamil Nadu, real-world customer behaviors may exhibit unmodeled structural macroeconomic shifts.
2. **Anonymous Walk-In Client Barrier:** The WhatsApp automated reminder and rescheduling system relies on verified mobile phone numbers. In-salon front-desk onboarding protocols are necessary to convert anonymous walk-in clients into enrolled digital accounts.
3. **Small Regional Cluster Count:** Regional Difference-in-Differences standard errors are clustered across $G = 5$ territories. While cluster-robust variance estimators are employed, wild cluster bootstrap procedures can provide additional small-cluster finite-sample asymptotic guarantees.
4. **Future Methodological Extensions:** Future iterations will explore Synthetic Control Methods (SCM) for individual outlet policy evaluations and Multi-Armed Bandit (MAB) reinforcement algorithms to dynamically optimize reminder send times.
