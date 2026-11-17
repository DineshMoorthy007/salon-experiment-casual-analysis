# Salon Experiment & Causal Analysis – Measuring the True Impact of Business Changes

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests Passing](https://img.shields.io/badge/tests-19%20passed-brightgreen.svg)]()
[![Streamlit App](https://img.shields.io/badge/Streamlit-Interactive%20Dashboard-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()
[![Status: Submission Ready](https://img.shields.io/badge/Internship%20Submission-Phase%201%2C%202%20%26%203%20Complete-success.svg)]()

> **Project Developed for:** Data Science with Python Analyst Internship (Yuvaintern)  
> **Author & Role:** Senior Data Scientist, Python Data Engineer & Data Analytics Architect  
> **Enterprise Analyzed:** *Aura Luxe Salon & Spa Chain*  
> **Location:** Coimbatore, Tamil Nadu, India (22 Outlets across 5 Geographic Regions)  
> **Scope:** ~300,000 Appointment Records (2022–2024), 18,000 A/B Trial Units, 48-Month Regional Panel

---

## 📌 Synthetic Data Disclosure & Ethical Notice

> [!IMPORTANT]
> **SIMULATION DISCLOSURE:**  
> All data files, customer demographics, booking logs, employee profiles, revenues, and experimental assignments included in this repository are **100% synthetically generated** using Python simulation scripts (Seed: `42`). The enterprise entity, outlet performance, and experimental results are modeled for academic, analytical, and portfolio demonstration purposes and must **not** be represented as actual proprietary corporate records of any existing salon or spa business.

---

## 🏢 Business Context & Problem Statement

*Aura Luxe Salon & Spa Chain* operates **22 retail outlets** strategically distributed across Coimbatore, Tamil Nadu. The network encompasses flagship luxury salons in high-income neighborhoods (*Race Course, RS Puram*), high-throughput salons in student and IT corridors (*Peelamedu, Saravanampatti*), and family-oriented salons in residential suburbs (*Vadavalli, Saibaba Colony*).

Prior to business interventions, exploratory analysis revealed two severe operational vulnerabilities:
1. **Unnotified Appointment No-Shows (~12.1%):** Customers failing to appear for booked slots resulted in over 420 hours of unrecoverable stylist chair vacancy each week across the chain.
2. **Customer Retention Plateau (~58.2%):** Over 41% of active clients failed to return for a completed appointment within a standard 60-day observation window, dampening customer lifetime value (LTV).

To address these challenges, the chain executed two strategic initiatives:
- **Prospective Randomized A/B Experiment:** Deploying an **Automated WhatsApp Appointment Confirmation & Smart Rescheduling System** tested against standard 24-hour SMS reminders across 18,000 active clients.
- **Quasi-Experimental Regional Overhaul:** Launching a **Centralized Regional Quality & Automated Service Guarantee Policy** in Central and North Coimbatore regions, evaluated via Difference-in-Differences (DiD).

---

## 🛠️ Technology Stack

- **Core Analytics & Engineering:** Python 3.10+, Pandas, NumPy
- **Statistical Inference & Econometrics:** Statsmodels, SciPy, Scikit-learn
- **Data Visualization & Plotting:** Matplotlib, Seaborn
- **Interactive UI Dashboard:** Streamlit
- **Technical Documentation & Reporting:** Python-docx (Word `.docx`)
- **Testing & Verification:** Pytest

---

## 📂 Project Architecture

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
│   ├── visualization.py                  # Complete visualization suite (15 figures, 300 DPI)
│   ├── generate_phase2_tables.py         # Summary results tables export script
│   └── generate_docx_reports.py          # Professional DOCX report generator
│
├── reports/
│   ├── experiment_design.docx            # Mandatory Deliverable 1: 15-Section Experiment Protocol
│   ├── decision_memo.docx                # Mandatory Deliverable 3: 1-Page A4 Executive Decision Memo
│   ├── data_quality_report.md            # Supporting: Data engineering & quality audit report
│   └── statistical_experimentation_report.md # Supporting: Statistical & causal technical report
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
│   ├── test_pipeline.py                  # Data engineering & pipeline tests (9 tests)
│   └── test_statistical_inference.py    # Statistical, DiD & CUPED tests (10 tests)
│
├── app.py                                # Interactive Multi-Page Streamlit Dashboard
├── requirements.txt                      # Project package dependencies
├── README.md                             # Project overview and reproduction guide
└── .gitignore                            # Git exclusion rules
```

---

## 🔬 Core Methodologies & Statistical Frameworks

### 1. Randomized A/B Experimentation
- **Intervention:** Automated WhatsApp Bot providing 48h/24h 1-click confirmation and slot-swap rescheduling.
- **Randomization:** Customer-level, stratified across 22 outlets (9,000 Treatment vs. 9,000 Control).
- **Power Sizing:** Statsmodels `NormalIndPower` with Cohen's $h$. Required $n = 4,218$ per arm for 3.0 pp MDE at 80% power; actual $N = 18,000$ achieves $> 98\%$ power.
- **Sample Ratio Mismatch (SRM):** Evaluated via Pearson's $\chi^2$ goodness-of-fit ($\chi^2 = 0.000, p = 1.000$), ruling out traffic skew.
- **Baseline Balance:** Evaluated across Age, Pre-trial visits, Spend, and Prior no-shows using Standardized Mean Differences (all SMDs $< 0.015$).

### 2. Difference-in-Differences (DiD) Econometric Modeling
- **Model:** $Y_{rt} = \beta_0 + \beta_1(\text{Treatment}_r) + \beta_2(\text{Post}_t) + \beta_3(\text{Treatment}_r \times \text{Post}_t) + \epsilon_{rt}$.
- **Clusters:** Clustered standard errors at regional level ($G = 5$).
- **Parallel Trends Validation:** Evaluated pre-intervention leads ($t = -24$ to $t = -2$) via dynamic event-study regression with omitted reference month $t = -1$ (2022-12). Over 95% of pre-treatment leads were statistically indistinguishable from zero ($p \ge 0.05$).

### 3. CUPED Variance Reduction
- **Formulation:** $Y_{i,\text{adj}} = Y_i - \theta (X_i - \bar{X})$, where $\theta = \frac{\widehat{\text{Cov}}(Y,X)}{\widehat{\text{Var}}(X)}$.
- **Covariate:** Pre-experiment spend ($X$) measured strictly before treatment kickoff (`2024-07-01`).
- **Precision Lift:** Achieved **13.17% variance reduction** in conditional spend, shrinking standard errors by **6.81%** and yielding an equivalent **13.17% trial runtime savings** without point-estimate bias.

---

## 📈 Key Empirical Findings Summary

| Metric Dimension | Control Group | Treatment Group | Estimated Treatment Effect | Relative Lift (%) | 95% Confidence Interval | $p$-value | Statistical & Business Conclusion |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **60-Day Retention (Primary)** | 57.84% | 63.79% | **+5.94 pp** | **+10.28%** | $[+4.52\text{ pp}, +7.37\text{ pp}]$ | $2.14 \times 10^{-16}$ | **Significant.** Boosts repeat visits by ~535 clients per 9,000 active base. |
| **No-Show Rate (Secondary)** | 13.11% | 8.41% | **-4.70 pp** | **-35.86%** | $[-5.63\text{ pp}, -3.77\text{ pp}]$ | $2.68 \times 10^{-23}$ | **Significant.** Drastic cut in vacant chair time; recaptures ~423 stylist hours. |
| **Revenue / Visit (Secondary)** | ₹1,247.50 | ₹1,355.00 | **+₹107.50** | **+8.62%** | $[+₹79.80, +₹135.20]$ | $4.31 \times 10^{-14}$ | **Significant.** Driven by lower abandonment and retail reminder links. |
| **DiD Regional Revenue Lift** | ₹1,278.30 | ₹1,646.20 | **+₹175.16** | **+13.70%** | $[+₹162.40, +₹187.92]$ | $< 0.0001$ | **Significant.** Regional quality policy sustained long-term revenue gains. |
| **DiD Retention Lift** | 56.10% | 65.80% | **+4.61 pp** | **+8.22%** | $[+4.06\text{ pp}, +5.16\text{ pp}]$ | $< 0.0001$ | **Significant.** Verified parallel trends across pre-intervention panel. |

---

## 🖼️ Visualizations & Screenshots Preview

The analytical pipeline generates 15 publication-grade figures (300 DPI) in `outputs/figures/`:

| Figure Preview | Description |
| :--- | :--- |
| **01. Status Distribution** | Appointment breakdown: Completed (80.5%), Cancelled (7.4%), No-Show (12.1%). |
| **02. Outlet Performance** | Revenue and appointment volume rankings across all 22 Coimbatore outlets. |
| **06. Monthly Revenue Trends** | Time series capturing Tamil Nadu festive peaks (Diwali, Pongal) and monsoon dips. |
| **10. Treatment vs Control** | 3-Panel comparison across Retention (+5.94 pp), Revenue (+₹107.5), and No-Shows (-4.7 pp). |
| **11. Forest Plot** | Point estimates and 95% Confidence Intervals for experimental metrics. |
| **12. DiD Parallel Trends** | 48-Month trajectory confirming shared pre-intervention dynamics across regions. |
| **13. DiD Event Study** | Dynamic leads and lags demonstrating zero pre-divergence and sustained post-lift. |
| **15. CUPED Variance Shrinkage**| Distribution KDE variance reduction (-13.2%) and confidence interval tightening. |

---

## 💻 Installation & Step-by-Step Execution Guide

### 1. Environment Setup
Clone the repository and install the verified dependencies:
```bash
git clone https://github.com/<your-username>/Salon-Experiment-Causal-Analysis.git
cd Salon-Experiment-Causal-Analysis
pip install -r requirements.txt
```

### 2. Run Data Engineering & Cleaning Pipelines
```bash
# Generate 8 raw synthetic datasets (Seed: 42)
python src/data_generation.py

# Clean datasets & de-duplicate records
python src/data_cleaning.py

# Engineer sequential, customer-level & outlet metrics
python src/feature_engineering.py
```

### 3. Generate Analytical Tables & Visualizations
```bash
# Generate Phase 2 summary tables
python src/generate_phase2_tables.py

# Generate all 15 publication-grade charts (300 DPI)
python src/visualization.py

# Generate Word (.docx) deliverables (Experiment Design & Decision Memo)
python src/generate_docx_reports.py
```

### 4. Execute Automated Verification Test Suite
Run the 19 automated tests validating data integrity, statistical formulas, DiD regression, and CUPED variance reduction:
```bash
python -m pytest tests/ -v
```
*(Expected: 19 passed in < 4.0s)*

### 5. Launch Interactive Streamlit Dashboard
```bash
streamlit run app.py
```
*The interactive dashboard will launch at `http://localhost:8501`.*

### 6. Inspect Jupyter Analysis Notebooks
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
jupyter notebook notebooks/02_ab_test_analysis.ipynb
jupyter notebook notebooks/03_difference_in_differences.ipynb
jupyter notebook notebooks/04_customer_segmentation.ipynb
jupyter notebook notebooks/05_cuped_analysis.ipynb
```

---

## 🚀 Interactive Streamlit Dashboard (`app.py`)

The project includes an optional interactive analytical web application with 5 dynamic pages:
- **Page 1: Business Overview:** High-level KPI cards, revenue in ₹ Crores, regional filters, and monthly revenue trends.
- **Page 2: A/B Experiment Analysis:** Control vs. Treatment comparisons, SRM diagnostic badge, and forest plots.
- **Page 3: Historical Causal Analysis:** DiD regression outcomes, parallel trends inspection, and event-study dynamics.
- **Page 4: Customer Segmentation:** Interactive subgroup exploration across Customer Personas, Outlet Tiers, and Regions.
- **Page 5: CUPED Variance Reduction:** Variance reduction statistics, distribution KDE plots, and an interactive runtime savings calculator.

---

## ⚠️ Limitations & Future Improvements

1. **Synthetic Nature:** All records are simulated via parametric distributions (Gamma, Pareto, Negative Binomial) calibrated to Coimbatore demographics. Real-world consumer behaviors may feature unforeseen micro-shocks.
2. **Anonymous Walk-In Barrier:** Walk-ins without registered mobile numbers cannot receive digital WhatsApp messages. In-salon front-counter conversion protocols are required to onboard these guests.
3. **Small Regional Cluster Count:** Regional DiD clustered standard errors are based on $G = 5$ administrative regions. While robust, asymptotic cluster assumptions should be monitored alongside outlet-level microdata.
4. **Future Extensions:** Implement synthetic control methods (SCM) for regional policy evaluation and deploy multi-armed bandit (MAB) algorithms to dynamically optimize message delivery timing.

---

## 📤 GitHub Push Instructions

To push this submission package to your personal GitHub repository:

```bash
# 1. Initialize Git repository
git init

# 2. Add files (honoring .gitignore)
git add .

# 3. Commit deliverables
git commit -m "feat: complete Phase 1, 2, and 3 submission package for Salon Experiment & Causal Analysis"

# 4. Link to GitHub remote
git remote add origin https://github.com/<your-username>/Salon-Experiment-Causal-Analysis.git

# 5. Push to main branch
git branch -M main
git push -u origin main
```
