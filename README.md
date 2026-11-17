# Salon Experiment & Causal Analysis – Measuring the True Impact of Business Changes

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests Passing](https://img.shields.io/badge/tests-9%20passed-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()
[![Status: Phase 1 Completed](https://img.shields.io/badge/Phase%201-Project%20Foundation%20%26%20Data%20Engineering-success.svg)]()

> **Project Developed for:** Data Science with Python Analyst Internship  
> **Author & Role:** Senior Data Scientist, Python Data Engineer & Data Analytics Architect  
> **Enterprise Analyzed:** *Aura Luxe Salon & Spa Chain*  
> **Location:** Coimbatore, Tamil Nadu, India (22 Outlets across 5 Geographic Regions)  
> **Data Scope:** ~300,000 Appointment Records (2022–2024)

---

## 📌 Synthetic Data Disclosure & Ethical Notice

> [!IMPORTANT]
> **SIMULATION DISCLOSURE:**  
> All data files, customer demographics, booking logs, employee profiles, revenues, and experimental assignments included in this repository are **100% synthetically generated** using Python simulation scripts. The enterprise entity, outlet performance, and experimental results are modeled for academic, analytical, and portfolio demonstration purposes and must **not** be represented as actual proprietary corporate records of any existing salon or spa business.

---

## 🏢 Business Context & Problem Framing

*Aura Luxe Salon & Spa Chain* is a premier regional lifestyle brand operating **22 retail outlets** strategically distributed across Coimbatore, Tamil Nadu. The network encompasses flagship luxury salons in high-income neighborhoods (Race Course, RS Puram), high-throughput salons in student and IT corridors (Peelamedu, Saravanampatti), and family-oriented salons in residential suburbs (Vadavalli, Saibaba Colony).

Over a 3-year period (2022–2024), the business observed two major operational bottlenecks:
1. **Elevated Appointment Abandonment:** An overall unnotified no-show rate of ~12.1% across digital and phone booking channels, resulting in underutilized stylist capacity and lost revenue.
2. **Customer Retention Plateaus:** 60-day repeat visit retention hovered around ~58.2%, with high churn among first-time visitors.

To resolve these challenges, management piloted:
- A prospective randomized A/B trial (**"Intelligent WhatsApp Confirmation & Smart Rescheduling System"**) across 18,000 active clients.
- A regional policy intervention (**"Centralized Regional Quality & Automated Service Guarantee"**) rolled out across Central and North Coimbatore regions for Difference-in-Differences (DiD) causal evaluation.

**Phase 1** builds the enterprise data foundation, raw-to-processed engineering pipelines, cleaning and de-duplication rules, behavioral feature engineering, and exploratory analytics.

---

## 🛠️ Technology Stack

- **Core Analytics & Data Engineering:** Python 3.10+, Pandas, NumPy
- **Statistical Modeling & Causal Foundation:** SciPy, Statsmodels, Scikit-learn
- **Data Visualization & Plotting:** Matplotlib, Seaborn
- **Interactive Exploration:** Jupyter Notebook
- **Testing & Verification:** Pytest

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
│   └── 01_data_exploration.ipynb         # Complete 13-section exploratory analysis notebook
│
├── src/
│   ├── data_generation.py                # Synthetic dataset generation engine (Seed: 42)
│   ├── data_cleaning.py                  # Deduplication, imputation, standardization & audit
│   ├── feature_engineering.py            # Sequential, customer-level, and outlet features
│   └── visualization.py                  # Publication-quality plotting module (300 DPI)
│
├── reports/
│   └── data_quality_report.md            # Comprehensive data engineering & quality report
│
├── outputs/
│   ├── figures/                          # 8 High-resolution analytical figures
│   │   ├── 01_appointment_status_distribution.png
│   │   ├── 02_outlet_performance_rankings.png
│   │   ├── 03_revenue_distributions.png
│   │   ├── 04_customer_segmentation.png
│   │   ├── 05_service_category_analysis.png
│   │   ├── 06_monthly_revenue_trends.png
│   │   ├── 07_noshow_and_cancellation_analysis.png
│   │   └── 08_experiment_baseline_and_outcomes.png
│   └── tables/                           # Summary audit tables and outlet performance CSVs
│       ├── cleaning_summary_audit.csv
│       └── outlet_performance_metrics.csv
│
├── tests/
│   └── test_pipeline.py                  # Pytest verification suite (9 comprehensive tests)
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
*Outputs saved to `data/raw/`.*

### 3. Execute Data Cleaning Pipeline
Run the data quality and cleaning suite to de-duplicate, impute missing values, standardize messy service names, recover misclassified no-shows, and rectify revenue anomalies:
```bash
python src/data_cleaning.py
```
*Outputs saved to `data/processed/` and audit log saved to `outputs/tables/cleaning_summary_audit.csv`.*

### 4. Execute Feature Engineering Pipeline
Engineer appointment-level sequential metrics, customer lifetime behavioral profiles, 60-day/90-day retention windows, and outlet performance metrics:
```bash
python src/feature_engineering.py
```
*Outputs saved to `data/processed/customer_features.csv` and `outputs/tables/outlet_performance_metrics.csv`.*

### 5. Generate Visualizations
Generate all 8 publication-grade analytical plots (300 DPI):
```bash
python src/visualization.py
```
*Figures saved to `outputs/figures/`.*

### 6. Run Jupyter Notebook
Open and explore the comprehensive 13-section analysis notebook:
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

### 7. Run Automated Tests
Validate pipeline integrity, foreign keys, de-duplication, and non-negative revenue constraints:
```bash
python -m pytest tests/ -v
```
*(Expected: 9 passed in < 5.0s)*

---

## 📊 Summary of Phase 1 Deliverables

| Deliverable Area | Scope & Outcome |
| :--- | :--- |
| **Synthetic Generation** | 8 datasets generated with realistic POS defects and Tamil Nadu festive seasonality (Diwali, Pongal). |
| **Data Cleaning** | Exactly 3,900 duplicate appointments removed; 7,532 misclassified no-shows recovered; 45 revenue glitches rectified; walk-ins standardized with guest tokens. |
| **Feature Engineering** | 60-day & 90-day retention windows computed; lifetime spend & visit cadence built; zero look-ahead data leakage in A/B baseline features. |
| **Exploratory Analytics** | Complete 13-section Jupyter notebook executed with inline tables and figures. |
| **Quality Reporting** | Comprehensive Markdown report created at `reports/data_quality_report.md`. |
| **Verification** | 100% test passage across 9 automated pytest test cases. |

---

## 🚀 Phase 2 Roadmap: Causal Inference & Econometric Analysis
- **A/B Testing:** Two-sample proportion tests, Welch's t-test, and Mann-Whitney U tests on retention lift and no-show reductions.
- **Difference-in-Differences (DiD):** Two-way fixed effects regression estimating regional policy impact across Coimbatore regions.
- **Propensity Score Matching (PSM):** Estimating causal treatment effects under observational confounding.
