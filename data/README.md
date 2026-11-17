# Salon Experiment & Causal Analysis - Data Directory

## Overview & Synthetic Data Disclosure

> **IMPORTANT ETHICAL NOTICE:**
> All datasets contained within `data/raw/` and `data/processed/` are **100% synthetically generated** for educational, causal inference experimentation, and data engineering demonstration purposes.
> The business entity ("Aura Luxe Salon & Spa Chain"), its 22 outlets across Coimbatore, Tamil Nadu, customer records, appointment bookings, revenue figures, and experimental treatment interventions are simulated and must **not** be represented as proprietary real company data.

---

## Directory Structure

```
data/
├── raw/                          # Unaltered synthetic raw datasets with realistic data quality defects
│   ├── outlets.csv               # 22 Coimbatore salon outlet profiles, locations, and tiers
│   ├── customers.csv             # 42,000 synthetic customer demographic & segment profiles
│   ├── services.csv              # Standard service catalogue with prices and durations
│   ├── stylists.csv              # 119 stylist profiles with experience and specialization
│   ├── appointments.csv          # 303,900 raw appointment records (2022-2024, includes duplicates & noise)
│   ├── product_sales.csv         # 48,000 retail product transaction records
│   ├── experiment_data.csv       # 18,000 randomized A/B trial records (stratified by outlet)
│   └── historical_regional_data.csv # 240 monthly panel records (Difference-in-Differences)
│
├── processed/                    # Cleaned, standardized, and validated analytical datasets
│   ├── outlets_clean.csv
│   ├── customers_clean.csv
│   ├── services_clean.csv
│   ├── stylists_clean.csv
│   ├── appointments_clean.csv
│   ├── product_sales_clean.csv
│   ├── experiment_data_clean.csv
│   ├── historical_regional_data_clean.csv
│   └── customer_features.csv     # Engineered customer-level analytical features
│
└── README.md                     # Data catalog and schema documentation
```

---

## Dataset Descriptions & Raw Schemas

### 1. `outlets.csv` (22 rows)
- `outlet_id`: Unique primary key (`OUT001` - `OUT022`)
- `outlet_name`: Full commercial name of outlet
- `location`: Neighborhood in Coimbatore (e.g., RS Puram, Race Course, Peelamedu, Saravanampatti)
- `region`: Coimbatore administrative region (`Central`, `East`, `North`, `South`, `West`)
- `tier`: Outlet classification (`Flagship Premium`, `High Volume Urban`, `Student & IT Hub`, `Residential Premium`, etc.)
- `opening_date`: Date outlet commenced operations (`YYYY-MM-DD`)

### 2. `services.csv` (24 rows)
- `service_id`: Unique primary key (`SRV001` - `SRV024`)
- `service_name`: Standardized official service title
- `service_category`: Service vertical (`Hair Care`, `Skin & Facial`, `Spa & Massage`, `Bridal & Makeup`, `Nail Care`)
- `standard_price`: List price in INR (₹300 - ₹16,000)
- `duration_minutes`: Estimated service duration in minutes (25 - 240 min)

### 3. `stylists.csv` (119 rows)
- `stylist_id`: Unique primary key (`STY001` - `STY119`)
- `outlet_id`: Foreign key referencing `outlets.csv`
- `experience_level`: Stylist seniority (`Junior`, `Mid-Level`, `Senior`, `Master Stylist`)
- `specialization`: Primary craft (`Hair Stylist`, `Skin Specialist`, `Aesthetician`, `Massage Therapist`, etc.)
- `base_retention_score`: Baseline client retention rating (0.65 - 0.95)

### 4. `customers.csv` (42,000 rows)
- `customer_id`: Unique primary key (`CUST00001` - `CUST42000`)
- `age`: Customer age (18 - 72 years)
- `gender`: Gender identity (`Female`, `Male`, `Other`)
- `registration_date`: Account creation date (`YYYY-MM-DD`)
- `customer_segment`: Marketing persona (`Budget`, `Regular`, `Premium`, `VIP`)
- `acquisition_source`: Marketing attribution (`Instagram`, `Google Search`, `Walk-in Referral`, etc.)
- `preferred_outlet_id`: Home outlet preference

### 5. `appointments.csv` (303,900 raw rows)
- `appointment_id`: Appointment identifier (`APT000001` - `APT300000`)
- `customer_id`: Foreign key to `customers.csv` (~4% missing specifically for walk-in appointments)
- `outlet_id`: Foreign key to `outlets.csv`
- `stylist_id`: Foreign key to `stylists.csv`
- `service_id`: Foreign key to `services.csv`
- `raw_service_name`: Free-text alias entered at outlet POS (includes casing, typos, abbreviations)
- `appointment_date`: Date of booking (`YYYY-MM-DD`, 2022-01-01 to 2024-12-31)
- `appointment_time`: Time slot (`HH:MM`)
- `booking_channel`: Channel (`Mobile App`, `Website`, `Phone Call`, `Walk-in`, with ~1.2% missing values)
- `appointment_status`: Recorded status (`Completed`, `Cancelled`, `No-Show`)
- `cancellation_indicator`: Binary flag (1 if Cancelled, 0 otherwise)
- `no_show_indicator`: Binary flag (1 if No-Show, 0 otherwise)
- `walk_in_indicator`: Binary flag (1 if Walk-in, 0 otherwise)
- `treatment_group`: Retrospective / prospective assignment tag (`Treatment`, `Control`, `Non-Experiment`)
- `revenue`: Monetary realization in INR (includes discounts, add-on upsells, negative refund glitches, and outlier typos)

### 6. `product_sales.csv` (48,000 rows)
- `transaction_id`: Transaction primary key (`TXN000001` - `TXN048000`)
- `customer_id`: Foreign key to `customers.csv` (~12% null for anonymous retail walk-ins)
- `appointment_id`: Foreign key to `appointments.csv` (linked for salon floor retail upsells; null for retail walk-ins)
- `outlet_id`: Foreign key to `outlets.csv`
- `product_id`: Product catalogue code (`PRD001` - `PRD009`)
- `product_name`: SKU name
- `product_category`: Product category (`Hair Care`, `Skin Care`, `Styling Tools`, `Spa Products`)
- `product_price`: Retail unit price in INR
- `quantity`: Units purchased (1 - 4)
- `total_amount`: Total transaction value in INR
- `transaction_date`: Date of transaction (`YYYY-MM-DD`)

### 7. `experiment_data.csv` (18,000 rows)
- `customer_id`: Unique customer enrolled in prospective randomized A/B trial
- `outlet_id`: Stratification unit (outlet)
- `treatment_assignment`: Experimental arm (`Treatment` vs. `Control`)
- `pre_experiment_visit_count`: Count of completed visits during 6-month pre-trial baseline
- `pre_experiment_revenue`: Cumulative spend during 6-month pre-trial baseline
- `previous_no_show_history`: Count of unnotified missed bookings in baseline
- `experiment_start_date`: Trial kickoff (`2024-07-01`)
- `retention_outcome`: Binary outcome (1 if completed visit within 60 days post-start, 0 otherwise)
- `revenue_per_visit`: Post-trial average revenue per visit in INR
- `no_show_outcome`: Binary outcome (1 if customer had at least one no-show during trial period)

### 8. `historical_regional_data.csv` (240 rows)
- `month`: Monthly panel period (`YYYY-MM`, 48 months: 2021-01 to 2024-12)
- `region`: Geographic region (`Central`, `North`, `East`, `South`, `West`)
- `treatment_region_indicator`: 1 if region received regional management policy (`Central`, `North`), 0 otherwise
- `post_intervention_indicator`: 1 if month >= `2023-01`, 0 otherwise
- `revenue_per_visit`: Regional average revenue per completed visit
- `customer_retention`: Regional monthly repeat customer retention rate
- `no_show_rate`: Regional monthly no-show proportion
- `number_of_appointments`: Total appointment volume
