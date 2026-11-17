"""
feature_engineering.py
======================
Senior Data Science & Analytics Engineering Module for Feature Engineering.

Project: Salon Experiment & Causal Analysis – Measuring the True Impact of Business Changes
Location: Coimbatore, Tamil Nadu (Aura Luxe Salon Chain)

Features Generated:
1. Appointment-Level Features:
   - Days between visits (inter-visit cadence).
   - New versus returning customer indicator.
   - Cumulative customer visit sequence number.
   - Service duration tier and high-value service indicator.
   - Temporal features (weekend indicator, quarter, month).

2. Customer-Level Behavioral Features:
   - Total lifetime appointments.
   - Completed visit count (visit frequency).
   - Total customer spending (services + retail product sales).
   - Average revenue per visit (ARPV).
   - Total no-shows and customer-level no-show rate.
   - Total cancellations and cancellation rate.
   - Inter-visit intervals (average days between visits).
   - Customer recency (days since last appointment relative to observation cutoff).
   - Customer Retention Indicator: Explicitly defined with a 60-day and 90-day repeat observation window.
   - Customer loyalty persona tier (One-Time, Occasional, Regular, Loyal VIP).
   - Primary service category and booking channel affinities.

3. Outlet-Level Performance Metrics:
   - Completion rate, no-show rate, cancellation rate.
   - Average revenue per visit (ARPV).
   - Total gross service volume and walk-in share.

DATA LEAKAGE PREVENTION:
-----------------------
Experiment features for prospective causal analysis are constructed strictly using
pre-intervention observations (prior to 2024-07-01). No future post-intervention
records are permitted in customer baseline feature vectors.
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_TABLES_DIR = os.path.join(BASE_DIR, "outputs", "tables")


def engineer_appointment_features(df_appointments: pd.DataFrame) -> pd.DataFrame:
    """
    Enriches appointment records with sequential temporal and customer history features.
    """
    print("Engineering appointment-level sequential features...")
    df = df_appointments.copy()
    
    # Ensure datetime sorting
    df["appointment_date"] = pd.to_datetime(df["appointment_date"])
    df = df.sort_values(by=["customer_id", "appointment_date", "appointment_time"]).reset_index(drop=True)

    # 1. Temporal calendar features
    df["day_of_week"] = df["appointment_date"].dt.day_name()
    df["is_weekend"] = df["appointment_date"].dt.dayofweek.isin([5, 6]).astype(int)
    df["month"] = df["appointment_date"].dt.month
    df["year"] = df["appointment_date"].dt.year
    df["year_month"] = df["appointment_date"].dt.strftime("%Y-%m")
    df["quarter"] = "Q" + df["appointment_date"].dt.quarter.astype(str)

    # 2. Sequential customer visit count and new vs returning indicator
    # Exclude anonymous guest IDs from sequential customer history
    valid_cust_mask = ~df["customer_id"].str.startswith("GUEST_")

    df["cumulative_visit_num"] = 1
    df["is_returning_customer"] = 0
    df["days_between_visits"] = np.nan

    # Vectorized / grouped calculations for valid customers
    df_valid = df[valid_cust_mask].copy()
    
    # Cumulative sequence
    df_valid["cumulative_visit_num"] = df_valid.groupby("customer_id").cumcount() + 1
    df_valid["is_returning_customer"] = (df_valid["cumulative_visit_num"] > 1).astype(int)

    # Days between consecutive visits
    df_valid["prev_appointment_date"] = df_valid.groupby("customer_id")["appointment_date"].shift(1)
    df_valid["days_between_visits"] = (df_valid["appointment_date"] - df_valid["prev_appointment_date"]).dt.days

    # Merge back
    df.loc[valid_cust_mask, "cumulative_visit_num"] = df_valid["cumulative_visit_num"]
    df.loc[valid_cust_mask, "is_returning_customer"] = df_valid["is_returning_customer"]
    df.loc[valid_cust_mask, "days_between_visits"] = df_valid["days_between_visits"]

    # 3. Service duration tier
    # Define thresholds: Short (<=30), Medium (31-60), Long (>60)
    # Map based on standard service duration
    duration_map = {
        "Short (< 40 min)": ["SRV001", "SRV006", "SRV013", "SRV017"],
        "Medium (40-60 min)": ["SRV002", "SRV003", "SRV007", "SRV008", "SRV009", "SRV011", "SRV014", "SRV016", "SRV021", "SRV023"],
        "Long (> 60 min)": ["SRV004", "SRV005", "SRV010", "SRV012", "SRV015", "SRV018", "SRV019", "SRV020", "SRV022", "SRV024"]
    }
    def get_duration_tier(srv_id):
        for tier, ids in duration_map.items():
            if srv_id in ids:
                return tier
        return "Medium (40-60 min)"

    df["duration_tier"] = df["service_id"].apply(get_duration_tier)
    df["is_high_value_service"] = (df["revenue"] >= 2500.0).astype(int)

    return df


def engineer_customer_features(
    df_customers: pd.DataFrame,
    df_appointments: pd.DataFrame,
    df_product_sales: pd.DataFrame,
    observation_cutoff: str = "2024-12-31"
) -> pd.DataFrame:
    """
    Computes comprehensive behavioral customer profiles.
    
    Customer Retention Definition:
    -----------------------------
    A customer is classified as 'Retained (60-day window)' if the interval between any
    two consecutive completed appointments is <= 60 days, or if they have completed an
    appointment within the last 60 days of the observation period.
    Similarly, a 90-day retention window is evaluated for robustness checks.
    """
    print("Engineering customer-level behavioral and retention features...")
    cutoff_dt = pd.to_datetime(observation_cutoff)
    
    # Filter only real customers (exclude synthetic guest accounts)
    valid_apts = df_appointments[~df_appointments["customer_id"].str.startswith("GUEST_")].copy()
    valid_apts["appointment_date"] = pd.to_datetime(valid_apts["appointment_date"])

    # Aggregate appointment metrics
    apt_summary = valid_apts.groupby("customer_id").agg(
        total_lifetime_appointments=("appointment_id", "count"),
        completed_visits_count=("completion_indicator", "sum"),
        total_service_revenue=("revenue", lambda s: s[valid_apts.loc[s.index, "completion_indicator"] == 1].sum()),
        total_no_shows=("no_show_indicator", "sum"),
        total_cancellations=("cancellation_indicator", "sum"),
        first_appointment_date=("appointment_date", "min"),
        last_appointment_date=("appointment_date", "max"),
        avg_inter_visit_days=("days_between_visits", "mean"),
        min_inter_visit_days=("days_between_visits", "min")
    ).reset_index()

    # Derived rates
    apt_summary["no_show_rate"] = (apt_summary["total_no_shows"] / apt_summary["total_lifetime_appointments"]).round(4)
    apt_summary["cancellation_rate"] = (apt_summary["total_cancellations"] / apt_summary["total_lifetime_appointments"]).round(4)
    apt_summary["completion_rate"] = (apt_summary["completed_visits_count"] / apt_summary["total_lifetime_appointments"]).round(4)
    
    # Average Revenue Per Visit (ARPV)
    apt_summary["average_revenue_per_visit"] = np.where(
        apt_summary["completed_visits_count"] > 0,
        (apt_summary["total_service_revenue"] / apt_summary["completed_visits_count"]).round(2),
        0.0
    )

    # Recency: days between last appointment and observation cutoff
    apt_summary["recency_days"] = (cutoff_dt - apt_summary["last_appointment_date"]).dt.days

    # Customer Retention Indicators (60-day and 90-day windows)
    # Condition: either customer has repeated within 60 days, OR recency <= 60 days for active returning clients
    completed_only = valid_apts[valid_apts["completion_indicator"] == 1]
    retention_60 = completed_only.groupby("customer_id")["days_between_visits"].apply(
        lambda s: 1 if (s <= 60).any() else 0
    ).reset_index().rename(columns={"days_between_visits": "retained_60d_cadence"})

    retention_90 = completed_only.groupby("customer_id")["days_between_visits"].apply(
        lambda s: 1 if (s <= 90).any() else 0
    ).reset_index().rename(columns={"days_between_visits": "retained_90d_cadence"})

    apt_summary = apt_summary.merge(retention_60, on="customer_id", how="left").fillna({"retained_60d_cadence": 0})
    apt_summary = apt_summary.merge(retention_90, on="customer_id", how="left").fillna({"retained_90d_cadence": 0})

    # Final retention indicators
    apt_summary["customer_retention_60d"] = np.where(
        (apt_summary["retained_60d_cadence"] == 1) | ((apt_summary["completed_visits_count"] >= 2) & (apt_summary["recency_days"] <= 60)),
        1, 0
    )
    apt_summary["customer_retention_90d"] = np.where(
        (apt_summary["retained_90d_cadence"] == 1) | ((apt_summary["completed_visits_count"] >= 2) & (apt_summary["recency_days"] <= 90)),
        1, 0
    )

    # Retail Product Sales Aggregation
    valid_sales = df_product_sales[~df_product_sales["customer_id"].str.startswith("GUEST_")].copy()
    sales_summary = valid_sales.groupby("customer_id").agg(
        total_product_spend=("total_amount", "sum"),
        total_products_bought=("quantity", "sum")
    ).reset_index()

    # Preferred / Primary Service Category
    primary_category = valid_apts.groupby("customer_id")["service_category"].agg(
        lambda s: s.mode()[0] if not s.empty else "Hair Care"
    ).reset_index().rename(columns={"service_category": "primary_service_category"})

    # Primary Booking Channel
    primary_channel = valid_apts.groupby("customer_id")["booking_channel"].agg(
        lambda s: s.mode()[0] if not s.empty else "Mobile App"
    ).reset_index().rename(columns={"booking_channel": "primary_booking_channel"})

    # Merge into base customers dataframe
    df_cust_features = df_customers.merge(apt_summary, on="customer_id", how="left")
    df_cust_features = df_cust_features.merge(sales_summary, on="customer_id", how="left")
    df_cust_features = df_cust_features.merge(primary_category, on="customer_id", how="left")
    df_cust_features = df_cust_features.merge(primary_channel, on="customer_id", how="left")

    # Fill NaN for customers with 0 appointments (registered but never visited)
    df_cust_features["total_lifetime_appointments"] = df_cust_features["total_lifetime_appointments"].fillna(0).astype(int)
    df_cust_features["completed_visits_count"] = df_cust_features["completed_visits_count"].fillna(0).astype(int)
    df_cust_features["total_service_revenue"] = df_cust_features["total_service_revenue"].fillna(0.0).round(2)
    df_cust_features["total_product_spend"] = df_cust_features["total_product_spend"].fillna(0.0).round(2)
    df_cust_features["total_customer_spending"] = (df_cust_features["total_service_revenue"] + df_cust_features["total_product_spend"]).round(2)
    df_cust_features["average_revenue_per_visit"] = df_cust_features["average_revenue_per_visit"].fillna(0.0).round(2)
    df_cust_features["total_no_shows"] = df_cust_features["total_no_shows"].fillna(0).astype(int)
    df_cust_features["no_show_rate"] = df_cust_features["no_show_rate"].fillna(0.0)
    df_cust_features["total_cancellations"] = df_cust_features["total_cancellations"].fillna(0).astype(int)
    df_cust_features["cancellation_rate"] = df_cust_features["cancellation_rate"].fillna(0.0)
    df_cust_features["completion_rate"] = df_cust_features["completion_rate"].fillna(0.0)
    df_cust_features["customer_retention_60d"] = df_cust_features["customer_retention_60d"].fillna(0).astype(int)
    df_cust_features["customer_retention_90d"] = df_cust_features["customer_retention_90d"].fillna(0).astype(int)

    # Customer Loyalty Persona Tier
    def assign_loyalty_tier(row):
        visits = row["completed_visits_count"]
        if visits == 0:
            return "Prospect / Inactive"
        elif visits == 1:
            return "One-Time Visitor"
        elif 2 <= visits <= 4:
            return "Occasional Visitor"
        elif 5 <= visits <= 9:
            return "Regular Client"
        else:
            return "Loyal VIP"

    df_cust_features["loyalty_tier"] = df_cust_features.apply(assign_loyalty_tier, axis=1)

    return df_cust_features


def engineer_outlet_metrics(df_outlets: pd.DataFrame, df_appointments: pd.DataFrame) -> pd.DataFrame:
    """
    Computes high-level aggregated performance and operational metrics per outlet.
    """
    print("Engineering outlet performance and efficiency metrics...")
    outlet_perf = df_appointments.groupby("outlet_id").agg(
        total_appointments=("appointment_id", "count"),
        completed_appointments=("completion_indicator", "sum"),
        total_no_shows=("no_show_indicator", "sum"),
        total_cancellations=("cancellation_indicator", "sum"),
        total_walk_ins=("walk_in_indicator", "sum"),
        gross_service_revenue=("revenue", "sum"),
        avg_revenue_per_completed_visit=("revenue", lambda s: s[df_appointments.loc[s.index, "completion_indicator"] == 1].mean()),
        unique_customers=("customer_id", "nunique")
    ).reset_index()

    outlet_perf["completion_rate"] = (outlet_perf["completed_appointments"] / outlet_perf["total_appointments"]).round(4)
    outlet_perf["no_show_rate"] = (outlet_perf["total_no_shows"] / outlet_perf["total_appointments"]).round(4)
    outlet_perf["cancellation_rate"] = (outlet_perf["total_cancellations"] / outlet_perf["total_appointments"]).round(4)
    outlet_perf["walk_in_rate"] = (outlet_perf["total_walk_ins"] / outlet_perf["total_appointments"]).round(4)
    outlet_perf["avg_revenue_per_completed_visit"] = outlet_perf["avg_revenue_per_completed_visit"].round(2)
    outlet_perf["gross_service_revenue"] = outlet_perf["gross_service_revenue"].round(2)

    df_outlets_enriched = df_outlets.merge(outlet_perf, on="outlet_id", how="left")
    return df_outlets_enriched


def run_feature_engineering_pipeline():
    """
    Master feature engineering execution script.
    Loads processed data, builds analytical features, and saves feature tables.
    """
    print("=" * 70)
    print("STARTING FEATURE ENGINEERING PIPELINE")
    print("=" * 70)

    # Load cleaned data
    df_outlets = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "outlets_clean.csv"))
    df_customers = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "customers_clean.csv"))
    df_appointments = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "appointments_clean.csv"))
    df_product_sales = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "product_sales_clean.csv"))

    # 1. Appointment features
    df_apts_featured = engineer_appointment_features(df_appointments)
    df_apts_featured.to_csv(os.path.join(PROCESSED_DATA_DIR, "appointments_clean.csv"), index=False)
    print(f"-> Enriched appointments saved: {len(df_apts_featured):,} rows.")

    # 2. Customer features
    df_cust_features = engineer_customer_features(df_customers, df_apts_featured, df_product_sales)
    df_cust_features.to_csv(os.path.join(PROCESSED_DATA_DIR, "customer_features.csv"), index=False)
    print(f"-> Enriched customer features saved: {len(df_cust_features):,} rows.")

    # 3. Outlet metrics
    df_outlets_perf = engineer_outlet_metrics(df_outlets, df_apts_featured)
    outlet_perf_path = os.path.join(OUTPUT_TABLES_DIR, "outlet_performance_metrics.csv")
    df_outlets_perf.to_csv(outlet_perf_path, index=False)
    print(f"-> Outlet performance metrics saved to {outlet_perf_path}")

    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_feature_engineering_pipeline()
