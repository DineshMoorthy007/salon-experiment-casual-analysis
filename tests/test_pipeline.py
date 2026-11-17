"""
test_pipeline.py
================
Comprehensive test suite verifying data engineering, cleaning, referential integrity,
and feature engineering across the Aura Luxe Salon datasets.
"""

import os
import pytest
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
TABLES_DIR = os.path.join(BASE_DIR, "outputs", "tables")


@pytest.fixture(scope="session")
def processed_data():
    """Loads processed datasets into memory for assertion checks."""
    datasets = {
        "outlets": pd.read_csv(os.path.join(PROCESSED_DIR, "outlets_clean.csv")),
        "services": pd.read_csv(os.path.join(PROCESSED_DIR, "services_clean.csv")),
        "stylists": pd.read_csv(os.path.join(PROCESSED_DIR, "stylists_clean.csv")),
        "customers": pd.read_csv(os.path.join(PROCESSED_DIR, "customers_clean.csv")),
        "appointments": pd.read_csv(os.path.join(PROCESSED_DIR, "appointments_clean.csv")),
        "product_sales": pd.read_csv(os.path.join(PROCESSED_DIR, "product_sales_clean.csv")),
        "experiment_data": pd.read_csv(os.path.join(PROCESSED_DIR, "experiment_data_clean.csv")),
        "historical_regional_data": pd.read_csv(os.path.join(PROCESSED_DIR, "historical_regional_data_clean.csv")),
        "customer_features": pd.read_csv(os.path.join(PROCESSED_DIR, "customer_features.csv")),
    }
    return datasets


def test_dataset_file_existence():
    """Verifies all raw and processed dataset files exist on disk."""
    raw_files = [
        "outlets.csv", "services.csv", "stylists.csv", "customers.csv",
        "appointments.csv", "product_sales.csv", "experiment_data.csv", "historical_regional_data.csv"
    ]
    for rf in raw_files:
        path = os.path.join(RAW_DIR, rf)
        assert os.path.exists(path), f"Missing raw file: {path}"

    processed_files = [
        "outlets_clean.csv", "services_clean.csv", "stylists_clean.csv", "customers_clean.csv",
        "appointments_clean.csv", "product_sales_clean.csv", "experiment_data_clean.csv",
        "historical_regional_data_clean.csv", "customer_features.csv"
    ]
    for pf in processed_files:
        path = os.path.join(PROCESSED_DIR, pf)
        assert os.path.exists(path), f"Missing processed file: {path}"


def test_appointment_record_counts(processed_data):
    """Verifies target appointment record count after de-duplication."""
    df_apts = processed_data["appointments"]
    assert len(df_apts) == 300000, f"Expected 300,000 cleaned appointments, got {len(df_apts)}"
    # Check no remaining duplicate appointment_ids
    assert df_apts["appointment_id"].duplicated().sum() == 0, "Duplicate appointment IDs found in processed data"


def test_outlet_specifications(processed_data):
    """Verifies exactly 22 outlets across Coimbatore regions."""
    df_outlets = processed_data["outlets"]
    assert len(df_outlets) == 22, f"Expected 22 outlets, got {len(df_outlets)}"
    assert df_outlets["outlet_id"].nunique() == 22
    regions = set(df_outlets["region"])
    expected_regions = {"Central", "East", "North", "South", "West"}
    assert regions == expected_regions, f"Region mismatch: {regions} != {expected_regions}"


def test_foreign_key_referential_integrity(processed_data):
    """Verifies all foreign key relationships in appointments."""
    df_apts = processed_data["appointments"]
    valid_outlets = set(processed_data["outlets"]["outlet_id"])
    valid_services = set(processed_data["services"]["service_id"])
    valid_stylists = set(processed_data["stylists"]["stylist_id"])
    valid_customers = set(processed_data["customers"]["customer_id"])

    # Outlets
    assert df_apts["outlet_id"].isin(valid_outlets).all(), "Invalid outlet foreign keys in appointments"
    # Services
    assert df_apts["service_id"].isin(valid_services).all(), "Invalid service foreign keys in appointments"
    # Stylists
    assert df_apts["stylist_id"].isin(valid_stylists).all(), "Invalid stylist foreign keys in appointments"

    # Non-guest customers must exist in customers.csv
    non_guest_apts = df_apts[~df_apts["customer_id"].str.startswith("GUEST_")]
    assert non_guest_apts["customer_id"].isin(valid_customers).all(), "Invalid customer foreign keys in appointments"


def test_revenue_validity(processed_data):
    """Verifies that revenues are non-negative and completed visits have positive revenue."""
    df_apts = processed_data["appointments"]
    # No negative revenue
    assert (df_apts["revenue"] < 0).sum() == 0, "Negative revenues detected"
    # Completed appointments must have positive revenue
    completed_zero_rev = (df_apts["appointment_status"] == "Completed") & (df_apts["revenue"] <= 0)
    assert completed_zero_rev.sum() == 0, "Completed appointments found with zero or negative revenue"
    # No-shows must have 0 revenue
    noshow_pos_rev = (df_apts["appointment_status"] == "No-Show") & (df_apts["revenue"] > 0)
    assert noshow_pos_rev.sum() == 0, "No-shows found with positive revenue"


def test_appointment_status_consistency(processed_data):
    """Verifies indicator alignment with appointment_status column."""
    df_apts = processed_data["appointments"]
    assert ((df_apts["appointment_status"] == "Completed") == (df_apts["completion_indicator"] == 1)).all()
    assert ((df_apts["appointment_status"] == "Cancelled") == (df_apts["cancellation_indicator"] == 1)).all()
    assert ((df_apts["appointment_status"] == "No-Show") == (df_apts["no_show_indicator"] == 1)).all()


def test_customer_features_validity(processed_data):
    """Verifies customer feature engineering logic and retention indicators."""
    df_cust = processed_data["customer_features"]
    assert len(df_cust) == 42000, f"Expected 42,000 customers, got {len(df_cust)}"
    assert (df_cust["total_customer_spending"] >= 0).all()
    assert (df_cust["average_revenue_per_visit"] >= 0).all()
    # Retention flags must be strictly binary 0 or 1
    assert set(df_cust["customer_retention_60d"].unique()).issubset({0, 1})
    assert set(df_cust["customer_retention_90d"].unique()).issubset({0, 1})
    # 90-day retention rate should be >= 60-day retention rate
    assert df_cust["customer_retention_90d"].mean() >= df_cust["customer_retention_60d"].mean()


def test_experiment_data_balance(processed_data):
    """Verifies prospective randomized A/B experiment structure."""
    df_exp = processed_data["experiment_data"]
    assert len(df_exp) == 18000, f"Expected 18,000 experiment records, got {len(df_exp)}"
    counts = df_exp["treatment_assignment"].value_counts()
    # 50-50 split
    assert abs(counts["Treatment"] - counts["Control"]) <= 10, "A/B assignment not balanced"
    # Pre-experiment visits non-negative
    assert (df_exp["pre_experiment_visit_count"] > 0).all()
    assert set(df_exp["retention_outcome"].unique()).issubset({0, 1})
    assert set(df_exp["no_show_outcome"].unique()).issubset({0, 1})


def test_historical_regional_data_structure(processed_data):
    """Verifies historical panel structure for Difference-in-Differences."""
    df_hist = processed_data["historical_regional_data"]
    assert len(df_hist) == 240, f"Expected 240 monthly panel records (5 regions * 48 months), got {len(df_hist)}"
    assert df_hist["region"].nunique() == 5
    assert df_hist["month"].nunique() == 48
    # Rates must be within [0, 1]
    assert (df_hist["customer_retention"] >= 0).all() and (df_hist["customer_retention"] <= 1.0).all()
    assert (df_hist["no_show_rate"] >= 0).all() and (df_hist["no_show_rate"] <= 1.0).all()
