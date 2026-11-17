"""
test_statistical_inference.py
=============================
Rigorous automated test suite verifying statistical inference, power analysis,
hypothesis testing, DiD econometric formulation, and CUPED variance reduction.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "src"))

from statistical_inference import (
    calculate_sample_size_proportions,
    power_sensitivity_grid,
    check_sample_ratio_mismatch,
    compute_standardized_mean_difference,
    assess_baseline_randomization_balance,
    estimate_proportions_treatment_effect,
    estimate_continuous_treatment_effect,
    bootstrap_treatment_effect_ci
)
from causal_inference import (
    fit_did_model,
    fit_event_study_model,
    compute_cuped_adjustment,
    evaluate_subgroup_effects
)

PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")


def test_sample_size_calculation():
    """Verifies that sample size calculation satisfies statistical power invariants."""
    res_large_mde = calculate_sample_size_proportions(p_baseline=0.55, mde_abs=0.05, alpha=0.05, power=0.80)
    res_small_mde = calculate_sample_size_proportions(p_baseline=0.55, mde_abs=0.02, alpha=0.05, power=0.80)

    # Detecting a smaller effect strictly requires a larger sample size
    assert res_small_mde["n_per_group"] > res_large_mde["n_per_group"], "Smaller MDE must require larger N"
    assert res_large_mde["n_total_required"] == res_large_mde["n_per_group"] * 2
    assert res_large_mde["cohens_h"] > 0


def test_power_sensitivity_grid():
    """Verifies sensitivity grid generation across multiple MDEs."""
    df_grid = power_sensitivity_grid(p_baseline=0.575)
    assert len(df_grid) >= 10
    assert "mde_absolute" in df_grid.columns
    assert "power" in df_grid.columns
    assert "n_per_group" in df_grid.columns
    # Sample size decreases monotonically as MDE increases
    df_pwr80 = df_grid[df_grid["power"] == 0.80].sort_values("mde_absolute")
    assert df_pwr80["n_per_group"].is_monotonic_decreasing


def test_sample_ratio_mismatch_check():
    """Verifies SRM Chi-square goodness-of-fit accurately detects and passes allocations."""
    # Balanced case (50/50 split)
    pass_res = check_sample_ratio_mismatch({"Control": 9000, "Treatment": 9000}, alpha=0.01)
    assert not pass_res["srm_detected"], "50/50 allocation must pass SRM check"
    assert pass_res["chi2_statistic"] == 0.0
    assert pass_res["p_value"] == 1.0

    # Imbalanced corrupted case (e.g., 55/45 split on 18,000 units)
    fail_res = check_sample_ratio_mismatch({"Control": 9900, "Treatment": 8100}, alpha=0.01)
    assert fail_res["srm_detected"], "Severe 55/45 allocation must trigger SRM warning"
    assert fail_res["p_value"] < 0.001


def test_standardized_mean_difference():
    """Verifies Cohen's d SMD formula calculation."""
    g1 = np.array([10.0, 12.0, 14.0, 16.0, 18.0])
    g2 = np.array([10.0, 12.0, 14.0, 16.0, 18.0])
    assert compute_standardized_mean_difference(g1, g2) == 0.0

    g3 = g1 + 5.0
    d = compute_standardized_mean_difference(g3, g1)
    assert d > 0, "Shifted distribution must have positive SMD"


def test_baseline_balance_diagnostics():
    """Verifies baseline balance evaluation across experiment covariates."""
    df_exp = pd.read_csv(os.path.join(PROCESSED_DIR, "experiment_data_clean.csv"))
    df_cust = pd.read_csv(os.path.join(PROCESSED_DIR, "customers_clean.csv"))
    df_m = df_exp.merge(df_cust[["customer_id", "age", "customer_segment"]], on="customer_id", how="left")

    balance = assess_baseline_randomization_balance(
        df_m,
        treatment_col="treatment_assignment",
        continuous_covariates=["age", "pre_experiment_visit_count", "pre_experiment_revenue"],
        categorical_covariates=["customer_segment"]
    )
    assert len(balance) == 4
    # Randomized dataset must be balanced: all SMDs < 0.05
    assert (balance["standardized_mean_diff (SMD)"] < 0.05).all(), "Randomized trial must have balanced covariates"


def test_two_proportion_treatment_effect():
    """Verifies two-proportion z-test and confidence intervals."""
    res = estimate_proportions_treatment_effect(
        success_t=600, n_t=1000,
        success_c=500, n_c=1000,
        alpha=0.05
    )
    assert res["control_rate"] == 0.50
    assert res["treatment_rate"] == 0.60
    assert np.isclose(res["absolute_effect"], 0.10)
    assert res["ci_lower_95"] < res["absolute_effect"] < res["ci_upper_95"]
    assert res["p_value"] < 0.001
    assert res["statistically_significant"]


def test_welch_ttest_continuous_effect():
    """Verifies Welch's t-test for continuous metrics."""
    np.random.seed(42)
    s_t = np.random.normal(loc=1500, scale=300, size=500)
    s_c = np.random.normal(loc=1350, scale=300, size=500)

    res = estimate_continuous_treatment_effect(s_t, s_c, alpha=0.05)
    assert res["absolute_effect"] > 100
    assert res["ci_lower_95"] < res["absolute_effect"] < res["ci_upper_95"]
    assert res["statistically_significant"]


def test_bootstrap_confidence_interval():
    """Verifies non-parametric percentile bootstrap CI coverage."""
    np.random.seed(42)
    s_t = np.random.normal(loc=100, scale=15, size=200)
    s_c = np.random.normal(loc=80, scale=15, size=200)

    est, ci_l, ci_u = bootstrap_treatment_effect_ci(s_t, s_c, n_boot=500, alpha=0.05)
    assert ci_l < est < ci_u
    assert np.isclose(est, 20.0, atol=3.0)


def test_difference_in_differences_empirical_equivalence():
    """Verifies that DiD regression beta3 matches the 2x2 sample double difference."""
    df_hist = pd.read_csv(os.path.join(PROCESSED_DIR, "historical_regional_data_clean.csv"))
    did_res = fit_did_model(df_hist, outcome_col="revenue_per_visit")

    # Regression coefficient beta3 must exactly equal empirical 2x2 double difference
    assert np.isclose(did_res["did_coefficient_beta3"], did_res["empirical_2x2_did"], atol=1e-3)
    assert did_res["statistically_significant"]
    assert did_res["ci_lower_95"] < did_res["did_coefficient_beta3"] < did_res["ci_upper_95"]


def test_cuped_variance_reduction_mechanics():
    """Verifies CUPED mathematical invariants: unbiased point estimate and reduced variance."""
    df_exp = pd.read_csv(os.path.join(PROCESSED_DIR, "experiment_data_clean.csv"))
    cuped = compute_cuped_adjustment(
        df_exp,
        treatment_col="treatment_assignment",
        outcome_col="revenue_per_visit",
        pre_covariate_col="pre_experiment_revenue"
    )

    # 1. Variance must strictly decrease when correlation rho > 0
    assert cuped["correlation_rho"] > 0
    assert cuped["variance_cuped"] < cuped["variance_raw"], "CUPED variance must be smaller than raw variance"
    assert cuped["actual_variance_reduction_pct"] > 0

    # 2. Point estimate should be consistent with raw estimate (unbiased)
    assert np.isclose(cuped["cuped_treatment_effect"], cuped["raw_treatment_effect"], atol=15.0)

    # 3. Standard error must shrink
    assert cuped["cuped_se"] <= cuped["raw_se"]
