"""
statistical_inference.py
========================
Senior Data Science & Experimentation Module for Statistical Inference.

Project: Salon Experiment & Causal Analysis – Measuring the True Impact of Business Changes
Enterprise: Aura Luxe Salon & Spa Chain (Coimbatore, Tamil Nadu)

Responsibilities:
1. Statistical power and sample size determination (proportions & continuous metrics).
2. Sample Ratio Mismatch (SRM) goodness-of-fit validation.
3. Baseline randomization balance checks (SMD / Cohen's d, t-tests, Chi-square).
4. Treatment effect estimation (Two-proportion z-tests, Welch's t-tests, Bootstrap CIs).
5. Statistical hypothesis reporting and business translation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import scipy.stats as stats
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportions_ztest, confint_proportions_2indep


def calculate_sample_size_proportions(
    p_baseline: float,
    mde_abs: float,
    alpha: float = 0.05,
    power: float = 0.80,
    ratio: float = 1.0
) -> Dict[str, float]:
    """
    Computes required sample size per group for a two-sample proportion test
    using Statsmodels NormalIndPower and Cohen's h effect size transformation.

    Parameters:
    -----------
    p_baseline : float
        Baseline conversion/retention rate (e.g., 0.575)
    mde_abs : float
        Minimum Detectable Effect in absolute percentage points (e.g., 0.03 for +3pp)
    alpha : float
        Significance level (Type I error probability, default: 0.05)
    power : float
        Statistical power (1 - Type II error probability, default: 0.80)
    ratio : float
        Allocation ratio n_treatment / n_control (default: 1.0)

    Returns:
    --------
    Dict containing Cohen's h, sample size per group, total sample size, and parameters.
    """
    p_target = p_baseline + mde_abs
    if not (0 < p_baseline < 1 and 0 < p_target < 1):
        raise ValueError("Proportions must strictly reside within (0, 1).")

    # Cohen's h effect size transformation for proportions:
    # h = 2 * arcsin(sqrt(p_target)) - 2 * arcsin(sqrt(p_baseline))
    h = 2 * np.arcsin(np.sqrt(p_target)) - 2 * np.arcsin(np.sqrt(p_baseline))

    power_analysis = NormalIndPower()
    n_per_group = power_analysis.solve_power(
        effect_size=abs(h),
        alpha=alpha,
        power=power,
        ratio=ratio,
        alternative="two-sided"
    )

    n_per_group_int = int(np.ceil(n_per_group))
    n_total = int(n_per_group_int * (1 + ratio))

    return {
        "baseline_rate": p_baseline,
        "mde_absolute": mde_abs,
        "mde_relative_pct": round((mde_abs / p_baseline) * 100, 2),
        "target_rate": p_target,
        "cohens_h": round(float(h), 4),
        "alpha": alpha,
        "power": power,
        "allocation_ratio": ratio,
        "n_per_group": n_per_group_int,
        "n_total_required": n_total
    }


def power_sensitivity_grid(
    p_baseline: float = 0.575,
    mde_range: Optional[List[float]] = None,
    powers: Optional[List[float]] = None,
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Generates a sensitivity grid of required sample sizes across multiple MDEs and power levels.
    """
    if mde_range is None:
        mde_range = [0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.050, 0.060]
    if powers is None:
        powers = [0.80, 0.90]

    records = []
    for mde in mde_range:
        for pwr in powers:
            res = calculate_sample_size_proportions(p_baseline, mde, alpha=alpha, power=pwr)
            records.append({
                "mde_absolute": mde,
                "mde_relative_pct": res["mde_relative_pct"],
                "power": pwr,
                "n_per_group": res["n_per_group"],
                "n_total": res["n_total_required"]
            })
    return pd.DataFrame(records)


def check_sample_ratio_mismatch(
    observed_counts: Dict[str, int],
    expected_ratio: Dict[str, float] = None,
    alpha: float = 0.01
) -> Dict[str, object]:
    """
    Conducts a Chi-square Goodness-of-Fit test to detect Sample Ratio Mismatch (SRM).

    Parameters:
    -----------
    observed_counts : Dict[str, int]
        Observed sample sizes per group, e.g. {"Control": 9000, "Treatment": 9000}
    expected_ratio : Dict[str, float]
        Expected theoretical proportion split, e.g. {"Control": 0.5, "Treatment": 0.5}
    alpha : float
        Decision threshold for SRM detection (conventionally 0.01 or 0.001)

    Returns:
    --------
    Dict containing observed counts, expected counts, chi2 stat, p-value, and diagnostic conclusion.
    """
    if expected_ratio is None:
        expected_ratio = {"Control": 0.5, "Treatment": 0.5}

    groups = list(expected_ratio.keys())
    total_obs = sum(observed_counts[g] for g in groups)

    obs = np.array([observed_counts[g] for g in groups], dtype=float)
    exp_probs = np.array([expected_ratio[g] for g in groups], dtype=float)
    exp_probs = exp_probs / exp_probs.sum()
    expected = total_obs * exp_probs

    chi2_stat, p_val = stats.chisquare(f_obs=obs, f_exp=expected)
    srm_flag = bool(p_val < alpha)

    if srm_flag:
        diag = (f"CRITICAL WARNING: Severe Sample Ratio Mismatch detected (p = {p_val:.5f} < {alpha}). "
                "Randomization mechanism or data collection pipeline may be corrupted. Investigate data filters.")
    else:
        diag = (f"SRM Check Passed (p = {p_val:.4f} >= {alpha}). "
                "Observed allocation closely matches theoretical allocation. No evidence of allocation bias.")

    return {
        "groups": groups,
        "observed_counts": dict(zip(groups, obs.astype(int))),
        "expected_counts": dict(zip(groups, expected.round(2))),
        "chi2_statistic": round(float(chi2_stat), 4),
        "p_value": float(p_val),
        "alpha_threshold": alpha,
        "srm_detected": srm_flag,
        "conclusion": diag
    }


def compute_standardized_mean_difference(
    group1: np.ndarray,
    group2: np.ndarray
) -> float:
    """
    Calculates Cohen's d (Standardized Mean Difference) between two groups.
    SMD = (Mean1 - Mean2) / Pooled_Standard_Deviation
    """
    n1, n2 = len(group1), len(group2)
    m1, m2 = np.mean(group1), np.mean(group2)
    v1, v2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    if pooled_sd == 0:
        return 0.0
    return float((m1 - m2) / pooled_sd)


def assess_baseline_randomization_balance(
    df: pd.DataFrame,
    treatment_col: str,
    continuous_covariates: List[str],
    categorical_covariates: List[str],
    treatment_val: str = "Treatment",
    control_val: str = "Control",
    smd_threshold: float = 0.05
) -> pd.DataFrame:
    """
    Performs rigorous baseline balance diagnostics across pre-experiment covariates.
    Computes means, standard deviations, differences, SMDs, and formal hypothesis test p-values.
    """
    df_t = df[df[treatment_col] == treatment_val]
    df_c = df[df[treatment_col] == control_val]

    balance_records = []

    # Continuous Covariates
    for col in continuous_covariates:
        c_vals = df_c[col].dropna().values
        t_vals = df_t[col].dropna().values

        mean_c, std_c = float(np.mean(c_vals)), float(np.std(c_vals, ddof=1))
        mean_t, std_t = float(np.mean(t_vals)), float(np.std(t_vals, ddof=1))
        smd = compute_standardized_mean_difference(t_vals, c_vals)

        # Welch's t-test
        t_stat, p_val = stats.ttest_ind(t_vals, c_vals, equal_var=False)

        balance_records.append({
            "covariate": col,
            "type": "Continuous",
            "control_mean": round(mean_c, 3),
            "control_std": round(std_c, 3),
            "treatment_mean": round(mean_t, 3),
            "treatment_std": round(std_t, 3),
            "raw_difference": round(mean_t - mean_c, 3),
            "standardized_mean_diff (SMD)": round(abs(smd), 4),
            "test_statistic": round(float(t_stat), 3),
            "p_value": round(float(p_val), 4),
            "balanced (SMD < threshold)": abs(smd) < smd_threshold
        })

    # Categorical Covariates
    for col in categorical_covariates:
        contingency = pd.crosstab(df[treatment_col], df[col])
        chi2, p_val, dof, _ = stats.chi2_contingency(contingency)
        # Cramer's V for categorical effect size
        n = contingency.sum().sum()
        r, k = contingency.shape
        cramers_v = np.sqrt(chi2 / (n * (min(r, k) - 1)))

        balance_records.append({
            "covariate": col,
            "type": "Categorical",
            "control_mean": np.nan,
            "control_std": np.nan,
            "treatment_mean": np.nan,
            "treatment_std": np.nan,
            "raw_difference": np.nan,
            "standardized_mean_diff (SMD)": round(float(cramers_v), 4),
            "test_statistic": round(float(chi2), 3),
            "p_value": round(float(p_val), 4),
            "balanced (SMD < threshold)": cramers_v < smd_threshold
        })

    return pd.DataFrame(balance_records)


def estimate_proportions_treatment_effect(
    success_t: int,
    n_t: int,
    success_c: int,
    n_c: int,
    metric_name: str = "Retention Rate",
    alpha: float = 0.05
) -> Dict[str, object]:
    """
    Estimates treatment effect for binary proportion metrics using a Two-Proportion Z-Test
    and calculates asymptotic Wald/Score 95% Confidence Intervals.
    """
    p_t = success_t / n_t
    p_c = success_c / n_c
    abs_effect = p_t - p_c
    rel_effect_pct = (abs_effect / p_c) * 100 if p_c > 0 else np.nan

    # Two-proportion z-test via Statsmodels
    count = np.array([success_t, success_c])
    nobs = np.array([n_t, n_c])
    z_stat, p_val = proportions_ztest(count, nobs, alternative="two-sided")

    # 95% Confidence Interval for difference in proportions (Newcombe / Score method)
    ci_low, ci_high = confint_proportions_2indep(
        count1=success_t, nobs1=n_t,
        count2=success_c, nobs2=n_c,
        method="score",
        alpha=alpha
    )

    is_significant = bool(p_val < alpha)

    return {
        "metric": metric_name,
        "control_n": n_c,
        "control_successes": success_c,
        "control_rate": round(p_c, 4),
        "treatment_n": n_t,
        "treatment_successes": success_t,
        "treatment_rate": round(p_t, 4),
        "absolute_effect": round(abs_effect, 4),
        "relative_effect_pct": round(rel_effect_pct, 2),
        "ci_lower_95": round(ci_low, 4),
        "ci_upper_95": round(ci_high, 4),
        "z_statistic": round(float(z_stat), 4),
        "p_value": float(p_val),
        "statistically_significant": is_significant
    }


def estimate_continuous_treatment_effect(
    series_t: np.ndarray,
    series_c: np.ndarray,
    metric_name: str = "Revenue per Visit",
    alpha: float = 0.05
) -> Dict[str, object]:
    """
    Estimates treatment effect for continuous metrics using Welch's t-test
    (unequal variances) and calculates exact 95% Confidence Intervals.
    """
    series_t = np.array(series_t, dtype=float)
    series_c = np.array(series_c, dtype=float)

    n_t, n_c = len(series_t), len(series_c)
    mean_t, mean_c = float(np.mean(series_t)), float(np.mean(series_c))
    var_t, var_c = float(np.var(series_t, ddof=1)), float(np.var(series_c, ddof=1))

    abs_effect = mean_t - mean_c
    rel_effect_pct = (abs_effect / mean_c) * 100 if mean_c > 0 else np.nan

    # Standard error of difference
    se_diff = np.sqrt((var_t / n_t) + (var_c / n_c))

    # Welch-Satterthwaite degrees of freedom
    df_welch = ((var_t / n_t + var_c / n_c) ** 2) / (
        ((var_t / n_t) ** 2) / (n_t - 1) + ((var_c / n_c) ** 2) / (n_c - 1)
    )

    t_stat, p_val = stats.ttest_ind(series_t, series_c, equal_var=False)

    t_crit = stats.t.ppf(1 - alpha / 2, df=df_welch)
    ci_low = abs_effect - t_crit * se_diff
    ci_high = abs_effect + t_crit * se_diff

    return {
        "metric": metric_name,
        "control_n": n_c,
        "control_mean": round(mean_c, 2),
        "control_std": round(np.sqrt(var_c), 2),
        "treatment_n": n_t,
        "treatment_mean": round(mean_t, 2),
        "treatment_std": round(np.sqrt(var_t), 2),
        "absolute_effect": round(abs_effect, 2),
        "relative_effect_pct": round(rel_effect_pct, 2),
        "ci_lower_95": round(ci_low, 2),
        "ci_upper_95": round(ci_high, 2),
        "t_statistic": round(float(t_stat), 4),
        "p_value": float(p_val),
        "statistically_significant": bool(p_val < alpha)
    }


def bootstrap_treatment_effect_ci(
    series_t: np.ndarray,
    series_c: np.ndarray,
    stat_func=np.mean,
    n_boot: int = 2000,
    alpha: float = 0.05,
    random_seed: int = 42
) -> Tuple[float, float, float]:
    """
    Computes a non-parametric percentile Bootstrap Confidence Interval for the treatment effect.

    Returns:
    --------
    (point_estimate, ci_lower, ci_upper)
    """
    rng = np.random.default_rng(random_seed)
    series_t = np.array(series_t)
    series_c = np.array(series_c)

    point_estimate = float(stat_func(series_t) - stat_func(series_c))
    boot_diffs = np.empty(n_boot)

    for i in range(n_boot):
        sample_t = rng.choice(series_t, size=len(series_t), replace=True)
        sample_c = rng.choice(series_c, size=len(series_c), replace=True)
        boot_diffs[i] = stat_func(sample_t) - stat_func(sample_c)

    ci_low = float(np.percentile(boot_diffs, 100 * (alpha / 2)))
    ci_high = float(np.percentile(boot_diffs, 100 * (1 - alpha / 2)))

    return point_estimate, ci_low, ci_high
