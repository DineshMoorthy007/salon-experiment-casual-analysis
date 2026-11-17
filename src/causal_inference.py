"""
causal_inference.py
===================
Senior Data Science & Econometric Module for Causal Inference.

Project: Salon Experiment & Causal Analysis – Measuring the True Impact of Business Changes
Enterprise: Aura Luxe Salon & Spa Chain (Coimbatore, Tamil Nadu)

Responsibilities:
1. Difference-in-Differences (DiD) two-way regression with cluster-robust standard errors.
2. Dynamic Event-Study regression for testing parallel trends assumptions.
3. Controlled-experiment Using Pre-Experiment Data (CUPED) variance reduction.
4. Subgroup heterogeneous treatment effect estimation across customer, outlet, and service segments.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import scipy.stats as stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant


# =====================================================================
# 1. DIFFERENCE-IN-DIFFERENCES (DiD) REGRESSION ENGINE
# =====================================================================

def fit_did_model(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str = "treatment_region_indicator",
    post_col: str = "post_intervention_indicator",
    cluster_col: Optional[str] = "region",
    alpha: float = 0.05
) -> Dict[str, object]:
    """
    Fits standard Difference-in-Differences regression:
    Y = beta0 + beta1 * Treatment + beta2 * Post + beta3 * (Treatment * Post) + epsilon

    Parameters:
    -----------
    df : pd.DataFrame
        Historical panel dataframe.
    outcome_col : str
        Target dependent variable (e.g. 'revenue_per_visit', 'customer_retention', 'no_show_rate').
    treatment_col : str
        Binary indicator for treatment group (1 if treated unit, 0 if control).
    post_col : str
        Binary indicator for post-intervention period (1 if post, 0 if pre).
    cluster_col : str, optional
        Variable to cluster standard errors (e.g. 'region').
    alpha : float
        Significance level (default 0.05).

    Returns:
    --------
    Dict containing estimated coefficients, standard errors, t-stats, p-values, 95% CIs, and DiD effect.
    """
    df_reg = df.copy()
    df_reg["interaction"] = df_reg[treatment_col] * df_reg[post_col]

    X = df_reg[[treatment_col, post_col, "interaction"]]
    X = add_constant(X)
    y = df_reg[outcome_col]

    # Fit OLS model with clustered standard errors
    if cluster_col is not None and cluster_col in df_reg.columns:
        groups = df_reg[cluster_col]
        n_clusters = groups.nunique()
        model = OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": groups})
    else:
        n_clusters = None
        model = OLS(y, X).fit(cov_type="HC1")

    # Extract DiD interaction coefficient (beta3)
    coef_did = float(model.params["interaction"])
    se_did = float(model.bse["interaction"])
    t_stat_did = float(model.tvalues["interaction"])
    p_val_did = float(model.pvalues["interaction"])
    ci_low = float(model.conf_int(alpha=alpha).loc["interaction", 0])
    ci_high = float(model.conf_int(alpha=alpha).loc["interaction", 1])

    # Counterfactual computation
    # Pre-treatment control mean and treatment mean
    pre_control = df_reg[(df_reg[treatment_col] == 0) & (df_reg[post_col] == 0)][outcome_col].mean()
    post_control = df_reg[(df_reg[treatment_col] == 0) & (df_reg[post_col] == 1)][outcome_col].mean()
    pre_treat = df_reg[(df_reg[treatment_col] == 1) & (df_reg[post_col] == 0)][outcome_col].mean()
    post_treat = df_reg[(df_reg[treatment_col] == 1) & (df_reg[post_col] == 1)][outcome_col].mean()

    empirical_did = (post_treat - pre_treat) - (post_control - pre_control)

    return {
        "outcome": outcome_col,
        "did_coefficient_beta3": round(coef_did, 4),
        "standard_error": round(se_did, 4),
        "t_statistic": round(t_stat_did, 4),
        "p_value": float(p_val_did),
        "ci_lower_95": round(ci_low, 4),
        "ci_upper_95": round(ci_high, 4),
        "r_squared": round(float(model.rsquared), 4),
        "pre_treatment_control_mean": round(float(pre_control), 4),
        "post_treatment_control_mean": round(float(post_control), 4),
        "pre_treatment_treated_mean": round(float(pre_treat), 4),
        "post_treatment_treated_mean": round(float(post_treat), 4),
        "empirical_2x2_did": round(float(empirical_did), 4),
        "n_observations": int(model.nobs),
        "n_clusters": n_clusters,
        "cov_type": model.cov_type,
        "statistically_significant": bool(p_val_did < alpha)
    }


def fit_event_study_model(
    df: pd.DataFrame,
    outcome_col: str,
    time_col: str = "month",
    treatment_col: str = "treatment_region_indicator",
    intervention_month: str = "2023-01",
    cluster_col: Optional[str] = "region"
) -> pd.DataFrame:
    """
    Fits an Event-Study (Dynamic DiD) specification to evaluate pre-intervention parallel trends:
    Y_rt = alpha_r + lambda_t + sum_{k != -1} gamma_k * (Treatment_r * 1(t = k)) + epsilon_rt

    Where k = -1 (Month 24, 2022-12) is the omitted reference period immediately prior to intervention.
    """
    df_es = df.copy()
    unique_months = sorted(df_es[time_col].unique())
    inter_idx = unique_months.index(intervention_month)

    # Relative time index (0 is intervention month, -1 is 1 month prior)
    df_es["rel_time"] = df_es[time_col].apply(lambda m: unique_months.index(m) - inter_idx)
    ref_period = -1  # Omitted base period

    # Generate interaction dummies for relative time
    lead_lags = [t for t in sorted(df_es["rel_time"].unique()) if t != ref_period]
    dummy_cols = []
    for t in lead_lags:
        col_name = f"lead_lag_{t}"
        df_es[col_name] = ((df_es["rel_time"] == t) & (df_es[treatment_col] == 1)).astype(int)
        dummy_cols.append(col_name)

    # Add Region Fixed Effects and Month Fixed Effects
    X_parts = [df_es[dummy_cols]]
    
    # Region dummies
    region_dummies = pd.get_dummies(df_es["region"], prefix="reg", drop_first=True, dtype=int)
    X_parts.append(region_dummies)
    
    # Month dummies
    month_dummies = pd.get_dummies(df_es["rel_time"], prefix="time", drop_first=True, dtype=int)
    X_parts.append(month_dummies)

    X = pd.concat(X_parts, axis=1)
    X = add_constant(X)
    y = df_es[outcome_col]

    if cluster_col is not None and cluster_col in df_es.columns:
        model = OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": df_es[cluster_col]})
    else:
        model = OLS(y, X).fit(cov_type="HC1")

    # Build event-study dataframe
    results = []
    for t in sorted(df_es["rel_time"].unique()):
        if t == ref_period:
            results.append({
                "rel_time": t,
                "calendar_month": unique_months[t + inter_idx],
                "coefficient": 0.0,
                "standard_error": 0.0,
                "ci_lower_95": 0.0,
                "ci_upper_95": 0.0,
                "p_value": 1.0,
                "period_type": "Reference (Base)"
            })
        else:
            col_name = f"lead_lag_{t}"
            coef = float(model.params[col_name])
            se = float(model.bse[col_name])
            ci = model.conf_int(alpha=0.05).loc[col_name]
            p = float(model.pvalues[col_name])
            results.append({
                "rel_time": t,
                "calendar_month": unique_months[t + inter_idx],
                "coefficient": round(coef, 4),
                "standard_error": round(se, 4),
                "ci_lower_95": round(float(ci[0]), 4),
                "ci_upper_95": round(float(ci[1]), 4),
                "p_value": round(p, 4),
                "period_type": "Pre-Intervention Lead" if t < 0 else "Post-Intervention Lag"
            })

    return pd.DataFrame(results)


# =====================================================================
# 2. CUPED VARIANCE REDUCTION ENGINE
# =====================================================================

def compute_cuped_adjustment(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    pre_covariate_col: str,
    treatment_val: str = "Treatment",
    control_val: str = "Control"
) -> Dict[str, object]:
    """
    Applies CUPED (Controlled-experiment Using Pre-Experiment Data) variance reduction:
    Y_adjusted = Y - theta * (X - mean(X))
    where theta = Cov(Y, X) / Var(X)

    Parameters:
    -----------
    df : pd.DataFrame
        Experiment dataset containing target metric and pre-experiment covariate.
    treatment_col : str
        Column designating assignment ('Treatment' vs 'Control').
    outcome_col : str
        Target experimental continuous metric (Y, e.g. 'revenue_per_visit').
    pre_covariate_col : str
        Pre-experiment continuous covariate (X, e.g. 'pre_experiment_revenue').

    Returns:
    --------
    Dict containing theta, correlation, variance reduction, raw vs adjusted treatment effects,
    sample size savings, and adjusted dataframe.
    """
    df_clean = df[[treatment_col, outcome_col, pre_covariate_col]].dropna().copy()

    Y = df_clean[outcome_col].values
    X = df_clean[pre_covariate_col].values

    # 1. Compute theta = Cov(Y, X) / Var(X)
    cov_matrix = np.cov(Y, X, ddof=1)
    cov_yx = cov_matrix[0, 1]
    var_x = cov_matrix[1, 1]

    if var_x == 0:
        theta = 0.0
        corr_xy = 0.0
    else:
        theta = cov_yx / var_x
        corr_xy = cov_yx / (np.std(Y, ddof=1) * np.std(X, ddof=1))

    # 2. Calculate adjusted outcome: Y_adj = Y - theta * (X - mean(X))
    mean_x = float(np.mean(X))
    y_adjusted = Y - theta * (X - mean_x)
    df_clean["y_adjusted"] = y_adjusted

    # 3. Variance comparison
    var_raw = float(np.var(Y, ddof=1))
    var_adj = float(np.var(y_adjusted, ddof=1))
    pct_var_reduction = ((var_raw - var_adj) / var_raw) * 100
    theoretical_reduction = (corr_xy ** 2) * 100

    # 4. Treatment effect estimation: Raw vs CUPED
    t_mask = (df_clean[treatment_col] == treatment_val)
    c_mask = (df_clean[treatment_col] == control_val)

    # Raw effect
    raw_mean_t = float(np.mean(Y[t_mask]))
    raw_mean_c = float(np.mean(Y[c_mask]))
    raw_effect = raw_mean_t - raw_mean_c
    raw_se = np.sqrt(np.var(Y[t_mask], ddof=1)/t_mask.sum() + np.var(Y[c_mask], ddof=1)/c_mask.sum())
    raw_t_stat, raw_pval = stats.ttest_ind(Y[t_mask], Y[c_mask], equal_var=False)

    # CUPED-adjusted effect
    adj_mean_t = float(np.mean(y_adjusted[t_mask]))
    adj_mean_c = float(np.mean(y_adjusted[c_mask]))
    adj_effect = adj_mean_t - adj_mean_c
    adj_se = np.sqrt(np.var(y_adjusted[t_mask], ddof=1)/t_mask.sum() + np.var(y_adjusted[c_mask], ddof=1)/c_mask.sum())
    adj_t_stat, adj_pval = stats.ttest_ind(y_adjusted[t_mask], y_adjusted[c_mask], equal_var=False)

    # Confidence intervals (95%)
    raw_ci = (raw_effect - 1.96 * raw_se, raw_effect + 1.96 * raw_se)
    adj_ci = (adj_effect - 1.96 * adj_se, adj_effect + 1.96 * adj_se)

    # Sample size reduction equivalent
    # N_new = N_old * (1 - rho^2)
    sample_size_factor = 1.0 - (corr_xy ** 2)
    duration_reduction_pct = (1.0 - sample_size_factor) * 100

    return {
        "theta": round(float(theta), 4),
        "correlation_rho": round(float(corr_xy), 4),
        "r_squared_pct": round(float(theoretical_reduction), 2),
        "variance_raw": round(var_raw, 2),
        "variance_cuped": round(var_adj, 2),
        "actual_variance_reduction_pct": round(pct_var_reduction, 2),
        "raw_treatment_effect": round(raw_effect, 2),
        "raw_se": round(float(raw_se), 3),
        "raw_ci_95": (round(float(raw_ci[0]), 2), round(float(raw_ci[1]), 2)),
        "raw_p_value": float(raw_pval),
        "cuped_treatment_effect": round(adj_effect, 2),
        "cuped_se": round(float(adj_se), 3),
        "cuped_ci_95": (round(float(adj_ci[0]), 2), round(float(adj_ci[1]), 2)),
        "cuped_p_value": float(adj_pval),
        "standard_error_reduction_pct": round(((raw_se - adj_se) / raw_se) * 100, 2),
        "equivalent_sample_size_savings_pct": round(duration_reduction_pct, 2),
        "adjusted_dataframe": df_clean
    }


# =====================================================================
# 3. SUBGROUP HETEROGENEITY ENGINE
# =====================================================================

def evaluate_subgroup_effects(
    df: pd.DataFrame,
    subgroup_col: str,
    outcome_col: str,
    treatment_col: str = "treatment_assignment",
    treatment_val: str = "Treatment",
    control_val: str = "Control",
    metric_type: str = "proportion"
) -> pd.DataFrame:
    """
    Estimates treatment effects, confidence intervals, and p-values across distinct subgroups.
    Supports proportion (binary) outcomes and continuous outcomes.
    """
    records = []
    unique_subgroups = [s for s in df[subgroup_col].dropna().unique()]

    for sg in sorted(unique_subgroups):
        sub_df = df[df[subgroup_col] == sg]
        t_sub = sub_df[sub_df[treatment_col] == treatment_val][outcome_col].dropna().values
        c_sub = sub_df[sub_df[treatment_col] == control_val][outcome_col].dropna().values

        n_t, n_c = len(t_sub), len(c_sub)
        if n_t < 10 or n_c < 10:
            continue

        if metric_type == "proportion":
            succ_t = int(np.sum(t_sub))
            succ_c = int(np.sum(c_sub))
            p_t = succ_t / n_t
            p_c = succ_c / n_c
            effect = p_t - p_c
            se = np.sqrt((p_t * (1 - p_t) / n_t) + (p_c * (1 - p_c) / n_c))
            ci_low = effect - 1.96 * se
            ci_high = effect + 1.96 * se
            z_stat = effect / se if se > 0 else 0.0
            p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            rel_effect = (effect / p_c) * 100 if p_c > 0 else np.nan

            records.append({
                "subgroup_dimension": subgroup_col,
                "subgroup_value": str(sg),
                "control_n": n_c,
                "treatment_n": n_t,
                "control_metric": round(p_c, 4),
                "treatment_metric": round(p_t, 4),
                "treatment_effect": round(effect, 4),
                "relative_effect_pct": round(rel_effect, 2),
                "standard_error": round(se, 4),
                "ci_lower_95": round(ci_low, 4),
                "ci_upper_95": round(ci_high, 4),
                "test_statistic": round(z_stat, 3),
                "p_value": float(p_val),
                "significant": bool(p_val < 0.05)
            })
        else:
            mean_t = float(np.mean(t_sub))
            mean_c = float(np.mean(c_sub))
            effect = mean_t - mean_c
            t_stat, p_val = stats.ttest_ind(t_sub, c_sub, equal_var=False)
            se = np.sqrt(np.var(t_sub, ddof=1)/n_t + np.var(c_sub, ddof=1)/n_c)
            ci_low = effect - 1.96 * se
            ci_high = effect + 1.96 * se
            rel_effect = (effect / mean_c) * 100 if mean_c > 0 else np.nan

            records.append({
                "subgroup_dimension": subgroup_col,
                "subgroup_value": str(sg),
                "control_n": n_c,
                "treatment_n": n_t,
                "control_metric": round(mean_c, 2),
                "treatment_metric": round(mean_t, 2),
                "treatment_effect": round(effect, 2),
                "relative_effect_pct": round(rel_effect, 2),
                "standard_error": round(se, 3),
                "ci_lower_95": round(ci_low, 2),
                "ci_upper_95": round(ci_high, 2),
                "test_statistic": round(float(t_stat), 3),
                "p_value": float(p_val),
                "significant": bool(p_val < 0.05)
            })

    return pd.DataFrame(records)
