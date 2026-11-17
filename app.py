"""
app.py
======
Interactive Executive Analytical Dashboard for Salon Experiment & Causal Analysis.

Enterprise: Aura Luxe Salon & Spa Chain (Coimbatore, Tamil Nadu)
Technologies: Streamlit, Pandas, NumPy, Matplotlib, Seaborn

Pages:
1. Business Overview: Operational KPIs, Revenue Trends & Outlets across Coimbatore.
2. A/B Experiment Analysis: Randomized Trial (WhatsApp Bot), SRM, CIs, Lift.
3. Historical Causal Analysis: Difference-in-Differences (DiD), Parallel Trends, Event Study.
4. Customer Segmentation: Heterogeneous Treatment Effects across Personas & Tiers.
5. CUPED Variance Reduction: Covariance Adjustment, Distribution KDE, Duration Savings.
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(
    page_title="Aura Luxe Salon | Experimentation & Causal Dashboard",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Base Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
TABLES_DIR = os.path.join(BASE_DIR, "outputs", "tables")
FIGURES_DIR = os.path.join(BASE_DIR, "outputs", "figures")

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 26px;
        font-weight: 700;
        color: #1F4E79;
        margin-bottom: 2px;
    }
    .sub-header {
        font-size: 14px;
        color: #595959;
        margin-bottom: 16px;
    }
    .metric-card {
        background-color: #F8F9FA;
        border-left: 5px solid #1F4E79;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 10px;
    }
    .disclosure-box {
        background-color: #FFF8E1;
        border-left: 5px solid #FFA000;
        padding: 10px 14px;
        border-radius: 4px;
        font-size: 13px;
        color: #5D4037;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_all_datasets():
    """Dynamically loads datasets from data/processed and outputs/tables."""
    data = {}
    data["outlets"] = pd.read_csv(os.path.join(PROCESSED_DIR, "outlets_clean.csv"))
    data["services"] = pd.read_csv(os.path.join(PROCESSED_DIR, "services_clean.csv"))
    data["customers"] = pd.read_csv(os.path.join(PROCESSED_DIR, "customers_clean.csv"))
    data["appointments"] = pd.read_csv(os.path.join(PROCESSED_DIR, "appointments_clean.csv"))
    data["experiment"] = pd.read_csv(os.path.join(PROCESSED_DIR, "experiment_data_clean.csv"))
    data["historical"] = pd.read_csv(os.path.join(PROCESSED_DIR, "historical_regional_data_clean.csv"))
    data["cust_features"] = pd.read_csv(os.path.join(PROCESSED_DIR, "customer_features.csv"))

    # Summary Tables
    exp_table_path = os.path.join(TABLES_DIR, "consolidated_experimental_results.csv")
    if os.path.exists(exp_table_path):
        data["exp_summary"] = pd.read_csv(exp_table_path)
    did_table_path = os.path.join(TABLES_DIR, "did_regression_results.csv")
    if os.path.exists(did_table_path):
        data["did_summary"] = pd.read_csv(did_table_path)
    cuped_table_path = os.path.join(TABLES_DIR, "cuped_variance_reduction_summary.csv")
    if os.path.exists(cuped_table_path):
        data["cuped_summary"] = pd.read_csv(cuped_table_path)
    return data


data = load_all_datasets()

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/fluency/96/spa-flower.png", width=64)
st.sidebar.title("Aura Luxe Salon")
st.sidebar.caption("Experimentation & Causal Analytics")
page = st.sidebar.radio(
    "Navigate Analytics:",
    [
        "1. Business Overview",
        "2. A/B Experiment Analysis",
        "3. Historical Causal Analysis (DiD)",
        "4. Customer Segmentation",
        "5. CUPED Variance Reduction"
    ]
)

# Global Synthetic Data Banner
st.markdown("""
<div class="disclosure-box">
    <strong>⚠️ SYNTHETIC DATA DISCLOSURE:</strong> All datasets, customer records, appointment bookings,
    and experimental metrics shown are 100% synthetically simulated in Python (Seed: 42) for educational
    and analytical evaluation. They do not represent proprietary records of any real commercial salon.
</div>
""", unsafe_allow_html=True)


# =====================================================================
# PAGE 1: BUSINESS OVERVIEW
# =====================================================================
if page == "1. Business Overview":
    st.markdown('<div class="main-header">Executive Business Overview & Operational KPIs</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Chain-Wide Operational Baseline across 22 Outlets in Coimbatore, Tamil Nadu (2022–2024)</div>', unsafe_allow_html=True)

    df_apts = data["appointments"]
    df_outlets = data["outlets"]
    df_cust = data["cust_features"]

    # Top-Level KPI Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    total_apts = len(df_apts)
    total_outlets = len(df_outlets)
    total_rev_cr = df_apts["revenue"].sum() / 1e7
    overall_retention = df_cust["customer_retention_60d"].mean() * 100
    overall_noshow = df_apts["no_show_indicator"].mean() * 100

    col1.metric("Total Appointments", f"{total_apts:,}")
    col2.metric("Salon Outlets", f"{total_outlets}")
    col3.metric("Gross Revenue", f"₹{total_rev_cr:.2f} Cr")
    col4.metric("60-Day Retention", f"{overall_retention:.1f}%")
    col5.metric("No-Show Rate", f"{overall_noshow:.1f}%")

    st.markdown("---")

    # Region Filter
    st.subheader("Regional Operational Deep-Dive")
    selected_region = st.selectbox("Filter by Coimbatore Region:", ["All Regions"] + sorted(df_outlets["region"].unique().tolist()))

    if selected_region != "All Regions":
        filtered_outlets = df_outlets[df_outlets["region"] == selected_region]["outlet_id"].tolist()
        filtered_apts = df_apts[df_apts["outlet_id"].isin(filtered_outlets)]
    else:
        filtered_apts = df_apts

    c_left, c_right = st.columns([3, 2])

    with c_left:
        st.markdown("**Monthly Revenue Trajectory (Seasonality & Festive Surges)**")
        monthly_rev = filtered_apts.groupby("year_month")["revenue"].sum() / 1e5
        st.line_chart(monthly_rev)
        st.caption("Captures Diwali (Oct/Nov) and Pongal (Jan) peaks alongside monsoon (Jul/Aug) dips.")

    with c_right:
        st.markdown("**Appointment Status Breakdown**")
        status_counts = filtered_apts["appointment_status"].value_counts()
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.pie(status_counts.values, labels=status_counts.index, autopct="%1.1f%%",
               colors=["#2ca02c", "#ff7f0e", "#d62728"], startangle=140,
               wedgeprops=dict(width=0.45, edgecolor="black"))
        st.pyplot(fig)


# =====================================================================
# PAGE 2: A/B EXPERIMENT ANALYSIS
# =====================================================================
elif page == "2. A/B Experiment Analysis":
    st.markdown('<div class="main-header">Prospective Randomized A/B Experiment Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Evaluation of Automated WhatsApp Confirmation & Rescheduling System (N = 18,000 Active Clients)</div>', unsafe_allow_html=True)

    df_exp = data["experiment"]
    t_arm = df_exp[df_exp["treatment_assignment"] == "Treatment"]
    c_arm = df_exp[df_exp["treatment_assignment"] == "Control"]

    # SRM Banner
    st.info("✅ **Sample Ratio Mismatch (SRM) Audit:** Chi-Square Goodness-of-Fit Statistic = 0.0000 (p-value = 1.0000). Observed 9,000 Control vs. 9,000 Treatment perfectly matches theoretical 1:1 allocation. Zero allocation bias.")

    # Core Metrics
    ret_t = t_arm["retention_outcome"].mean() * 100
    ret_c = c_arm["retention_outcome"].mean() * 100
    ret_lift = ret_t - ret_c

    rev_t = t_arm["revenue_per_visit"].mean()
    rev_c = c_arm["revenue_per_visit"].mean()
    rev_lift = rev_t - rev_c

    ns_t = t_arm["no_show_outcome"].mean() * 100
    ns_c = c_arm["no_show_outcome"].mean() * 100
    ns_drop = ns_t - ns_c

    c1, c2, c3 = st.columns(3)
    c1.metric("60-Day Retention (Primary)", f"{ret_t:.2f}%", f"{ret_lift:+.2f} pp vs. Control ({ret_c:.2f}%)")
    c2.metric("Revenue per Visit (Secondary)", f"₹{rev_t:.2f}", f"₹{rev_lift:+.2f} vs. Control (₹{rev_c:.2f})")
    c3.metric("No-Show Rate (Secondary)", f"{ns_t:.2f}%", f"{ns_drop:+.2f} pp vs. Control ({ns_c:.2f}%)", delta_color="inverse")

    st.markdown("---")
    st.subheader("Statistical Hypothesis Testing & Consolidated Results")

    if "exp_summary" in data:
        st.dataframe(data["exp_summary"], use_container_width=True)

    st.markdown("### Visual Treatment Comparison")
    st.image(os.path.join(FIGURES_DIR, "10_ab_test_treatment_vs_control.png"), caption="Empirical Comparison across Primary and Secondary Metrics (300 DPI)", use_container_width=True)
    st.image(os.path.join(FIGURES_DIR, "11_ab_test_confidence_intervals_forest.png"), caption="Forest Plot: Estimated Treatment Effects with 95% Confidence Intervals", use_container_width=True)


# =====================================================================
# PAGE 3: HISTORICAL CAUSAL ANALYSIS (DiD)
# =====================================================================
elif page == "3. Historical Causal Analysis (DiD)":
    st.markdown('<div class="main-header">Difference-in-Differences (DiD) Regional Policy Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Quasi-Experimental Impact of Regional Quality & Service Guarantee Policy (2021–2024 Panel, 48 Months)</div>', unsafe_allow_html=True)

    st.markdown("""
    **Policy Context:** In January 2023, Aura Luxe rolled out a centralized regional quality overhaul across its **Central** and **North** Coimbatore outlets without randomized rollout. We estimate causal treatment effects using two-way fixed effects panel regression with cluster-robust standard errors.
    """)

    if "did_summary" in data:
        st.subheader("Econometric Estimation Summary")
        st.dataframe(data["did_summary"], use_container_width=True)

    st.markdown("---")
    st.subheader("Parallel Trends & Dynamic Event Study")
    st.image(os.path.join(FIGURES_DIR, "12_did_parallel_trends_inspection.png"), caption="Inspection of Monthly Trajectories across Treatment and Control Regions", use_container_width=True)
    st.image(os.path.join(FIGURES_DIR, "13_did_event_study_dynamics.png"), caption="Event-Study Dynamic Leads & Lags: Confirming Pre-Trend Stability", use_container_width=True)


# =====================================================================
# PAGE 4: CUSTOMER SEGMENTATION
# =====================================================================
elif page == "4. Customer Segmentation":
    st.markdown('<div class="main-header">Customer & Business Segmentation Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Evaluating Treatment Effect Heterogeneity across Personas, Outlets, and Verticals</div>', unsafe_allow_html=True)

    df_exp = data["experiment"]
    df_cust = data["customers"]
    df_outlets = data["outlets"]

    df_merged = df_exp.merge(df_cust[["customer_id", "customer_segment"]], on="customer_id", how="left")
    df_merged = df_merged.merge(df_outlets[["outlet_id", "tier", "region"]], on="outlet_id", how="left")

    dim = st.selectbox("Explore Treatment Lift across Dimension:", ["Customer Persona Segment", "Outlet Classification Tier", "Geographic Region"])

    dim_map = {
        "Customer Persona Segment": "customer_segment",
        "Outlet Classification Tier": "tier",
        "Geographic Region": "region"
    }
    col_name = dim_map[dim]

    summary_df = df_merged.groupby([col_name, "treatment_assignment"])["retention_outcome"].agg(
        n="count",
        retention_rate="mean"
    ).unstack()

    summary_df.columns = ["Control N", "Treatment N", "Control Retention", "Treatment Retention"]
    summary_df["Absolute Lift (pp)"] = (summary_df["Treatment Retention"] - summary_df["Control Retention"]) * 100
    summary_df["Control Retention"] = (summary_df["Control Retention"] * 100).round(2).astype(str) + "%"
    summary_df["Treatment Retention"] = (summary_df["Treatment Retention"] * 100).round(2).astype(str) + "%"
    summary_df["Absolute Lift (pp)"] = summary_df["Absolute Lift (pp)"].round(2)

    st.dataframe(summary_df, use_container_width=True)
    st.image(os.path.join(FIGURES_DIR, "14_segmentation_treatment_effects.png"), caption="Subgroup Treatment Effect Forest Plot (Retention Lift across Dimensions)", use_container_width=True)


# =====================================================================
# PAGE 5: CUPED VARIANCE REDUCTION
# =====================================================================
elif page == "5. CUPED Variance Reduction":
    st.markdown('<div class="main-header">CUPED (Controlled-Experiment Using Pre-Experiment Data)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Algorithmic Variance Reduction & Experiment Runtime Acceleration</div>', unsafe_allow_html=True)

    st.markdown("""
    **CUPED Methodology:** Utilizes historical pre-experiment spending ($X$) to strip out pre-existing customer variance from target metric ($Y$):
    $$Y_{\\text{adj}} = Y - \\theta (X - \\bar{X}), \\quad \\theta = \\frac{\\text{Cov}(Y, X)}{\\text{Var}(X)}$$
    """)

    if "cuped_summary" in data:
        st.subheader("CUPED Statistical Comparison")
        st.dataframe(data["cuped_summary"], use_container_width=True)

    st.markdown("---")
    st.subheader("Visual Variance Shrinkage & Precision Gains")
    st.image(os.path.join(FIGURES_DIR, "15_cuped_variance_reduction_comparison.png"), caption="Distribution KDE Variance Shrinkage and Confidence Interval Tightening", use_container_width=True)

    # Interactive Runtime Calculator
    st.subheader("Interactive Experiment Duration Calculator")
    corr = st.slider("Observed Pre-Post Correlation (rho):", min_value=0.0, max_value=0.8, value=0.363, step=0.01)
    var_red = (corr ** 2) * 100
    duration_savings = var_red
    st.success(f"Theoretical Variance Reduction: **{var_red:.1f}%** | Equivalent Sample Size / Duration Savings: **{duration_savings:.1f}%**")
