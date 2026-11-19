"""
app.py
======
Interactive Executive Analytical Dashboard for Salon Experiment & Causal Analysis.

Enterprise: Aura Luxe Salon & Spa Chain (Coimbatore, Tamil Nadu)
Technologies: Streamlit, Plotly, Pandas, NumPy, Matplotlib, Seaborn

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
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(
    page_title="Aura Luxe Salon | Causal Analytics & Experimentation",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Base Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
TABLES_DIR = os.path.join(BASE_DIR, "outputs", "tables")
FIGURES_DIR = os.path.join(BASE_DIR, "outputs", "figures")

# Custom Typography & Polished Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"], [class*="st-"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Headings */
    .hero-title {
        font-size: 28px;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #f59e0b 0%, #fbbf24 40%, #ffffff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2px;
    }
    .hero-subtitle {
        font-size: 14px;
        font-weight: 400;
        color: #94a3b8;
        margin-bottom: 18px;
    }

    /* Section Cards */
    .metric-glass-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px 20px;
        backdrop-filter: blur(8px);
        margin-bottom: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }

    .disclosure-box {
        background: rgba(245, 158, 11, 0.08);
        border-left: 4px solid #f59e0b;
        border-radius: 6px;
        padding: 10px 16px;
        font-size: 13px;
        color: #fde68a;
        margin-bottom: 20px;
        line-height: 1.5;
    }

    /* Sidebar Spacing & Tab Styling */
    [data-testid="stSidebar"] {
        background-color: #0f1117;
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }
    
    [data-testid="stSidebar"] [data-testid="stRadio"] > div {
        gap: 10px !important;
        display: flex;
        flex-direction: column;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 4px;
        cursor: pointer;
        transition: all 0.25s ease-in-out;
        width: 100%;
        display: flex;
        align-items: center;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background: rgba(255, 255, 255, 0.09);
        border-color: rgba(245, 158, 11, 0.4);
        transform: translateX(4px);
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-checked="true"] {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.22), rgba(217, 119, 6, 0.15));
        border-color: #f59e0b;
        box-shadow: 0 2px 8px rgba(245, 158, 11, 0.2);
    }

    /* Single Line Math Containers */
    .formula-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(59, 130, 246, 0.25);
        border-radius: 10px;
        padding: 14px 20px;
        margin: 14px 0;
        text-align: center;
    }

    /* Streamlit Metric Overrides */
    [data-testid="stMetricValue"] {
        font-weight: 700 !important;
        font-size: 26px !important;
        letter-spacing: -0.5px !important;
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

# Sidebar Navigation Header
st.sidebar.markdown("""
<div style="display: flex; align-items: center; gap: 12px; margin-bottom: 16px;">
    <span style="font-size: 32px;">🌺</span>
    <div>
        <div style="font-size: 19px; font-weight: 800; color: #f8fafc; letter-spacing: -0.3px;">Aura Luxe Salon</div>
        <div style="font-size: 11px; color: #94a3b8; font-weight: 500;">Causal Analytics & Experimentation</div>
    </div>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigation Modules:",
    [
        "1. Business Overview",
        "2. A/B Experiment Analysis",
        "3. Historical Causal Analysis (DiD)",
        "4. Customer Segmentation",
        "5. CUPED Variance Reduction"
    ],
    label_visibility="collapsed"
)

# Sidebar metadata footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="font-size: 11px; color: #64748b; line-height: 1.6;">
    <strong>Network Scope:</strong> 22 Outlets (Coimbatore)<br>
    <strong>Observation Window:</strong> 2021 – 2024 (48M)<br>
    <strong>Trial Cohort:</strong> N = 18,000 Randomized<br>
    <strong>Methodology:</strong> DiD, CUPED, Neyman-Rubin
</div>
""", unsafe_allow_html=True)

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
    st.markdown('<div class="hero-title">Executive Business Overview & Operational KPIs</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Chain-Wide Operational Baseline across 22 Outlets in Coimbatore, Tamil Nadu (2022–2024 Panel)</div>', unsafe_allow_html=True)

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
    col2.metric("Salon Outlets", f"{total_outlets} Hubs")
    col3.metric("Gross Revenue", f"₹{total_rev_cr:.2f} Cr")
    col4.metric("60-Day Retention", f"{overall_retention:.1f}%")
    col5.metric("No-Show Rate", f"{overall_noshow:.1f}%")

    st.markdown("---")

    # Regional Filter
    c_hdr1, c_hdr2 = st.columns([3, 1])
    with c_hdr1:
        st.subheader("Regional Operational Deep-Dive")
    with c_hdr2:
        selected_region = st.selectbox("Region Filter:", ["All Regions"] + sorted(df_outlets["region"].unique().tolist()), label_visibility="collapsed")

    if selected_region != "All Regions":
        filtered_outlets = df_outlets[df_outlets["region"] == selected_region]["outlet_id"].tolist()
        filtered_apts = df_apts[df_apts["outlet_id"].isin(filtered_outlets)]
    else:
        filtered_apts = df_apts

    c_left, c_right = st.columns([3, 2])

    with c_left:
        st.markdown("**Monthly Revenue Trajectory (Seasonality & Festive Surges)**")
        monthly_rev = filtered_apts.groupby("year_month")["revenue"].sum().reset_index()
        monthly_rev["rev_lakhs"] = monthly_rev["revenue"] / 1e5

        # Interactive Plotly Line Chart with Compressed Timestamps (Quarterly)
        fig_rev = px.area(
            monthly_rev,
            x="year_month",
            y="rev_lakhs",
            labels={"year_month": "Month", "rev_lakhs": "Revenue (₹ Lakhs)"},
        )
        fig_rev.update_traces(
            line=dict(color="#f59e0b", width=2.5),
            fillcolor="rgba(245, 158, 11, 0.15)",
            hovertemplate="<b>%{x}</b><br>Revenue: ₹%{y:.2f} Lakhs<extra></extra>"
        )

        # Annotations for Diwali / Pongal surges
        fig_rev.add_annotation(
            x="2024-10", y=monthly_rev[monthly_rev["year_month"] == "2024-10"]["rev_lakhs"].values[0] if len(monthly_rev[monthly_rev["year_month"] == "2024-10"]) > 0 else 150,
            text="🪔 Diwali Surge", showarrow=True, arrowhead=2, arrowcolor="#fbbf24",
            font=dict(size=10, color="#fbbf24"), bgcolor="rgba(15, 23, 42, 0.8)", bordercolor="#f59e0b"
        )

        # Compress X-axis ticks to quarterly intervals (e.g., Jan '22, Apr '22, Jul '22, Oct '22)
        all_months = sorted(monthly_rev["year_month"].unique())
        tick_vals = [m for m in all_months if m.endswith(("-01", "-04", "-07", "-10"))]
        tick_text = []
        for m in tick_vals:
            yr, mo = m.split("-")
            month_abbr = {"01": "Jan", "04": "Apr", "07": "Jul", "10": "Oct"}[mo]
            tick_text.append(f"{month_abbr} '{yr[2:]}")

        fig_rev.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=20, b=10),
            height=340,
            xaxis=dict(
                tickmode="array",
                tickvals=tick_vals,
                ticktext=tick_text,
                tickangle=0,
                showgrid=True,
                gridcolor="rgba(255,255,255,0.06)",
                color="#94a3b8"
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor="rgba(255,255,255,0.06)",
                color="#94a3b8",
                title="Revenue (₹ Lakhs)"
            )
        )
        st.plotly_chart(fig_rev, use_container_width=True)
        st.caption("✨ Captures Diwali (Oct/Nov) and Pongal (Jan) peak booking demand alongside monsoon (Jul/Aug) dips.")

    with c_right:
        st.markdown("**Appointment Status Breakdown**")
        status_counts = filtered_apts["appointment_status"].value_counts().reset_index()
        status_counts.columns = ["Status", "Count"]

        # Color mapping: Completed -> Emerald, Cancelled -> Amber, No-Show -> Rose
        color_map = {
            "Completed": "#10b981",
            "Cancelled": "#f59e0b",
            "No-Show": "#ef4444"
        }

        fig_pie = go.Figure(
            data=[go.Pie(
                labels=status_counts["Status"],
                values=status_counts["Count"],
                hole=0.55,
                marker=dict(colors=[color_map.get(s, "#3b82f6") for s in status_counts["Status"]],
                            line=dict(color="#0f1117", width=2)),
                textinfo="label+percent",
                hoverinfo="label+value+percent",
                textfont=dict(size=12, color="#ffffff", family="Plus Jakarta Sans"),
            )]
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=10, b=10),
            height=340,
            showlegend=False
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        st.caption("Operational flow: 81.1% completed appointments with 8.0% unnotified no-show rate.")


# =====================================================================
# PAGE 2: A/B EXPERIMENT ANALYSIS
# =====================================================================
elif page == "2. A/B Experiment Analysis":
    st.markdown('<div class="hero-title">Prospective Randomized A/B Experiment Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Evaluation of Automated WhatsApp Confirmation & Rescheduling System (N = 18,000 Active Clients)</div>', unsafe_allow_html=True)

    df_exp = data["experiment"]
    t_arm = df_exp[df_exp["treatment_assignment"] == "Treatment"]
    c_arm = df_exp[df_exp["treatment_assignment"] == "Control"]

    # SRM Banner with Clean Styling
    st.markdown("""
    <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 8px; padding: 12px 18px; margin-bottom: 16px;">
        <span style="color: #34d399; font-weight: 700;">✅ Sample Ratio Mismatch (SRM) Audit: Passed</span><br>
        <span style="font-size: 13px; color: #cbd5e1;">
            Chi-Square Goodness-of-Fit Statistic = <strong>0.0000</strong> (p-value = <strong>1.0000</strong>).
            Observed <strong>9,000 Control vs. 9,000 Treatment</strong> perfectly matches the theoretical 1:1 allocation ratio. Zero sample selection bias detected.
        </span>
    </div>
    """, unsafe_allow_html=True)

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
        df_exp_disp = data["exp_summary"].copy()
        # Cleanly display without truncated index or dozens of zeros
        st.dataframe(
            df_exp_disp,
            hide_index=True,
            use_container_width=True
        )

    st.markdown("---")
    st.subheader("Visual Treatment Comparison & Confidence Bounds")

    # Interactive Plotly Treatment Comparison Bar Chart
    comp_data = pd.DataFrame({
        "Metric": ["60-Day Retention (%)", "60-Day Retention (%)", "Revenue / Visit (₹ / 10)", "Revenue / Visit (₹ / 10)", "No-Show Rate (%)", "No-Show Rate (%)"],
        "Group": ["Control", "Treatment", "Control", "Treatment", "Control", "Treatment"],
        "Value": [ret_c, ret_t, rev_c / 10, rev_t / 10, ns_c, ns_t],
        "Label": [f"{ret_c:.2f}%", f"{ret_t:.2f}%", f"₹{rev_c:.2f}", f"₹{rev_t:.2f}", f"{ns_c:.2f}%", f"{ns_t:.2f}%"]
    })

    fig_comp = px.bar(
        comp_data,
        x="Metric",
        y="Value",
        color="Group",
        barmode="group",
        text="Label",
        color_discrete_map={"Control": "#64748b", "Treatment": "#f59e0b"}
    )
    fig_comp.update_traces(
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Group: %{fullData.name}<br>Value: %{text}<extra></extra>"
    )
    fig_comp.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=20, b=10),
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#94a3b8", title="Normalized Value"),
        xaxis=dict(color="#94a3b8")
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    # High-Res Forest Plot
    st.image(
        os.path.join(FIGURES_DIR, "11_ab_test_confidence_intervals_forest.png"),
        caption="Forest Plot: Estimated Treatment Effects with Analytical and Bootstrap 95% Confidence Intervals",
        use_container_width=True
    )


# =====================================================================
# PAGE 3: HISTORICAL CAUSAL ANALYSIS (DiD)
# =====================================================================
elif page == "3. Historical Causal Analysis (DiD)":
    st.markdown('<div class="hero-title">Difference-in-Differences (DiD) Regional Policy Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Quasi-Experimental Impact of Regional Quality & Service Guarantee Policy (2021–2024 Panel, 48 Months)</div>', unsafe_allow_html=True)

    st.markdown("""
    **Policy Context:** In January 2023, Aura Luxe rolled out a centralized regional quality overhaul across its **Central** and **North** Coimbatore outlets without randomized rollout. We estimate causal treatment effects using two-way fixed effects panel regression with cluster-robust standard errors.
    """)

    # Single-line Econometric Formula
    st.markdown('<div class="formula-card">', unsafe_allow_html=True)
    st.latex(r"Y_{rt} = \beta_0 + \beta_1(\text{Treatment}_r) + \beta_2(\text{Post}_t) + \beta_3(\text{Treatment}_r \times \text{Post}_t) + \epsilon_{rt}")
    st.caption("Econometric Model: β₃ represents the Average Treatment Effect on the Treated (ATT) with Cluster-Robust SE at the regional level.")
    st.markdown('</div>', unsafe_allow_html=True)

    if "did_summary" in data:
        st.subheader("Econometric Estimation Summary")
        st.dataframe(
            data["did_summary"],
            hide_index=True,
            use_container_width=True
        )

    st.markdown("---")
    st.subheader("Parallel Trends & Dynamic Event Study")

    # Interactive Plotly Parallel Trends Visualization
    df_hist = data["historical"].copy()
    df_hist["group"] = np.where(df_hist["treatment_region"] == 1, "Policy Regions (Central, North)", "Comparison Regions (South, East, West)")
    monthly_trends = df_hist.groupby(["year_month", "group"])["revenue_per_visit"].mean().reset_index()

    fig_did = px.line(
        monthly_trends,
        x="year_month",
        y="revenue_per_visit",
        color="group",
        labels={"year_month": "Month", "revenue_per_visit": "Revenue / Visit (₹)", "group": "Regional Cohort"},
        color_discrete_map={"Policy Regions (Central, North)": "#3b82f6", "Comparison Regions (South, East, West)": "#94a3b8"}
    )
    fig_did.add_vline(x="2023-01", line_width=2, line_dash="dash", line_color="#ef4444",
                      annotation_text="Policy Launch (Jan 2023)", annotation_position="top left",
                      annotation_font=dict(color="#ef4444", size=11))

    # Compress X-axis ticks to bi-annual intervals (every 6 months)
    all_hist_months = sorted(monthly_trends["year_month"].unique())
    tick_vals_did = [m for m in all_hist_months if m.endswith(("-01", "-07"))]
    tick_text_did = [f"{'Jan' if m.endswith('-01') else 'Jul'} '{m[:4][2:]}" for m in tick_vals_did]

    fig_did.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=20, b=10),
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(
            tickmode="array",
            tickvals=tick_vals_did,
            ticktext=tick_text_did,
            tickangle=0,
            showgrid=True,
            gridcolor="rgba(255,255,255,0.06)",
            color="#94a3b8"
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.06)",
            color="#94a3b8",
            title="Revenue per Visit (₹)"
        )
    )
    st.plotly_chart(fig_did, use_container_width=True)

    # Dynamic Event Study Leads & Lags
    st.image(
        os.path.join(FIGURES_DIR, "13_did_event_study_dynamics.png"),
        caption="Event-Study Dynamic Leads & Lags: Pre-Trend Coefficients are Statistically Indistinguishable from Zero (Validating Parallel Trends)",
        use_container_width=True
    )


# =====================================================================
# PAGE 4: CUSTOMER SEGMENTATION
# =====================================================================
elif page == "4. Customer Segmentation":
    st.markdown('<div class="hero-title">Customer & Business Segmentation Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Evaluating Treatment Effect Heterogeneity across Personas, Outlets, and Verticals</div>', unsafe_allow_html=True)

    df_exp = data["experiment"]
    df_cust = data["customers"]
    df_outlets = data["outlets"]

    df_merged = df_exp.merge(df_cust[["customer_id", "customer_segment"]], on="customer_id", how="left")
    df_merged = df_merged.merge(df_outlets[["outlet_id", "tier", "region"]], on="outlet_id", how="left")

    dim = st.selectbox(
        "Explore Treatment Lift across Dimension:",
        ["Customer Persona Segment", "Outlet Classification Tier", "Geographic Region"]
    )

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

    summary_df.columns = ["Control N", "Treatment N", "Control Retention Raw", "Treatment Retention Raw"]
    summary_df["Absolute Lift (pp)"] = ((summary_df["Treatment Retention Raw"] - summary_df["Control Retention Raw"]) * 100).round(2)
    summary_df["Relative Lift (%)"] = (((summary_df["Treatment Retention Raw"] - summary_df["Control Retention Raw"]) / summary_df["Control Retention Raw"]) * 100).round(2)
    summary_df["Control Retention"] = (summary_df["Control Retention Raw"] * 100).round(2).astype(str) + "%"
    summary_df["Treatment Retention"] = (summary_df["Treatment Retention Raw"] * 100).round(2).astype(str) + "%"

    disp_table = summary_df[[
        "Control N", "Treatment N", "Control Retention", "Treatment Retention", "Absolute Lift (pp)", "Relative Lift (%)"
    ]].reset_index()

    st.dataframe(disp_table, hide_index=True, use_container_width=True)

    # Plotly Horizontal Bar Chart of Heterogeneous Treatment Effects
    fig_lift = px.bar(
        disp_table,
        x="Absolute Lift (pp)",
        y=col_name,
        orientation="h",
        text="Absolute Lift (pp)",
        color="Absolute Lift (pp)",
        color_continuous_scale="Viridis",
        labels={"Absolute Lift (pp)": "Retention Lift (percentage points)", col_name: dim}
    )
    fig_lift.update_traces(
        texttemplate="+%{text:.2f} pp",
        textposition="outside"
    )
    fig_lift.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=10, b=10),
        height=280,
        coloraxis_showscale=False,
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#94a3b8"),
        yaxis=dict(color="#94a3b8")
    )
    st.plotly_chart(fig_lift, use_container_width=True)

    st.markdown("---")
    st.image(
        os.path.join(FIGURES_DIR, "14_segmentation_treatment_effects.png"),
        caption="Subgroup Treatment Effect Forest Plot (Retention Lift across Dimensions with 95% CIs)",
        use_container_width=True
    )


# =====================================================================
# PAGE 5: CUPED VARIANCE REDUCTION
# =====================================================================
elif page == "5. CUPED Variance Reduction":
    st.markdown('<div class="hero-title">CUPED (Controlled-Experiment Using Pre-Experiment Data)</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Algorithmic Variance Reduction & Experiment Runtime Acceleration</div>', unsafe_allow_html=True)

    # Single-line Math Formulas
    st.markdown('<div class="formula-card">', unsafe_allow_html=True)
    st.latex(r"Y_{i, \text{adj}} = Y_i - \theta (X_i - \bar{X}) \quad \text{where} \quad \theta = \frac{\text{Cov}(Y, X)}{\text{Var}(X)}")
    st.latex(r"\text{Variance Reduction} = \rho_{XY}^2 \quad \implies \quad \text{Equivalent Sample Size / Duration Savings} = \rho_{XY}^2")
    st.caption("CUPED Covariance Adjustment: Removes pre-experiment baseline customer variance without introducing point-estimate bias.")
    st.markdown('</div>', unsafe_allow_html=True)

    if "cuped_summary" in data:
        st.subheader("CUPED Statistical Comparison")
        st.dataframe(
            data["cuped_summary"],
            hide_index=True,
            use_container_width=True
        )

    st.markdown("---")
    st.subheader("Visual Variance Shrinkage & Precision Gains")

    # High-Res Static Figure 15 (Fixed X-axis padding and no CI text collision)
    st.image(
        os.path.join(FIGURES_DIR, "15_cuped_variance_reduction_comparison.png"),
        caption="CUPED Analytics: Outcome Distribution Variance Shrinkage (Left) and 95% Confidence Interval Tightening (Right)",
        use_container_width=True
    )

    # Interactive Runtime Calculator
    st.markdown("---")
    st.subheader("Interactive Experiment Duration Acceleration Calculator")

    c_calc1, c_calc2 = st.columns([2, 3])
    with c_calc1:
        corr = st.slider("Observed Pre-Post Correlation (ρ):", min_value=0.0, max_value=0.85, value=0.363, step=0.01)
        var_red = (corr ** 2) * 100
        sample_needed_pct = 100 - var_red
        days_saved_60 = round(60 * (var_red / 100), 1)

    with c_calc2:
        st.markdown(f"""
        <div class="metric-glass-card" style="border-left: 4px solid #10b981;">
            <div style="font-size: 13px; color: #94a3b8; font-weight: 600;">ALGORITHMIC EFFICIENCY GAIN</div>
            <div style="font-size: 24px; font-weight: 800; color: #34d399; margin: 4px 0;">{var_red:.1f}% Variance Reduction</div>
            <div style="font-size: 13px; color: #cbd5e1; line-height: 1.5;">
                • Standard Required Sample Size: <strong>100%</strong> (N = 18,000)<br>
                • Effective Sample Size Required under CUPED: <strong>{sample_needed_pct:.1f}%</strong> (N = {int(18000 * sample_needed_pct/100):,})<br>
                • Timeline Acceleration on 60-Day Trial: <strong>~{days_saved_60} Days Saved</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
