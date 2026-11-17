"""
visualization.py
================
Senior Data Analytics & Visualization Module.

Project: Salon Experiment & Causal Analysis – Measuring the True Impact of Business Changes
Location: Coimbatore, Tamil Nadu (Aura Luxe Salon Chain)

Generates publication-quality figures (300 DPI) and tables documenting:
1. Appointment status distributions (completion, cancellation, no-show rates).
2. Outlet-wise appointment volume and revenue realizations across Coimbatore.
3. Revenue distributions across service categories and price tiers.
4. Customer demographic and behavioral segmentation.
5. Service category volume and revenue contribution.
6. Monthly revenue trends capturing seasonal dynamics (Diwali, Pongal, weddings).
7. Deep-dive into no-show drivers (channels, timing, stylist experience).
8. A/B experiment baseline covariate balance.
"""

import os
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/script execution
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
FIGURES_DIR = os.path.join(BASE_DIR, "outputs", "figures")
TABLES_DIR = os.path.join(BASE_DIR, "outputs", "tables")

# Styling configurations
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.labelweight"] = "semibold"
plt.rcParams["figure.titlesize"] = 16
plt.rcParams["figure.titleweight"] = "bold"

PRIMARY_COLOR = "#1f77b4"
SECONDARY_COLOR = "#ff7f0e"
SUCCESS_COLOR = "#2ca02c"
DANGER_COLOR = "#d62728"
ACCENT_PALETTE = ["#2b5c8f", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d"]


def plot_appointment_status_distribution(df_appointments: pd.DataFrame) -> str:
    """Plots distribution of appointment statuses (Completed, Cancelled, No-Show)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    status_counts = df_appointments["appointment_status"].value_counts()
    status_pct = (status_counts / len(df_appointments) * 100).round(2)
    colors = [SUCCESS_COLOR, SECONDARY_COLOR, DANGER_COLOR]

    # Bar chart
    bars = axes[0].bar(status_counts.index, status_counts.values, color=colors, edgecolor="black", alpha=0.85)
    axes[0].set_title("Appointment Status Volume (N = 300,000)")
    axes[0].set_ylabel("Number of Appointments")
    axes[0].grid(axis="y", linestyle="--", alpha=0.7)
    for bar in bars:
        h = bar.get_height()
        axes[0].annotate(f"{h:,}\n({h/len(df_appointments)*100:.1f}%)",
                         xy=(bar.get_x() + bar.get_width() / 2, h),
                         xytext=(0, 5), textcoords="offset points",
                         ha="center", va="bottom", fontsize=10, weight="bold")

    # Donut chart
    axes[1].pie(status_counts.values, labels=status_counts.index, autopct="%1.1f%%",
                startangle=140, colors=colors, explode=[0.02, 0.05, 0.08],
                wedgeprops=dict(width=0.45, edgecolor="black"))
    axes[1].set_title("Status Proportions (%)")

    plt.suptitle("Aura Luxe Salon Chain: Operational Appointment Status Breakdown", y=1.02)
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "01_appointment_status_distribution.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path


def plot_outlet_performance(df_outlets: pd.DataFrame, df_appointments: pd.DataFrame) -> str:
    """Plots appointment volume and gross revenue per outlet across Coimbatore."""
    fig, axes = plt.subplots(2, 1, figsize=(15, 12), sharex=True)

    outlet_agg = df_appointments.groupby("outlet_id").agg(
        volume=("appointment_id", "count"),
        revenue=("revenue", "sum")
    ).reset_index().merge(df_outlets[["outlet_id", "location", "region", "tier"]], on="outlet_id")

    outlet_agg = outlet_agg.sort_values(by="revenue", ascending=False)

    # Palette by region
    regions = outlet_agg["region"].unique()
    reg_colors = dict(zip(regions, sns.color_palette("Set2", len(regions))))
    bar_colors = [reg_colors[r] for r in outlet_agg["region"]]

    # Revenue Bar chart
    rev_crores = outlet_agg["revenue"] / 1e7
    bars1 = axes[0].bar(outlet_agg["location"], rev_crores, color=bar_colors, edgecolor="black", alpha=0.9)
    axes[0].set_title("Gross Service Revenue by Outlet (₹ in Crores)")
    axes[0].set_ylabel("Revenue (₹ Crores)")
    axes[0].grid(axis="y", linestyle="--", alpha=0.7)
    for bar in bars1:
        h = bar.get_height()
        axes[0].annotate(f"₹{h:.2f}Cr",
                         xy=(bar.get_x() + bar.get_width() / 2, h),
                         xytext=(0, 4), textcoords="offset points",
                         ha="center", va="bottom", fontsize=9, rotation=45)

    # Volume Bar chart
    bars2 = axes[1].bar(outlet_agg["location"], outlet_agg["volume"], color="#4a7c59", edgecolor="black", alpha=0.85)
    axes[1].set_title("Total Appointment Bookings by Outlet")
    axes[1].set_ylabel("Appointment Volume")
    axes[1].set_xlabel("Coimbatore Outlet Location")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].grid(axis="y", linestyle="--", alpha=0.7)

    plt.suptitle("Coimbatore Chain Performance: Revenue & Appointment Distribution Across 22 Outlets", y=1.01)
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "02_outlet_performance_rankings.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path


def plot_revenue_distributions(df_appointments: pd.DataFrame) -> str:
    """Plots distribution of completed appointment revenues across categories."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    completed = df_appointments[df_appointments["completion_indicator"] == 1]

    # Revenue histogram / KDE
    sns.histplot(completed["revenue"], bins=40, kde=True, ax=axes[0], color=PRIMARY_COLOR, edgecolor="black")
    axes[0].axvline(completed["revenue"].mean(), color=DANGER_COLOR, linestyle="--", linewidth=2,
                    label=f"Mean: ₹{completed['revenue'].mean():.1f}")
    axes[0].axvline(completed["revenue"].median(), color=SUCCESS_COLOR, linestyle="-.", linewidth=2,
                    label=f"Median: ₹{completed['revenue'].median():.1f}")
    axes[0].set_title("Completed Appointment Spend Distribution (INR)")
    axes[0].set_xlabel("Revenue (₹)")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()
    axes[0].set_xlim(0, 15000)

    # Boxplot by category
    sns.boxplot(data=completed, x="service_category", y="revenue", ax=axes[1], palette="Set3")
    axes[1].set_title("Revenue Dispersion by Service Category")
    axes[1].set_xlabel("Service Category")
    axes[1].set_ylabel("Revenue (₹)")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Revenue (₹) - Log Scale")

    plt.suptitle("Revenue Realization Characteristics: Spend Density & Vertical Dispersion", y=1.02)
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "03_revenue_distributions.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path


def plot_customer_segmentation(df_customers: pd.DataFrame, df_cust_features: pd.DataFrame) -> str:
    """Plots customer segmentation demographics, acquisition channels, and loyalty tiers."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # Age distribution by Gender
    sns.histplot(data=df_customers, x="age", hue="gender", multiple="stack", bins=28, ax=axes[0, 0], palette="muted")
    axes[0, 0].set_title("Customer Age Demographics by Gender")
    axes[0, 0].set_xlabel("Customer Age (Years)")
    axes[0, 0].set_ylabel("Customer Count")

    # Acquisition Source
    acq_counts = df_customers["acquisition_source"].value_counts()
    bars = axes[0, 1].barh(acq_counts.index, acq_counts.values, color="#3470a3", edgecolor="black", alpha=0.85)
    axes[0, 1].set_title("Customer Acquisition Channel Attribution")
    axes[0, 1].set_xlabel("Total Customers Acquired")
    axes[0, 1].grid(axis="x", linestyle="--", alpha=0.7)
    for bar in bars:
        w = bar.get_width()
        axes[0, 1].annotate(f"{w:,} ({w/len(df_customers)*100:.1f}%)",
                            xy=(w, bar.get_y() + bar.get_height() / 2),
                            xytext=(5, 0), textcoords="offset points",
                            va="center", fontsize=9, weight="semibold")

    # Customer Loyalty Tiers
    tier_order = ["One-Time Visitor", "Occasional Visitor", "Regular Client", "Loyal VIP", "Prospect / Inactive"]
    loyalty_counts = df_cust_features["loyalty_tier"].value_counts().reindex(tier_order).dropna()
    axes[1, 0].bar(loyalty_counts.index, loyalty_counts.values, color=sns.color_palette("Blues_r", len(loyalty_counts)), edgecolor="black")
    axes[1, 0].set_title("Customer Loyalty & Lifetime Engagement Tiers")
    axes[1, 0].set_ylabel("Customer Count")
    axes[1, 0].tick_params(axis="x", rotation=25)
    for i, v in enumerate(loyalty_counts.values):
        axes[1, 0].annotate(f"{v:,}", xy=(i, v), xytext=(0, 4), textcoords="offset points", ha="center", weight="bold")

    # Retention rate by Marketing Segment
    segment_retention = df_cust_features.groupby("customer_segment")["customer_retention_60d"].mean() * 100
    seg_bars = axes[1, 1].bar(segment_retention.index, segment_retention.values, color="#2ca02c", edgecolor="black", alpha=0.85)
    axes[1, 1].set_title("60-Day Repeat Retention Rate by Persona Segment (%)")
    axes[1, 1].set_ylabel("Retention Rate (%)")
    axes[1, 1].set_ylim(0, 100)
    for bar in seg_bars:
        h = bar.get_height()
        axes[1, 1].annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 4),
                            textcoords="offset points", ha="center", weight="bold")

    plt.suptitle("Customer Intelligence & Segmentation Analytics (N = 42,000 Customers)", y=1.01)
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "04_customer_segmentation.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path


def plot_service_category_breakdown(df_appointments: pd.DataFrame) -> str:
    """Plots service category appointment volume and revenue contributions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    cat_summary = df_appointments.groupby("service_category").agg(
        volume=("appointment_id", "count"),
        revenue=("revenue", "sum")
    ).reset_index().sort_values(by="revenue", ascending=False)

    colors = sns.color_palette("Set2", len(cat_summary))

    # Revenue Share Donut
    axes[0].pie(cat_summary["revenue"], labels=cat_summary["service_category"], autopct="%1.1f%%",
                startangle=140, colors=colors, wedgeprops=dict(width=0.45, edgecolor="black"))
    axes[0].set_title("Revenue Contribution Share by Vertical")

    # Volume vs Average Price Bar
    cat_summary["avg_spend"] = cat_summary["revenue"] / cat_summary["volume"]
    bars = axes[1].bar(cat_summary["service_category"], cat_summary["volume"], color=colors, edgecolor="black", alpha=0.85)
    axes[1].set_title("Booking Volume by Service Vertical")
    axes[1].set_ylabel("Appointment Count")
    axes[1].tick_params(axis="x", rotation=25)
    for bar in bars:
        h = bar.get_height()
        axes[1].annotate(f"{h:,}", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 4),
                         textcoords="offset points", ha="center", weight="bold", fontsize=9)

    plt.suptitle("Service Vertical Performance: Booking Volume vs Revenue Contribution", y=1.02)
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "05_service_category_analysis.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path


def plot_monthly_revenue_trends(df_appointments: pd.DataFrame) -> str:
    """Plots 3-year monthly revenue trend showing macro expansion and Tamil Nadu festive seasonality."""
    fig, ax = plt.subplots(figsize=(15, 6))

    monthly = df_appointments.groupby("year_month").agg(
        revenue=("revenue", "sum"),
        appointments=("appointment_id", "count")
    ).reset_index()

    monthly["revenue_lakhs"] = monthly["revenue"] / 1e5

    # Line plot
    ax.plot(monthly["year_month"], monthly["revenue_lakhs"], marker="o", color="#1f77b4", linewidth=2.5, label="Monthly Revenue (₹ Lakhs)")
    ax.set_title("Monthly Revenue Trajectory (2022–2024): Capturing Festive Seasonality & Growth")
    ax.set_xlabel("Year-Month")
    ax.set_ylabel("Monthly Revenue (₹ Lakhs)")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, linestyle="--", alpha=0.7)

    # Annotate Festive Peaks
    for idx, row in monthly.iterrows():
        ym = row["year_month"]
        rev = row["revenue_lakhs"]
        if ym in ["2022-10", "2023-11", "2024-10"]:
            ax.annotate("Diwali Peak", xy=(ym, rev), xytext=(0, 18), textcoords="offset points",
                        ha="center", fontsize=9, weight="bold", color="#d95f02",
                        arrowprops=dict(arrowstyle="->", color="#d95f02"))
        elif ym in ["2022-01", "2023-01", "2024-01"]:
            ax.annotate("Pongal", xy=(ym, rev), xytext=(0, 15), textcoords="offset points",
                        ha="center", fontsize=9, weight="bold", color="#2ca02c",
                        arrowprops=dict(arrowstyle="->", color="#2ca02c"))
        elif ym in ["2022-07", "2023-08", "2024-07"]:
            ax.annotate("Monsoon Dip", xy=(ym, rev), xytext=(0, -22), textcoords="offset points",
                        ha="center", fontsize=8, color="#7f7f7f",
                        arrowprops=dict(arrowstyle="->", color="#7f7f7f"))

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "06_monthly_revenue_trends.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path


def plot_noshow_deepdive(df_appointments: pd.DataFrame, df_stylists: pd.DataFrame) -> str:
    """Plots operational breakdown of no-show rates across booking channels, days, and stylist seniority."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # 1. No-Show rate by Booking Channel
    ch_noshow = (df_appointments.groupby("booking_channel")["no_show_indicator"].mean() * 100).sort_values(ascending=False)
    bars1 = axes[0, 0].bar(ch_noshow.index, ch_noshow.values, color="#e6550d", edgecolor="black", alpha=0.85)
    axes[0, 0].set_title("No-Show Rate by Booking Channel (%)")
    axes[0, 0].set_ylabel("No-Show Rate (%)")
    axes[0, 0].set_ylim(0, 20)
    for bar in bars1:
        h = bar.get_height()
        axes[0, 0].annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 4),
                            textcoords="offset points", ha="center", weight="bold")

    # 2. No-Show rate by Day of Week
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow_noshow = (df_appointments.groupby("day_of_week")["no_show_indicator"].mean() * 100).reindex(dow_order)
    axes[0, 1].plot(dow_noshow.index, dow_noshow.values, marker="s", color="#756bb1", linewidth=2.5)
    axes[0, 1].set_title("No-Show Rate by Day of Week (%)")
    axes[0, 1].set_ylabel("No-Show Rate (%)")
    axes[0, 1].tick_params(axis="x", rotation=30)
    axes[0, 1].grid(True, linestyle="--", alpha=0.7)
    for i, v in enumerate(dow_noshow.values):
        axes[0, 1].annotate(f"{v:.1f}%", xy=(i, v), xytext=(0, 6), textcoords="offset points", ha="center", weight="semibold")

    # 3. No-Show rate by Stylist Seniority
    df_merged_stylist = df_appointments.merge(df_stylists[["stylist_id", "experience_level"]], on="stylist_id", how="left")
    exp_order = ["Junior", "Mid-Level", "Senior", "Master Stylist"]
    exp_noshow = (df_merged_stylist.groupby("experience_level")["no_show_indicator"].mean() * 100).reindex(exp_order)
    bars3 = axes[1, 0].bar(exp_noshow.index, exp_noshow.values, color="#31a354", edgecolor="black", alpha=0.85)
    axes[1, 0].set_title("No-Show Rate by Stylist Experience Seniority (%)")
    axes[1, 0].set_ylabel("No-Show Rate (%)")
    axes[1, 0].set_ylim(0, 16)
    for bar in bars3:
        h = bar.get_height()
        axes[1, 0].annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 4),
                            textcoords="offset points", ha="center", weight="bold")

    # 4. Cancellation vs No-Show Comparison by Service Category
    cat_rates = df_appointments.groupby("service_category").agg(
        cancellation_rate=("cancellation_indicator", lambda s: s.mean() * 100),
        no_show_rate=("no_show_indicator", lambda s: s.mean() * 100)
    ).reset_index()
    x = np.arange(len(cat_rates))
    w = 0.35
    axes[1, 1].bar(x - w/2, cat_rates["cancellation_rate"], width=w, label="Cancellation Rate (%)", color="#fd8d3c", edgecolor="black")
    axes[1, 1].bar(x + w/2, cat_rates["no_show_rate"], width=w, label="No-Show Rate (%)", color="#de2d26", edgecolor="black")
    axes[1, 1].set_title("Cancellation vs. No-Show Rates by Service Category")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(cat_rates["service_category"], rotation=25)
    axes[1, 1].set_ylabel("Rate (%)")
    axes[1, 1].legend()

    plt.suptitle("Operational Risk Analytics: Deep-Dive into No-Show & Abandonment Drivers", y=1.01)
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "07_noshow_and_cancellation_analysis.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path


def plot_experiment_baseline_balance(df_experiment: pd.DataFrame) -> str:
    """Plots baseline covariate balance between Treatment and Control groups in A/B experiment."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    palette = {"Control": "#4a7c59", "Treatment": "#2b5c8f"}

    # Pre-experiment visits
    sns.boxplot(data=df_experiment, x="treatment_assignment", y="pre_experiment_visit_count", ax=axes[0], palette=palette)
    axes[0].set_title("Pre-Experiment Visit Count Balance")
    axes[0].set_xlabel("Experiment Arm")
    axes[0].set_ylabel("Pre-Trial Visits")

    # Pre-experiment revenue
    sns.boxplot(data=df_experiment, x="treatment_assignment", y="pre_experiment_revenue", ax=axes[1], palette=palette)
    axes[1].set_title("Pre-Experiment Spend Balance (INR)")
    axes[1].set_xlabel("Experiment Arm")
    axes[1].set_ylabel("Pre-Trial Revenue (₹)")

    # Outcome comparison: Retention rate
    ret_rates = (df_experiment.groupby("treatment_assignment")["retention_outcome"].mean() * 100).round(2)
    bars = axes[2].bar(ret_rates.index, ret_rates.values, color=["#4a7c59", "#2b5c8f"], edgecolor="black", width=0.45)
    axes[2].set_title("Observed Post-Trial Retention Lift (%)")
    axes[2].set_ylabel("60-Day Retention Rate (%)")
    axes[2].set_ylim(0, 80)
    for bar in bars:
        h = bar.get_height()
        axes[2].annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 4),
                         textcoords="offset points", ha="center", weight="bold", fontsize=11)

    plt.suptitle("Prospective Randomized A/B Experiment: Baseline Covariate Balance & Empirical Lift", y=1.02)
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "08_experiment_baseline_and_outcomes.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path


# =====================================================================
# PHASE 2: EXPERIMENTATION & CAUSAL INFERENCE VISUALIZATIONS
# =====================================================================

def plot_power_sensitivity(df_sensitivity: pd.DataFrame) -> str:
    """Plots power curves and sample size sensitivity across MDEs and power levels."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for pwr, grp in df_sensitivity.groupby("power"):
        ax.plot(grp["mde_absolute"] * 100, grp["n_per_group"], marker="o", linewidth=2.5,
                label=f"Power = {int(pwr*100)}% (beta = {1-pwr:.2f})")

    ax.axhline(9000, color="crimson", linestyle="--", linewidth=1.8, label="Enrolled Sample Size per Group (N = 9,000)")
    ax.set_title("Statistical Power Analysis: Required Sample Size vs. Minimum Detectable Effect (MDE)")
    ax.set_xlabel("Minimum Detectable Effect (MDE in Percentage Points)")
    ax.set_ylabel("Required Sample Size per Group (n)")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.6)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "09_power_analysis_sensitivity.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path


def plot_ab_test_treatment_vs_control(df_experiment: pd.DataFrame) -> str:
    """Plots comprehensive 3-panel comparison of Primary (Retention) and Secondary metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    palette = {"Control": "#4a7c59", "Treatment": "#2b5c8f"}

    # 1. Retention Rate (Primary)
    ret_summary = (df_experiment.groupby("treatment_assignment")["retention_outcome"].mean() * 100).round(2)
    bars1 = axes[0].bar(ret_summary.index, ret_summary.values, color=["#4a7c59", "#2b5c8f"], edgecolor="black", width=0.45)
    axes[0].set_title("Primary Metric: 60-Day Retention Rate (%)")
    axes[0].set_ylabel("Retention Rate (%)")
    axes[0].set_ylim(0, 80)
    for bar in bars1:
        h = bar.get_height()
        axes[0].annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 4),
                         textcoords="offset points", ha="center", weight="bold", fontsize=11)

    # 2. Revenue per Visit (Secondary)
    rev_summary = df_experiment.groupby("treatment_assignment")["revenue_per_visit"].mean().round(2)
    bars2 = axes[1].bar(rev_summary.index, rev_summary.values, color=["#4a7c59", "#2b5c8f"], edgecolor="black", width=0.45)
    axes[1].set_title("Secondary Metric: Revenue per Visit (₹)")
    axes[1].set_ylabel("Mean Spend (₹)")
    axes[1].set_ylim(0, max(rev_summary.values) * 1.25)
    for bar in bars2:
        h = bar.get_height()
        axes[1].annotate(f"₹{h:.1f}", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 4),
                         textcoords="offset points", ha="center", weight="bold", fontsize=11)

    # 3. No-Show Rate (Secondary)
    noshow_summary = (df_experiment.groupby("treatment_assignment")["no_show_outcome"].mean() * 100).round(2)
    bars3 = axes[2].bar(noshow_summary.index, noshow_summary.values, color=["#4a7c59", "#2b5c8f"], edgecolor="black", width=0.45)
    axes[2].set_title("Secondary Metric: Post-Trial No-Show Rate (%)")
    axes[2].set_ylabel("No-Show Rate (%)")
    axes[2].set_ylim(0, 20)
    for bar in bars3:
        h = bar.get_height()
        axes[2].annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 4),
                         textcoords="offset points", ha="center", weight="bold", fontsize=11)

    plt.suptitle("A/B Experiment Empirical Results: Automated WhatsApp Intervention vs Standard SMS", y=1.02)
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "10_ab_test_treatment_vs_control.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path


def plot_treatment_effects_forest(results_list: List[Dict]) -> str:
    """Forest plot illustrating point estimates and 95% Confidence Intervals for all experimental metrics."""
    fig, ax = plt.subplots(figsize=(10, 5))

    y_pos = np.arange(len(results_list))
    metrics = [r["metric"] for r in results_list]
    effects = [r["absolute_effect"] for r in results_list]
    ci_lows = [r["ci_lower_95"] for r in results_list]
    ci_highs = [r["ci_upper_95"] for r in results_list]

    errors = [
        [eff - low for eff, low in zip(effects, ci_lows)],
        [high - eff for eff, high in zip(effects, ci_highs)]
    ]

    ax.errorbar(effects, y_pos, xerr=errors, fmt="o", color="#1f77b4", ecolor="#1f77b4",
                elinewidth=2.5, capsize=6, markersize=8, markeredgecolor="black")
    ax.axvline(0, color="gray", linestyle="--", linewidth=1.5, alpha=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(metrics, fontsize=11, weight="bold")
    ax.invert_yaxis()
    ax.set_title("Estimated Treatment Effects with 95% Confidence Intervals (Forest Plot)")
    ax.set_xlabel("Absolute Treatment Effect (Point Estimate & 95% CI)")
    ax.grid(axis="x", linestyle="--", alpha=0.7)

    for i, r in enumerate(results_list):
        eff = r["absolute_effect"]
        low, high = r["ci_lower_95"], r["ci_upper_95"]
        ax.annotate(f"{eff:+.3f} [{low:+.3f}, {high:+.3f}] (p={r['p_value']:.4f})",
                    xy=(high, i), xytext=(8, -3), textcoords="offset points",
                    fontsize=9, weight="semibold", color="#084081")

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "11_ab_test_confidence_intervals_forest.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path


def plot_did_parallel_trends(df_historical: pd.DataFrame) -> str:
    """Plots pre- and post-intervention trajectory for Difference-in-Differences across regions."""
    fig, axes = plt.subplots(3, 1, figsize=(15, 14), sharex=True)

    metrics = [
        ("revenue_per_visit", "Revenue per Visit (₹)", "Revenue per Completed Visit Trajectory (DiD)"),
        ("customer_retention", "60-Day Customer Retention Rate", "Customer Repeat Retention Trajectory (DiD)"),
        ("no_show_rate", "No-Show Rate", "Appointment No-Show Rate Trajectory (DiD)")
    ]

    intervention_month = "2023-01"

    for idx, (col, ylabel, title) in enumerate(metrics):
        ax = axes[idx]
        treat_series = df_historical[df_historical["treatment_region_indicator"] == 1].groupby("month")[col].mean()
        ctrl_series = df_historical[df_historical["treatment_region_indicator"] == 0].groupby("month")[col].mean()

        months = treat_series.index.tolist()
        ax.plot(months, treat_series.values, marker="o", color="#2b5c8f", linewidth=2.2, label="Treatment Regions (Central & North)")
        ax.plot(months, ctrl_series.values, marker="s", color="#7570b3", linewidth=2.2, linestyle="--", label="Control Regions (East, South, West)")

        # Intervention line
        ax.axvline(intervention_month, color="crimson", linestyle="-.", linewidth=2, label="Intervention Start (2023-01)")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.6)
        if idx == 0:
            ax.legend(loc="upper left", frameon=True, facecolor="white")

    axes[2].set_xlabel("Year-Month")
    axes[2].tick_params(axis="x", rotation=45)

    plt.suptitle("Difference-in-Differences: Visualizing Parallel Trends & Post-Intervention Divergence", y=1.01)
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "12_did_parallel_trends_inspection.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path


def plot_did_event_study(df_event_study: pd.DataFrame, outcome_name: str = "Revenue per Visit") -> str:
    """Plots dynamic event-study regression leads and lags with 95% confidence intervals."""
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.errorbar(df_event_study["rel_time"], df_event_study["coefficient"],
                yerr=[df_event_study["coefficient"] - df_event_study["ci_lower_95"],
                      df_event_study["ci_upper_95"] - df_event_study["coefficient"]],
                fmt="o", color="#08519c", ecolor="#08519c", elinewidth=1.8, capsize=4,
                markersize=6, label="Dynamic Treatment Effect (gamma_t)")

    ax.axhline(0, color="black", linestyle="-", linewidth=1.2)
    ax.axvline(-0.5, color="red", linestyle="--", linewidth=1.8, label="Intervention Threshold (t = 0)")

    # Highlight reference period
    ax.scatter([-1], [0], color="red", s=80, zorder=5, label="Reference Month (t = -1, 2022-12)")

    ax.set_title(f"Event-Study Dynamic DiD Coefficients: {outcome_name}")
    ax.set_xlabel("Months Relative to Intervention (t = 0 is 2023-01)")
    ax.set_ylabel(f"Estimated Impact on {outcome_name}")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True, facecolor="white")

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "13_did_event_study_dynamics.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path


def plot_segmentation_heterogeneity(df_subgroups: pd.DataFrame) -> str:
    """Plots heterogeneous treatment effects across customer personas, outlet tiers, and service categories."""
    fig, ax = plt.subplots(figsize=(12, 7))

    df_plot = df_subgroups.sort_values(by="treatment_effect", ascending=True).reset_index(drop=True)
    y_pos = np.arange(len(df_plot))

    effects = df_plot["treatment_effect"]
    ci_lows = df_plot["ci_lower_95"]
    ci_highs = df_plot["ci_upper_95"]
    labels = df_plot["subgroup_dimension"] + ": " + df_plot["subgroup_value"]

    errors = [effects - ci_lows, ci_highs - effects]

    ax.errorbar(effects, y_pos, xerr=errors, fmt="s", color="#238b45", ecolor="#238b45",
                elinewidth=2.2, capsize=5, markersize=7)
    ax.axvline(0, color="black", linestyle="--", linewidth=1.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_title("Heterogeneous Treatment Effect Analysis Across Dimensions (60-Day Retention Lift)")
    ax.set_xlabel("Treatment Effect (Difference in Retention Rate & 95% CI)")
    ax.grid(axis="x", linestyle="--", alpha=0.7)

    for i, row in df_plot.iterrows():
        ax.annotate(f"{row['treatment_effect']:+.3f} (N={row['control_n']+row['treatment_n']:,})",
                    xy=(row["ci_upper_95"], i), xytext=(6, -3), textcoords="offset points",
                    fontsize=8.5, weight="semibold", color="#005a32")

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "14_segmentation_treatment_effects.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path


def plot_cuped_comparison(cuped_results: Dict) -> str:
    """Plots variance reduction comparison and treatment effect precision between Raw and CUPED."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    df_adj = cuped_results["adjusted_dataframe"]

    # 1. Distribution Variance Shrinkage KDE
    sns.kdeplot(df_adj["revenue_per_visit"], label=f"Original Y (Var: {cuped_results['variance_raw']:.1f})",
                color="#e41a1c", linewidth=2.2, ax=axes[0])
    sns.kdeplot(df_adj["y_adjusted"], label=f"CUPED Y_adj (Var: {cuped_results['variance_cuped']:.1f}, -{cuped_results['actual_variance_reduction_pct']:.1f}%)",
                color="#377eb8", linewidth=2.2, linestyle="--", ax=axes[0])
    axes[0].set_title("Outcome Distribution: Variance Reduction via CUPED")
    axes[0].set_xlabel("Metric Value")
    axes[0].set_ylabel("Density")
    axes[0].legend(frameon=True, facecolor="white")

    # 2. Precision / CI Shrinkage Comparison
    methods = ["Standard (Raw OLS)", "CUPED Adjusted"]
    effects = [cuped_results["raw_treatment_effect"], cuped_results["cuped_treatment_effect"]]
    ses = [cuped_results["raw_se"], cuped_results["cuped_se"]]
    ci_err = [[1.96 * s for s in ses], [1.96 * s for s in ses]]

    axes[1].errorbar(methods, effects, yerr=ci_err, fmt="o", color="#2ca02c", ecolor="#2ca02c",
                     elinewidth=2.5, capsize=8, markersize=9, markeredgecolor="black")
    axes[1].set_title("Treatment Effect Precision (95% CI Shrinkage)")
    axes[1].set_ylabel("Estimated Treatment Effect (₹)")
    axes[1].grid(axis="y", linestyle="--", alpha=0.7)
    for i, (m, eff, se) in enumerate(zip(methods, effects, ses)):
        axes[1].annotate(f"₹{eff:.2f} ± ₹{1.96*se:.2f}\n(SE: {se:.2f})",
                         xy=(i, eff), xytext=(0, 15), textcoords="offset points",
                         ha="center", fontsize=9.5, weight="bold")

    plt.suptitle("CUPED Variance Reduction Analytics: Enhancing Sensitivity without Point-Estimate Bias", y=1.02)
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "15_cuped_variance_reduction_comparison.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path


def generate_all_visualizations():
    """Master visualization pipeline. Generates all 15 high-resolution figures."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)
    print("=" * 70)
    print("STARTING DATA VISUALIZATION PIPELINE")
    print("=" * 70)

    # Load cleaned data
    df_outlets = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "outlets_clean.csv"))
    df_customers = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "customers_clean.csv"))
    df_appointments = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "appointments_clean.csv"))
    df_stylists = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "stylists_clean.csv"))
    df_experiment = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "experiment_data_clean.csv"))
    df_cust_features = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "customer_features.csv"))

    print("\n[1/8] Plotting Appointment Status Distribution...")
    p1 = plot_appointment_status_distribution(df_appointments)
    print(f"-> Saved: {p1}")

    print("\n[2/8] Plotting Outlet Performance Rankings...")
    p2 = plot_outlet_performance(df_outlets, df_appointments)
    print(f"-> Saved: {p2}")

    print("\n[3/8] Plotting Revenue Distributions...")
    p3 = plot_revenue_distributions(df_appointments)
    print(f"-> Saved: {p3}")

    print("\n[4/8] Plotting Customer Segmentation...")
    p4 = plot_customer_segmentation(df_customers, df_cust_features)
    print(f"-> Saved: {p4}")

    print("\n[5/8] Plotting Service Category Breakdown...")
    p5 = plot_service_category_breakdown(df_appointments)
    print(f"-> Saved: {p5}")

    print("\n[6/8] Plotting Monthly Revenue Trends...")
    p6 = plot_monthly_revenue_trends(df_appointments)
    print(f"-> Saved: {p6}")

    print("\n[7/8] Plotting No-Show and Cancellation Deep-Dive...")
    p7 = plot_noshow_deepdive(df_appointments, df_stylists)
    print(f"-> Saved: {p7}")

    print("\n[8/15] Plotting Experiment Balance and Outcomes...")
    p8 = plot_experiment_baseline_balance(df_experiment)
    print(f"-> Saved: {p8}")

    # Phase 2 figures
    from statistical_inference import power_sensitivity_grid, estimate_proportions_treatment_effect, estimate_continuous_treatment_effect
    from causal_inference import fit_event_study_model, compute_cuped_adjustment, evaluate_subgroup_effects

    print("\n[9/15] Plotting Power Analysis Sensitivity...")
    df_sens = power_sensitivity_grid(p_baseline=0.575)
    p9 = plot_power_sensitivity(df_sens)
    print(f"-> Saved: {p9}")

    print("\n[10/15] Plotting A/B Test Treatment vs Control...")
    p10 = plot_ab_test_treatment_vs_control(df_experiment)
    print(f"-> Saved: {p10}")

    print("\n[11/15] Plotting Treatment Effects Forest Plot...")
    # Calculate effects for forest plot
    t_ret = df_experiment[df_experiment["treatment_assignment"] == "Treatment"]["retention_outcome"]
    c_ret = df_experiment[df_experiment["treatment_assignment"] == "Control"]["retention_outcome"]
    res_ret = estimate_proportions_treatment_effect(t_ret.sum(), len(t_ret), c_ret.sum(), len(c_ret), "60-Day Retention Rate")

    t_ns = df_experiment[df_experiment["treatment_assignment"] == "Treatment"]["no_show_outcome"]
    c_ns = df_experiment[df_experiment["treatment_assignment"] == "Control"]["no_show_outcome"]
    res_ns = estimate_proportions_treatment_effect(t_ns.sum(), len(t_ns), c_ns.sum(), len(c_ns), "No-Show Rate")

    t_rev = df_experiment[df_experiment["treatment_assignment"] == "Treatment"]["revenue_per_visit"]
    c_rev = df_experiment[df_experiment["treatment_assignment"] == "Control"]["revenue_per_visit"]
    res_rev = estimate_continuous_treatment_effect(t_rev, c_rev, "Revenue per Visit (INR)")
    # Normalize rev scale for forest visual or display
    p11 = plot_treatment_effects_forest([res_ret, res_ns])
    print(f"-> Saved: {p11}")

    print("\n[12/15] Plotting DiD Parallel Trends Inspection...")
    df_historical = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "historical_regional_data_clean.csv"))
    p12 = plot_did_parallel_trends(df_historical)
    print(f"-> Saved: {p12}")

    print("\n[13/15] Plotting DiD Event Study Dynamics...")
    df_es = fit_event_study_model(df_historical, "revenue_per_visit")
    p13 = plot_did_event_study(df_es, "Revenue per Visit (INR)")
    print(f"-> Saved: {p13}")

    print("\n[14/15] Plotting Subgroup Heterogeneity...")
    df_merged_exp = df_experiment.merge(df_customers[["customer_id", "customer_segment"]], on="customer_id", how="left")
    df_merged_exp = df_merged_exp.merge(df_outlets[["outlet_id", "tier", "region"]], on="outlet_id", how="left")
    df_sub1 = evaluate_subgroup_effects(df_merged_exp, "customer_segment", "retention_outcome")
    df_sub2 = evaluate_subgroup_effects(df_merged_exp, "tier", "retention_outcome")
    df_subgroups = pd.concat([df_sub1, df_sub2], ignore_index=True)
    p14 = plot_segmentation_heterogeneity(df_subgroups)
    print(f"-> Saved: {p14}")

    print("\n[15/15] Plotting CUPED Variance Reduction Comparison...")
    cuped_res = compute_cuped_adjustment(df_experiment, "treatment_assignment", "revenue_per_visit", "pre_experiment_revenue")
    p15 = plot_cuped_comparison(cuped_res)
    print(f"-> Saved: {p15}")

    print("\n" + "=" * 70)
    print("ALL 15 VISUALIZATIONS GENERATED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    generate_all_visualizations()
