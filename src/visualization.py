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


def generate_all_visualizations():
    """Master visualization pipeline. Generates all 8 high-resolution figures."""
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

    print("\n[8/8] Plotting Experiment Balance and Outcomes...")
    p8 = plot_experiment_baseline_balance(df_experiment)
    print(f"-> Saved: {p8}")

    print("\n" + "=" * 70)
    print("ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    generate_all_visualizations()
