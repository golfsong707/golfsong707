# ============================================================
# CORRECTED PIPELINE 2.0
# STAGE 1 — RAW DATA AUDIT AND EXPLORATORY DATA ANALYSIS
# Print edition: as stage1_audit_eda.py, with the
# repeated banner / save / plot boilerplate folded
# into helpers for a larger type size.
# ============================================================
import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# --- HELPERS --------------------------------------
def banner(title):
    """Print a section banner."""
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def save(report, filename, **kwargs):
    """Write a table to the Stage 1 folder."""
    report.to_csv(os.path.join(stage1_dir, filename), **kwargs)


def finish_plot(filename):
    """Save, display and close the figure."""
    plt.tight_layout()
    plt.savefig(os.path.join(stage1_dir, filename), dpi=300)
    plt.show()
    plt.close()


def histogram(series, xlabel, title, filename, figsize=(7, 5)):
    """Plot one raw distribution."""
    plt.figure(figsize=figsize)
    series.dropna().plot(kind="hist", bins=50)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.title(title)
    finish_plot(filename)


banner("CORRECTED PIPELINE 2.0\n"
       "STAGE 1 — RAW DATA AUDIT AND EXPLORATORY DATA ANALYSIS")

# --- 1. PATHS -------------------------------------
raw_data_path = (
    r"D:\Data_Workspace\Capstone_Project"
    r"\NLGN_PLC_Predictive_Analystic\data\01_raw"
    r"\NLNG_Dirty_Data.csv"
)
corrected_root = (
    r"D:\Data_Workspace\Capstone_Project"
    r"\NLGN_PLC_Predictive_Analystic\data"
    r"\04_corrected_pipeline"
)
stage1_dir = os.path.join(corrected_root, "stage_1_audit_eda")
os.makedirs(stage1_dir, exist_ok=True)
print(f"\nRaw dataset:\n{raw_data_path}")
print(f"\nCorrected pipeline output:\n{corrected_root}")

# --- 2. LOAD RAW DATA -----------------------------
raw_df = pd.read_csv(raw_data_path)
# Standardise headers only. Values are not modified.
raw_df.columns = raw_df.columns.str.strip().str.lower()
banner("RAW DATASET")
print(f"Rows:    {len(raw_df):,}")
print(f"Columns: {len(raw_df.columns):,}")
mem_mb = raw_df.memory_usage(deep=True).sum() / 1024**2
print(f"Memory:  {mem_mb:.2f} MB")
print("\nFirst 5 rows:")
print(raw_df.head())

# --- 3. SCHEMA AUDIT ------------------------------
schema_report = pd.DataFrame({
    "column": raw_df.columns,
    "dtype": [str(raw_df[c].dtype) for c in raw_df.columns],
    "missing_count": [raw_df[c].isna().sum()
                      for c in raw_df.columns],
    "missing_pct": [raw_df[c].isna().mean() * 100
                    for c in raw_df.columns],
    "unique_values": [raw_df[c].nunique(dropna=False)
                      for c in raw_df.columns],
})
schema_report = schema_report.sort_values("missing_pct",
                                          ascending=False)
save(schema_report, "01_schema_audit.csv", index=False)
print("\nSchema audit saved.")

# --- 4. DUPLICATE AUDIT ---------------------------
exact_duplicates = raw_df.duplicated().sum()
if {"equipment_id", "timestamp"}.issubset(raw_df.columns):
    equipment_timestamp_duplicates = raw_df.duplicated(
        subset=["equipment_id", "timestamp"], keep=False).sum()
else:
    equipment_timestamp_duplicates = np.nan
duplicate_report = pd.DataFrame({
    "audit_item": ["exact_duplicate_rows",
                   "equipment_timestamp_duplicate_rows"],
    "count": [exact_duplicates, equipment_timestamp_duplicates],
})
save(duplicate_report, "02_duplicate_audit.csv", index=False)
print("\nDuplicate audit:")
print(duplicate_report)

# --- 5. TIMESTAMP AUDIT ---------------------------
raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"],
                                     errors="coerce")
timestamp_report = pd.DataFrame({
    "metric": ["invalid_or_missing_timestamps",
               "minimum_timestamp", "maximum_timestamp",
               "unique_timestamps"],
    "value": [raw_df["timestamp"].isna().sum(),
              raw_df["timestamp"].min(),
              raw_df["timestamp"].max(),
              raw_df["timestamp"].nunique()],
})
save(timestamp_report, "03_timestamp_audit.csv", index=False)
print("\nTimestamp audit:")
print(timestamp_report)

# --- 6. EQUIPMENT AND TRAIN COVERAGE --------------
if "equipment_id" in raw_df.columns:
    equipment_coverage = (
        raw_df["equipment_id"].value_counts()
        .rename_axis("equipment_id")
        .reset_index(name="record_count")
    )
    save(equipment_coverage, "04_equipment_coverage.csv",
         index=False)
if "train" in raw_df.columns:
    train_coverage = (
        raw_df["train"].value_counts().rename_axis("train")
        .reset_index(name="record_count")
    )
    save(train_coverage, "05_train_coverage.csv", index=False)
n_eq = (raw_df["equipment_id"].nunique()
        if "equipment_id" in raw_df.columns else "N/A")
n_tr = (raw_df["train"].nunique()
        if "train" in raw_df.columns else "N/A")
print(f"\nEquipment assets: {n_eq}")
print(f"Trains: {n_tr}")

# --- 7. MISSING VALUE AUDIT -----------------------
missing_report = pd.DataFrame({
    "column": raw_df.columns,
    "missing_count": raw_df.isna().sum().values,
    "missing_percentage": raw_df.isna().mean().values * 100,
})
missing_report = missing_report.sort_values("missing_percentage",
                                            ascending=False)
save(missing_report, "06_missing_value_audit.csv", index=False)
banner("MISSING VALUE AUDIT")
print(missing_report[missing_report["missing_count"] > 0])

# --- 8. TARGET AUDIT — FAILURE WITHIN 24 HOURS ----
classification_target = "failure_within_24h"
if classification_target in raw_df.columns:
    failure_counts = (raw_df[classification_target]
                      .value_counts(dropna=False).sort_index())
    failure_distribution = pd.DataFrame({
        "count": failure_counts,
        "percentage": failure_counts / len(raw_df) * 100,
    })
    save(failure_distribution,
         "07_failure_24h_distribution.csv")
    banner("FAILURE WITHIN 24 HOURS")
    print(failure_distribution)

# --- 9. RUL AUDIT ---------------------------------
rul_target = "rul_days"
if rul_target in raw_df.columns:
    rul_audit = pd.DataFrame({
        "metric": ["missing", "negative", "mean", "median",
                   "minimum", "maximum"],
        "value": [raw_df[rul_target].isna().sum(),
                  (raw_df[rul_target] < 0).sum(),
                  raw_df[rul_target].mean(),
                  raw_df[rul_target].median(),
                  raw_df[rul_target].min(),
                  raw_df[rul_target].max()],
    })
    save(rul_audit, "08_rul_audit.csv", index=False)
    banner("RUL AUDIT")
    print(rul_audit)

# --- 10. PHYSICAL VALIDITY AUDIT ------------------
physical_rules = {
    "load_factor": (0, 1), "wear_level": (0, 1),
    "lubrication_health_index": (0, 100),
    "oil_pressure": (0, np.inf),
    "oil_particles_ppm": (0, np.inf),
    "overall_vibration": (0, np.inf),
    "vibration": (0, np.inf), "rpm": (0, np.inf),
    "hours_since_maint": (0, np.inf),
    "cumulative_op_hours": (0, np.inf),
    "rul_days": (0, np.inf),
}
physical_results = []
for col, (lower, upper) in physical_rules.items():
    if col not in raw_df.columns:
        continue
    invalid_mask = ((raw_df[col] < lower)
                    | (raw_df[col] > upper)) & raw_df[col].notna()
    physical_results.append({
        "variable": col, "lower_bound": lower,
        "upper_bound": upper,
        "invalid_count": int(invalid_mask.sum()),
        "invalid_pct": float(invalid_mask.mean() * 100),
    })
physical_report = pd.DataFrame(physical_results)
save(physical_report, "09_physical_validity_audit.csv",
     index=False)
banner("PHYSICAL VALIDITY AUDIT")
print(physical_report)

# --- 11. OUTLIER AUDIT — IQR ----------------------
numeric_columns = raw_df.select_dtypes(include=np.number).columns
outlier_results = []
for col in numeric_columns:
    series = raw_df[col].dropna()
    if len(series) == 0:
        continue
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = (raw_df[col] < lower) | (raw_df[col] > upper)
    outlier_results.append({
        "variable": col, "lower_iqr": lower,
        "upper_iqr": upper,
        "outlier_count": int(mask.sum()),
        "outlier_pct": float(mask.mean() * 100),
    })
outlier_report = (pd.DataFrame(outlier_results)
                  .sort_values("outlier_pct", ascending=False))
save(outlier_report, "10_outlier_audit_iqr.csv", index=False)
banner("TOP OUTLIER VARIABLES")
print(outlier_report.head(15))

# --- 12. LEAKAGE SCREENING ------------------------
potential_leakage = ["failure_within_72h", "failure_within_7d",
                     "rul_days", "rul_censored", "failure_rate"]
treatments = []
for c in potential_leakage:
    if c in ["failure_within_72h", "failure_within_7d"]:
        treatments.append("EXCLUDE — future failure target")
    elif c in ["rul_days", "rul_censored"]:
        treatments.append("EXCLUDE — regression target")
    elif c == "failure_rate":
        treatments.append(
            "EXCLUDE CONSERVATIVELY — verify provenance")
    else:
        treatments.append("Not present")
leakage_report = pd.DataFrame({
    "variable": potential_leakage,
    "present": [c in raw_df.columns for c in potential_leakage],
    "treatment": treatments,
})
save(leakage_report, "11_leakage_screening.csv", index=False)
banner("LEAKAGE SCREENING")
print(leakage_report)

# --- 13. DESCRIPTIVE STATISTICS -------------------
numeric_profile = (raw_df.select_dtypes(include=np.number)
                   .describe().T)
numeric_profile["skewness"] = (raw_df
                               .select_dtypes(include=np.number)
                               .skew())
save(numeric_profile, "12_raw_statistical_profile.csv")
banner("RAW NUMERICAL PROFILE")
print(numeric_profile)

# --- 14. RUL DISTRIBUTION -------------------------
if rul_target in raw_df.columns:
    histogram(raw_df[rul_target], "RUL (Days)",
              "Raw RUL Distribution", "eda_rul_distribution.png",
              figsize=(8, 5))

# --- 15. FAILURE DISTRIBUTION ---------------------
if classification_target in raw_df.columns:
    failure_counts = (raw_df[classification_target]
                      .value_counts().sort_index())
    plt.figure(figsize=(7, 5))
    failure_counts.plot(kind="bar")
    plt.xlabel("Failure Within 24 Hours")
    plt.ylabel("Number of Observations")
    plt.title("24-Hour Failure Class Distribution")
    plt.xticks(rotation=0)
    finish_plot("eda_failure_distribution.png")

# --- 16. KEY SENSOR DISTRIBUTIONS -----------------
key_sensors = ["overall_vibration", "vibration",
               "oil_particles_ppm", "oil_pressure",
               "lubrication_health_index", "load_factor",
               "wear_level"]
available_sensors = [c for c in key_sensors if c in raw_df.columns]
for col in available_sensors:
    histogram(raw_df[col], col, f"Distribution of {col}",
              f"eda_{col}_distribution.png")

# --- 17. SENSOR VALUES BY FAILURE STATUS ----------
if classification_target in raw_df.columns:
    sensor_failure_summary = []
    for col in available_sensors:
        normal = raw_df.loc[raw_df[classification_target] == 0,
                            col].dropna()
        failure = raw_df.loc[raw_df[classification_target] == 1,
                             col].dropna()
        sensor_failure_summary.append({
            "sensor": col, "normal_n": len(normal),
            "failure_n": len(failure),
            "normal_median": normal.median(),
            "failure_median": failure.median(),
            "median_difference": failure.median() - normal.median(),
        })
    sensor_failure_report = pd.DataFrame(sensor_failure_summary)
    save(sensor_failure_report, "13_sensor_failure_comparison.csv",
         index=False)
    banner("SENSOR VALUES BY FAILURE STATUS")
    print(sensor_failure_report)

# --- 18. STAGE 1 SUMMARY --------------------------
stage1_summary = {
    "rows": int(len(raw_df)),
    "columns": int(len(raw_df.columns)),
    "exact_duplicates": int(exact_duplicates),
    "equipment_timestamp_duplicates": (
        int(equipment_timestamp_duplicates)
        if not pd.isna(equipment_timestamp_duplicates) else None),
    "missing_value_columns": int((raw_df.isna().sum() > 0).sum()),
    "equipment_assets": (int(raw_df["equipment_id"].nunique())
                         if "equipment_id" in raw_df.columns
                         else None),
    "trains": (int(raw_df["train"].nunique())
               if "train" in raw_df.columns else None),
}
with open(os.path.join(stage1_dir, "stage1_summary.json"), "w") as f:
    json.dump(stage1_summary, f, indent=4)
banner("STAGE 1 COMPLETE")
print(json.dumps(stage1_summary, indent=4))
