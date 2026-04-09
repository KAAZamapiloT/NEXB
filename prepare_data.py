
import pandas as pd
import numpy as np

# ================================
# CONFIG
# ================================
INPUT_FILE = "5g_network_data.csv"
OUTPUT_FILE = "5g_network_data_processed.csv"


# ================================
# LOAD DATA
# ================================
df = pd.read_csv(INPUT_FILE)

print(f"Loaded dataset with shape: {df.shape}")


# ================================
# CLEAN COLUMN NAMES
# ================================
df.columns = (
    df.columns
    .str.strip()
    .str.replace(" ", "_")
    .str.replace("(", "")
    .str.replace(")", "")
    .str.replace("%", "percent")
)

print("Columns cleaned.")


# ================================
# DROP UNUSED COLUMNS
# ================================
df = df.drop(columns=["Timestamp"], errors="ignore")


# ================================
# FEATURE ENGINEERING
# ================================
df["Speed_Ratio"] = df["Download_Speed_Mbps"] / (df["Upload_Speed_Mbps"] + 1e-6)
df["Latency_Jitter"] = df["Latency_ms"] * df["Jitter_ms"]
df["Signal_Latency"] = df["Signal_Strength_dBm"] / (df["Latency_ms"] + 1)
df["Load_Index"] = df["Data_Usage_MB"] / (df["Connected_Duration_min"] + 1)

print("Feature engineering complete.")


# ================================
# NORMALIZATION FUNCTION
# ================================
def normalize(series):
    return (series - series.min()) / (series.max() - series.min() + 1e-6)


# ================================
# NORMALIZE KEY METRICS
# ================================
lat = normalize(df["Latency_ms"])
jit = normalize(df["Jitter_ms"])
ping = normalize(df["Ping_to_Google_ms"])

dl = normalize(df["Download_Speed_Mbps"])
ul = normalize(df["Upload_Speed_Mbps"])

sig = normalize(df["Signal_Strength_dBm"])

load = normalize(df["Data_Usage_MB"])
handover = normalize(df["Handover_Count"])


# ================================
# BUILD STRESS SCORE
# ================================
df["Stress_Score"] = (
    0.2 * lat +
    0.15 * jit +
    0.15 * ping +
    0.15 * (1 - dl) +
    0.1 * (1 - ul) +
    0.15 * (1 - sig) +
    0.07 * load +
    0.03 * handover
)

print("Stress score computed.")


# ================================
# CREATE NEW LABELS
# ================================
df["Network_Congestion_Level"] = pd.cut(
    df["Stress_Score"],
    bins=[-1, 0.4, 0.7, 1.1],
    labels=["Low", "Medium", "High"]
)

print("New congestion labels created.")


# ================================
# VALIDATION CHECKS
# ================================
print("\nClass distribution:")
print(df["Network_Congestion_Level"].value_counts())

print("\nFeature means by class:")
print(df.groupby("Network_Congestion_Level").mean(numeric_only=True))


# ================================
# SAVE NEW DATASET
# ================================
df.to_csv(OUTPUT_FILE, index=False)

print(f"\nProcessed dataset saved to: {OUTPUT_FILE}")