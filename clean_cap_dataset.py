import pandas as pd
import numpy as np
import re

# ==================================================
# LOAD DATASET
# ==================================================

df = pd.read_csv("FINAL_CAP_MASTER_DATASET.csv", low_memory=False)

print("Original Shape:", df.shape)

# ==================================================
# FIX COLUMN NAMES (HANDLE UNICODE HYPHEN)
# ==================================================

df.columns = (
    df.columns
      .str.replace("-", "_")   # Fix special hyphen
      .str.replace("-", "_")
      .str.strip()
      .str.lower()
      .str.replace(" ", "_")
)

print("\nColumns After Cleaning:\n", df.columns)

# ==================================================
# UNIFY RANK & PERCENTILE
# ==================================================

rank_col = None
percentile_col = None

for col in df.columns:
    if "cut" in col and "rank" in col and "percentile" not in col:
        rank_col = col
    if "cut" in col and "percentile" in col:
        percentile_col = col

if rank_col is None:
    rank_col = "rank"

if percentile_col is None:
    percentile_col = "percentile"

df["final_rank"] = df[rank_col].combine_first(df.get("rank"))
df["final_percentile"] = df[percentile_col].combine_first(df.get("percentile"))

df["final_rank"] = pd.to_numeric(df["final_rank"], errors="coerce")
df["final_percentile"] = pd.to_numeric(df["final_percentile"], errors="coerce")

df = df.dropna(subset=["final_rank", "final_percentile"])

df["final_rank"] = df["final_rank"].astype(int)
df["final_percentile"] = df["final_percentile"].astype(float)

# ==================================================
# CLEAN STRING COLUMNS
# ==================================================

string_cols = [
    "college_name",
    "branch_name",
    "course_name",
    "category",
    "status",
    "stage",
    "home_university"
]

for col in string_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()

# ==================================================
# EXTRACT CATEGORY + GENDER + QUOTA
# ==================================================

def extract_details(cat):
    cat = str(cat).upper()

    # ----------------- CATEGORY -----------------
    if "OPEN" in cat:
        category = "OPEN"
    elif "SC" in cat:
        category = "SC"
    elif "ST" in cat:
        category = "ST"
    elif "OBC" in cat:
        category = "OBC"
    elif "NT1" in cat:
        category = "NT1"
    elif "NT2" in cat:
        category = "NT2"
    elif "NT3" in cat:
        category = "NT3"
    elif "VJ" in cat:
        category = "VJ"
    elif "EWS" in cat:
        category = "EWS"
    else:
        category = "OTHER"

    # ----------------- GENDER -----------------
    if cat.startswith("L"):  
        gender = "FEMALE"
    else:
        gender = "GENERAL"

    # ----------------- QUOTA TYPE -----------------
    if "PWD" in cat:
        quota_type = "PWD"
    elif "DEF" in cat:
        quota_type = "DEFENCE"
    elif "TFWS" in cat:
        quota_type = "TFWS"
    elif "EWS" in cat:
        quota_type = "EWS"
    else:
        quota_type = "REGULAR"

    return pd.Series([category, gender, quota_type])


df[["normalized_category", "gender", "quota_type"]] = df["category"].apply(extract_details)

# ==================================================
# FIX UNIVERSITY TYPE
# ==================================================

if "home_university" in df.columns:
    df["university_type"] = df["home_university"].fillna("State")
else:
    df["university_type"] = "State"

# ==================================================
# ENCODE ROUND NUMBER
# ==================================================

round_map = {
    "CAP1": 1,
    "CAP2": 2,
    "CAP3": 3,
    "CAP4": 4
}

df["round_number"] = df["round"].map(round_map)

# ==================================================
# DROP USELESS COLUMNS
# ==================================================

drop_cols = [
    "sr._no.",
    "page",
    "rank",
    "percentile"
]

df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# ==================================================
# FINAL STRUCTURED COLUMNS
# ==================================================

final_columns = [
    "year",
    "round",
    "round_number",
    "college_code",
    "college_name",
    "branch_code",
    "branch_name",
    "normalized_category",
    "gender",
    "quota_type",
    "stage",
    "university_type",
    "status",
    "final_rank",
    "final_percentile"
]

df = df[[c for c in final_columns if c in df.columns]]

df = df.drop_duplicates()

print("\nFinal Shape:", df.shape)

df.to_csv("CAP_MASTER_STRUCTURED.csv", index=False)

print("\nCAP_MASTER_STRUCTURED.csv created successfully.")
