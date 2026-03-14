import pandas as pd
import re
import os

# ==============================
# CONFIGURATION
# ==============================

DATA_FILES = [
    
    # 2024
    {"path": "2024CAP_I.csv", "year": 2024, "round": "CAP1"},
    {"path": "2024CAP_III.csv", "year": 2024, "round": "CAP3"},
    
    # 2025
    {"path": "2025ENGG_CAP1_CutOff_contains_additional_info.xlsx", "year": 2025, "round": "CAP1"},
    {"path": "2025ENGG_CAP2_CutOff_contains_additional_info.xlsx", "year": 2025, "round": "CAP2"},
    {"path": "2025ENGG_CAP3_CutOff_contains_additional_info.xlsx", "year": 2025, "round": "CAP3"},
    {"path": "2025ENGG_CAP4_CutOff_contains_additional_info.xlsx", "year": 2025, "round": "CAP4"},
]

# ==============================
# HELPER FUNCTIONS
# ==============================

def clean_columns(df):
    df.columns = [re.sub(r'\s+', '_', col.strip().lower()) for col in df.columns]
    return df

def extract_rank_percentile(cell):
    if pd.isna(cell):
        return None, None
    match = re.search(r'(\d+)\s*\(([\d\.]+)\)', str(cell))
    if match:
        return int(match.group(1)), float(match.group(2))
    return None, None

def standardize_dataframe(df, year, round_name):
    df = clean_columns(df)
    
    # Add year and round
    df["year"] = year
    df["round"] = round_name
    
    # Try to detect rank + percentile columns automatically
    for col in df.columns:
        if "cutoff" in col or "rank" in col:
            df[[f"{col}_rank", f"{col}_percentile"]] = df[col].apply(
                lambda x: pd.Series(extract_rank_percentile(x))
            )
    
    return df

# ==============================
# MAIN PIPELINE
# ==============================

all_data = []

for file_info in DATA_FILES:
    
    path = file_info["path"]
    year = file_info["year"]
    round_name = file_info["round"]
    
    if not os.path.exists(path):
        print(f"Skipping missing file: {path}")
        continue
    
    print(f"Processing: {path}")
    
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)
    
    df = standardize_dataframe(df, year, round_name)
    
    all_data.append(df)

# Combine everything
master_df = pd.concat(all_data, ignore_index=True)

# Remove duplicates
master_df.drop_duplicates(inplace=True)

# Remove fully empty rows
master_df.dropna(how="all", inplace=True)

# Save master file
master_df.to_csv("FINAL_CAP_MASTER_DATASET.csv", index=False)

print("\nMaster Dataset Created Successfully!")
print("Total Rows:", len(master_df))
