import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
import joblib

# ==========================================================
# LOAD DATA
# ==========================================================

df = pd.read_csv("CAP_MASTER_STRUCTURED.csv")
print("Original Dataset Shape:", df.shape)

# ==========================================================
# CREATE GROUP KEY (College + Branch + Category)
# ==========================================================

df["group_key"] = (
    df["college_code"].astype(str) + "_" +
    df["branch_code"].astype(str) + "_" +
    df["normalized_category"].astype(str)
)

# ==========================================================
# SORT FOR TIME FEATURES
# ==========================================================

df = df.sort_values(["group_key", "year"])

# ==========================================================
# CREATE LAG FEATURES (PERCENTILE BASED)
# ==========================================================

df["prev_year_percentile"] = df.groupby("group_key")["final_percentile"].shift(1)

df["two_year_percentile_avg"] = (
    df.groupby("group_key")["final_percentile"]
    .rolling(2)
    .mean()
    .reset_index(level=0, drop=True)
)

df["percentile_change"] = df["final_percentile"] - df["prev_year_percentile"]

# ==========================================================
# REMOVE ROWS WITHOUT HISTORY
# ==========================================================

df = df.dropna(subset=["prev_year_percentile"])
print("After removing first-year rows:", df.shape)

df["two_year_percentile_avg"] = df["two_year_percentile_avg"].fillna(df["prev_year_percentile"])
df["percentile_change"] = df["percentile_change"].fillna(0)

# ==========================================================
# TARGET
# ==========================================================

y = df["final_percentile"]

# ==========================================================
# FEATURES
# ==========================================================

features = [
    "year",
    "round_number",
    "college_code",
    "branch_code",
    "prev_year_percentile",
    "two_year_percentile_avg",
    "percentile_change",
    "normalized_category",
    "gender",
    "quota_type",
    "stage",
    "university_type"
]

X = df[features].copy()

# ==========================================================
# ENCODE CATEGORICAL FEATURES
# ==========================================================

categorical_cols = [
    "normalized_category",
    "gender",
    "quota_type",
    "stage",
    "university_type"
]

label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Train Size:", X_train.shape)
print("Test Size:", X_test.shape)

# ==========================================================
# XGBOOST MODEL
# ==========================================================

model = XGBRegressor(
    n_estimators=800,
    max_depth=9,
    learning_rate=0.04,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_lambda=2,
    reg_alpha=1,
    random_state=42
)

model.fit(X_train, y_train)

# ==========================================================
# EVALUATION
# ==========================================================

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMean Absolute Error:", mae)
print("R2 Score:", r2)

# ==========================================================
# SAVE MODEL
# ==========================================================

joblib.dump(model, "cutoff_percentile_model_v3.pkl")
joblib.dump(label_encoders, "cutoff_percentile_encoders_v3.pkl")

print("\nImproved Percentile Regression Model Saved Successfully")
