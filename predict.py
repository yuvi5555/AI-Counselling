import pandas as pd
import numpy as np
import joblib

# ==========================================================
# LOAD MODELS
# ==========================================================

rank_model = joblib.load("cutoff_regression_model_v3.pkl")
rank_encoders = joblib.load("cutoff_label_encoders_v3.pkl")

percentile_model = joblib.load("cutoff_percentile_model_v3.pkl")
percentile_encoders = joblib.load("cutoff_percentile_encoders_v3.pkl")

# ==========================================================
# LOAD HISTORICAL DATA (FOR LAG LOOKUP)
# ==========================================================

historical_df = pd.read_csv("CAP_MASTER_STRUCTURED.csv")

historical_df["group_key"] = (
    historical_df["college_code"].astype(str) + "_" +
    historical_df["branch_code"].astype(str) + "_" +
    historical_df["normalized_category"].astype(str)
)

historical_df = historical_df.sort_values(["group_key", "year"])


# ==========================================================
# SAFE ENCODING
# ==========================================================

def safe_encode(encoders, column, value):
    le = encoders[column]
    value = str(value)
    if value in le.classes_:
        return le.transform([value])[0]
    else:
        print(f"Warning: '{value}' not seen before. Using default.")
        return le.transform([le.classes_[0]])[0]


# ==========================================================
# GET LAG FEATURES
# ==========================================================

def get_lag_features(input_data):

    group_key = (
        str(input_data["college_code"]) + "_" +
        str(input_data["branch_code"]) + "_" +
        str(input_data["normalized_category"])
    )

    group_data = historical_df[historical_df["group_key"] == group_key]

    if group_data.empty:
        return 50000, 50000, 0, 90, 90, 0

    group_data = group_data.sort_values("year")

    prev_rank = group_data.iloc[-1]["final_rank"]
    prev_percentile = group_data.iloc[-1]["final_percentile"]

    if len(group_data) >= 2:
        two_year_rank_avg = group_data.tail(2)["final_rank"].mean()
        two_year_percentile_avg = group_data.tail(2)["final_percentile"].mean()
    else:
        two_year_rank_avg = prev_rank
        two_year_percentile_avg = prev_percentile

    return (
        prev_rank,
        two_year_rank_avg,
        0,  # rank change assumed stable
        prev_percentile,
        two_year_percentile_avg,
        0   # percentile change assumed stable
    )


# ==========================================================
# PREDICTION FUNCTION
# ==========================================================

def predict_dual(student_rank, student_percentile, input_data):

    (
        prev_rank,
        two_year_rank_avg,
        rank_change,
        prev_percentile,
        two_year_percentile_avg,
        percentile_change
    ) = get_lag_features(input_data)

    # ================= RANK MODEL =================

    rank_input = input_data.copy()
    rank_input["prev_year_cutoff"] = prev_rank
    rank_input["two_year_avg"] = two_year_rank_avg
    rank_input["yearly_change"] = rank_change

    rank_df = pd.DataFrame([rank_input])

    rank_cat_cols = [
        "normalized_category",
        "gender",
        "quota_type",
        "stage",
        "university_type"
    ]

    for col in rank_cat_cols:
        rank_df[col] = safe_encode(rank_encoders, col, rank_df[col].iloc[0])

    rank_features = [
        "year",
        "round_number",
        "college_code",
        "branch_code",
        "prev_year_cutoff",
        "two_year_avg",
        "yearly_change",
        "normalized_category",
        "gender",
        "quota_type",
        "stage",
        "university_type"
    ]

    rank_df = rank_df[rank_features]
    predicted_rank_cutoff = rank_model.predict(rank_df)[0]

    # ================= PERCENTILE MODEL =================

    percentile_input = input_data.copy()
    percentile_input["prev_year_percentile"] = prev_percentile
    percentile_input["two_year_percentile_avg"] = two_year_percentile_avg
    percentile_input["percentile_change"] = percentile_change

    percentile_df = pd.DataFrame([percentile_input])

    for col in rank_cat_cols:
        percentile_df[col] = safe_encode(percentile_encoders, col, percentile_df[col].iloc[0])

    percentile_features = [
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

    percentile_df = percentile_df[percentile_features]
    predicted_percentile_cutoff = percentile_model.predict(percentile_df)[0]

    # ================= PROBABILITY CALCULATION =================

    rank_prob = 1 / (1 + np.exp((student_rank - predicted_rank_cutoff) / 3000))
    percentile_prob = 1 / (1 + np.exp(( predicted_percentile_cutoff - student_percentile ) / 1.5))

    # Weighted Fusion
    final_probability = 0.3 * rank_prob + 0.7 * percentile_prob

    return predicted_rank_cutoff, predicted_percentile_cutoff, final_probability


# ==========================================================
# USER INPUT
# ==========================================================

print("\n====== Intelligent Dual Admission Predictor ======\n")

student_rank = int(input("Enter Your Rank: "))
student_percentile = float(input("Enter Your Percentile: "))

input_data = {
    "year": int(input("Enter Target Year (e.g., 2026): ")),
    "round_number": int(input("Enter Round Number (1-4): ")),
    "college_code": int(input("Enter College Code: ")),
    "branch_code": int(input("Enter Branch Code: ")),
    "normalized_category": input("Enter Category: ").upper(),
    "gender": input("Enter Gender (GENERAL/FEMALE): ").upper(),
    "quota_type": input("Enter Quota Type: ").upper(),
    "stage": input("Enter Stage (I/II): ").upper(),
    "university_type": input("Enter University Type: ")
}

pred_rank, pred_percentile, prob = predict_dual(
    student_rank,
    student_percentile,
    input_data
)

print("\n======================================")
print("Predicted Cutoff Rank:", int(pred_rank))
print("Predicted Cutoff Percentile:", round(pred_percentile, 2))
print("Final Admission Probability:", round(prob * 100, 2), "%")

if prob >= 0.85:
    print("SAFE OPTION 🟢")
elif prob >= 0.60:
    print("MODERATE OPTION 🟡")
else:
    print("DREAM OPTION 🔴")

print("======================================")
