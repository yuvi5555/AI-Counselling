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
# LOAD DATA
# ==========================================================

historical_df = pd.read_csv("CAP_MASTER_STRUCTURED.csv")

historical_df["branch_name"] = historical_df["branch_name"].str.strip().str.lower()

historical_df["group_key"] = (
    historical_df["college_code"].astype(str) + "_" +
    historical_df["branch_code"].astype(str) + "_" +
    historical_df["normalized_category"].astype(str)
)

historical_df = historical_df.sort_values(["group_key", "year"])


# ==========================================================
# SAFE ENCODE
# ==========================================================

def safe_encode(encoders, column, value):
    le = encoders[column]
    value = str(value)
    if value in le.classes_:
        return le.transform([value])[0]
    else:
        return le.transform([le.classes_[0]])[0]


# ==========================================================
# GET LAG FEATURES
# ==========================================================

def get_lag_features(row):

    group_data = historical_df[historical_df["group_key"] == row["group_key"]]

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

    return prev_rank, two_year_rank_avg, 0, prev_percentile, two_year_percentile_avg, 0


# ==========================================================
# RECOMMENDATION ENGINE
# ==========================================================

def recommend_top_30(student_data):

    branch_name_input = student_data["branch_name"].strip().lower()

    filtered = historical_df[
        (historical_df["branch_name"] == branch_name_input) &
        (historical_df["normalized_category"] == student_data["normalized_category"])
    ]

    if filtered.empty:
        print("\n❌ Branch not found. Available branches:\n")
        print(historical_df["branch_name"].drop_duplicates().head(20).to_list())
        return pd.DataFrame()

    unique_colleges = filtered.drop_duplicates(subset=["college_code"])

    recommendations = []

    for _, row in unique_colleges.iterrows():

        prev_rank, two_rank_avg, rank_change, \
        prev_percentile, two_percentile_avg, percentile_change = get_lag_features(row)

        # ================= RANK MODEL =================
        rank_input = {
            "year": student_data["year"],
            "round_number": student_data["round_number"],
            "college_code": row["college_code"],
            "branch_code": row["branch_code"],
            "prev_year_cutoff": prev_rank,
            "two_year_avg": two_rank_avg,
            "yearly_change": rank_change,
            "normalized_category": student_data["normalized_category"],
            "gender": student_data["gender"],
            "quota_type": student_data["quota_type"],
            "stage": student_data["stage"],
            "university_type": student_data["university_type"]
        }

        rank_df = pd.DataFrame([rank_input])

        for col in ["normalized_category","gender","quota_type","stage","university_type"]:
            rank_df[col] = safe_encode(rank_encoders, col, rank_df[col].iloc[0])

        rank_df = rank_df[[
            "year","round_number","college_code","branch_code",
            "prev_year_cutoff","two_year_avg","yearly_change",
            "normalized_category","gender","quota_type","stage","university_type"
        ]]

        predicted_rank = rank_model.predict(rank_df)[0]

        # ================= PERCENTILE MODEL =================
        percentile_input = {
            "year": student_data["year"],
            "round_number": student_data["round_number"],
            "college_code": row["college_code"],
            "branch_code": row["branch_code"],
            "prev_year_percentile": prev_percentile,
            "two_year_percentile_avg": two_percentile_avg,
            "percentile_change": percentile_change,
            "normalized_category": student_data["normalized_category"],
            "gender": student_data["gender"],
            "quota_type": student_data["quota_type"],
            "stage": student_data["stage"],
            "university_type": student_data["university_type"]
        }

        percentile_df = pd.DataFrame([percentile_input])

        for col in ["normalized_category","gender","quota_type","stage","university_type"]:
            percentile_df[col] = safe_encode(percentile_encoders, col, percentile_df[col].iloc[0])

        percentile_df = percentile_df[[
            "year","round_number","college_code","branch_code",
            "prev_year_percentile","two_year_percentile_avg","percentile_change",
            "normalized_category","gender","quota_type","stage","university_type"
        ]]

        predicted_percentile = percentile_model.predict(percentile_df)[0]

        # ================= PROBABILITY =================
        rank_prob = 1 / (1 + np.exp((student_data["student_rank"] - predicted_rank) / 3000))
        percentile_prob = 1 / (1 + np.exp((student_data["student_percentile"] - predicted_percentile) / 1.5))

        final_prob = 0.3 * rank_prob + 0.7 * percentile_prob

        recommendations.append({
            "college_code": row["college_code"],
            "college_name": row["college_name"],
            "previous_year_cutoff_rank": int(prev_rank),
            "previous_year_cutoff_percentile": round(prev_percentile, 2),
            "predicted_cutoff_rank": int(predicted_rank),
            "predicted_cutoff_percentile": round(predicted_percentile, 2),
            "admission_probability (%)": round(final_prob * 100, 2)
        })

    rec_df = pd.DataFrame(recommendations)
    rec_df = rec_df.sort_values("admission_probability (%)", ascending=False)

    return rec_df.head(30)


# ==========================================================
# USER INPUT
# ==========================================================

print("\n====== TOP 30 COLLEGE RECOMMENDER (Branch Name Version) ======\n")

student_data = {
    "student_rank": int(input("Enter Your Rank: ")),
    "student_percentile": float(input("Enter Your Percentile: ")),
    "year": int(input("Enter Target Year: ")),
    "round_number": int(input("Enter Round Number: ")),
    "branch_name": input("Enter Preferred Branch Name: "),
    "normalized_category": input("Enter Category: ").upper(),
    "gender": input("Enter Gender: ").upper(),
    "quota_type": input("Enter Quota Type: ").upper(),
    "stage": input("Enter Stage (I/II): ").upper(),
    "university_type": input("Enter University Type: ")
}

top_30 = recommend_top_30(student_data)

if not top_30.empty:
    print("\n========= TOP 30 RECOMMENDED COLLEGES =========\n")
    print(top_30.to_string(index=False))
