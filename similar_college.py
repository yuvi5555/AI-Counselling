import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib

# ==========================================================
# LOAD STRUCTURED DATASET
# ==========================================================

df = pd.read_csv("CAP_MASTER_STRUCTURED.csv")

print("Dataset Loaded:", df.shape)

# ==========================================================
# CREATE COLLEGE-LEVEL FEATURES
# ==========================================================

college_features = df.groupby("college_code").agg(
    college_name=("college_name", "first"),
    avg_percentile=("final_percentile", "mean"),
    avg_rank=("final_rank", "mean"),
    percentile_std=("final_percentile", "std"),
    rank_std=("final_rank", "std"),
    data_points=("year", "count")
).reset_index()

# Fill missing std values
college_features["percentile_std"] = college_features["percentile_std"].fillna(0)
college_features["rank_std"] = college_features["rank_std"].fillna(0)

# Stability score (lower std = more stable)
college_features["stability_score"] = (
    1 / (1 + college_features["percentile_std"])
)

print("College Feature Table Created:", college_features.shape)

# ==========================================================
# SELECT FEATURES FOR CLUSTERING
# ==========================================================

feature_columns = [
    "avg_percentile",
    "avg_rank",
    "percentile_std",
    "rank_std",
    "data_points",
    "stability_score"
]

X = college_features[feature_columns].copy()

# ==========================================================
# SCALE FEATURES
# ==========================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================================================
# CLUSTER COLLEGES
# ==========================================================

n_clusters = 8  # You can tune this

kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
college_features["cluster"] = kmeans.fit_predict(X_scaled)

print("Clustering Completed")

# ==========================================================
# SAVE MODELS
# ==========================================================

joblib.dump(kmeans, "college_kmeans_model.pkl")
joblib.dump(scaler, "college_scaler.pkl")
college_features.to_csv("college_similarity_features.csv", index=False)

print("Similarity Engine Saved Successfully")

# ==========================================================
# SIMILAR COLLEGE RECOMMENDER FUNCTION
# ==========================================================

def recommend_similar_colleges(college_code, top_n=5):

    if college_code not in college_features["college_code"].values:
        print("College code not found.")
        return pd.DataFrame()

    cluster_id = college_features[
        college_features["college_code"] == college_code
    ]["cluster"].values[0]

    similar = college_features[
        (college_features["cluster"] == cluster_id) &
        (college_features["college_code"] != college_code)
    ]

    # Sort by closeness in avg_percentile
    target_percentile = college_features[
        college_features["college_code"] == college_code
    ]["avg_percentile"].values[0]

    similar["distance"] = abs(similar["avg_percentile"] - target_percentile)

    similar = similar.sort_values("distance")

    return similar[[
        "college_code",
        "college_name",
        "avg_percentile",
        "avg_rank",
        "stability_score"
    ]].head(top_n)


# ==========================================================
# TEST EXAMPLE
# ==========================================================

if __name__ == "__main__":

    print("\n==== College Similarity Test ====\n")

    code = int(input("Enter College Code: "))

    result = recommend_similar_colleges(code)

    if not result.empty:
        print("\nSimilar Colleges:\n")
        print(result.to_string(index=False))
