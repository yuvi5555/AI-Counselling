# ==========================================================
# COLLEGE ASSISTANT - UNIFIED INTERFACE
# ==========================================================
# This file imports and calls functions from:
# - predict.py (for admission prediction)
# - recommend_college.py (for top 30 recommendations)
# - similar_college.py (for similar college recommendations)
# - rag_assistant.py (for RAG-based Q&A)
# ==========================================================

import sys
from predict import predict_dual
from recommend_college import recommend_top_30
from similar_college import recommend_similar_colleges
from rag_assistant import generate_answer


class CollegeAssistant:
    """Unified College Counselling Assistant"""

    def __init__(self):
        """Initialize the assistant"""
        print("College Assistant initialized successfully!")

    # =====================================================
    # FEATURE 1: PREDICT ADMISSION
    # =====================================================

    def predict_admission(self):
        """Predict admission probability - calls predict.py"""
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
            print("Status: SAFE")
        elif prob >= 0.60:
            print("Status: MODERATE")
        else:
            print("Status: DREAM")

        print("======================================")

    # =====================================================
    # FEATURE 2: RECOMMEND COLLEGES
    # =====================================================

    def recommend_colleges(self):
        """Recommend top 30 colleges - calls recommend_college.py"""
        print("\n====== TOP 30 COLLEGE RECOMMENDER ======\n")

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
        else:
            print("\nNo recommendations found!")

    # =====================================================
    # FEATURE 3: FIND SIMILAR COLLEGES
    # =====================================================

    def find_similar_colleges(self):
        """Find similar colleges - calls similar_college.py"""
        print("\n==== COLLEGE SIMILARITY SEARCH ====\n")

        college_code = int(input("Enter College Code: "))
        top_n = int(input("Number of similar colleges to show (default 5): ") or "5")

        result = recommend_similar_colleges(college_code, top_n)

        if result is not None:
            print("\n========= SIMILAR COLLEGES =========\n")
            print(result.to_string(index=False))
        else:
            print("College code not found!")

    # =====================================================
    # FEATURE 4: RAG ASSISTANT
    # =====================================================

    def ask_rag_question(self):
        """Ask questions using RAG - calls rag_assistant.py"""
        print("\n===== OpenRouter RAG Counselling Assistant =====\n")

        query = input("Ask your question: ")
        answer = generate_answer(query)

        print("\nAnswer:\n")
        print(answer)
        print("\n" + "="*60)


# ==========================================================
# MAIN EXECUTION
# ==========================================================

if __name__ == "__main__":
    assistant = CollegeAssistant()

    print("\n========== COLLEGE ASSISTANT - ALL FEATURES ==========\n")

    while True:
        print("\n1. Predict Admission")
        print("2. Recommend Top 30 Colleges")
        print("3. Find Similar Colleges")
        print("4. Ask RAG Question")
        print("5. Exit")

        choice = input("\nSelect option (1-5): ").strip()

        try:
            if choice == "1":
                assistant.predict_admission()

            elif choice == "2":
                assistant.recommend_colleges()

            elif choice == "3":
                assistant.find_similar_colleges()

            elif choice == "4":
                assistant.ask_rag_question()

            elif choice == "5":
                print("Exiting...")
                break

            else:
                print("Invalid choice! Please select 1-5.")

        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try again with valid inputs.\n")