import pandas as pd
from scripts.preprocess_data import preprocess_data
from models.answer_analysis.check_answers import AnswerAnalyzer
from models.reason_quality_analyzer.infer import ReasonQualityAnalyzer
from models.doubt_detection.detect_doubts import DoubtDetector
from models.weak_area_profiler.profiler import WeakAreaProfiler
from models.adaptive_quiz_generator.generate_quiz import AdaptiveQuizGenerator
from models.bloom_classifier.infer import predict_bloom_level  # Import Bloom's level classifier

# Paths
MCQ_DATA_PATH = 'data/mcqs/raw_mcqs.csv'  # Raw MCQ file path
STUDENT_ATTEMPTS_PATH = 'data/student_attempts/pre_assessment.csv'  # Pre-assessment file path
PREPROCESSED_MCQ_PATH = 'data/mcqs/blooms_labeled_mcqs.csv'  # Preprocessed MCQ file path
PREPROCESSED_ATTEMPTS_PATH = 'data/student_attempts/processed_pre_assessment.csv'  # Processed pre-assessment path
REASON_MODEL_PATH = 'models/reason_quality_analyzer/model/'

def main():
    print("Starting Inference Pipeline...")

    # Step 1: Preprocess MCQs and Student Attempts
    print("Preprocessing MCQs and Student Attempts...")
    
    # Preprocess the raw MCQs into the labeled format (including Bloom's Taxonomy classification)
    preprocess_data(input_file=MCQ_DATA_PATH, output_file=PREPROCESSED_MCQ_PATH)
    
    # You can also preprocess student attempts if needed (e.g., clean missing data, add any required columns)
    student_df = pd.read_csv(STUDENT_ATTEMPTS_PATH,on_bad_lines='skip')
    student_df.to_csv(PREPROCESSED_ATTEMPTS_PATH, index=False,on_bad_lines='skip')
    
    print(f"Preprocessed MCQs saved at {PREPROCESSED_MCQ_PATH}")
    print(f"Preprocessed Student Attempts saved at {PREPROCESSED_ATTEMPTS_PATH}")

    # Step 2: Load Preprocessed Data
    mcq_df = pd.read_csv(PREPROCESSED_MCQ_PATH,on_bad_lines='skip')
    student_df = pd.read_csv(PREPROCESSED_ATTEMPTS_PATH,on_bad_lines='skip')
    
    print(f"Loaded {len(mcq_df)} MCQs.")
    print(f"Loaded {len(student_df)} student attempts.")

    # Step 3: Initialize models
    doubt_detector = DoubtDetector(mcq_data_path=PREPROCESSED_MCQ_PATH, reason_model_path=REASON_MODEL_PATH)
    weak_profiler = WeakAreaProfiler()
    quiz_generator = AdaptiveQuizGenerator(mcq_data_path=PREPROCESSED_MCQ_PATH)

    # Step 4: Analyze each student attempt
    analyzed_attempts = []
    
    for idx, row in student_df.iterrows():
        question = row['question']
        selected_option = row['selected_option']
        reason_text = row['reason']
        topic = row['topic']  # Assuming topic is already included in pre-assessment data

        # Step 5: Detect doubts in the student’s answer
        doubt_level = doubt_detector.detect_doubt(question, selected_option, reason_text)

        # Step 6: Classify Bloom's level using the infer.py model
        bloom_level = predict_bloom_level([question])[0]  # List of one question

        analyzed_attempts.append({
            "question": question,
            "topic": topic,
            "doubt_level": doubt_level,
            "bloom_level": bloom_level  # Store the Bloom level prediction
        })

    print("Doubt detection and Bloom level classification completed for all questions.")

    # Step 7: Build Weak Area Profile
    weak_profile = weak_profiler.build_weak_profile(analyzed_attempts)
    print(f"Weak Area Profile: {weak_profile}")

    # Step 8: Generate Main Quiz based on the Weak Area Profile
    main_quiz = quiz_generator.generate_quiz(weak_profile, num_questions=10)
    
    print("\nGenerated Personalized Main Quiz:")
    for idx, q in enumerate(main_quiz):
        print(f"{idx+1}. {q['question']} [{q['difficulty_level']}]")

if __name__ == "__main__":
    main()
# This script serves as an inference pipeline that integrates all the components of the PARAKH AI system.
# It preprocesses the data, analyzes student attempts, detects doubts, classifies Bloom's levels, builds a weak area profile,