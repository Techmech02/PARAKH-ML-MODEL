import pandas as pd
import random

class AdaptiveQuizGenerator:
    def __init__(self, mcq_data_path):
        """
        mcq_data_path: Path to MCQs CSV file (must contain columns: question, topic, difficulty_level)
        """
        self.mcq_df = pd.read_csv(mcq_data_path)
        self.mcq_df['topic'] = self.mcq_df['topic'].astype(str)

    def generate_quiz(self, doubt_analysis, num_questions=10):
        """
        Generates a personalized quiz based on doubt analysis.

        Parameters:
        - doubt_analysis: Dict of {topic: doubt_level} for the student
        - num_questions: Total questions to generate

        Returns:
        - List of selected MCQs
        """
        selected_questions = []

        # Priority - high_doubt > low_doubt > no_doubt
        priority = {"high_doubt": 3, "low_doubt": 2, "no_doubt": 1}

        # Sort topics by doubt level priority
        sorted_topics = sorted(doubt_analysis.items(), key=lambda x: priority[x[1]], reverse=True)

        # Loop through topics by doubt level
        for topic, doubt_level in sorted_topics:
            topic_mcqs = self.mcq_df[self.mcq_df['topic'] == topic]

            if topic_mcqs.empty:
                continue

            if doubt_level == "high_doubt":
                # Prefer harder MCQs (Application, Analysis, Evaluation)
                hard_mcqs = topic_mcqs[topic_mcqs['difficulty_level'].isin(["Application", "Analysis", "Evaluation"])]
                questions = hard_mcqs.sample(min(3, len(hard_mcqs)), random_state=42).to_dict(orient='records')
            elif doubt_level == "low_doubt":
                # Prefer medium MCQs (Understanding)
                medium_mcqs = topic_mcqs[topic_mcqs['difficulty_level'] == "Understanding"]
                questions = medium_mcqs.sample(min(2, len(medium_mcqs)), random_state=42).to_dict(orient='records')
            else:
                # Pick maybe 1 easy MCQ (Knowledge)
                easy_mcqs = topic_mcqs[topic_mcqs['difficulty_level'] == "Knowledge"]
                questions = easy_mcqs.sample(min(1, len(easy_mcqs)), random_state=42).to_dict(orient='records')

            selected_questions.extend(questions)

            # Stop if enough questions selected
            if len(selected_questions) >= num_questions:
                break

        # Final trim
        selected_questions = selected_questions[:num_questions]

        return selected_questions
