import pandas as pd

class AnswerAnalyzer:
    def __init__(self, mcq_data_path):
    
        self.mcq_df = pd.read_csv(mcq_data_path)
        self.mcq_df['question'] = self.mcq_df['question'].astype(str)

    def check_answer(self, question_text, selected_option):
    
        question_row = self.mcq_df[self.mcq_df['question'] == question_text]

        if question_row.empty:
            raise ValueError("Question not found in MCQ dataset.")

        correct_option = question_row['correct_option'].values[0]

        if selected_option.strip().lower() == correct_option.strip().lower():
            return 1  # Correct
        else:
            return 0  # Wrong
