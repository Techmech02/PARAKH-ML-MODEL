from models.answer_analysis.check_answers import AnswerAnalyzer
from models.reason_quality_analyzer.infer import ReasonQualityAnalyzer

class DoubtDetector:
    def __init__(self, mcq_data_path, reason_model_path):
        self.answer_checker = AnswerAnalyzer(mcq_data_path)
        self.reason_analyzer = ReasonQualityAnalyzer(reason_model_path)

    def detect_doubt(self, question_text, selected_option, reason_text):
        """
        Detects doubt level based on student's answer and reason.

        Parameters:
        - question_text: The question attempted by student.
        - selected_option: Student's selected option.
        - reason_text: Student's explanation.

        Returns:
        - "no_doubt", "low_doubt", or "high_doubt"
        """
        # Check if answer is correct or wrong
        answer_result = self.answer_checker.check_answer(question_text, selected_option)  # 1=correct, 0=wrong

        # Analyze the reason
        reason_quality = self.reason_analyzer.predict_reason_quality(reason_text)  # strong / confused / unclear

        # Apply logic
        if answer_result == 1 and reason_quality == "strong":
            return "no_doubt"
        elif (answer_result == 1 and reason_quality in ["confused", "unclear"]) or (answer_result == 0 and reason_quality == "strong"):
            return "low_doubt"
        elif answer_result == 0 and reason_quality in ["confused", "unclear"]:
            return "high_doubt"
        else:
            return "low_doubt"  # Safe fallback (should not occur)

