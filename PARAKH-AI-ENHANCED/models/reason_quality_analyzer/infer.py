import os
import joblib
from sentence_transformers import SentenceTransformer
import numpy as np

class ReasonQualityAnalyzer:
    def __init__(self, model_dir):
        """
        model_dir: Folder where trained model and sentence transformer are saved.
        """
        self.model = joblib.load(os.path.join(model_dir, 'reason_quality_classifier.pkl'))
        self.encoder = SentenceTransformer(os.path.join(model_dir, 'sentence_transformer'))

        # Define labels mapping
        self.label_mapping = {
            0: "strong",
            1: "confused",
            2: "unclear"
        }

    def predict_reason_quality(self, reason_text):
        """
        Predicts if the student's reason is strong, confused, or unclear.

        Parameters:
        - reason_text: The explanation written by the student.

        Returns:
        - Predicted label: "strong", "confused", or "unclear"
        """
        reason_embedded = self.encoder.encode([reason_text])
        prediction = self.model.predict(reason_embedded)
        label = self.label_mapping.get(prediction[0], "unknown")
        return label
