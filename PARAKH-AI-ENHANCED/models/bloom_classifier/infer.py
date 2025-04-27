import joblib
from sentence_transformers import SentenceTransformer
import pandas as pd
import os
from config.settings import Config

# Load configuration
cfg = Config(config_path='config/config.yaml')
model_save_path = cfg.get('paths', 'bloom_model')

# Load the trained model and encoder
clf = joblib.load(os.path.join(model_save_path, 'bloom_classifier.pkl'))
encoder = SentenceTransformer(os.path.join(model_save_path, 'sentence_transformer'))

def predict_bloom_level(questions):
    # Embed questions using the pre-trained encoder
    embeddings = encoder.encode(questions)

    # Predict Bloom's level using the trained classifier
    predictions = clf.predict(embeddings)
    return predictions

def main():
    # Load new MCQs or student responses (can replace with your actual data source)
    new_data_path = 'data/new_mcqs.csv'  # Example: path to new questions
    df = pd.read_csv(new_data_path)
    questions = df['question'].tolist()

    # Predict Bloom's levels
    predictions = predict_bloom_level(questions)

    # Print predictions (or do further processing)
    for i, question in enumerate(questions):
        print(f"Question: {question}\nPredicted Bloom Level: {predictions[i]}\n")

if __name__ == "__main__":
    main()
