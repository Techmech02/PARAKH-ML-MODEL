import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer
import yaml

from config.settings import Config

cfg = Config(config_path='config/config.yaml')  # Give correct relative path

# Now use cfg.get() as needed
data_path = cfg.get('paths', 'bloom_labeled_mcqs')
model_save_path = cfg.get('paths', 'bloom_model')
encoder_name = cfg.get('models', 'embedding_model')




# Load data
df = pd.read_csv(data_path)
questions = df['question'].astype(str).tolist()
labels = df['bloom_level'].tolist()

# Embedding
print(f"Loading embedding model: {encoder_name}")
encoder = SentenceTransformer(encoder_name)
X = encoder.encode(questions, show_progress_bar=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Train classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model + encoder
os.makedirs(model_save_path, exist_ok=True)
joblib.dump(clf, os.path.join(model_save_path, 'bloom_classifier.pkl'))
encoder.save(os.path.join(model_save_path, 'sentence_transformer'))

print(f"\n Model saved to {model_save_path}")
