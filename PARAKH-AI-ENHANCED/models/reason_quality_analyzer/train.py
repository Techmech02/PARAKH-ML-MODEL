import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer
from config.settings import Config

# Load config
cfg = Config()

# Paths
data_path = cfg.get('paths', 'reason_labeled_data')
model_save_path = cfg.get('paths', 'reason_model')
encoder_name = cfg.get('models', 'embedding_model')

# Load data
df = pd.read_csv(data_path)
reasons = df['reason'].astype(str).tolist()
labels = df['quality'].tolist()

# Encode labels: strong -> 0, confused -> 1, unclear -> 2
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Save label mapping for future use
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print(f"Label Mapping: {label_mapping}")

# Embedding
print(f"Loading embedding model: {encoder_name}")
encoder = SentenceTransformer(encoder_name)
X = encoder.encode(reasons, show_progress_bar=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, labels_encoded,
    test_size=cfg.get('training', 'test_size'),
    random_state=cfg.get('training', 'random_seed')
)

# Train classifier
clf = LogisticRegression(max_iter=cfg.get('training', 'max_iter'), multi_class='multinomial', solver='lbfgs')
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Save model + encoder
os.makedirs(model_save_path, exist_ok=True)
joblib.dump(clf, os.path.join(model_save_path, 'reason_quality_classifier.pkl'))
encoder.save(os.path.join(model_save_path, 'sentence_transformer'))

print(f"\n✅ Reason Quality Model (3-class) saved to {model_save_path}")
