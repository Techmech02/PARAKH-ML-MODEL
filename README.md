# PARAKH-ML-MODEL 🧠

> The machine-learning brain behind **PARAKH** — an AI-driven adaptive assessment platform that understands *what* a student knows and *where* they struggle, then builds the right quiz to help them improve.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Sentence-Transformers](https://img.shields.io/badge/Sentence--Transformers-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/status-research%2FWIP-blue?style=flat-square)

---

## ✨ What it does

PARAKH turns raw MCQs and student attempts into **personalized, adaptive quizzes**. The pipeline chains together five ML components:

| Module | Path | What it does |
| --- | --- | --- |
| 🏷️ **Bloom's Taxonomy Classifier** | `models/bloom_classifier/` | Classifies each question into one of the 6 Bloom's cognitive levels (Remember → Create) using Sentence-BERT (`all-MiniLM-L6-v2`) embeddings + a logistic-regression head. |
| ✅ **Answer Analysis** | `models/answer_analysis/` | Scores student responses and flags incorrect/uncertain answers. |
| 📝 **Reason-Quality Analyzer** | `models/reason_quality_analyzer/` | Evaluates the *quality* of a student's written reasoning, not just correctness. |
| ❓ **Doubt Detection** | `models/doubt_detection/` | Detects topics/questions where the student is unsure. |
| 📉 **Weak-Area Profiler** | `models/weak_area_profiler/` | Aggregates signals into a per-topic, per-Bloom-level weakness profile. |
| 🎯 **Adaptive Quiz Generator** | `models/adaptive_quiz_generator/` | Generates a follow-up quiz targeting the student's weak areas, scaling difficulty as they improve. |

These are orchestrated end-to-end by **`scripts/inference_pipeline.py`**.

---

## 🏗️ Project structure

```
PARAKH-AI-ENHANCED/
├── config/
│   ├── config.yaml          # Paths, model names, training + quiz-gen settings
│   └── settings.py          # Config loader (Config.get / get_section)
├── data/
│   ├── mcqs/                # raw_mcqs.csv, blooms_labeled_mcqs.csv
│   └── student_attempts/    # pre-assessment answers
├── models/
│   ├── bloom_classifier/        # train.py / infer.py + saved model
│   ├── answer_analysis/         # check_answers.py
│   ├── reason_quality_analyzer/ # train.py / infer.py
│   ├── doubt_detection/         # detect_doubts.py
│   ├── weak_area_profiler/      # profiler.py
│   └── adaptive_quiz_generator/ # generate_quiz.py
├── notebooks/               # 01 data exploration → 04 quiz-generation testing
└── scripts/
    ├── preprocess_data.py   # Clean + label raw MCQs
    └── inference_pipeline.py# End-to-end orchestration
```

---

## ⚙️ How it works

```
raw MCQs ─┐
          ├─▶ preprocess_data ─▶ Bloom classifier ─▶ labeled question bank ─┐
student ──┘                                                                  │
attempts ─▶ answer analysis ─▶ reason quality ─▶ doubt detection ─▶ weak-area profile
                                                                             │
                                                          adaptive quiz generator ◀┘
                                                                             │
                                                                  personalized quiz
```

All paths and hyper-parameters live in [`config/config.yaml`](PARAKH-AI-ENHANCED/config/config.yaml) so the pipeline is fully configurable (embedding model, classifier type, test split, difficulty scaling, min MCQs per topic, etc.).

---

## 🚀 Getting started

```bash
git clone https://github.com/Techmech02/PARAKH-ML-MODEL.git
cd PARAKH-ML-MODEL/PARAKH-AI-ENHANCED

python -m venv venv && source venv/bin/activate
pip install -r requirements.txt        # sentence-transformers, scikit-learn, pandas, torch, pyyaml

# 1) Train the Bloom's classifier
python -m models.bloom_classifier.train

# 2) Run the full adaptive pipeline
python -m scripts.inference_pipeline
```

> Explore the `notebooks/` folder for a guided walkthrough: data exploration, classifier training, reason-quality training, and quiz-generation testing.

---

## 🧩 Related repositories

- **[Parakh-DL](https://github.com/Techmech02/Parakh-DL)** — Flask REST API + React front-end that serves these models.
- **[Quiz_Gen](https://github.com/Techmech02/Quiz_Gen)** — earlier quiz-generation experiments.

---

## 🗺️ Roadmap

- [ ] Publish evaluation metrics for each model
- [ ] Add a `requirements.txt` with pinned versions
- [ ] Containerize the inference pipeline
- [ ] CI for the notebooks/scripts

---

<sub>Built by <a href="https://github.com/Techmech02">@Techmech02</a>. Contributions & feedback welcome.</sub>
