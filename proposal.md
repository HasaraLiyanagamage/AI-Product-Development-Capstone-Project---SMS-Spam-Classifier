# Project Proposal — SMS Spam Classifier

## 1. Problem Statement
- Build an AI product that classifies SMS messages as spam or ham (not spam).
- Target users: individuals and small teams needing a lightweight spam detection demo or filter.
- Value: Demonstrates a common NLP classification use-case with measurable impact and a clear UI.

## 2. Dataset
- Primary: SMS Spam Collection (public dataset)
  - UCI page: https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
  - Raw mirrors referenced inside `ml_pipeline/train.py` for automated download.
- Data policy: purely public, no PII; for educational demonstration only.

## 3. ML Model
- Text classification pipeline using scikit-learn:
  - `TfidfVectorizer` for feature extraction (unigrams + bigrams, stopwords=english).
  - `LogisticRegression` (liblinear) with class_weight="balanced" for robust baseline.
- Metrics: Accuracy, classification report (precision/recall/F1), confusion matrix image.
- Model artifact stored as `backend/model/sms_spam_model.joblib`.

## 4. Tech Stack
- **Backend**: Python, FastAPI, Uvicorn; serves `POST /predict` and static frontend files.
- **Frontend**: Vanilla HTML/CSS/JS for simplicity and portability.
- **ML**: scikit-learn, pandas, numpy; seaborn/matplotlib for visualization.
- **Packaging**: `requirements.txt`; optional containerization for deployment.

## 5. Scope & Milestones
- M1: Pipeline & metrics (train + evaluate + save model).
- M2: REST API for predictions (model loaded at startup, probability output).
- M3: Frontend UI to submit message and display prediction + confidence.
- M4: Documentation (README, usage guide, screenshots).

## 6. Risks & Mitigations
- Dataset availability: use multiple mirrors and a small fallback sample for first-run training.
- Class imbalance: use class_weight and proper evaluation metrics.
- Deployment: serve static frontend from FastAPI; option to add CORS if hosting separately.

## 7. Success Criteria
- Working UI that calls the API and displays prediction.
- Reproducible training script producing metrics and a model artifact.
- Clean repo structure with clear instructions.
