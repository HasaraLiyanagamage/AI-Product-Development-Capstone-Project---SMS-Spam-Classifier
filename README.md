# AI Product Development Capstone Project — SMS Spam Classifier

## Overview
- **Problem**: Classify SMS messages as spam or ham (not spam).
- **Users**: Anyone needing a quick filter or demo for spam detection.
- **Deliverable**: End-to-end product with ML pipeline, backend API, and frontend UI.

## Architecture
- **ML Pipeline** (`ml_pipeline/train.py`):
  - Loads dataset (downloads public dataset automatically; falls back to a small built-in sample so it always runs).
  - Trains a `TfidfVectorizer + LogisticRegression` pipeline.
  - Evaluates on a holdout set and saves metrics and a confusion matrix.
  - Saves the trained pipeline to `backend/model/sms_spam_model.joblib`.
- **Backend** (`backend/app.py`):
  - FastAPI app exposing `POST /predict`.
  - Loads the saved model at startup; if missing, trains it automatically.
  - Serves the frontend static files.
- **Frontend** (`frontend/`):
  - Simple HTML/CSS/JS UI to enter an SMS and view the prediction.

## Quickstart
1. Ensure Python 3 is available:
   ```bash
   python3 --version
   ```
2. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # zsh/bash
   python3 -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
   If you prefer not to activate the venv, prefix commands with `.venv/bin/`:
   ```bash
   .venv/bin/pip install -r requirements.txt
   ```
3. (Optional) Train the model explicitly:
   ```bash
   .venv/bin/python ml_pipeline/train.py
   ```
   This writes the model to `backend/model/sms_spam_model.joblib` and metrics to `artifacts/`.
4. Run the backend + frontend server:
   ```bash
   .venv/bin/uvicorn backend.app:app --reload
   ```
5. Open the app:
   - http://127.0.0.1:8000/

## API
- **Endpoint**: `POST /predict`
- **Request JSON**:
  ```json
  { "text": "Free entry in 2 a weekly competition..." }
  ```
- **Response JSON**:
  ```json
  { "label": "spam", "probability": 0.97 }
  ```

## Dataset
- Primary: SMS Spam Collection (publicly available)
  - UCI Link: https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
  - Raw (mirrors used in training script) are embedded in `ml_pipeline/train.py`.

## Screenshots to Capture (as required by the brief)
- **Frontend before prediction** (landing page with the input form).
- **Frontend after prediction** (shows label and probability).
- **ML pipeline results**: include `artifacts/metrics.json` and `artifacts/confusion_matrix.png`.

## Repo Structure
```
/ (project root)
├── README.md
├── requirements.txt
├── proposal.md
├── artifacts/                      # metrics & plots (created by training)
├── backend/
│   ├── app.py
│   └── model/
│       └── sms_spam_model.joblib   # created by training
├── frontend/
│   ├── index.html
│   ├── script.js
│   └── style.css
└── ml_pipeline/
    ├── __init__.py
    └── train.py
```

## Notes
- The backend will auto-train a model on first run if none exists.
- You can replace the dataset path in `ml_pipeline/train.py` to use your own data.
- For deployment, you can containerize with Uvicorn/Gunicorn or host on a PaaS.

## Troubleshooting
- **pip/python not found** (e.g., `zsh: command not found: pip` or `python`):
  - Use Python 3 explicitly: `python3 -m pip ...` and `python3 -m venv .venv`.
  - If `python3` is missing, install Python 3:
    - Homebrew:
      ```bash
      /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
      brew install python
      ```
    - Or install from https://www.python.org/downloads/
- **uvicorn not found**: ensure dependencies are installed in the venv or use `.venv/bin/uvicorn`.
- **PATH issues on macOS/Homebrew (Apple Silicon)**: ensure Homebrew is on PATH:
  ```bash
  echo 'export PATH="/opt/homebrew/bin:$PATH"' >> ~/.zprofile
  exec zsh
  ```
# AI-Product-Development-Capstone-Project---SMS-Spam-Classifier
