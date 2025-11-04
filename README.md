# Lung Condition Predictor

A small project to predict lung conditions from respiratory audio recordings. This repository contains two simple front-ends: a Flask web app (`app.py`) and a Streamlit app (`lung_app.py`). The project demonstrates feature extraction from audio (MFCC), model prediction integration, and generation of a simple PDF medical-style report from the Streamlit UI.

## Features

- Flask-based web UI for single-file predictions (`app.py`).
- Streamlit app with richer visuals and PDF report download (`lung_app.py`).
- MFCC-based feature extraction (4 seconds of audio, 13 MFCCs with statistics).
- Example placeholders for model loading and prediction (replace mock model with your trained model or provide the `best_random_forest_model.pkl`).
- Utility to generate a PDF medical report (Streamlit app).

## Repository structure

- `app.py` - Flask web app to upload an audio file and get a prediction (mock by default).
- `lung_app.py` - Streamlit app with audio visualizations and PDF report generation.
- `model.ipynb` - Notebook used for experimentation and model training (if present).
- `requirement.txt` - Python dependencies (use `pip install -r requirement.txt`).
- `Respiratory_Sound_Database/` - Included dataset directory (audio and metadata).
- `uploads/` - Upload folder used by the Flask app (should be ignored in git).
- `best_random_forest_model.pkl` (optional) - Trained model file for production use (not included by default).
- `class_names.pkl` (optional) - Class name mapping for the model (if needed).

> Note: The repository currently uses a mock predictor in `app.py`. Replace the `mock_predict` with your real model inference code or place your serialized model file in the repository root and update the loading code.

## Requirements

Install dependencies listed in `requirement.txt` (file in repo root). If you prefer a manual install, main packages used by the apps include:

- Python 3.8+
- librosa
- numpy
- matplotlib
- flask
- streamlit
- joblib
- pandas

Install with pip:

```powershell
pip install -r requirement.txt
```

If you don't have `requirement.txt`, you can install the essentials manually:

```powershell
pip install numpy librosa matplotlib flask streamlit joblib pandas
```

## Running the apps (Windows PowerShell)

Flask app (simple web UI):

```powershell
# From project root
python app.py
# Then open http://localhost:5000 in your browser
```

Streamlit app (visuals + PDF report):

```powershell
# From project root
streamlit run lung_app.py
# Streamlit will open a local URL (usually http://localhost:8501)
```

## Model integration

- `lung_app.py` expects a trained Random Forest model serialized as `best_random_forest_model.pkl` in the project root (joblib). If you do not have it, the Streamlit app will show an error when trying to load the model.
- `app.py` currently uses the `mock_predict` function — replace it by loading your model and performing inference on `features` (the `extract_features` function returns a (1, N) feature vector). Example pseudocode:

```python
# example in app.py
from joblib import load
model = load('best_random_forest_model.pkl')
prediction = model.predict(features)
probs = model.predict_proba(features)[0]
```

## Dataset

The repository contains `Respiratory_Sound_Database/` with audio and a metadata CSV `patient_diagnosis.csv`. Use these for training or testing. The apps analyze the first 4 seconds of audio (padding or truncating as needed) and extract MFCC-based features.

## PDF Medical Report (Streamlit)

`lung_app.py` includes `create_medical_report_pdf` which creates a two-page PDF: first page contains a textual medical-style report and probabilities, second page contains waveform/spectrogram/MFCC visualizations. The PDF is offered as a download in the Streamlit UI.

## Recommended .gitignore

Add at least the following to `.gitignore` before pushing:

```
# Python
__pycache__/
*.py[cod]
*.pyo
*.pyd

# Jupyter
.ipynb_checkpoints

# Environment
.env
venv/

# App uploads and generated files
uploads/
*.pkl
*.h5
*.ckpt
*.pdf

# OS
Thumbs.db
.DS_Store
```

## How to push to GitHub (example PowerShell commands)

Replace `<your-repo-url>` with your GitHub repository remote URL (HTTPS or SSH):

```powershell
# initialize repo (if not already a git repo)
git init ;
# add files
git add . ;
# commit
git commit -m "Initial commit - Lung Condition Predictor" ;
# create main branch (optional)
git branch -M main ;
# add remote
git remote add origin <your-repo-url> ;
# push
git push -u origin main
```

If you already have a repo and remote configured, you only need to add, commit and push.

## Troubleshooting

- If `librosa` fails to load certain audio formats, convert to WAV or install `soundfile` (pip install soundfile).
- If the Streamlit app fails to find `best_random_forest_model.pkl`, ensure the file is present or update `load_rf_model` to the correct path.
- If you get port conflicts, specify a different port for Streamlit: `streamlit run lung_app.py --server.port 8502`.

## Contributing

Small improvements are welcome:
- Add model training scripts and clear model artifact placement.
- Add unit tests for `extract_features` and prediction wrappers.
- Improve README with sample screenshots.

## License

This repository is provided as-is. Add a license file as appropriate (e.g., `LICENSE` with MIT license) before publishing if you want to specify terms.

## Contact

For questions, add an issue or create a pull request in the repository.
