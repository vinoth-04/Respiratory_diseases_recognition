# Installation Guide

## System Requirements

- **Python**: 3.8 or higher
- **OS**: Windows, macOS, or Linux
- **RAM**: 8GB minimum (16GB recommended for model training)
- **GPU**: Optional but recommended for faster training (NVIDIA GPU with CUDA support)

## Step-by-Step Installation

### 1. Clone Repository

```bash
git clone https://github.com/vinoth-04/Respiratory_diseases_recognition.git
cd Respiratory_diseases_recognition
```

### 2. Create Virtual Environment

#### Windows
```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS/Linux
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import librosa, numpy, tensorflow; print('✅ All dependencies installed successfully')"
```

## GPU Support (Optional)

For faster model training with GPU:

```bash
pip install tensorflow-gpu>=2.10.0
```

Verify GPU:
```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## Troubleshooting

### TensorFlow Installation Issues
```bash
pip install --force-reinstall tensorflow==2.12.0
```

### Librosa/Audio Issues
```bash
pip install librosa --force-reinstall
```

### Windows - Long Path Issues
```bash
pip install -r requirements.txt --use-deprecated=legacy-resolver
```

## Jupyter Notebook Setup

```bash
jupyter notebook
```

Then open `notebooks/model.ipynb`

## Streamlit Setup

```bash
streamlit run src/lung_app.py
```

The app will open in your browser at `http://localhost:8501`
