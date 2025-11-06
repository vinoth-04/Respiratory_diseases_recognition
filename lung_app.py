import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import joblib
import pickle
import pandas as pd
from io import BytesIO
import datetime

# Page configuration
st.set_page_config(
    page_title="Lung Condition Predictor",
    page_icon="🫁",
    layout="wide"
)

# Title
st.title("🫁 Lung Condition Prediction from Audio")
st.markdown("Upload a respiratory sound recording to predict the lung condition")

# Patient name input
patient_name = st.text_input("Patient Name", placeholder="Enter patient's full name")

# Load RF model
@st.cache_resource
def load_rf_model():
    try:
        rf_model = joblib.load('best_random_forest_model.pkl')
        return rf_model
    except Exception as e:
        st.error(f"Error loading Random Forest model: {str(e)}")
        return None

# Feature extraction function
def extract_features(audio_path, sr=22050):
    """Extract MFCC features from audio file"""
    try:
        audio, _ = librosa.load(audio_path, sr=sr)
        
        if len(audio) < sr * 4:
            audio = np.pad(audio, (0, sr * 4 - len(audio)), mode='constant')
        else:
            audio = audio[:sr * 4]
        
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mean = np.mean(mfccs, axis=1)
        std = np.std(mfccs, axis=1)
        min_ = np.min(mfccs, axis=1)
        max_ = np.max(mfccs, axis=1)
        features = np.concatenate([mean, std, min_, max_])
        
        return features.reshape(1, -1), audio, sr
    
    except Exception as e:
        st.error(f"Error extracting features: {str(e)}")
        return None, None, None

# Create audio visualizations
def create_audio_visualizations(audio, sr):
    """Create waveform and spectrogram plots"""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    # 1. Waveform
    ax1.plot(audio, color='blue', alpha=0.7)
    ax1.set_title('Audio Waveform', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time (samples)')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True, alpha=0.3)
    
    # 2. Spectrogram
    D = librosa.stft(audio)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    img2 = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', ax=ax2)
    ax2.set_title('Spectrogram', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Frequency (Hz)')
    plt.colorbar(img2, ax=ax2, format='%+2.0f dB')
    
    # 3. MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    img3 = librosa.display.specshow(mfccs, sr=sr, x_axis='time', ax=ax3)
    ax3.set_title('MFCCs', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Coefficients')
    plt.colorbar(img3, ax=ax3)
    
    plt.tight_layout()
    return fig

# Create PDF with report first, visualizations second
def create_medical_report_pdf(audio, sr, pred_class, pred_proba, class_names, filename, patient_name):
    """Create PDF: First page = Report, Second page = Visualizations"""
    pdf_buffer = BytesIO()
    
    with PdfPages(pdf_buffer) as pdf:
        # FIRST PAGE: Medical Report
        fig1, ax1 = plt.subplots(figsize=(8.5, 11))  # A4 size
        ax1.axis('off')
        
        # Current date
        current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Report content
        report_text = f"""
        MEDICAL REPORT - LUNG CONDITION ANALYSIS

        Patient: {patient_name if patient_name else 'Unknown Patient'}
        File: {filename}
        Analysis Date: {current_date}
        Duration: 4 seconds (analyzed)
        Sample Rate: {sr} Hz

        ┌─────────────────────────────────────────┐
        │            PREDICTION RESULTS           │
        └─────────────────────────────────────────┘

        PREDICTED CONDITION: {pred_class}
        CONFIDENCE LEVEL: {np.max(pred_proba):.2%}

        ┌─────────────────────────────────────────┐
        │        DETAILED PROBABILITIES           │
        └─────────────────────────────────────────┘
        """
        
        for i, class_name in enumerate(class_names):
            prob_percent = pred_proba[i] * 100
            marker = " → PREDICTED" if class_name == pred_class else ""
            report_text += f"\n{class_name:<20} {prob_percent:>6.2f}%{marker}"
        
        report_text += f"""

        ┌─────────────────────────────────────────┐
        │        MEDICAL INTERPRETATION           │
        └─────────────────────────────────────────┘
        """
        
        condition_info = {
            "Healthy": "Normal respiratory sounds - no abnormalities detected\nNo signs of respiratory disease observed in the audio sample.",
            "COPD": "Chronic Obstructive Pulmonary Disease\nMay show wheezing or prolonged expiration patterns in respiratory sounds.",
            "Asthma": "Asthma\nMay show wheezing during expiration, indicating airway constriction.",
            "Bronchiectasis": "Bronchiectasis\nMay show crackles or coarse breath sounds indicating airway damage.",
            "Other": "Other respiratory condition\nRequires further medical evaluation and clinical correlation."
        }
        
        interpretation = condition_info.get(pred_class, "No specific interpretation available")
        report_text += f"\n{interpretation}"
        
        report_text += f"""

        ┌─────────────────────────────────────────┐
        │            ANALYSIS NOTES               │
        └─────────────────────────────────────────┘
        • This analysis is based on respiratory sound patterns
        • Results should be used as a screening tool only
        • Always consult with a healthcare professional
        • Audio quality and recording conditions may affect results
        • Model accuracy: 86.87% (Random Forest classifier)
        """
        
        ax1.text(0.05, 0.95, report_text, fontsize=10, verticalalignment='top',
                 horizontalalignment='left', transform=ax1.transAxes, fontfamily='monospace')
        
        pdf.savefig(fig1, bbox_inches='tight')
        plt.close(fig1)
        
        # SECOND PAGE: Visualizations
        fig2 = create_audio_visualizations(audio, sr)
        pdf.savefig(fig2, bbox_inches='tight')
        plt.close(fig2)
    
    pdf_buffer.seek(0)
    return pdf_buffer

# Main app
def main():
    st.sidebar.header("Model Information")
    st.sidebar.markdown("""
    - **Model**: Random Forest (86.87%)
    - **Input**: Respiratory sound (WAV)
    - **Output**: Lung condition prediction
    """)
    
    uploaded_file = st.file_uploader(
        "Upload Respiratory Sound File (WAV format)", 
        type=['wav', 'mp3', 'm4a']
    )
    
    if uploaded_file is not None:
        if not patient_name:
            st.warning("⚠️ Please enter the patient's name above")
            return
        
        rf_model = load_rf_model()
        if rf_model is None:
            return
        
        # Extract features
        with st.spinner('Processing audio...'):
            features, audio_data, sample_rate = extract_features(uploaded_file)
        
        if features is not None and audio_data is not None:
            # Make prediction
            pred_proba = rf_model.predict_proba(features)[0]
            pred_class = rf_model.predict(features)[0]
            
            try:
                with open('class_names.pkl', 'rb') as f:
                    class_names = pickle.load(f)
            except:
                class_names = rf_model.classes_ if hasattr(rf_model, 'classes_') else ['Class 1', 'Class 2', 'Class 3']
            
            # LEFT COLUMN: Prediction | RIGHT COLUMN: Visualizations
            col_left, col_right = st.columns([1, 2])
            
            # LEFT COLUMN - PREDICTION ONLY
            with col_left:
                st.subheader("📊 Prediction Results")
                st.metric("Patient", patient_name)
                st.metric("Diagnosis", pred_class)
                
                max_proba = np.max(pred_proba)
                st.metric("Confidence", f"{max_proba:.2%}")
                
                # Show only the predicted disease
                st.subheader("Predicted Disease")
                st.success(f"✅ **{pred_class}**")
                st.info(f"Confidence: **{max_proba:.2%}**")
                
                # Medical interpretation
                condition_info = {
                    "Healthy": "Normal respiratory sounds",
                    "COPD": "Chronic Obstructive Pulmonary Disease",
                    "Asthma": "Asthma - may show wheezing",
                    "Bronchiectasis": "Bronchiectasis - may show crackles",
                    "Other": "Other respiratory condition"
                }
                interpretation = condition_info.get(pred_class, "No interpretation available")
                st.write(f"**Medical Info:** {interpretation}")
                
                # PDF Download button
                pdf_buffer = create_medical_report_pdf(
                    audio_data, sample_rate, pred_class, pred_proba, class_names, 
                    uploaded_file.name, patient_name
                )
                
                st.download_button(
                    label="📥 Download Medical Report (PDF)",
                    data=pdf_buffer,
                    file_name=f"lung_prediction_report_{patient_name.replace(' ', '_')}_{uploaded_file.name.replace('.wav', '')}.pdf",
                    mime="application/pdf"
                )
            
            # RIGHT COLUMN - VISUALIZATIONS ONLY
            with col_right:
                st.subheader("🎨 Audio Analysis")
                st.audio(uploaded_file, format='audio/wav')
                
                with st.spinner('Generating visualizations...'):
                    fig = create_audio_visualizations(audio_data, sample_rate)
                    st.pyplot(fig)
                    plt.close(fig)
        
        else:
            st.error("Failed to process audio file.")

if __name__ == "__main__":
    main()