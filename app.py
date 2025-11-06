import streamlit as st
import numpy as np
import pickle
import sounddevice as sd
from scipy.io.wavfile import write
import tempfile
import os
from extract_features import extract_features

# ==============================
# Load Model
# ==============================
with open("voice_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🎙️ Sistem Deteksi Suara - Buka / Tutup")
st.markdown("Upload file suara atau rekam langsung untuk mendeteksi kata.")

# ==============================
# Fungsi Prediksi
# ==============================
def predict(file_path):
    feat = extract_features(file_path).reshape(1, -1)
    pred = model.predict(feat)[0]
    if pred == 0:
        return "🔓 BUKA"
    elif pred == 1:
        return "🔒 TUTUP"
    else:
        return "❓ Tidak Dikenali"

# ==============================
# Upload File
# ==============================
st.header("📁 Upload File Suara")
uploaded_file = st.file_uploader("Upload file .wav", type=["wav"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        st.audio(tmp_file.name)
        if st.button("🔍 Deteksi dari File"):
            hasil = predict(tmp_file.name)
            st.success(f"Hasil Deteksi: {hasil}")

# ==============================
# Rekam Suara Langsung
# ==============================
st.header("🎤 Rekam Langsung")
duration = st.slider("Durasi Rekaman (detik):", 1, 5, 2)

if st.button("🎙️ Mulai Rekam"):
    st.info("Sedang merekam... Silakan ucapkan 'buka' atau 'tutup'")
    fs = 16000
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    
    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    write(temp_wav.name, fs, recording)
    st.audio(temp_wav.name)
    
    hasil = predict(temp_wav.name)
    st.success(f"Hasil Deteksi: {hasil}")
