import streamlit as st
import numpy as np
import pickle
import tempfile
import os
from extract_features import extract_features
from st_audiorec import st_audiorec  # komponen untuk rekam di browser

# ==============================
# Load Model
# ==============================
with open("voice_model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Deteksi Suara Buka/Tutup", page_icon="🎙️")
st.title("🎙️ Sistem Deteksi Suara - Buka / Tutup")
st.markdown("Upload file suara atau rekam langsung dari browser untuk mendeteksi kata.")

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
# Rekam Langsung (Browser Mic)
# ==============================
st.header("🎤 Rekam Langsung dari Browser")

wav_audio_data = st_audiorec()

if wav_audio_data is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(wav_audio_data)
        st.audio(temp_audio.name)
        if st.button("🎯 Deteksi dari Rekaman"):
            hasil = predict(temp_audio.name)
            st.success(f"Hasil Deteksi: {hasil}")
