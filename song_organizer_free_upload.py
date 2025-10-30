import streamlit as st
from faster_whisper import WhisperModel
import tempfile
import subprocess
import os
import zipfile
from collections import Counter
import re

# Load Whisper model (CPU-friendly)
@st.cache_resource
def load_whisper_model():
    return WhisperModel("base")  # change to "tiny" for faster CPU processing

model = load_whisper_model()

# Convert any audio file to WAV using ffmpeg
def convert_to_wav(input_file):
    tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp_wav.close()  # Close the file so ffmpeg can write to it
    subprocess.run(["ffmpeg", "-y", "-i", input_file, tmp_wav.name], check=True)
    return tmp_wav.name

def transcribe_audio(file_path):
    wav_path = convert_to_wav(file_path)
    segments, info = model.transcribe(wav_path)
    os.remove(wav_path)
    full_text = " ".join([segment.text for segment in segments])
    return full_text

st.title("Song Organizer")

uploaded_file = st.file_uploader("Upload audio files", type=["mp3", "wav", "m4a", "flac"], accept_multiple_files=True)

if uploaded_file:
    for uploaded in uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1]) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        transcription = transcribe_audio(tmp_path)
        st.subheader(f"Transcription for {uploaded.name}")
        st.text_area("Transcription", transcription, height=200)

        os.remove(tmp_path)
