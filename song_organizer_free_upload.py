# song_organizer_free_upload.py

import streamlit as st
from faster_whisper import WhisperModel
import tempfile
from pydub import AudioSegment
import zipfile
from collections import Counter
import re

# -------------------------
# Load Whisper model (CPU-friendly)
# -------------------------
@st.cache_resource
def load_whisper_model():
    return WhisperModel("base")  # tiny/base/small/medium/large

model = load_whisper_model()

# -------------------------
# Convert any audio to WAV
# -------------------------
def convert_to_wav(input_file):
    tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    audio = AudioSegment.from_file(input_file)
    audio.export(tmp_wav.name, format="wav")
    return tmp_wav.name

# -------------------------
# Transcribe audio
# -------------------------
def transcribe_audio(file_path):
    segments, info = model.transcribe(file_path)
    transcription = " ".join([seg.text for seg in segments])
    return transcription

# -------------------------
# Recommend title from chorus-like repetition
# -------------------------
def recommend_title(transcription, top_n=3):
    # Split into lines or phrases
    lines = re.split(r"[.\n]", transcription)
    lines = [line.strip() for line in lines if line.strip()]
    
    # Count repeated lines
    counter = Counter(lines)
    most_common = counter.most_common(top_n)
    
    # Return the most repeated line as suggested title
    if most_common:
        return most_common[0][0]
    return "Untitled"

# -------------------------
# Streamlit UI
# -------------------------
st.title("🎵 Song Organizer")

uploaded_files = st.file_uploader(
    "Upload one or more audio files",
    type=["mp3", "wav", "m4a", "flac"],
    accept_multiple_files=True
)

if uploaded_files:
    st.info("Processing files... This may take a moment.")
    
    for uploaded_file in uploaded_files:
        st.subheader(f"File: {uploaded_file.name}")
        
        # Convert to WAV
        wav_path = convert_to_wav(uploaded_file)
        
        # Transcribe
        transcription = transcribe_audio(wav_path)
        st.text_area("Transcription", transcription, height=200)
        
        # Recommend title
        suggested_title = recommend_title(transcription)
        st.text_input("Suggested Title", value=suggested_title)
