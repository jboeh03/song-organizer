import streamlit as st
from faster_whisper import WhisperModel
import tempfile
from pydub import AudioSegment
import zipfile
from collections import Counter
import re
import os

st.set_page_config(page_title="Song Organizer", layout="wide")

st.title("🎵 Song Organizer")

# -------------------------------
# Load Whisper model (CPU-friendly)
# -------------------------------
@st.cache_resource
def load_whisper_model():
    # Change to "tiny", "base", "small", "medium", "large" as needed
    return WhisperModel("base")  

model = load_whisper_model()

# -------------------------------
# Convert audio to WAV
# -------------------------------
def convert_to_wav(input_file):
    tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    audio = AudioSegment.from_file(input_file)
    audio.export(tmp_wav.name, format="wav")
    return tmp_wav.name

# -------------------------------
# Transcribe audio
# -------------------------------
def transcribe_audio(file_path):
    segments, info = model.transcribe(file_path)
    full_text = " ".join([segment.text for segment in segments])
    return full_text

# -------------------------------
# Extract common words (simple lyrics analysis)
# -------------------------------
def extract_common_words(text, top_n=10):
    words = re.findall(r"\b\w+\b", text.lower())
    counter = Counter(words)
    return counter.most_common(top_n)

# -------------------------------
# Streamlit File Uploader
# -------------------------------
uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a", "ogg", "flac"])

if uploaded_file:
    st.info("Converting audio...")
    wav_file = convert_to_wav(uploaded_file)

    st.info("Transcribing audio (this may take a while)...")
    transcription = transcribe_audio(wav_file)
    st.success("Transcription completed!")

    st.subheader("🎤 Transcription")
    st.text_area("Lyrics / Text", transcription, height=300)

    st.subheader("📊 Most Common Words")
    common_words = extract_common_words(transcription)
    st.table(common_words)

    # Optional: allow download of transcription
    st.download_button(
        label="Download Transcription",
        data=transcription,
        file_name=os.path.splitext(uploaded_file.name)[0] + "_transcription.txt",
        mime="text/plain"
    )
