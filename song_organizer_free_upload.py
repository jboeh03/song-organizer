# song_organizer_free_upload.py
import streamlit as st
import whisper
import tempfile
import subprocess
import os
from collections import Counter
import re

st.set_page_config(page_title="Song Organizer", layout="wide")

st.title("🎵 Song Organizer - Audio to Titles")

# Load Whisper model (CPU-friendly)
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")  # change to "tiny" for faster CPU

model = load_whisper_model()

def convert_to_wav(input_file):
    tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    subprocess.run([
        "ffmpeg", "-y", "-i", input_file, "-ac", "1", "-ar", "16000", tmp_wav.name
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return tmp_wav.name

def extract_chorus_like_title(text, top_n=5):
    # Basic approach: most common phrases of 1-3 words
    words = re.findall(r"\b\w+\b", text.lower())
    ngrams = []
    for n in range(1, 4):
        for i in range(len(words)-n+1):
            ngrams.append(" ".join(words[i:i+n]))
    count = Counter(ngrams)
    most_common = [phrase for phrase, _ in count.most_common(top_n*2)]
    # Deduplicate and return top_n
    seen = set()
    suggestions = []
    for phrase in most_common:
        if phrase not in seen:
            seen.add(phrase)
            suggestions.append(phrase)
        if len(suggestions) >= top_n:
            break
    return suggestions

uploaded_files = st.file_uploader(
    "Upload your audio files", accept_multiple_files=True, type=["mp3", "wav", "m4a"]
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.subheader(uploaded_file.name)

        # Convert to WAV
        wav_path = convert_to_wav(uploaded_file)

        # Transcribe
        st.info("Transcribing audio...")
        result = model.transcribe(wav_path)
        transcription = result["text"]
        st.text_area("Transcribed Lyrics", transcription, height=200)

        # Generate suggested titles
        suggestions = extract_chorus_like_title(transcription)
        selected_title = st.selectbox(
            "Select a title for this song", ["--None--"] + suggestions, index=0
        )

        if selected_title != "--None--":
            new_filename = f"{selected_title}.wav"
            st.success(f"Suggested filename: {new_filename}")
        else:
            st.warning("No title selected yet.")

        # Optional: download renamed file
        if selected_title != "--None__":
            with open(wav_path, "rb") as f:
                st.download_button(
                    label="Download WAV with suggested title",
                    data=f,
                    file_name=new_filename,
                    mime="audio/wav"
                )
