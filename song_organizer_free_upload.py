import os
import warnings
import streamlit as st
import whisper
import tempfile
from pydub import AudioSegment
import zipfile
from collections import Counter
import re

# --- Silence warnings ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU.*")

st.set_page_config(page_title="🎵 Song Idea Organizer", layout="wide")
st.title("🎶 Song Idea Organizer — Offline Phrase Detection & Titles")
st.write(
    "Upload your song ideas, review the transcripts with repeated phrases highlighted, "
    "approve/edit suggested titles, then download all as a ZIP."
)

# --- Load Whisper model ---
@st.cache_resource
def load_whisper():
    return whisper.load_model("medium")  # medium = best tradeoff for CPU

whisper_model = load_whisper()

# --- Helper functions ---
def preprocess_audio(file_path):
    sound = AudioSegment.from_file(file_path)
    sound = sound.set_channels(1)
    sound = sound.set_frame_rate(16000)
    sound.export(file_path, format="wav")

def split_audio(file_path, chunk_length_ms=60000):
    audio = AudioSegment.from_file(file_path)
    chunks = []
    for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i:i+chunk_length_ms]
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        chunk.export(tmp_file.name, format="wav")
        chunks.append(tmp_file.name)
    return chunks

def transcribe_audio(file_path):
    preprocess_audio(file_path)
    chunks = split_audio(file_path)
    full_text = ""
    for chunk_path in chunks:
        result = whisper_model.transcribe(chunk_path)
        full_text += result["text"] + "\n"
    return full_text.strip()

def detect_repeated_phrases(transcript, min_words=2, max_words=5, top_n=1):
    words = transcript.split()
    phrases = []
    for n in range(min_words, max_words+1):
        for i in range(len(words)-n+1):
            phrase = " ".join(words[i:i+n])
            phrases.append(phrase)
    counter = Counter(phrases)
    most_common = [phrase for phrase, count in counter.most_common(top_n)]
    return most_common

def highlight_phrases(transcript, phrases):
    highlighted = transcript
    for phrase in phrases:
        if phrase:
            escaped = re.escape(phrase)
            highlighted = re.sub(
                escaped,
                f"<mark style='background-color: #FFD700'>{phrase}</mark>",
                highlighted,
                flags=re.IGNORECASE
            )
    return highlighted

def clean_filename(title, max_length=50):
    safe = "".join(c for c in title if c.isalnum() or c in (" ", "-", "_")).strip()
    safe = safe.replace(" ", "_")
    return safe[:max_length] or "Untitled"

# --- File upload ---
uploaded_files = st.file_uploader(
    "🎧 Upload your song idea files (.wav, .mp3, .m4a):",
    type=["wav", "mp3", "m4a"],
    accept_multiple_files=True
)

if uploaded_files:
    st.write(f"Processing {len(uploaded_files)} files...")
    results = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for file in uploaded_files:
            st.header(file.name)

            # Save temp
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(file.read())
                tmp_path = tmp.name

            # Transcribe
            transcript = transcribe_audio(tmp_path)

            # Detect repeated phrases (chorus)
            repeated_phrases = detect_repeated_phrases(transcript, min_words=2, max_words=5, top_n=1)

            # Highlight repeated phrases
            highlighted_transcript = highlight_phrases(transcript, repeated_phrases)
            st.subheader("📝 Transcript (Repeated phrases highlighted)")
            st.markdown(f"<div style='white-space: pre-wrap'>{highlighted_transcript}</div>", unsafe_allow_html=True)

            # Title suggestions = top repeated phrase
            first_suggestion = repeated_phrases[0] if repeated_phrases else "Untitled"

            st.subheader("🎯 Suggested Titles")
            title_choice = st.radio("Pick a title", options=[first_suggestion, "Custom Title"], key=file.name)
            if title_choice == "Custom Title":
                title_choice = st.text_input("Type your custom title:", key=f"custom_{file.name}", value=first_suggestion)

            # Export audio with safe filename
            safe_title = clean_filename(title_choice)
            new_filename = f"{safe_title}.wav"
            new_path = os.path.join(temp_dir, new_filename)
            AudioSegment.from_file(tmp_path).export(new_path, format="wav")
            results.append({"original": file.name, "new": new_filename, "path": new_path})

        # ZIP download
        zip_path = os.path.join(temp_dir, "renamed_songs.zip")
        with zipfile.ZipFile(zip_path, "w") as zipf:
            for r in results:
                zipf.write(r["path"], arcname=r["new"])

        st.subheader("✅ Summary of Renamed Files")
        for r in results:
            st.write(f"- **{r['original']}** → **{r['new']}**")

        with open(zip_path, "rb") as f:
            st.download_button(
                label="⬇️ Download All as ZIP",
                data=f,
                file_name="renamed_song_ideas.zip",
                mime="application/zip"
            )
else:
    st.info("Upload one or more audio files to get started.")
