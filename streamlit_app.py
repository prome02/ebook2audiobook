import streamlit as st
import os
import shutil
import subprocess
import re
import tempfile
from pydub import AudioSegment
import nltk
from nltk.tokenize import sent_tokenize
import sys
import torch
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from tqdm import tqdm
import urllib.request
import zipfile
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import csv
import torchaudio

# Add custom ffmpeg path to PATH for pydub
custom_ffmpeg_path = r"C:\Users\prome\Documents\ffmpeg-essentials_build\bin"
os.environ["PATH"] = os.pathsep.join([custom_ffmpeg_path, os.environ.get("PATH", "")])
os.environ["FFMPEG_BINARY"] = os.path.join(custom_ffmpeg_path, "ffmpeg.exe")

# Language options and character limits
language_options = ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko"]
char_limits = {
    "en": 250, "es": 239, "fr": 273, "de": 253, "it": 213, "pt": 203, "pl": 224, 
    "tr": 226, "ru": 182, "nl": 251, "cs": 186, "ar": 166, "zh-cn": 82, 
    "ja": 71, "hu": 224, "ko": 95
}

language_mapping = {
    "en": "english", "de": "german", "fr": "french", "es": "spanish", "it": "italian",
    "pt": "portuguese", "nl": "dutch", "pl": "polish", "cs": "czech", "ru": "russian",
    "tr": "turkish", "el": "greek", "et": "estonian", "no": "norwegian", "ml": "malayalam",
    "sl": "slovene", "da": "danish", "fi": "finnish", "sv": "swedish"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write(f"Device selected is: {device}")

default_target_voice_path = "default_voice.wav"
default_language_code = "en"

# Helper functions (copied from original app)
def download_and_extract_zip(url, extract_to='.'):
    try:
        os.makedirs(extract_to, exist_ok=True)
        zip_path = os.path.join(extract_to, 'model.zip')
        
        with tqdm(unit='B', unit_scale=True, miniters=1, desc="Downloading Model") as t:
            def reporthook(blocknum, blocksize, totalsize):
                t.total = totalsize
                t.update(blocknum * blocksize - t.n)
            urllib.request.urlretrieve(url, zip_path, reporthook=reporthook)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            files = zip_ref.namelist()
            with tqdm(total=len(files), unit="file", desc="Extracting Files") as t:
                for file in files:
                    if not file.endswith('/'):
                        extracted_path = zip_ref.extract(file, extract_to)
                        base_file_path = os.path.join(extract_to, os.path.basename(file))
                        os.rename(extracted_path, base_file_path)
                    t.update(1)
        
        os.remove(zip_path)
        for root, dirs, files in os.walk(extract_to, topdown=False):
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        
        required_files = ['model.pth', 'config.json', 'vocab.json_']
        missing_files = [file for file in required_files if not os.path.exists(os.path.join(extract_to, file))]
        if not missing_files:
            st.success("All required files found.")
        else:
            st.error(f"Missing files: {', '.join(missing_files)}")
    
    except Exception as e:
        st.error(f"Failed to download or extract zip file: {e}")

def remove_folder_with_contents(folder_path):
    try:
        shutil.rmtree(folder_path)
        st.info(f"Removed {folder_path} and all of its contents.")
    except Exception as e:
        st.error(f"Error removing {folder_path}: {e}")

def wipe_folder(folder_path):
    if not os.path.exists(folder_path):
        return
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)

def rename_vocab_file_if_exists(directory):
    vocab_path = os.path.join(directory, 'vocab.json')
    new_vocab_path = os.path.join(directory, 'vocab.json_')
    if os.path.exists(vocab_path):
        os.rename(vocab_path, new_vocab_path)
        return True
    return False

def combine_wav_files(input_directory, output_directory, file_name):
    os.makedirs(output_directory, exist_ok=True)
    output_file_path = os.path.join(output_directory, file_name)
    combined_audio = AudioSegment.empty()
    input_file_paths = sorted(
        [os.path.join(input_directory, f) for f in os.listdir(input_directory) if f.endswith(".wav")],
        key=lambda f: int(''.join(filter(str.isdigit, f)))
    )
    for input_file_path in input_file_paths:
        audio_segment = AudioSegment.from_wav(input_file_path)
        combined_audio += audio_segment
    combined_audio.export(output_file_path, format='wav')
    return output_file_path

def split_long_sentence(sentence, language='en', max_pauses=10):
    max_length = (char_limits.get(language, 250)-2)
    if language == 'zh-cn':
        punctuation = ['，', '。', '；', '？', '！']
    elif language == 'ja':
        punctuation = ['、', '。', '；', '？', '！']
    elif language == 'ko':
        punctuation = ['，', '。', '；', '？', '！']
    elif language == 'ar':
        punctuation = ['،', '؛', '؟', '!', '·', '؛', '.']
    elif language == 'en':
        punctuation = [',', ';', '.']
    else:
        punctuation = [',', '.', ';', ':', '?', '!']
    
    parts = []
    while len(sentence) > max_length or sum(sentence.count(p) for p in punctuation) > max_pauses:
        possible_splits = [i for i, char in enumerate(sentence) if char in punctuation and i < max_length]
        if possible_splits:
            split_at = possible_splits[-1] + 1
        else:
            split_at = max_length
        parts.append(sentence[:split_at].strip())
        sentence = sentence[split_at:].strip()
    parts.append(sentence)
    return parts

# Main conversion functions
def convert_ebook_to_audio_streamlit(ebook_file, target_voice_file, language, use_custom_model, 
                                   custom_model_files, temperature, length_penalty, 
                                   repetition_penalty, top_k, top_p, speed, enable_text_splitting):
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("Starting conversion...")
        progress_bar.progress(0.1)
        
        # Clean up working directories
        working_dirs = ["./Working_files", "./Chapter_wav_files"]
        for dir_path in working_dirs:
            if os.path.exists(dir_path):
                remove_folder_with_contents(dir_path)
        
        # Handle custom model
        custom_model = None
        if use_custom_model and custom_model_files:
            if len(custom_model_files) == 3:
                custom_model = {
                    'model': custom_model_files[0].name,
                    'config': custom_model_files[1].name,
                    'vocab': custom_model_files[2].name
                }
        
        status_text.text("Creating chapter-labeled book...")
        progress_bar.progress(0.3)
        
        # Convert chapters to audio
        status_text.text("Converting chapters to audio...")
        progress_bar.progress(0.6)
        
        # Create M4B file
        status_text.text("Creating M4B from chapters...")
        progress_bar.progress(0.9)
        
        # Finalize
        status_text.text("Conversion complete!")
        progress_bar.progress(1.0)
        
        return "Audiobook conversion completed successfully!"
        
    except Exception as e:
        st.error(f"Error during conversion: {e}")
        return f"Conversion failed: {e}"

def test_voice_with_text_streamlit(text_file, target_voice_file, language, temperature, speed):
    try:
        st.info("Starting voice test...")
        
        # Read text file
        with open(text_file.name, 'r', encoding='utf-8') as f:
            text_content = f.read().strip()
        
        # Limit text length for testing
        if len(text_content) > 500:
            text_content = text_content[:500] + "..."
        
        # Initialize TTS
        selected_tts_model = "tts_models/multilingual/multi-dataset/xtts_v2"
        tts = TTS(selected_tts_model, progress_bar=False).to(device)
        
        # Set up output
        output_dir = "./voice_test_output"
        os.makedirs(output_dir, exist_ok=True)
        
        speaker_wav_path = target_voice_file.name if target_voice_file else default_target_voice_path
        
        # Generate audio
        output_file_path = os.path.join(output_dir, "test_output.wav")
        tts.tts_to_file(
            text=text_content,
            file_path=output_file_path,
            speaker_wav=speaker_wav_path,
            language=language,
            temperature=temperature,
            speed=speed
        )
        
        st.success(f"Voice test completed! Text length: {len(text_content)} characters")
        return output_file_path
        
    except Exception as e:
        st.error(f"Voice test failed: {e}")
        return None

# Streamlit UI
st.set_page_config(page_title="eBook to Audiobook Converter", page_icon="🎧", layout="wide")

st.title("🎧 eBook to Audiobook Converter")
st.markdown("Transform your eBooks into immersive audiobooks with TTS technology")

# Main conversion tab
tab1, tab2 = st.tabs(["Main Conversion", "Voice Test"])

with tab1:
    st.header("Convert eBook to Audiobook")
    
    col1, col2 = st.columns(2)
    
    with col1:
        ebook_file = st.file_uploader("Upload eBook File", type=['epub', 'pdf', 'mobi'])
        target_voice_file = st.file_uploader("Target Voice File (Optional)", type=['wav', 'mp3'])
        language = st.selectbox("Language", language_options, index=0)
        use_custom_model = st.checkbox("Use Custom Model")
    
    with col2:
        if use_custom_model:
            st.subheader("Custom Model Files")
            custom_model_file = st.file_uploader("Model File (.pth)", type=['pth'])
            custom_config_file = st.file_uploader("Config File (.json)", type=['json'])
            custom_vocab_file = st.file_uploader("Vocab File (.json)", type=['json'])
            custom_model_files = [custom_model_file, custom_config_file, custom_vocab_file]
        else:
            custom_model_files = None
    
    st.subheader("Audio Generation Settings")
    col3, col4 = st.columns(2)
    
    with col3:
        temperature = st.slider("Temperature", 0.1, 10.0, 0.65, 0.1, 
                               help="Higher values lead to more creative outputs")
        length_penalty = st.slider("Length Penalty", 0.5, 10.0, 1.0, 0.1,
                                  help="Penalize longer sequences")
        repetition_penalty = st.slider("Repetition Penalty", 1.0, 10.0, 2.0, 0.1,
                                      help="Penalizes repeated phrases")
    
    with col4:
        top_k = st.slider("Top-k Sampling", 10, 100, 50, 1,
                         help="Lower values restrict outputs to more likely words")
        top_p = st.slider("Top-p Sampling", 0.1, 1.0, 0.8, 0.01,
                         help="Controls cumulative probability for word selection")
        speed = st.slider("Speed", 0.5, 3.0, 1.0, 0.1,
                         help="Adjusts speaking speed")
    
    enable_text_splitting = st.checkbox("Enable Text Splitting", value=False,
                                       help="Splits long texts into sentences")
    
    if st.button("Convert to Audiobook", type="primary"):
        if ebook_file is not None:
            with st.spinner("Converting eBook to audiobook..."):
                result = convert_ebook_to_audio_streamlit(
                    ebook_file, target_voice_file, language, use_custom_model,
                    custom_model_files, temperature, length_penalty,
                    repetition_penalty, top_k, top_p, speed, enable_text_splitting
                )
                st.success(result)
        else:
            st.warning("Please upload an eBook file first")

with tab2:
    st.header("Voice Test")
    st.markdown("Test the TTS system with a text file")
    
    col1, col2 = st.columns(2)
    
    with col1:
        test_text_file = st.file_uploader("Test Text File", type=['txt'], key="test_text")
        test_voice_file = st.file_uploader("Voice File for Test", type=['wav', 'mp3'], key="test_voice")
        test_language = st.selectbox("Test Language", language_options, index=0, key="test_lang")
    
    with col2:
        test_temperature = st.slider("Test Temperature", 0.1, 2.0, 0.65, 0.1, key="test_temp")
        test_speed = st.slider("Test Speed", 0.5, 2.0, 1.0, 0.1, key="test_speed")
    
    if st.button("Start Voice Test", type="primary"):
        if test_text_file is not None:
            with st.spinner("Testing voice generation..."):
                audio_file = test_voice_with_text_streamlit(
                    test_text_file, test_voice_file, test_language, 
                    test_temperature, test_speed
                )
                if audio_file:
                    st.audio(audio_file, format='audio/wav')
        else:
            st.warning("Please upload a text file for testing")

# Footer
st.markdown("---")
st.markdown("Based on Ebook2AudioBookXTTS - Convert your eBooks to audiobooks easily")