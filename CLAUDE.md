# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**ebook2audiobook** is a Python application that converts eBooks to audiobooks with chapters and metadata using Calibre and Coqui XTTS (Text-to-Speech). The project supports voice cloning, multiple languages, and offers both a web GUI interface and headless command-line operation.

## Development Commands

### Installation Requirements
```bash
# System dependencies (example for Ubuntu)
sudo apt-get install -y calibre ffmpeg mecab libmecab-dev mecab-ipadic-utf8

# Python dependencies
pip install coqui-tts==0.24.2 pydub nltk beautifulsoup4 ebooklib tqdm gradio==4.44.0
python -m nltk.downloader punkt punkt_tab

# For non-Latin languages
pip install mecab mecab-python3 unidic
python -m unidic download
```

### Running the Application
```bash
# Web GUI mode (default)
python app.py

# Web GUI with public sharing
python app.py --share True

# Headless mode
python app.py --headless True --ebook <path_to_ebook> --voice [voice_file] --language [lang_code]

# Help and parameter listing
python app.py -h
```

### Docker Commands
```bash
# CPU only
docker run -it --rm -p 7860:7860 --platform=linux/amd64 athomasson2/ebook2audiobookxtts:huggingface python app.py

# GPU accelerated
docker run -it --rm --gpus all -p 7860:7860 --platform=linux/amd64 athomasson2/ebook2audiobookxtts:huggingface python app.py
```

## Code Architecture

### Primary Entry Point
- **`app.py`** (1,041 lines) - Main application file containing all core functionality

### Key Functions in app.py
- `convert_ebook_to_audio()` - Main conversion pipeline that orchestrates the entire process
- `create_chapter_labeled_book()` - Extracts and processes chapters from eBooks using Calibre
- `convert_chapters_to_audio_standard_model()` - Handles TTS conversion using standard XTTS models
- `convert_chapters_to_audio_custom_model()` - Handles TTS conversion using custom fine-tuned models
- `create_m4b_from_chapters()` - Assembles final M4B audiobook with metadata and chapter markers
- `run_gradio_interface()` - Launches the web GUI interface

### Core Dependencies
- **Coqui TTS (XTTS v2)** - Advanced text-to-speech with voice cloning
- **Gradio 4.44.0** - Web GUI framework (exact version required)
- **Calibre** - eBook format conversion and chapter extraction
- **FFmpeg** - Audio processing and M4B file creation
- **PyTorch** - Deep learning framework for TTS models

### Supported Formats
- **Input**: `.epub`, `.pdf`, `.mobi`, `.txt`, `.html`, `.rtf`, `.chm`, `.lit`, `.pdb`, `.fb2`, `.odt`, `.cbr`, `.cbz`, `.prc`, `.lrf`, `.pml`, `.snb`, `.cbc`, `.rb`, `.tcr`
- **Output**: M4B format with chapter markers and metadata
- **Note**: `.epub` and `.mobi` provide optimal chapter detection

### Language Support
16 languages supported: English, Spanish, French, German, Italian, Portuguese, Polish, Turkish, Russian, Dutch, Czech, Arabic, Chinese (Simplified), Japanese, Hungarian, Korean

### Voice Options
- Default built-in voice (`default_voice.wav`)
- Voice cloning from uploaded audio samples
- Custom fine-tuned XTTS models from Hugging Face
- Downloadable custom models via URL

## Important Parameters
- `--temperature` (0.65 default) - Controls TTS creativity/hallucinations
- `--speed` (1.0 default) - Narrator speaking speed
- `--language` - Target language for TTS
- `--use_custom_model` - Enable custom XTTS models
- `--custom_model_url` - Download custom models from URLs

## Directory Structure Notes
- `legacy/` - Contains deprecated scripts (do not modify)
- `Notebooks/` - Jupyter notebooks for Colab/Kaggle deployment
- `samples/` - Example outputs and test files for supported languages
- `readme/` - Internationalized documentation