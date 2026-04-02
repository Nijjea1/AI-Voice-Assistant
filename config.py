"""
Jarvis — Centralized Configuration
====================================
Every tunable value lives here. No magic numbers scattered across files.

WHY THIS PATTERN?
  - Change the Ollama model? One line here, works everywhere.
  - New developer joins? They read this file to understand all the knobs.
  - Debugging? You know exactly what values the system is using.

HOW IT WORKS:
  Other files import what they need:
      from config import RESPONDER_MODEL, OLLAMA_URL

  They never hardcode values like "http://localhost:11434" directly.
"""

# ─────────────────────────────────────────────
# LLM (Ollama)
# ─────────────────────────────────────────────
# Ollama runs as a local server. We talk to it over HTTP.
# Pull a model first:  ollama pull qwen3:1.7b
OLLAMA_URL = "http://localhost:11434/api"
RESPONDER_MODEL = "qwen3:1.7b"       # The model that generates responses
MAX_HISTORY = 20                       # Max conversation turns to keep in context
QWEN_TIMEOUT_SECONDS = 300            # Unload model after 5 min idle (saves VRAM)
QWEN_KEEP_ALIVE = "5m"               # Tell Ollama to keep model warm for 5 min

# ─────────────────────────────────────────────
# Router (FunctionGemma)
# ─────────────────────────────────────────────
# A tiny fine-tuned model (~270M params) that classifies user intent.
# "Turn on the lights" → control_light
# "What's 2+2?"        → thinking
# Auto-downloads from HuggingFace on first run.
LOCAL_ROUTER_PATH = "./merged_model"
HF_ROUTER_REPO = "nlouis/pocket-ai-router"

# ─────────────────────────────────────────────
# TTS — Kokoro (Text-to-Speech)
# ─────────────────────────────────────────────
# Kokoro-82M: lightweight, natural-sounding, runs locally.
# Voices: af_heart (American female), bf_emma (British female),
#          am_adam (American male), bm_george (British male), etc.
# Full list: https://huggingface.co/hexgrad/Kokoro-82M
KOKORO_VOICE = "bm_george"              # British male — clear and natural like Jarvis
KOKORO_LANG = "b"                     # 'a' = American, 'b' = British
KOKORO_SAMPLE_RATE = 24000            # Kokoro outputs audio at 24kHz
KOKORO_SPEED = 1.1                     # 0.5-2.0, default 1.0. Talking speed.

# ─────────────────────────────────────────────
# STT — Speech-to-Text
# ─────────────────────────────────────────────
# RealTimeSTT handles microphone capture + Whisper transcription.
# Porcupine handles wake word detection ("Jarvis" is a built-in keyword).
WAKE_WORD = "jarvis"
WAKE_WORD_SENSITIVITY = 0.4           # 0.0-1.0, lower = fewer false positives
REALTIMESTT_MODEL = "base"            # Whisper model size: tiny/base/small/medium/large
STT_RECORD_TIMEOUT = 5.0              # Max seconds to listen after wake word

# ─────────────────────────────────────────────
# Voice Assistant
# ─────────────────────────────────────────────
VOICE_ASSISTANT_ENABLED = True        # Set False to disable voice entirely

# ─────────────────────────────────────────────
# GUI
# ─────────────────────────────────────────────
APP_NAME = "Jarvis"
APP_MIN_WIDTH = 1100
APP_MIN_HEIGHT = 750

# ─────────────────────────────────────────────
# Smart Home (TP-Link Kasa)
# ─────────────────────────────────────────────
KASA_DISCOVERY_TIMEOUT = 5            # Seconds to scan for devices on LAN

# ─────────────────────────────────────────────
# Weather (Open-Meteo — free, no API key)
# ─────────────────────────────────────────────
DEFAULT_LATITUDE = 43.7315            # Brampton, ON
DEFAULT_LONGITUDE = -79.7624

# ─────────────────────────────────────────────
# Console Colors (ANSI escape codes)
# ─────────────────────────────────────────────
# These make terminal output readable during development.
# Usage: print(f"{CYAN}[Module] message{RESET}")
GRAY = "\033[90m"
RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
