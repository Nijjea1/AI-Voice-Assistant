# Jarvis — Product Requirements Document

**Author:** Avneet  
**Date:** March 28, 2026  
**Status:** Planning  
**Version:** 1.0

---

## 1. Vision

Jarvis is a **fully local, privacy-first AI assistant** for Windows that combines a polished desktop GUI with natural voice control. It runs entirely on your machine — no cloud APIs, no subscriptions, no data leaving your computer. Think of it as a personal Iron Man Jarvis that actually works.

The architecture is a hybrid of two designs:
- **ada_local** — a working repo with a proven GUI, fine-tuned router, VRAM management, and browser agent
- **Jarvis spec** — a multi-agent architecture with a Central Dispatcher, plugin agents, and planned integrations (Spotify, Gmail, Google Drive, Face ID)

We take the best engineering from both and build something better than either.

---

## 2. Core Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    USER INPUT                            │
│              (Microphone / Text Box)                     │
└────────────────────┬─────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────┐
│                  STT (RealTimeSTT)                       │
│         Wake word "Jarvis" via Porcupine                 │
│         Whisper transcription on GPU                     │
└────────────────────┬─────────────────────────────────────┘
                     │ text
                     ▼
┌──────────────────────────────────────────────────────────┐
│             CENTRAL DISPATCHER                           │
│                                                          │
│   ┌─────────────┐    ┌──────────────────────────┐       │
│   │   Router     │───▶│    Agent Registry         │      │
│   │ (FunctionGemma)│  │  (auto-discovers agents)  │      │
│   └─────────────┘    └──────────┬───────────────┘       │
│                                 │                        │
│         ┌──────────┬────────────┼────────────┬────────┐  │
│         ▼          ▼            ▼            ▼        ▼  │
│   ┌─────────┐ ┌────────┐ ┌──────────┐ ┌────────┐ ┌───┐ │
│   │ Lights  │ │ Timer  │ │ Calendar │ │ Search │ │...│ │
│   │ Agent   │ │ Agent  │ │ Agent    │ │ Agent  │ │   │ │
│   └─────────┘ └────────┘ └──────────┘ └────────┘ └───┘ │
│         │          │            │            │           │
│         └──────────┴────────────┴────────────┘           │
│                        │ result                          │
│                        ▼                                 │
│              ┌────────────────┐                          │
│              │  Ollama LLM    │  (Qwen 3 1.7B)           │
│              │  Natural lang  │                          │
│              │  response gen  │                          │
│              └───────┬────────┘                          │
│                      │ text                              │
│                      ▼                                   │
│              ┌────────────────┐                          │
│              │  Kokoro TTS    │  (82M params)            │
│              │  Voice output  │                          │
│              └────────────────┘                          │
└──────────────────────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────┐
│                   PySide6 GUI                            │
│          QFluentWidgets (Windows 11 design)              │
│                                                          │
│   ┌───────────┬────────┬─────────┬──────────┬─────────┐ │
│   │ Dashboard │  Chat  │ Planner │ Briefing │ Settings│ │
│   └───────────┴────────┴─────────┴──────────┴─────────┘ │
└──────────────────────────────────────────────────────────┘
```

---

## 3. Technology Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| **Language** | Python 3.11 | Ecosystem, ML libraries |
| **GUI Framework** | PySide6 + QFluentWidgets | Windows 11 Fluent Design, lazy-loading tabs |
| **LLM** | Ollama → Qwen 3 1.7B | Fast local inference, thinking mode |
| **Router** | FunctionGemma 270M (fine-tuned) | 50ms intent classification on GPU |
| **TTS** | Kokoro-82M (`pip install kokoro`) | 82M params, Apache license, natural voice |
| **STT** | RealTimeSTT + Porcupine + Whisper | Real-time transcription, built-in "Jarvis" wake word |
| **Smart Home** | python-kasa | TP-Link Kasa devices, async |
| **Web Search** | duckduckgo-search | No API key needed |
| **Browser Agent** | Playwright + Qwen3-VL:4B | VLM-driven autonomous browsing |
| **Weather** | Open-Meteo API | Free, no key needed |
| **Storage** | SQLite | Chat history, calendar, tasks |
| **Settings** | JSON file (~/.jarvis/) | Thread-safe, Qt signal reactivity |

---

## 4. Agent Plugin System

Every feature is an **Agent**. Agents are self-contained Python files in `agents/`. They auto-register on startup — no core code changes needed to add a feature.

### 4.1 BaseAgent Interface

```python
class MyAgent(BaseAgent):
    name = "my_agent"
    description = "Does something cool"

    def get_functions(self) -> List[FunctionDef]:
        # Declare what functions this agent handles
        return [FunctionDef(name="do_thing", ...)]

    def execute(self, func_name, params) -> AgentResult:
        # Handle the function call
        return AgentResult(success=True, message="Done!")

    def get_system_info(self) -> Optional[Dict]:
        # Contribute to dashboard / system status
        return {"status": "running"}
```

### 4.2 Agent Roster

**Phase 1 — MVP (ada_local parity):**

| Agent | Functions | Status |
|-------|-----------|--------|
| Passthrough | `thinking`, `nonthinking`, `get_system_info` | Core |
| Light | `control_light` | Smart home |
| Timer | `set_timer`, `set_alarm` | Planner |
| Calendar | `create_calendar_event`, `read_calendar` | Planner |
| Task | `add_task`, `complete_task` | Planner |
| Search | `web_search` | Web |
| News | `get_briefing` | Dashboard |
| Weather | `get_weather` | Dashboard |
| Browser | `browse_web` | Web Agent |

**Phase 2 — Jarvis Extras (post-MVP):**

| Agent | Functions | Status |
|-------|-----------|--------|
| Spotify | `play_music`, `pause`, `skip`, `now_playing` | Entertainment |
| Email (Gmail) | `read_email`, `send_email`, `search_email` | Communication |
| Google Drive | `search_files`, `upload_file` | Productivity |
| Google Calendar | `sync_events` (replaces local calendar) | Cloud sync |
| Face ID | `authenticate_user` | Security |
| System Control | `open_app`, `screenshot`, `system_stats` | OS integration |

---

## 5. Feature Specifications

### 5.1 Voice Pipeline

- **Wake word:** "Jarvis" via Porcupine (pre-trained, no API key for built-in words)
- **STT:** Whisper base model via RealTimeSTT, GPU-accelerated
- **TTS:** Kokoro-82M, British English voice `bf_emma`, 24kHz sample rate
- **Streaming:** Sentence-buffered — TTS starts speaking as soon as the first sentence completes from the LLM, not after the full response

### 5.2 Router

- Fine-tuned FunctionGemma 270M from `nlouis/pocket-ai-router` on HuggingFace
- Classifies user intent into one of N registered functions in ~50ms on GPU
- Falls back to `nonthinking` if no clear match
- Tool schemas are dynamically built from the Agent Registry (new agents automatically appear in routing)

### 5.3 VRAM Management

- **Auto-load:** Qwen model loads on first query
- **Auto-unload:** After 5 minutes idle, Qwen unloads to free VRAM
- **Exclusive mode:** When the Browser Agent (VLM) needs VRAM, it unloads Qwen first
- **Parallel startup:** Router, Responder, and TTS models load in parallel threads

### 5.4 GUI Tabs

| Tab | Contents |
|-----|----------|
| **Dashboard** | Greeting, clock, weather widget, upcoming events, tasks, device status, news cards |
| **Chat** | Full chat interface with streaming, thinking expander, search indicator, session sidebar |
| **Planner** | Calendar view, task list, timers, alarms |
| **Briefing** | AI-curated news from DuckDuckGo (Tech, Science, Top Stories) |
| **Home Auto** | Kasa device discovery, on/off/brightness/color controls |
| **Web Agent** | VLM browser automation with live screenshot, action log, thinking stream |
| **Settings** | Model selection, Ollama URL, TTS voice, weather location, theme |

### 5.5 Smart Home

- TP-Link Kasa: bulbs, plugs, light strips
- Auto-discovery on local network
- Fuzzy device name matching ("living room" matches "Living Room Lamp")
- HSV color control with named colors (red, blue, warm white, etc.)

---

## 6. Build Plan — Git Commit Milestones

The project is built in **8 phases**. Each phase is a working, runnable checkpoint you can push to GitHub. You learn each layer before building the next.

---

### Phase 1: Skeleton + Config *(Commit 1)*
**What you learn:** Project structure, Python packaging, configuration patterns

```
jarvis/
├── main.py              # Entry point (just prints "Jarvis ready")
├── config.py            # All configuration constants
├── requirements.txt     # Dependencies
├── README.md            # Project overview
├── .gitignore
├── agents/
│   ├── __init__.py
│   └── base.py          # BaseAgent, AgentRegistry, FunctionDef, AgentResult
├── core/
│   └── __init__.py
├── gui/
│   └── __init__.py
└── data/
```

**Test:** `python main.py` prints startup message. Import `agents.base` works.  
**Commit message:** `feat: project skeleton with agent plugin architecture`

---

### Phase 2: Central Dispatcher + Ollama LLM *(Commits 2-3)*
**What you learn:** HTTP streaming, Ollama API, the dispatcher pattern

Files added:
- `core/dispatcher.py` — Central Dispatcher (routes queries, streams LLM)
- `core/model_manager.py` — VRAM lifecycle (load/unload/timeout)
- `core/settings_store.py` — Persistent settings
- `agents/passthrough_agent.py` — thinking/nonthinking agent

**Test:** Run from terminal, type a message, get a streamed response from Ollama.  
**Commit 2:** `feat: central dispatcher with Ollama LLM streaming`  
**Commit 3:** `feat: VRAM model manager with auto-unload timeout`

---

### Phase 3: Kokoro TTS *(Commit 4)*
**What you learn:** Audio synthesis, threading, queue patterns

Files added:
- `core/tts.py` — Kokoro TTS engine with sentence queue + interrupt

**Test:** Type a message, hear Jarvis speak the response.  
**Commit:** `feat: Kokoro TTS with sentence-buffered streaming`

---

### Phase 4: FunctionGemma Router + First Agents *(Commits 5-7)*
**What you learn:** Fine-tuned model inference, agent execution, the full pipeline

Files added:
- `core/router.py` — FunctionGemma router (auto-downloads from HuggingFace)
- `agents/timer_agent.py` — Timers and alarms
- `agents/task_agent.py` — To-do list (SQLite)
- `agents/calendar_agent.py` — Local calendar (SQLite)
- `agents/search_agent.py` — DuckDuckGo web search
- `agents/weather_agent.py` — Open-Meteo weather
- `agents/light_agent.py` — Kasa smart home
- `agents/news_agent.py` — AI-curated news briefing

**Test:** Type "set a timer for 5 minutes" → routed to timer agent. Type "turn on the lights" → routed to light agent.  
**Commit 5:** `feat: FunctionGemma router with auto-download`  
**Commit 6:** `feat: timer, task, calendar, search agents`  
**Commit 7:** `feat: weather, news, light agents`

---

### Phase 5: GUI Shell *(Commits 8-10)*
**What you learn:** PySide6, QFluentWidgets, lazy tab loading, Qt signals

Files added:
- `gui/app.py` — Main window with Fluent Design
- `gui/styles.py` — Aura dark theme stylesheet
- `gui/tabs/dashboard.py` — Dashboard with weather, clock, greeting
- `gui/tabs/chat.py` — Chat interface with streaming bubbles
- `gui/tabs/planner.py` — Calendar + tasks + timers
- `gui/tabs/settings.py` — Configuration panel
- `gui/components/message_bubble.py` — Chat message widget
- `gui/components/thinking_expander.py` — Collapsible thinking display
- `gui/components/system_monitor.py` — CPU/RAM in title bar

**Test:** Full windowed app launches, chat works, dashboard shows weather.  
**Commit 8:** `feat: GUI shell with dashboard and chat tabs`  
**Commit 9:** `feat: planner tab with calendar, tasks, timers`  
**Commit 10:** `feat: settings tab and system monitor`

---

### Phase 6: Voice Assistant *(Commit 11)*
**What you learn:** Real-time audio capture, wake word detection, STT pipeline

Files added:
- `core/stt.py` — RealTimeSTT with Porcupine wake word
- `core/voice_assistant.py` — Voice pipeline orchestrator
- `gui/components/voice_indicator.py` — Listening indicator

**Test:** Say "Jarvis, what time is it?" → hear a spoken response.  
**Commit:** `feat: voice assistant with wake word detection`

---

### Phase 7: Browser Agent *(Commit 12)*
**What you learn:** Playwright automation, VLM inference, agentic loops

Files added:
- `agents/browser_agent.py` — VLM browser automation
- `core/browser_controller.py` — Playwright wrapper
- `core/vlm_client.py` — Qwen3-VL Ollama client
- `gui/tabs/browser.py` — Browser tab with live viewport

**Test:** Type "Go to google.com and search for Python tutorials" → watch it browse autonomously.  
**Commit:** `feat: VLM browser agent with live viewport`

---

### Phase 8: Briefing + Home Automation Tabs *(Commit 13)*
**What you learn:** Async device discovery, AI content curation

Files added:
- `gui/tabs/briefing.py` — News briefing view
- `gui/tabs/home_automation.py` — Device control panel
- `gui/components/news_card.py` — News card widget

**Test:** Briefing tab shows AI-curated headlines. Home tab discovers Kasa devices.  
**Commit:** `feat: briefing and home automation tabs`

---

### Phase 9+: Jarvis Extras (Future)

These come after MVP is solid:

| Feature | Commit Message |
|---------|---------------|
| Spotify Agent | `feat: spotify agent with playback control` |
| Gmail Agent | `feat: gmail agent with OAuth2` |
| Google Drive Agent | `feat: google drive agent` |
| Google Calendar Sync | `feat: google calendar API replacing local calendar` |
| Face ID | `feat: face recognition authentication` |
| Encrypted Vault | `feat: secure credential storage` |

---

## 7. File Structure (Complete)

```
jarvis/
├── main.py
├── config.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── agents/                     # Plugin agents (auto-discovered)
│   ├── __init__.py
│   ├── base.py                 # BaseAgent, AgentRegistry, FunctionDef
│   ├── passthrough_agent.py    # thinking/nonthinking/get_system_info
│   ├── light_agent.py          # Kasa smart home
│   ├── timer_agent.py          # Timers + alarms
│   ├── calendar_agent.py       # Local SQLite calendar
│   ├── task_agent.py           # To-do list
│   ├── search_agent.py         # DuckDuckGo web search
│   ├── weather_agent.py        # Open-Meteo weather
│   ├── news_agent.py           # AI-curated news
│   └── browser_agent.py        # VLM browser automation
│
├── core/                       # Core engine (not agents)
│   ├── __init__.py
│   ├── dispatcher.py           # Central Dispatcher (the brain)
│   ├── router.py               # FunctionGemma intent classifier
│   ├── tts.py                  # Kokoro TTS engine
│   ├── stt.py                  # RealTimeSTT + wake word
│   ├── voice_assistant.py      # Voice pipeline orchestrator
│   ├── model_manager.py        # VRAM lifecycle management
│   ├── history.py              # SQLite chat history
│   ├── settings_store.py       # Persistent JSON settings
│   ├── browser_controller.py   # Playwright wrapper
│   └── vlm_client.py           # Qwen3-VL client
│
├── gui/                        # PySide6 GUI
│   ├── __init__.py
│   ├── app.py                  # Main window
│   ├── handlers.py             # Chat message handling
│   ├── styles.py               # Aura dark theme
│   ├── tabs/
│   │   ├── __init__.py
│   │   ├── dashboard.py
│   │   ├── chat.py
│   │   ├── planner.py
│   │   ├── briefing.py
│   │   ├── home_automation.py
│   │   ├── browser.py
│   │   └── settings.py
│   ├── components/
│   │   ├── __init__.py
│   │   ├── message_bubble.py
│   │   ├── thinking_expander.py
│   │   ├── search_indicator.py
│   │   ├── system_monitor.py
│   │   ├── voice_indicator.py
│   │   ├── timer.py
│   │   ├── alarm.py
│   │   ├── schedule.py
│   │   ├── news_card.py
│   │   ├── toast.py
│   │   └── toggle_switch.py
│   └── assets/
│       └── logo.png
│
├── data/                       # SQLite databases (gitignored)
│   ├── chat_history.db
│   ├── calendar.db
│   └── tasks.db
│
├── merged_model/               # Fine-tuned router (auto-downloaded)
│
└── training/                   # Router training pipeline
    ├── generate_training_data.py
    └── train_function_gemma.py
```

---

## 8. Dependencies

```
# Core GUI
PySide6>=6.10.0
PySide6-Fluent-Widgets>=1.10.0
darkdetect>=0.8.0
markdown>=3.4.0
pygments>=2.15.0

# AI / ML (install PyTorch with CUDA separately)
transformers>=4.57.0
accelerate>=1.12.0
safetensors>=0.7.0
huggingface-hub>=0.36.0

# TTS (Kokoro)
kokoro>=0.9.4
soundfile>=0.13.0
sounddevice>=0.5.0
numpy>=2.0.0
misaki[en]>=0.3.0

# STT
realtimestt>=0.3.0
PyAudio>=0.2.14

# Browser Agent
playwright>=1.57.0

# Smart Home
python-kasa>=0.10.0

# Web / API
requests>=2.32.0
duckduckgo-search>=8.0.0

# System
psutil>=7.0.0
pynvml>=13.0.0
```

**System prerequisites:**
- Miniconda (Python 3.11)
- Ollama (with `qwen3:1.7b` pulled)
- espeak-ng (for Kokoro TTS phoneme processing)
- NVIDIA GPU with 6GB+ VRAM (recommended)
- PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu124`

---

## 9. Hardware Requirements

| Tier | RAM | GPU | Experience |
|------|-----|-----|-----------|
| Minimum | 8GB | None (CPU) | Slower, but functional |
| Recommended | 16GB | NVIDIA 6GB VRAM | Smooth, real-time voice |
| Optimal | 32GB | NVIDIA 8GB+ VRAM | Browser agent + chat simultaneously |

---

## 10. What Makes This Different from ada_local

| Aspect | ada_local | Jarvis |
|--------|----------|--------|
| Architecture | Flat function executor | Plugin agent system with auto-discovery |
| TTS | Piper (robotic) | Kokoro-82M (natural) |
| Dispatcher | Split across 3 files | Single Central Dispatcher class |
| Adding features | Edit router + executor + config | Drop a file in agents/ |
| Security | None | Planned: Face ID + encrypted vault |
| Integrations | None | Planned: Spotify, Gmail, Google Drive |
| Calendar | Local SQLite only | Local first, Google Calendar API planned |
| Code quality | Working prototype | Structured, documented, testable |
