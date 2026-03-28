# Jarvis — Local AI Voice Assistant

A **fully local, privacy-first AI assistant** for Windows. Voice-controlled with a polished desktop GUI, running entirely on your machine.

> 🔒 No cloud APIs. No subscriptions. No data leaves your computer.

## Features (Planned)

| Feature | Description |
|---------|-------------|
| 🎤 Voice Control | Wake word "Jarvis" with natural language commands |
| 💬 AI Chat | Local LLMs via Ollama with streaming responses |
| 🏠 Smart Home | TP-Link Kasa smart light/plug control |
| 📅 Planner | Calendar events, timers, alarms, to-do lists |
| 📰 Briefing | AI-curated news from Technology, Science, Top Stories |
| 🌤️ Weather | Current weather and hourly forecast |
| 🔍 Web Search | DuckDuckGo search via voice or text |
| 🌐 Web Agent | VLM-powered autonomous browser automation |

## Architecture

Every feature is a **plugin agent** — a self-contained Python file in `agents/` that auto-registers on startup.

```
User Input → STT → Router (FunctionGemma) → Agent → LLM → Kokoro TTS → Audio
```

## Quick Start

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/jarvis.git
cd jarvis

# Create environment
conda create -n jarvis python=3.11 -y
conda activate jarvis

# Install dependencies
pip install -r requirements.txt

# Run
python main.py
```

## Project Structure

```
jarvis/
├── main.py              # Entry point
├── config.py            # All configuration
├── agents/              # Plugin agents (auto-discovered)
│   ├── base.py          # BaseAgent, AgentRegistry
│   └── passthrough_agent.py
├── core/                # Engine (dispatcher, TTS, STT, router)
├── gui/                 # PySide6 GUI
└── data/                # SQLite databases (gitignored)
```

## Tech Stack

- **LLM:** Ollama → Qwen 3 1.7B
- **Router:** FunctionGemma 270M (fine-tuned)
- **TTS:** Kokoro-82M
- **STT:** RealTimeSTT + Whisper + Porcupine
- **GUI:** PySide6 + QFluentWidgets
- **Smart Home:** python-kasa

## Build Progress

- [x] Phase 1: Skeleton + agent plugin architecture
- [ ] Phase 2: Central Dispatcher + Ollama LLM
- [ ] Phase 3: Kokoro TTS
- [ ] Phase 4: Router + all agents
- [ ] Phase 5: GUI shell
- [ ] Phase 6: Voice assistant
- [ ] Phase 7: Browser agent
- [ ] Phase 8: Briefing + home automation tabs

## License

MIT
