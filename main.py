"""
Jarvis — Main Entry Point
===========================
Phase 1: Discover agents, print registry status
Phase 2: Terminal chat with Ollama
Phase 3: Kokoro TTS voice output
Phase 4: System prompt, personality, time-based greeting

Commands:
    Type a message to chat
    'voice'  — toggle TTS on/off
    'clear'  — reset conversation
    'status' — show system info
    'quit'   — exit
"""

import json
from datetime import datetime

from config import (
    APP_NAME, AI_NAME, GREETING_NAME, RESPONDER_MODEL, SYSTEM_PROMPT,
    CYAN, GREEN, RESET, GRAY, BOLD, YELLOW
)
from agents.base import registry, discover_agents
from core.dispatcher import dispatcher
from core.model_manager import unload_all_models


def main():
    """Application entry point."""

    # ── Banner ──
    print(f"""
{CYAN}{'='*50}
   {BOLD}{APP_NAME} — Advanced AI Assistant{RESET}{CYAN}
{'='*50}{RESET}
""")

    # ── Step 1: Discover and register all agents ──
    discover_agents(registry)

    # ── Step 2: Phase 1 debug output ──
    print(f"\n{GREEN}{'─'*50}{RESET}")
    print(f"{GREEN}Registry Status:{RESET}")
    print(f"  Agents:    {registry.agent_count}")
    print(f"  Functions: {registry.function_count}")

    print(f"\n{GREEN}Function Map:{RESET}")
    for func_def in registry.get_all_functions():
        agent = registry.get_agent_for_function(func_def.name)
        agent_name = agent.name if agent else "???"
        print(f"  {func_def.name:30s} → {agent_name}")

    print(f"\n{GREEN}Lookup Test:{RESET}")
    test_functions = ["thinking", "nonthinking", "get_system_info", "control_light"]
    for func_name in test_functions:
        agent = registry.get_agent_for_function(func_name)
        if agent:
            print(f"  '{func_name}' → {agent.name} agent ✓")
        else:
            print(f"  '{func_name}' → not registered (expected — agent not built yet)")

    schemas = registry.get_all_tool_schemas()
    print(f"\n{GREEN}Tool Schemas:{RESET} {len(schemas)} schemas generated for router model")
    if schemas:
        print(f"\n{GRAY}Example schema (first function):{RESET}")
        print(f"{GRAY}{json.dumps(schemas[0], indent=2)}{RESET}")

    # ── Step 3: Print chat instructions ──
    print(f"\n{GREEN}{'─'*50}{RESET}")
    print(f"\n{CYAN}Commands:{RESET}")
    print(f"  Type a message to chat with {AI_NAME}")
    print(f"  'voice'      — toggle TTS on/off")
    print(f"  'clear'      — reset conversation")
    print(f"  'status'     — show model status")
    print(f"  'quit'       — exit")
    print(f"\n{YELLOW}Note: Make sure Ollama is running with '{RESPONDER_MODEL}' pulled.{RESET}")
    print(f"{GRAY}{'─'*50}{RESET}\n")

    # ── Step 4: Time-based greeting ──
    # Jarvis greets you differently depending on the time of day,
    # just like the Friday repo does. If voice is enabled, he speaks it.

    from core.tts import tts_engine
    print(f"{CYAN}[TTS] Loading voice model...{RESET}")
    if tts_engine.toggle(True):
        print(f"{GREEN}[TTS] ✓ Voice is ON.{RESET}")
    else:
        print(f"{YELLOW}[TTS] ✗ Could not enable voice. Text-only mode.{RESET}")

    # Greet based on time of day (spoken aloud)
    _greet()

    # ── Step 5: Chat loop ──
    try:
        while True:
            try:
                user_input = input(f"{CYAN}You:{RESET} ").strip()
            except EOFError:
                break

            if not user_input:
                continue

            if user_input.lower() in ("quit", "exit", "q"):
                break

            if user_input.lower() == "clear":
                dispatcher.clear_history()
                continue

            if user_input.lower() == "status":
                _print_status()
                continue

            if user_input.lower() == "voice":
                _toggle_voice()
                continue

            # Send to Dispatcher — streams response + speaks via TTS
            dispatcher.process(user_input)

    except KeyboardInterrupt:
        print(f"\n{GRAY}Interrupted.{RESET}")

    # ── Cleanup ──
    print(f"\n{CYAN}Shutting down...{RESET}")
    dispatcher.shutdown()
    unload_all_models(sync=True)
    print(f"{CYAN}Goodbye!{RESET}\n")


def _greet():
    """
    Print and optionally speak a time-based greeting.
    
    Morning (5-12):   "Good morning, sir."
    Afternoon (12-17): "Good afternoon, sir."
    Evening (17-21):   "Good evening, sir."
    Night (21-5):      "You're up late, sir."
    """
    hour = datetime.now().hour

    if 5 <= hour < 12:
        greeting = f"Good morning, {GREETING_NAME}. How can I help you today?"
    elif 12 <= hour < 17:
        greeting = f"Good afternoon, {GREETING_NAME}. What can I do for you?"
    elif 17 <= hour < 21:
        greeting = f"Good evening, {GREETING_NAME}. What are we working on tonight?"
    else:
        greeting = f"You're up late, {GREETING_NAME}. What can I help you with?"

    print(f"\n{GREEN}{AI_NAME}:{RESET} {greeting}\n")

    # If TTS is enabled, speak the greeting
    from core.tts import tts_engine
    if tts_engine.enabled:
        tts_engine.queue_sentence(greeting)


def _print_status():
    """Print current model and registry status."""
    from core.model_manager import get_running_models
    from core.tts import tts_engine

    running = get_running_models()
    print(f"\n{GREEN}Status:{RESET}")
    print(f"  Personality: {AI_NAME}")
    print(f"  Agents registered: {registry.agent_count}")
    print(f"  Functions available: {registry.function_count}")
    print(f"  Ollama models loaded: {running if running else 'none'}")
    print(f"  TTS enabled: {tts_engine.enabled}")
    print(f"  TTS voice: {tts_engine.voice}")
    print()


def _toggle_voice():
    """Toggle TTS on or off."""
    from core.tts import tts_engine

    if tts_engine.enabled:
        tts_engine.toggle(False)
    else:
        print(f"{CYAN}[TTS] Loading voice model (first time may take a moment)...{RESET}")
        if tts_engine.toggle(True):
            print(f"{GREEN}[TTS] ✓ Voice is ON. {AI_NAME} will speak responses.{RESET}")
        else:
            print(f"{YELLOW}[TTS] ✗ Could not enable voice.{RESET}")


if __name__ == "__main__":
    main()