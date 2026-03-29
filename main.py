"""
Jarvis — Main Entry Point
===========================
Phase 1: Discover agents, print registry status
Phase 2: Terminal chat with Ollama

Run:
    python main.py

Type messages and get streamed AI responses.
Type 'quit' or 'exit' to stop.
Type 'clear' to reset the conversation.
"""

import json

from config import APP_NAME, RESPONDER_MODEL, CYAN, GREEN, RESET, GRAY, BOLD, YELLOW
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
    print(f"  Type a message to chat with Jarvis")
    print(f"  'clear'      — reset conversation")
    print(f"  'status'     — show model status")
    print(f"  'quit'       — exit")
    print(f"\n{YELLOW}Note: Make sure Ollama is running with '{RESPONDER_MODEL}' pulled.{RESET}")
    print(f"{GRAY}{'─'*50}{RESET}\n")

    #Step 4: Chat loop 
    #Added in phase 2 for chatting with the ollama model and testing.
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

            # Send to Dispatcher — streams response from Ollama
            dispatcher.process(user_input)

    except KeyboardInterrupt:
        print(f"\n{GRAY}Interrupted.{RESET}")

    # ── Cleanup ──
    print(f"\n{CYAN}Shutting down...{RESET}")
    dispatcher.shutdown()
    unload_all_models(sync=True)
    print(f"{CYAN}Goodbye!{RESET}\n")


def _print_status():
    """Print current model and registry status."""
    from core.model_manager import get_running_models

    running = get_running_models()
    print(f"\n{GREEN}Status:{RESET}")
    print(f"  Agents registered: {registry.agent_count}")
    print(f"  Functions available: {registry.function_count}")
    print(f"  Ollama models loaded: {running if running else 'none'}")
    print()


if __name__ == "__main__":
    main()