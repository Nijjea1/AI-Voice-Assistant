"""
Jarvis — Main Entry Point
===========================
Phase 1: Skeleton test.

This file will grow as we add phases:
  Phase 1: Discover agents, print registry status (YOU ARE HERE)
  Phase 2: Add terminal chat with Ollama
  Phase 5: Launch the PySide6 GUI
  Phase 6: Start the voice assistant

For now, it just proves the plugin architecture works:
  1. Discovers all agent files in agents/
  2. Registers them in the AgentRegistry
  3. Prints what was found
"""

from config import APP_NAME, CYAN, GREEN, RESET, GRAY, BOLD
from agents.base import registry, discover_agents


def main():
    """Application entry point."""
    
    # ── Banner ──
    print(f"""
{CYAN}{'='*50}
   {BOLD}{APP_NAME} — Advanced AI Assistant{RESET}{CYAN}
{'='*50}{RESET}
""")
    
    # ── Step 1: Discover and register all agents ──
    # This scans agents/ for any Python files containing BaseAgent subclasses
    # and registers them. New files are found automatically.
    discover_agents(registry)
    
    # ── Step 2: Print what we found ──
    print(f"\n{GREEN}{'─'*50}{RESET}")
    print(f"{GREEN}Registry Status:{RESET}")
    print(f"  Agents:    {registry.agent_count}")
    print(f"  Functions: {registry.function_count}")
    
    # Show all registered functions and which agent owns them
    print(f"\n{GREEN}Function Map:{RESET}")
    for func_def in registry.get_all_functions():
        agent = registry.get_agent_for_function(func_def.name)
        agent_name = agent.name if agent else "???"
        print(f"  {func_def.name:30s} → {agent_name}")
    
    # ── Step 3: Test a lookup ──
    print(f"\n{GREEN}Lookup Test:{RESET}")
    test_functions = ["thinking", "nonthinking", "get_system_info", "control_light"]
    for func_name in test_functions:
        agent = registry.get_agent_for_function(func_name)
        if agent:
            print(f"  '{func_name}' → {agent.name} agent ✓")
        else:
            print(f"  '{func_name}' → not registered (expected — agent not built yet)")
    
    # ── Step 4: Test tool schema generation ──
    schemas = registry.get_all_tool_schemas()
    print(f"\n{GREEN}Tool Schemas:{RESET} {len(schemas)} schemas generated for router model")
    
    # Show one schema as an example
    if schemas:
        import json
        print(f"\n{GRAY}Example schema (first function):{RESET}")
        print(f"{GRAY}{json.dumps(schemas[0], indent=2)}{RESET}")
    
    print(f"\n{CYAN}{'='*50}")
    print(f"   {APP_NAME} Phase 1 — Skeleton OK")
    print(f"{'='*50}{RESET}\n")


if __name__ == "__main__":
    main()
