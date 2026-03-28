"""
Agent Base Class & Registry
============================

THE BIG IDEA:
  Every capability in Jarvis is an "Agent." Lights? That's an agent. Timers?
  Agent. Spotify? Agent. Each agent is a self-contained Python file that
  declares what it can do and how to do it.

  When Jarvis starts, it automatically discovers every agent file in this
  folder and registers it. The Central Dispatcher then knows "if the user
  says 'turn on the lights', send it to the LightAgent."

  Adding a new feature = creating one new file. No other code changes needed.

THIS FILE DEFINES:
  1. FunctionDef    — Describes a single capability ("set_timer", "control_light")
  2. AgentResult    — Standardized response from any agent action
  3. BaseAgent      — Abstract class every agent must subclass
  4. AgentRegistry  — The lookup table that maps functions → agents
  5. discover_agents() — Auto-finds and registers all agent files

DESIGN PATTERNS USED:
  - Abstract Base Class (ABC): Forces every agent to implement required methods
  - Registry Pattern: Central lookup table for routing
  - Plugin Architecture: New files auto-register without touching existing code
  - Dataclasses: Clean data containers without boilerplate
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import importlib
import pkgutil
import os

from config import GRAY, RESET, CYAN, GREEN


# ═══════════════════════════════════════════════
# 1. FunctionDef — What an agent can do
# ═══════════════════════════════════════════════

@dataclass
class FunctionDef:
    """
    Describes a single function an agent can handle.
    
    Think of it as a menu item: "Here's what I can do, and here's
    what information I need from you to do it."
    
    Example:
        FunctionDef(
            name="set_timer",
            description="Set a countdown timer",
            parameters={
                "duration": {
                    "type": "string",
                    "description": "e.g. '5 minutes' or '1 hour'"
                },
                "label": {
                    "type": "string",
                    "description": "Optional name for the timer"
                }
            },
            required_params=["duration"]
        )
    
    WHAT IS @dataclass?
        A Python decorator that auto-generates __init__, __repr__, and __eq__
        methods from the class attributes. Without it, you'd write:
        
            def __init__(self, name, description, parameters=None, required_params=None):
                self.name = name
                self.description = description
                self.parameters = parameters if parameters is not None else {}
                self.required_params = required_params if required_params is not None else []
        
        With @dataclass, Python generates all of that from the type annotations.
    """
    name: str                                          # Function name (e.g. "set_timer")
    description: str                                   # What it does (shown to the router model)
    parameters: Dict[str, Any] = field(default_factory=dict)      # Parameter definitions
    required_params: List[str] = field(default_factory=list)       # Which params are mandatory
    
    # WHY field(default_factory=dict)?
    #   In Python, mutable defaults are shared across all instances:
    #       parameters: Dict = {}     # ❌ DANGEROUS — all instances share same dict
    #       parameters: Dict = field(default_factory=dict)  # ✓ Each instance gets its own dict
    
    def to_tool_schema(self) -> Dict:
        """
        Convert to JSON Schema format for the FunctionGemma router model.
        
        The router model was trained to understand functions in this exact
        JSON format. It reads the schema and decides which function matches
        the user's intent.
        
        Returns something like:
            {
                "type": "function",
                "function": {
                    "name": "set_timer",
                    "description": "Set a countdown timer",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "duration": {"type": "string", "description": "..."}
                        },
                        "required": ["duration"]
                    }
                }
            }
        """
        properties = {}
        for param_name, param_info in self.parameters.items():
            properties[param_name] = {
                "type": param_info.get("type", "string"),
                "description": param_info.get("description", ""),
            }
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": self.required_params,
                },
            },
        }


# ═══════════════════════════════════════════════
# 2. AgentResult — What comes back from an action
# ═══════════════════════════════════════════════

@dataclass
class AgentResult:
    """
    Standardized result from any agent action.
    
    Every agent returns this same structure, so the Dispatcher doesn't
    need to know anything about the agent's internals. It just checks:
      - Did it succeed?
      - What's the human-readable message?
      - Is there raw data?
      - Should the GUI update something?
    
    Example:
        AgentResult(
            success=True,
            message="Timer set for 5 minutes",
            data={"seconds": 300, "label": "Pasta"},
            gui_signal="timer_set",
            gui_data={"seconds": 300, "label": "Pasta"}
        )
    """
    success: bool                          # Did the action work?
    message: str                           # Human-readable result (shown to user/LLM)
    data: Any = None                       # Raw data (for programmatic use)
    gui_signal: Optional[str] = None       # Name of GUI signal to emit (e.g. "timer_set")
    gui_data: Optional[Dict] = None        # Payload for the GUI signal


# ═══════════════════════════════════════════════
# 3. BaseAgent — The contract every agent follows
# ═══════════════════════════════════════════════

class BaseAgent(ABC):
    """
    Abstract base class for all Jarvis agents.
    
    WHAT IS ABC (Abstract Base Class)?
        It's Python's way of saying "you MUST implement these methods."
        If you create a subclass and forget to implement get_functions()
        or execute(), Python will raise an error immediately — not at
        runtime when a user happens to trigger that code path.
        
        It's like a contract: "If you want to be an agent, you MUST
        provide these capabilities."
    
    WHAT IS @abstractmethod?
        Marks a method as "required to override." A class with any
        unimplemented abstract methods cannot be instantiated:
        
            class BadAgent(BaseAgent):
                name = "bad"
                # Forgot to implement get_functions() and execute()
            
            agent = BadAgent()  # ❌ TypeError: Can't instantiate abstract class
    
    HOW TO CREATE A NEW AGENT:
        1. Create a new file in agents/ (e.g. agents/spotify_agent.py)
        2. Subclass BaseAgent
        3. Set `name` and `description`
        4. Implement get_functions() — declare what you can do
        5. Implement execute() — do it
        6. That's it. The registry finds it automatically on startup.
    
    EXAMPLE:
        class SpotifyAgent(BaseAgent):
            name = "spotify"
            description = "Controls Spotify playback"
            
            def get_functions(self) -> List[FunctionDef]:
                return [
                    FunctionDef(
                        name="play_music",
                        description="Play a song or playlist",
                        parameters={
                            "query": {"type": "string", "description": "Song name"}
                        },
                        required_params=["query"]
                    )
                ]
            
            def execute(self, func_name: str, params: Dict[str, Any]) -> AgentResult:
                if func_name == "play_music":
                    # ... actual Spotify API call here ...
                    return AgentResult(True, f"Playing {params['query']}")
                return AgentResult(False, f"Unknown function: {func_name}")
    """
    
    # Class-level attributes — override these in your subclass
    name: str = "base"
    description: str = "Base agent"
    
    def __init__(self):
        self._initialized = False
    
    # ── Lifecycle ──
    
    def initialize(self) -> bool:
        """
        Called once when the agent is first needed (LAZY initialization).
        
        WHY LAZY?
            Not all agents are used in every session. If the user never
            asks about Spotify, we never connect to the Spotify API.
            This saves startup time and avoids errors from missing credentials.
        
        Override this to:
            - Connect to APIs
            - Load models or databases
            - Validate credentials
        
        Returns True if initialization succeeded.
        """
        self._initialized = True
        return True
    
    @property
    def is_initialized(self) -> bool:
        """Check if initialize() has been called successfully."""
        return self._initialized
    
    # ── Core Interface (MUST override) ──
    
    @abstractmethod
    def get_functions(self) -> List[FunctionDef]:
        """
        Declare what functions this agent handles.
        
        Called at registration time (startup). The returned FunctionDefs
        are used to:
          1. Build the router's tool schema (so it knows these functions exist)
          2. Map function names to this agent in the registry
        """
        ...
    
    @abstractmethod
    def execute(self, func_name: str, params: Dict[str, Any]) -> AgentResult:
        """
        Execute a function call.
        
        Args:
            func_name: Which function to run (from get_functions())
            params: The parameters extracted by the router
        
        Returns:
            AgentResult with success status, message, and optional data
        
        Must handle ALL functions declared in get_functions().
        """
        ...
    
    # ── Optional Interface ──
    
    def get_system_info(self) -> Optional[Dict[str, Any]]:
        """
        Contribute data to the system info aggregator.
        
        When the user asks "what's my status?" or the dashboard loads,
        the Dispatcher calls get_system_info() on every agent and
        merges the results.
        
        Override to report your agent's state. Return None if you
        have nothing to report.
        
        Example (from a timer agent):
            return {
                "active_timers": [
                    {"label": "Pasta", "remaining": "3m 22s"},
                    {"label": "Laundry", "remaining": "45m 10s"}
                ]
            }
        """
        return None
    
    def shutdown(self):
        """
        Clean up resources. Called when the app exits.
        
        Override to:
            - Close API connections
            - Save state
            - Release hardware resources
        """
        pass


# ═══════════════════════════════════════════════
# 4. AgentRegistry — The central lookup table
# ═══════════════════════════════════════════════

class AgentRegistry:
    """
    Discovers, registers, and manages all agents.
    
    HOW IT WORKS:
        1. Each agent registers itself:  registry.register(my_agent)
        2. The agent's functions are mapped:  "set_timer" → TimerAgent
        3. When a query is routed to "set_timer", the dispatcher asks:
           registry.get_agent_for_function("set_timer")  → TimerAgent
    
    INTERNAL DATA STRUCTURES:
        _agents:       {"timer": TimerAgent, "lights": LightAgent, ...}
        _function_map: {"set_timer": "timer", "control_light": "lights", ...}
    
    This is the REGISTRY PATTERN — a central directory that decouples
    the "thing that routes" from the "things that execute." The dispatcher
    doesn't import TimerAgent directly; it asks the registry.
    """
    
    def __init__(self):
        # Agent instances, keyed by agent name
        self._agents: Dict[str, BaseAgent] = {}
        
        # Maps function_name → agent_name for O(1) lookup
        self._function_map: Dict[str, str] = {}
    
    def register(self, agent: BaseAgent):
        """
        Register an agent and map all its functions.
        
        If two agents try to register the same function name,
        the second one wins (with a warning). This prevents silent conflicts.
        """
        self._agents[agent.name] = agent
        
        for func_def in agent.get_functions():
            if func_def.name in self._function_map:
                existing = self._function_map[func_def.name]
                print(
                    f"{GRAY}[Registry] ⚠ Function '{func_def.name}' already "
                    f"registered by '{existing}', overriding with '{agent.name}'{RESET}"
                )
            self._function_map[func_def.name] = agent.name
        
        func_count = len(agent.get_functions())
        func_names = [f.name for f in agent.get_functions()]
        print(
            f"{GREEN}[Registry] ✓ {agent.name} — "
            f"{func_count} function(s): {func_names}{RESET}"
        )
    
    # ── Lookups ──
    
    def get_agent(self, agent_name: str) -> Optional[BaseAgent]:
        """Get an agent by its name."""
        return self._agents.get(agent_name)
    
    def get_agent_for_function(self, func_name: str) -> Optional[BaseAgent]:
        """
        Get the agent that handles a specific function.
        
        This is the key method the Dispatcher uses:
            agent = registry.get_agent_for_function("set_timer")
            result = agent.execute("set_timer", {"duration": "5 minutes"})
        """
        agent_name = self._function_map.get(func_name)
        if agent_name:
            return self._agents.get(agent_name)
        return None
    
    # ── Bulk Operations ──
    
    def get_all_functions(self) -> List[FunctionDef]:
        """Get every registered function across all agents."""
        functions = []
        for agent in self._agents.values():
            functions.extend(agent.get_functions())
        return functions
    
    def get_all_tool_schemas(self) -> List[Dict]:
        """
        Get JSON tool schemas for the router model.
        
        The router model needs to know ALL available functions to make
        routing decisions. This builds the complete schema list dynamically
        from whatever agents are registered.
        
        When you add a new agent, its functions automatically appear here.
        """
        return [f.to_tool_schema() for f in self.get_all_functions()]
    
    def get_all_function_names(self) -> set:
        """Get set of all registered function names."""
        return set(self._function_map.keys())
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Aggregate system info from all initialized agents.
        
        Calls get_system_info() on each agent and merges results.
        Used by the dashboard and the "get_system_info" function.
        """
        info = {}
        for name, agent in self._agents.items():
            if agent.is_initialized:
                try:
                    agent_info = agent.get_system_info()
                    if agent_info:
                        info[name] = agent_info
                except Exception as e:
                    print(f"{GRAY}[Registry] Error from {name}: {e}{RESET}")
        return info
    
    def shutdown_all(self):
        """Shut down every agent cleanly."""
        for name, agent in self._agents.items():
            try:
                agent.shutdown()
                print(f"{GRAY}[Registry] Shut down: {name}{RESET}")
            except Exception as e:
                print(f"{GRAY}[Registry] Error shutting down {name}: {e}{RESET}")
    
    # ── Info ──
    
    @property
    def agent_count(self) -> int:
        return len(self._agents)
    
    @property
    def function_count(self) -> int:
        return len(self._function_map)
    
    def __repr__(self):
        return f"AgentRegistry({self.agent_count} agents, {self.function_count} functions)"


# ═══════════════════════════════════════════════
# 5. Auto-Discovery — Find and register all agents
# ═══════════════════════════════════════════════

def discover_agents(registry: AgentRegistry, agents_dir: str = None):
    """
    Auto-discover and register all agents in the agents/ package.
    
    HOW IT WORKS:
        1. Scans every .py file in agents/
        2. Imports each module
        3. Finds any class that subclasses BaseAgent
        4. Creates an instance and registers it
    
    This is the PLUGIN PATTERN — drop a new file in agents/ and it's
    automatically part of the system. No import statements to add,
    no lists to update, no config to change.
    
    WHY pkgutil.iter_modules?
        It's Python's built-in way to enumerate modules in a package.
        Better than os.listdir() because it handles edge cases like
        namespace packages and compiled modules.
    """
    if agents_dir is None:
        agents_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    
    print(f"{CYAN}[Registry] Discovering agents...{RESET}")
    
    for _, module_name, _ in pkgutil.iter_modules([agents_dir]):
        # Skip private modules and this file
        if module_name.startswith("_") or module_name == "base":
            continue
        
        try:
            # Import the module (e.g. agents.timer_agent)
            module = importlib.import_module(f"agents.{module_name}")
            
            # Find all BaseAgent subclasses in the module
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                
                # Check: is it a class? Is it a subclass of BaseAgent?
                # Is it not BaseAgent itself? Does it have no unimplemented abstract methods?
                if (isinstance(attr, type)
                        and issubclass(attr, BaseAgent)
                        and attr is not BaseAgent
                        and not getattr(attr, "__abstractmethods__", None)):
                    
                    # Create an instance and register it
                    agent_instance = attr()
                    registry.register(agent_instance)
        
        except Exception as e:
            print(f"{GRAY}[Registry] ✗ Failed to load {module_name}: {e}{RESET}")
    
    print(f"{CYAN}[Registry] Discovery complete: {registry}{RESET}")


# ═══════════════════════════════════════════════
# Global Registry Instance
# ═══════════════════════════════════════════════
# There's exactly one registry for the whole application.
# Every module imports this same instance.

registry = AgentRegistry()
