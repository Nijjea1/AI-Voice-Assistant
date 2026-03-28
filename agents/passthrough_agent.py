"""
Passthrough Agent
==================
Handles general queries that don't need a specialized agent.

This is the "catch-all" — if no other agent matches, the query
comes here and gets passed through to the LLM directly.

Two modes:
  - "thinking"    → Complex queries (math, code, reasoning). 
                     Tells the LLM to use chain-of-thought.
  - "nonthinking" → Simple queries (greetings, facts, chitchat).
                     Tells the LLM to respond quickly.

Also handles "get_system_info" which aggregates status from all agents.

NOTE: The Dispatcher handles the actual LLM call and system info
      aggregation. This agent just declares the functions exist so
      the router knows about them.
"""

from typing import List, Dict, Any

from agents.base import BaseAgent, FunctionDef, AgentResult


class PassthroughAgent(BaseAgent):
    """Routes general queries to the LLM."""
    
    name = "passthrough"
    description = "Handles general conversation, reasoning, and system info"
    
    def get_functions(self) -> List[FunctionDef]:
        return [
            FunctionDef(
                name="thinking",
                description=(
                    "Use for complex queries requiring reasoning, math, "
                    "coding, or multi-step analysis."
                ),
                parameters={
                    "prompt": {
                        "type": "string",
                        "description": "The user's original prompt",
                    }
                },
                required_params=["prompt"],
            ),
            FunctionDef(
                name="nonthinking",
                description=(
                    "DEFAULT function. Use for simple queries, greetings, "
                    "factual questions, and anything that doesn't clearly "
                    "need another function."
                ),
                parameters={
                    "prompt": {
                        "type": "string",
                        "description": "The user's original prompt",
                    }
                },
                required_params=["prompt"],
            ),
            FunctionDef(
                name="get_system_info",
                description=(
                    "Get current system state: time, active timers, alarms, "
                    "calendar events, pending tasks, smart device status, "
                    "weather, and recent news headlines."
                ),
                parameters={},
                required_params=[],
            ),
        ]
    
    def execute(self, func_name: str, params: Dict[str, Any]) -> AgentResult:
        """
        The Dispatcher handles these functions directly:
          - thinking/nonthinking → streamed to LLM
          - get_system_info → aggregated from all agents
        
        This execute() is technically reachable but the Dispatcher
        short-circuits before calling it. It exists to satisfy the
        abstract method requirement.
        """
        return AgentResult(
            success=True,
            message="Passthrough to LLM",
        )
