"""
Central Dispatcher
===================
The brain of Jarvis. Takes user input and produces a response.

THE FULL PIPELINE (Phase 4):
  1. User types or speaks a message
  2. Router (FunctionGemma) classifies the intent in ~200ms
  3. If it's a passthrough (thinking/nonthinking):
       → Send directly to Ollama LLM
  4. If it's a system info request:
       → Gather info from all agents → send to Ollama to phrase naturally
  5. If it's an agent action (set_timer, web_search, etc.):
       → Find the right agent in the registry
       → Execute the action
       → Send the result + original question to Ollama
       → Ollama phrases it naturally: "Done! Timer set for 5 minutes."
  6. The response streams token by token
  7. Complete sentences are sent to TTS for speaking
"""

import json
import requests
from datetime import datetime

from config import (
    RESPONDER_MODEL, OLLAMA_URL, MAX_HISTORY, SYSTEM_PROMPT, AI_NAME,
    GRAY, RESET, CYAN, GREEN, YELLOW
)
from core.model_manager import ensure_qwen_loaded, mark_qwen_used
from agents.base import registry


# Functions that skip the agent system and go straight to Ollama
PASSTHROUGH_FUNCTIONS = {"thinking", "nonthinking"}


class SentenceBuffer:
    """
    Collects streaming text tokens and extracts complete sentences.
    (Unchanged from Phase 3)
    """

    ENDINGS = ".!?"

    def __init__(self):
        self.buffer = ""

    def add(self, text: str) -> list:
        self.buffer += text
        sentences = []

        while True:
            end_idx = -1
            for i, char in enumerate(self.buffer):
                if char in self.ENDINGS:
                    if i + 1 >= len(self.buffer) or self.buffer[i + 1] == " ":
                        end_idx = i + 1
                        break

            if end_idx == -1:
                break

            sentence = self.buffer[:end_idx].strip()
            self.buffer = self.buffer[end_idx:].lstrip()

            if sentence:
                sentences.append(sentence)

        return sentences

    def flush(self) -> str:
        remaining = self.buffer.strip()
        self.buffer = ""
        return remaining if remaining else None


class Dispatcher:
    """
    Central command that routes queries, executes agents,
    and generates natural language responses.
    """

    def __init__(self):
        self.http_session = requests.Session()

        # Conversation history with personality-based system prompt
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

        # The router is loaded lazily (first time a message is sent)
        # because it takes a few seconds to load the model
        self.router = None

    # ─────────────────────────────────────────
    # Router (Lazy Loading)
    # ─────────────────────────────────────────

    def _ensure_router(self):
        """
        Load the FunctionGemma router on first use.
        
        We don't load it at startup because:
          1. It takes 2-5 seconds to load
          2. It downloads 500MB on first run
          3. The user might just want to chat (no routing needed)
        
        After the first call, self.router is set and reused.
        """
        if self.router is not None:
            return

        try:
            from core.router import FunctionGemmaRouter
            self.router = FunctionGemmaRouter(compile_model=False)
        except Exception as e:
            print(f"{YELLOW}[Dispatcher] Router failed to load: {e}{RESET}")
            print(f"{YELLOW}[Dispatcher] All queries will go to LLM directly.{RESET}")

    # ─────────────────────────────────────────
    # Main Processing Pipeline
    # ─────────────────────────────────────────

    def process(self, user_text: str):
        """
        Main entry point. Process a user message through the full pipeline.
        
        The flow:
          1. Route the message (which function should handle it?)
          2. If passthrough → send to LLM directly
          3. If agent action → execute agent → then ask LLM to phrase result
          4. Stream the response and send to TTS
        """
        # Step 1: Route the message
        func_name, params = self._route(user_text)
        print(f"{GRAY}[Dispatcher] Routed → {func_name}{RESET}")

        # Step 2: Handle based on function type
        if func_name in PASSTHROUGH_FUNCTIONS:
            # General chat — send directly to LLM
            # Note: we always disable explicit thinking mode and let
            # the model decide naturally. The router's thinking vs
            # nonthinking classification is unreliable on edge cases.
            self._stream_llm(user_text, enable_thinking=False)

        elif func_name == "get_system_info":
            # System info — gather from all agents, then phrase naturally
            result = self._execute_system_info()
            self._respond_with_context(func_name, result, user_text)

        else:
            # Agent action — execute the agent, then phrase the result
            result = self._execute_agent(func_name, params)

            # Print the raw result for debugging
            status = "✓" if result.success else "✗"
            print(f"{GRAY}[Dispatcher] Agent result: {status} {result.message}{RESET}")

            # Ask Ollama to phrase the result naturally
            self._respond_with_context(func_name, result, user_text)

    # ─────────────────────────────────────────
    # Routing
    # ─────────────────────────────────────────

    def _route(self, user_text: str):
        """
        Classify the user's intent.
        
        First checks keyword triggers for agents the router wasn't
        trained on (like news). Then falls through to FunctionGemma
        for everything else.
        
        WHY KEYWORD PRE-ROUTING?
          FunctionGemma was fine-tuned on a specific set of functions.
          When we add new agents (news, weather, etc.), the router
          doesn't know they exist. Keyword matching catches these
          before the router ever sees them.
          
          This is a bridge — eventually you'd retrain the router
          with the new functions. But keywords work perfectly for now.
        """
        lower = user_text.lower()

        # ── Keyword pre-routing for agents the router doesn't know ──
        # Check these BEFORE sending to FunctionGemma

       # News agent triggers
        news_triggers = [
            "news", "headlines", "brief me", "catch me up",
            "what's happening", "whats happening", "what is happening",
            "what did i miss", "world update", "world monitor",
            "open world monitor", "briefing",
        ]
        if any(trigger in lower for trigger in news_triggers):
            print(f"{GRAY}[Dispatcher] Keyword matched → get_news{RESET}")
            return "get_news", {"query": user_text}

        # Task agent triggers — adding tasks
        add_task_triggers = [
            "add to my list", "add to my to do", "add to my todo",
            "add task", "new task", "remind me to", "i need to",
            "to do list add", "todo add",
        ]
        if any(trigger in lower for trigger in add_task_triggers):
            print(f"{GRAY}[Dispatcher] Keyword matched → add_task{RESET}")
            return "add_task", {"text": user_text}

        # Task agent triggers — completing tasks
        complete_task_triggers = [
            "mark", "complete task", "finish task", "check off",
            "task done", "task complete", "i finished",
        ]
        if any(trigger in lower for trigger in complete_task_triggers):
            print(f"{GRAY}[Dispatcher] Keyword matched → complete_task{RESET}")
            return "complete_task", {"text": user_text}

        # Task agent triggers — listing tasks
        list_task_triggers = [
            "my to do", "my todo", "my tasks", "to-do list",
            "what do i need to do", "show my tasks", "pending tasks",
            "task list",
        ]
        if any(trigger in lower for trigger in list_task_triggers):
            print(f"{GRAY}[Dispatcher] Keyword matched → list_tasks{RESET}")
            return "list_tasks", {}

        # Calendar agent triggers — creating events
        create_event_triggers = [
            "schedule", "create event", "add event", "new event",
            "book a", "plan a", "set up a meeting",
            "i have a meeting", "i have an appointment",
        ]
        if any(trigger in lower for trigger in create_event_triggers):
            print(f"{GRAY}[Dispatcher] Keyword matched → create_event{RESET}")
            return "create_event", {"title": user_text}

        # Calendar agent triggers — reading events
        read_calendar_triggers = [
            "my schedule", "my calendar", "what's on my calendar",
            "whats on my calendar", "what is on my calendar",
            "any events", "am i free", "what do i have",
            "coming up", "upcoming events", "plans for",
        ]
        if any(trigger in lower for trigger in read_calendar_triggers):
            print(f"{GRAY}[Dispatcher] Keyword matched → read_calendar{RESET}")
            return "read_calendar", {"date": user_text}

        # Calendar agent triggers — deleting events
        delete_event_triggers = [
            "cancel event", "cancel the", "cancel my",
            "delete event", "remove event", "drop event",
        ]
        if any(trigger in lower for trigger in delete_event_triggers):
            print(f"{GRAY}[Dispatcher] Keyword matched → delete_event{RESET}")
            return "delete_event", {"title": user_text}

        # ── FunctionGemma router for everything else ──
        self._ensure_router()

        if self.router is None:
            return "nonthinking", {"prompt": user_text}

        try:
            (func_name, params), elapsed = self.router.route_with_timing(user_text)
            print(f"{GRAY}[Dispatcher] Routed in {elapsed*1000:.0f}ms{RESET}")
            return func_name, params
        except Exception as e:
            print(f"{GRAY}[Dispatcher] Route error: {e}{RESET}")
            return "nonthinking", {"prompt": user_text}

        # Timer agent triggers
        timer_triggers = [
            "set a timer", "set timer", "timer for", "start a timer",
            "countdown", "count down",
        ]
        if any(trigger in lower for trigger in timer_triggers):
            # Extract the duration from the query
            print(f"{GRAY}[Dispatcher] Keyword matched → set_timer{RESET}")
            return "set_timer", {"duration": user_text}

        # Alarm agent triggers
        alarm_triggers = [
            "set an alarm", "set alarm", "alarm for", "alarm at",
            "wake me up", "wake me at", "remind me at",
        ]
        if any(trigger in lower for trigger in alarm_triggers):
            print(f"{GRAY}[Dispatcher] Keyword matched → set_alarm{RESET}")
            return "set_alarm", {"time": user_text}
        
        # Search agent triggers
        search_triggers = [
            "search for", "search the web", "look up", "lookup",
            "google", "find information", "find info",
        ]
        if any(trigger in lower for trigger in search_triggers):
            print(f"{GRAY}[Dispatcher] Keyword matched → web_search{RESET}")
            return "web_search", {"query": user_text}

        # Weather agent triggers
        weather_triggers = [
            "weather", "temperature", "forecast",
            "how is it outside", "how's it outside",
            "do i need an umbrella", "will it rain",
            "is it cold", "is it hot", "is it warm",
        ]
        if any(trigger in lower for trigger in weather_triggers):
            print(f"{GRAY}[Dispatcher] Keyword matched → get_weather{RESET}")
            return "get_weather", {}

    # ─────────────────────────────────────────
    # Agent Execution
    # ─────────────────────────────────────────

    def _execute_agent(self, func_name: str, params: dict):
        """
        Find the right agent and execute the function.
        
        Uses the registry to look up which agent handles this function.
        If the agent hasn't been initialized yet, initializes it first
        (lazy loading — don't connect to APIs until actually needed).
        """
        from agents.base import AgentResult

        # Look up which agent handles this function
        agent = registry.get_agent_for_function(func_name)

        if agent is None:
            return AgentResult(
                success=False,
                message=f"No agent registered for '{func_name}'"
            )

        # Lazy initialize the agent if needed
        if not agent.is_initialized:
            try:
                if not agent.initialize():
                    return AgentResult(
                        success=False,
                        message=f"Agent '{agent.name}' failed to initialize"
                    )
            except Exception as e:
                return AgentResult(
                    success=False,
                    message=f"Agent '{agent.name}' init error: {e}"
                )

        # Execute the function
        try:
            return agent.execute(func_name, params)
        except Exception as e:
            return AgentResult(
                success=False,
                message=f"Agent '{agent.name}' error: {e}"
            )

    def _execute_system_info(self):
        """
        Gather system info from all initialized agents.
        
        Calls get_system_info() on every agent and merges the results.
        Used when the user asks "what's my status?" or "what do I have going on?"
        """
        from agents.base import AgentResult

        info = {
            "current_time": datetime.now().strftime("%A, %B %d, %Y at %I:%M %p"),
        }

        # Collect info from all agents
        agent_info = registry.get_system_info()
        info.update(agent_info)

        return AgentResult(
            success=True,
            message="System info retrieved",
            data=info,
        )

    # ─────────────────────────────────────────
    # LLM Response Generation
    # ─────────────────────────────────────────

    def _respond_with_context(self, func_name: str, result, user_text: str):
        if not ensure_qwen_loaded():
            return
        mark_qwen_used()

        if func_name == "get_news" and result.success:
            context_msg = (
                f"You just fetched the latest news for the user. "
                f"Here are the actual headlines:\n\n"
                f"{result.message}\n\n"
                f"Brief the user conversationally, like a news anchor speaking. "
                f"Weave the stories together naturally in flowing sentences. "
                f"Do NOT use numbered lists, bullet points, or any list format. "
                f"Do NOT start sentences with numbers. "
                f"Speak as if you are talking, not writing a report. "
                f"Cover the key stories but keep it to 3-4 sentences total. "
                f"Do not make up any stories. "
                f"Do not mention a world monitor unless the headlines explicitly say so. "
                f"Do not say you opened anything unless the headlines say you did."
            )
        elif func_name == "get_system_info" and result.success:
            data = result.data or {}
            context_parts = []
            for key, value in data.items():
                context_parts.append(f"{key}: {json.dumps(value, default=str)}")
            context_msg = "CURRENT SYSTEM STATE:\n" + "\n".join(context_parts)
        else:
            context_msg = (
                f"Action: {func_name}\n"
                f"Success: {result.success}\n"
                f"Result: {result.message}"
            )

        prompt = f"{context_msg}\n\nUser said: {user_text}\n\nRespond naturally and concisely."
        self._stream_llm(prompt, enable_thinking=False)

    def _stream_llm(self, user_text: str, enable_thinking: bool = False):
        """
        Stream a response from Ollama, printing tokens and queuing TTS.
        (Same as Phase 3, but uses personality-based system prompt)
        """
        if not ensure_qwen_loaded():
            print(f"{YELLOW}[{AI_NAME}] Could not load model. Is Ollama running?{RESET}")
            return ""

        mark_qwen_used()

        # Trim history if too long
        if len(self.messages) > MAX_HISTORY:
            self.messages = [self.messages[0]] + self.messages[-(MAX_HISTORY - 1):]

        self.messages.append({"role": "user", "content": user_text})

        payload = {
            "model": RESPONDER_MODEL,
            "messages": self.messages,
            "stream": True,
        }

        full_response = ""
        sentence_buffer = SentenceBuffer()

        from core.tts import tts_engine

        try:
            with self.http_session.post(
                f"{OLLAMA_URL}/chat",
                json=payload,
                stream=True,
            ) as response:
                response.raise_for_status()

                print(f"\n{GREEN}{AI_NAME}:{RESET} ", end="", flush=True)

                for line in response.iter_lines():
                    if not line:
                        continue

                    try:
                        chunk = json.loads(line.decode("utf-8"))
                        message = chunk.get("message", {})
                        content = message.get("content", "")

                        if content:
                            print(content, end="", flush=True)
                            full_response += content

                            complete_sentences = sentence_buffer.add(content)
                            for sentence in complete_sentences:
                                tts_engine.queue_sentence(sentence)

                    except json.JSONDecodeError:
                        continue

                remaining = sentence_buffer.flush()
                if remaining:
                    tts_engine.queue_sentence(remaining)

                print()

        except requests.exceptions.ConnectionError:
            print(f"\n{YELLOW}[{AI_NAME}] Cannot connect to Ollama at {OLLAMA_URL}{RESET}")
            print(f"{YELLOW}         Make sure Ollama is running: ollama serve{RESET}")

        except requests.exceptions.HTTPError as e:
            print(f"\n{YELLOW}[{AI_NAME}] HTTP error: {e}{RESET}")

        except Exception as e:
            print(f"\n{YELLOW}[{AI_NAME}] Error: {e}{RESET}")

        if full_response:
            self.messages.append({"role": "assistant", "content": full_response})

        mark_qwen_used()
        return full_response

    def clear_history(self):
        """Reset conversation."""
        self.messages = [self.messages[0]]
        print(f"{CYAN}[{AI_NAME}] Conversation cleared.{RESET}")

    def shutdown(self):
        """Clean up resources."""
        from core.tts import tts_engine
        tts_engine.shutdown()
        self.http_session.close()
        print(f"{CYAN}[{AI_NAME}] Dispatcher shut down.{RESET}")


# Global Instance
dispatcher = Dispatcher()