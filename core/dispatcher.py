"""
Central Dispatcher
===================
The brain of Jarvis. Takes user input and produces a response.

WHAT IT DOES NOW (Phase 2):
  User types text → Dispatcher sends to Ollama → streams response back

WHAT IT WILL DO LATER:
  User types text → Router classifies intent → Agent executes action
  → Dispatcher generates natural language response via Ollama → TTS speaks it

THE STREAMING CONCEPT:
  When you chat with ChatGPT, you see words appear one at a time, not all
  at once. That's "streaming." Ollama supports this too.
  
  Without streaming: Send prompt → wait 10 seconds → get entire response at once
  With streaming:    Send prompt → get first word in 0.5s → more words trickle in
  
  Streaming works by making an HTTP request with stream=True. Instead of
  waiting for the entire response, we read it line by line as Ollama generates
  tokens. Each line is a tiny JSON object with one chunk of text.

HOW OLLAMA'S CHAT API WORKS:
  We POST to /api/chat with:
  {
      "model": "qwen3:1.7b",
      "messages": [
          {"role": "system", "content": "You are Jarvis..."},
          {"role": "user", "content": "Hello"},
          {"role": "assistant", "content": "Hi there!"},
          {"role": "user", "content": "What's the weather?"}
      ],
      "stream": true
  }
  
  Ollama responds with many lines, one per token:
      {"message": {"content": "The"}, "done": false}
      {"message": {"content": " weather"}, "done": false}
      {"message": {"content": " is"}, "done": false}
      ...
      {"message": {"content": "."}, "done": true}
  
  We print each token as it arrives, giving that "typing" effect.
  
  The "messages" list is the CONVERSATION HISTORY. We include previous
  turns so the model has context. Without it, every message would be
  like talking to someone with amnesia.
"""

import json
import requests

from config import (
    RESPONDER_MODEL, OLLAMA_URL, MAX_HISTORY,
    GRAY, RESET, CYAN, GREEN, YELLOW
)
from core.model_manager import ensure_qwen_loaded, mark_qwen_used


class Dispatcher:
    """
    Central command that processes user queries.
    
    For now: sends everything to Ollama.
    Later: routes to agents first, then Ollama for natural language response.
    """
    
    def __init__(self):
        # HTTP session — reuses connections to Ollama (faster)
        self.http_session = requests.Session()
        
        # Conversation history — the model needs this for context
        # The system message sets the model's personality
        self.messages = [
            {
                "role": "system",
                "content": (
                    "You are Jarvis, a helpful AI assistant. "
                    "Respond in short, complete sentences. "
                    "Never use emojis or special characters. "
                    "Keep responses concise and conversational."
                ),
            }
        ]
    
    def process(self, user_text: str):
        """
        Process a user message. Sends to Ollama and streams the response.
        
        This is the main entry point. In the terminal chat loop,
        we call this for every line the user types.
        
        Args:
            user_text: What the user typed
            
        Returns:
            The full response string (also printed during streaming)
        """
        # Step 1: Make sure the model is loaded in VRAM
        if not ensure_qwen_loaded():
            print(f"{YELLOW}[Jarvis] Could not load model. Is Ollama running?{RESET}")
            print(f"{YELLOW}         Start it with: ollama serve{RESET}")
            print(f"{YELLOW}         Pull model with: ollama pull {RESPONDER_MODEL}{RESET}")
            return ""
        
        # Step 2: Add user message to conversation history
        self.messages.append({"role": "user", "content": user_text})
        
        # Step 3: Trim history if it's too long
        # We keep the system message (index 0) and the most recent messages.
        # Without trimming, the conversation would eventually exceed the
        # model's context window (how much text it can "see" at once).
        if len(self.messages) > MAX_HISTORY:
            # Keep system message + last (MAX_HISTORY - 1) messages
            self.messages = [self.messages[0]] + self.messages[-(MAX_HISTORY - 1):]
        
        # Step 4: Stream the response from Ollama
        full_response = self._stream_response()
        
        # Step 5: Add assistant response to history (for future context)
        if full_response:
            self.messages.append({"role": "assistant", "content": full_response})
        
        # Step 6: Tell the model manager we used the model (resets idle timer)
        mark_qwen_used()
        
        return full_response
    
    def _stream_response(self) -> str:
        """
        Send the conversation to Ollama and stream the response token by token.
        
        Returns the complete response as a string.
        
        HOW STREAMING WORKS UNDER THE HOOD:
          1. We make an HTTP POST with stream=True
          2. This tells the `requests` library: "Don't wait for the full response.
             Give me data as it arrives."
          3. r.iter_lines() yields one line at a time as Ollama generates tokens
          4. Each line is JSON with a "message.content" field containing one token
          5. We print each token immediately (no newline) using end="" and flush=True
          6. flush=True forces Python to display the text immediately instead of
             buffering it (Python normally waits until a full line before displaying)
        """
        # Build the request payload
        payload = {
            "model": RESPONDER_MODEL,
            "messages": self.messages,
            "stream": True,             # Stream tokens one at a time
        }
        
        full_response = ""
        
        try:
            # Make the HTTP request
            # "with" ensures the connection is properly closed when we're done
            with self.http_session.post(
                f"{OLLAMA_URL}/chat",
                json=payload,           # Automatically converts dict to JSON
                stream=True,            # Tell requests to stream the response
            ) as response:
                
                # Check for errors (wrong URL, model not found, etc.)
                response.raise_for_status()
                
                # Print a colored prefix for the response
                print(f"\n{GREEN}Jarvis:{RESET} ", end="", flush=True)
                
                # Read response line by line as Ollama generates tokens
                for line in response.iter_lines():
                    if not line:
                        continue
                    
                    try:
                        # Each line is a JSON object like:
                        # {"message": {"content": "Hello"}, "done": false}
                        chunk = json.loads(line.decode("utf-8"))
                        
                        # Extract the text content from this chunk
                        message = chunk.get("message", {})
                        content = message.get("content", "")
                        
                        if content:
                            # Print this token immediately
                            print(content, end="", flush=True)
                            full_response += content
                        
                    except json.JSONDecodeError:
                        # Malformed line — skip it
                        continue
                
                # After all tokens, print a newline
                print()
        
        except requests.exceptions.ConnectionError:
            print(f"\n{YELLOW}[Jarvis] Cannot connect to Ollama at {OLLAMA_URL}{RESET}")
            print(f"{YELLOW}         Make sure Ollama is running: ollama serve{RESET}")
        
        except requests.exceptions.HTTPError as e:
            print(f"\n{YELLOW}[Jarvis] HTTP error: {e}{RESET}")
            print(f"{YELLOW}         Is the model pulled? Try: ollama pull {RESPONDER_MODEL}{RESET}")
        
        except Exception as e:
            print(f"\n{YELLOW}[Jarvis] Error: {e}{RESET}")
        
        return full_response
    
    def clear_history(self):
        """Reset conversation — start fresh."""
        self.messages = [self.messages[0]]  # Keep only the system message
        print(f"{CYAN}[Jarvis] Conversation cleared.{RESET}")
    
    def shutdown(self):
        """Clean up resources."""
        self.http_session.close()
        print(f"{CYAN}[Jarvis] Dispatcher shut down.{RESET}")


# ─────────────────────────────────────────────
# Global Instance
# ─────────────────────────────────────────────
dispatcher = Dispatcher()