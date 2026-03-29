"""
Model Manager
==============
Manages the Ollama LLM model's lifecycle in GPU memory (VRAM).

THE PROBLEM:
  Your GPU has limited memory in my case 8GB. The chat model (Qwen) that we use takes ~2GB.
  If it sits loaded all day doing nothing, that's 2GB wasted. And when you
  want to run the browser agent (which needs its own model), there's no room.

THE SOLUTION:
  - Load the model the first time someone chats (not at startup)
  - Keep it loaded while the user is active
  - After 5 minutes of silence, unload it automatically
  - Before loading a different model, unload the current one first

HOW OLLAMA WORKS:
  Ollama is a server running on your machine at localhost:11434.
  We talk to it over HTTP, just like a web browser talks to a website.
  
  To load a model:   POST /api/generate with {"model": "qwen3:1.7b", "keep_alive": "5m"}
  To unload a model:  POST /api/generate with {"model": "qwen3:1.7b", "keep_alive": 0}
  To check what's loaded: GET /api/ps
  
  "keep_alive" tells Ollama how long to keep the model in memory.
  Setting it to 0 means "unload immediately."
"""

import threading
import time
import requests

from config import (
    RESPONDER_MODEL, OLLAMA_URL, QWEN_TIMEOUT_SECONDS,
    QWEN_KEEP_ALIVE, GRAY, RESET, CYAN
)


# ═══════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════

def get_running_models() -> list:
    """
    Ask Ollama "what models are currently loaded in memory?"
    
    Returns a list of model name strings, e.g. ["qwen3:1.7b"]
    Returns empty list if Ollama isn't running or errors out.
    """
    try:
        response = requests.get(f"{OLLAMA_URL}/ps", timeout=2)
        if response.status_code == 200:
            data = response.json()
            # data looks like: {"models": [{"name": "qwen3:1.7b", ...}]}
            return [m.get("name", "") for m in data.get("models", [])]
    except Exception:
        # Ollama might not be running — that's fine, return empty
        pass
    return []


def sync_unload_model(model_name: str):
    """
    Tell Ollama to immediately unload a model from memory.
    
    "sync" means this function waits until the unload is done
    before returning (as opposed to firing and forgetting).
    """
    try:
        response = requests.post(
            f"{OLLAMA_URL}/generate",
            json={
                "model": model_name,
                "prompt": "",           # Empty prompt — we don't want a response
                "keep_alive": 0         # 0 = unload immediately
            },
            timeout=5
        )
        if response.status_code == 200:
            print(f"{GRAY}[ModelMgr] Unloaded: {model_name}{RESET}")
    except Exception as e:
        print(f"{GRAY}[ModelMgr] Unload error ({model_name}): {e}{RESET}")


def unload_all_models(sync: bool = False):
    """
    Unload every model Ollama currently has loaded.
    
    sync=True:  Wait for each unload to finish (used during app shutdown)
    sync=False: Fire and forget (faster, used during normal operation)
    """
    for model_name in get_running_models():
        if model_name:
            if sync:
                sync_unload_model(model_name)
            else:
                # Run in background thread so we don't block
                threading.Thread(
                    target=sync_unload_model,
                    args=(model_name,),
                    daemon=True     # daemon=True means thread dies when app exits
                ).start()


# ═══════════════════════════════════════════════
# QwenModelManager — The Main Class
# ═══════════════════════════════════════════════

class QwenModelManager:
    """
    Manages the responder model (Qwen) with automatic idle timeout.
    
    LIFECYCLE:
        1. User sends first message
        2. ensure_loaded() loads the model into VRAM
        3. mark_used() updates the "last used" timestamp on each interaction
        4. A background thread watches the timestamp
        5. After 5 minutes of no activity, it unloads the model
        6. Next message triggers ensure_loaded() again
    """
    
    def __init__(self):
        self.model_name = RESPONDER_MODEL
        self.last_used = None           # Timestamp of last interaction (float)
        self.is_loaded = False           # Is the model currently in VRAM?
        self.lock = threading.Lock()     # Prevent race conditions
        self.monitoring = False          # Is the timeout watcher running?
        self.http = requests.Session()   # Reuse HTTP connections (faster)
        
        # WHY requests.Session()?
        # Normal requests.get() opens a new TCP connection each time.
        # Session() reuses the same connection, which is faster when
        # you're making many requests to the same server (Ollama).
    
    def ensure_loaded(self) -> bool:
        """
        Make sure the model is in VRAM. Load it if it isn't.
        
        Returns True if model is ready, False if loading failed.
        
        This is the method the Dispatcher calls before every LLM request.
        """
        with self.lock:
            if self.is_loaded:
                # Already loaded — just update the timestamp
                self.last_used = time.time()
                return True
            
            # Maybe it's already running (e.g. user loaded it manually)
            running = get_running_models()
            if any(self.model_name in m for m in running):
                print(f"{CYAN}[ModelMgr] {self.model_name} already running{RESET}")
                self.is_loaded = True
                self.last_used = time.time()
                self._start_monitor()
                return True
            
            # Actually load it by sending a tiny request
            try:
                print(f"{CYAN}[ModelMgr] Loading {self.model_name}...{RESET}")
                response = self.http.post(
                    f"{OLLAMA_URL}/generate",
                    json={
                        "model": self.model_name,
                        "prompt": "hi",            # Minimal prompt to trigger loading
                        "stream": False,            # Wait for full response
                        "keep_alive": QWEN_KEEP_ALIVE,
                        "options": {"num_predict": 1}  # Generate just 1 token (fast)
                    },
                    timeout=120   # 2 minutes — first load downloads the model
                )
                
                if response.status_code == 200:
                    self.is_loaded = True
                    self.last_used = time.time()
                    print(f"{CYAN}[ModelMgr] ✓ {self.model_name} loaded{RESET}")
                    self._start_monitor()
                    return True
                else:
                    print(f"{GRAY}[ModelMgr] Load failed: HTTP {response.status_code}{RESET}")
                    return False
                    
            except Exception as e:
                print(f"{GRAY}[ModelMgr] Load error: {e}{RESET}")
                return False
    
    def mark_used(self):
        """
        Update the "last used" timestamp.
        
        Called by the Dispatcher after every interaction. This pushes
        back the idle timeout — as long as you keep chatting, the
        model stays loaded.
        """
        with self.lock:
            self.last_used = time.time()
    
    def unload(self, reason: str = "manual"):
        """Unload the model from VRAM."""
        with self.lock:
            if not self.is_loaded:
                return
            self.monitoring = False
            sync_unload_model(self.model_name)
            self.is_loaded = False
            self.last_used = None
    
    def _start_monitor(self):
        """Start the background thread that watches for idle timeout."""
        self.monitoring = True
        # daemon=True: thread dies automatically when the app exits
        # No need to manually stop it
        threading.Thread(target=self._monitor_loop, daemon=True).start()
    
    def _monitor_loop(self):
        """
        Background loop that checks every 10 seconds:
        "Has it been more than 5 minutes since the last interaction?"
        If yes, unload the model.
        
        WHY A LOOP AND NOT A TIMER?
          A single timer would fire once. But each new message should
          reset the countdown. With a loop, we just check the timestamp
          each time — if it's been updated, the countdown effectively resets.
        """
        while self.monitoring:
            time.sleep(10)   # Check every 10 seconds
            
            with self.lock:
                if not self.monitoring or not self.is_loaded or not self.last_used:
                    break
                
                elapsed = time.time() - self.last_used
                
                if elapsed >= QWEN_TIMEOUT_SECONDS:
                    print(f"{GRAY}[ModelMgr] Idle for {elapsed:.0f}s, unloading...{RESET}")
                    # Can't call self.unload() here because we already hold the lock
                    # So we do the unload inline
                    self.monitoring = False
                    sync_unload_model(self.model_name)
                    self.is_loaded = False
                    self.last_used = None
                    break


# ═══════════════════════════════════════════════
# Global Instance + Public Functions
# ═══════════════════════════════════════════════
# Other files call these simple functions instead of
# touching the manager object directly.

_qwen_mgr = QwenModelManager()


def ensure_qwen_loaded() -> bool:
    """Make sure the chat model is loaded. Returns True if ready."""
    return _qwen_mgr.ensure_loaded()


def mark_qwen_used():
    """Tell the manager we just used the model (resets idle timer)."""
    _qwen_mgr.mark_used()


def unload_qwen(reason: str = "manual"):
    """Manually unload the chat model."""
    _qwen_mgr.unload(reason)