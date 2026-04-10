"""
FunctionGemma Router
=====================
A fine-tuned 270M parameter model that classifies user intent.

WHAT IT DOES:
  You say: "Set a timer for 5 minutes"
  Router says: function="set_timer", params={"duration": "5 minutes"}

  You say: "Hello, how are you?"
  Router says: function="nonthinking", params={"prompt": "Hello, how are you?"}

  It does this in ~50ms on GPU or ~200ms on CPU.

HOW IT WORKS:
  1. We load a fine-tuned version of Google's FunctionGemma model
  2. We give it a list of ALL available functions (from the agent registry)
  3. We give it the user's text
  4. It outputs which function to call and what parameters to pass

  The model was trained specifically on this task — it's not a general
  chatbot, it's a classifier. It reads the function descriptions and
  decides which one matches.

AUTO-DOWNLOAD:
  The model lives on HuggingFace at "nlouis/pocket-ai-router".
  First time you run the router, it downloads ~500MB of model weights.
  After that, it loads from the local ./merged_model/ folder instantly.

CPU vs GPU:
  The code auto-detects CUDA availability. If your GPU supports it,
  it runs on GPU (~50ms). Otherwise CPU (~200ms). Both are fine —
  the bottleneck is the LLM, not the router.
"""

import os
import re
import time
import warnings

# Suppress noisy warnings from transformers library
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore", message=".*generation flags.*")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import logging as tf_logging
from huggingface_hub import snapshot_download
from typing import Tuple, Dict, Any

# Suppress transformers logging (it's very chatty)
tf_logging.set_verbosity_error()

from config import LOCAL_ROUTER_PATH, HF_ROUTER_REPO, GRAY, RESET, GREEN, CYAN, YELLOW
from agents.base import registry


def ensure_model_available(model_path: str = LOCAL_ROUTER_PATH) -> str:
    """
    Make sure the router model exists locally.
    If not, download it from HuggingFace automatically.
    
    This is why the first run takes a minute — it's downloading ~500MB.
    After that, the model is cached in ./merged_model/ and loads instantly.
    
    Returns the path to the model directory.
    """
    # Check if model files already exist locally
    if os.path.exists(model_path) and os.path.exists(
        os.path.join(model_path, "model.safetensors")
    ):
        return model_path

    # Not found locally — download from HuggingFace
    print(f"{CYAN}[Router] Model not found locally. Downloading from HuggingFace...{RESET}")
    print(f"{CYAN}[Router] Repo: {HF_ROUTER_REPO} (this may take a minute){RESET}")

    try:
        # snapshot_download downloads the entire model repository
        # local_dir: where to save it
        # local_dir_use_symlinks=False: copy files instead of symlinking
        #   (symlinks cause issues on Windows)
        downloaded_path = snapshot_download(
            repo_id=HF_ROUTER_REPO,
            local_dir=model_path,
            local_dir_use_symlinks=False,
        )
        print(f"{GREEN}[Router] ✓ Model downloaded to {downloaded_path}{RESET}")
        return downloaded_path
    except Exception as e:
        raise RuntimeError(
            f"Failed to download router model from {HF_ROUTER_REPO}: {e}\n"
            f"Check your internet connection and try again."
        )


class FunctionGemmaRouter:
    """
    Routes user prompts to the correct agent function.
    
    HOW THE MODEL WORKS INTERNALLY:
      FunctionGemma is a tiny language model (270M params) that was
      fine-tuned on examples like:
      
        Input:  "Turn on the living room lights"
        Output: "call:control_light{action:<escape>on<escape>,device_name:<escape>living room<escape>}"
      
      It learned to map natural language to function calls. The output
      format is specific to how it was trained — we parse it with regex.
    
    TOOL SCHEMAS:
      We give the model a list of all available functions (from the
      agent registry) so it knows what options exist. If a new agent
      is registered, its functions automatically appear in the schema.
    """

    def __init__(self, model_path: str = LOCAL_ROUTER_PATH, compile_model: bool = False):
        # Step 1: Make sure model is downloaded
        model_path = ensure_model_available(model_path)

        # Step 2: Detect device (GPU vs CPU)
        # torch.cuda.is_available() returns True if:
        #   - You have an NVIDIA GPU
        #   - CUDA drivers are installed
        #   - PyTorch was built with CUDA support for your GPU architecture
        # For RTX 5060 (sm_120), this may return False on stable PyTorch
        # builds. That's fine — we fall back to CPU.
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # bfloat16 is a compact number format that uses half the memory
        # of regular floats. GPUs handle it natively. CPUs often don't,
        # so we use regular float32 on CPU.
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        print(f"{CYAN}[Router] Loading FunctionGemma on {device.upper()}...{RESET}")
        start = time.time()

        # Step 3: Load the tokenizer
        # The tokenizer converts text to numbers (tokens) and back.
        # "Hello world" → [15496, 995] → model processes → [1234, 5678] → "call:nonthinking"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Step 4: Load the model weights
        # AutoModelForCausalLM loads the neural network with the right architecture.
        # torch_dtype: use bfloat16 on GPU for speed, float32 on CPU for compatibility
        # device_map: put the model on the right device (GPU or CPU)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device,
        )

        # Step 5: Set to evaluation mode
        # In training mode, the model tracks gradients (for learning).
        # In eval mode, it skips that work (faster inference).
        self.model.eval()

        # Optional: torch.compile for extra speed (PyTorch 2.0+)
        # This JIT-compiles the model for faster execution.
        # Sometimes fails on certain hardware, so we catch errors.
        if compile_model:
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print(f"{CYAN}[Router] Model compiled with torch.compile(){RESET}")
            except Exception:
                pass

        # Step 6: Build tool schemas from the agent registry
        # These tell the model what functions exist and what they do.
        # When a new agent registers, its functions appear here automatically.
        self._tools = registry.get_all_tool_schemas()
        self._valid_functions = registry.get_all_function_names()
        self._system_msg = "You are a model that can do function calling with the following functions"

        elapsed = time.time() - start
        print(f"{GREEN}[Router] ✓ Loaded in {elapsed:.2f}s on {device.upper()} "
              f"({len(self._valid_functions)} functions){RESET}")

    def refresh_tools(self):
        """
        Rebuild the tool list from the registry.
        Call this if agents are registered after the router is loaded.
        """
        self._tools = registry.get_all_tool_schemas()
        self._valid_functions = registry.get_all_function_names()

    @torch.inference_mode()
    def route(self, user_prompt: str) -> Tuple[str, Dict[str, Any]]:
        """
        Route a user prompt to a function.
        
        Args:
            user_prompt: What the user said/typed
            
        Returns:
            Tuple of (function_name, parameters_dict)
            
        Example:
            route("Set a timer for 5 minutes")
            → ("set_timer", {"duration": "5 minutes"})
            
            route("Hello!")
            → ("nonthinking", {"prompt": "Hello!"})
        
        @torch.inference_mode() is a decorator that tells PyTorch:
        "I'm only doing inference (prediction), not training. Don't track
        gradients, don't store computation graphs." This makes inference
        faster and uses less memory.
        """
        # Step 1: Build the prompt in the format the model expects
        # The model was trained on a specific chat template with tools
        messages = [
            {"role": "developer", "content": self._system_msg},
            {"role": "user", "content": user_prompt},
        ]

        # apply_chat_template formats the messages + tool schemas into
        # the exact text format the model was trained on
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tools=self._tools,
            add_generation_prompt=True,  # Add the "assistant:" prefix
            tokenize=False,             # Return text, not token IDs
        )

        # Step 2: Tokenize (convert text to numbers)
        # return_tensors="pt" means return PyTorch tensors
        # .to(self.model.device) moves the data to the same device as the model
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # Step 3: Generate the model's response
        # max_new_tokens=100: generate at most 100 tokens (function calls are short)
        # do_sample=False: always pick the most likely token (deterministic)
        # use_cache=True: cache internal computations for speed
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            use_cache=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # Step 4: Decode only the NEW tokens (not the input prompt)
        # The model outputs the entire sequence (input + generated).
        # We slice off the input part to get just the generated text.
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=False)

        # Step 5: Parse the response to extract function name and arguments
        return self._parse(response, user_prompt)

    def route_with_timing(self, user_prompt: str) -> Tuple[Tuple[str, Dict], float]:
        """Route with timing info. Returns ((func_name, params), elapsed_seconds)."""
        start = time.time()
        result = self.route(user_prompt)
        return result, time.time() - start

    def _parse(self, response: str, user_prompt: str) -> Tuple[str, Dict[str, Any]]:
        """
        Parse the model's raw text output into a function name and arguments.
        
        The model outputs text like:
            "call:set_timer{duration:<escape>5 minutes<escape>,label:<escape>pasta<escape>}"
        
        We need to extract:
            function_name = "set_timer"
            arguments = {"duration": "5 minutes", "label": "pasta"}
        """
        # Look for "call:function_name" pattern
        for func_name in self._valid_functions:
            if f"call:{func_name}" in response:
                args = self._extract_args(response, func_name, user_prompt)
                return func_name, args

        # No function matched — default to nonthinking (general chat)
        return "nonthinking", {"prompt": user_prompt}

    def _extract_args(self, response: str, func_name: str, user_prompt: str) -> Dict[str, Any]:
        """
        Extract function arguments from the model's response.
        
        The model uses a custom format for arguments:
            {key:<escape>value<escape>,key2:<escape>value2<escape>}
        
        The <escape> tags wrap string values. Numbers and booleans
        appear without escape tags.
        """
        # Passthrough functions just need the original prompt
        if func_name in ("thinking", "nonthinking"):
            return {"prompt": user_prompt}

        # get_system_info takes no arguments
        if func_name == "get_system_info":
            return {}

        # Try to find the arguments block: call:func_name{...}
        pattern = rf"call:{func_name}\{{([^}}]+)\}}"
        match = re.search(pattern, response)

        if match:
            args = {}
            # Parse key:value pairs
            # Handles both <escape>string values<escape> and raw numbers
            arg_pattern = r'(\w+):(?:<escape>([^<]*)<escape>|([^,]+))'
            for m in re.finditer(arg_pattern, match.group(1)):
                key = m.group(1)
                # group(2) = escaped string value, group(3) = raw value
                value = m.group(2) if m.group(2) is not None else m.group(3)

                # Convert types: "42" → 42, "true" → True
                if value.isdigit():
                    args[key] = int(value)
                elif value.lower() in ("true", "false"):
                    args[key] = value.lower() == "true"
                else:
                    args[key] = value

            if args:
                return args

        # Fallback: couldn't parse args, use sensible defaults
        fallback_map = {
            "control_light": {"action": "toggle", "device_name": user_prompt},
            "set_timer": {"duration": user_prompt},
            "set_alarm": {"time": user_prompt},
            "create_calendar_event": {"title": user_prompt},
            "add_task": {"text": user_prompt},
            "web_search": {"query": user_prompt},
        }
        return fallback_map.get(func_name, {})