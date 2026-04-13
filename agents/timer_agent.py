"""
Timer Agent
============
Manages countdown timers and alarms.

TIMERS vs ALARMS:
  Timer: counts DOWN from a duration. "Set a timer for 5 minutes."
         Stored as: start_time + duration_seconds. We compute remaining
         time by subtracting elapsed time from the duration.
         
  Alarm: fires at a specific TIME. "Wake me up at 7am."
         Stored as: target_time. We check if current time has passed it.

HOW TIMERS WORK INTERNALLY:
  We don't use sleep() — that would block the program. Instead:
  
  1. When user says "set timer for 5 min", we record:
     - start_time = now (e.g. 8:00:00 PM)
     - duration = 300 seconds
     - label = "Timer"
  
  2. A background thread checks every second:
     - elapsed = now - start_time
     - remaining = duration - elapsed
     - If remaining <= 0, the timer has expired → announce it
  
  3. When user asks "how much time is left?":
     - We compute remaining on the fly from start_time and duration

DURATION PARSING:
  Users say durations in many ways. We need to handle:
    "5 minutes"          → 300s
    "1 hour"             → 3600s
    "30 seconds"         → 30s
    "1 hour 30 minutes"  → 5400s
    "10 min"             → 600s
    "2h 15m"             → 8100s
    "90 seconds"         → 90s
  
  We use regex to find all number+unit pairs in the string and sum them.

TIME PARSING (for alarms):
  Users say times in many ways:
    "7am"       → 07:00
    "7:30 pm"   → 19:30
    "14:30"     → 14:30
    "3pm"       → 15:00
  
  We normalize everything to 24-hour format.
"""

import re
import time
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from agents.base import BaseAgent, FunctionDef, AgentResult
from config import GRAY, RESET, CYAN, GREEN, YELLOW


class TimerAgent(BaseAgent):
    """Manages countdown timers and alarms."""

    name = "timer"
    description = "Sets countdown timers and alarms"

    def __init__(self):
        super().__init__()
        # Active timers: {label: {"start": float, "duration": int, "label": str}}
        self._timers = {}
        # Active alarms: {label: {"target_time": str, "label": str}}
        self._alarms = {}
        # Lock for thread-safe access to timers/alarms
        self._lock = threading.Lock()
        # Background monitor thread
        self._monitor_thread = None
        self._monitoring = False

    def get_functions(self) -> List[FunctionDef]:
        return [
            FunctionDef(
                name="set_timer",
                description=(
                    "Set a countdown timer. Use when the user says: "
                    "'Set a timer for 5 minutes', 'Timer 10 min', "
                    "'Start a 30 second timer', 'Count down from 2 minutes'"
                ),
                parameters={
                    "duration": {
                        "type": "string",
                        "description": "Duration like '5 minutes', '1 hour', '30 seconds'",
                    },
                    "label": {
                        "type": "string",
                        "description": "Optional label like 'pasta' or 'laundry'",
                    },
                },
                required_params=["duration"],
            ),
            FunctionDef(
                name="set_alarm",
                description=(
                    "Set an alarm for a specific time. Use when the user says: "
                    "'Set an alarm for 7am', 'Wake me up at 6:30', "
                    "'Alarm at 3pm', 'Remind me at 14:30'"
                ),
                parameters={
                    "time": {
                        "type": "string",
                        "description": "Time like '7am', '14:30', '3:30 pm'",
                    },
                    "label": {
                        "type": "string",
                        "description": "Optional label like 'meeting' or 'workout'",
                    },
                },
                required_params=["time"],
            ),
        ]

    def initialize(self) -> bool:
        """Start the background monitor that checks for expired timers."""
        self._initialized = True
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
        )
        self._monitor_thread.start()
        return True

    def execute(self, func_name: str, params: Dict[str, Any]) -> AgentResult:
        if func_name == "set_timer":
            return self._set_timer(params)
        elif func_name == "set_alarm":
            return self._set_alarm(params)
        return AgentResult(False, f"Unknown function: {func_name}")

    # ─────────────────────────────────────────
    # Set Timer
    # ─────────────────────────────────────────

    def _set_timer(self, params: Dict) -> AgentResult:
        """
        Parse the duration and start a countdown timer.
        
        The timer is stored in memory with:
          - start: timestamp when it was created
          - duration: total seconds to count down
          - label: user-provided name (or "Timer")
        
        The background monitor thread will detect when it expires.
        """
        duration_str = params.get("duration", "")
        label = params.get("label", "").strip()

        # Parse the duration string into seconds
        seconds = self._parse_duration(duration_str)

        if seconds <= 0:
            return AgentResult(
                False,
                f"Could not understand the duration '{duration_str}'. "
                f"Try something like '5 minutes' or '1 hour 30 minutes'."
            )

        # Generate a label if none provided
        if not label:
            label = f"Timer {len(self._timers) + 1}"

        # Store the timer
        with self._lock:
            self._timers[label] = {
                "start": time.time(),
                "duration": seconds,
                "label": label,
            }

        # Format a human-readable duration for the response
        human_duration = self._format_duration(seconds)

        print(f"{GREEN}[Timer] ✓ Set '{label}' for {human_duration}{RESET}")

        return AgentResult(
            success=True,
            message=f"Timer '{label}' set for {human_duration}.",
            data={"seconds": seconds, "label": label},
        )

    # ─────────────────────────────────────────
    # Set Alarm
    # ─────────────────────────────────────────

    def _set_alarm(self, params: Dict) -> AgentResult:
        """
        Parse the time and set an alarm.
        
        The alarm fires when the current time reaches the target time.
        If the target time has already passed today, we set it for tomorrow.
        """
        time_str = params.get("time", "")
        label = params.get("label", "").strip()

        # Parse the time string into hours and minutes
        target = self._parse_time(time_str)

        if target is None:
            return AgentResult(
                False,
                f"Could not understand the time '{time_str}'. "
                f"Try something like '7am', '3:30 pm', or '14:30'."
            )

        hours, minutes = target

        # Build the target datetime
        now = datetime.now()
        target_dt = now.replace(hour=hours, minute=minutes, second=0, microsecond=0)

        # If the time already passed today, set for tomorrow
        if target_dt <= now:
            target_dt += timedelta(days=1)

        if not label:
            label = f"Alarm {len(self._alarms) + 1}"

        # Store the alarm
        with self._lock:
            self._alarms[label] = {
                "target_time": target_dt,
                "label": label,
            }

        # Format for display
        display_time = target_dt.strftime("%I:%M %p")
        day_str = "today" if target_dt.date() == now.date() else "tomorrow"

        print(f"{GREEN}[Timer] ✓ Alarm '{label}' set for {display_time} {day_str}{RESET}")

        return AgentResult(
            success=True,
            message=f"Alarm '{label}' set for {display_time} {day_str}.",
            data={"time": display_time, "label": label},
        )

    # ─────────────────────────────────────────
    # Duration Parsing
    # ─────────────────────────────────────────

    def _parse_duration(self, text: str) -> int:
        """
        Parse a natural language duration into seconds.
        
        Strategy: find all number+unit pairs using regex and sum them.
        
        Examples:
          "5 minutes"              → 300
          "1 hour 30 minutes"      → 5400
          "2h 15m"                 → 8100
          "90 seconds"             → 90
          "10 min"                 → 600
          "1 hour and 30 minutes"  → 5400 (the "and" is ignored)
        
        HOW THE REGEX WORKS:
          (\d+)\s*(hours?|hrs?|h|minutes?|mins?|m|seconds?|secs?|s)
          
          (\d+)      — capture one or more digits (the number)
          \s*        — optional whitespace between number and unit
          (hours?|...) — capture the unit (with optional 's' for plurals)
        """
        text = text.lower().strip()
        total_seconds = 0

        # Define unit multipliers (how many seconds each unit is worth)
        unit_map = {
            "hour": 3600, "hours": 3600, "hr": 3600, "hrs": 3600, "h": 3600,
            "minute": 60, "minutes": 60, "min": 60, "mins": 60, "m": 60,
            "second": 1, "seconds": 1, "sec": 1, "secs": 1, "s": 1,
        }

        # Find all number+unit pairs
        # The regex captures: (number)(optional space)(unit)
        pattern = r"(\d+)\s*(hours?|hrs?|h|minutes?|mins?|m|seconds?|secs?|s)\b"
        matches = re.findall(pattern, text)

        if matches:
            for number_str, unit in matches:
                number = int(number_str)
                multiplier = unit_map.get(unit, 0)
                total_seconds += number * multiplier
        else:
            # Fallback: if just a bare number, assume minutes
            # "timer 5" → 5 minutes
            bare_number = re.search(r"(\d+)", text)
            if bare_number:
                total_seconds = int(bare_number.group(1)) * 60

        return total_seconds

    # ─────────────────────────────────────────
    # Time Parsing
    # ─────────────────────────────────────────

    def _parse_time(self, text: str) -> Optional[tuple]:
        """
        Parse a time string into (hours, minutes) in 24-hour format.
        
        Examples:
          "7am"     → (7, 0)
          "7:30pm"  → (19, 30)
          "14:30"   → (14, 30)
          "3 pm"    → (15, 0)
          "12:00am" → (0, 0)    (midnight)
          "12:00pm" → (12, 0)   (noon)
        
        Returns None if parsing fails.
        """
        text = text.lower().strip().replace(".", "")

        # Pattern 1: "7:30 pm" or "7:30pm" or "7:30 am"
        match = re.match(r"(\d{1,2}):(\d{2})\s*(am|pm)?", text)
        if match:
            hours = int(match.group(1))
            minutes = int(match.group(2))
            period = match.group(3)

            if period:
                hours = self._convert_to_24h(hours, period)
            return (hours, minutes)

        # Pattern 2: "7am" or "7 pm" (no minutes)
        match = re.match(r"(\d{1,2})\s*(am|pm)", text)
        if match:
            hours = int(match.group(1))
            period = match.group(2)
            hours = self._convert_to_24h(hours, period)
            return (hours, 0)

        # Pattern 3: "14:30" (24-hour, no am/pm)
        match = re.match(r"(\d{1,2}):(\d{2})$", text)
        if match:
            hours = int(match.group(1))
            minutes = int(match.group(2))
            if 0 <= hours <= 23 and 0 <= minutes <= 59:
                return (hours, minutes)

        return None

    def _convert_to_24h(self, hours: int, period: str) -> int:
        """
        Convert 12-hour time to 24-hour.
        
        12am = 0 (midnight), 12pm = 12 (noon)
        1am = 1, 1pm = 13
        11am = 11, 11pm = 23
        """
        if period == "am":
            return 0 if hours == 12 else hours
        else:  # pm
            return hours if hours == 12 else hours + 12

    # ─────────────────────────────────────────
    # Duration Formatting
    # ─────────────────────────────────────────

    def _format_duration(self, seconds: int) -> str:
        """
        Convert seconds to human-readable duration.
        
        300   → "5 minutes"
        3661  → "1 hour 1 minute 1 second"
        90    → "1 minute 30 seconds"
        """
        parts = []
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60

        if hours > 0:
            parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
        if minutes > 0:
            parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
        if secs > 0 and hours == 0:  # Only show seconds if under 1 hour
            parts.append(f"{secs} second{'s' if secs != 1 else ''}")

        return " ".join(parts) if parts else "0 seconds"

    # ─────────────────────────────────────────
    # Background Monitor
    # ─────────────────────────────────────────

    def _monitor_loop(self):
        """
        Background thread that checks for expired timers and alarms.
        
        Runs every second. When a timer or alarm expires:
          1. Prints a notification to the console
          2. Queues a TTS announcement so Jarvis speaks it
          3. Removes it from the active list
        
        WHY NOT USE sleep(duration)?
          If we did sleep(300) for a 5-minute timer, we couldn't:
          - Check multiple timers at once
          - Cancel timers
          - Report remaining time
          - Handle alarms (which are time-based, not duration-based)
        
        Polling every second uses negligible CPU and handles all cases.
        """
        while self._monitoring:
            time.sleep(1)

            with self._lock:
                # Check timers
                expired_timers = []
                for label, timer in self._timers.items():
                    elapsed = time.time() - timer["start"]
                    if elapsed >= timer["duration"]:
                        expired_timers.append(label)

                for label in expired_timers:
                    print(f"\n{GREEN}[Timer] ✓ '{label}' has finished!{RESET}")
                    self._announce(f"Your timer '{label}' has finished, sir.")
                    del self._timers[label]

                # Check alarms
                expired_alarms = []
                now = datetime.now()
                for label, alarm in self._alarms.items():
                    if now >= alarm["target_time"]:
                        expired_alarms.append(label)

                for label in expired_alarms:
                    print(f"\n{GREEN}[Timer] ✓ Alarm '{label}' is ringing!{RESET}")
                    self._announce(f"Your alarm '{label}' is going off, sir. Time to get up.")
                    del self._alarms[label]

    def _announce(self, message: str):
        """
        Speak an announcement via TTS.
        
        We import tts_engine here (not at the top) to avoid circular
        imports — the TTS module doesn't depend on agents, and agents
        shouldn't depend on TTS at import time. Importing inside the
        function means it only happens when actually needed.
        """
        try:
            from core.tts import tts_engine
            if tts_engine.enabled:
                tts_engine.queue_sentence(message)
        except Exception:
            pass

    # ─────────────────────────────────────────
    # System Info
    # ─────────────────────────────────────────

    def get_system_info(self) -> Optional[Dict]:
        """
        Report active timers and alarms for the system info aggregator.
        
        Called when user asks "what's my status?" or "how much time is left?"
        """
        with self._lock:
            info = {}

            if self._timers:
                timer_list = []
                for label, timer in self._timers.items():
                    elapsed = time.time() - timer["start"]
                    remaining = max(0, int(timer["duration"] - elapsed))
                    timer_list.append({
                        "label": label,
                        "remaining": self._format_duration(remaining),
                    })
                info["active_timers"] = timer_list

            if self._alarms:
                alarm_list = []
                for label, alarm in self._alarms.items():
                    target = alarm["target_time"].strftime("%I:%M %p")
                    alarm_list.append({
                        "label": label,
                        "time": target,
                    })
                info["active_alarms"] = alarm_list

            return info if info else None

    def shutdown(self):
        """Stop the monitor thread."""
        self._monitoring = False