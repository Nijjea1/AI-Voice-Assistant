"""
Calendar Agent
===============
Manages calendar events with SQLite persistence.

DATE PARSING:
  Users say dates in many ways. We need to handle:
    "today"            → today's date
    "tomorrow"         → today + 1 day
    "monday"           → next Monday
    "next friday"      → the Friday after this one
    "march 15"         → March 15 of this year (or next year if passed)
    "2026-04-20"       → exact ISO date
    "day after tomorrow" → today + 2 days

  Strategy: check relative words first (today/tomorrow/day names),
  then try common date formats, then fall back to today.

TIME PARSING:
  Same as the timer agent — "3pm", "14:30", "3:30 pm"
  Plus some fuzzy ones:
    "morning"    → 09:00
    "afternoon"  → 14:00
    "evening"    → 18:00
    "noon"       → 12:00
    "midnight"   → 00:00

EVENT DURATION:
  Default is 1 hour. If the user specifies a duration
  ("meeting for 30 minutes"), we use that instead.

SQLITE SCHEMA:
  events table:
    id         — UUID primary key
    title      — "Team meeting" 
    start_time — "2026-04-18 15:00:00"
    end_time   — "2026-04-18 16:00:00"
    category   — "WORK", "PERSONAL", etc.
    description — optional notes
"""

import os
import re
import sqlite3
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from agents.base import BaseAgent, FunctionDef, AgentResult
from config import GRAY, RESET, GREEN, YELLOW


# Day name to weekday number (Monday=0, Sunday=6)
DAY_NAMES = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
    # Short versions
    "mon": 0, "tue": 1, "wed": 2, "thu": 3,
    "fri": 4, "sat": 5, "sun": 6,
}

# Month name to number
MONTH_NAMES = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "jun": 6, "jul": 7, "aug": 8, "sep": 9,
    "oct": 10, "nov": 11, "dec": 12,
}

# Fuzzy time words
FUZZY_TIMES = {
    "morning": (9, 0),
    "noon": (12, 0),
    "afternoon": (14, 0),
    "evening": (18, 0),
    "night": (20, 0),
    "midnight": (0, 0),
}


class CalendarAgent(BaseAgent):
    """Manages calendar events with SQLite persistence."""

    name = "calendar"
    description = "Manages calendar events — create, read, and delete"

    def __init__(self):
        super().__init__()
        self.db_path = "data/calendar.db"

    def get_functions(self) -> List[FunctionDef]:
        return [
            FunctionDef(
                name="create_event",
                description=(
                    "Create a calendar event. Use when the user says: "
                    "'Schedule a meeting tomorrow at 3pm', "
                    "'Add dentist appointment on Friday', "
                    "'Create event: team standup Monday 10am', "
                    "'I have a meeting next Wednesday at 2'"
                ),
                parameters={
                    "title": {
                        "type": "string",
                        "description": "Event title like 'Team meeting'",
                    },
                    "date": {
                        "type": "string",
                        "description": "Date like 'tomorrow', 'Monday', 'March 15'",
                    },
                    "time": {
                        "type": "string",
                        "description": "Time like '3pm', '14:30', 'morning'",
                    },
                    "duration": {
                        "type": "integer",
                        "description": "Duration in minutes (default 60)",
                    },
                },
                required_params=["title"],
            ),
            FunctionDef(
                name="read_calendar",
                description=(
                    "Read calendar events for a date. Use when the user says: "
                    "'What is on my schedule today?', 'Any events tomorrow?', "
                    "'What do I have on Monday?', 'My calendar', "
                    "'Am I free on Friday?', 'What is coming up?'"
                ),
                parameters={
                    "date": {
                        "type": "string",
                        "description": "Date like 'today', 'tomorrow', 'Monday'",
                    },
                },
                required_params=[],
            ),
            FunctionDef(
                name="delete_event",
                description=(
                    "Delete a calendar event. Use when the user says: "
                    "'Cancel the meeting tomorrow', 'Remove dentist appointment', "
                    "'Delete the standup on Monday'"
                ),
                parameters={
                    "title": {
                        "type": "string",
                        "description": "Event title to search for (fuzzy match)",
                    },
                },
                required_params=["title"],
            ),
        ]

    def initialize(self) -> bool:
        """Create the database and events table."""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS events (
                        id TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        start_time TIMESTAMP NOT NULL,
                        end_time TIMESTAMP NOT NULL,
                        category TEXT DEFAULT 'GENERAL',
                        description TEXT DEFAULT ''
                    )
                """)

            self._initialized = True
            return True
        except Exception as e:
            print(f"{GRAY}[Calendar] Database init failed: {e}{RESET}")
            return False

    def execute(self, func_name: str, params: Dict[str, Any]) -> AgentResult:
        if func_name == "create_event":
            return self._create_event(params)
        elif func_name == "read_calendar":
            return self._read_calendar(params)
        elif func_name == "delete_event":
            return self._delete_event(params)
        return AgentResult(False, f"Unknown function: {func_name}")

    # ─────────────────────────────────────────
    # Create Event
    # ─────────────────────────────────────────

    def _create_event(self, params: Dict) -> AgentResult:
        """
        Parse the user's input and create a calendar event.
        
        The router/keyword system passes the raw user text as params.
        We need to extract: title, date, time, and duration from
        natural language. This is imperfect but handles common cases.
        """
        title = params.get("title", "").strip()
        date_str = params.get("date", "")
        time_str = params.get("time", "")
        duration = params.get("duration", 60)

        if not title:
            return AgentResult(False, "No event title provided.")

        # Clean up the title — remove scheduling phrases
        title = self._clean_title(title)

        if not title:
            return AgentResult(False, "Could not determine the event title.")

        # Parse the date
        event_date = self._parse_date(date_str if date_str else title)

        # Parse the time
        hours, minutes = self._parse_time_from_text(time_str if time_str else title)

        # Build start and end datetimes
        start_dt = event_date.replace(hour=hours, minute=minutes, second=0, microsecond=0)

        # Duration: try to parse from params, default to 60 minutes
        if isinstance(duration, str):
            try:
                duration = int(duration)
            except ValueError:
                duration = 60
        end_dt = start_dt + timedelta(minutes=duration)

        # Save to database
        event_id = str(uuid.uuid4())
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO events (id, title, start_time, end_time) VALUES (?, ?, ?, ?)",
                    (event_id, title, start_dt.strftime("%Y-%m-%d %H:%M:%S"),
                     end_dt.strftime("%Y-%m-%d %H:%M:%S")),
                )

            # Format for response
            display_date = start_dt.strftime("%A, %B %d")
            display_time = start_dt.strftime("%I:%M %p").lstrip("0")

            print(f"{GREEN}[Calendar] ✓ Created: '{title}' on {display_date} at {display_time}{RESET}")

            return AgentResult(
                success=True,
                message=f"Event '{title}' created for {display_date} at {display_time}.",
                data={"id": event_id, "title": title, "date": display_date, "time": display_time},
            )
        except Exception as e:
            return AgentResult(False, f"Failed to create event: {e}")

    def _clean_title(self, text: str) -> str:
        """
        Extract the event title from the user's natural language.
        
        "Schedule a meeting tomorrow at 3pm" → "Meeting"
        "Add dentist appointment on Friday" → "Dentist appointment"
        
        We strip scheduling phrases, dates, and times, then capitalize.
        """
        lower = text.lower()

        # Remove scheduling prefixes
        prefixes = [
            "schedule a", "schedule", "create event", "create an event",
            "add event", "add an event", "add a", "add",
            "i have a", "i have an", "i have", "plan a", "plan",
            "book a", "book", "set up a", "set up",
        ]
        for prefix in prefixes:
            if lower.startswith(prefix):
                text = text[len(prefix):].strip()
                lower = text.lower()
                break

        # Remove date/time phrases from the end
        # We strip common patterns like "tomorrow at 3pm", "on Monday", etc.
        patterns_to_strip = [
            r"\s+(?:today|tomorrow|next\s+\w+)\s*(?:at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?)?$",
            r"\s+on\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s*(?:at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?)?$",
            r"\s+at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?$",
            r"\s+(?:in the\s+)?(?:morning|afternoon|evening|night)$",
            r"\s+for\s+\d+\s*(?:minutes?|mins?|hours?|hrs?)$",
        ]
        for pattern in patterns_to_strip:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()

        # Capitalize
        if text:
            text = text[0].upper() + text[1:]

        return text

    # ─────────────────────────────────────────
    # Read Calendar
    # ─────────────────────────────────────────

    def _read_calendar(self, params: Dict) -> AgentResult:
        """Read events for a specific date."""
        date_str = params.get("date", "today")
        target_date = self._parse_date(date_str)
        date_formatted = target_date.strftime("%Y-%m-%d")
        display_date = target_date.strftime("%A, %B %d")

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT title, start_time, end_time FROM events "
                    "WHERE start_time BETWEEN ? AND ? ORDER BY start_time ASC",
                    (f"{date_formatted} 00:00:00", f"{date_formatted} 23:59:59"),
                )
                events = [dict(row) for row in cursor.fetchall()]

            if not events:
                return AgentResult(
                    success=True,
                    message=f"You have no events scheduled for {display_date}. Your day is free.",
                )

            # Format events for the LLM
            lines = [f"Events for {display_date}:"]
            for event in events:
                start = datetime.strptime(event["start_time"], "%Y-%m-%d %H:%M:%S")
                end = datetime.strptime(event["end_time"], "%Y-%m-%d %H:%M:%S")
                time_range = f"{start.strftime('%I:%M %p').lstrip('0')} - {end.strftime('%I:%M %p').lstrip('0')}"
                lines.append(f"  {event['title']} ({time_range})")

            return AgentResult(
                success=True,
                message="\n".join(lines),
                data=events,
            )
        except Exception as e:
            return AgentResult(False, f"Failed to read calendar: {e}")

    # ─────────────────────────────────────────
    # Delete Event
    # ─────────────────────────────────────────

    def _delete_event(self, params: Dict) -> AgentResult:
        """Delete an event using fuzzy title matching."""
        search_text = params.get("title", "").strip().lower()

        if not search_text:
            return AgentResult(False, "No event specified to delete.")

        # Clean search text — remove action phrases
        for phrase in ["cancel", "delete", "remove", "drop",
                       "the", "my", "event", "appointment"]:
            search_text = search_text.replace(phrase, "")
        search_text = search_text.strip()

        if not search_text:
            return AgentResult(False, "Could not determine which event to delete.")

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                # Get all future events
                cursor = conn.execute(
                    "SELECT id, title, start_time FROM events "
                    "WHERE start_time >= ? ORDER BY start_time ASC",
                    (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),),
                )
                events = [dict(row) for row in cursor.fetchall()]

            if not events:
                return AgentResult(False, "You have no upcoming events to delete.")

            # Fuzzy match
            matches = [e for e in events if search_text in e["title"].lower()]

            if not matches:
                titles = ", ".join(e["title"] for e in events)
                return AgentResult(
                    False,
                    f"No event matching '{search_text}'. Your upcoming events: {titles}"
                )

            # Delete first match
            matched = matches[0]
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM events WHERE id = ?", (matched["id"],))

            print(f"{GREEN}[Calendar] ✓ Deleted: '{matched['title']}'{RESET}")
            return AgentResult(
                success=True,
                message=f"Deleted '{matched['title']}' from your calendar.",
                data=matched,
            )
        except Exception as e:
            return AgentResult(False, f"Failed to delete event: {e}")

    # ─────────────────────────────────────────
    # Date Parsing
    # ─────────────────────────────────────────

    def _parse_date(self, text: str) -> datetime:
        """
        Parse a natural language date string into a datetime.
        
        Checks in order:
          1. Relative: "today", "tomorrow", "day after tomorrow"
          2. Day names: "monday", "next friday"
          3. Month + day: "march 15", "jan 3"
          4. ISO format: "2026-04-20"
          5. Fallback: today
        """
        lower = text.lower().strip()
        today = datetime.now()

        # ── Relative dates ──
        if "today" in lower:
            return today
        if "day after tomorrow" in lower:
            return today + timedelta(days=2)
        if "tomorrow" in lower:
            return today + timedelta(days=1)
        if "yesterday" in lower:
            return today - timedelta(days=1)

        # ── Day names: "monday", "next friday" ──
        is_next = "next" in lower
        for day_name, day_num in DAY_NAMES.items():
            if day_name in lower:
                current_day = today.weekday()
                days_ahead = day_num - current_day
                if days_ahead <= 0:
                    days_ahead += 7
                if is_next and days_ahead <= 7:
                    days_ahead += 7
                return today + timedelta(days=days_ahead)

        # ── Month + day: "march 15", "jan 3" ──
        for month_name, month_num in MONTH_NAMES.items():
            if month_name in lower:
                # Find the day number after the month name
                day_match = re.search(rf"{month_name}\s+(\d{{1,2}})", lower)
                if day_match:
                    day = int(day_match.group(1))
                    try:
                        target = today.replace(month=month_num, day=day)
                        # If the date already passed this year, use next year
                        if target < today:
                            target = target.replace(year=today.year + 1)
                        return target
                    except ValueError:
                        pass
                break

        # ── ISO format: "2026-04-20" ──
        iso_match = re.search(r"(\d{4})-(\d{2})-(\d{2})", lower)
        if iso_match:
            try:
                return datetime(
                    int(iso_match.group(1)),
                    int(iso_match.group(2)),
                    int(iso_match.group(3)),
                )
            except ValueError:
                pass

        # ── Fallback: today ──
        return today

    # ─────────────────────────────────────────
    # Time Parsing
    # ─────────────────────────────────────────

    def _parse_time_from_text(self, text: str) -> tuple:
        """
        Extract time from natural language text.
        
        Returns (hours, minutes) in 24-hour format.
        Falls back to 09:00 if no time found.
        """
        lower = text.lower()

        # ── Fuzzy times: "morning", "afternoon", etc. ──
        for word, (h, m) in FUZZY_TIMES.items():
            if word in lower:
                return (h, m)

        # ── "at 3pm", "at 14:30", "at 3:30 pm" ──
        # Pattern: optional "at", then time
        time_match = re.search(
            r"(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?",
            lower
        )
        if time_match:
            hours = int(time_match.group(1))
            minutes = int(time_match.group(2)) if time_match.group(2) else 0
            period = time_match.group(3)

            if period:
                if period == "am":
                    hours = 0 if hours == 12 else hours
                else:
                    hours = hours if hours == 12 else hours + 12

            # Sanity check
            if 0 <= hours <= 23 and 0 <= minutes <= 59:
                return (hours, minutes)

        # ── Fallback: 9am ──
        return (9, 0)

    # ─────────────────────────────────────────
    # System Info
    # ─────────────────────────────────────────

    def get_system_info(self) -> Optional[Dict]:
        """Report today's events for the dashboard / system info."""
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT title, start_time FROM events "
                    "WHERE start_time BETWEEN ? AND ? ORDER BY start_time ASC",
                    (f"{today} 00:00:00", f"{today} 23:59:59"),
                )
                events = [dict(row) for row in cursor.fetchall()]

            if not events:
                return None

            return {
                "today_events": [
                    {
                        "title": e["title"],
                        "time": datetime.strptime(
                            e["start_time"], "%Y-%m-%d %H:%M:%S"
                        ).strftime("%I:%M %p").lstrip("0"),
                    }
                    for e in events
                ]
            }
        except Exception:
            return None