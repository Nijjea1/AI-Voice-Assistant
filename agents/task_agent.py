"""
Task Agent
===========
Manages a to-do list with SQLite persistence.

WHY SQLITE?
  We need tasks to survive between app restarts. Options:
    - Text file: fragile, no structure, hard to update individual items
    - JSON file: better, but concurrent writes can corrupt it
    - SQLite: built into Python, handles concurrency, structured queries
  
  SQLite stores data in a single file (data/tasks.db). Python's sqlite3
  module is built-in — no pip install needed. The database is created
  automatically on first use.

HOW SQLITE WORKS (simplified):
  It's a tiny database engine inside a single file. You talk to it
  using SQL (Structured Query Language):
  
    INSERT INTO tasks (text) VALUES ('Buy groceries')   — add a row
    SELECT * FROM tasks                                  — get all rows
    UPDATE tasks SET completed = 1 WHERE id = '...'      — update a row
    DELETE FROM tasks WHERE id = '...'                   — remove a row
  
  Python's sqlite3 module handles all the file I/O. You just write SQL.

FUZZY MATCHING:
  When the user says "mark groceries as done", we need to find the task
  that matches "groceries". We can't require an exact match because the
  task might be "Buy groceries from Walmart". So we check if the user's
  text appears INSIDE any task text (case-insensitive substring match).
"""

import os
import sqlite3
import uuid
from typing import List, Dict, Any, Optional

from agents.base import BaseAgent, FunctionDef, AgentResult
from config import GRAY, RESET, GREEN


class TaskAgent(BaseAgent):
    """Manages a persistent to-do list using SQLite."""

    name = "tasks"
    description = "Manages a to-do list — add, complete, and list tasks"

    def __init__(self):
        super().__init__()
        self.db_path = "data/tasks.db"

    def get_functions(self) -> List[FunctionDef]:
        return [
            FunctionDef(
                name="add_task",
                description=(
                    "Add a task to the to-do list. Use when the user says: "
                    "'Add buy groceries to my list', 'Remind me to call mom', "
                    "'Add task: finish homework', 'To do: clean the house', "
                    "'I need to pick up laundry'"
                ),
                parameters={
                    "text": {
                        "type": "string",
                        "description": "The task description",
                    },
                },
                required_params=["text"],
            ),
            FunctionDef(
                name="complete_task",
                description=(
                    "Mark a task as done. Use when the user says: "
                    "'Mark groceries as done', 'I finished homework', "
                    "'Complete the laundry task', 'Check off groceries'"
                ),
                parameters={
                    "text": {
                        "type": "string",
                        "description": "The task to mark as complete (fuzzy match)",
                    },
                },
                required_params=["text"],
            ),
            FunctionDef(
                name="list_tasks",
                description=(
                    "Show all tasks on the to-do list. Use when the user says: "
                    "'What is on my to-do list?', 'Show my tasks', "
                    "'What do I need to do?', 'My tasks', 'To-do list'"
                ),
                parameters={},
                required_params=[],
            ),
        ]

    def initialize(self) -> bool:
        """Create the database and tasks table if they don't exist."""
        try:
            # Create the data/ directory if it doesn't exist
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

            # Connect and create the table
            # "with" ensures the connection is closed properly
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS tasks (
                        id TEXT PRIMARY KEY,
                        text TEXT NOT NULL,
                        completed INTEGER DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                # id: unique identifier (UUID)
                # text: the task description ("Buy groceries")
                # completed: 0 = pending, 1 = done
                # created_at: when it was added (auto-filled by SQLite)

            self._initialized = True
            return True
        except Exception as e:
            print(f"{GRAY}[Tasks] Database init failed: {e}{RESET}")
            return False

    def execute(self, func_name: str, params: Dict[str, Any]) -> AgentResult:
        if func_name == "add_task":
            return self._add_task(params)
        elif func_name == "complete_task":
            return self._complete_task(params)
        elif func_name == "list_tasks":
            return self._list_tasks()
        return AgentResult(False, f"Unknown function: {func_name}")

    # ─────────────────────────────────────────
    # Add Task
    # ─────────────────────────────────────────

    def _add_task(self, params: Dict) -> AgentResult:
        """
        Add a new task to the database.
        
        UUID is used as the ID so every task has a unique identifier
        regardless of when or where it was created. uuid4() generates
        a random UUID — no collisions in practice.
        """
        text = params.get("text", "").strip()

        if not text:
            return AgentResult(False, "No task text provided.")

        # Clean up the text — remove common prefixes the user might say
        # "add buy groceries" → "buy groceries"
        # "remind me to call mom" → "call mom"
        text = self._clean_task_text(text)

        task_id = str(uuid.uuid4())

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO tasks (id, text, completed) VALUES (?, ?, 0)",
                    (task_id, text),
                )

            print(f"{GREEN}[Tasks] ✓ Added: '{text}'{RESET}")
            return AgentResult(
                success=True,
                message=f"Added '{text}' to your to-do list.",
                data={"id": task_id, "text": text},
            )
        except Exception as e:
            return AgentResult(False, f"Failed to add task: {e}")

    def _clean_task_text(self, text: str) -> str:
        """
        Remove common prefixes from task text.
        
        Users say: "add buy groceries to my list"
        We want to store: "Buy groceries"
        
        We strip out phrases like "add", "remind me to", "to my list", etc.
        """
        lower = text.lower()

        # Prefixes to strip (order matters — longer first)
        prefixes = [
            "add task", "add a task", "add to my list",
            "add to my to do list", "add to my todo list",
            "remind me to", "i need to", "i have to",
            "todo", "to do", "task",
            "add",
        ]

        for prefix in prefixes:
            if lower.startswith(prefix):
                text = text[len(prefix):].strip()
                lower = text.lower()
                break

        # Suffixes to strip
        suffixes = [
            "to my list", "to my to do list", "to my todo list",
            "to the list",
        ]

        for suffix in suffixes:
            if lower.endswith(suffix):
                text = text[:-len(suffix)].strip()
                break

        # Capitalize first letter
        if text:
            text = text[0].upper() + text[1:]

        return text

    # ─────────────────────────────────────────
    # Complete Task
    # ─────────────────────────────────────────

    def _complete_task(self, params: Dict) -> AgentResult:
        """
        Mark a task as done using fuzzy matching.
        
        The user says "mark groceries as done" — we need to find the
        task whose text contains "groceries". We do a case-insensitive
        substring search across all pending tasks.
        """
        search_text = params.get("text", "").strip().lower()

        if not search_text:
            return AgentResult(False, "No task specified to complete.")

        # Clean the search text — remove action phrases
        for phrase in ["mark", "complete", "finish", "done", "check off",
                       "check", "as done", "as complete", "as finished", "task"]:
            search_text = search_text.replace(phrase, "")
        search_text = search_text.strip()

        if not search_text:
            return AgentResult(False, "Could not determine which task to complete.")

        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get all pending (not completed) tasks
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT id, text FROM tasks WHERE completed = 0"
                )
                pending = cursor.fetchall()

            if not pending:
                return AgentResult(False, "You have no pending tasks.")

            # Fuzzy match: find tasks where the user's text appears
            # inside the task text (case-insensitive)
            matches = []
            for task in pending:
                if search_text in task["text"].lower():
                    matches.append(dict(task))

            if not matches:
                return AgentResult(
                    False,
                    f"No pending task matching '{search_text}'. "
                    f"Your pending tasks are: {', '.join(t['text'] for t in pending)}"
                )

            # Complete the first match
            matched_task = matches[0]
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "UPDATE tasks SET completed = 1 WHERE id = ?",
                    (matched_task["id"],),
                )

            print(f"{GREEN}[Tasks] ✓ Completed: '{matched_task['text']}'{RESET}")
            return AgentResult(
                success=True,
                message=f"Marked '{matched_task['text']}' as done.",
                data=matched_task,
            )

        except Exception as e:
            return AgentResult(False, f"Failed to complete task: {e}")

    # ─────────────────────────────────────────
    # List Tasks
    # ─────────────────────────────────────────

    def _list_tasks(self) -> AgentResult:
        """
        Get all tasks, grouped by status.
        
        Returns pending tasks first, then completed tasks.
        The Dispatcher will send this to Ollama to phrase naturally.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT text, completed FROM tasks ORDER BY created_at ASC"
                )
                tasks = [dict(row) for row in cursor.fetchall()]

            if not tasks:
                return AgentResult(
                    success=True,
                    message="Your to-do list is empty. Nothing to do right now.",
                )

            pending = [t for t in tasks if not t["completed"]]
            completed = [t for t in tasks if t["completed"]]

            # Build a message for the LLM to rephrase
            lines = []
            if pending:
                lines.append(f"Pending tasks ({len(pending)}):")
                for t in pending:
                    lines.append(f"  - {t['text']}")

            if completed:
                lines.append(f"Completed tasks ({len(completed)}):")
                for t in completed:
                    lines.append(f"  - {t['text']}")

            return AgentResult(
                success=True,
                message="\n".join(lines),
                data={"pending": pending, "completed": completed},
            )

        except Exception as e:
            return AgentResult(False, f"Failed to load tasks: {e}")

    # ─────────────────────────────────────────
    # System Info
    # ─────────────────────────────────────────

    def get_system_info(self) -> Optional[Dict]:
        """
        Report pending task count and list for the dashboard.
        Called when user asks "what's my status?"
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT text FROM tasks WHERE completed = 0 ORDER BY created_at ASC"
                )
                pending = [dict(row)["text"] for row in cursor.fetchall()]

            if not pending:
                return None

            return {
                "pending_tasks": pending,
                "task_count": len(pending),
            }
        except Exception:
            return None