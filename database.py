"""
database.py
-----------
Handles a simple SQLite database that logs every conversation:
who said what, what the bot replied, and how confident it was.

SQLite needs no server or installation - Python has it built in.
The database is just a single file: chatbot_logs.db, created automatically
the first time you run the app.
"""

import sqlite3
from datetime import datetime

DB_PATH = "chatbot_logs.db"


def init_db():
    """Create the logs table if it doesn't already exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS conversation_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            user_message TEXT,
            bot_response TEXT,
            matched_question TEXT,
            confidence REAL,
            timestamp TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def log_conversation(session_id, user_message, bot_response, matched_question, confidence):
    """Insert one exchange into the log table."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO conversation_logs
        (session_id, user_message, bot_response, matched_question, confidence, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            session_id,
            user_message,
            bot_response,
            matched_question,
            confidence,
            datetime.utcnow().isoformat(),
        ),
    )
    conn.commit()
    conn.close()


def get_all_logs(limit=100):
    """Fetch recent logs, most recent first - useful for a debug/admin view."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM conversation_logs ORDER BY id DESC LIMIT ?", (limit,)
    )
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows
