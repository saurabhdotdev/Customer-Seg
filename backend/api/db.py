import os
import sqlite3
import json
import secrets
from datetime import datetime
from backend.config import BASE_DIR, IS_VERCEL

DB_PATH = "/tmp/customer_seg.db" if IS_VERCEL else os.path.join(BASE_DIR, "customer_seg.db")

def get_db_connection():
    """
    Returns a database connection to SQLite / PostgreSQL compatible SQL engine.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """
    Initializes SQL database tables if they do not exist.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. Users Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    # 2. Ingested Transactions Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            customer_id TEXT NOT NULL,
            recency_days INTEGER NOT NULL,
            frequency_orders INTEGER NOT NULL,
            monetary_spend REAL NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    # 3. Audit Logs Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS audit_logs (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            action TEXT NOT NULL,
            details TEXT,
            timestamp TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()

# Auto-initialize on import
init_db()

def log_audit_event(user_id: str, action: str, details: dict = None):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        event_id = f"evt_{secrets.token_hex(6)}"
        cursor.execute(
            "INSERT INTO audit_logs (id, user_id, action, details, timestamp) VALUES (?, ?, ?, ?, ?)",
            (event_id, user_id or "guest", action, json.dumps(details or {}), datetime.utcnow().isoformat())
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        print("Audit log failed:", exc)
