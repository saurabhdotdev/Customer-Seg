import os
import json
import hmac
import hashlib
import secrets
from datetime import datetime, timezone, timedelta
import jwt
from fastapi import Header, HTTPException, Depends
from backend.config import WORK_DIR, BASE_DIR, IS_VERCEL

JWT_SECRET = os.environ.get("JWT_SECRET", "customer-seg-jwt-secret-key-2026-high-yield")
JWT_ALGORITHM = "HS256"
TOKEN_EXPIRE_DAYS = 7

USERS_FILE = os.path.join(WORK_DIR, "users_db.json")
BASE_USERS_FILE = os.path.join(BASE_DIR, "data", "users_db.json")

def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    hashed = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000)
    return f"{salt}:{hashed.hex()}"

def verify_password(password: str, hashed_password: str) -> bool:
    try:
        salt, stored_hash = hashed_password.split(":")
        calc_hash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000)
        return hmac.compare_digest(calc_hash.hex(), stored_hash)
    except Exception:
        return False

from backend.api.db import get_db_connection, log_audit_event

def load_users() -> dict:
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, email, password_hash, created_at FROM users")
        rows = cursor.fetchall()
        conn.close()
        
        users = {}
        for r in rows:
            users[r["email"]] = {
                "id": r["id"],
                "user_id": r["id"],
                "name": r["name"],
                "email": r["email"],
                "password": r["password_hash"],
                "password_hash": r["password_hash"],
                "created_at": r["created_at"]
            }
        return users
    except Exception as exc:
        print("Failed to load users from DB:", exc)
        return {}

def save_user_to_db(user_id: str, name: str, email: str, password_hash: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    created_at = datetime.utcnow().isoformat()
    cursor.execute(
        "INSERT OR REPLACE INTO users (id, name, email, password_hash, created_at) VALUES (?, ?, ?, ?, ?)",
        (user_id, name, email, password_hash, created_at)
    )
    conn.commit()
    conn.close()
    log_audit_event(user_id, "user_registered", {"email": email, "name": name})

def save_users(users: dict):
    for email, user in users.items():
        uid = user.get("id") or user.get("user_id") or f"usr_{secrets.token_hex(6)}"
        pwd = user.get("password_hash") or user.get("password") or ""
        save_user_to_db(uid, user["name"], user["email"], pwd)

def create_jwt_token(user_id: str, email: str, name: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "email": email,
        "name": name,
        "iat": now,
        "exp": now + timedelta(days=TOKEN_EXPIRE_DAYS)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def decode_jwt_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Session expired. Please log in again.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid authentication token.")

def get_current_user_optional(authorization: str = Header(None)) -> dict | None:
    if not authorization:
        return None
    token = authorization.replace("Bearer ", "").strip()
    if not token:
        return None
    try:
        return decode_jwt_token(token)
    except HTTPException:
        return None

def get_current_user_required(authorization: str = Header(None)) -> dict:
    if not authorization:
        raise HTTPException(status_code=401, detail="Authentication required. Please sign in.")
    token = authorization.replace("Bearer ", "").strip()
    if not token:
        raise HTTPException(status_code=401, detail="Authentication token missing.")
    return decode_jwt_token(token)
