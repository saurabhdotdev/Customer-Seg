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

def load_users() -> dict:
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    if os.path.exists(BASE_USERS_FILE):
        try:
            with open(BASE_USERS_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_users(users: dict):
    os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)

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
