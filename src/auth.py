"""
Authentication and security module for Employee Burnout Prediction System.
Handles password hashing, token generation, and Role-Based Access Control (RBAC).
"""

import os
import hashlib
import secrets
import hmac
import json
import base64
import time

SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "burnout-barrier-secret-key-2026")

ROLES = {
    "HR": "HR Manager",
    "ADMIN": "Administrator",
    "MLE": "Machine Learning Engineer"
}

def hash_password(password: str) -> str:
    """Hash a password using PBKDF2 with HMAC-SHA256."""
    salt = secrets.token_hex(16)
    pwd_hash = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt.encode('utf-8'),
        100000
    ).hex()
    return f"{salt}:{pwd_hash}"

def verify_password(password: str, stored_hash: str) -> bool:
    """Verify a plain password against stored salt:hash format."""
    if not stored_hash or ":" not in stored_hash:
        return False
    try:
        salt, expected_hash = stored_hash.split(":", 1)
        calc_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        ).hex()
        return hmac.compare_digest(expected_hash, calc_hash)
    except Exception:
        return False

def generate_token(user_id: int, username: str, role: str, expires_in: int = 86400) -> str:
    """Generate a lightweight HMAC-signed access token (JWT alternative)."""
    payload = {
        "user_id": user_id,
        "username": username,
        "role": role,
        "exp": int(time.time()) + expires_in
    }
    payload_bytes = json.dumps(payload).encode('utf-8')
    payload_b64 = base64.urlsafe_b64encode(payload_bytes).decode('utf-8').rstrip('=')
    
    signature = hmac.new(
        SECRET_KEY.encode('utf-8'),
        payload_b64.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    return f"{payload_b64}.{signature}"

def decode_token(token: str) -> dict:
    """Verify and decode a token string."""
    if not token or "." not in token:
        return None
    try:
        payload_b64, signature = token.split(".", 1)
        expected_sig = hmac.new(
            SECRET_KEY.encode('utf-8'),
            payload_b64.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        if not hmac.compare_digest(signature, expected_sig):
            return None
        
        # Add padding back if needed
        padding = 4 - (len(payload_b64) % 4)
        if padding != 4:
            payload_b64 += "=" * padding
            
        payload_json = base64.urlsafe_b64decode(payload_b64.encode('utf-8')).decode('utf-8')
        payload = json.loads(payload_json)
        
        if payload.get("exp", 0) < time.time():
            return None
            
        return payload
    except Exception:
        return None
