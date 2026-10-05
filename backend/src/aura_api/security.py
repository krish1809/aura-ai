"""Password hashing (bcrypt) and session tokens (JWT in an httpOnly cookie)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import bcrypt
import jwt

from aura_api.config import Settings

ALGORITHM = "HS256"


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode()[:72], bcrypt.gensalt()).decode()


def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode()[:72], hashed.encode())
    except ValueError:
        return False


def create_token(user_id: str, settings: Settings) -> str:
    now = datetime.now(UTC)
    payload = {
        "sub": user_id,
        "iat": now,
        "exp": now + timedelta(minutes=settings.jwt_expire_minutes),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=ALGORITHM)


def decode_token(token: str, settings: Settings) -> str | None:
    try:
        return jwt.decode(token, settings.jwt_secret, algorithms=[ALGORITHM])["sub"]
    except jwt.PyJWTError:
        return None
