from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, Response, status
from sqlalchemy import delete, select

from aura_api.deps import CurrentUser, Session, SettingsDep
from aura_api.models import GraphEdge, GraphNode, User, UserMemory
from aura_api.ratelimit import client_ip
from aura_api.schemas import LoginIn, RegisterIn, UserOut
from aura_api.security import create_token, hash_password, verify_password

router = APIRouter(prefix="/api/auth", tags=["auth"])


def _set_session_cookie(response: Response, user_id: str, settings) -> None:
    response.set_cookie(
        settings.cookie_name,
        create_token(user_id, settings),
        max_age=settings.jwt_expire_minutes * 60,
        httponly=True,
        secure=settings.cookie_secure,
        samesite="lax",
        path="/",
    )


@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
async def register(
    body: RegisterIn, request: Request, response: Response, session: Session, settings: SettingsDep
) -> User:
    limiter, key = request.app.state.register_limiter, client_ip(request)
    limiter.check(key)
    limiter.record(key)
    email = body.email.lower()
    if await session.scalar(select(User).where(User.email == email)):
        raise HTTPException(status.HTTP_409_CONFLICT, "An account with this email already exists")
    user = User(email=email, name=body.name.strip(), password_hash=hash_password(body.password))
    session.add(user)
    await session.commit()
    _set_session_cookie(response, user.id, settings)
    return user


@router.post("/login", response_model=UserOut)
async def login(
    body: LoginIn, request: Request, response: Response, session: Session, settings: SettingsDep
) -> User:
    # Only failed attempts count, per email+IP, so a password-guessing run is slowed down
    # without locking out the account owner from their own network.
    limiter, key = request.app.state.login_limiter, f"{client_ip(request)}|{body.email.lower()}"
    limiter.check(key)
    user = await session.scalar(select(User).where(User.email == body.email.lower()))
    if user is None or not verify_password(body.password, user.password_hash):
        limiter.record(key)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Incorrect email or password")
    limiter.reset(key)
    _set_session_cookie(response, user.id, settings)
    return user


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(response: Response, settings: SettingsDep) -> None:
    response.delete_cookie(settings.cookie_name, path="/")


@router.get("/me", response_model=UserOut)
async def me(user: CurrentUser) -> User:
    return user


@router.delete("/me", status_code=status.HTTP_204_NO_CONTENT)
async def delete_account(
    user: CurrentUser, response: Response, session: Session, settings: SettingsDep
) -> None:
    """Permanently delete the account and everything Aura stored about the user."""
    for model in (GraphEdge, GraphNode, UserMemory):
        await session.execute(delete(model).where(model.user_id == user.id))
    await session.delete(user)
    await session.commit()
    response.delete_cookie(settings.cookie_name, path="/")
