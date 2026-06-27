"""
DermoScan – User-related service helpers.
"""
from __future__ import annotations

from typing import Optional
from types import SimpleNamespace

from flask import session

from database import get_user_by_id
from app.models import AppUser, build_user


def get_current_user() -> Optional[AppUser]:
    """Return an AppUser for the current session, or None."""
    user_id = session.get("user_id")
    if not user_id:
        return None

    profile = get_user_by_id(user_id)
    if isinstance(profile, dict):
        return build_user(profile)

    # Fallback: build a minimal user from session data
    return AppUser(
        id=user_id,
        first_name=session.get("user_name") or "",
        last_name="",
        email="",
        username="",
        is_premium=bool(session.get("is_premium", False)),
        is_admin=bool(session.get("is_admin", False)),
        created_at=None,
    )


def set_session(user: dict) -> None:
    """Populate session keys from a user dict returned by the DB layer."""
    session.clear()
    session["user_id"]   = user["id"]
    session["user_name"] = user.get("first_name") or user.get("username", "")
    session["is_admin"]  = bool(user.get("is_admin", False))
    session["is_premium"]= bool(user.get("is_premium", False))
