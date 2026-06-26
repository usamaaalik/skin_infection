"""
DermoScan - Supabase client bootstrap
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "").strip()

try:
    from supabase import client as supabase_client
except ImportError:  # pragma: no cover - dependency may not be installed in some environments
    supabase_client = None


if SUPABASE_URL and SUPABASE_KEY and supabase_client:
    supabase = supabase_client.create_client(SUPABASE_URL, SUPABASE_KEY)
else:
    supabase = None


class SupabaseDB:
    """Compatibility shim for the old SQLAlchemy-style initialization."""

    def __init__(self, client: Any | None = None) -> None:
        self.client = client

    def init_app(self, app: Any) -> None:  # noqa: ARG002
        return None

    @property
    def session(self) -> Any:
        raise RuntimeError("Supabase persistence is not wired yet; use supabase.client directly.")


# Replace the old SQLAlchemy instance with a Supabase-backed wrapper.
db = SupabaseDB(supabase)


@dataclass
class User:
    id: int | None = None
    first_name: str = ""
    last_name: str = ""
    email: str = ""
    password_hash: str = ""
    is_premium: bool = False
    is_admin: bool = False
    created_at: Any = None

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}".strip()

    @property
    def initials(self) -> str:
        return f"{self.first_name[0]}{self.last_name[0]}".upper() if self.first_name and self.last_name else ""


@dataclass
class ScanHistory:
    id: int | None = None
    user_id: int | None = None
    image_filename: str = ""
    predicted_class: str = ""
    confidence: float = 0.0
    all_scores: str | None = None
    created_at: Any = None

    @property
    def confidence_pct(self) -> float:
        return round(self.confidence * 100, 1)

    @property
    def badge_color(self) -> str:
        mapping = {
            "normal": "success",
            "acne": "warning",
            "eczema": "info",
            "psoriasis": "danger",
            "ringworm": "secondary",
        }
        return mapping.get(self.predicted_class.lower(), "primary")


@dataclass
class Feedback:
    id: int | None = None
    user_id: int | None = None
    message: str = ""
    rating: int = 5
    created_at: Any = None
