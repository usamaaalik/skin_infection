"""
DermoScan – In-memory model objects built from Supabase row dicts.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Optional

from dermoscan.utils import build_image_data_url


class AppUser(SimpleNamespace):
    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}".strip()

    @property
    def initials(self) -> str:
        f = getattr(self, "first_name", "")
        l = getattr(self, "last_name", "")
        return f"{f[0]}{l[0]}".upper() if f and l else ""


def build_user(profile: Optional[dict], email: Optional[str] = None) -> Optional[AppUser]:
    if not isinstance(profile, dict) or not profile.get("id"):
        return None
    return AppUser(
        id=profile.get("id"),
        first_name=profile.get("first_name", ""),
        last_name=profile.get("last_name", ""),
        email=email or "",
        username=profile.get("username", ""),
        is_premium=bool(profile.get("is_premium", False)),
        is_admin=bool(profile.get("is_admin", False)),
        created_at=_parse_dt(profile.get("created_at")),
    )


def build_scan(scan_data: Optional[dict]) -> Optional[SimpleNamespace]:
    if not isinstance(scan_data, dict):
        return None
    scan = SimpleNamespace(**scan_data)
    scan.created_at     = _parse_dt(scan_data.get("created_at"), aware=True)
    scan.confidence_pct = round(float(scan.confidence or 0) * 100, 1) if scan.confidence is not None else 0.0
    scan.badge_color    = _badge_color(getattr(scan, "predicted_class", ""))
    scan.scores_dict    = _parse_scores(getattr(scan, "all_scores", None))
    scan.image_url      = _resolve_image_url(scan)
    return scan


def build_feedback(feedback_data: Optional[dict]) -> Optional[SimpleNamespace]:
    if not isinstance(feedback_data, dict):
        return None
    fb = SimpleNamespace(**feedback_data)
    fb.created_at = _parse_dt(feedback_data.get("created_at"))
    return fb


def _parse_dt(value: Optional[str], aware: bool = False) -> Optional[datetime]:
    if not isinstance(value, str):
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if aware and dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


_BADGE_MAP = {"normal": "success", "acne": "warning", "eczema": "info",
              "psoriasis": "danger", "ringworm": "secondary"}


def _badge_color(predicted_class: str) -> str:
    return _BADGE_MAP.get(str(predicted_class).lower(), "primary")


def _parse_scores(all_scores) -> dict:
    if not all_scores:
        return {}
    try:
        return json.loads(all_scores) if isinstance(all_scores, str) else all_scores
    except Exception:
        return {}


def _resolve_image_url(scan: SimpleNamespace) -> Optional[str]:
    image_bytes = getattr(scan, "image_bytes", None)
    if image_bytes and isinstance(image_bytes, str) and image_bytes.startswith("data:image"):
        return image_bytes
    if image_bytes:
        return build_image_data_url(image_bytes, getattr(scan, "image_filename", None))
    return None
