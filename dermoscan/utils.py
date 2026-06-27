"""
DermoScan – Shared utility helpers.
"""
from __future__ import annotations

import base64
import os
from typing import Any, Optional

from dermoscan.config import ALLOWED_EXTENSIONS


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def build_image_data_url(image_bytes: Any, filename: Optional[str] = None) -> Optional[str]:
    if not image_bytes:
        return None
    if isinstance(image_bytes, (bytes, bytearray)):
        encoded = base64.b64encode(bytes(image_bytes)).decode("utf-8")
    else:
        encoded = str(image_bytes)
    ext = (os.path.splitext(filename or "")[1].lstrip(".") or "png").lower()
    if ext == "jpg":
        ext = "jpeg"
    return f"data:image/{ext};base64,{encoded}"


def safe_supabase_data(response: Any) -> list[dict]:
    if response is None:
        return []
    if isinstance(response, dict):
        data = response.get("data")
        return data if isinstance(data, list) else []
    data = getattr(response, "data", None)
    return data if isinstance(data, list) else []


def safe_supabase_single(response: Any) -> Optional[dict]:
    if response is None:
        return None
    if isinstance(response, dict):
        data = response.get("data")
        return data if isinstance(data, dict) else None
    data = getattr(response, "data", None)
    return data if isinstance(data, dict) else None


_SYMBOLS = set("!@#$%^&*()-_=+[]{}|;:',.<>?/`~\"\\ ")


def validate_password(password: str) -> Optional[str]:
    msg = "Password must be at least 8 characters and include uppercase, lowercase, a number, and a symbol."
    if (len(password) < 8
            or not any(c.isupper() for c in password)
            or not any(c.islower() for c in password)
            or not any(c.isdigit() for c in password)
            or not any(c in _SYMBOLS for c in password)):
        return msg
    return None
