"""
DermoScan – Application configuration constants.
"""
from __future__ import annotations

import os

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")

ALLOWED_EXTENSIONS  = {"png", "jpg", "jpeg"}
MAX_FREE_SCANS      = 5
FREE_REPORT_DAYS    = 7
MAX_CONTENT_LENGTH  = 10 * 1024 * 1024  # 10 MB

SECRET_KEY        = os.environ.get("SECRET_KEY", "dermoscan-secret-key-change-in-prod")
SITE_URL          = os.environ.get("SITE_URL", "http://127.0.0.1:5000").strip().rstrip("/")
AUTH_CONFIRM_PATH = os.environ.get("AUTH_CONFIRM_PATH", "/auth/confirm")
