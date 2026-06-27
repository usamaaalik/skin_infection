"""
DermoScan – Scan retrieval and filtering helpers.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from database import supabase
from dermoscan.config import FREE_REPORT_DAYS
from dermoscan.models import build_scan
from dermoscan.utils import safe_supabase_data


def get_user_scans(user_id: Optional[str]) -> list:
    if not user_id or not supabase:
        return []
    try:
        response = (
            supabase.table("scan_history")
            .select("*").eq("user_id", user_id)
            .order("created_at", desc=True).execute()
        )
        return [s for s in (build_scan(r) for r in safe_supabase_data(response)) if s]
    except Exception:
        return []


def get_user_scan_count(user_id: Optional[str]) -> int:
    return len(get_user_scans(user_id))


def filter_scans(user_id: str, date_from, date_to) -> list:
    if not supabase:
        return []
    query = supabase.table("scan_history").select("*").eq("user_id", user_id)
    if date_from:
        query = query.gte("created_at", datetime.combine(date_from, datetime.min.time()).isoformat())
    if date_to:
        query = query.lte("created_at", datetime.combine(date_to, datetime.max.time()).isoformat())
    try:
        response = query.order("created_at", desc=True).execute()
        return [s for s in (build_scan(r) for r in safe_supabase_data(response)) if s]
    except Exception:
        return []


def parse_report_dates(user, date_from_str: str, date_to_str: str):
    today    = datetime.utcnow().date()
    free_min = today - timedelta(days=FREE_REPORT_DAYS - 1)
    try:
        date_from = datetime.strptime(date_from_str, "%Y-%m-%d").date() if date_from_str else None
        date_to   = datetime.strptime(date_to_str,   "%Y-%m-%d").date() if date_to_str   else None
    except ValueError:
        return None, None, "Invalid date format."
    if not user.is_premium:
        if date_from and date_from < free_min:
            return None, None, (
                f"Free accounts can only download reports for the last "
                f"{FREE_REPORT_DAYS} days. Upgrade to Premium for full history."
            )
        if date_from is None:
            date_from = free_min
    return date_from, date_to, None
