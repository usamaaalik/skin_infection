"""
DermoScan – Admin panel route.
"""
from __future__ import annotations

from flask import Blueprint, render_template

from database import list_users, supabase
from dermoscan.decorators import admin_required
from dermoscan.models import build_feedback, build_scan, build_user
from dermoscan.utils import safe_supabase_data

bp = Blueprint("admin", __name__)


@bp.route("/admin")
@admin_required
def admin_panel():
    users = [u for u in (build_user(row) for row in list_users()) if u]

    try:
        scans_resp = (
            supabase.table("scan_history")
            .select("*").order("created_at", desc=True).limit(50).execute()
        )
        scans = [s for s in (build_scan(r) for r in safe_supabase_data(scans_resp)) if s]
    except Exception:
        scans = []

    try:
        fb_resp   = supabase.table("feedback").select("*").order("created_at", desc=True).execute()
        feedbacks = [f for f in (build_feedback(r) for r in safe_supabase_data(fb_resp)) if f]
    except Exception:
        feedbacks = []

    return render_template(
        "admin.html",
        users=users, scans=scans, feedbacks=feedbacks,
        total_scans=len(scans), total_users=len(users),
    )
