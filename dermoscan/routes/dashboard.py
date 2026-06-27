"""
DermoScan – Dashboard route.
"""
from __future__ import annotations

from flask import Blueprint, flash, redirect, render_template, session, url_for

from dermoscan.config import MAX_FREE_SCANS
from dermoscan.decorators import login_required
from dermoscan.services.scan_service import get_user_scans
from dermoscan.services.user_service import get_current_user

bp = Blueprint("dashboard", __name__)


@bp.route("/dashboard")
@login_required
def dashboard():
    user = get_current_user()
    if not user:
        session.clear()
        flash("Your session could not be restored. Please log in again.", "warning")
        return redirect(url_for("auth.login"))
    recent_scans = get_user_scans(user.id)[:3]
    return render_template(
        "dashboard.html",
        user=user,
        scan_count=len(recent_scans),
        recent_scans=recent_scans,
        max_free=MAX_FREE_SCANS,
    )
