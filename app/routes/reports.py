"""
DermoScan – PDF report download routes.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from flask import Blueprint, flash, make_response, redirect, session, url_for

from report import generate_history_pdf, generate_single_pdf
from database import supabase
from app.config import FREE_REPORT_DAYS, UPLOAD_FOLDER
from app.decorators import login_required
from app.models import build_scan
from app.services.scan_service import filter_scans, parse_report_dates
from app.services.user_service import get_current_user
from app.utils import safe_supabase_single

bp = Blueprint("reports", __name__)


@bp.route("/report/history")
@login_required
def report_history():
    from flask import request
    user = get_current_user()
    if not user:
        session.clear()
        flash("Your session could not be restored. Please log in again.", "warning")
        return redirect(url_for("auth.login"))

    date_from_str = request.args.get("date_from", "").strip()
    date_to_str   = request.args.get("date_to",   "").strip()
    date_from, date_to, err = parse_report_dates(user, date_from_str, date_to_str)
    if err:
        flash(err, "warning")
        return redirect(url_for("scan.history"))

    scans = filter_scans(user.id, date_from, date_to)
    if not scans:
        flash("No scans found for the selected date range.", "info")
        return redirect(url_for("scan.history"))

    pdf_bytes = generate_history_pdf(user, scans, UPLOAD_FOLDER, date_from=date_from, date_to=date_to)
    response  = make_response(pdf_bytes)
    response.headers["Content-Type"]        = "application/pdf"
    response.headers["Content-Disposition"] = (
        f"attachment; filename=dermoscan_report_{user.id}_{datetime.utcnow().strftime('%Y%m%d')}.pdf"
    )
    return response


@bp.route("/report/scan/<int:scan_id>")
@login_required
def report_scan(scan_id):
    user = get_current_user()
    if not user:
        session.clear()
        flash("Your session could not be restored. Please log in again.", "warning")
        return redirect(url_for("auth.login"))

    try:
        response = (
            supabase.table("scan_history")
            .select("*").eq("id", scan_id).eq("user_id", user.id)
            .maybe_single().execute()
        )
        scan = build_scan(safe_supabase_single(response))
    except Exception:
        scan = None

    if not scan:
        flash("The requested scan could not be found.", "warning")
        return redirect(url_for("scan.history"))

    if not user.is_premium:
        cutoff  = datetime.now(timezone.utc) - timedelta(days=FREE_REPORT_DAYS)
        scan_dt = scan.created_at
        if scan_dt and scan_dt.tzinfo is None:
            scan_dt = scan_dt.replace(tzinfo=timezone.utc)
        if scan_dt and scan_dt < cutoff:
            flash(
                f"Free accounts can only download reports for scans from the last "
                f"{FREE_REPORT_DAYS} days. Upgrade to Premium for full access.",
                "warning",
            )
            return redirect(url_for("scan.scan_detail", scan_id=scan_id))

    pdf_bytes = generate_single_pdf(user, scan, UPLOAD_FOLDER)
    response  = make_response(pdf_bytes)
    response.headers["Content-Type"]        = "application/pdf"
    response.headers["Content-Disposition"] = (
        f"attachment; filename=dermoscan_scan_{scan_id}_{datetime.utcnow().strftime('%Y%m%d')}.pdf"
    )
    return response
