"""
DermoScan – Analysis and scan history routes.
"""
from __future__ import annotations

import base64
import os
import tempfile
import uuid
from datetime import datetime, timedelta

from flask import Blueprint, flash, redirect, render_template, request, session, url_for
from werkzeug.utils import secure_filename

from database import insert_scan_record, supabase
from dermoscan.config import FREE_REPORT_DAYS, MAX_FREE_SCANS
from dermoscan.decorators import login_required
from dermoscan.models import build_scan
from dermoscan.services.predictor_service import get_predictor
from dermoscan.services.scan_service import get_user_scan_count
from dermoscan.services.user_service import get_current_user
from dermoscan.utils import allowed_file, build_image_data_url, safe_supabase_data, safe_supabase_single

bp = Blueprint("scan", __name__)


@bp.route("/analysis", methods=["GET", "POST"])
@login_required
def analysis():
    user = get_current_user()
    if not user:
        session.clear()
        flash("Your session could not be restored. Please log in again.", "warning")
        return redirect(url_for("auth.login"))

    scan_count = get_user_scan_count(user.id)

    if request.method == "POST":
        if not user.is_premium and scan_count >= MAX_FREE_SCANS:
            flash(f"Free limit of {MAX_FREE_SCANS} scans reached. Upgrade to Premium.", "warning")
            return redirect(url_for("dashboard.dashboard"))

        file = request.files.get("image")
        if not file or file.filename == "":
            flash("No file selected.", "danger")
            return render_template("analysis.html", user=user, scan_count=scan_count, max_free=MAX_FREE_SCANS)

        if not allowed_file(file.filename):
            flash("Only JPG and PNG files are accepted.", "danger")
            return render_template("analysis.html", user=user, scan_count=scan_count, max_free=MAX_FREE_SCANS)

        ext       = secure_filename(file.filename).rsplit(".", 1)[1].lower()
        filename  = f"{uuid.uuid4().hex}.{ext}"
        img_bytes = file.read()
        img_b64   = base64.b64encode(img_bytes).decode("utf-8")

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=f".{ext}")
        os.close(tmp_fd)
        try:
            with open(tmp_path, "wb") as fh:
                fh.write(img_bytes)
            result = get_predictor().predict(tmp_path)
        except Exception as exc:
            flash(f"Prediction error: {exc}", "danger")
            return render_template("analysis.html", user=user, scan_count=scan_count, max_free=MAX_FREE_SCANS)
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

        try:
            inserted   = insert_scan_record({
                "user_id": user.id, "image_filename": filename,
                "image_bytes": img_b64, "predicted_class": result["predicted_class"],
                "confidence": result["confidence"], "all_scores": result["all_scores"],
            })
            history_id = inserted.get("id") if inserted else None
        except Exception as exc:
            flash(f"Unable to save your scan history right now: {exc}", "warning")
            history_id = None

        return render_template(
            "result.html", user=user, result=result,
            image_url=build_image_data_url(img_b64, filename), history_id=history_id,
        )

    return render_template("analysis.html", user=user, scan_count=scan_count, max_free=MAX_FREE_SCANS)


@bp.route("/history")
@login_required
def history():
    user = get_current_user()
    if not user:
        session.clear()
        flash("Your session could not be restored. Please log in again.", "warning")
        return redirect(url_for("auth.login"))

    date_from_str = request.args.get("date_from", "").strip()
    date_to_str   = request.args.get("date_to",   "").strip()
    today         = datetime.utcnow().date()
    free_min_date = today - timedelta(days=FREE_REPORT_DAYS - 1)

    date_from = date_to = None
    try:
        if date_from_str:
            date_from = datetime.strptime(date_from_str, "%Y-%m-%d").date()
        if date_to_str:
            date_to = datetime.strptime(date_to_str, "%Y-%m-%d").date()
    except ValueError:
        flash("Invalid date format.", "warning")

    query = supabase.table("scan_history").select("*").eq("user_id", user.id)
    if date_from:
        query = query.gte("created_at", datetime.combine(date_from, datetime.min.time()).isoformat())
    if date_to:
        query = query.lte("created_at", datetime.combine(date_to, datetime.max.time()).isoformat())

    try:
        response = query.order("created_at", desc=True).execute()
        scans = [s for s in (build_scan(r) for r in safe_supabase_data(response)) if s]
    except Exception as exc:
        flash(f"Unable to load your scan history right now: {exc}", "warning")
        scans = []

    return render_template(
        "history.html", user=user, scans=scans,
        date_from=date_from_str, date_to=date_to_str,
        free_report_days=FREE_REPORT_DAYS,
        free_min_date=free_min_date.strftime("%Y-%m-%d"),
        today=today.strftime("%Y-%m-%d"),
    )


@bp.route("/history/<int:scan_id>")
@login_required
def scan_detail(scan_id):
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

    image_url   = None
    image_bytes = getattr(scan, "image_bytes", None)
    if image_bytes:
        image_url = build_image_data_url(image_bytes, getattr(scan, "image_filename", None))
    elif getattr(scan, "image_filename", None):
        image_url = url_for("static", filename=f"uploads/{scan.image_filename}")

    return render_template("scan_detail.html", user=user, scan=scan, image_url=image_url)
