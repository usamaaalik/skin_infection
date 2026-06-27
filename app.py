"""
DermoScan - Skin Infection Detection and Classification System
Main Flask application
"""
from __future__ import annotations

import base64
import os
import tempfile
import uuid
import json
from datetime import datetime, timedelta, timezone
from functools import wraps
from types import SimpleNamespace
from typing import Optional

from flask import (
    Flask, render_template, request, redirect, url_for,
    session, flash, jsonify, make_response
)
from werkzeug.utils import secure_filename

import base64

from database import (
    db,
    supabase,
    User,
    ScanHistory,
    Feedback,
    authenticate_user,
    create_user,
    get_user_by_id,
    get_user_by_username,
    init_db,
    insert_scan_record,
    list_users,
    update_user,
)
from predictor import SkinPredictor
from report import generate_history_pdf, generate_single_pdf

# ─── App Configuration ───────────────────────────────────────────────────────

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dermoscan-secret-key-change-in-prod")
SITE_URL = os.environ.get("SITE_URL", "http://127.0.0.1:5000").strip().rstrip("/")
AUTH_CONFIRM_PATH = os.environ.get("AUTH_CONFIRM_PATH", "/auth/confirm")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MAX_FREE_SCANS = 5

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{os.path.join(BASE_DIR, 'dermoscan.db')}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB

db.init_app(app)
init_db()

# Lazy-load predictor so app starts even if model isn't trained yet
predictor: Optional[SkinPredictor] = None


def get_predictor() -> SkinPredictor:
    global predictor
    if predictor is None:
        predictor = SkinPredictor()
    return predictor


# ─── Helpers ─────────────────────────────────────────────────────────────────

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _build_image_data_url(image_bytes, filename: Optional[str] = None) -> Optional[str]:
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


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to continue.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        user = _get_current_user()
        if not user or not user.is_admin:
            flash("Admin access required.", "danger")
            return redirect(url_for("dashboard"))
        return f(*args, **kwargs)
    return decorated


def _get_user_payload(payload) -> tuple[Optional[object], Optional[object]]:
    if payload is None:
        return None, None
    if isinstance(payload, dict):
        return payload.get("user"), payload.get("session")
    return getattr(payload, "user", None), getattr(payload, "session", None)


def _supabase_redirect_url(path: str) -> str:
    configured = os.environ.get("SITE_URL", "").strip().rstrip("/")
    if configured:
        return f"{configured}{path}"
    return f"{request.host_url.rstrip('/')}{path}"


class AppUser(SimpleNamespace):
    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}".strip()

    @property
    def initials(self) -> str:
        return f"{self.first_name[0]}{self.last_name[0]}".upper() if self.first_name and self.last_name else ""


def _safe_supabase_data(response) -> list[dict]:
    if response is None:
        return []
    if isinstance(response, dict):
        data = response.get("data")
        return data if isinstance(data, list) else []
    data = getattr(response, "data", None)
    return data if isinstance(data, list) else []


def _safe_supabase_single(response) -> Optional[dict]:
    if response is None:
        return None
    if isinstance(response, dict):
        data = response.get("data")
        return data if isinstance(data, dict) else None
    data = getattr(response, "data", None)
    return data if isinstance(data, dict) else None


def _build_user(profile: Optional[dict], email: Optional[str] = None) -> Optional[AppUser]:
    if not isinstance(profile, dict) or not profile.get("id"):
        return None
    created_at = profile.get("created_at")
    if isinstance(created_at, str):
        try:
            created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        except ValueError:
            created_at = None
    return AppUser(
        id=profile.get("id"),
        first_name=profile.get("first_name", ""),
        last_name=profile.get("last_name", ""),
        email=email or "",
        username=profile.get("username", ""),
        is_premium=bool(profile.get("is_premium", False)),
        is_admin=bool(profile.get("is_admin", False)),
        created_at=created_at,
    )


def _build_scan(scan_data: Optional[dict]) -> Optional[AppUser]:
    if not isinstance(scan_data, dict):
        return None
    created_at = scan_data.get("created_at")
    if isinstance(created_at, str):
        try:
            created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            # Ensure always timezone-aware
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
        except ValueError:
            created_at = None
    scan = SimpleNamespace(**scan_data)
    scan.created_at = created_at
    scan.confidence_pct = round(float(scan.confidence or 0) * 100, 1) if scan.confidence is not None else 0.0
    scan.badge_color = {
        "normal": "success",
        "acne": "warning",
        "eczema": "info",
        "psoriasis": "danger",
        "ringworm": "secondary",
    }.get(str(getattr(scan, "predicted_class", "")).lower(), "primary")
    scan.scores_dict = {}
    try:
        if scan.all_scores:
            scan.scores_dict = json.loads(scan.all_scores) if isinstance(scan.all_scores, str) else scan.all_scores
    except Exception:
        scan.scores_dict = {}

    image_bytes = getattr(scan, "image_bytes", None)
    if image_bytes and isinstance(image_bytes, str) and image_bytes.startswith("data:image"):
        scan.image_url = image_bytes
    elif image_bytes:
        scan.image_url = _build_image_data_url(image_bytes, getattr(scan, "image_filename", None))
    else:
        scan.image_url = None
    return scan


def _build_feedback(feedback_data: Optional[dict]) -> Optional[SimpleNamespace]:
    if not isinstance(feedback_data, dict):
        return None
    created_at = feedback_data.get("created_at")
    if isinstance(created_at, str):
        try:
            created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        except ValueError:
            created_at = None
    feedback = SimpleNamespace(**feedback_data)
    feedback.created_at = created_at
    return feedback


def _get_current_user() -> Optional[AppUser]:
    user_id = session.get("user_id")
    if not user_id:
        return None

    profile = get_user_by_id(user_id)
    if isinstance(profile, dict):
        return _build_user(profile)

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


def _get_user_scans(user_id: Optional[str]) -> list[SimpleNamespace]:
    if not user_id or not supabase:
        return []
    try:
        response = supabase.table("scan_history").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
        scans = []
        for item in _safe_supabase_data(response):
            scan = _build_scan(item)
            if scan is not None:
                scans.append(scan)
        return scans
    except Exception:
        return []


def _get_user_scan_count(user_id: Optional[str]) -> int:
    return len(_get_user_scans(user_id))


def _upsert_profile_for_user(user_obj: object, email: Optional[str] = None) -> None:
    if isinstance(user_obj, dict):
        user_id = user_obj.get("id")
        first_name = (user_obj.get("user_metadata") or {}).get("first_name", "") if isinstance(user_obj.get("user_metadata"), dict) else ""
        last_name = (user_obj.get("user_metadata") or {}).get("last_name", "") if isinstance(user_obj.get("user_metadata"), dict) else ""
        username = (user_obj.get("user_metadata") or {}).get("username", "") if isinstance(user_obj.get("user_metadata"), dict) else ""
    else:
        user_id = getattr(user_obj, "id", None)
        user_meta = getattr(user_obj, "user_metadata", None) or {}
        first_name = user_meta.get("first_name", "") if isinstance(user_meta, dict) else ""
        last_name = user_meta.get("last_name", "") if isinstance(user_meta, dict) else ""
        username = user_meta.get("username", "") if isinstance(user_meta, dict) else ""

    if not user_id:
        return

    existing = get_user_by_id(user_id)
    if existing:
        update_user(user_id, {
            "first_name": first_name or existing.get("first_name", ""),
            "last_name": last_name or existing.get("last_name", ""),
            "username": username or existing.get("username", ""),
        })
    elif username:
        create_user(first_name or "", last_name or "", username, "temporary-password")


# ─── Auth Routes ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        first = request.form.get("first_name", "").strip()
        last = request.form.get("last_name", "").strip()
        username = request.form.get("username", "").strip().lower()
        password = request.form.get("password", "")

        if not all([first, last, username, password]):
            flash("All fields are required.", "danger")
            return render_template("register.html")
        if len(password) < 8 or not any(c.isupper() for c in password) \
                or not any(c.islower() for c in password) \
                or not any(c.isdigit() for c in password) \
                or not any(c in "!@#$%^&*()-_=+[]{}|;:',.<>?/`~\"\\\" " for c in password):
            flash("Password must be at least 8 characters and include uppercase, lowercase, a number, and a symbol.", "danger")
            return render_template("register.html")
        if len(username) < 3:
            flash("Username must be at least 3 characters.", "warning")
            return render_template("register.html")

        try:
            created_user = create_user(first, last, username, password)
        except RuntimeError as exc:
            flash(str(exc), "warning")
            return render_template("register.html")

        if created_user is None:
            flash("That username is already taken. Please choose another one.", "warning")
            return render_template("register.html")

        session.clear()
        session["user_id"] = created_user["id"]
        session["user_name"] = created_user["first_name"] or username
        session["is_admin"] = bool(created_user.get("is_admin", False))
        session["is_premium"] = bool(created_user.get("is_premium", False))
        flash("Your account was created. You can now sign in with your username and password.", "success")
        return redirect(url_for("dashboard"))
    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        username = request.form.get("identifier", "").strip().lower()
        password = request.form.get("password", "")

        if not username or not password:
            flash("Please enter your credentials.", "warning")
            return render_template("login.html")

        user = authenticate_user(username, password)
        if not user:
            session.clear()
            flash("Invalid username or password.", "danger")
            return render_template("login.html")

        session.clear()
        session["user_id"] = user["id"]
        session["user_name"] = user.get("first_name") or username
        session["is_admin"] = bool(user.get("is_admin", False))
        session["is_premium"] = bool(user.get("is_premium", False))
        return redirect(url_for("dashboard"))
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/forgot-password")
def forgot_password():
    flash("Password reset via email is disabled. Please contact support to change your password.", "info")
    return redirect(url_for("login"))


# ─── Dashboard ───────────────────────────────────────────────────────────────

@app.route("/dashboard")
@login_required
def dashboard():
    user = _get_current_user()
    if not user:
        session.clear()
        flash("Your session could not be restored. Please log in again.", "warning")
        return redirect(url_for("login"))

    recent_scans = _get_user_scans(user.id)[:3]
    scan_count = len(recent_scans)
    return render_template(
        "dashboard.html",
        user=user,
        scan_count=scan_count,
        recent_scans=recent_scans,
        max_free=MAX_FREE_SCANS,
    )


# ─── Analysis / Scan ─────────────────────────────────────────────────────────

@app.route("/analysis", methods=["GET", "POST"])
@login_required
def analysis():
    user = _get_current_user()
    if not user:
        session.clear()
        flash("Your session could not be restored. Please log in again.", "warning")
        return redirect(url_for("login"))

    scan_count = _get_user_scan_count(user.id)

    if request.method == "POST":
        # Usage limit check
        if not user.is_premium and scan_count >= MAX_FREE_SCANS:
            flash(f"Free limit of {MAX_FREE_SCANS} scans reached. Upgrade to Premium.", "warning")
            return redirect(url_for("dashboard"))

        if "image" not in request.files:
            flash("No file uploaded.", "danger")
            return render_template("analysis.html", user=user, scan_count=scan_count)

        file = request.files["image"]
        if file.filename == "":
            flash("No file selected.", "danger")
            return render_template("analysis.html", user=user, scan_count=scan_count)

        if not allowed_file(file.filename):
            flash("Only JPG and PNG files are accepted.", "danger")
            return render_template("analysis.html", user=user, scan_count=scan_count)

        ext = secure_filename(file.filename).rsplit(".", 1)[1].lower()
        filename = f"{uuid.uuid4().hex}.{ext}"
        image_bytes = file.read()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        temp_handle, temp_path = tempfile.mkstemp(suffix=f".{ext}")
        os.close(temp_handle)
        with open(temp_path, "wb") as handle:
            handle.write(image_bytes)

        try:
            pred = get_predictor()
            result = pred.predict(temp_path)
        except Exception as e:
            flash(f"Prediction error: {e}", "danger")
            return render_template("analysis.html", user=user, scan_count=scan_count)
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass

        # Save to history
        try:
            inserted = insert_scan_record({
                "user_id": user.id,
                "image_filename": filename,
                "image_bytes": image_b64,
                "predicted_class": result["predicted_class"],
                "confidence": result["confidence"],
                "all_scores": result["all_scores"],
            })
            history_id = inserted.get("id") if inserted else None
        except Exception as exc:
            flash(f"Unable to save your scan history right now: {exc}", "warning")
            history_id = None

        return render_template(
            "result.html",
            user=user,
            result=result,
            image_url=_build_image_data_url(image_b64, filename),
            history_id=history_id,
        )

    return render_template("analysis.html", user=user, scan_count=scan_count, max_free=MAX_FREE_SCANS)


# ─── Scan History ────────────────────────────────────────────────────────────

# Free users can only filter/download reports within this many days
FREE_REPORT_DAYS = 7

@app.route("/history")
@login_required
def history():
    user = _get_current_user()
    if not user:
        session.clear()
        flash("Your session could not be restored. Please log in again.", "warning")
        return redirect(url_for("login"))

    # ── Date filter params ────────────────────────────────────────────────────
    date_from_str = request.args.get("date_from", "").strip()
    date_to_str   = request.args.get("date_to",   "").strip()

    # Free users: clamp date range to last FREE_REPORT_DAYS days
    today = datetime.utcnow().date()
    free_min_date = today - timedelta(days=FREE_REPORT_DAYS - 1)

    date_from = None
    date_to   = None

    try:
        if date_from_str:
            date_from = datetime.strptime(date_from_str, "%Y-%m-%d").date()
        if date_to_str:
            date_to = datetime.strptime(date_to_str, "%Y-%m-%d").date()
    except ValueError:
        flash("Invalid date format.", "warning")

    # Build query
    query = supabase.table("scan_history").select("*").eq("user_id", user.id)
    if date_from:
        query = query.gte("created_at", datetime.combine(date_from, datetime.min.time()).isoformat())
    if date_to:
        query = query.lte("created_at", datetime.combine(date_to, datetime.max.time()).isoformat())

    try:
        response = query.order("created_at", desc=True).execute()
        scans = [_build_scan(item) for item in _safe_supabase_data(response) if _build_scan(item)]
    except Exception as exc:
        flash(f"Unable to load your scan history right now: {exc}", "warning")
        scans = []

    return render_template(
        "history.html",
        user=user,
        scans=scans,
        date_from=date_from_str,
        date_to=date_to_str,
        free_report_days=FREE_REPORT_DAYS,
        free_min_date=free_min_date.strftime("%Y-%m-%d"),
        today=today.strftime("%Y-%m-%d"),
    )


@app.route("/history/<int:scan_id>")
@login_required
def scan_detail(scan_id):
    user = _get_current_user()
    if not user:
        session.clear()
        flash("Your session could not be restored. Please log in again.", "warning")
        return redirect(url_for("login"))

    try:
        response = supabase.table("scan_history").select("*").eq("id", scan_id).eq("user_id", user.id).maybe_single().execute()
        scan = _build_scan(_safe_supabase_single(response))
    except Exception:
        scan = None

    if not scan:
        flash("The requested scan could not be found.", "warning")
        return redirect(url_for("history"))

    image_url = None
    image_bytes = getattr(scan, "image_bytes", None)
    if image_bytes:
        image_url = _build_image_data_url(image_bytes, getattr(scan, "image_filename", None))
    elif getattr(scan, "image_filename", None):
        image_url = url_for("static", filename=f"uploads/{scan.image_filename}")
    return render_template("scan_detail.html", user=user, scan=scan, image_url=image_url)


# ─── Profile ─────────────────────────────────────────────────────────────────

@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    user = _get_current_user()
    if not user:
        session.clear()
        flash("Your session could not be restored. Please log in again.", "warning")
        return redirect(url_for("login"))
    if request.method == "POST":
        first = request.form.get("first_name", "").strip()
        last = request.form.get("last_name", "").strip()
        new_pass = request.form.get("new_password", "")
        updates = {}
        if first:
            updates["first_name"] = first
            session["user_name"] = first
        if last:
            updates["last_name"] = last
        if new_pass:
            if len(new_pass) < 8 or not any(c.isupper() for c in new_pass) \
                    or not any(c.islower() for c in new_pass) \
                    or not any(c.isdigit() for c in new_pass) \
                    or not any(c in "!@#$%^&*()-_=+[]{}|;:',.<>?/`~\"\\\" " for c in new_pass):
                flash("Password must be at least 8 characters and include uppercase, lowercase, a number, and a symbol.", "warning")
                return render_template("profile.html", user=user)
            update_user(user.id, {"password": new_pass})

        if updates:
            updated = update_user(user.id, updates)
            if not updated:
                flash("Unable to update your profile right now.", "warning")
                return render_template("profile.html", user=user)

        flash("Profile updated successfully.", "success")
        user = _get_current_user() or user
    return render_template("profile.html", user=user)


# ─── Feedback ────────────────────────────────────────────────────────────────

@app.route("/feedback", methods=["GET", "POST"])
@login_required
def feedback():
    user = _get_current_user()
    if not user:
        session.clear()
        flash("Your session could not be restored. Please log in again.", "warning")
        return redirect(url_for("login"))
    if request.method == "POST":
        message = request.form.get("message", "").strip()
        rating = request.form.get("rating", "5")
        if not message:
            flash("Feedback message is required.", "danger")
            return render_template("feedback.html", user=user)
        try:
            supabase.table("feedback").insert({
                "user_id": user.id,
                "message": message,
                "rating": int(rating),
            }).execute()
        except Exception as exc:
            flash(f"Unable to save feedback right now: {exc}", "warning")
            return render_template("feedback.html", user=user)
        flash("Thank you for your feedback!", "success")
        return redirect(url_for("dashboard"))
    return render_template("feedback.html", user=user)


# ─── Subscription ────────────────────────────────────────────────────────────

@app.route("/upgrade")
@login_required
def upgrade():
    user = _get_current_user()
    return render_template("upgrade.html", user=user)


@app.route("/upgrade/confirm", methods=["POST"])
@login_required
def upgrade_confirm():
    user = _get_current_user()
    if not user:
        session.clear()
        flash("Your session could not be restored. Please log in again.", "warning")
        return redirect(url_for("login"))
    updated = update_user(user.id, {"is_premium": True})
    if not updated:
        flash("Unable to activate Premium right now.", "warning")
        return redirect(url_for("dashboard"))
    flash("Congratulations! You are now a Premium member.", "success")
    return redirect(url_for("dashboard"))


# ─── PDF Reports ─────────────────────────────────────────────────────────────

def _parse_report_dates(user):
    """
    Parse date_from / date_to from query params.
    Free users are restricted to the last FREE_REPORT_DAYS days.
    Returns (date_from, date_to, error_msg_or_None).
    """
    today       = datetime.utcnow().date()
    free_min    = today - timedelta(days=FREE_REPORT_DAYS - 1)

    date_from_str = request.args.get("date_from", "").strip()
    date_to_str   = request.args.get("date_to",   "").strip()

    try:
        date_from = datetime.strptime(date_from_str, "%Y-%m-%d").date() if date_from_str else None
        date_to   = datetime.strptime(date_to_str,   "%Y-%m-%d").date() if date_to_str   else None
    except ValueError:
        return None, None, "Invalid date format."

    # Enforce free-user limit
    if not user.is_premium:
        if date_from and date_from < free_min:
            return None, None, (
                f"Free accounts can only download reports for the last "
                f"{FREE_REPORT_DAYS} days. Upgrade to Premium for full history."
            )
        # If no date_from supplied for free user, default to free_min
        if date_from is None:
            date_from = free_min

    return date_from, date_to, None


def _filter_scans(user_id, date_from, date_to):
    query = supabase.table("scan_history").select("*").eq("user_id", user_id)
    if date_from:
        query = query.gte("created_at", datetime.combine(date_from, datetime.min.time()).isoformat())
    if date_to:
        query = query.lte("created_at", datetime.combine(date_to, datetime.max.time()).isoformat())
    try:
        response = query.order("created_at", desc=True).execute()
        return [_build_scan(item) for item in _safe_supabase_data(response) if _build_scan(item)]
    except Exception:
        return []


@app.route("/report/history")
@login_required
def report_history():
    """Download filtered scan history as PDF."""
    user = _get_current_user()
    if not user:
        session.clear()
        flash("Your session could not be restored. Please log in again.", "warning")
        return redirect(url_for("login"))

    date_from, date_to, err = _parse_report_dates(user)
    if err:
        flash(err, "warning")
        return redirect(url_for("history"))

    scans = _filter_scans(user.id, date_from, date_to)

    if not scans:
        flash("No scans found for the selected date range.", "info")
        return redirect(url_for("history"))

    pdf_bytes = generate_history_pdf(
        user, scans, app.config["UPLOAD_FOLDER"],
        date_from=date_from, date_to=date_to,
    )
    response = make_response(pdf_bytes)
    response.headers["Content-Type"] = "application/pdf"
    fname = f"dermoscan_report_{user.id}_{datetime.utcnow().strftime('%Y%m%d')}.pdf"
    response.headers["Content-Disposition"] = f"attachment; filename={fname}"
    return response


@app.route("/report/scan/<int:scan_id>")
@login_required
def report_scan(scan_id):
    """Download a single scan as PDF."""
    user = _get_current_user()
    if not user:
        session.clear()
        flash("Your session could not be restored. Please log in again.", "warning")
        return redirect(url_for("login"))

    try:
        response = supabase.table("scan_history").select("*").eq("id", scan_id).eq("user_id", user.id).maybe_single().execute()
        scan = _build_scan(_safe_supabase_single(response))
    except Exception:
        scan = None

    if not scan:
        flash("The requested scan could not be found.", "warning")
        return redirect(url_for("history"))

    # Free user: only allow report if scan is within FREE_REPORT_DAYS
    if not user.is_premium:
        cutoff = datetime.now(timezone.utc) - timedelta(days=FREE_REPORT_DAYS)
        scan_dt = scan.created_at
        # Ensure scan_dt is timezone-aware for comparison
        if scan_dt and scan_dt.tzinfo is None:
            from datetime import timezone as _tz
            scan_dt = scan_dt.replace(tzinfo=_tz.utc)
        if scan_dt and scan_dt < cutoff:
            flash(
                f"Free accounts can only download reports for scans from the last "
                f"{FREE_REPORT_DAYS} days. Upgrade to Premium for full access.",
                "warning",
            )
            return redirect(url_for("scan_detail", scan_id=scan_id))

    pdf_bytes = generate_single_pdf(user, scan, app.config["UPLOAD_FOLDER"])
    response = make_response(pdf_bytes)
    response.headers["Content-Type"] = "application/pdf"
    fname = f"dermoscan_scan_{scan_id}_{datetime.utcnow().strftime('%Y%m%d')}.pdf"
    response.headers["Content-Disposition"] = f"attachment; filename={fname}"
    return response


# ─── Admin Panel ─────────────────────────────────────────────────────────────

@app.route("/admin")
@admin_required
def admin_panel():
    users = []
    for item in list_users():
        built = _build_user(item)
        if built:
            users.append(built)

    try:
        scans_response = supabase.table("scan_history").select("*").order("created_at", desc=True).limit(50).execute()
        scans = [_build_scan(item) for item in _safe_supabase_data(scans_response) if _build_scan(item)]
    except Exception:
        scans = []

    try:
        feedback_response = supabase.table("feedback").select("*").order("created_at", desc=True).execute()
        feedbacks = [_build_feedback(item) for item in _safe_supabase_data(feedback_response) if _build_feedback(item)]
    except Exception:
        feedbacks = []

    total_scans = len(scans)
    total_users = len(users)
    return render_template(
        "admin.html",
        users=users,
        scans=scans,
        feedbacks=feedbacks,
        total_scans=total_scans,
        total_users=total_users,
    )


# ─── API endpoints ───────────────────────────────────────────────────────────

@app.route("/api/predict", methods=["POST"])
@login_required
def api_predict():
    """JSON API for prediction (AJAX use)."""
    user = _get_current_user()
    if not user:
        return jsonify({"error": "Session expired"}), 401
    scan_count = _get_user_scan_count(user.id)
    if not user.is_premium and scan_count >= MAX_FREE_SCANS:
        return jsonify({"error": "Scan limit reached"}), 403

    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    ext = secure_filename(file.filename).rsplit(".", 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    image_bytes = file.read()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    temp_handle, temp_path = tempfile.mkstemp(suffix=f".{ext}")
    os.close(temp_handle)
    with open(temp_path, "wb") as handle:
        handle.write(image_bytes)

    try:
        pred = get_predictor()
        result = pred.predict(temp_path)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    try:
        inserted = insert_scan_record({
            "user_id": user.id,
            "image_filename": filename,
            "image_bytes": image_b64,
            "predicted_class": result["predicted_class"],
            "confidence": result["confidence"],
            "all_scores": result["all_scores"],
        })
        history_id = inserted.get("id") if inserted else None
    except Exception:
        history_id = None

    result["history_id"] = history_id
    result["image_url"] = _build_image_data_url(image_b64, filename)
    return jsonify(result)


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=5000)
