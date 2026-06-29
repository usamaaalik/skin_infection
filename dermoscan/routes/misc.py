"""
DermoScan – Miscellaneous routes: feedback, upgrade, Stripe payment, JSON API.
"""
from __future__ import annotations

import base64
import os
import tempfile
import uuid

import stripe
from flask import Blueprint, flash, jsonify, redirect, render_template, request, session, url_for
from werkzeug.utils import secure_filename

from database import insert_scan_record, supabase, update_user
from dermoscan.config import (
    MAX_FREE_SCANS,
    SITE_URL,
    STRIPE_PRICE_ID,
    STRIPE_PUBLISHABLE_KEY,
    STRIPE_SECRET_KEY,
    STRIPE_WEBHOOK_SECRET,
)
from dermoscan.decorators import login_required
from dermoscan.services.predictor_service import get_predictor
from dermoscan.services.scan_service import get_user_scan_count
from dermoscan.services.user_service import get_current_user
from dermoscan.utils import allowed_file, build_image_data_url

# Set Stripe secret key once at startup
stripe.api_key = STRIPE_SECRET_KEY

bp = Blueprint("misc", __name__)


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------

@bp.route("/feedback", methods=["GET", "POST"])
@login_required
def feedback():
    user = get_current_user()
    if not user:
        session.clear()
        flash("Your session could not be restored. Please log in again.", "warning")
        return redirect(url_for("auth.login"))
    if request.method == "POST":
        message = request.form.get("message", "").strip()
        rating  = request.form.get("rating", "5")
        if not message:
            flash("Feedback message is required.", "danger")
            return render_template("feedback.html", user=user)
        try:
            supabase.table("feedback").insert({
                "user_id": user.id, "message": message, "rating": int(rating),
            }).execute()
        except Exception as exc:
            flash(f"Unable to save feedback right now: {exc}", "warning")
            return render_template("feedback.html", user=user)
        flash("Thank you for your feedback!", "success")
        return redirect(url_for("dashboard.dashboard"))
    return render_template("feedback.html", user=user)


# ---------------------------------------------------------------------------
# Upgrade Page
# ---------------------------------------------------------------------------

@bp.route("/upgrade")
@login_required
def upgrade():
    # Pass publishable key to frontend so Stripe.js can use it
    return render_template(
        "upgrade.html",
        user=get_current_user(),
        stripe_publishable_key=STRIPE_PUBLISHABLE_KEY,
    )


# ---------------------------------------------------------------------------
# Stripe Checkout — Create Session
# ---------------------------------------------------------------------------

@bp.route("/upgrade/checkout", methods=["POST"])
@login_required
def upgrade_checkout():
    """
    Creates a Stripe Checkout Session and redirects user to Stripe payment page.
    In test mode, no real money is charged.
    """
    user = get_current_user()
    if not user:
        session.clear()
        flash("Your session could not be restored. Please log in again.", "warning")
        return redirect(url_for("auth.login"))

    # Already premium — no need to charge
    if user.is_premium:
        flash("You are already a Premium member!", "info")
        return redirect(url_for("dashboard.dashboard"))

    try:
        # Create Stripe Checkout Session
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[{
                "price": STRIPE_PRICE_ID,  # Price ID from Stripe Dashboard
                "quantity": 1,
            }],
            mode="subscription",  # Monthly recurring
            success_url=SITE_URL + url_for("misc.upgrade_success") + "?session_id={CHECKOUT_SESSION_ID}",
            cancel_url=SITE_URL + url_for("misc.upgrade_cancel"),
            metadata={"user_id": str(user.id)},  # Store user ID for later use
        )
        return redirect(checkout_session.url, code=303)

    except stripe.error.StripeError as exc:
        flash(f"Payment error: {exc.user_message}", "danger")
        return redirect(url_for("misc.upgrade"))

    except Exception as exc:
        flash(f"Something went wrong: {exc}", "danger")
        return redirect(url_for("misc.upgrade"))


# ---------------------------------------------------------------------------
# Stripe Success — After Payment
# ---------------------------------------------------------------------------

@bp.route("/upgrade/success")
@login_required
def upgrade_success():
    user = get_current_user()
    if not user:
        session.clear()
        return redirect(url_for("auth.login"))

    session_id = request.args.get("session_id")

    if session_id:
        try:
            # Verify payment with Stripe
            checkout_session = stripe.checkout.Session.retrieve(session_id)

            # For subscriptions use status == "complete" not payment_status == "paid"
            if checkout_session.status == "complete":
                update_user(user.id, {"is_premium": True})
                flash("🎉 Congratulations! You are now a Premium member. Enjoy unlimited scans!", "success")
                return redirect(url_for("dashboard.dashboard"))

        except stripe.error.StripeError as exc:
            flash(f"Stripe error: {exc.user_message}", "warning")
            return redirect(url_for("misc.upgrade"))

    flash("Payment could not be verified. Please contact support.", "warning")
    return redirect(url_for("misc.upgrade"))


# ---------------------------------------------------------------------------
# Stripe Cancel — User Cancelled Payment
# ---------------------------------------------------------------------------

@bp.route("/upgrade/cancel")
@login_required
def upgrade_cancel():
    """Stripe redirects here if user clicks Back/Cancel on payment page."""
    flash("Payment cancelled. You can upgrade anytime.", "info")
    return redirect(url_for("misc.upgrade"))


# ---------------------------------------------------------------------------
# Old confirm route kept for backward compatibility (demo mode fallback)
# ---------------------------------------------------------------------------

@bp.route("/upgrade/confirm", methods=["POST"])
@login_required
def upgrade_confirm():
    user = get_current_user()
    if not user:
        session.clear()
        flash("Your session could not be restored. Please log in again.", "warning")
        return redirect(url_for("auth.login"))
    if not update_user(user.id, {"is_premium": True}):
        flash("Unable to activate Premium right now.", "warning")
        return redirect(url_for("dashboard.dashboard"))
    flash("Congratulations! You are now a Premium member.", "success")
    return redirect(url_for("dashboard.dashboard"))


# ---------------------------------------------------------------------------
# API Predict
# ---------------------------------------------------------------------------

@bp.route("/api/predict", methods=["POST"])
@login_required
def api_predict():
    user = get_current_user()
    if not user:
        return jsonify({"error": "Session expired"}), 401
    if not user.is_premium and get_user_scan_count(user.id) >= MAX_FREE_SCANS:
        return jsonify({"error": "Scan limit reached"}), 403

    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image provided"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

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
        return jsonify({"error": str(exc)}), 500
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    # Reject non-skin images
    if result.get("rejected"):
        return jsonify({"error": result.get("rejection_reason", "Not a valid skin image.")}), 422

    try:
        inserted = insert_scan_record({
            "user_id": user.id, "image_filename": filename, "image_bytes": img_b64,
            "predicted_class": result["predicted_class"], "confidence": result["confidence"],
            "all_scores": result["all_scores"],
        })
        result["history_id"] = inserted.get("id") if inserted else None
    except Exception:
        result["history_id"] = None

    result["image_url"] = build_image_data_url(img_b64, filename)
    return jsonify(result)