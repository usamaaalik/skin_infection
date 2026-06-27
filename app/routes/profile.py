"""
DermoScan – Profile route.
"""
from __future__ import annotations

from flask import Blueprint, flash, redirect, render_template, request, session, url_for

from database import update_user
from app.decorators import login_required
from app.services.user_service import get_current_user
from app.utils import validate_password

bp = Blueprint("profile", __name__)


@bp.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    user = get_current_user()
    if not user:
        session.clear()
        flash("Your session could not be restored. Please log in again.", "warning")
        return redirect(url_for("auth.login"))

    if request.method == "POST":
        first    = request.form.get("first_name",   "").strip()
        last     = request.form.get("last_name",    "").strip()
        new_pass = request.form.get("new_password", "")

        updates = {}
        if first:
            updates["first_name"] = first
            session["user_name"]  = first
        if last:
            updates["last_name"] = last

        if new_pass:
            pw_error = validate_password(new_pass)
            if pw_error:
                flash(pw_error, "warning")
                return render_template("profile.html", user=user)
            update_user(user.id, {"password": new_pass})

        if updates:
            if not update_user(user.id, updates):
                flash("Unable to update your profile right now.", "warning")
                return render_template("profile.html", user=user)

        flash("Profile updated successfully.", "success")
        user = get_current_user() or user

    return render_template("profile.html", user=user)
