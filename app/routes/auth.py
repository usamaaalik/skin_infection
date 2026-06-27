"""
DermoScan – Authentication routes.
"""
from __future__ import annotations

from flask import Blueprint, flash, redirect, render_template, request, session, url_for

from database import authenticate_user, create_user
from app.services.user_service import set_session
from app.utils import validate_password

bp = Blueprint("auth", __name__)


@bp.route("/")
def index():
    if "user_id" in session:
        return redirect(url_for("dashboard.dashboard"))
    return redirect(url_for("auth.login"))


@bp.route("/register", methods=["GET", "POST"])
def register():
    if "user_id" in session:
        return redirect(url_for("dashboard.dashboard"))

    if request.method == "POST":
        first    = request.form.get("first_name", "").strip()
        last     = request.form.get("last_name",  "").strip()
        username = request.form.get("username",   "").strip().lower()
        password = request.form.get("password",   "")

        if not all([first, last, username, password]):
            flash("All fields are required.", "danger")
            return render_template("register.html")

        pw_error = validate_password(password)
        if pw_error:
            flash(pw_error, "danger")
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

        set_session(created_user)
        flash("Your account was created. You can now sign in with your username and password.", "success")
        return redirect(url_for("dashboard.dashboard"))

    return render_template("register.html")


@bp.route("/login", methods=["GET", "POST"])
def login():
    if "user_id" in session:
        return redirect(url_for("dashboard.dashboard"))

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

        set_session(user)
        return redirect(url_for("dashboard.dashboard"))

    return render_template("login.html")


@bp.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("auth.login"))


@bp.route("/forgot-password")
def forgot_password():
    flash("Password reset via email is disabled. Please contact support to change your password.", "info")
    return redirect(url_for("auth.login"))


@bp.route("/update-password", methods=["GET", "POST"])
def update_password():
    if request.method == "POST":
        from database import update_user
        from app.utils import validate_password
        user_id  = session.get("user_id")
        new_pass = request.form.get("new_password", "")
        pw_error = validate_password(new_pass)
        if pw_error:
            flash(pw_error, "danger")
            return render_template("update_password.html")
        if user_id:
            update_user(user_id, {"password": new_pass})
            flash("Password updated successfully.", "success")
            return redirect(url_for("auth.login"))
    return render_template("update_password.html")
