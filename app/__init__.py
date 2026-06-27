"""
DermoScan – Application factory.
"""
from __future__ import annotations

import os

from flask import Flask

from database import db, init_db
from app.config import (
    MAX_CONTENT_LENGTH,
    SECRET_KEY,
    UPLOAD_FOLDER,
)


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates"),
        static_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), "static"),
    )

    # ── Configuration ──────────────────────────────────────────────────────────
    app.secret_key                       = SECRET_KEY
    app.config["UPLOAD_FOLDER"]          = UPLOAD_FOLDER
    app.config["MAX_CONTENT_LENGTH"]     = MAX_CONTENT_LENGTH
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    # ── Database ───────────────────────────────────────────────────────────────
    db.init_app(app)
    init_db()

    # ── Blueprints ─────────────────────────────────────────────────────────────
    from app.routes.auth      import bp as auth_bp
    from app.routes.dashboard import bp as dashboard_bp
    from app.routes.scan      import bp as scan_bp
    from app.routes.profile   import bp as profile_bp
    from app.routes.reports   import bp as reports_bp
    from app.routes.admin     import bp as admin_bp
    from app.routes.misc      import bp as misc_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(scan_bp)
    app.register_blueprint(profile_bp)
    app.register_blueprint(reports_bp)
    app.register_blueprint(admin_bp)
    app.register_blueprint(misc_bp)

    return app
