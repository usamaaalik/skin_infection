"""
DermoScan – Application factory.
"""
from __future__ import annotations

import os

from flask import Flask

from database import db, init_db
from dermoscan.config import (
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

    app.secret_key                            = SECRET_KEY
    app.config["UPLOAD_FOLDER"]               = UPLOAD_FOLDER
    app.config["MAX_CONTENT_LENGTH"]          = MAX_CONTENT_LENGTH
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    db.init_app(app)
    init_db()

    from dermoscan.routes.auth      import bp as auth_bp
    from dermoscan.routes.dashboard import bp as dashboard_bp
    from dermoscan.routes.scan      import bp as scan_bp
    from dermoscan.routes.profile   import bp as profile_bp
    from dermoscan.routes.reports   import bp as reports_bp
    from dermoscan.routes.admin     import bp as admin_bp
    from dermoscan.routes.misc      import bp as misc_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(scan_bp)
    app.register_blueprint(profile_bp)
    app.register_blueprint(reports_bp)
    app.register_blueprint(admin_bp)
    app.register_blueprint(misc_bp)

    return app
