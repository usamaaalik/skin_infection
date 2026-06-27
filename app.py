"""
DermoScan – Entry point.
Imports the application factory and exposes `app` for gunicorn / Flask CLI.
"""
from app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
