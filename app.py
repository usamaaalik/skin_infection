"""
DermoScan – Entry point.
Exposes `app` for gunicorn (gunicorn app:app) and the Flask CLI.
"""
from dermoscan import create_app

app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
