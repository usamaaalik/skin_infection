# Vercel deployment notes

## Requirements
- Python 3.11
- Add these environment variables in Vercel:
  - SECRET_KEY
  - SUPABASE_URL
  - SUPABASE_KEY
  - SUPABASE_SERVICE_ROLE_KEY

## Deploy
1. Push this repository to GitHub.
2. Open Vercel and import the project.
3. Set the root directory to the project folder.
4. Vercel will use vercel.json and api/index.py automatically.

## Notes
- The app now exposes a Vercel-compatible entry point through api/index.py.
- Heavy machine-learning startup can be slow in serverless environments, so the app should be tested with the model fallback enabled.
