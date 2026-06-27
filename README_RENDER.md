# Render deployment notes

## Deploy on Render
1. Push this repository to GitHub.
2. Create a new Web Service on Render.
3. Connect the repository.
4. Use the following start command:
   ```bash
   gunicorn app:app
   ```
5. Add these environment variables in Render:
   - SECRET_KEY
   - SITE_URL
   - AUTH_CONFIRM_PATH
   - SUPABASE_URL
   - SUPABASE_KEY
   - SUPABASE_SERVICE_ROLE_KEY

## Notes
- The app uses Supabase for users and scan history.
- The service is configured with a free plan in render.yaml.
- The model can fall back to a mock predictor if the trained model is unavailable.
