# TENX AI

A student-friendly AI chat app using Flask, Vercel, and the Gemini API.

## Files
- `app.py`: Flask application and `/api/chat` endpoint
- `public/index.html`: responsive TENX AI chat interface
- `requirements.txt`: Python dependencies

## Deploy on Vercel
1. Upload these files to the root of a GitHub repository.
2. Import the repository in Vercel. Framework detection should identify Flask from `app.py` and `requirements.txt`.
3. In Vercel: Project → Settings → Environment Variables, add:
   - Name: `GEMINI_API_KEY`
   - Value: your Gemini API key (keep it secret)
   - Environment: Production (and Preview if you need preview deployments)
4. Redeploy after adding or changing the environment variable.
5. Test `https://YOUR-DOMAIN/api/health`; it should return JSON with `ok: true` and `configured: true`.
6. Open the website and send a message.

Never put the API key in `public/index.html`, frontend JavaScript, GitHub, or chat messages.
