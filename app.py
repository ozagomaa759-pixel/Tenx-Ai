import os
import requests
from flask import Flask, jsonify, request

app = Flask(__name__)

@app.get("/")
def home():
    # Vercel serves files in /public through its CDN; this route is a fallback
    from flask import send_from_directory
    return send_from_directory("public", "index.html")

@app.get("/api/health")
def health():
    return jsonify(ok=True, configured=bool(os.getenv("GEMINI_API_KEY")))

@app.post("/api/chat")
def chat():
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return jsonify(error="The server is missing GEMINI_API_KEY. Add it in Vercel Environment Variables, then redeploy."), 503

    body = request.get_json(silent=True) or {}
    messages = body.get("messages", [])
    if not isinstance(messages, list) or not messages:
        return jsonify(error="Write a message first."), 400

    contents = []
    for item in messages[-16:]:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            continue
        content = content.strip()[:6000]
        if content:
            contents.append({
                "role": "user" if role == "user" else "model",
                "parts": [{"text": content}]
            })

    if not contents or contents[-1]["role"] != "user":
        return jsonify(error="Please send a new message."), 400

    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    payload = {
        "systemInstruction": {
            "parts": [{
                "text": "You are TENX AI, a helpful educational AI assistant for students. Answer accurately, clearly, and in the same language as the user. Explain step by step when useful. If unsure, say so."
            }]
        },
        "contents": contents,
        "generationConfig": {"temperature": 0.7, "maxOutputTokens": 1500}
    }

    try:
        response = requests.post(
            url,
            headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
            json=payload,
            timeout=45
        )
        try:
            data = response.json()
        except ValueError:
            data = {}

        if not response.ok:
            # Avoid logging or returning the API key.
            message = data.get("error", {}).get("message", "Gemini request failed.")
            return jsonify(error=f"Gemini API error ({response.status_code}): {message}"), 502

        candidates = data.get("candidates", [])
        parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
        answer = "".join(p.get("text", "") for p in parts if isinstance(p, dict) and isinstance(p.get("text"), str)).strip()
        if not answer:
            return jsonify(error="Gemini returned an empty response. Please try again."), 502
        return jsonify(reply=answer)
    except requests.Timeout:
        return jsonify(error="The AI took too long to respond. Try again."), 504
    except requests.RequestException:
        return jsonify(error="Could not connect to Gemini. Check the deployment logs and try again."), 502

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
