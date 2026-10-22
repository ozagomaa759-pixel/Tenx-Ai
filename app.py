import os
from flask import Flask, request, jsonify, send_from_directory
import requests

app = Flask(__name__, static_folder="static")

@app.get("/")
def home():
    return send_from_directory("static", "index.html")

@app.post("/api/chat")
def chat():
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        return jsonify(error="Missing DEEPSEEK_API_KEY. Configure it in your terminal before starting the server."), 503
    body = request.get_json(silent=True) or {}
    messages = body.get("messages", [])
    if not isinstance(messages, list) or not messages:
        return jsonify(error="Please enter a message."), 400
    safe_messages = []
    for m in messages[-20:]:
        if isinstance(m, dict) and m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str):
            safe_messages.append({"role": m["role"], "content": m["content"][:8000]})
    try:
        r = requests.post(
            "https://api.deepseek.com/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": "deepseek-chat", "messages": [
                {"role": "system", "content": "You are TENX AI, a helpful, friendly educational assistant. Explain clearly, support learning, and answer in the language the user uses."},
                *safe_messages
            ], "temperature": 0.7},
            timeout=60
        )
        if not r.ok:
            return jsonify(error=f"AI provider returned status {r.status}. Check the API key, account balance, and model access."), 502
        data = r.json()
        answer = data["choices"][0]["message"]["content"]
        return jsonify(reply=answer)
    except requests.Timeout:
        return jsonify(error="The AI request timed out. Please try again."), 504
    except Exception:
        app.logger.exception("TENX AI request failed")
        return jsonify(error="Could not reach the AI service. Check your internet connection and server logs."), 502

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
