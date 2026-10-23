
import os
from flask import Flask, request, jsonify, send_from_directory
import requests
app = Flask(__name__, static_folder=None)


@app.get("/")
def home():
    return send_from_directory("public", "index.html")


@app.post("/api/chat")
def chat():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return jsonify(
            error="Gemini API key is not configured on the server."
        ), 503

    body = request.get_json(silent=True) or {}
    messages = body.get("messages", [])

    if not isinstance(messages, list) or not messages:
        return jsonify(error="Please enter a message."), 400

    contents = []

    for message in messages[-20:]:
        if not isinstance(message, dict):
            continue

        role = message.get("role")
        content = message.get("content")

        if role not in ("user", "assistant") or not isinstance(content, str):
            continue

        content = content[:8000].strip()

        if not content:
            continue

        contents.append({
            "role": "user" if role == "user" else "model",
            "parts": [{"text": content}]
        })

    if not contents or contents[-1]["role"] != "user":
        return jsonify(error="Please send a new message."), 400

    
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={os.environ.get('GEMINI_API_KEY')}"

    )

    payload = {
        "systemInstruction": {
            "parts": [{
                "text": (
                    "You are TENX AI, a friendly educational assistant. "
                    "Explain clearly, support learning, and answer in "
                    "the same language the user uses."
                )
            }]
        },
        "contents": contents,
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 2048
        }
    }

    try:
        response = requests.post(
            url,
            headers={
                "x-goog-api-key": api_key,
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=60
        )

        if not response.ok:
            app.logger.error(
                "Gemini returned status %s: %s",
                response.status_code,
                response.text[:1000]
            )
            return jsonify(
                error="Gemini could not answer. Check your API key, "
                      "model access, and API quota in Google AI Studio."
            ), 502

        data = response.json()
        candidates = data.get("candidates", [])

        if not candidates:
            return jsonify(
                error="Gemini returned no answer. Please try again."
            ), 502

        parts = candidates[0].get("content", {}).get("parts", [])
        answer = "".join(
            part.get("text", "")
            for part in parts
            if isinstance(part.get("text"), str)
        ).strip()

        if not answer:
            return jsonify(error="The AI returned an empty answer."), 502

        return jsonify(reply=answer)

    except requests.Timeout:
        return jsonify(
            error="The request timed out. Please try again."
        ), 504

    except requests.RequestException:
        app.logger.exception("TENX AI connection failed")
        return jsonify(
            error="Could not connect to Gemini. Please try again."
        ), 502

    except Exception:
        app.logger.exception("TENX AI request failed")
        return jsonify(error="An unexpected server error occurred."), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
