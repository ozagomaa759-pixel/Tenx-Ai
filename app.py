import os
import requests
from flask import Flask, jsonify, request

app = Flask(__name__)

@app.get("/")
def home():
    # Vercel serves files in /public through its CDN; this route is a fallback
    from flask import send_from_directory
    return send_from_directory(".", "index.html")

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

    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash-lite:generateContent"
    payload = {
        
    "systemInstruction": {
    "parts": [{
        "text": """
You are TENX AI, a high-quality educational AI assistant for students.

ACCURACY:
- Give accurate, relevant, and clear answers.
- Never invent facts. If uncertain, state the uncertainty honestly.
- Explain reasoning step by step when it helps the student understand.
- For simple questions, answer directly without unnecessary detail.
- For programming questions, explain the code clearly and use correct code examples.

LANGUAGE:
- Reply in the same language as the user's message.
- For Arabic, use natural, clear Arabic.
- When Arabic and English appear together, keep English technical terms spelled correctly.
- Do not reverse letters, words, numbers, or code.

FORMATTING:
- Use plain text by default.
- Do not use asterisks (*) for decoration or emphasis.
- Avoid unnecessary Markdown symbols, headings, and bullet points.
- Use short paragraphs and numbered steps when helpful.
- Put programming code in fenced code blocks using the correct language.
- Keep mathematical expressions and technical terms readable.

PERSONALITY:
- Be respectful, encouraging, and professional.
- Teach students instead of simply overwhelming them with information.
- Do not claim certainty when the answer is not verified.
"""
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
            timeout=30
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
