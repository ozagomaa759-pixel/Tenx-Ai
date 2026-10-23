import os
import requests
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder=None)

@app.get("/")
def home():
    return send_from_directory(".", "index.html")


@app.post("/api/chat")
def chat():
    data = request.get_json(silent=True) or {}
    contents = data.get("contents", [])
    
    if not contents or contents[-1]["role"] != "user":
        return jsonify(error="Please send a new message."), 400

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return jsonify(error="Gemini API Key is not configured."), 500

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    
    payload = {
        "systemInstruction": {
            "parts": [{
                "text": "You are TENX AI, a friendly educational assistant. Explain clearly, support learning, and answer in the same language the user uses."
            }]
        },
        "contents": contents,
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 2048
        }
    }
    
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        res_data = response.json()
        return jsonify(res_data)
    except Exception as e:
        return jsonify(error=str(e)), 500

if __name__ == "__main__":
    app.run(debug=True)
