
import os
import json
import logging
import requests

from flask import Flask, request, Response, jsonify, send_from_directory

app = Flask(__name__, static_folder=None)
logging.basicConfig(level=logging.INFO)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")

MAX_HISTORY = 8
MAX_MESSAGE_LENGTH = 4000

SYSTEM_PROMPT = """
You are TENX AI, a helpful educational AI assistant for students.
Reply in the same language as the user's latest message.
When the user writes Arabic, reply in clear natural Arabic.
When the user writes Egyptian Arabic, friendly Egyptian Arabic is welcome.
Explain programming, science, mathematics, and technology clearly.
When an image is provided, analyze its visible content.
Be accurate, concise for simple questions, and honest about uncertainty.
"""

session = requests.Session()


@app.route("/", methods=["GET"])
def home():
    return send_from_directory(BASE_DIR, "Index.html")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model": MODEL,
        "api_key_configured": bool(API_KEY)
    })


def build_contents(messages, image=None):
    contents = []

    for message in messages[-MAX_HISTORY:]:
        if not isinstance(message, dict):
            continue

        role = message.get("role", "user")
        text = message.get("content", message.get("text", ""))

        if role not in ("user", "assistant") or not isinstance(text, str):
            continue

        contents.append({
            "role": "model" if role == "assistant" else "user",
            "parts": [{"text": text[:MAX_MESSAGE_LENGTH]}]
        })

    if image and isinstance(image, dict):
        mime_type = image.get("mimeType", "")
        image_data = image.get("data", "")

        allowed_types = {
            "image/jpeg", "image/png", "image/webp",
            "image/heic", "image/heif"
        }

        if (
            mime_type in allowed_types
            and isinstance(image_data, str)
            and image_data
            and len(image_data) <= 12_000_000
        ):
            if not contents or contents[-1]["role"] != "user":
                contents.append({"role": "user", "parts": []})

            contents[-1]["parts"].append({
                "inline_data": {
                    "mime_type": mime_type,
                    "data": image_data
                }
            })

    return contents


@app.route("/api/chat", methods=["POST"])
def chat():
    if not API_KEY:
        return jsonify({
            "error": "GEMINI_API_KEY is missing from Vercel Environment Variables."
        }), 500

    data = request.get_json(silent=True) or {}
    messages = data.get("messages", [])
    image = data.get("image")

    if not isinstance(messages, list) or not messages:
        return jsonify({"error": "Please send a message."}), 400

    contents = build_contents(messages, image)

    if not contents:
        return jsonify({"error": "No valid message was provided."}), 400

    payload = {
        "system_instruction": {
            "parts": [{"text": SYSTEM_PROMPT}]
        },
        "contents": contents,
        "generationConfig": {
            "temperature": 0.5,
            "maxOutputTokens": 1024
        }
    }

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{MODEL}:streamGenerateContent?alt=sse&key={API_KEY}"
    )

    def generate():
        upstream = None

        try:
            upstream = session.post(
                url,
                json=payload,
                headers={
                    "Content-Type": "application/json; charset=utf-8",
                    "Accept": "text/event-stream"
                },
                stream=True,
                timeout=(10, 60)
            )

            if upstream.status_code != 200:
                details = upstream.text[:2000]
                logging.error(
                    "Gemini HTTP %s: %s",
                    upstream.status_code,
                    details
                )
                yield "[TENX_ERROR]" + json.dumps({
                    "error": f"Gemini returned HTTP {upstream.status_code}.",
                    "details": details
                }, ensure_ascii=False)
                return

            for line in upstream.iter_lines(decode_unicode=False):
                if not line:
                    continue

                if isinstance(line, bytes):
                    line = line.decode("utf-8", errors="replace")

                line = line.strip()
                if not line.startswith("data:"):
                    continue

                raw = line[5:].strip()
                if not raw or raw == "[DONE]":
                    continue

                try:
                    chunk = json.loads(raw)
                except json.JSONDecodeError:
                    logging.warning("Invalid JSON chunk from Gemini.")
                    continue

                for candidate in chunk.get("candidates", []):
                    content = candidate.get("content", {})
                    for part in content.get("parts", []):
                        text = part.get("text", "")
                        if text:
                            yield text

        except requests.Timeout:
            logging.exception("Gemini request timed out.")
            yield "[TENX_ERROR]" + json.dumps({
                "error": "The AI service took too long to respond."
            }, ensure_ascii=False)

        except requests.RequestException:
            logging.exception("Network error contacting Gemini.")
            yield "[TENX_ERROR]" + json.dumps({
                "error": "Could not connect to the AI service."
            }, ensure_ascii=False)

        except Exception:
            logging.exception("Unexpected error in streaming response.")
            yield "[TENX_ERROR]" + json.dumps({
                "error": "An unexpected server error occurred."
            }, ensure_ascii=False)

        finally:
            if upstream is not None:
                upstream.close()

    return Response(
        generate(),
        content_type="text/plain; charset=utf-8",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no"
        }
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "5000"))
    )
