
import os
import json
import time
import logging
import requests

from flask import Flask, request, Response, jsonify, send_from_directory

app = Flask(__name__, static_folder=None)
logging.basicConfig(level=logging.INFO)

API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
MAX_HISTORY = 8
MAX_MESSAGE_LENGTH = 4000

SYSTEM_PROMPT = """
You are TENX AI, a helpful educational AI assistant created for students.

Rules:
- Reply in the same language as the user's latest message.
- If the user writes Arabic, answer in clear, natural Arabic.
- If the user writes Egyptian Arabic, you may reply in friendly Egyptian Arabic.
- Preserve correct Unicode text. Never intentionally corrupt Arabic characters.
- Explain programming, science, mathematics, and technology step by step.
- For programming requests, provide accurate code and explain it simply.
- If an image is provided, inspect it and answer questions about its visible content.
- If you are uncertain, say so honestly.
- Be concise for simple questions and detailed for complex questions.
"""

session = requests.Session()


def repair_mojibake(text):
    """Repair common UTF-8 text accidentally decoded as Latin-1/Windows-1252."""
    if not isinstance(text, str) or not text:
        return text

    # Repair only when the text strongly resembles encoding corruption.
    suspicious = ("Ã", "Â", "Ø", "Ù", "Ð", "Ñ")
    if any(marker in text for marker in suspicious):
        try:
            repaired = text.encode("latin-1").decode("utf-8")
            if repaired and repaired != text:
                return repaired
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass

        try:
            repaired = text.encode("cp1252").decode("utf-8")
            if repaired and repaired != text:
                return repaired
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass

    return text


def build_contents(messages, image=None):
    contents = []

    for message in messages[-MAX_HISTORY:]:
        role = message.get("role", "user")
        text = message.get("content", message.get("text", ""))

        if not isinstance(text, str):
            continue

        text = text[:MAX_MESSAGE_LENGTH]
        text = repair_mojibake(text)

        if role not in ("user", "assistant"):
            continue

        gemini_role = "model" if role == "assistant" else "user"
        contents.append({
            "role": gemini_role,
            "parts": [{"text": text}]
        })

    # Attach the image to the most recent user message.
    if image and isinstance(image, dict):
        mime_type = image.get("mimeType", "")
        image_data = image.get("data", "")

        allowed_types = {
            "image/jpeg",
            "image/png",
            "image/webp",
            "image/heic",
            "image/heif",
        }

        if (
            mime_type in allowed_types
            and isinstance(image_data, str)
            and image_data
            and len(image_data) <= 12_000_000
        ):
            if not contents or contents[-1]["role"] != "user":
                contents.append({
                    "role": "user",
                    "parts": []
                })

            contents[-1]["parts"].append({
                "inline_data": {
                    "mime_type": mime_type,
                    "data": image_data
                }
            })

    return contents


@app.route("/", methods=["GET"])
def home():
    return send_from_directory("public", "index.html")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model": MODEL,
        "api_key_configured": bool(API_KEY)
    })


@app.route("/api/chat", methods=["POST"])
def chat():
    if not API_KEY:
        return jsonify({
            "error": "GEMINI_API_KEY is missing in Vercel environment variables."
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
                error_text = upstream.text[:3000]
                logging.error(
                    "Gemini API error %s: %s",
                    upstream.status_code,
                    error_text
                )
                yield "\n[ TENX_ERROR ] " + json.dumps({
                    "error": f"Gemini returned HTTP {upstream.status_code}.",
                    "details": error_text
                }, ensure_ascii=False)
                return

            # Explicit UTF-8 decoding for the upstream SSE stream.
            upstream.encoding = "utf-8"

            for line in upstream.iter_lines(decode_unicode=True):
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
                    logging.warning("Could not parse Gemini stream chunk.")
                    continue

                candidates = chunk.get("candidates", [])
                for candidate in candidates:
                    content = candidate.get("content", {})
                    for part in content.get("parts", []):
                        text = part.get("text", "")
                        if text:
                            text = repair_mojibake(text)
                            yield text

        except requests.Timeout:
            logging.exception("Gemini request timed out.")
            yield "\n[ TENX_ERROR ] The AI service took too long to respond."
        except requests.RequestException:
            logging.exception("Network error calling Gemini.")
            yield "\n[ TENX_ERROR ] Could not connect to the AI service."
        except Exception:
            logging.exception("Unexpected streaming error.")
            yield "\n[ TENX_ERROR ] An unexpected server error occurred."
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
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
