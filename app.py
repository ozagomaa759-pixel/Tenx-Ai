
import os
import io
import json
import time
import base64
import logging
import requests

from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from flask import (
    Flask, jsonify, request, send_from_directory,
    Response, stream_with_context
)

register_heif_opener()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
logging.basicConfig(level=logging.INFO)

MODEL = "gemini-3.5-flash-lite"
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_IMAGE_DIMENSION = 1280
MAX_HISTORY_MESSAGES = 8
MAX_MESSAGE_LENGTH = 4000

session = requests.Session()

SYSTEM_PROMPT = """
You are TENX AI, an educational AI assistant for students.

ACCURACY:
- Give accurate, relevant, clear answers.
- Never invent facts. Admit uncertainty when necessary.
- Answer simple questions directly and briefly.
- Explain difficult questions step by step when useful.
- Provide correct, runnable code when requested.
- Analyze attached images and school questions carefully.
- If an image is unclear, say what cannot be read.

LANGUAGE:
- Reply in the same language as the user.
- Use natural, clear Arabic for Arabic messages.
- Preserve English technical terms correctly.
- Never alter letters, numbers, or code.

FORMATTING:
- Use short paragraphs and numbered steps when useful.
- Put programming code in fenced code blocks.
- Avoid unnecessary introductions and repetition.

PERSONALITY:
- Be respectful, encouraging, and professional.
- Help students understand their work.

CREATOR:
- If asked who created TENX AI, say Mohamed Gomaa (Oza).
- Do not claim its creator developed Gemini itself.
"""


@app.get("/")
def home():
    return send_from_directory(".", "index.html")


@app.get("/api/health")
def health():
    return jsonify(
        ok=True,
        configured=bool(os.getenv("GEMINI_API_KEY"))
    )


@app.errorhandler(413)
def request_too_large(error):
    return jsonify(
        error="Request too large. Please choose a smaller image."
    ), 413


def prepare_image(image_data):
    try:
        raw = base64.b64decode(image_data, validate=True)

        if not raw:
            raise ValueError("The image is empty.")

        if len(raw) > MAX_IMAGE_BYTES:
            raise ValueError("The image must be under 5 MB.")

        with Image.open(io.BytesIO(raw)) as source:
            image = ImageOps.exif_transpose(source)
            image.load()
            image.thumbnail(
                (MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION)
            )

            if image.mode in ("RGBA", "LA"):
                rgba = image.convert("RGBA")
                background = Image.new("RGB", rgba.size, "white")
                background.paste(
                    rgba, mask=rgba.getchannel("A")
                )
                image = background
            else:
                image = image.convert("RGB")

            output = io.BytesIO()
            image.save(
                output, format="JPEG",
                quality=82, optimize=True
            )

            return base64.b64encode(
                output.getvalue()
            ).decode("ascii")

    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(
            "Could not read this image. Try another image."
        ) from exc


@app.post("/api/chat")
def chat():
    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if not api_key:
        return jsonify(
            error="GEMINI_API_KEY is missing from Vercel settings."
        ), 503

    body = request.get_json(silent=True)

    if not isinstance(body, dict):
        return jsonify(error="Invalid request data."), 400

    messages = body.get("messages")

    if not isinstance(messages, list) or not messages:
        return jsonify(error="Write a message first."), 400

    contents = []

    for item in messages[-MAX_HISTORY_MESSAGES:]:
        if not isinstance(item, dict):
            continue

        role = item.get("role")
        content = item.get("content")

        if role not in ("user", "assistant"):
            continue
        if not isinstance(content, str):
            continue

        content = content.strip()[:MAX_MESSAGE_LENGTH]
        if not content:
            continue

        api_role = "user" if role == "user" else "model"

        if contents and contents[-1]["role"] == api_role:
            contents[-1]["parts"][0]["text"] += "\n" + content
        else:
            contents.append({
                "role": api_role,
                "parts": [{"text": content}]
            })

    if not contents or contents[-1]["role"] != "user":
        return jsonify(
            error="Please send a new message."
        ), 400

    image = body.get("image")

    if image is not None:
        if not isinstance(image, dict):
            return jsonify(error="Invalid image data."), 400

        image_data = image.get("data", "")

        if not isinstance(image_data, str) or not image_data:
            return jsonify(error="No image data received."), 400

        try:
            converted = prepare_image(image_data)
        except ValueError as exc:
            return jsonify(error=str(exc)), 400

        contents[-1]["parts"].append({
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": converted
            }
        })

    url = (
        "https://generativelanguage.googleapis.com/"
        f"v1beta/models/{MODEL}:streamGenerateContent?alt=sse"
    )

    payload = {
        "systemInstruction": {
            "parts": [{"text": SYSTEM_PROMPT}]
        },
        "contents": contents,
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 1000
        }
    }

    started = time.perf_counter()

    try:
        upstream = session.post(
            url,
            headers={
                "x-goog-api-key": api_key,
                "Content-Type": "application/json"
            },
            json=payload,
            stream=True,
            timeout=(5, 25)
        )

    except requests.Timeout:
        return jsonify(
            error="Gemini took too long to start responding."
        ), 504
    except requests.RequestException:
        app.logger.exception("Gemini connection failed")
        return jsonify(
            error="Could not connect to Gemini."
        ), 502

    if not upstream.ok:
        try:
            data = upstream.json()
        except ValueError:
            data = {}

        message = data.get("error", {}).get(
            "message", "Gemini request failed."
        )
        status = upstream.status_code
        upstream.close()

        app.logger.warning(
            "Gemini HTTP %s: %s", status, message[:300]
        )

        return jsonify(
            error=f"Gemini API error ({status}): {message}"
        ), 502

    @stream_with_context
    def generate():
        first_chunk = True

        try:
            for line in upstream.iter_lines(decode_unicode=True):
                if not line:
                    continue

                if isinstance(line, bytes):
                    line = line.decode("utf-8", errors="replace")

                if not line.startswith("data:"):
                    continue

                raw_event = line[5:].strip()

                if not raw_event or raw_event == "[DONE]":
                    continue

                try:
                    event = json.loads(raw_event)
                except json.JSONDecodeError:
                    continue

                candidates = event.get("candidates", [])
                if not candidates:
                    continue

                parts = (
                    candidates[0]
                    .get("content", {})
                    .get("parts", [])
                )

                for part in parts:
                    text = part.get("text")

                    if isinstance(text, str) and text:
                        if first_chunk:
                            elapsed = round(
                                time.perf_counter() - started, 2
                            )
                            app.logger.info(
                                "TENX first Gemini chunk: %.2fs",
                                elapsed
                            )
                            first_chunk = False

                        yield text

        except requests.RequestException:
            app.logger.exception("Gemini stream interrupted")
        finally:
            upstream.close()
            app.logger.info(
                "TENX stream finished in %.2fs",
                time.perf_counter() - started
            )

    return Response(
        generate(),
        mimetype="text/plain",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
