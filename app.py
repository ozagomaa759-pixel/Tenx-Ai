
import os
import io
import json
import base64
import logging

import requests
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from flask import Flask, jsonify, request, send_from_directory

register_heif_opener()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

logging.basicConfig(level=logging.INFO)

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip()
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_IMAGE_DIMENSION = 1600
MAX_MESSAGES = 8
MAX_MESSAGE_LENGTH = 6000

SYSTEM_PROMPT = """
You are TENX AI, a reliable educational assistant.
Help students understand school subjects, mathematics,
programming, artificial intelligence, and educational images.
Explain difficult topics step by step and keep simple answers concise.
Never invent facts or claim capabilities you do not have.
Reply in the same language as the user's latest message.
Use clear, friendly language suitable for students.
If asked who created TENX AI, say Mohamed Gomaa (Oza).
Do not claim the creator developed Gemini itself.
"""


def error(message, status=400):
    return jsonify(error=message), status


@app.get("/")
def home():
    return send_from_directory(".", "index.html")


@app.get("/api/health")
def health():
    return jsonify(
        ok=True,
        configured=bool(os.getenv("GEMINI_API_KEY", "").strip()),
        model=MODEL,
    )


@app.errorhandler(413)
def too_large(exc):
    return error("The request is too large. Choose a smaller image.", 413)


def prepare_image(data):
    try:
        raw = base64.b64decode(data, validate=True)

        if not raw or len(raw) > MAX_IMAGE_BYTES:
            raise ValueError("The image must be under 5 MB.")

        with Image.open(io.BytesIO(raw)) as source:
            image = ImageOps.exif_transpose(source)
            image.load()
            image.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION))

            if image.mode in ("RGBA", "LA"):
                rgba = image.convert("RGBA")
                background = Image.new("RGB", rgba.size, "white")
                background.paste(rgba, mask=rgba.getchannel("A"))
                image = background
            else:
                image = image.convert("RGB")

            output = io.BytesIO()
            image.save(output, format="JPEG", quality=85, optimize=True)

            return base64.b64encode(output.getvalue()).decode("ascii")

    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(
            "Could not open that image. Try another image."
        ) from exc


def validate_messages(messages):
    if not isinstance(messages, list) or not messages:
        raise ValueError("Write a message first.")

    contents = []

    for item in messages[-MAX_MESSAGES:]:
        if not isinstance(item, dict):
            continue

        role = item.get("role")
        text = item.get("content")

        if role not in ("user", "assistant") or not isinstance(text, str):
            continue

        text = text.strip()[:MAX_MESSAGE_LENGTH]
        if not text:
            continue

        contents.append({
            "role": "user" if role == "user" else "model",
            "parts": [{"text": text}],
        })

    if not contents or contents[-1]["role"] != "user":
        raise ValueError("Please send a new message.")

    return contents


def get_error_message(response):
    try:
        data = response.json()
        details = data.get("error", {})
        if isinstance(details, dict):
            message = details.get("message")
            if isinstance(message, str) and message.strip():
                return message.strip()
    except (ValueError, requests.RequestException):
        pass

    return "Google returned an API error. Check the Vercel logs."


@app.post("/api/chat")
def chat():
    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if not api_key:
        return error("GEMINI_API_KEY is missing from Vercel settings.", 503)

    if not MODEL:
        return error("GEMINI_MODEL is not configured.", 503)

    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return error("Invalid JSON request.", 400)

    try:
        contents = validate_messages(body.get("messages", []))
    except ValueError as exc:
        return error(str(exc), 400)

    image = body.get("image")

    if image is not None:
        if not isinstance(image, dict):
            return error("Invalid image data.", 400)

        image_data = image.get("data", "")
        if not isinstance(image_data, str) or not image_data:
            return error("No image data received.", 400)

        try:
            converted = prepare_image(image_data)
        except ValueError as exc:
            return error(str(exc), 400)

        contents[-1]["parts"].append({
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": converted,
            }
        })

    url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        f"models/{MODEL}:generateContent"
    )

    payload = {
        "systemInstruction": {
            "parts": [{"text": SYSTEM_PROMPT}]
        },
        "contents": contents,
        "generationConfig": {
            "temperature": 0.4,
            "maxOutputTokens": 1000,
        },
    }

    try:
        app.logger.info("[TENX] Sending request to Gemini model=%s", MODEL)

        response = requests.post(
            url,
            headers={
                "x-goog-api-key": api_key,
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=(10, 55),
        )

        if not response.ok:
            message = get_error_message(response)
            app.logger.error(
                "[TENX] Gemini HTTP status=%s message=%s",
                response.status_code,
                message,
            )
            return error(
                f"Gemini API error ({response.status_code}): {message}",
                502,
            )

        try:
            data = response.json()
        except ValueError:
            app.logger.error("[TENX] Gemini returned invalid JSON.")
            return error("Gemini returned an invalid response.", 502)

        feedback = data.get("promptFeedback") or {}
        if feedback.get("blockReason"):
            reason = feedback["blockReason"]
            app.logger.warning("[TENX] Request blocked: %s", reason)
            return error(
                f"Gemini blocked this request ({reason}). Try another prompt.",
                422,
            )

        candidates = data.get("candidates") or []
        if not candidates:
            app.logger.error(
                "[TENX] No candidates. Response keys=%s",
                list(data.keys()),
            )
            return error(
                "Gemini returned no answer. Check model access and Vercel logs.",
                502,
            )

        candidate = candidates[0] or {}
        content = candidate.get("content") or {}
        parts = content.get("parts") or []

        answer_parts = [
            part["text"]
            for part in parts
            if isinstance(part, dict)
            and isinstance(part.get("text"), str)
            and part["text"].strip()
            and not part.get("thought", False)
        ]

        answer = "".join(answer_parts).strip()

        if not answer:
            finish_reason = candidate.get("finishReason", "unknown")
            app.logger.error(
                "[TENX] Empty answer. finishReason=%s candidateKeys=%s",
                finish_reason,
                list(candidate.keys()),
            )
            return error(
                "Gemini returned no visible text "
                f"(finish reason: {finish_reason}). Try again or check logs.",
                502,
            )

        app.logger.info(
            "[TENX] Successful response. chars=%d",
            len(answer),
        )

        return jsonify(reply=answer)

    except requests.Timeout:
        app.logger.exception("[TENX] Gemini request timed out.")
        return error("Gemini timed out. Please try again.", 504)

    except requests.RequestException:
        app.logger.exception("[TENX] Gemini connection failed.")
        return error("Could not connect to Gemini. Check Vercel logs.", 502)

    except Exception:
        app.logger.exception("[TENX] Unexpected server error.")
        return error("Unexpected server error. Check Vercel logs.", 500)


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000")),
        debug=False,
    )
