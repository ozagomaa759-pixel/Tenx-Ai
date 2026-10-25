
import os
import io
import base64
import requests
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from flask import Flask, jsonify, request, send_from_directory

register_heif_opener()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_IMAGE_DIMENSION = 1600

SYSTEM_PROMPT = """
You are TENX AI, an educational AI assistant for students.

ACCURACY:
- Give accurate, relevant, clear answers.
- Never invent facts. State uncertainty honestly.
- Explain step by step when useful.
- Answer simple questions directly.
- Explain code and use correct examples.
- When an image is attached, analyze its contents carefully.
- Read school questions from images and explain the solution.
- If text is blurry or unreadable, say so honestly.

LANGUAGE:
- Reply in the same language as the user.
- Use natural, clear Arabic when the user writes Arabic.
- Keep English technical terms correctly spelled.
- Never reverse letters, numbers, words, or code.

FORMATTING:
- Use plain text by default.
- Use short paragraphs and numbered steps when helpful.
- Put programming code in fenced code blocks.
- Keep math and technical terms readable.

PERSONALITY:
- Be respectful, encouraging, and professional.
- Teach students clearly without unnecessary detail.

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
        error="The request is too large. Please choose a smaller image."
    ), 413


def prepare_image(image_data):
    try:
        raw = base64.b64decode(image_data, validate=True)

        if not raw:
            raise ValueError("The image is empty.")

        if len(raw) > MAX_IMAGE_BYTES:
            raise ValueError("The original image must be under 5 MB.")

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
                    rgba,
                    mask=rgba.getchannel("A")
                )
                image = background
            else:
                image = image.convert("RGB")

            output = io.BytesIO()
            image.save(
                output,
                format="JPEG",
                quality=85,
                optimize=True
            )

            return base64.b64encode(
                output.getvalue()
            ).decode("ascii")

    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(
            "The image could not be opened. Try another image."
        ) from exc


@app.post("/api/chat")
def chat():
    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if not api_key:
        return jsonify(
            error="GEMINI_API_KEY is missing from Vercel settings."
        ), 503

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

        if role not in ("user", "assistant"):
            continue
        if not isinstance(content, str):
            continue

        content = content.strip()[:6000]
        if content:
            contents.append({
                "role": "user" if role == "user" else "model",
                "parts": [{"text": content}]
            })

    if not contents or contents[-1]["role"] != "user":
        return jsonify(error="Please send a new message."), 400

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

    model = "gemini-3.5-flash-lite"
    url = (
        "https://generativelanguage.googleapis.com/"
        f"v1beta/models/{model}:generateContent"
    )

    payload = {
        "systemInstruction": {
            "parts": [{"text": SYSTEM_PROMPT}]
        },
        "contents": contents,
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 1500
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
            timeout=25
        )

        try:
            data = response.json()
        except ValueError:
            data = {}

        if not response.ok:
            message = data.get("error", {}).get(
                "message",
                "Gemini request failed."
            )
            return jsonify(
                error=f"Gemini API error ({response.status_code}): {message}"
            ), 502

        candidates = data.get("candidates", [])
        parts = (
            candidates[0].get("content", {}).get("parts", [])
            if candidates else []
        )

        answer = "".join(
            part.get("text", "")
            for part in parts
            if isinstance(part, dict)
            and isinstance(part.get("text"), str)
        ).strip()

        if not answer:
            return jsonify(
                error="Gemini returned an empty response. Please try again."
            ), 502

        return jsonify(reply=answer)

    except requests.Timeout:
        return jsonify(
            error="The AI took too long to respond. Please try again."
        ), 504
    except requests.RequestException:
        return jsonify(
            error="Could not connect to Gemini. Please try again."
        ), 502


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
