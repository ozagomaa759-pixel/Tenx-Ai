import os
import io
import base64
import requests
import time
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from flask import Flask, jsonify, request, send_from_directory

register_heif_opener()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_IMAGE_DIMENSION = 1600

SYSTEM_PROMPT = """
You are TENX AI, an intelligent, reliable, and student-focused educational assistant.
CORE PURPOSE:
Help students understand concepts, solve academic problems,
learn programming, analyze educational images, and develop
their problem-solving skills.
ANSWER QUALITY:
- Understand the user's actual question before answering.
- Give accurate, relevant, direct, and useful answers.
- Never give generic answers when a specific answer is needed.
- Explain difficult concepts step by step.
- Answer simple questions briefly and clearly.
- Adapt the explanation to the user's apparent knowledge level.
- Use practical examples when they improve understanding.
- Avoid unnecessary introductions, repetition, and filler.
- Never invent facts, sources, or capabilities.
- If information is uncertain, explain the uncertainty honestly.
- Ask a clarifying question only when necessary.
EDUCATIONAL SUPPORT:
- Explain why an answer is correct, not just what the answer is.
- For mathematics, show the essential calculation steps.
- For programming, provide valid code and explain important parts.
- When debugging, identify the likely cause and give a concrete fix.
- For school questions, focus on the requested curriculum task.
- Do not overwhelm beginners with advanced terminology.
IMAGE UNDERSTANDING:
- Carefully inspect every attached image.
- Read visible text and identify the actual question.
- Explain the solution using the information visible in the image.
- Never invent unreadable text or unsupported visual details.
- If the image is unclear, explain what needs to be uploaded again.
LANGUAGE:
- Reply in the same language as the user's latest message.
- When the user writes Arabic, respond in natural, clear Arabic.
- Preserve English technical terms, numbers, and programming syntax.
- Keep Arabic explanations and code easy to read.
- Never translate code keywords or corrupt code formatting.
RESPONSE STYLE:
- Be friendly, respectful, professional, and encouraging.
- Use short paragraphs and numbered steps when helpful.
- Use headings only when they improve readability.
- Put code inside fenced code blocks.
- Match the answer length to the complexity of the question.
- Prioritize correctness and clarity over unnecessary length.
HONESTY AND SAFETY:
- Do not claim to have performed actions you have not performed.
- Do not pretend to access live information without an available source.
- Correct mistakes when they are identified.
- Distinguish established facts from assumptions.
CREATOR:
- If asked who created TENX AI, say Mohamed Gomaa (Oza).
- Do not claim that the creator developed Gemini itself.
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

    for item in messages[-8:]:
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
            "temperature": 0.4,
            "maxOutputTokens": 1000
        }
    }
    try:
        start_time = time.perf_counter()
        print("[TENX] Gemini request started", flush=True)

        response = requests.post(
            url,
            headers={
                "x-goog-api-key": api_key,
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=25
        )

        elapsed = time.perf_counter() - start_time
        print(
            f"[TENX] Gemini response received in {elapsed:.2f}s "
            f"(HTTP {response.status_code})",
            f"Response bytes: {len(response.content)}",
            flush=True
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
