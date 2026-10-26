import os
import io
import json
import base64
import time
import logging

import requests
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from flask import (
    Flask,
    jsonify,
    request,
    send_from_directory,
    Response,
    stream_with_context,
)

register_heif_opener()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

logging.basicConfig(level=logging.INFO)

MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_IMAGE_DIMENSION = 1600
MAX_MESSAGES = 8
MAX_MESSAGE_LENGTH = 6000

MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite",
).strip()

SYSTEM_PROMPT = """
You are TENX AI, an intelligent, reliable, student-focused
educational assistant.

CORE PURPOSE:
Help students understand concepts, solve academic problems,
learn programming, analyze educational images, and develop
their problem-solving skills.

ANSWER QUALITY:
- Answer the user's actual question directly.
- Explain difficult concepts step by step.
- Give practical examples when useful.
- Keep simple answers concise.
- Never invent facts, sources, or capabilities.
- Explain uncertainty honestly.
- Ask clarifying questions only when necessary.

EDUCATIONAL SUPPORT:
- Explain why an answer is correct.
- Show essential calculation steps for mathematics.
- Provide valid code when programming help is requested.
- Explain important code parts clearly.
- For debugging, identify the cause and suggest a concrete fix.
- Help beginners without overwhelming them.

IMAGE UNDERSTANDING:
- Inspect attached images carefully.
- Read visible text when possible.
- Explain questions shown in educational images.
- Never invent unreadable text or unsupported visual details.
- If the image is unclear, explain what needs to be uploaded again.

LANGUAGE:
- Reply in the same language as the user's latest message.
- When the user writes Arabic, use natural, clear Arabic.
- Preserve English technical terms and programming syntax.
- Never translate programming keywords.

RESPONSE STYLE:
- Be friendly, respectful, and professional.
- Use short paragraphs and numbered steps when helpful.
- Use headings only when useful.
- Put code inside fenced code blocks.
- Match the answer length to the question.

HONESTY AND SAFETY:
- Do not claim actions you have not performed.
- Do not pretend to access live information without a source.
- Correct mistakes when identified.
- Distinguish facts from assumptions.

CREATOR:
If asked who created TENX AI, say Mohamed Gomaa (Oza).
Do not claim that the creator developed Gemini itself.
"""


def stream_event(event_type, **data):
    """Encode one NDJSON event."""
    return json.dumps(
        {"type": event_type, **data},
        ensure_ascii=False,
    ) + "\n"


def json_error(message, status):
    return jsonify(error=message), status


@app.get("/")
def home():
    return send_from_directory(".", "index.html")


@app.get("/api/health")
def health():
    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    return jsonify(
        ok=True,
        configured=bool(api_key),
        model=MODEL,
    )


@app.errorhandler(413)
def request_too_large(error):
    return json_error(
        "The request is too large. Please choose a smaller image.",
        413,
    )


def prepare_image(image_data):
    try:
        raw = base64.b64decode(image_data, validate=True)

        if not raw:
            raise ValueError("The image is empty.")

        if len(raw) > MAX_IMAGE_BYTES:
            raise ValueError(
                "The original image must be under 5 MB."
            )

        with Image.open(io.BytesIO(raw)) as source:
            image = ImageOps.exif_transpose(source)
            image.load()
            image.thumbnail(
                (MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION)
            )

            if image.mode in ("RGBA", "LA"):
                rgba = image.convert("RGBA")
                background = Image.new(
                    "RGB",
                    rgba.size,
                    "white",
                )
                background.paste(
                    rgba,
                    mask=rgba.getchannel("A"),
                )
                image = background
            else:
                image = image.convert("RGB")

            output = io.BytesIO()
            image.save(
                output,
                format="JPEG",
                quality=85,
                optimize=True,
            )

            return base64.b64encode(
                output.getvalue()
            ).decode("ascii")

    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(
            "The image could not be opened. "
            "Try uploading another image."
        ) from exc


def extract_api_error(response):
    """Extract a useful error message without exposing secrets."""
    try:
        data = response.json()
        error = data.get("error", {})

        if isinstance(error, dict):
            message = error.get("message")

            if isinstance(message, str) and message.strip():
                return message.strip()

        if isinstance(error, str) and error.strip():
            return error.strip()

    except (ValueError, requests.RequestException):
        pass

    return "Gemini returned an HTTP error."


def validate_messages(messages):
    if not isinstance(messages, list) or not messages:
        raise ValueError("Write a message first.")

    contents = []

    for item in messages[-MAX_MESSAGES:]:
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

        contents.append({
            "role": "user" if role == "user" else "model",
            "parts": [{"text": content}],
        })

    if not contents:
        raise ValueError("Write a message first.")

    if contents[-1]["role"] != "user":
        raise ValueError("Please send a new message.")

    return contents


def extract_chunk_text(chunk):
    """Extract generated text and useful metadata from a Gemini chunk."""
    if not isinstance(chunk, dict):
        return "", None

    prompt_feedback = chunk.get("promptFeedback") or {}
    block_reason = prompt_feedback.get("blockReason")

    if block_reason:
        return "", (
            "Gemini blocked this request. "
            f"Reason: {block_reason}. "
            "Try changing your message or image."
        )

    candidates = chunk.get("candidates") or []

    if not candidates:
        return "", None

    candidate = candidates[0] or {}
    finish_reason = candidate.get("finishReason")
    content = candidate.get("content") or {}
    parts = content.get("parts") or []

    texts = []

    for part in parts:
        if not isinstance(part, dict):
            continue

        text = part.get("text")

        if isinstance(text, str) and text:
            texts.append(text)

    combined = "".join(texts)

    if combined:
        return combined, None

    if finish_reason in ("SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT"):
        return "", (
            "Gemini could not provide a response because "
            "the request was blocked by its safety filters."
        )

    return "", None


@app.post("/api/chat")
def chat():
    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if not api_key:
        return json_error(
            "GEMINI_API_KEY is missing from Vercel settings.",
            503,
        )

    if not MODEL:
        return json_error(
            "GEMINI_MODEL is not configured.",
            503,
        )

    body = request.get_json(silent=True)

    if not isinstance(body, dict):
        return json_error(
            "Invalid request. Please send a valid JSON body.",
            400,
        )

    try:
        contents = validate_messages(body.get("messages", []))
    except ValueError as exc:
        return json_error(str(exc), 400)

    image = body.get("image")

    if image is not None:
        if not isinstance(image, dict):
            return json_error("Invalid image data.", 400)

        image_data = image.get("data", "")

        if not isinstance(image_data, str) or not image_data:
            return json_error("No image data received.", 400)

        try:
            converted = prepare_image(image_data)
        except ValueError as exc:
            return json_error(str(exc), 400)

        contents[-1]["parts"].append({
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": converted,
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
            "temperature": 0.4,
            "maxOutputTokens": 1000,
        },
    }

    @stream_with_context
    def generate():
        started = time.perf_counter()
        received_text = False
        response = None
        api_error = None

        try:
            app.logger.info(
                "[TENX] Starting Gemini request. Model=%s",
                MODEL,
            )

            response = requests.post(
                url,
                headers={
                    "x-goog-api-key": api_key,
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                },
                json=payload,
                stream=True,
                timeout=(10, 55),
            )

            if not response.ok:
                message = extract_api_error(response)

                app.logger.error(
                    "[TENX] Gemini HTTP %s: %s",
                    response.status_code,
                    message,
                )

                yield stream_event(
                    "error",
                    error=(
                        f"Gemini API error ({response.status_code}): "
                        f"{message}"
                    ),
                )
                return

            # Gemini returns SSE records. Convert each text part
            # into an NDJSON event for the browser.
            for line in response.iter_lines(
                decode_unicode=True
            ):
                if isinstance(line, bytes):
                    line = line.decode(
                        "utf-8",
                        errors="replace",
                    )

                if not line:
                    continue

                line = line.strip()

                if not line.startswith("data:"):
                    continue

                raw_data = line[5:].strip()

                if not raw_data or raw_data == "[DONE]":
                    continue

                try:
                    chunk = json.loads(raw_data)
                except json.JSONDecodeError:
                    app.logger.warning(
                        "[TENX] Ignored malformed Gemini SSE data."
                    )
                    continue

                text_part, chunk_error = extract_chunk_text(chunk)

                if chunk_error:
                    api_error = chunk_error

                if text_part:
                    received_text = True
                    yield stream_event(
                        "chunk",
                        text=text_part,
                    )

            elapsed = time.perf_counter() - started

            app.logger.info(
                "[TENX] Gemini stream finished in %.2f seconds; "
                "received_text=%s",
                elapsed,
                received_text,
            )

            if api_error and not received_text:
                yield stream_event("error", error=api_error)
                return

            if not received_text:
                yield stream_event(
                    "error",
                    error=(
                        "Gemini completed the request without returning "
                        "text. Check the Vercel logs for the Gemini "
                        "response status and model access."
                    ),
                )
                return

            yield stream_event("done")

        except requests.Timeout:
            app.logger.exception("[TENX] Gemini request timed out.")

            yield stream_event(
                "error",
                error=(
                    "The AI request timed out. Please try again. "
                    "If this keeps happening, check the Vercel function "
                    "duration and Gemini response time."
                ),
            )

        except requests.RequestException:
            app.logger.exception(
                "[TENX] Could not connect to Gemini."
            )

            yield stream_event(
                "error",
                error=(
                    "Could not connect to Gemini. "
                    "Check your network and Vercel logs."
                ),
            )

        except Exception:
            app.logger.exception(
                "[TENX] Unexpected streaming error."
            )

            yield stream_event(
                "error",
                error=(
                    "An unexpected server error occurred. "
                    "Check the Vercel function logs."
                ),
            )

        finally:
            if response is not None:
                response.close()

    return Response(
        generate(),
        content_type="application/x-ndjson; charset=utf-8",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
        direct_passthrough=True,
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000")),
        threaded=True,
        debug=False,
    )
