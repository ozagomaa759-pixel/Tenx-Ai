import os
import io
import base64
import html
import logging
import requests

from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from flask import Flask, jsonify, request, send_from_directory, send_file
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_RIGHT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, KeepTogether
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    ARABIC_SUPPORT = True
except ImportError:
    ARABIC_SUPPORT = False

register_heif_opener()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tenx-ai")

MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_IMAGE_DIMENSION = 1600
CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.5-flash-lite")
IMAGE_MODEL = os.getenv("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models/"


SYSTEM_PROMPT = """
You are TENX AI, an educational AI assistant for students.
Give accurate, clear, relevant answers. Never invent facts.
Explain step by step when useful. Analyze attached images carefully.
If text is blurry or unreadable, say so honestly.
Reply in the same language as the user. Use natural, clear Arabic
when the user writes Arabic. Keep code correctly formatted in fenced blocks.
Be respectful, encouraging, and professional.
If asked who created TENX AI, say Mohamed Gomaa (Oza).
Do not claim the creator developed Gemini itself.

For arithmetic and mathematical calculations, use simple, readable
plain text whenever possible. Write expressions like 5 + 10 = 15.
Avoid LaTeX commands such as \text{}, dollar-sign math delimiters,
and unnecessary mathematical symbols. Never wrap simple calculations
in LaTeX. Keep the answer natural and easy to read.
"""



@app.get("/")
def home():
    return send_from_directory(".", "index.html")


def api_key():
    return os.getenv("GEMINI_API_KEY", "").strip()


@app.get("/api/health")
def health():
    return jsonify(ok=True, configured=bool(api_key()))


@app.errorhandler(413)
def too_large(_error):
    return jsonify(error="Ø§ÙÙÙÙ ÙØ¨ÙØ± Ø¬Ø¯ÙØ§. Ø¬Ø±ÙØ¨ ÙÙÙÙØ§ Ø£ØµØºØ±."), 413


def prepare_image(encoded):
    if not isinstance(encoded, str) or not encoded:
        raise ValueError("ÙÙ ÙØªÙ Ø§Ø³ØªÙØ§Ù Ø¨ÙØ§ÙØ§Øª Ø§ÙØµÙØ±Ø©.")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise ValueError("Ø¨ÙØ§ÙØ§Øª Ø§ÙØµÙØ±Ø© ØºÙØ± ØµØ§ÙØ­Ø©.") from exc
    if not raw:
        raise ValueError("Ø§ÙØµÙØ±Ø© ÙØ§Ø±ØºØ©.")
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("Ø­Ø¬Ù Ø§ÙØµÙØ±Ø© ÙØ¬Ø¨ Ø£ÙØ§ ÙØªØ¬Ø§ÙØ² 5 ÙÙØ¬Ø§Ø¨Ø§ÙØª.")
    try:
        with Image.open(io.BytesIO(raw)) as source:
            image = ImageOps.exif_transpose(source)
            image.load()
            image.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION))
            if image.mode in ("RGBA", "LA") or "transparency" in image.info:
                rgba = image.convert("RGBA")
                bg = Image.new("RGB", rgba.size, "white")
                bg.paste(rgba, mask=rgba.getchannel("A"))
                image = bg
            else:
                image = image.convert("RGB")
            out = io.BytesIO()
            image.save(out, format="JPEG", quality=85, optimize=True)
            return base64.b64encode(out.getvalue()).decode("ascii")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("ØªØ¹Ø°ÙØ± ÙØªØ­ Ø§ÙØµÙØ±Ø©. Ø¬Ø±ÙØ¨ ØµÙØ±Ø© Ø¨ØµÙØºØ© JPG Ø£Ù PNG.") from exc


def gemini_request(model, payload, timeout=35):
    key = api_key()
    if not key:
        raise RuntimeError("ÙÙØªØ§Ø­ GEMINI_API_KEY ØºÙØ± ÙØ¶Ø¨ÙØ· ÙÙ Ø¥Ø¹Ø¯Ø§Ø¯Ø§Øª Vercel.")
    url = f"{GEMINI_API_BASE}{model}:generateContent"
    response = requests.post(
        url,
        headers={"x-goog-api-key": key, "Content-Type": "application/json"},
        json=payload,
        timeout=(8, timeout),
    )
    try:
        data = response.json()
    except ValueError:
        data = {}
    if not response.ok:
        api_error = data.get("error", {}).get("message", "Ø·ÙØ¨ Gemini ÙÙ ÙÙØ¬Ø­.")
        logger.warning("Gemini API returned HTTP %s", response.status_code)
        # Avoid sending internal server details or credentials to the browser.
        if response.status_code in (401, 403):
            raise RuntimeError("ÙÙØªØ§Ø­ Gemini ØºÙØ± ØµØ§ÙØ­ Ø£Ù ÙØ§ ÙÙÙÙ ØµÙØ§Ø­ÙØ© Ø§Ø³ØªØ®Ø¯Ø§Ù Ø§ÙÙÙÙØ°Ø¬.")
        if response.status_code == 429:
            raise RuntimeError("ØªÙ ØªØ¬Ø§ÙØ² Ø­Ø¯ Ø§ÙØ§Ø³ØªØ®Ø¯Ø§Ù ÙØ¤ÙØªÙØ§. Ø§ÙØªØ¸Ø± ÙÙÙÙÙØ§ Ø«Ù Ø­Ø§ÙÙ ÙØ¬Ø¯Ø¯ÙØ§.")
        if response.status_code == 404:
            raise RuntimeError("Ø§ÙÙÙÙØ°Ø¬ ØºÙØ± ÙØªØ§Ø­ ÙÙØ°Ø§ Ø§ÙØ­Ø³Ø§Ø¨. Ø±Ø§Ø¬Ø¹ Ø§Ø³Ù Ø§ÙÙÙÙØ°Ø¬ ÙØ¥ØªØ§Ø­ØªÙ.")
        raise RuntimeError(f"Ø®Ø¯ÙØ© Gemini Ø±ÙØ¶Øª Ø§ÙØ·ÙØ¨ (HTTP {response.status_code}).")
    return data


def response_parts(data):
    candidates = data.get("candidates") or []
    if not candidates:
        block = data.get("promptFeedback", {}).get("blockReason")
        if block:
            raise RuntimeError("ÙÙ ÙØªÙ ØªÙÙÙØ° Ø§ÙØ·ÙØ¨ Ø¨Ø³Ø¨Ø¨ ÙÙÙØ¯ Ø§ÙØ³ÙØ§ÙØ©.")
        return []
    return candidates[0].get("content", {}).get("parts", []) or []


def text_from_parts(parts):
    return "".join(
        p.get("text", "") for p in parts
        if isinstance(p, dict) and isinstance(p.get("text"), str)
    ).strip()


@app.post("/api/chat")
def chat():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify(error="Ø¨ÙØ§ÙØ§Øª Ø§ÙØ·ÙØ¨ ØºÙØ± ØµØ§ÙØ­Ø©."), 400
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return jsonify(error="Ø§ÙØªØ¨ Ø±Ø³Ø§ÙØ© Ø£ÙÙÙØ§."), 400

    contents = []
    for item in messages[-16:]:
        if not isinstance(item, dict):
            continue
        role, text = item.get("role"), item.get("content")
        if role not in ("user", "assistant") or not isinstance(text, str):
            continue
        text = text.strip()[:6000]
        if text:
            contents.append({
                "role": "user" if role == "user" else "model",
                "parts": [{"text": text}],
            })

    if not contents or contents[-1]["role"] != "user":
        return jsonify(error="Ø£Ø±Ø³Ù Ø±Ø³Ø§ÙØ© Ø¬Ø¯ÙØ¯Ø© ÙÙ ÙØ¶ÙÙ."), 400

    image = body.get("image")
    if image is not None:
        if not isinstance(image, dict):
            return jsonify(error="Ø¨ÙØ§ÙØ§Øª Ø§ÙØµÙØ±Ø© ØºÙØ± ØµØ§ÙØ­Ø©."), 400
        try:
            encoded = prepare_image(image.get("data", ""))
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        contents[-1]["parts"].append({
            "inline_data": {"mime_type": "image/jpeg", "data": encoded}
        })

    payload = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": contents,
        "generationConfig": {"temperature": 0.7, "maxOutputTokens": 1500},
    }
    try:
        data = gemini_request(CHAT_MODEL, payload, timeout=30)
        answer = text_from_parts(response_parts(data))
        if not answer:
            return jsonify(error="ÙÙ ÙØµÙ Ø±Ø¯ ÙØµÙ ÙÙ Gemini. Ø­Ø§ÙÙ ÙØ±Ø© Ø£Ø®Ø±Ù."), 502
        return jsonify(reply=answer)
    except requests.Timeout:
        return jsonify(error="Ø§Ø³ØªØºØ±Ù Ø§ÙØ±Ø¯ ÙÙØªÙØ§ Ø·ÙÙÙÙØ§. Ø­Ø§ÙÙ ÙØ±Ø© Ø£Ø®Ø±Ù."), 504
    except requests.RequestException:
        logger.exception("Network error while calling Gemini chat")
        return jsonify(error="ØªØ¹Ø°ÙØ± Ø§ÙØ§ØªØµØ§Ù Ø¨Ø®Ø¯ÙØ© Ø§ÙØ°ÙØ§Ø¡ Ø§ÙØ§ØµØ·ÙØ§Ø¹Ù."), 502
    except RuntimeError as exc:
        return jsonify(error=str(exc)), 502


@app.post("/api/generate-image")
def generate_image():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify(error="Ø¨ÙØ§ÙØ§Øª Ø§ÙØ·ÙØ¨ ØºÙØ± ØµØ§ÙØ­Ø©."), 400
    prompt = body.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return jsonify(error="Ø§ÙØªØ¨ ÙØµÙ Ø§ÙØµÙØ±Ø© Ø§ÙØªÙ ØªØ±ÙØ¯ Ø¥ÙØ´Ø§Ø¡ÙØ§."), 400
    prompt = prompt.strip()[:3000]
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]},
    }
    try:
        data = gemini_request(IMAGE_MODEL, payload, timeout=55)
        parts = response_parts(data)
        for part in parts:
            if not isinstance(part, dict):
                continue
            img = part.get("inlineData") or part.get("inline_data")
            if not isinstance(img, dict):
                continue
            encoded = img.get("data")
            mime = img.get("mimeType") or img.get("mime_type") or "image/png"
            if isinstance(encoded, str) and encoded and mime in (
                "image/png", "image/jpeg", "image/webp"
            ):
                # Return the image as base64 for the existing frontend to display.
                return jsonify(image=encoded, mimeType=mime,
                               text=text_from_parts(parts))
        return jsonify(error="Ø§ÙÙÙÙØ°Ø¬ ÙÙ ÙÙØ±Ø¬Ø¹ ØµÙØ±Ø©. ØªØ£ÙØ¯ ÙÙ Ø¥ØªØ§Ø­ØªÙ ÙØ­Ø³Ø§Ø¨ Gemini API."), 502
    except requests.Timeout:
        return jsonify(error="Ø§ÙØªÙØª ÙÙÙØ© Ø¥ÙØ´Ø§Ø¡ Ø§ÙØµÙØ±Ø©. Ø­Ø§ÙÙ Ø¨ÙØµÙ Ø£ÙØµØ±."), 504
    except requests.RequestException:
        logger.exception("Network error while generating image")
        return jsonify(error="ØªØ¹Ø°ÙØ± Ø§ÙØ§ØªØµØ§Ù Ø¨Ø®Ø¯ÙØ© Ø¥ÙØ´Ø§Ø¡ Ø§ÙØµÙØ±."), 502
    except RuntimeError as exc:
        return jsonify(error=str(exc)), 502


def register_pdf_font():
    if "TENXArabic" in pdfmetrics.getRegisteredFontNames():
        return "TENXArabic"
    candidates = [
        os.getenv("PDF_FONT_PATH", "").strip(),
        os.path.join(os.path.dirname(__file__), "fonts", "NotoSansArabic-Regular.ttf"),
        os.path.join(os.path.dirname(__file__), "fonts", "Amiri-Regular.ttf"),
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if not path or not os.path.isfile(path):
            continue
        try:
            pdfmetrics.registerFont(TTFont("TENXArabic", path))
            return "TENXArabic"
        except Exception:
            logger.warning("Unable to load configured PDF font")
    return "Helvetica"


def shape_arabic(text):
    if not ARABIC_SUPPORT:
        return text
    try:
        return get_display(arabic_reshaper.reshape(text))
    except Exception:
        logger.exception("Arabic shaping failed")
        return text


@app.post("/api/create-pdf")
def create_pdf():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify(error="Ø¨ÙØ§ÙØ§Øª Ø§ÙØ·ÙØ¨ ØºÙØ± ØµØ§ÙØ­Ø©."), 400
    title = body.get("title", "TENX AI Summary")
    content = body.get("text", "")
    if not isinstance(title, str) or not isinstance(content, str):
        return jsonify(error="Ø§ÙØ¹ÙÙØ§Ù ÙØ§ÙÙØ­ØªÙÙ ÙØ¬Ø¨ Ø£Ù ÙÙÙÙØ§ ÙØµÙØ§."), 400
    title = title.strip()[:180] or "TENX AI Summary"
    content = content.strip()[:35000]
    if not content:
        return jsonify(error="ÙØ§ ÙÙØ¬Ø¯ ÙØµ ÙØ¥ÙØ´Ø§Ø¡ ÙÙÙ PDF."), 400

    font_name = register_pdf_font()
    # Arabic needs BOTH a font and shaping/bidi packages for correct output.
    contains_arabic = any("\u0600" <= ch <= "\u06ff" for ch in title + content)
    if contains_arabic and (not ARABIC_SUPPORT or font_name == "Helvetica"):
        return jsonify(error=(
            "ÙØ¥ÙØ´Ø§Ø¡ PDF Ø¹Ø±Ø¨Ù ØµØ­ÙØ­Ø ØªØ£ÙØ¯ ÙÙ ØªØ«Ø¨ÙØª arabic-reshaper Ùpython-bidi "
            "ÙØ¥Ø¶Ø§ÙØ© Ø®Ø· Ø¹Ø±Ø¨Ù Ø¯Ø§Ø®Ù ÙØ¬ÙØ¯ fonts."
        )), 500

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="TENXPDFTitle", parent=styles["Title"], fontName=font_name,
        fontSize=19, leading=27, alignment=TA_CENTER,
        textColor=colors.HexColor("#183B67"), spaceAfter=20,
    ))
    styles.add(ParagraphStyle(
        name="TENXPDFBody", parent=styles["BodyText"], fontName=font_name,
        fontSize=10.5, leading=17,
        alignment=TA_RIGHT if contains_arabic else 0,
        textColor=colors.HexColor("#202938"), spaceAfter=8,
    ))

    def safe_paragraph(raw, style):
        # Escape HTML, preserve line breaks, then shape each line independently.
        chunks = []
        for line in raw.splitlines() or [""]:
            if contains_arabic:
                line = shape_arabic(line)
            chunks.append(html.escape(line) or "&#160;")
        return Paragraph("<br/>".join(chunks), style)

    output = io.BytesIO()
    try:
        doc = SimpleDocTemplate(
            output, pagesize=A4, rightMargin=48, leftMargin=48,
            topMargin=52, bottomMargin=48, title=title, author="TENX AI",
        )
        story = [safe_paragraph(title, styles["TENXPDFTitle"]), Spacer(1, 8)]
        for paragraph in content.split("\n\n"):
            story.append(safe_paragraph(paragraph.replace("\n", "<br/>"), styles["TENXPDFBody"]))
        doc.build(story)
        output.seek(0)
        return send_file(
            output, mimetype="application/pdf", as_attachment=True,
            download_name="TENX_AI_Summary.pdf",
        )
    except Exception:
        logger.exception("PDF generation failed")
        return jsonify(error="ØªØ¹Ø°ÙØ± Ø¥ÙØ´Ø§Ø¡ ÙÙÙ PDF. Ø±Ø§Ø¬Ø¹ Ø³Ø¬ÙØ§Øª Vercel."), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
