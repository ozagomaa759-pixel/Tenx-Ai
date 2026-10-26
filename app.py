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
MAX_IMAGE_DIMENSION = 1200  # تقليل الأبعاد قليلاً لتسريع معالجة الصور

# إنشاء Session ثابتة لتسريع الاتصالات المتكررة بـ API
api_session = requests.Session()

SYSTEM_PROMPT = """
You are TENX AI, an educational AI assistant for students.
ACCURACY: Give accurate, direct answers. Explain code clearly.
LANGUAGE: Reply in the same language as the user. Use natural Arabic for Arabic text. Never reverse letters or code.
FORMATTING: Short paragraphs, plain text, fenced code blocks for code.
CREATOR: If asked who created TENX AI, say Mohamed Gomaa (Oza).
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
    return jsonify(error="الصورة كبيرة جداً، اختر صورة أقل من 5 ميجابايت."), 413

def prepare_image(image_data):
    try:
        raw = base64.b64decode(image_data, validate=True)
        if not raw or len(raw) > MAX_IMAGE_BYTES:
            raise ValueError("حجم الصورة غير مسموح به.")

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
            image.save(output, format="JPEG", quality=80, optimize=True) # تقليل الجودة لـ 80 لتسريع الرفع والتحليل
            return base64.b64encode(output.getvalue()).decode("ascii")
    except Exception as exc:
        raise ValueError("فشل معالجة الصورة.") from exc

@app.post("/api/chat")
def chat():
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return jsonify(error="مفتاح API غير موجود في الإعدادات."), 503

    body = request.get_json(silent=True) or {}
    messages = body.get("messages", [])

    if not isinstance(messages, list) or not messages:
        return jsonify(error="اكتب رسالة أولاً."), 400

    contents = []
    # تقليص الذاكرة المؤقتة لآخر 8 رسائل فقط لتقليل الضغط وسرعة الرد
    for item in messages[-8:]:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            continue
        content = content.strip()[:3000]
        if content:
            contents.append({
                "role": "user" if role == "user" else "model",
                "parts": [{"text": content}]
            })

    if not contents or contents[-1]["role"] != "user":
        return jsonify(error="أرسل رسالة صحيحة."), 400

    image = body.get("image")
    if image is not None and isinstance(image, dict):
        image_data = image.get("data", "")
        if image_data:
            try:
                converted = prepare_image(image_data)
                contents[-1]["parts"].append({
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": converted
                    }
                })
            except ValueError as exc:
                return jsonify(error=str(exc)), 400

    model = "gemini-3.5-flash-lite"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    payload = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": contents,
        "generationConfig": {
            "temperature": 0.6,
            "maxOutputTokens": 800  # تقليص الحد الأقصى للتوكنز لتوليد الرد بسرعة صاروخية
        }
    }

    try:
        # استخدام api_session لتسريع الاتصال عبر تكرار فتح الـ sockets
        response = api_session.post(
            url,
            headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
            json=payload,
            timeout=15
        )

        data = response.json() if response.ok else {}
        if not response.ok:
            msg = data.get("error", {}).get("message", "حدث خطأ في الاتصال.")
            return jsonify(error=msg), 502

        candidates = data.get("candidates", [])
        parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
        answer = "".join(p.get("text", "") for p in parts if isinstance(p, dict)).strip()

        if not answer:
            return jsonify(error="الذكاء الاصطناعي لم يعيد أي رد، حاول مرة أخرى."), 502

        return jsonify(reply=answer)

    except requests.Timeout:
        return jsonify(error="استغرق الرد وقتاً طويلاً، حاول مرة أخرى."), 504
    except requests.RequestException:
        return jsonify(error="فشل الاتصال بخدمة الذكاء الاصطناعي."), 502

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
