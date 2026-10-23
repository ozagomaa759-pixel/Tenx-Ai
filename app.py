"""
================================================================================
Tenx-AI Zenith - Advanced Core Server Engine
================================================================================
Framework: Flask (WSGI Application)
Deployment Target: Vercel Serverless Functions / Cloud Hosting
Authentication: Bearer Token Mechanism (Optimized for AQ-style tokens)
Model Engine: Google Gemini API (gemini-1.5-flash)

Description:
    This server module powers the Tenx-AI Zenith platform. It manages routing,
    API request formatting, session history tracking, response parsing, error
    resilience, and security headers.
================================================================================
"""

import os
import sys
import time
import json
import uuid
import logging
import datetime
from typing import Dict, Any, List, Optional, Tuple
import requests
from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    Response,
    stream_with_context,
    make_response,
    session
)

# ==============================================================================
# 1. LOGGING & APPLICATION INITIALIZATION
# ==============================================================================

# Configure detailed logging format for production monitoring
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("ZenithEngine")

app = Flask(__name__)

# Application Configurations
app.config['SECRET_KEY'] = os.environ.get("SECRET_KEY", "zenith-ultra-secure-key-9988776655")
app.config['JSON_SORT_KEYS'] = False
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
app.config['PERMANENT_SESSION_LIFETIME'] = datetime.timedelta(days=7)

# Load the primary API key / Access Token from Environment Variables
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# AI Model Endpoint Configurations
GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_MODEL = "gemini-1.5-flash"

# ==============================================================================
# 2. IN-MEMORY SESSION & CONVERSATION MANAGER
# ==============================================================================

class ConversationManager:
    """
    In-memory storage for handling user session histories, system prompts,
    and message formatting before sending payloads to Gemini API.
    """
    
    def __init__(self):
        self._conversations: Dict[str, List[Dict[str, str]]] = {}
        self._system_instruction = (
            "You are Tenx-AI Zenith, a hyper-intelligent, dark-themed AI assistant. "
            "You provide precise, concise, and highly creative responses. "
            "Always respond naturally in the user's language."
        )

    def get_session_id(self, req: request) -> str:
        """Retrieves or creates a unique session identifier."""
        sid = req.headers.get("X-Session-ID")
        if not sid:
            sid = str(uuid.uuid4())
        return sid

    def get_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieves history for a given session."""
        return self._conversations.get(session_id, [])

    def append_message(self, session_id: str, role: str, text: str) -> None:
        """Appends a new message entry to session history."""
        if session_id not in self._conversations:
            self._conversations[session_id] = []
            
        self._conversations[session_id].append({
            "role": role,
            "parts": [{"text": text}],
            "timestamp": datetime.datetime.utcnow().isoformat()
        })

        # Cap history to last 30 interactions to save memory
        if len(self._conversations[session_id]) > 30:
            self._conversations[session_id] = self._conversations[session_id][-30:]

    def clear_history(self, session_id: str) -> bool:
        """Clears stored history for a session."""
        if session_id in self._conversations:
            del self._conversations[session_id]
            return True
        return False

    def build_payload(self, user_message: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Constructs the complete payload including optional system instructions
        and conversation history formatted for Google's API.
        """
        contents = []

        # Include prior conversation context if session ID is valid
        if session_id and session_id in self._conversations:
            for item in self._conversations[session_id]:
                contents.append({
                    "role": item["role"],
                    "parts": item["parts"]
                })

        # Append current user prompt
        contents.append({
            "role": "user",
            "parts": [{"text": user_message}]
        })

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": 0.7,
                "topP": 0.95,
                "topK": 40,
                "maxOutputTokens": 2048,
            },
            "systemInstruction": {
                "parts": [{"text": self._system_instruction}]
            }
        }
        return payload

conv_manager = ConversationManager()

# ==============================================================================
# 3. HTTP HEADER & AUTHENTICATION BUILDER
# ==============================================================================

def build_api_headers() -> Dict[str, str]:
    """
    Constructs the exact HTTP headers needed by Google API.
    Uses 'Authorization: Bearer <TOKEN>' which seamlessly accepts 
    OAuth tokens starting with 'AQ.'.
    """
    if not GEMINI_API_KEY:
        logger.warning("Attempting to build headers with empty GEMINI_API_KEY.")
        
    return {
        "Authorization": f"Bearer {GEMINI_API_KEY.strip()}",
        "Content-Type": "application/json",
        "User-Agent": "Tenx-AI-Zenith-Server/1.0",
        "Accept": "application/json"
    }

def build_target_url(model_name: str = DEFAULT_MODEL, stream: bool = False) -> str:
    """Generates the clean endpoint URL without embedding sensitive keys in query params."""
    action = "streamGenerateContent" if stream else "generateContent"
    return f"{GEMINI_API_BASE_URL}/{model_name}:{action}"

# ==============================================================================
# 4. RESPONSE PARSING & RESILIENCE UTILITIES
# ==============================================================================

def extract_text_from_gemini_response(response_json: Dict[str, Any]) -> str:
    """
    Safely navigates through Gemini API nested JSON structure to extract output text.
    Handles potential edge cases where response structure differs.
    """
    try:
        candidates = response_json.get("candidates", [])
        if not candidates:
            # Check for safety blocks or alternate fields
            prompt_feedback = response_json.get("promptFeedback", {})
            if "blockReason" in prompt_feedback:
                reason = prompt_feedback.get("blockReason")
                return f"[تم حظر الإجابة بواسطة نظام الأمان: {reason}]"
            return "لم يتم استلام أي إجابة من نموذج الذكاء الاصطناعي."

        first_candidate = candidates[0]
        finish_reason = first_candidate.get("finishReason", "")
        
        if finish_reason and finish_reason not in ["STOP", "MAX_TOKENS"]:
            logger.warning(f"Candidate finished with unusual reason: {finish_reason}")

        content = first_candidate.get("content", {})
        parts = content.get("parts", [])
        
        if not parts:
            return "محتوى الإجابة فارغ."

        extracted_text = "".join([part.get("text", "") for part in parts])
        return extracted_text.strip()

    except Exception as err:
        logger.error(f"Error during JSON extraction: {str(err)}")
        return "حدث خطأ أثناء معالجة رد السيرفر."

# ==============================================================================
# 5. CORE ROUTING ENGINE
# ==============================================================================

@app.before_request
def log_incoming_request():
    """Logs metadata for every incoming request for debugging and analytics."""
    logger.info(f"--> Incoming {request.method} request to {request.path} | IP: {request.remote_addr}")

@app.after_request
def apply_security_headers(response: Response) -> Response:
    """Applies modern security and CORS headers to outgoing responses."""
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Server"] = "Tenx-AI-Zenith-Core"
    return response

# ------------------------------------------------------------------------------
# Route 1: Frontend Main Portal
# ------------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    """Renders the main dark-themed web application UI."""
    try:
        return render_template("index.html", year=datetime.datetime.now().year)
    except Exception as e:
        logger.error(f"Template rendering error: {str(e)}")
        return "<h1>Tenx-AI Zenith Core Server Active</h1><p>UI Template loading issue.</p>", 200

# ------------------------------------------------------------------------------
# Route 2: Server & Key Health Check
# ------------------------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health_check():
    """Diagnostic endpoint to verify environmental setup and server health."""
    has_key = bool(GEMINI_API_KEY and len(GEMINI_API_KEY.strip()) > 5)
    key_type = "AQ_OAuth_Token" if GEMINI_API_KEY.startswith("AQ.") else "Standard_API_Key"
    
    health_payload = {
        "status": "online",
        "engine": "Tenx-AI Zenith",
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "authentication": {
            "configured": has_key,
            "token_type": key_type if has_key else "None"
        },
        "python_version": sys.version.split()[0]
    }
    return jsonify(health_payload), 200

# ------------------------------------------------------------------------------
# Route 3: Primary Chat Endpoint (Standard JSON REST)
# ------------------------------------------------------------------------------
@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Main communication channel for chat queries.
    Receives JSON: {"message": "..."}
    Returns JSON: {"reply": "...", "status": "success"}
    """
    start_time = time.time()
    
    # 1. Ensure API Key exists
    if not GEMINI_API_KEY:
        logger.error("Request blocked: GEMINI_API_KEY environment variable is missing.")
        return jsonify({
            "error": "مفتاح API غير مضبوط في إعدادات البيئة (Vercel Variables).",
            "code": "MISSING_API_KEY"
        }), 500

    # 2. Parse payload
    try:
        req_data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "تنسيق طلب غير صالح (Invalid JSON)."}), 400

    if not req_data or "message" not in req_data:
        return jsonify({"error": "الطلب لا يحتوي على حقل الرسالة 'message'."}), 400

    user_message = str(req_data.get("message", "")).strip()
    if not user_message:
        return jsonify({"error": "لا يمكن إرسال رسالة فارغة."}), 400

    session_id = conv_manager.get_session_id(request)
    
    # 3. Build payload and headers
    payload = conv_manager.build_payload(user_message, session_id=session_id)
    headers = build_api_headers()
    target_url = build_target_url(DEFAULT_MODEL, stream=False)

    logger.info(f"Sending request to Gemini API via Bearer Token. Prompt Length: {len(user_message)}")

    # 4. Dispatch request to Google API
    try:
        api_response = requests.post(
            target_url,
            headers=headers,
            json=payload,
            timeout=35
        )

        elapsed_time = round(time.time() - start_time, 2)
        logger.info(f"Gemini API answered in {elapsed_time}s | Status: {api_response.status_code}")

        # 5. Handle success
        if api_response.status_code == 200:
            res_json = api_response.json()
            ai_reply = extract_text_from_gemini_response(res_json)

            # Store in session memory
            conv_manager.append_message(session_id, "user", user_message)
            conv_manager.append_message(session_id, "model", ai_reply)

            return jsonify({
                "status": "success",
                "reply": ai_reply,
                "session_id": session_id,
                "latency_seconds": elapsed_time
            }), 200

        # 6. Handle API Errors
        else:
            err_data = api_response.json() if api_response.content else {}
            err_msg = err_data.get("error", {}).get("message", api_response.text)
            logger.error(f"Gemini API Error [{api_response.status_code}]: {err_msg}")

            return jsonify({
                "status": "error",
                "error": f"خطأ في خوادم الذكاء الاصطناعي ({api_response.status_code}): {err_msg}",
                "code": "API_RESPONSE_ERROR"
            }), api_response.status_code

    except requests.exceptions.Timeout:
        logger.error("HTTP Request to Gemini timed out after 35 seconds.")
        return jsonify({
            "error": "انتهت مهلة الاتصال بالخادم. يرجى المحاولة مرة أخرى.",
            "code": "REQUEST_TIMEOUT"
        }), 504

    except requests.exceptions.RequestException as req_err:
        logger.error(f"Network error contacting Gemini: {str(req_err)}")
        return jsonify({
            "error": "حدث خطأ في الاتصال بالشبكة أثناء الاتصال بالذكاء الاصطناعي.",
            "code": "NETWORK_ERROR"
        }), 502

    except Exception as fatal_err:
        logger.exception("Unexpected exception occurred during /api/chat execution.")
        return jsonify({
            "error": f"خطأ غير متوقع في السيرفر الداخلي: {str(fatal_err)}",
            "code": "INTERNAL_SERVER_ERROR"
        }), 500

# ------------------------------------------------------------------------------
# Route 4: Streaming Chat Endpoint (Server-Sent Events / SSE)
# ------------------------------------------------------------------------------
@app.route("/api/chat/stream", methods=["POST"])
def chat_stream():
    """Streaming response generator for real-time text output."""
    if not GEMINI_API_KEY:
        return jsonify({"error": "GEMINI_API_KEY غير موجود."}), 500

    req_data = request.get_json(silent=True) or {}
    user_message = req_data.get("message", "").strip()

    if not user_message:
        return jsonify({"error": "رسالة فارغة."}), 400

    payload = conv_manager.build_payload(user_message)
    headers = build_api_headers()
    target_url = build_target_url(DEFAULT_MODEL, stream=True)

    def generate():
        try:
            with requests.post(target_url, headers=headers, json=payload, stream=True, timeout=30) as r:
                if r.status_code != 200:
                    yield f"data: {json.dumps({'error': f'API Error {r.status_code}'})}\n\n"
                    return

                for line in r.iter_lines():
                    if line:
                        decoded = line.decode('utf-8')
                        if decoded.startswith("data: "):
                            yield f"{decoded}\n\n"
        except Exception as stream_err:
            yield f"data: {json.dumps({'error': str(stream_err)})}\n\n"

    return Response(stream_with_context(generate()), content_type='text/event-stream')

# ------------------------------------------------------------------------------
# Route 5: Session History & Management
# ------------------------------------------------------------------------------
@app.route("/api/history", methods=["GET", "DELETE"])
def handle_history():
    """Retrieve or clear current conversation session history."""
    session_id = conv_manager.get_session_id(request)

    if request.method == "DELETE":
        cleared = conv_manager.clear_history(session_id)
        return jsonify({"status": "cleared" if cleared else "not_found"}), 200

    history = conv_manager.get_history(session_id)
    return jsonify({
        "session_id": session_id,
        "total_messages": len(history),
        "history": history
    }), 200

# ------------------------------------------------------------------------------
# Route 6: System Information
# ------------------------------------------------------------------------------
@app.route("/api/system/info", methods=["GET"])
def system_info():
    """Returns platform capabilities and system metadata."""
    return jsonify({
        "platform_name": "Tenx-AI Zenith",
        "version": "2.5.0-Enterprise",
        "supported_models": ["gemini-1.5-flash", "gemini-1.5-pro"],
        "active_model": DEFAULT_MODEL,
        "auth_method": "Bearer Token Header",
        "status": "Operational"
    }), 200

# ==============================================================================
# 6. GLOBAL ERROR HANDLERS
# ==============================================================================

@app.errorhandler(404)
def handle_not_found(e):
    """Custom 404 JSON error response."""
    return jsonify({"error": "المسار المطلوب غير موجود على السيرفر.", "code": "NOT_FOUND"}), 404

@app.errorhandler(500)
def handle_server_error(e):
    """Custom 500 JSON error response."""
    return jsonify({"error": "حدث خطأ داخلي في الخادم.", "code": "SERVER_ERROR"}), 500

# ==============================================================================
# 7. LOCAL APPLICATION ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting Tenx-AI Zenith Engine on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=True)
