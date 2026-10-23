import os
import json
import requests
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# HTML Template Embedded Directly to Avoid Vercel Path Issues
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tenx-AI Zenith</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
        body { background-color: #0f172a; color: #f8fafc; font-family: 'Segoe UI', Tahoma, sans-serif; height: 100vh; display: flex; flex-direction: column; margin: 0; }
        .header { background: #1e293b; padding: 15px 20px; border-bottom: 1px solid #334155; display: flex; justify-content: space-between; align-items: center; }
        .chat-container { flex: 1; overflow-y: auto; padding: 20px; max-width: 900px; margin: 0 auto; width: 100%; box-sizing: border-box; }
        .message { margin-bottom: 20px; display: flex; align-items: flex-start; gap: 12px; }
        .message.user { flex-direction: row-reverse; }
        .avatar { width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; flex-shrink: 0; }
        .user .avatar { background: #2563eb; color: white; }
        .bot .avatar { background: #10b981; color: white; }
        .content { background: #1e293b; padding: 12px 18px; border-radius: 16px; max-width: 75%; line-height: 1.6; word-break: break-word; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }
        .user .content { background: #2563eb; color: #fff; }
        .input-area { background: #1e293b; padding: 15px; border-top: 1px solid #334155; }
        .input-box { max-width: 900px; margin: 0 auto; display: flex; gap: 10px; }
        textarea { background: #0f172a; border: 1px solid #334155; color: #fff; border-radius: 12px; padding: 12px 15px; resize: none; flex: 1; height: 50px; outline: none; }
        textarea:focus { border-color: #3b82f6; }
        .btn-send { background: #2563eb; color: white; border: none; border-radius: 12px; padding: 0 25px; cursor: pointer; transition: 0.2s; font-size: 18px; }
        .btn-send:hover { background: #1d4ed8; }
    </style>
</head>
<body>

    <div class="header">
        <h5 class="m-0 fw-bold text-primary"><i class="fa-solid fa-brain me-2"></i>Tenx-AI Zenith</h5>
        <span class="badge bg-success">متصل ومعالج</span>
    </div>

    <div class="chat-container" id="chatContainer">
        <div class="message bot">
            <div class="avatar"><i class="fa-solid fa-robot"></i></div>
            <div class="content">أهلاً بك في Tenx-AI Zenith! أنا جاهز لمساعدتك في أي سؤال أو مهمة.</div>
        </div>
    </div>

    <div class="input-area">
        <div class="input-box">
            <textarea id="userInput" placeholder="اكتب رسالتك هنا..." rows="1"></textarea>
            <button class="btn-send" onclick="sendMessage()"><i class="fa-solid fa-paper-plane"></i></button>
        </div>
    </div>

    <script>
        async function sendMessage() {
            const input = document.getElementById('userInput');
            const message = input.value.trim();
            if (!message) return;

            const chatContainer = document.getElementById('chatContainer');

            chatContainer.innerHTML += `
                <div class="message user">
                    <div class="avatar"><i class="fa-solid fa-user"></i></div>
                    <div class="content">${escapeHtml(message)}</div>
                </div>
            `;
            
            input.value = '';
            chatContainer.scrollTop = chatContainer.scrollHeight;

            const loadingId = 'loading-' + Date.now();
            chatContainer.innerHTML += `
                <div class="message bot" id="${loadingId}">
                    <div class="avatar"><i class="fa-solid fa-robot"></i></div>
                    <div class="content"><i class="fa-solid fa-spinner fa-spin me-2"></i>جاري التفكير...</div>
                </div>
            `;
            chatContainer.scrollTop = chatContainer.scrollHeight;

            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: message })
                });

                const data = await response.json();
                const loadingElem = document.getElementById(loadingId);

                if (response.ok && data.response) {
                    loadingElem.querySelector('.content').innerHTML = data.response.replace(/\\n/g, '<br>');
                } else {
                    loadingElem.querySelector('.content').innerHTML = '<span class="text-danger">حدث خطأ: ' + (data.error || 'فشل معالجة الطلب') + '</span>';
                }
            } catch (err) {
                document.getElementById(loadingId).querySelector('.content').innerHTML = '<span class="text-danger">عفواً، تعذر الاتصال بالسيرفر.</span>';
            }
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        function escapeHtml(text) {
            return text.replace(/[&<"']/g, function(m) {
                return { '&': '&amp;', '<': '&lt;', '"': '&quot;', "'": '&#039;' }[m];
            });
        }

        document.getElementById('userInput').addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/health')
def health():
    token = os.environ.get("GEMINI_API_KEY", "")
    return jsonify({
        "status": "online",
        "has_token": bool(token)
    })

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json() or {}
        user_message = data.get('message', '')

        if not user_message:
            return jsonify({'error': 'الرسالة فارغة'}), 400

        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            return jsonify({'error': 'لم يتم العثور على GEMINI_API_KEY في إعدادات البيئة'}), 500

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
        headers = {'Content-Type': 'application/json'}
        payload = {
            "contents": [{
                "parts": [{"text": user_message}]
            }]
        }

        res = requests.post(url, headers=headers, json=payload, timeout=30)
        res_data = res.json()

        if res.status_code == 200:
            bot_reply = res_data['candidates'][0]['content']['parts'][0]['text']
            return jsonify({'response': bot_reply})
        else:
            error_msg = res_data.get('error', {}).get('message', 'خطأ غير معروف من Google API')
            return jsonify({'error': error_msg}), res.status_code

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
