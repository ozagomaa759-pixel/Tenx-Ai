# TENX AI — school presentation project

A polished, responsive AI-chat website prototype. It connects to DeepSeek through a small Python server so the API key is **not exposed in the browser code**.

## Run on Windows

1. Install Python 3 from https://www.python.org/downloads/ (enable **Add Python to PATH**).
2. Extract this ZIP to a folder.
3. Open Command Prompt in that folder and run:

   ```bat
   py -m pip install flask requests
   ```

4. Set your DeepSeek API key for this Command Prompt window (replace `YOUR_KEY` with your own key):

   ```bat
   set DEEPSEEK_API_KEY=YOUR_KEY
   ```

5. Start the website:

   ```bat
   py app.py
   ```

6. Open http://127.0.0.1:5000 in your browser.

Keep the Command Prompt window open while presenting. To stop the server, press `Ctrl+C`.

## Important

- Never paste an API key into `static/index.html`, a public GitHub repository, or a screenshot.
- If you previously posted a key publicly or shared it in a chat, revoke/rotate it in the provider dashboard and create a fresh one.
- Live responses require internet access and a working provider API key/account. The website will show a helpful message if the server is not configured.
- This project is a prototype, not a hosted public website. `127.0.0.1` works on the computer running it.
