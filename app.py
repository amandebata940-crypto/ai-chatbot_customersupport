"""
app.py
------
The web server. This is what you run to start the chatbot.

It does 3 jobs:
1. Serves the chat webpage (templates/index.html) at "/"
2. Exposes a POST endpoint at "/chat" that the webpage calls every time
   you send a message - it returns the bot's reply as JSON
3. Exposes "/logs" so you (the developer) can see every conversation
   that's been logged to the database

Run it with:  python app.py
Then open:    http://127.0.0.1:5000 in your browser
"""

import uuid
from flask import Flask, render_template, request, jsonify, session

from chatbot import FAQChatbot
import database

app = Flask(__name__)
app.secret_key = "dev-secret-key-change-this-in-production"  # needed for session cookies

# Load the chatbot once when the server starts (not on every request - that'd be slow)
bot = FAQChatbot(faq_path="data/faqs.json")

# Create the database/table if this is the first run
database.init_db()


@app.route("/")
def home():
    # Give each browser tab a unique session ID so we can group its messages in the logs
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    user_message = (data or {}).get("message", "").strip()

    if not user_message:
        return jsonify({"reply": "Please type something!"}), 400

    session_id = session.get("session_id", "anonymous")

    reply, matched_question, confidence = bot.get_response(user_message)

    database.log_conversation(
        session_id=session_id,
        user_message=user_message,
        bot_response=reply,
        matched_question=matched_question,
        confidence=confidence,
    )

    return jsonify({"reply": reply, "confidence": round(confidence, 2)})


@app.route("/logs")
def logs():
    """Simple JSON view of recent conversations - handy for the 'user interaction logs' requirement."""
    return jsonify(database.get_all_logs())


if __name__ == "__main__":
    app.run(debug=True, port=5000)
