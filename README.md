# AI-Powered FAQ Chatbot

An NLP-based chatbot for customer support, built with Python, NLTK, scikit-learn (TF-IDF),
Flask, and SQLite. It matches user questions to the closest FAQ using text similarity
and logs every conversation.

## How it works

1. **NLTK** cleans and normalizes user input (lowercasing, removing filler words,
   reducing words to their root form).
2. **scikit-learn's TF-IDF + cosine similarity** compares the cleaned input against
   a set of known FAQ questions to find the best match.
3. **Flask** serves a simple chat webpage and a `/chat` API endpoint.
4. **SQLite** logs every message, the bot's reply, and its confidence score to
   `chatbot_logs.db`.

## Project structure

```
ai-chatbot/
├── app.py              # Flask server (run this)
├── chatbot.py          # NLP matching logic
├── database.py         # SQLite logging
├── requirements.txt    # Python dependencies
├── data/
│   └── faqs.json        # FAQ question/answer pairs (edit to add your own)
├── templates/
│   └── index.html        # Chat webpage
└── static/
    └── style.css          # Chat styling
```

## Setup & run (local machine)

1. **Install Python 3.9+** if you don't already have it: https://python.org/downloads
   Confirm with: `python --version`

2. **Open a terminal in this folder** and create a virtual environment (keeps
   dependencies isolated from the rest of your system):
   ```
   python -m venv venv
   ```
   Activate it:
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`

3. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

4. **Run the app:**
   ```
   python app.py
   ```

5. **Open your browser** to `http://127.0.0.1:5000` and start chatting.

6. **View logs** anytime at `http://127.0.0.1:5000/logs` (raw JSON of every
   conversation - this satisfies the "user interaction logs" requirement).

## Customizing

- Add more FAQs by editing `data/faqs.json` - just follow the existing
  `{"question": ..., "answer": ...}` format. No retraining needed, it reloads
  automatically on server restart.
- Adjust how strict matching is by changing `CONFIDENCE_THRESHOLD` in `chatbot.py`
  (lower = more lenient, higher = more cautious about unknown questions).
- Want deeper semantic understanding instead of keyword-based matching? See the
  commented-out `TransformerFAQChatbot` class at the bottom of `chatbot.py` for
  a drop-in upgrade using sentence-transformers.

## Publishing to GitHub

```
git init
git add .
git commit -m "AI-powered FAQ chatbot with NLP, Flask, SQLite"
```
Create a new repo on GitHub, then:
```
git remote add origin https://github.com/<your-username>/ai-chatbot.git
git branch -M main
git push -u origin main
```

Add a `.gitignore` with at least:
```
venv/
__pycache__/
chatbot_logs.db
```
