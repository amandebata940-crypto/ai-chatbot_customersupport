"""
chatbot.py
----------
This is the "brain" of the chatbot.

How it works (in plain English):
1. We load a list of FAQ question/answer pairs.
2. We clean up text using NLTK (lowercase it, remove filler words like
   "the"/"is", and reduce words to their root form - e.g. "running" -> "run").
   This is called PREPROCESSING.
3. We turn every FAQ question into a vector of numbers using TF-IDF
   (a technique that scores how important each word is to a sentence).
4. When the user types something, we clean their message the same way,
   turn it into a TF-IDF vector, and measure its COSINE SIMILARITY
   (a math measure of "how similar are these two vectors") against every
   FAQ question.
5. Whichever FAQ question is most similar wins, and we return its answer -
   but only if the similarity score clears a minimum threshold, otherwise
   we admit we don't know and offer to connect to a human.

This is a classic, lightweight NLP approach used in real production FAQ bots.
It needs no internet connection and no huge model download.

OPTIONAL UPGRADE: near the bottom of this file there's a commented-out
version that uses a pretrained Transformer model (sentence-transformers)
for deeper contextual understanding, if you want to swap it in later.
"""

import json
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- One-time NLTK downloads (safe to run every time, it skips if cached) ---
for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"tokenizers/{pkg}") if "punkt" in pkg else nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

lemmatizer = WordNetLemmatizer()
STOP_WORDS = set(stopwords.words("english"))

# Below this similarity score, we treat the question as "unknown"
CONFIDENCE_THRESHOLD = 0.30


def preprocess(text: str) -> str:
    """Lowercase, strip punctuation, remove stopwords, lemmatize."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = nltk.word_tokenize(text)
    cleaned = [lemmatizer.lemmatize(tok) for tok in tokens if tok not in STOP_WORDS]
    return " ".join(cleaned)


class FAQChatbot:
    def __init__(self, faq_path: str = "data/faqs.json"):
        with open(faq_path, "r", encoding="utf-8") as f:
            self.faqs = json.load(f)

        self.questions = [item["question"] for item in self.faqs]
        self.answers = [item["answer"] for item in self.faqs]

        # Preprocess all FAQ questions once, up front
        self.processed_questions = [preprocess(q) for q in self.questions]

        # Fit TF-IDF on the FAQ questions. This builds the vocabulary and
        # learns which words are distinctive vs. common.
        self.vectorizer = TfidfVectorizer()
        self.question_vectors = self.vectorizer.fit_transform(self.processed_questions)

    def get_response(self, user_message: str):
        """
        Returns a tuple: (answer_text, matched_question_or_None, confidence_score)
        """
        cleaned = preprocess(user_message)

        if not cleaned.strip():
            return (
                "Could you rephrase that? I didn't catch any keywords I recognize.",
                None,
                0.0,
            )

        user_vector = self.vectorizer.transform([cleaned])
        similarities = cosine_similarity(user_vector, self.question_vectors)[0]

        best_idx = similarities.argmax()
        best_score = float(similarities[best_idx])

        if best_score < CONFIDENCE_THRESHOLD:
            return (
                "I'm not confident I understand that. Could you rephrase, "
                "or type 'talk to agent' to reach a human?",
                None,
                best_score,
            )

        return (self.answers[best_idx], self.questions[best_idx], best_score)


# ---------------------------------------------------------------------------
# OPTIONAL UPGRADE: Transformer-based contextual matching
# ---------------------------------------------------------------------------
# If you want deeper semantic understanding (e.g. matching "my package is
# late" to "how long does shipping take?" even with zero shared keywords),
# install sentence-transformers and swap in this class instead:
#
#   pip install sentence-transformers
#
# from sentence_transformers import SentenceTransformer, util
#
# class TransformerFAQChatbot:
#     def __init__(self, faq_path="data/faqs.json"):
#         with open(faq_path) as f:
#             self.faqs = json.load(f)
#         self.questions = [i["question"] for i in self.faqs]
#         self.answers = [i["answer"] for i in self.faqs]
#         self.model = SentenceTransformer("all-MiniLM-L6-v2")  # downloads ~80MB once
#         self.question_embeddings = self.model.encode(self.questions, convert_to_tensor=True)
#
#     def get_response(self, user_message):
#         user_embedding = self.model.encode(user_message, convert_to_tensor=True)
#         scores = util.cos_sim(user_embedding, self.question_embeddings)[0]
#         best_idx = int(scores.argmax())
#         best_score = float(scores[best_idx])
#         if best_score < 0.5:
#             return ("I'm not sure I understand. Could you rephrase?", None, best_score)
#         return (self.answers[best_idx], self.questions[best_idx], best_score)
