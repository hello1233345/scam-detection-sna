import os
import re
import sys
import joblib
from scipy.sparse import hstack

# Work out where this file lives, so it runs from anywhere (including Person C's app)
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from features import extra_features

_model = joblib.load(os.path.join(HERE, "model.pkl"))
_vectorizer = joblib.load(os.path.join(HERE, "vectorizer.pkl"))

def _clean(text):
    text = str(text)
    text = re.sub(r"<[^>]+>", " ", text)      # remove HTML tags
    text = text.lower()
    return re.sub(r"\s+", " ", text).strip()  # collapse whitespace

def predict_score(text):
    """Return the probability (0 to 1) that the message is a scam."""
    text = _clean(text)
    X = hstack([_vectorizer.transform([text]), extra_features([text])]).tocsr()
    return float(_model.predict_proba(X)[0][1])

if __name__ == "__main__":
    tests = [
        "Congratulations! You are a winner. Click http://free-prize.example.com to claim your free prize now",
        "URGENT: your account has been suspended. Verify your password at www.bank-secure.example.com",
        "Hey, are we still meeting for lunch tomorrow at 1?",
        "Can you send me the notes from today's class?",
    ]
    for t in tests:
        print(round(predict_score(t), 3), "|", t[:60])