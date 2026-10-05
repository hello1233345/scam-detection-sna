import re
import numpy as np
from scipy.sparse import csr_matrix

URL_PATTERN = r'https?://[^\s<>"\']+|www\.[^\s<>"\']+'
URGENCY_WORDS = ["free", "winner", "click", "verify", "suspended",
                 "urgent", "prize", "claim", "password", "account"]

def extra_features(texts):
    """Turn each message into 3 numbers: has_url, log(length), urgency count."""
    rows = []
    for t in texts:
        t = str(t).lower()
        has_url = 1 if re.search(URL_PATTERN, t) else 0
        length = np.log1p(len(t))
        urgency = sum(t.count(w) for w in URGENCY_WORDS)
        rows.append([has_url, length, urgency])
    return csr_matrix(np.array(rows, dtype=float))