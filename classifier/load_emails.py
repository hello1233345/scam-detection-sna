import os
import email
from email import policy
import pandas as pd

def parse_eml(path):
    with open(path, "rb") as f:
        msg = email.message_from_binary_file(f, policy=policy.default)
    subject = msg["subject"] or ""
    body = msg.get_body(preferencelist=("plain", "html"))
    body = body.get_content() if body else ""
    return str(subject) + " " + body

def load_folder(folder, label, source):
    rows = []
    for name in os.listdir(folder):
        if name.startswith("cmds"):   # skip the corpus's helper file
            continue
        try:
            text = parse_eml(os.path.join(folder, name))
            rows.append({"text": text, "label": label, "source": source})
        except Exception:
            continue                  # skip broken files
    return pd.DataFrame(rows)

ham = load_folder("data/raw/easy_ham", 0, "spamassassin")
spam = load_folder("data/raw/spam", 1, "spamassassin")

emails = pd.concat([ham, spam], ignore_index=True)
emails.to_csv("data/raw/emails_spamassassin.csv", index=False)

print(emails.shape)
print(emails["label"].value_counts())