import pandas as pd
import joblib
import matplotlib.pyplot as plt
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay)

from features import extra_features

# Load data and recreate the SAME split used in training
df = pd.read_csv("data/clean_combined.csv")
df["text"] = df["text"].fillna("")
_, X_test, _, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42,
    stratify=df["label"])

# Load the saved model and vectorizer
model = joblib.load("classifier/model.pkl")
vectorizer = joblib.load("classifier/vectorizer.pkl")

Xte = hstack([vectorizer.transform(X_test), extra_features(X_test)]).tocsr()
pred = model.predict(Xte)

# Metrics
report = classification_report(y_test, pred, target_names=["legit", "scam"])
print(report)
with open("report/metrics.txt", "w") as f:
    f.write(report)

# Confusion matrix image
cm = confusion_matrix(y_test, pred)
ConfusionMatrixDisplay(cm, display_labels=["legit", "scam"]).plot(cmap="Blues")
plt.title("Scam classifier: confusion matrix")
plt.savefig("report/confusion_matrix.png", dpi=150, bbox_inches="tight")
print("Saved report/metrics.txt and report/confusion_matrix.png")
print("False alarms (legit flagged as scam):", cm[0][1])
print("Missed scams (scam flagged as legit):", cm[1][0])