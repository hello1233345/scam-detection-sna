import pandas as pd
import joblib
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, recall_score

from features import extra_features

# 1. Load the cleaned data
df = pd.read_csv("data/clean_combined.csv")
df["text"] = df["text"].fillna("")

# 2. Split 80/20 (stratify keeps the scam percentage equal in both parts)
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42,
    stratify=df["label"])

# 3. Turn text into numbers (TF-IDF) and add our 3 extra features
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
Xtr = hstack([vectorizer.fit_transform(X_train), extra_features(X_train)]).tocsr()
Xte = hstack([vectorizer.transform(X_test), extra_features(X_test)]).tocsr()

# 4. Train two models and compare
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "RandomForest": RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                           random_state=42, n_jobs=-1),
}

best_name, best_model, best_recall = None, None, -1
for name, model in models.items():
    model.fit(Xtr, y_train)
    pred = model.predict(Xte)
    print("=" * 50)
    print(name)
    print(classification_report(y_test, pred, target_names=["legit", "scam"]))
    rec = recall_score(y_test, pred)
    if rec > best_recall:
        best_name, best_model, best_recall = name, model, rec

# 5. Save the winner (highest scam recall)
print("Best model:", best_name, "with scam recall", round(best_recall, 3))
joblib.dump(best_model, "classifier/model.pkl")
joblib.dump(vectorizer, "classifier/vectorizer.pkl")
print("Saved classifier/model.pkl and classifier/vectorizer.pkl")