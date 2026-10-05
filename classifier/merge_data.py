import pandas as pd

# 1. Load the SMS data
sms = pd.read_csv("data/raw/SMSSpamCollection", sep="\t",
                  names=["label", "text"])
sms["label"] = sms["label"].map({"ham": 0, "spam": 1})
sms["source"] = "uci_sms"

# 2. Load the email data you just made
emails = pd.read_csv("data/raw/emails_spamassassin.csv")

# 3. Stack them together
df = pd.concat([sms, emails], ignore_index=True)

# 4. Clean up: drop empty texts and duplicates
df = df.dropna(subset=["text"])
df = df.drop_duplicates(subset=["text"])

# 5. Add an id column and put columns in the agreed order
df.insert(0, "id", range(len(df)))
df = df[["id", "text", "label", "source"]]

# 6. Save
df.to_csv("data/raw_combined.csv", index=False)

print(df.shape)
print(df["label"].value_counts())
print(df["source"].value_counts())