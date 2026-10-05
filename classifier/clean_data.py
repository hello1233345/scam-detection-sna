import re
import pandas as pd
from urllib.parse import urlparse

df = pd.read_csv("data/raw_combined.csv")

# Matches http(s):// links and www. links, stopping at spaces, quotes and <>
URL_PATTERN = r'https?://[^\s<>"\']+|www\.[^\s<>"\']+'

def get_urls(text):
    return re.findall(URL_PATTERN, str(text))

def get_domain(url):
    if not url.startswith("http"):
        url = "http://" + url          # so urlparse can read www. links
    try:
        host = urlparse(url).hostname or ""
    except ValueError:
        return ""
    return host.lower().removeprefix("www.")

def clean_text(text):
    text = str(text)
    text = re.sub(r"<[^>]+>", " ", text)   # remove HTML tags
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()  # collapse extra whitespace
    return text

# Extract URLs first, from the ORIGINAL text, before the HTML tags are removed
urls = df["text"].apply(get_urls)
df["has_url"] = urls.apply(lambda u: 1 if len(u) > 0 else 0)
df["domain"] = urls.apply(lambda u: get_domain(u[0]) if len(u) > 0 else "")

df["text"] = df["text"].apply(clean_text)

df.to_csv("data/clean_combined.csv", index=False)

print(df.shape)
print(df[["text", "domain"]].head())
print("Messages with a domain:", (df["domain"] != "").sum())
print("\nTop 10 domains:")
print(df[df["domain"] != ""]["domain"].value_counts().head(10))