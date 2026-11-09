"""
graph_builder.py  --  Person B, Day 2-3
Builds the message <-> domain / phone-number graph and saves it.

Run from the repo root:   python network/graph_builder.py

WHY THIS DESIGN (be ready to explain in an interview)
-----------------------------------------------------
Nodes:  message nodes  (msg_<id>)       -> one per message that has a link or phone
        domain nodes   (dom_<domain>)   -> e.g. dom_newsisfree.com
        phone nodes    (tel_<number>)   -> phone numbers / SMS shortcodes
Edges:  message -- domain   (the message contains a link to that domain)
        message -- phone    (the SMS tells you to call/text that number)

This is a *bipartite* graph: edges only go between a message and an
"infrastructure" node (domain or phone), never message-message or
domain-domain. Two messages that never touch each other directly become
connected through a shared domain or shared phone number. That indirect
link is exactly how a scam campaign shows up.

WHY NOT JUST "ONE DOMAIN PER MESSAGE" (what the plan PDF says)?
If every message has exactly one domain, each domain is the centre of its
own separate star and no star touches another. We measured this on the real
data: 642 domains -> 642 connected components. Betweenness would be 0
everywhere and Louvain would just return each star. So we add:
  (a) ALL domains in a message, not only the first one
  (b) phone numbers / shortcodes found in SMS text
which join the stars together.
"""
import re
import pickle
from urllib.parse import urlparse

import pandas as pd
import networkx as nx

RAW_CSV = "data/raw_combined.csv"      # original text (we need every URL, and the
                                       # HTML tags are still there, so href links survive)
CLEAN_CSV = "data/clean_combined.csv"  # not strictly needed; kept for reference
OUT_GRAPH = "network/graph.pkl"

# http(s):// links and www. links, stopping at spaces, quotes and angle brackets.
URL_PATTERN = r'https?://[^\s<>"\']+|www\.[^\s<>"\']+'

# UK-style phone numbers (0800 123 4567 -> 11 digits), longer international
# numbers, and 5-digit SMS shortcodes like 86688 (these are very common in
# the UCI SMS scam texts: "text WIN to 87066").
PHONE_PATTERN = r"(?<!\d)(?:0\d{9,10}|\+?\d{10,12}|\d{5})(?!\d)"


def get_domain(url: str) -> str:
    """http://www.Example.com/a/b?x=1  ->  example.com"""
    if not url.startswith("http"):
        url = "http://" + url           # urlparse needs a scheme to find the host
    try:
        host = urlparse(url).hostname or ""
    except ValueError:                  # malformed url, e.g. broken IPv6 brackets
        return ""
    return host.lower().removeprefix("www.")


def extract_domains(text: str) -> list[str]:
    """All DISTINCT domains in a message (the preprocessing step kept only the first)."""
    found = {get_domain(u) for u in re.findall(URL_PATTERN, str(text))}
    found.discard("")
    return sorted(found)


def extract_phones(text: str) -> list[str]:
    return sorted(set(re.findall(PHONE_PATTERN, str(text))))


def build_graph(df: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()
    for row in df.itertuples():
        msg = f"msg_{row.id}"
        domains = extract_domains(row.text)
        # Phone numbers only for SMS: email bodies are full of digit strings
        # (dates, ids) that are not phone numbers and would create fake links.
        phones = extract_phones(row.text) if row.source == "uci_sms" else []
        if not domains and not phones:
            continue                    # nothing to link on -> not part of the graph
        G.add_node(msg, kind="message", label=int(row.label), source=row.source)
        for d in domains:
            G.add_node(f"dom_{d}", kind="domain")
            G.add_edge(msg, f"dom_{d}")
        for p in phones:
            G.add_node(f"tel_{p}", kind="phone")
            G.add_edge(msg, f"tel_{p}")
    return G


def describe(G: nx.Graph) -> None:
    kinds = pd.Series(dict(G.nodes(data="kind"))).value_counts()
    comps = sorted((len(c) for c in nx.connected_components(G)), reverse=True)
    print("nodes:", G.number_of_nodes(), "| edges:", G.number_of_edges())
    print(kinds.to_string())
    print("connected components:", len(comps))
    print("largest components:", comps[:8])


if __name__ == "__main__":
    # keep_default_na=False stops pandas turning an empty string into NaN
    df = pd.read_csv(RAW_CSV, keep_default_na=False)
    G = build_graph(df)
    describe(G)
    with open(OUT_GRAPH, "wb") as f:
        pickle.dump(G, f)
    print("saved ->", OUT_GRAPH)
