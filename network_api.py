"""
network_api.py  --  Person B, hand-off to Person C (the Streamlit app)

Two functions:
  get_cluster_risk(message_id)  -> for a message that is ALREADY in the dataset
  get_domain_risk(text)         -> for a NEW pasted message (the one the app really uses)

Why two? The graph is built from the dataset. A brand-new pasted message has
no node in it, so we cannot look up its cluster. Instead we extract its
domains / phone numbers and check whether they are known hubs, using the
scam_rate we measured for each hub (network/hub_domains.csv).

Usage from the app:
    import sys; sys.path.insert(0, "network")
    from network_api import get_domain_risk
    result = get_domain_risk(user_text)
"""
import os
import pickle
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from graph_builder import extract_domains, extract_phones  # noqa: E402

# A hub seen in fewer messages than this is too weak to trust: 1 message that
# happened to be a scam would give "100% scam rate" from a single example.
MIN_SUPPORT = 3

_partition = pickle.load(open(os.path.join(HERE, "partition.pkl"), "rb"))
_clusters = pd.read_csv(os.path.join(HERE, "cluster_risk.csv")).set_index("cluster")
_hubs = pd.read_csv(os.path.join(HERE, "hub_domains.csv")).set_index("node")


def get_cluster_risk(message_id) -> dict | None:
    """Cluster id, size and scam rate for a message that is in the graph.
    Returns None if the message has no domain/phone (so it is not in the graph)."""
    node = f"msg_{message_id}"
    if node not in _partition:
        return None
    cid = _partition[node]
    row = _clusters.loc[cid]
    return {
        "cluster": int(cid),
        "n_messages": int(row.n_messages),
        "scam_rate": float(row.scam_rate),
        "top_hub": row.top_hub,
    }


def get_domain_risk(text: str) -> dict:
    """Risk of a NEW message from the domains / phone numbers it contains.

    Returns {"risk": float|None, "matches": [...]}. risk is the highest scam
    rate among known hubs with >= MIN_SUPPORT messages; None means "nothing in
    this message is a known hub", so the app should rely on the content score.
    """
    keys = [f"dom_{d}" for d in extract_domains(text)]
    keys += [f"tel_{p}" for p in extract_phones(text)]
    matches = []
    for k in keys:
        if k in _hubs.index and _hubs.loc[k, "n_messages"] >= MIN_SUPPORT:
            h = _hubs.loc[k]
            cid = int(h.cluster)
            matches.append({
                "hub": h["name"], "type": h["type"],
                "n_messages": int(h.n_messages),
                "scam_rate": float(h.scam_rate),
                "cluster": cid,
                "cluster_scam_rate": float(_clusters.loc[cid, "scam_rate"]),
            })
    risk = max((m["scam_rate"] for m in matches), default=None)
    return {"risk": risk, "matches": matches}
