"""
analyze.py  --  Person B, Day 4-5
Centrality (hub domains), Louvain communities (scam rings), cluster risk,
and an honest train/test validation.

Run from the repo root (after graph_builder.py):   python network/analyze.py
Outputs: network/hub_domains.csv, network/cluster_risk.csv, network/partition.pkl
"""
import pickle

import pandas as pd
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

SEED = 42
G = pickle.load(open("network/graph.pkl", "rb"))

messages = [n for n, k in G.nodes(data="kind") if k == "message"]

# ---------------------------------------------------------------------------
# 1. CENTRALITY  -- which domains / phone numbers are the busiest "hubs"?
# ---------------------------------------------------------------------------
# degree centrality = (number of neighbours) / (n - 1).  For a domain node the
# number of neighbours is simply "how many messages link to it".
deg = nx.degree_centrality(G)

# betweenness centrality = how often a node sits on the shortest path between
# two OTHER nodes. High betweenness = a bridge between otherwise separate
# groups. Exact betweenness is O(n*m); our graph is small (~4.5k nodes) so we
# can afford the exact value. (On a big graph you would pass k=500 to sample.)
bet = nx.betweenness_centrality(G, seed=SEED)

# ---------------------------------------------------------------------------
# 2. LOUVAIN COMMUNITIES  -- groups of nodes tightly connected to each other
# ---------------------------------------------------------------------------
# Louvain greedily moves nodes between groups to maximise MODULARITY:
#   modularity = (fraction of edges inside groups) - (fraction expected if the
#   edges were wired at random). Closer to 1 = clearer community structure.
# (networkx >= 2.8 ships this; no separate python-louvain install is needed.)
communities = nx.community.louvain_communities(G, seed=SEED)
partition = {node: cid for cid, comm in enumerate(communities) for node in comm}
Q = nx.community.modularity(G, communities)
print(f"Louvain: {len(communities)} communities, modularity Q = {Q:.3f}")

# ---------------------------------------------------------------------------
# 3. VALIDATION WITHOUT LEAKAGE
# ---------------------------------------------------------------------------
# If cluster risk is computed from ALL labels and then checked on those same
# messages, we are using the answer to predict the answer. So: hide the labels
# of 20% of messages, compute cluster risk from the other 80%, then test on
# the hidden 20%. The hidden messages stay in the graph (the structure is not
# secret) but their labels are not used.
labels = pd.Series({m: G.nodes[m]["label"] for m in messages})
train_ids, test_ids = train_test_split(
    labels.index, test_size=0.2, random_state=SEED, stratify=labels
)
msg_cluster = pd.Series({m: partition[m] for m in messages})
train_risk = labels[train_ids].groupby(msg_cluster[train_ids]).mean()
global_rate = labels[train_ids].mean()

def predict(m):
    # a cluster with no training messages falls back to the overall scam rate
    return train_risk.get(msg_cluster[m], global_rate)

test_score = pd.Series({m: predict(m) for m in test_ids})
test_pred = (test_score > 0.5).astype(int)
print("\nVALIDATION (cluster risk from 80% labels, tested on hidden 20%):")
print(classification_report(labels[test_ids], test_pred,
                            target_names=["legit", "scam"], digits=2))

# ---------------------------------------------------------------------------
# 4. FINAL TABLES (use all labels -- these describe the data, they are not a test)
# ---------------------------------------------------------------------------
cl = pd.DataFrame({"cluster": msg_cluster, "label": labels})
cluster_risk = cl.groupby("cluster").agg(
    n_messages=("label", "size"), scam_rate=("label", "mean")).reset_index()

# size of each community in nodes, and its busiest domain/phone, for readability
rows = []
for cid, comm in enumerate(communities):
    infra = [n for n in comm if G.nodes[n]["kind"] != "message"]
    top = max(infra, key=G.degree) if infra else ""
    rows.append((cid, len(comm), top))
cluster_risk = cluster_risk.merge(
    pd.DataFrame(rows, columns=["cluster", "n_nodes", "top_hub"]), on="cluster")
cluster_risk = cluster_risk.sort_values("n_messages", ascending=False)
cluster_risk.to_csv("network/cluster_risk.csv", index=False)

hub_rows = []
for n, kind in G.nodes(data="kind"):
    if kind == "message":
        continue
    labs = [G.nodes[m]["label"] for m in G.neighbors(n)]
    hub_rows.append({
        "node": n, "type": kind, "name": n.split("_", 1)[1],
        "n_messages": len(labs), "scam_rate": sum(labs) / len(labs),
        "degree_centrality": deg[n], "betweenness": bet[n],
        "cluster": partition[n],
    })
hubs = pd.DataFrame(hub_rows).sort_values("n_messages", ascending=False)
hubs.to_csv("network/hub_domains.csv", index=False)

pickle.dump(partition, open("network/partition.pkl", "wb"))

pd.set_option("display.width", 200)
print("TOP 10 BUSIEST HUBS:")
print(hubs.head(10)[["name", "type", "n_messages", "scam_rate", "betweenness"]].to_string(index=False))
print("\nTOP 10 SCAM-HEAVY HUBS (>= 5 messages):")
sc = hubs[hubs.n_messages >= 5].sort_values(["scam_rate", "n_messages"], ascending=False)
print(sc.head(10)[["name", "type", "n_messages", "scam_rate", "betweenness"]].to_string(index=False))
print("\nTOP 10 CLUSTERS BY SIZE:")
print(cluster_risk.head(10).to_string(index=False))
print("\nHigh-risk clusters (scam_rate >= 0.8, >= 5 msgs):",
      ((cluster_risk.scam_rate >= 0.8) & (cluster_risk.n_messages >= 5)).sum())
