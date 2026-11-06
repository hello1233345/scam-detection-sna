"""
draw_graph.py  --  Person B, Day 6
Draws 3 scam rings + 1 mixed cluster as a 2x2 figure -> network/graph_visual.png

Run from the repo root:   python network/draw_graph.py

Design choices (explain these in the demo):
  * Drawing all ~4,500 nodes gives an unreadable blob, so we draw 4 clusters.
  * Colour = scam (orange) vs legit (blue). Shape = node type, so the picture
    still reads for colour-blind viewers or in black-and-white print:
    circle = message, square = domain, diamond = phone number.
  * Only the hub of each cluster is labelled; labelling every node is noise.
"""
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import pandas as pd

SCAM, LEGIT = "#eb6834", "#2a78d6"          # categorical slots 2 and 1
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"

G = pickle.load(open("network/graph.pkl", "rb"))
partition = pickle.load(open("network/partition.pkl", "rb"))
cr = pd.read_csv("network/cluster_risk.csv")

rings = cr[(cr.scam_rate >= 0.8) & (cr.n_messages >= 5)].head(3)       # biggest pure scam rings
mixed = cr[(cr.scam_rate.between(0.2, 0.8)) & (cr.n_messages >= 50)].head(1)  # a mixed one
chosen = pd.concat([rings, mixed])

fig, axes = plt.subplots(2, 2, figsize=(12, 10), facecolor=SURFACE)
for ax, (_, c) in zip(axes.flat, chosen.iterrows()):
    nodes = [n for n, cid in partition.items() if cid == c.cluster]
    H = G.subgraph(nodes)
    pos = nx.spring_layout(H, seed=42, k=0.35 if len(H) > 100 else None)
    ax.set_facecolor(SURFACE)
    nx.draw_networkx_edges(H, pos, ax=ax, edge_color="#c3c2b7", width=0.6)
    def is_scam(n):
        """message: its own label. domain/phone: majority label of its messages."""
        if H.nodes[n]["kind"] == "message":
            return H.nodes[n]["label"] == 1
        labs = [G.nodes[m]["label"] for m in G.neighbors(n)]
        return sum(labs) / len(labs) >= 0.5

    for kind, shape, size in [("message", "o", 28), ("domain", "s", 90), ("phone", "D", 90)]:
        for scam, col in [(True, SCAM), (False, LEGIT)]:
            sel = [n for n in H if H.nodes[n]["kind"] == kind and is_scam(n) == scam]
            if sel:
                nx.draw_networkx_nodes(H, pos, nodelist=sel, ax=ax, node_color=col,
                                       node_shape=shape, node_size=size,
                                       edgecolors=SURFACE, linewidths=1.2)
    hub = c.top_hub
    ax.annotate(hub.split("_", 1)[1], pos[hub], xytext=(0, 12), textcoords="offset points",
                ha="center", fontsize=9, color=INK,
                bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec="none", alpha=0.85))
    ax.set_title(f"Cluster {int(c.cluster)}: {int(c.n_messages)} messages, "
                 f"{c.scam_rate:.0%} scam", fontsize=11, color=INK, loc="left")
    ax.axis("off")

legend = [
    Line2D([], [], marker="o", ls="", color=SCAM, label="scam message"),
    Line2D([], [], marker="o", ls="", color=LEGIT, label="legit message"),
    Line2D([], [], marker="o", ls="", color=MUTED, markersize=5, label="message"),
    Line2D([], [], marker="s", ls="", color=MUTED, label="domain"),
    Line2D([], [], marker="D", ls="", color=MUTED, label="phone number"),
]
fig.legend(handles=legend, loc="lower center", ncol=5, frameon=False,
           labelcolor=INK, fontsize=10)
fig.suptitle("Scam rings found by Louvain community detection", fontsize=14,
             color=INK, x=0.07, ha="left")
fig.tight_layout(rect=(0, 0.04, 1, 0.97))
fig.savefig("network/graph_visual.png", dpi=200, facecolor=SURFACE)
print("saved -> network/graph_visual.png")
