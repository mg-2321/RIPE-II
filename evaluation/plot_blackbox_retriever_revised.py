#!/usr/bin/env python3
"""High-quality matplotlib blackbox retriever plots for RIPE-II.

Replaces the PIL-drawn versions in evaluation/plots/:
  blackbox_retriever_risk_heatmap.*
  blackbox_poisoning_rate_sensitivity_bars.*   (grouped bar chart)
  blackbox_poisoning_rate_sensitivity_lines.*  (line / time-series chart)
  blackbox_exposure_to_top1_funnel.*

Run from the project root:
    python evaluation/plot_blackbox_retriever_revised.py
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
RATE_DIR = Path("/gscratch/uwb/gayat23/GuardRAG/results/component_study/rate_sweep_blackbox")
OUT = Path(__file__).parent / "plots"
OUT.mkdir(parents=True, exist_ok=True)

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.titlesize":     19,
    "axes.labelsize":     17,
    "xtick.labelsize":    15,
    "ytick.labelsize":    15,
    "legend.fontsize":    14,
    "figure.titlesize":   20,
    "figure.dpi":         180,
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
})

# ── Corpora ───────────────────────────────────────────────────────────────────
# Heatmap: no MSMARCO (incomplete retrievers), yes HotpotQA (all 8 present)
HEATMAP_CORPORA = ["nfcorpus", "scifact", "fiqa", "nq", "hotpotqa"]
# Rate sensitivity: all corpora with sweep data
RATE_CORPORA    = ["nfcorpus", "scifact", "fiqa", "nq", "msmarco", "hotpotqa"]
# Funnel: same as heatmap
FUNNEL_CORPORA  = ["nfcorpus", "scifact", "fiqa", "nq", "hotpotqa"]

CORPUS_LABEL = {
    "nfcorpus": "NFCorpus",
    "scifact":  "SciFact",
    "fiqa":     "FIQA",
    "nq":       "NQ",
    "msmarco":  "MSMARCO",
    "hotpotqa": "HotpotQA",
}
CORPUS_COLOR = {
    "nfcorpus": "#e63946",
    "scifact":  "#f4a261",
    "fiqa":     "#2a9d8f",
    "nq":       "#457b9d",
    "msmarco":  "#8338ec",
    "hotpotqa": "#fb8500",
}

# ── Retriever ordering & labels ───────────────────────────────────────────────
RETRIEVERS = [
    ("bm25",        "BM25"),
    ("bm25_ce",     "BM25+CE"),
    ("dense_e5",    "E5"),
    ("dense_e5_ce", "E5+CE"),
    ("hybrid",      "Hybrid"),
    ("hybrid_ce",   "Hybrid+CE"),
    ("splade",      "SPLADE++"),
    ("splade_ce",   "SPLADE++CE"),
]
RET_ORDER  = [k for k, _ in RETRIEVERS]
RET_LABEL  = dict(RETRIEVERS)

# Family colour for funnel plot (softer palette)
RET_COLOR = {
    "bm25":        "#1f77b4",
    "bm25_ce":     "#aec7e8",
    "dense_e5":    "#2ca02c",
    "dense_e5_ce": "#98df8a",
    "hybrid":      "#d62728",
    "hybrid_ce":   "#ff9896",
    "splade":      "#9467bd",
    "splade_ce":   "#c5b0d5",
}

# Rate-sweep retrievers available per corpus
RATE_RETS = {
    "fiqa":     [("bm25", "BM25"), ("dense_e5", "E5"), ("hybrid_ce", "Hybrid+CE")],
    "nfcorpus": [("dense_e5", "E5"), ("hybrid_ce", "Hybrid+CE"), ("splade_ce", "SPLADE++CE")],
    "nq":       [("bm25", "BM25"), ("dense_e5_ce", "E5+CE"), ("splade_ce", "SPLADE++CE")],
    "msmarco":  [("bm25", "BM25"), ("bm25_ce", "BM25+CE")],
    "scifact":  [("bm25", "BM25"), ("dense_e5", "E5"), ("hybrid_ce", "Hybrid+CE")],
    "hotpotqa": [("bm25", "BM25"), ("dense_e5_ce", "E5+CE"), ("hybrid_ce", "Hybrid+CE")],
}

# ── Main evaluation data (realistic ~3% poison rate) ─────────────────────────
#   columns: corpus, retriever_key, ER@10, Target-Retrieved, Target-Top-1
MAIN_DATA = [
    ("nfcorpus", "bm25",        91.0, 87.0, 57.0),
    ("nfcorpus", "bm25_ce",     91.0, 86.0, 45.0),
    ("nfcorpus", "dense_e5",    89.0, 86.0, 40.0),
    ("nfcorpus", "dense_e5_ce", 93.0, 89.0, 40.0),
    ("nfcorpus", "hybrid",      91.0, 88.0, 60.0),
    ("nfcorpus", "hybrid_ce",   92.0, 88.0, 41.0),
    ("nfcorpus", "splade",      89.0, 87.0, 50.0),
    ("nfcorpus", "splade_ce",   93.0, 91.0, 42.0),

    ("scifact",  "bm25",        87.4, 86.2, 65.5),
    ("scifact",  "bm25_ce",     88.5, 87.4, 67.8),
    ("scifact",  "dense_e5",    95.4, 93.1, 67.8),
    ("scifact",  "dense_e5_ce", 94.3, 93.1, 67.8),
    ("scifact",  "hybrid",      88.5, 87.4, 67.8),
    ("scifact",  "hybrid_ce",   88.5, 87.4, 67.8),
    ("scifact",  "splade",      95.4, 93.1, 64.4),
    ("scifact",  "splade_ce",   90.8, 89.7, 67.8),

    ("fiqa",     "bm25",        98.5, 98.5, 94.0),
    ("fiqa",     "bm25_ce",     98.0, 98.0, 89.5),
    ("fiqa",     "dense_e5",    31.5, 29.5, 12.0),
    ("fiqa",     "dense_e5_ce", 36.0, 32.5, 32.0),
    ("fiqa",     "hybrid",      98.0, 98.0, 94.0),
    ("fiqa",     "hybrid_ce",   98.0, 98.0, 88.5),
    ("fiqa",     "splade",      93.0, 93.0, 71.5),
    ("fiqa",     "splade_ce",   93.0, 93.0, 87.5),

    ("nq",       "bm25",        86.2, 84.8, 40.0),
    ("nq",       "bm25_ce",     90.0, 89.8, 63.4),
    ("nq",       "dense_e5",    96.0, 95.6, 60.0),
    ("nq",       "dense_e5_ce", 97.0, 96.6, 65.4),
    ("nq",       "hybrid",      89.2, 89.0, 49.8),
    ("nq",       "hybrid_ce",   93.6, 93.6, 64.6),
    ("nq",       "splade",      94.4, 94.2, 55.4),
    ("nq",       "splade_ce",   97.0, 97.0, 65.4),

    ("msmarco",  "bm25",        97.3, 97.3, 77.7),
    ("msmarco",  "bm25_ce",     98.7, 98.7, 75.0),
    ("msmarco",  "dense_e5",    71.0, 71.0, 33.7),

    ("hotpotqa", "bm25",        84.8, 83.6, 60.6),
    ("hotpotqa", "bm25_ce",     86.6, 85.4, 75.2),
    ("hotpotqa", "dense_e5",    90.6, 89.8, 61.8),
    ("hotpotqa", "dense_e5_ce", 91.8, 91.2, 75.2),
    ("hotpotqa", "hybrid",      87.4, 86.4, 65.8),
    ("hotpotqa", "hybrid_ce",   91.2, 90.4, 76.2),
    ("hotpotqa", "splade",      86.6, 85.4, 62.4),
    ("hotpotqa", "splade_ce",   89.2, 88.0, 73.0),
]
# keyed lookup: (corpus, ret_key) → (er, tr, top1)
MAIN = {(c, r): (er, tr, t1) for c, r, er, tr, t1 in MAIN_DATA}

# ── Rate sweep data loader ────────────────────────────────────────────────────
RATE_INTS = [25, 50, 100]

def load_rate_sweep():
    """Returns {(corpus, ret_key, rate_int): {er, tr, top1}}."""
    result = {}
    rate_map = {"025": 25, "050": 50, "100": 100}
    for f in RATE_DIR.glob("*.summary.json"):
        stem = f.stem.replace(".summary", "")
        if "_rate" not in stem:
            continue
        corpus, rest = stem.split("_rate", 1)
        rate_str, ret_key = rest[:3], rest[4:]
        ALL_CORPORA = set(HEATMAP_CORPORA) | set(RATE_CORPORA) | set(FUNNEL_CORPORA)
        if corpus not in ALL_CORPORA or rate_str not in rate_map:
            continue
        d = json.loads(f.read_text())
        result[(corpus, ret_key, rate_map[rate_str])] = {
            "er":   float(d.get("poison_exposure_pct",   0.0)),
            "tr":   float(d.get("target_retrieved_pct",  0.0)),
            "top1": float(d.get("target_top1_pct",       0.0)),
        }
    return result


# ═════════════════════════════════════════════════════════════════════════════
# PLOT 1 — Risk Heatmap  (Top-1 % by retriever × corpus)
# ═════════════════════════════════════════════════════════════════════════════
def plot_heatmap():
    corpora = HEATMAP_CORPORA
    n_corp = len(corpora)
    n_ret  = len(RETRIEVERS)

    matrix  = np.full((n_corp, n_ret), np.nan)
    er_mat  = np.full((n_corp, n_ret), np.nan)
    for ci, corpus in enumerate(corpora):
        for ri, (rkey, _) in enumerate(RETRIEVERS):
            row = MAIN.get((corpus, rkey))
            if row:
                er_mat[ci, ri], _, matrix[ci, ri] = row

    # find best (lowest Top-1) and worst (highest Top-1) per corpus
    best_col  = [int(np.nanargmin(matrix[ci])) if not np.all(np.isnan(matrix[ci])) else -1
                 for ci in range(n_corp)]
    worst_col = [int(np.nanargmax(matrix[ci])) if not np.all(np.isnan(matrix[ci])) else -1
                 for ci in range(n_corp)]

    fig, ax = plt.subplots(figsize=(15, 7))
    cmap = matplotlib.colormaps["RdYlGn_r"].copy()   # green=safe, red=high risk
    cmap.set_bad("#e8e8e8")

    im = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap,
                   vmin=0, vmax=100, aspect="auto")

    # cell annotations
    for ci in range(n_corp):
        for ri in range(n_ret):
            if np.isnan(matrix[ci, ri]):
                ax.text(ri, ci, "—", ha="center", va="center",
                        fontsize=14, color="#555")
                continue
            top1 = matrix[ci, ri]
            er   = er_mat[ci, ri]
            ax.text(ri, ci - 0.18, f"{top1:.0f}%",
                    ha="center", va="center", fontsize=15,
                    fontweight="bold", color="black")
            ax.text(ri, ci + 0.22, f"ER {er:.0f}%",
                    ha="center", va="center", fontsize=13,
                    color="black")

    # border colours — both must be visible on coloured cells AND white legend bg
    BEST_CLR  = "#c0c0c0"   # light silver: visible on orange/red cells and on white paper
    WORST_CLR = "#111111"   # near-black

    # highlight best/worst: inset 0.40 from cell centre so borders stay well clear
    # of the white grid lines that sit exactly at ±0.5 boundaries
    INSET = 0.40
    SZ    = INSET * 2
    for ci in range(n_corp):
        if best_col[ci] >= 0:
            ri = best_col[ci]
            rect = plt.Rectangle((ri - INSET, ci - INSET), SZ, SZ,
                                  linewidth=2.5, edgecolor=BEST_CLR,
                                  facecolor="none", zorder=5)
            ax.add_patch(rect)
        if worst_col[ci] >= 0 and worst_col[ci] != best_col[ci]:
            ri = worst_col[ci]
            rect = plt.Rectangle((ri - INSET, ci - INSET), SZ, SZ,
                                  linewidth=2.5, edgecolor=WORST_CLR,
                                  facecolor="none", zorder=5)
            ax.add_patch(rect)

    ax.set_xticks(range(n_ret))
    ax.set_xticklabels([lbl for _, lbl in RETRIEVERS], fontsize=14,
                        rotation=40, ha="right")
    ax.tick_params(axis="x", pad=6)
    ax.set_yticks(range(n_corp))
    ax.set_yticklabels([CORPUS_LABEL[c] for c in corpora], fontsize=15,
                        fontweight="bold")
    ax.set_ylabel("Dataset", fontsize=16, labelpad=10)

    # grid lines
    for x in np.arange(0.5, n_ret - 1, 1):
        ax.axvline(x, color="white", lw=1.5, zorder=3)
    for y in np.arange(0.5, n_corp - 1, 1):
        ax.axhline(y, color="white", lw=1.5, zorder=3)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, aspect=25)
    cbar.set_label("Target Top-1  (%)", fontsize=15)
    cbar.ax.tick_params(labelsize=13)

    # legend — facecolor="none" so border colour matches exactly what's on the cells
    worst_patch = mpatches.Patch(edgecolor=WORST_CLR, facecolor="none",
                                  linewidth=2.5, label="Worst per corpus (highest Top-1)")
    best_patch  = mpatches.Patch(edgecolor=BEST_CLR,  facecolor="none",
                                  linewidth=2.5, label="Best per corpus (lowest Top-1)")
    # place legend centred below x-tick labels at the same vertical level
    ax.legend(handles=[worst_patch, best_patch], loc="upper center",
              bbox_to_anchor=(0.5, -0.22), framealpha=0.92, ncol=2,
              fontsize=13, edgecolor="#ccc")

    ax.set_title(
        "Retriever Risk Landscape — Target Top-1 Poison Control\n"
        "Cell: Top-1% · ER@10% · Realistic poison rate",
        fontsize=17, pad=12,
    )
    fig.tight_layout()
    for ext in (".png", ".pdf"):
        fig.savefig(OUT / f"blackbox_retriever_risk_heatmap{ext}",
                    bbox_inches="tight", dpi=220)
    print("Saved: blackbox_retriever_risk_heatmap")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# PLOT 2 — Rate Sensitivity: Grouped bar chart (ER@10 by poison rate)
# ═════════════════════════════════════════════════════════════════════════════
RATE_COLORS = {
    "realistic": "#a8dadc",   # mint  (~3%)
    25:          "#ffd166",   # yellow
    50:          "#ef9c57",   # orange
    100:         "#e63946",   # red
}
RATE_LABELS = {
    "realistic": "Realistic",
    25:          "25%",
    50:          "50%",
    100:         "100%",
}

def _rate_vals(corpus, rate_data):
    """Return list of (ret_label, [realistic, 25, 50, 100]) for a corpus."""
    result = []
    for rkey, rlabel in RATE_RETS[corpus]:
        realistic = MAIN.get((corpus, rkey))
        er_real   = realistic[0] if realistic else np.nan
        sweep     = [rate_data.get((corpus, rkey, r), {}).get("er", np.nan)
                     for r in RATE_INTS]
        result.append((rlabel, [er_real] + sweep))
    return result


def _draw_bar_group(axes, group, rate_data, bar_width, rate_order):
    """Shared helper — fills one row of 3 bar subplots."""
    n_rates = len(rate_order)
    for ax, corpus in zip(axes, group):
        rets = _rate_vals(corpus, rate_data)
        xs   = np.arange(len(rets))
        for ri, rate in enumerate(rate_order):
            offsets = xs + (ri - n_rates / 2 + 0.5) * bar_width
            vals    = [rv[ri] for _, rv in rets]
            ax.bar(offsets, vals, width=bar_width * 0.88,
                   color=RATE_COLORS[rate], edgecolor="white",
                   linewidth=0.8, zorder=3)
        ax.set_xticks(xs)
        ax.set_xticklabels([lbl for lbl, _ in rets], fontsize=20,
                            rotation=40, ha="right")
        ax.tick_params(axis="y", labelsize=19)
        ax.set_ylim(0, 110)
        ax.set_title(CORPUS_LABEL[corpus], fontsize=24,
                     fontweight="bold", color="#222222", pad=10)
        ax.yaxis.set_major_locator(plt.MultipleLocator(20))
        ax.grid(axis="y", alpha=0.3, linestyle="--", color="#bbb", zorder=0)
        ax.set_axisbelow(True)


def plot_rate_bars():
    """Two separate 1×3 bar charts — 3 corpora each."""
    rate_data  = load_rate_sweep()
    bar_width  = 0.18
    rate_order = ["realistic", 25, 50, 100]

    legend_handles = [mpatches.Patch(color=RATE_COLORS[r], label=RATE_LABELS[r])
                      for r in rate_order]

    groups = [
        ("1", RATE_CORPORA[:3]),   # NFCorpus, SciFact, FIQA
        ("2", RATE_CORPORA[3:]),   # NQ, MSMARCO, HotpotQA
    ]

    for suffix, group in groups:
        # 16" wide: fonts at ~2× print size so they read ≥9pt when scaled to 7" textwidth
        fig, axes = plt.subplots(1, 3, figsize=(16, 7),
                                 gridspec_kw={"wspace": 0.44})
        _draw_bar_group(axes, group, rate_data, bar_width, rate_order)

        fig.legend(handles=legend_handles, title="Poison rate",
                   title_fontsize=20, fontsize=20,
                   loc="upper center", bbox_to_anchor=(0.5, 0.0),
                   framealpha=0.95, edgecolor="#ccc",
                   ncol=4, handlelength=1.8, columnspacing=1.2)

        fig.supylabel("Exposure Rate — ER@10 (%)", fontsize=21, x=0.01)
        fig.suptitle("Poison Rate Sensitivity — Exposure Rate (ER@10)",
                     fontsize=24, y=1.01)
        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.22, top=0.90)

        for ext in (".png", ".pdf"):
            fig.savefig(OUT / f"blackbox_poisoning_rate_sensitivity_bars_{suffix}{ext}",
                        bbox_inches="tight", dpi=300)
        print(f"Saved: blackbox_poisoning_rate_sensitivity_bars_{suffix}")
        plt.close(fig)


def plot_rate_lines():
    """Line / time-series chart — one line per retriever, X = poison rate."""
    rate_data = load_rate_sweep()

    # X positions: use actual numeric rates but with ≈log spacing for clarity
    # realistic ≈ 3.5%, then 25, 50, 100
    X_VALS   = [3.5, 25, 50, 100]
    X_LABELS = ["Realistic\n(~3%)", "25%", "50%", "100%"]

    # retriever line styles (cycle through a vivid palette)
    LINE_STYLES = ["-", "--", "-."]
    LINE_COLORS = ["#1f77b4", "#2ca02c", "#d62728"]
    MARKERS     = ["o", "s", "^"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5),
                             gridspec_kw={"hspace": 0.52, "wspace": 0.40})
    corpus_axes = list(zip(RATE_CORPORA, axes.flat))

    for corpus, ax in corpus_axes:
        rets = _rate_vals(corpus, rate_data)

        for idx, (rlabel, yvals) in enumerate(rets):
            color  = LINE_COLORS[idx % len(LINE_COLORS)]
            ls     = LINE_STYLES[idx % len(LINE_STYLES)]
            marker = MARKERS[idx % len(MARKERS)]

            # split into realistic star + sweep line
            y_real  = yvals[0]
            y_sweep = yvals[1:]   # 25, 50, 100

            # sweep line (solid)
            valid_x = [X_VALS[i+1] for i, v in enumerate(y_sweep) if not np.isnan(v)]
            valid_y = [v for v in y_sweep if not np.isnan(v)]
            if valid_x:
                ax.plot(valid_x, valid_y, color=color, ls=ls, lw=2.4,
                        marker=marker, ms=7, zorder=4, label=rlabel)
                # dashed connector from realistic to first sweep point
                if not np.isnan(y_real) and valid_x:
                    ax.plot([X_VALS[0], valid_x[0]], [y_real, valid_y[0]],
                            color=color, ls=":", lw=1.4, alpha=0.55, zorder=3)

            # realistic star
            if not np.isnan(y_real):
                ax.scatter([X_VALS[0]], [y_real], color=color, marker="*",
                           s=220, zorder=5, edgecolors="white", linewidths=0.8)

        # shade realistic zone
        ax.axvspan(0, 8, alpha=0.10, color="#2a9d8f", zorder=0)
        ax.text(5.5, 5, "Realistic\nzone", fontsize=10, color="#2a9d8f",
                ha="center", va="bottom", style="italic")

        ax.set_xlim(-2, 107)
        ax.set_ylim(0, 108)
        ax.set_xticks(X_VALS)
        ax.set_xticklabels(X_LABELS, fontsize=11)
        ax.set_xlabel("Poison rate", fontsize=12)
        ax.set_ylabel("ER@10  (%)", fontsize=12)
        ax.set_title(CORPUS_LABEL[corpus], fontsize=14, fontweight="bold",
                     color=CORPUS_COLOR[corpus], pad=6)
        ax.yaxis.set_major_locator(plt.MultipleLocator(20))
        ax.grid(alpha=0.28, linestyle="--", color="#bbb", zorder=0)
        ax.legend(fontsize=10.5, loc="upper left", framealpha=0.88,
                  edgecolor="#ddd", handlelength=2)

    fig.suptitle(
        "Poison Rate Sensitivity — Exposure Rate (ER@10)\n"
        "★ = realistic baseline  ·  lines = rate-sweep  ·  shaded = realistic zone (~3%)",
        fontsize=15, y=1.01,
    )
    for ext in (".png", ".pdf"):
        fig.savefig(OUT / f"blackbox_poisoning_rate_sensitivity_lines{ext}",
                    bbox_inches="tight", dpi=220)
    print("Saved: blackbox_poisoning_rate_sensitivity_lines")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# PLOT 3 — Exposure-to-Top-1 Funnel (parallel coordinates, matplotlib)
# ═════════════════════════════════════════════════════════════════════════════
STAGE_KEYS   = ["er", "tr", "top1"]
STAGE_LABELS = ["ER@10\n(Exposure)", "Target\nRetrieved", "Target\nTop-1"]

def _best_worst(corpus_rows):
    """Return (best_rkey, worst_rkey) by Top-1 (lowest=best, highest=worst)."""
    top1s = {rkey: top1 for _, rkey, _, _, top1 in corpus_rows}
    if not top1s:
        return None, None
    best  = min(top1s, key=top1s.__getitem__)
    worst = max(top1s, key=top1s.__getitem__)
    if best == worst:
        worst = None
    return best, worst

def plot_funnel():
    fig, axes = plt.subplots(2, 3, figsize=(14, 8),
                             gridspec_kw={"hspace": 0.55, "wspace": 0.38})
    corpus_axes = [
        ("nfcorpus", axes[0, 0]),
        ("scifact",  axes[0, 1]),
        ("fiqa",     axes[0, 2]),
        ("nq",       axes[1, 0]),
        ("hotpotqa", axes[1, 1]),
    ]
    axes[1, 2].set_visible(False)

    xs = np.array([0, 1, 2])

    for corpus, ax in corpus_axes:
        corpus_rows = [row for row in MAIN_DATA if row[0] == corpus]
        best_rkey, worst_rkey = _best_worst(corpus_rows)

        for _, rkey, er, tr, top1 in corpus_rows:
            ys = np.array([er, tr, top1])
            is_best  = (rkey == best_rkey)
            is_worst = (rkey == worst_rkey)
            color = RET_COLOR.get(rkey, "#888")

            if is_best or is_worst:
                ax.plot(xs, ys, color=color, lw=2.8, marker="o",
                        ms=6, zorder=4, alpha=0.95,
                        label=f"{'Best' if is_best else 'Worst'}: {RET_LABEL[rkey]}")
                ax.annotate(
                    f"{RET_LABEL[rkey]}\n({top1:.0f}%)",
                    xy=(2, top1), xytext=(2.08, top1),
                    fontsize=10, va="center", color=color,
                    fontweight="bold",
                )
            else:
                ax.plot(xs, ys, color=color, lw=1.2, marker="o",
                        ms=3.5, zorder=2, alpha=0.38)

        # shade the promotion gap (TR→Top1) for all lines lightly
        ax.fill_betweenx([0, 100], 1, 2, color="#f0f0f0", zorder=0, alpha=0.5)
        ax.text(1.5, 4, "promotion\ngap", ha="center", fontsize=10,
                color="#bbb", style="italic")

        ax.set_xticks(xs)
        ax.set_xticklabels(STAGE_LABELS, fontsize=11)
        ax.set_ylim(0, 105)
        ax.set_ylabel("Queries  (%)", fontsize=12)
        ax.set_title(CORPUS_LABEL[corpus], fontsize=14, fontweight="bold",
                     color=CORPUS_COLOR[corpus], pad=6)
        ax.yaxis.set_major_locator(plt.MultipleLocator(25))
        ax.grid(axis="y", alpha=0.25, linestyle="--", color="#bbb", zorder=0)
        ax.set_axisbelow(True)
        for sx in xs:
            ax.axvline(sx, color="#ddd", lw=0.8, zorder=1)

        # minimal best/worst legend per panel
        if best_rkey:
            ax.legend(fontsize=10, loc="lower left",
                      framealpha=0.85, edgecolor="#ddd",
                      handlelength=1.4, labelspacing=0.3)

    # shared retriever legend at bottom
    ret_handles = [
        mlines.Line2D([], [], color=RET_COLOR[k], lw=2, marker="o", ms=5,
                      label=RET_LABEL[k])
        for k, _ in RETRIEVERS
        if k in RET_COLOR
    ]
    fig.legend(handles=ret_handles, title="Retriever",
               title_fontsize=12, fontsize=10.5,
               loc="lower right", bbox_to_anchor=(0.97, 0.04),
               framealpha=0.93, edgecolor="#ddd", ncol=2)

    fig.suptitle(
        "Exposure-to-Top-1 Funnel — Pipeline Stage Dropout\n"
        "Each line: ER@10 → Target Retrieved → Target Top-1  "
        "·  bold = best / worst per corpus",
        fontsize=15, y=1.01,
    )
    for ext in (".png", ".pdf"):
        fig.savefig(OUT / f"blackbox_exposure_to_top1_funnel{ext}",
                    bbox_inches="tight", dpi=220)
    print("Saved: blackbox_exposure_to_top1_funnel")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# BONUS: CE Effect Heatmap (Δ Top-1 from adding cross-encoder reranker)
# ═════════════════════════════════════════════════════════════════════════════
def plot_ce_delta():
    BASE_RETS = ["bm25", "dense_e5", "hybrid", "splade"]
    BASE_LABELS = ["BM25", "E5", "Hybrid", "SPLADE++"]
    corpora = HEATMAP_CORPORA

    delta = np.full((len(corpora), len(BASE_RETS)), np.nan)
    for ci, corpus in enumerate(corpora):
        for ri, base in enumerate(BASE_RETS):
            brow = MAIN.get((corpus, base))
            crow = MAIN.get((corpus, f"{base}_ce"))
            if brow and crow:
                delta[ci, ri] = crow[2] - brow[2]   # Top-1 delta

    vmax = max(abs(np.nanmin(delta)), abs(np.nanmax(delta)), 1)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    cmap = matplotlib.colormaps["RdYlGn"].copy()
    cmap.set_bad("#e8e8e8")

    im = ax.imshow(np.ma.masked_invalid(delta), cmap=cmap,
                   vmin=-vmax, vmax=vmax, aspect="auto")

    for ci in range(len(corpora)):
        for ri in range(len(BASE_RETS)):
            if np.isnan(delta[ci, ri]):
                ax.text(ri, ci, "—", ha="center", va="center",
                        fontsize=15, color="#aaa")
                continue
            d = delta[ci, ri]
            sign = "+" if d >= 0 else ""
            text_color = "white" if abs(d) > vmax * 0.55 else "#222"
            ax.text(ri, ci, f"{sign}{d:.1f}pp", ha="center", va="center",
                    fontsize=15, fontweight="bold", color=text_color)

    for x in np.arange(0.5, len(BASE_RETS) - 1, 1):
        ax.axvline(x, color="white", lw=2, zorder=3)
    for y in np.arange(0.5, len(corpora) - 1, 1):
        ax.axhline(y, color="white", lw=2, zorder=3)

    ax.set_xticks(range(len(BASE_RETS)))
    ax.set_xticklabels(BASE_LABELS, fontsize=15, fontweight="bold")
    ax.set_yticks(range(len(corpora)))
    ax.set_yticklabels([CORPUS_LABEL[c] for c in corpora], fontsize=15,
                        fontweight="bold")
    ax.set_xlabel("Base retriever  (before CE reranking)", fontsize=16, labelpad=10)
    ax.set_ylabel("Dataset", fontsize=16, labelpad=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, aspect=20)
    cbar.set_label("Δ Target Top-1  (pp)", fontsize=15)
    cbar.ax.tick_params(labelsize=13)

    ax.set_title(
        "Cross-Encoder Reranker Effect on Attack Precision\n"
        "Green = CE increases attack Top-1  ·  Red = CE reduces it",
        fontsize=17, pad=12,
    )
    fig.tight_layout()
    for ext in (".png", ".pdf"):
        fig.savefig(OUT / f"blackbox_ce_delta_heatmap{ext}",
                    bbox_inches="tight", dpi=220)
    print("Saved: blackbox_ce_delta_heatmap")
    plt.close(fig)


if __name__ == "__main__":
    plot_heatmap()
    plot_rate_bars()
    plot_rate_lines()
    plot_funnel()
    plot_ce_delta()
    print("All plots written to", OUT)
