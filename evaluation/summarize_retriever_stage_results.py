#!/usr/bin/env python3
"""
Build RIPE-II retriever-stage tables and paper figures.

Author: Gayatri Malladi
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path("/mmfs1/home/gayat23/projects/guardrag-thesis")
COMPONENT_ROOT = Path("/gscratch/uwb/gayat23/GuardRAG/results/component_study")
RATE_COMPONENT_ROOT = COMPONENT_ROOT / "rate_sweep_blackbox"
OUT_DIR = PROJECT_ROOT / "results" / "retriever_stage"
PLOT_DIR = PROJECT_ROOT / "evaluation" / "plots"

CORPORA = ["nfcorpus", "scifact", "fiqa", "nq", "msmarco", "hotpotqa"]
RATE_CORPORA = ["nq", "msmarco", "hotpotqa"]
RATES = ["025", "050", "100"]

COMBOS: List[Tuple[str, str]] = [
    ("bm25", "BM25"),
    ("bm25_ce", "BM25 + CE"),
    ("dense_e5", "E5"),
    ("dense_e5_ce", "E5 + CE"),
    ("hybrid", "Hybrid"),
    ("hybrid_ce", "Hybrid + CE"),
    ("splade", "SPLADE++"),
    ("splade_ce", "SPLADE++ + CE"),
]

AVAILABLE_RATE_COMBOS = {
    "nq": ["bm25", "dense_e5_ce", "splade_ce"],
    "msmarco": ["bm25", "bm25_ce"],
    "hotpotqa": ["bm25", "dense_e5_ce", "hybrid_ce"],
}

CORPUS_LABELS = {
    "nfcorpus": "NFCorpus",
    "scifact": "SciFact",
    "fiqa": "FIQA",
    "nq": "NQ",
    "msmarco": "MSMARCO",
    "hotpotqa": "HotpotQA",
}

PALETTE = {
    "bm25": "#1B6CA8",
    "bm25_ce": "#5AA9E6",
    "dense_e5": "#2F8F5B",
    "dense_e5_ce": "#7BC96F",
    "hybrid": "#C46A2B",
    "hybrid_ce": "#F2A541",
    "splade": "#7A4D9E",
    "splade_ce": "#C678DD",
}

STAGE_LABELS = ["Poison Exposure", "Target Retrieved", "Target Top-1"]


@dataclass(frozen=True)
class RetrieverRow:
    corpus: str
    retriever: str
    label: str
    n: int
    poison_exposure_pct: float
    target_retrieved_pct: float
    target_top1_pct: float
    top1_poisoned_pct: float
    mean_poison_count_final_topk: float
    mean_target_rank_retrieved: float
    median_target_rank_retrieved: float
    status: str
    source: str
    rate: str = ""
    role: str = ""

    def selection_key(self) -> tuple[float, float, float, float]:
        """Security-risk ordering for best/worst retriever selection."""
        return (
            self.poison_exposure_pct,
            self.target_retrieved_pct,
            self.target_top1_pct,
            self.top1_poisoned_pct,
        )


def load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def row_from_summary(
    corpus: str,
    combo: str,
    label: str,
    path: Path,
    rate: str = "",
    role: str = "",
) -> RetrieverRow:
    data = load_json(path)
    if data is None:
        return RetrieverRow(
            corpus=corpus,
            retriever=combo,
            label=label,
            n=0,
            poison_exposure_pct=0.0,
            target_retrieved_pct=0.0,
            target_top1_pct=0.0,
            top1_poisoned_pct=0.0,
            mean_poison_count_final_topk=0.0,
            mean_target_rank_retrieved=0.0,
            median_target_rank_retrieved=0.0,
            status="missing",
            source=str(path),
            rate=rate,
            role=role,
        )
    return RetrieverRow(
        corpus=corpus,
        retriever=combo,
        label=label,
        n=int(data.get("n", 0)),
        poison_exposure_pct=float(data.get("poison_exposure_pct", 0.0)),
        target_retrieved_pct=float(data.get("target_retrieved_pct", 0.0)),
        target_top1_pct=float(data.get("target_top1_pct", 0.0)),
        top1_poisoned_pct=float(data.get("top1_poisoned_pct", 0.0)),
        mean_poison_count_final_topk=float(data.get("mean_poison_count_final_topk", 0.0)),
        mean_target_rank_retrieved=float(data.get("mean_target_rank_retrieved", 0.0)),
        median_target_rank_retrieved=float(data.get("median_target_rank_retrieved", 0.0)),
        status="complete",
        source=str(path),
        rate=rate,
        role=role,
    )


def collect_main_rows() -> List[RetrieverRow]:
    rows: List[RetrieverRow] = []
    for corpus in CORPORA:
        for combo, label in COMBOS:
            path = COMPONENT_ROOT / f"{corpus}_main_candidate_{combo}.summary.json"
            rows.append(row_from_summary(corpus, combo, label, path))
    return rows


def complete_rows(rows: Iterable[RetrieverRow]) -> List[RetrieverRow]:
    return [row for row in rows if row.status == "complete"]


def best_worst_map(rows: List[RetrieverRow]) -> Dict[str, Tuple[RetrieverRow, RetrieverRow]]:
    mapping: Dict[str, Tuple[RetrieverRow, RetrieverRow]] = {}
    for corpus in CORPORA:
        corpus_rows = [r for r in rows if r.corpus == corpus and r.status == "complete"]
        if not corpus_rows:
            continue
        worst = min(corpus_rows, key=lambda r: r.selection_key())
        best = max(corpus_rows, key=lambda r: r.selection_key())
        mapping[corpus] = (worst, best)
    return mapping


def best_worst_rows(rows: List[RetrieverRow]) -> List[RetrieverRow]:
    selected: List[RetrieverRow] = []
    for corpus, (worst, best) in best_worst_map(rows).items():
        selected.append(replace(worst, role="worst"))
        if best.retriever != worst.retriever:
            selected.append(replace(best, role="best"))
        else:
            selected[-1] = replace(selected[-1], role="best_and_worst")
    return selected


def rate_pair_for_corpus(corpus: str, bw: Dict[str, Tuple[RetrieverRow, RetrieverRow]]) -> List[Tuple[str, str]]:
    available = set(AVAILABLE_RATE_COMBOS.get(corpus, []))
    pairs: List[Tuple[str, str]] = []
    if corpus in bw:
        worst, best = bw[corpus]
        for role, row in [("worst", worst), ("best", best)]:
            if row.retriever in available:
                pairs.append((role, row.retriever))
    for combo in AVAILABLE_RATE_COMBOS.get(corpus, []):
        if combo not in {c for _, c in pairs}:
            label = "comparison" if pairs else "available"
            pairs.append((label, combo))
    # Keep tables compact: best/worst when available, otherwise at most 3 comparison lines.
    return pairs[:3]


def collect_rate_rows(main_rows: List[RetrieverRow]) -> List[RetrieverRow]:
    rows: List[RetrieverRow] = []
    labels = dict(COMBOS)
    bw = best_worst_map(main_rows)
    for corpus in RATE_CORPORA:
        for role, combo in rate_pair_for_corpus(corpus, bw):
            for rate in RATES:
                path = RATE_COMPONENT_ROOT / f"{corpus}_rate{rate}_{combo}.summary.json"
                rows.append(row_from_summary(corpus, combo, labels[combo], path, rate=rate, role=role))
    return rows


def write_csv(path: Path, rows: Iterable[RetrieverRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "corpus",
        "rate",
        "retriever",
        "label",
        "role",
        "status",
        "n",
        "poison_exposure_pct",
        "target_retrieved_pct",
        "target_top1_pct",
        "top1_poisoned_pct",
        "mean_poison_count_final_topk",
        "mean_target_rank_retrieved",
        "median_target_rank_retrieved",
        "source",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: getattr(row, name) for name in fieldnames})


def fmt(value: float) -> str:
    return f"{value:.1f}"


def write_markdown_table(path: Path, rows: List[RetrieverRow], title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {title}",
        "",
        "| Corpus | Rate | Role | Retriever | Status | N | ER@10 | Target Retrieved | Target Top-1 | Top-1 Poisoned | Mean Poisons@10 | Mean Target Rank | Median Target Rank |",
        "| --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    CORPUS_LABELS.get(row.corpus, row.corpus),
                    row.rate or "-",
                    row.role or "-",
                    row.label,
                    row.status,
                    str(row.n),
                    fmt(row.poison_exposure_pct),
                    fmt(row.target_retrieved_pct),
                    fmt(row.target_top1_pct),
                    fmt(row.top1_poisoned_pct),
                    f"{row.mean_poison_count_final_topk:.3f}",
                    f"{row.mean_target_rank_retrieved:.2f}",
                    f"{row.median_target_rank_retrieved:.2f}",
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    font_path = "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf"
    return ImageFont.truetype(font_path, size=size)


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont) -> Tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0], box[3] - box[1]


def draw_centered(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[float, float],
    text: str,
    fnt: ImageFont.FreeTypeFont,
    fill: str = "#111111",
) -> None:
    w, h = text_size(draw, text, fnt)
    draw.text((xy[0] - w / 2, xy[1] - h / 2), text, font=fnt, fill=fill)


def metric_values(row: RetrieverRow) -> List[float]:
    return [row.poison_exposure_pct, row.target_retrieved_pct, row.target_top1_pct]


def save_with_pdf(img: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    img.save(path.with_suffix(".pdf"), "PDF", resolution=220.0)


def draw_heatmap(rows: List[RetrieverRow], out_path: Path) -> None:
    width, height = 1850, 1050
    left, top = 270, 150
    cell_w, cell_h = 180, 105
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    f_title, f_axis, f_cell, f_small = font(34, True), font(19, True), font(20, True), font(16)

    draw.text((90, 42), "Retriever Risk Landscape: Target Top-1 Poison Control", font=f_title, fill="#111111")
    draw.text((90, 88), "Cell value is target poisoned document ranked #1 in final top-k (%)", font=f_small, fill="#555555")

    by_key = {(row.corpus, row.retriever): row for row in rows if row.status == "complete"}
    bw = best_worst_map(rows)
    values = [row.target_top1_pct for row in by_key.values()]
    vmax = max(values) if values else 100.0

    for ci, (_, label) in enumerate(COMBOS):
        x = left + ci * cell_w + cell_w / 2
        draw_centered(draw, (x, top - 38), label.replace(" + ", "\n+ "), f_axis, "#222222")

    for ri, corpus in enumerate(CORPORA):
        y = top + ri * cell_h
        draw_centered(draw, (left - 105, y + cell_h / 2), CORPUS_LABELS[corpus], f_axis, "#222222")
        worst, best = bw.get(corpus, (None, None))
        for ci, (combo, _) in enumerate(COMBOS):
            x = left + ci * cell_w
            row = by_key.get((corpus, combo))
            if row is None:
                fill = "#F0F0F0"
                label = "missing"
                text_fill = "#777777"
            else:
                t = row.target_top1_pct / max(vmax, 1.0)
                # Green to gold to red, where deeper red means higher security risk.
                if t < 0.5:
                    alpha = t / 0.5
                    r = int(235 * alpha + 46 * (1 - alpha))
                    g = int(183 * alpha + 125 * (1 - alpha))
                    b = int(76 * alpha + 95 * (1 - alpha))
                else:
                    alpha = (t - 0.5) / 0.5
                    r = int(193 * alpha + 235 * (1 - alpha))
                    g = int(73 * alpha + 183 * (1 - alpha))
                    b = int(57 * alpha + 76 * (1 - alpha))
                fill = f"#{r:02x}{g:02x}{b:02x}"
                label = f"{row.target_top1_pct:.1f}"
                text_fill = "white" if t > 0.58 else "#111111"
            draw.rounded_rectangle((x + 7, y + 7, x + cell_w - 7, y + cell_h - 7), radius=10, fill=fill)
            if best and combo == best.retriever:
                draw.rounded_rectangle((x + 5, y + 5, x + cell_w - 5, y + cell_h - 5), radius=12, outline="#111111", width=5)
            if worst and combo == worst.retriever:
                draw.rounded_rectangle((x + 11, y + 11, x + cell_w - 11, y + cell_h - 11), radius=8, outline="#FFFFFF", width=4)
            draw_centered(draw, (x + cell_w / 2, y + cell_h / 2), label, f_cell, text_fill)

    legend_y = top + len(CORPORA) * cell_h + 45
    draw.rounded_rectangle((left, legend_y, left + 28, legend_y + 28), radius=6, fill="#C14939")
    draw.text((left + 40, legend_y + 2), "higher target top-1 risk", font=f_small, fill="#333333")
    draw.rounded_rectangle((left + 330, legend_y, left + 358, legend_y + 28), radius=6, outline="#111111", width=4)
    draw.text((left + 370, legend_y + 2), "best per corpus", font=f_small, fill="#333333")
    draw.rounded_rectangle((left + 590, legend_y, left + 618, legend_y + 28), radius=6, outline="#777777", width=4)
    draw.text((left + 630, legend_y + 2), "worst per corpus", font=f_small, fill="#333333")
    save_with_pdf(img, out_path)


def draw_funnel(rows: List[RetrieverRow], out_path: Path) -> None:
    width, height = 2000, 1320
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    f_title, f_axis, f_label, f_small = font(34, True), font(18, True), font(15), font(13)

    draw.text((80, 34), "Exposure-to-Top-1 Funnel Across Blackbox Retriever Settings", font=f_title, fill="#111111")
    draw.text((80, 82), "Each line tracks ER@10 -> Target Retrieved -> Target Top-1. Thick lines mark best and worst retrievers per corpus.", font=f_small, fill="#555555")

    complete = complete_rows(rows)
    by_corpus = {corpus: [r for r in complete if r.corpus == corpus] for corpus in CORPORA}
    bw = best_worst_map(rows)
    panel_w, panel_h = 600, 340
    gap_x, gap_y = 40, 80
    start_x, start_y = 90, 150

    for idx, corpus in enumerate(CORPORA):
        col, row_idx = idx % 3, idx // 3
        x0 = start_x + col * (panel_w + gap_x)
        y0 = start_y + row_idx * (panel_h + gap_y)
        draw.text((x0, y0 - 36), CORPUS_LABELS[corpus], font=f_axis, fill="#111111")

        chart_l, chart_t = x0 + 58, y0 + 22
        chart_w, chart_h = panel_w - 96, panel_h - 86
        for tick in range(0, 101, 25):
            y = chart_t + chart_h - (tick / 100) * chart_h
            draw.line((chart_l, y, chart_l + chart_w, y), fill="#E6E6E6", width=1)
            draw.text((x0 + 10, y - 8), str(tick), font=f_small, fill="#666666")
        stage_x = [chart_l, chart_l + chart_w / 2, chart_l + chart_w]
        for sx, stage in zip(stage_x, STAGE_LABELS):
            draw.line((sx, chart_t, sx, chart_t + chart_h), fill="#DDDDDD", width=1)
            draw_centered(draw, (sx, chart_t + chart_h + 32), stage.replace(" ", "\n"), f_small, "#333333")

        worst, best = bw.get(corpus, (None, None))
        for r in by_corpus[corpus]:
            vals = metric_values(r)
            points = [(stage_x[i], chart_t + chart_h - (vals[i] / 100) * chart_h) for i in range(3)]
            is_best = best and r.retriever == best.retriever
            is_worst = worst and r.retriever == worst.retriever
            width_line = 5 if (is_best or is_worst) else 2
            fill = PALETTE[r.retriever]
            if not (is_best or is_worst):
                fill = fill + "99" if len(fill) == 7 else fill
            draw.line(points, fill=fill[:7], width=width_line)
            for px, py in points:
                radius = 5 if (is_best or is_worst) else 3
                draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=PALETTE[r.retriever], outline="white")
        if best:
            draw.text((chart_l, y0 + panel_h - 18), f"Best: {best.label} ({best.target_top1_pct:.1f}% top-1)", font=f_small, fill="#111111")
        if worst:
            draw.text((chart_l + 270, y0 + panel_h - 18), f"Worst: {worst.label} ({worst.target_top1_pct:.1f}% top-1)", font=f_small, fill="#555555")

    legend_y = height - 118
    draw.text((80, legend_y - 30), "Retriever combinations", font=f_axis, fill="#111111")
    for i, (combo, label) in enumerate(COMBOS):
        x = 80 + (i % 4) * 440
        y = legend_y + (i // 4) * 36
        draw.line((x, y + 10, x + 42, y + 10), fill=PALETTE[combo], width=6)
        draw.text((x + 54, y), label, font=f_label, fill="#222222")
    save_with_pdf(img, out_path)


def draw_rate_sensitivity(rate_rows: List[RetrieverRow], metric: str, title: str, out_path: Path) -> None:
    width, height = 1760, 720
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    f_title, f_axis, f_label, f_small = font(32, True), font(18, True), font(16), font(13)
    draw.text((70, 34), title, font=f_title, fill="#111111")
    draw.text((70, 78), "Poisoning-rate variants keep the full query workload and subset the poisoned attack pool.", font=f_small, fill="#555555")

    panel_w, panel_h = 500, 470
    start_x, start_y = 80, 145
    gap = 55
    x_rate = {"025": 0, "050": 1, "100": 2}

    for idx, corpus in enumerate(RATE_CORPORA):
        x0 = start_x + idx * (panel_w + gap)
        y0 = start_y
        draw.text((x0, y0 - 34), CORPUS_LABELS[corpus], font=f_axis, fill="#111111")
        chart_l, chart_t = x0 + 62, y0 + 16
        chart_w, chart_h = panel_w - 98, panel_h - 88
        for tick in range(0, 101, 20):
            y = chart_t + chart_h - (tick / 100) * chart_h
            draw.line((chart_l, y, chart_l + chart_w, y), fill="#E7E7E7", width=1)
            draw.text((x0 + 18, y - 8), str(tick), font=f_small, fill="#666666")
        xs = [chart_l, chart_l + chart_w / 2, chart_l + chart_w]
        for rate, label in [("025", "25%"), ("050", "50%"), ("100", "100%")]:
            draw_centered(draw, (xs[x_rate[rate]], chart_t + chart_h + 28), label, f_label, "#333333")

        corpus_rows = [r for r in rate_rows if r.corpus == corpus and r.status == "complete"]
        retrievers = []
        for r in corpus_rows:
            if r.retriever not in retrievers:
                retrievers.append(r.retriever)
        for combo in retrievers:
            series = sorted([r for r in corpus_rows if r.retriever == combo], key=lambda r: RATES.index(r.rate))
            if len(series) < 2:
                continue
            points = []
            for r in series:
                value = getattr(r, metric)
                points.append((xs[x_rate[r.rate]], chart_t + chart_h - (value / 100) * chart_h))
            label = next((lbl for c, lbl in COMBOS if c == combo), combo)
            role = next((r.role for r in series if r.role), "")
            line_w = 6 if role in {"best", "worst"} else 3
            color = PALETTE.get(combo, "#555555")
            draw.line(points, fill=color, width=line_w)
            for px, py in points:
                draw.ellipse((px - 6, py - 6, px + 6, py + 6), fill=color, outline="white", width=2)
            lx, ly = points[-1]
            suffix = f" ({role})" if role in {"best", "worst"} else ""
            draw.text((lx + 10, ly - 10), f"{label}{suffix}", font=f_small, fill="#222222")

        draw.text((chart_l + chart_w / 2 - 50, y0 + panel_h - 10), "Poisoning rate", font=f_label, fill="#333333")
    draw.text((20, 325), "Percent", font=f_axis, fill="#333333")
    save_with_pdf(img, out_path)


def write_status(main_rows: List[RetrieverRow], rate_rows: List[RetrieverRow]) -> None:
    missing = [r for r in main_rows if r.status == "missing"]
    rate_missing = [r for r in rate_rows if r.status == "missing"]
    lines = [
        "# Retriever Stage Completion Status",
        "",
        f"Main 8-way missing rows: {len(missing)}",
    ]
    for row in missing:
        lines.append(f"- {CORPUS_LABELS[row.corpus]}: {row.label}")
    lines.extend(["", f"Rate-sweep selected-row missing rows: {len(rate_missing)}"])
    for row in rate_missing:
        lines.append(f"- {CORPUS_LABELS[row.corpus]} rate {row.rate}: {row.label} ({row.role or 'comparison'})")
    (OUT_DIR / "retriever_stage_completion_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    main_rows = collect_main_rows()
    bw_rows = best_worst_rows(main_rows)
    rate_rows = collect_rate_rows(main_rows)

    write_csv(OUT_DIR / "blackbox_retriever_all.csv", main_rows)
    write_csv(OUT_DIR / "blackbox_retriever_best_worst.csv", bw_rows)
    write_csv(OUT_DIR / "blackbox_rate_retriever_selected.csv", rate_rows)
    write_markdown_table(OUT_DIR / "blackbox_retriever_all.md", main_rows, "Blackbox Retriever Stage: All 8 Combinations")
    write_markdown_table(OUT_DIR / "blackbox_retriever_best_worst.md", bw_rows, "Blackbox Retriever Stage: Strongest and Weakest")
    write_markdown_table(OUT_DIR / "blackbox_rate_retriever_selected.md", rate_rows, "Blackbox Poisoning-Rate Retriever Stage")
    write_status(main_rows, rate_rows)

    draw_funnel(main_rows, PLOT_DIR / "blackbox_exposure_to_top1_funnel.png")
    draw_heatmap(main_rows, PLOT_DIR / "blackbox_retriever_risk_heatmap.png")
    draw_rate_sensitivity(
        rate_rows,
        "poison_exposure_pct",
        "Blackbox Poisoning-Rate Sensitivity: Exposure Rate",
        PLOT_DIR / "blackbox_poisoning_rate_exposure_curve.png",
    )
    draw_rate_sensitivity(
        rate_rows,
        "target_top1_pct",
        "Blackbox Poisoning-Rate Sensitivity: Target Top-1",
        PLOT_DIR / "blackbox_poisoning_rate_target_top1_curve.png",
    )

    missing = [r for r in main_rows if r.status == "missing"]
    rate_missing = [r for r in rate_rows if r.status == "missing"]
    print(f"Wrote tables to {OUT_DIR}")
    print(f"Wrote plots to {PLOT_DIR}")
    print(f"Main missing rows: {len(missing)}")
    for row in missing:
        print(f"  MISSING main: {row.corpus} {row.label}")
    print(f"Rate missing rows: {len(rate_missing)}")
    for row in rate_missing:
        print(f"  MISSING rate: {row.corpus} {row.rate} {row.label} ({row.role or 'comparison'})")


if __name__ == "__main__":
    main()
