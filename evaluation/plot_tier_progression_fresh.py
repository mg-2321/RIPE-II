#!/usr/bin/env python3
"""
Render a fresh poisoning-rate progression figure for the generation stage.

The plot auto-discovers completed rate-sweep live_judge JSONL files and draws
one grouped bar per corpus using judge-based ASR overall. To keep the
comparison clean, it uses the common completed model shared across corpora
(`llama-3.1-8b`) rather than pooling mismatched generators together.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT_PATH = "/usr/share/fonts/dejavu/DejaVuSans.ttf"
FONT_BOLD_PATH = "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"
RATE_DIR = Path("/gscratch/uwb/gayat23/GuardRAG/results/live_judge/rate_sweep_blackbox")
COMMON_MODEL = "llama-3.1-8b"

RATE_RE = re.compile(r"^(?P<corpus>.+)_rate(?P<rate>\d{3})_.+\.jsonl$")

CORPUS_LABELS = {
    "fiqa": "FIQA (Financial)",
    "hotpotqa": "HotpotQA (Multi-hop)",
    "msmarco": "MSMARCO (Web Search)",
    "nfcorpus": "NFCorpus (Biomedical)",
    "nq": "NQ (General QA)",
    "scifact": "SciFact (Scientific)",
}

CORPUS_ORDER = [
    "nfcorpus",
    "scifact",
    "nq",
    "hotpotqa",
    "fiqa",
    "msmarco",
]

COLORS = {
    "fiqa": "#4CAF50",
    "hotpotqa": "#F01F1F",
    "msmarco": "#8C564B",
    "nfcorpus": "#9467BD",
    "nq": "#FF7A00",
    "scifact": "#3F7FBF",
}

MARKERS = {
    "fiqa": "triangle",
    "hotpotqa": "circle",
    "msmarco": "diamond",
    "nfcorpus": "circle",
    "nq": "diamond",
    "scifact": "square",
}

C = {
    "bg": "#FFFFFF",
    "text": "#111111",
    "muted": "#666666",
    "axis": "#2D2D2D",
    "grid": "#D6D6D6",
    "sparse_fill": "#A8E6A0",
    "sparse_border": "#5E6E5A",
    "moderate_text": "#FF8A00",
    "callout_fill": "#FFFDF8",
    "callout_border": "#B22222",
    "callout_text": "#8B0000",
    "cat_fill": "#F6A0A0",
    "legend_fill": "#F7F7F7",
    "legend_border": "#B8B8B8",
    "legend_shadow": "#A0A0A0",
    "white": "#FFFFFF",
}


def fnt(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD_PATH if bold else FONT_PATH, size=size)


def center_text(draw: ImageDraw.ImageDraw, cx: float, cy: float, text: str, font, fill: str) -> None:
    lines = text.split("\n")
    line_gap = 4
    heights = []
    widths = []
    for line in lines:
        bb = draw.textbbox((0, 0), line, font=font)
        widths.append(bb[2] - bb[0])
        heights.append(bb[3] - bb[1])
    total_h = sum(heights) + line_gap * (len(lines) - 1)
    y = cy - total_h / 2
    for line, w, h in zip(lines, widths, heights):
        draw.text((cx - w / 2, y), line, font=font, fill=fill)
        y += h + line_gap


def text_box(
    draw: ImageDraw.ImageDraw,
    cx: float,
    cy: float,
    text: str,
    font,
    fill: str,
    bg: str,
    border: str,
    pad_x: int = 12,
    pad_y: int = 10,
    border_w: int = 2,
) -> tuple[float, float, float, float]:
    lines = text.split("\n")
    widths = []
    heights = []
    for line in lines:
        bb = draw.textbbox((0, 0), line, font=font)
        widths.append(bb[2] - bb[0])
        heights.append(bb[3] - bb[1])
    line_gap = 4
    box_w = max(widths) + pad_x * 2
    box_h = sum(heights) + line_gap * (len(lines) - 1) + pad_y * 2
    x0 = cx - box_w / 2
    y0 = cy - box_h / 2
    x1 = cx + box_w / 2
    y1 = cy + box_h / 2
    draw.rounded_rectangle((x0, y0, x1, y1), radius=12, fill=bg, outline=border, width=border_w)
    center_text(draw, cx, cy, text, font, fill)
    return (x0, y0, x1, y1)


def arrow(draw: ImageDraw.ImageDraw, x0: float, y0: float, x1: float, y1: float, fill: str, width: int = 4) -> None:
    draw.line((x0, y0, x1, y1), fill=fill, width=width)
    dx = x1 - x0
    dy = y1 - y0
    length = max((dx * dx + dy * dy) ** 0.5, 1.0)
    ux = dx / length
    uy = dy / length
    px = -uy
    py = ux
    head = 14
    half = 7
    draw.polygon(
        [
            (x1, y1),
            (x1 - ux * head + px * half, y1 - uy * head + py * half),
            (x1 - ux * head - px * half, y1 - uy * head - py * half),
        ],
        fill=fill,
    )


def draw_marker(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    style: str,
    color: str,
    size: int = 18,
) -> None:
    outer = size
    inner = size - 4

    if style == "circle":
        draw.ellipse((x - outer, y - outer, x + outer, y + outer), fill=C["white"])
        draw.ellipse((x - inner, y - inner, x + inner, y + inner), fill=color)
        return

    if style == "square":
        draw.rectangle((x - outer, y - outer, x + outer, y + outer), fill=C["white"])
        draw.rectangle((x - inner, y - inner, x + inner, y + inner), fill=color)
        return

    if style == "triangle":
        outer_pts = [(x, y - outer), (x - outer, y + outer), (x + outer, y + outer)]
        inner_pts = [(x, y - inner), (x - inner, y + inner), (x + inner, y + inner)]
        draw.polygon(outer_pts, fill=C["white"])
        draw.polygon(inner_pts, fill=color)
        return

    if style == "diamond":
        outer_pts = [(x, y - outer), (x - outer, y), (x, y + outer), (x + outer, y)]
        inner_pts = [(x, y - inner), (x - inner, y), (x, y + inner), (x + inner, y)]
        draw.polygon(outer_pts, fill=C["white"])
        draw.polygon(inner_pts, fill=color)
        return


def load_rate_series() -> dict[str, dict]:
    latest_paths: dict[tuple[str, int], Path] = {}

    for path in sorted(RATE_DIR.glob("*.jsonl")):
        match = RATE_RE.match(path.name)
        if not match:
            continue
        if f"_{COMMON_MODEL}_" not in path.name:
            continue
        corpus = match.group("corpus")
        rate = int(match.group("rate"))
        latest_paths[(corpus, rate)] = path

    raw: dict[str, dict[int, dict]] = defaultdict(dict)
    for (corpus, rate), path in sorted(latest_paths.items()):
        total = 0
        exposed = 0
        judge_yes = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                total += 1
                row = json.loads(line)
                if row.get("any_poison_retrieved"):
                    exposed += 1
                if row.get("judge_verdict") is True:
                    judge_yes += 1
        exposure_rate = round((exposed / total) * 100, 1) if total else 0.0
        asr_rate = round((judge_yes / total) * 100, 1) if total else 0.0
        raw[corpus][rate] = {
            "rate": rate,
            "rows": total,
            "exposed": exposed,
            "judge_yes": judge_yes,
            "exposure_rate": exposure_rate,
            "asr_rate": asr_rate,
            "file": path.name,
        }

    cleaned: dict[str, dict] = {}
    for corpus in CORPUS_ORDER:
        entries = raw.get(corpus)
        if not entries:
            continue
        ordered_rates = sorted(entries)
        # Only plot corpora with a full current progression.
        if len(ordered_rates) < 3:
            continue
        cleaned[corpus] = {
            "label": CORPUS_LABELS.get(corpus, corpus),
            "color": COLORS.get(corpus, "#4C78A8"),
            "marker": MARKERS.get(corpus, "circle"),
            "rates": [entries[r] for r in ordered_rates],
        }
    return cleaned


def save_data_json(data: dict, path: Path) -> None:
    payload = {
        "source_dir": str(RATE_DIR),
        "metric": "judge_asr_overall",
        "model": COMMON_MODEL,
        "note": (
            "Fresh completed poisoning-rate sweeps only, using the common "
            f"generation model {COMMON_MODEL}. Corpora without all three "
            "completed 25/50/100 rate points are omitted."
        ),
        "corpora": data,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def render_plot(data: dict, out_path: Path) -> None:
    width, height = 1800, 1080
    img = Image.new("RGB", (width, height), C["bg"])
    draw = ImageDraw.Draw(img)

    left = 210
    right = width - 90
    top = 220
    bottom = 840
    plot_w = right - left
    plot_h = bottom - top
    ymax = 110.0

    all_rates = sorted({entry["rate"] for corpus in data.values() for entry in corpus["rates"]})
    if not all_rates:
        raise ValueError("No completed rate-sweep data found to plot.")

    def x_of(rate: int) -> float:
        if len(all_rates) == 1:
            return left + plot_w / 2
        idx = all_rates.index(rate)
        return left + (plot_w * (idx + 0.5) / len(all_rates))

    def y_of(value: float) -> float:
        return bottom - (value / ymax) * plot_h

    center_text(
        draw,
        width / 2,
        40,
        "Attack Success vs Poisoning Tier",
        fnt(38, True),
        C["text"],
    )

    for tick in range(0, 101, 20):
        y = y_of(tick)
        draw.line((left, y, right, y), fill=C["grid"], width=2)
        bb = draw.textbbox((0, 0), str(tick), font=fnt(20))
        draw.text((left - 18 - (bb[2] - bb[0]), y - (bb[3] - bb[1]) / 2), str(tick), font=fnt(20), fill=C["axis"])

    draw.line((left, top, left, bottom), fill=C["axis"], width=3)
    draw.line((left, bottom, right, bottom), fill=C["axis"], width=3)

    y_label = "Judge ASR Overall (%)"
    y_bb = draw.textbbox((0, 0), y_label, font=fnt(30, True))
    y_img = Image.new("RGBA", (y_bb[2] - y_bb[0] + 18, y_bb[3] - y_bb[1] + 18), (255, 255, 255, 0))
    y_draw = ImageDraw.Draw(y_img)
    y_draw.text((9, 9), y_label, font=fnt(30, True), fill=C["text"])
    y_img = y_img.rotate(90, expand=True)
    img.paste(y_img, (58, top + plot_h // 2 - y_img.height // 2), y_img)

    center_text(draw, (left + right) / 2, bottom + 110, "Poisoning Tier", fnt(34, True), C["text"])

    x_labels = {
        25: "Realistic\n(25%)",
        50: "Hard\n(50%)",
        100: "Stress\n(100%)",
    }
    for rate in all_rates:
        x = x_of(rate)
        draw.line((x, bottom, x, bottom + 10), fill=C["axis"], width=2)
        center_text(draw, x, bottom + 42, x_labels.get(rate, f"{rate}%"), fnt(24, True), C["text"])

    # Compact legend above the plot so it never overlaps the data.
    legend_entries = list(data.values())
    cols = 2
    col_w = 360
    row_h = 44
    rows = (len(legend_entries) + cols - 1) // cols
    legend_w = cols * col_w
    legend_h = rows * row_h + 24
    legend_x0 = int((width - legend_w) / 2)
    legend_y0 = 92
    legend_x1 = legend_x0 + legend_w
    legend_y1 = legend_y0 + legend_h
    draw.rounded_rectangle(
        (legend_x0, legend_y0, legend_x1, legend_y1),
        radius=10,
        fill=C["legend_fill"],
        outline=C["legend_border"],
        width=2,
    )

    for idx, payload in enumerate(legend_entries):
        color = payload["color"]
        row = idx // cols
        col = idx % cols
        x_base = legend_x0 + 26 + col * col_w
        y = legend_y0 + 22 + row * row_h
        draw.rectangle((x_base, y - 10, x_base + 34, y + 10), fill=color, outline=C["axis"], width=2)
        draw.text((x_base + 50, y - 16), payload["label"], font=fnt(18), fill=C["text"])

    # Vertical guide lines at each tier point
    for rate in all_rates:
        x = x_of(rate)
        draw.line((x, top, x, bottom), fill=C["grid"], width=2)

    # Grouped bar chart: poisoning tiers are categorical, not temporal.
    corpus_items = list(data.items())
    n_corpora = len(corpus_items)
    if n_corpora == 0:
        raise ValueError("No corpora available to plot.")

    if len(all_rates) > 1:
        min_group_gap = min(x_of(all_rates[i + 1]) - x_of(all_rates[i]) for i in range(len(all_rates) - 1))
    else:
        min_group_gap = plot_w

    group_width = min(360, max(240, int(min_group_gap * 0.52)))
    gap = 16
    bar_width = int((group_width - gap * (n_corpora - 1)) / n_corpora)
    if bar_width < 34:
        gap = 10
        bar_width = int((group_width - gap * (n_corpora - 1)) / n_corpora)

    for rate in all_rates:
        center_x = x_of(rate)
        start_x = center_x - (n_corpora * bar_width + (n_corpora - 1) * gap) / 2
        for idx, (_corpus, payload) in enumerate(corpus_items):
            entry = next((item for item in payload["rates"] if item["rate"] == rate), None)
            if entry is None:
                continue
            x0 = int(start_x + idx * (bar_width + gap))
            x1 = int(x0 + bar_width)
            y0 = int(y_of(entry["asr_rate"]))
            y1 = bottom
            draw.rectangle((x0, y0, x1, y1), fill=payload["color"], outline=C["axis"], width=2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    img.save(out_path.with_suffix(".pdf"), "PDF", resolution=200.0)


def main() -> None:
    data = load_rate_series()
    root = Path(__file__).resolve().parent.parent
    json_name = "plot2_tier_progression_fresh_data.json"
    png_name = "plot2_tier_progression_fresh.png"

    save_data_json(data, root / json_name)
    save_data_json(data, Path(__file__).resolve().parent / "plots" / json_name)
    render_plot(data, root / png_name)
    render_plot(data, Path(__file__).resolve().parent / "plots" / png_name)
    print(f"Saved: {root / png_name}")
    print(f"Saved: {root / png_name.replace('.png', '.pdf')}")


if __name__ == "__main__":
    main()
