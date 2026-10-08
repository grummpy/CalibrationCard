"""Small dependency-free SVG charts for the report card."""

from __future__ import annotations

from html import escape

INK = "#1f2a44"
MUTED = "#7a8094"
GRID = "#e6e0d2"
GREEN = "#2e7d5b"
RED = "#d64541"
GOLD = "#e0a526"
BLUE = "#3b6ea8"


def _frame(w, h, pad, title, xlabel, ylabel):
    parts = [f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{escape(title)}" '
             f'font-family="system-ui, -apple-system, Segoe UI, sans-serif">']
    parts.append(f'<text x="{pad}" y="16" font-size="13" font-weight="600" fill="{INK}">{escape(title)}</text>')
    parts.append(f'<text x="{w / 2}" y="{h - 4}" font-size="11" fill="{MUTED}" text-anchor="middle">{escape(xlabel)}</text>')
    parts.append(f'<text transform="translate(12,{h / 2}) rotate(-90)" font-size="11" fill="{MUTED}" '
                 f'text-anchor="middle">{escape(ylabel)}</text>')
    return parts


def reliability(bins: list[dict], *, title="Reliability diagram", overlays: list[tuple[str, list[dict], str]] | None = None,
                width=420, height=360, floor: float = 0.0) -> str:
    pad_l, pad_r, pad_t, pad_b = 46, 14, 28, 40
    pw, ph = width - pad_l - pad_r, height - pad_t - pad_b
    lo = max(0.0, min(floor, 0.9))

    def x(v):
        return pad_l + (v - lo) / (1 - lo) * pw

    def y(v):
        return pad_t + (1 - v) * ph

    parts = _frame(width, height, pad_l, title, "Confidence (what the model says)", "Accuracy (how often it is right)")
    for t in range(0, 11, 2):
        v = t / 10
        parts.append(f'<line x1="{pad_l}" x2="{pad_l + pw}" y1="{y(v)}" y2="{y(v)}" stroke="{GRID}"/>')
        parts.append(f'<text x="{pad_l - 6}" y="{y(v) + 4}" font-size="10" fill="{MUTED}" text-anchor="end">{v:.1f}</text>')
    for k in range(6):
        v = lo + (1 - lo) * k / 5
        parts.append(f'<text x="{x(v)}" y="{pad_t + ph + 14}" font-size="10" fill="{MUTED}" text-anchor="middle">{v:.2f}</text>')
    total = sum(b["count"] for b in bins) or 1
    for b in bins:
        if not b["count"]:
            continue
        x0, x1 = x(max(b["low"], lo)), x(b["high"])
        acc, conf = b["accuracy"], b["confidence"]
        w = max(1.0, x1 - x0 - 1.5)
        alpha = 0.35 + 0.65 * min(1.0, b["count"] / total * len(bins))
        parts.append(f'<rect x="{x0 + 0.75:.2f}" y="{y(acc):.2f}" width="{w:.2f}" height="{(y(0) - y(acc)):.2f}" '
                     f'fill="{BLUE}" fill-opacity="{alpha:.2f}"><title>{b["count"]} rows · says {conf:.1%} · right {acc:.1%}'
                     f'</title></rect>')
        top, bottom = (y(conf), y(acc)) if conf > acc else (y(acc), y(conf))
        if abs(conf - acc) > 0.002:
            parts.append(f'<rect x="{x0 + 0.75:.2f}" y="{top:.2f}" width="{w:.2f}" height="{bottom - top:.2f}" '
                         f'fill="{RED}" fill-opacity="0.28" stroke="{RED}" stroke-opacity="0.6" stroke-width="0.8"/>')
    parts.append(f'<line x1="{x(lo)}" y1="{y(lo)}" x2="{x(1)}" y2="{y(1)}" stroke="{INK}" stroke-dasharray="5 4" '
                 'stroke-width="1.2"/>')
    legend_y = pad_t + 8
    for i, (label, obins, color) in enumerate(overlays or []):
        pts = [(x(b["confidence"]), y(b["accuracy"])) for b in obins if b["count"] and b["confidence"] is not None
               and b["confidence"] >= lo]
        if len(pts) > 1:
            path = " ".join(f"{px:.1f},{py:.1f}" for px, py in pts)
            parts.append(f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="2"/>')
            parts.extend(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="2.6" fill="{color}"/>' for px, py in pts)
        parts.append(f'<rect x="{pad_l + 10}" y="{legend_y + i * 15}" width="10" height="3" fill="{color}"/>'
                     f'<text x="{pad_l + 25}" y="{legend_y + 4 + i * 15}" font-size="10" fill="{INK}">{escape(label)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def confidence_histogram(hist: list[dict], *, width=420, height=170, title="How confident is it?") -> str:
    pad_l, pad_r, pad_t, pad_b = 46, 14, 28, 34
    pw, ph = width - pad_l - pad_r, height - pad_t - pad_b
    top = max((h["count"] for h in hist), default=1) or 1
    parts = _frame(width, height, pad_l, title, "Confidence of the predicted class", "Rows")
    bw = pw / len(hist)
    for i, h in enumerate(hist):
        bh = h["count"] / top * ph
        parts.append(f'<rect x="{pad_l + i * bw + 1:.2f}" y="{pad_t + ph - bh:.2f}" width="{bw - 2:.2f}" height="{bh:.2f}" '
                     f'fill="{GOLD}"><title>{h["low"]:.2f}-{h["high"]:.2f}: {h["count"]}</title></rect>')
    for k in range(6):
        v = k / 5
        parts.append(f'<text x="{pad_l + v * pw}" y="{pad_t + ph + 13}" font-size="10" fill="{MUTED}" '
                     f'text-anchor="middle">{v:.1f}</text>')
    parts.append(f'<text x="{pad_l - 6}" y="{pad_t + 8}" font-size="10" fill="{MUTED}" text-anchor="end">{top}</text>')
    parts.append("</svg>")
    return "".join(parts)


def class_direction(row: dict) -> str:
    """'on target', 'over', 'under' or 'mixed' (errors in both directions that cancel out on average)."""
    if row["ece"] < 0.02:
        return "on target"
    if abs(row["signed_gap"]) < 0.25 * row["ece"]:
        return "mixed"
    return "over" if row["signed_gap"] > 0 else "under"


def class_color(row: dict) -> str:
    return {"on target": GREEN, "over": RED, "under": BLUE, "mixed": GOLD}[class_direction(row)]


def class_bars(classwise: list[dict], *, width=420, height=None, title="Per-class calibration error") -> str:
    rows = classwise[:20]
    height = height or 40 + 18 * len(rows)
    pad_l, pad_r, pad_t = 110, 50, 26
    pw = width - pad_l - pad_r
    top = max([r["ece"] for r in rows] + [0.05])
    parts = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
             f'font-family="system-ui, -apple-system, Segoe UI, sans-serif">',
             f'<text x="0" y="16" font-size="13" font-weight="600" fill="{INK}">{escape(title)}</text>']
    for i, r in enumerate(rows):
        yy = pad_t + i * 18
        w = r["ece"] / top * pw
        color = class_color(r)
        label = r["class"] if len(r["class"]) <= 16 else r["class"][:15] + "…"
        parts.append(f'<text x="{pad_l - 6}" y="{yy + 11}" font-size="11" fill="{INK}" text-anchor="end">{escape(label)}</text>')
        parts.append(f'<rect x="{pad_l}" y="{yy + 2}" width="{max(w, 1):.2f}" height="12" rx="2" fill="{color}"/>')
        parts.append(f'<text x="{pad_l + max(w, 1) + 4:.2f}" y="{yy + 12}" font-size="10" fill="{MUTED}">{r["ece"]:.1%}</text>')
    parts.append(f'<text x="{pad_l}" y="{height - 4}" font-size="10" fill="{MUTED}">red = over-predicted, blue = '
                 'under-predicted, gold = mixed, green = on target</text>')
    parts.append("</svg>")
    return "".join(parts)
