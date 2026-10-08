#!/usr/bin/env python3
"""Render the app icon (SVG, PNG, ICO, ICNS) and copy it into the web UI.

    python scripts/render_assets.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
WEB = ROOT / "calibrationcard" / "web"
STEPS = [(250, 760, 340, 700), (340, 700, 430, 610), (430, 610, 520, 530), (520, 530, 610, 470), (610, 470, 700, 380)]

SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1024 1024">
  <rect width="1024" height="1024" rx="224" fill="#1f2a44"/>
  <rect x="170" y="150" width="684" height="724" rx="56" fill="#fbf6ea"/>
  <line x1="250" y1="790" x2="774" y2="266" stroke="#7a8094" stroke-width="14" stroke-dasharray="34 26" stroke-linecap="round"/>
  <g fill="#3b6ea8">
    <rect x="250" y="700" width="78" height="90" rx="10"/>
    <rect x="340" y="610" width="78" height="180" rx="10"/>
    <rect x="430" y="530" width="78" height="260" rx="10"/>
    <rect x="520" y="470" width="78" height="320" rx="10"/>
    <rect x="610" y="380" width="78" height="410" rx="10"/>
  </g>
  <circle cx="700" cy="300" r="128" fill="#fbf6ea" stroke="#d64541" stroke-width="30"/>
  <text x="700" y="352" font-family="Georgia, 'Times New Roman', serif" font-size="160" font-weight="700"
        fill="#d64541" text-anchor="middle">A</text>
</svg>
"""


def render(size: int = 1024) -> Image.Image:
    scale = 4
    s = size * scale
    k = s / 1024
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, s - 1, s - 1], radius=int(224 * k), fill=(31, 42, 68, 255))
    d.rounded_rectangle([170 * k, 150 * k, 854 * k, 874 * k], radius=int(56 * k), fill=(251, 246, 234, 255))
    # dashed diagonal (perfect calibration)
    x0, y0, x1, y1 = 250, 790, 774, 266
    length = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
    pos = 0.0
    while pos < length:
        a = pos / length
        b = min(length, pos + 34) / length
        d.line([((x0 + (x1 - x0) * a) * k, (y0 + (y1 - y0) * a) * k), ((x0 + (x1 - x0) * b) * k, (y0 + (y1 - y0) * b) * k)],
               fill=(122, 128, 148, 255), width=int(14 * k))
        pos += 60
    for x, top in ((250, 700), (340, 610), (430, 530), (520, 470), (610, 380)):
        d.rounded_rectangle([x * k, top * k, (x + 78) * k, 790 * k], radius=int(10 * k), fill=(59, 110, 168, 255))
    r = 128 * k
    cx, cy = 700 * k, 300 * k
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(251, 246, 234, 255), outline=(214, 69, 65, 255), width=int(30 * k))
    font = None
    for name in ("DejaVuSerif-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf", "Georgia Bold.ttf"):
        try:
            font = ImageFont.truetype(name, int(170 * k))
            break
        except OSError:
            continue
    font = font or ImageFont.load_default()
    d.text((cx, cy + 6 * k), "A", font=font, fill=(214, 69, 65, 255), anchor="mm")
    return img.resize((size, size), Image.LANCZOS)


def main() -> int:
    ASSETS.mkdir(exist_ok=True)
    (ASSETS / "icon.svg").write_text(SVG, encoding="utf-8")
    big = render(1024)
    big.save(ASSETS / "icon-1024.png")
    small = big.resize((512, 512), Image.LANCZOS)
    small.save(ASSETS / "icon-512.png")
    small.save(ASSETS / "icon.png")
    big.save(ASSETS / "icon.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    big.save(ASSETS / "icon.icns")
    shutil.copy(ASSETS / "icon.svg", WEB / "icon.svg")
    big.resize((256, 256), Image.LANCZOS).save(WEB / "icon.png")
    big.save(WEB / "favicon.ico", sizes=[(16, 16), (32, 32), (48, 48), (64, 64)])
    print("Icons written to assets/ and calibrationcard/web/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
