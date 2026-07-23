"""Clean-vector SVG toolkit for Chapter 16 figures.

Authored as SVG (canonical) for precise typography, curved callouts, muted pastel fills, and
data-bound labels. Renders PNG through rsvg-convert at 2x, and emits a companion Excalidraw
scene so the source stays importable. The visual language: warm off-white ground, charcoal
outlines, two or three restrained accents, direct labels with thin curved arrows.
"""

from __future__ import annotations

import subprocess
from html import escape
from pathlib import Path

CHAPTER_DIR = Path(__file__).resolve().parents[1]
FIGURE_DIR = CHAPTER_DIR / "assets" / "figures"

# --- restrained palette (muted, inspired by the references, not copied) -------------------
PAPER = "#FAF7F1"       # warm off-white ground
INK = "#2B2A28"         # charcoal outline and text
MUTED = "#7C776D"       # secondary text
FAINT = "#B7B2A7"       # hairlines, ticks
BLUE = "#3C6E8F"        # observed input / evidence / annotation
BLUE_FILL = "#CFE0EA"
SAGE = "#5E8A6E"        # approved decision / learned outcome
SAGE_FILL = "#D6E7DA"
OCHRE = "#BE8A3A"       # rule / calculation / human gate
OCHRE_FILL = "#F1E4C6"
BRICK = "#B0563F"       # blocked / missing / warning
BRICK_FILL = "#ECD6CE"
GRAY = "#8C8A84"        # protected / secondary
GRAY_FILL = "#E7E3DB"
CARD = "#FFFFFF"
CONTAINER = "#F1ECE1"
FONT = "Helvetica, Arial, sans-serif"


class Figure:
    """Accumulates a canonical editable SVG scene."""

    def __init__(self, width: int = 1600, height: int = 1000, scale: int = 2):
        self.width = width
        self.height = height
        self.scale = scale
        self.svg: list[str] = []
        self.elements: list[dict] = []
        self._n = 0

    # --- excalidraw mirror helpers ------------------------------------------
    def _ex(self, kind: str, x: float, y: float, w: float, h: float, **kw) -> None:
        self._n += 1
        base = {
            "id": f"e{self._n:03d}", "type": kind, "x": x, "y": y, "width": w, "height": h,
            "angle": 0, "strokeColor": kw.get("stroke", INK),
            "backgroundColor": kw.get("fill", "transparent"), "fillStyle": "solid",
            "strokeWidth": 1.5, "strokeStyle": "solid", "roughness": 0, "opacity": 100,
            "groupIds": [], "frameId": None, "roundness": {"type": 3} if kw.get("round") else None,
            "seed": 16000 + self._n, "version": 1, "versionNonce": 26000 + self._n,
            "isDeleted": False, "boundElements": [], "updated": 1784580000000,
            "link": None, "locked": False,
        }
        if kind == "text":
            base.update({"text": kw.get("text", ""), "fontSize": kw.get("size", 20),
                         "fontFamily": 2, "textAlign": "left", "verticalAlign": "top",
                         "containerId": None, "originalText": kw.get("text", ""),
                         "autoResize": False, "lineHeight": 1.2})
        self.elements.append(base)

    # --- primitives ----------------------------------------------------------
    def rrect(self, x, y, w, h, fill=CARD, stroke=INK, r=14, sw=2.0, shadow=False, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        if shadow:
            self.svg.append(
                f'<rect x="{x+3}" y="{y+5}" width="{w}" height="{h}" rx="{r}" '
                f'fill="#000000" opacity="0.06"/>'
            )
        self.svg.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}"{d}/>'
        )
        self._ex("rectangle", x, y, w, h, fill=fill, stroke=stroke, round=bool(r))

    def circle(self, cx, cy, rad, fill=CARD, stroke=INK, sw=2.0):
        self.svg.append(
            f'<circle cx="{cx}" cy="{cy}" r="{rad}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{sw}"/>'
        )
        self._ex("ellipse", cx - rad, cy - rad, 2 * rad, 2 * rad, fill=fill, stroke=stroke)

    def diamond(self, cx, cy, rad, fill=OCHRE_FILL, stroke=OCHRE, sw=2.2):
        pts = f"{cx},{cy-rad} {cx+rad},{cy} {cx},{cy+rad} {cx-rad},{cy}"
        self.svg.append(
            f'<polygon points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" '
            f'stroke-linejoin="round"/>'
        )
        self._ex("diamond", cx - rad, cy - rad, 2 * rad, 2 * rad, fill=fill, stroke=stroke)

    def text(self, x, y, s, size=22, weight=400, color=INK, anchor="start", spacing=0.0,
             italic=False, family=FONT):
        anchor_map = {"start": "start", "middle": "middle", "end": "end"}
        ls = f' letter-spacing="{spacing}"' if spacing else ""
        st = ' font-style="italic"' if italic else ""
        # support simple two-line via \n
        if "\n" in s:
            lines = s.split("\n")
            lh = size * 1.24
            for i, line in enumerate(lines):
                self.svg.append(
                    f'<text x="{x}" y="{y + i*lh}" fill="{color}" font-family="{family}" '
                    f'font-size="{size}" font-weight="{weight}"{ls}{st} '
                    f'text-anchor="{anchor_map[anchor]}">{escape(line)}</text>'
                )
        else:
            self.svg.append(
                f'<text x="{x}" y="{y}" fill="{color}" font-family="{family}" font-size="{size}" '
                f'font-weight="{weight}"{ls}{st} text-anchor="{anchor_map[anchor]}">{escape(s)}</text>'
            )
        self._ex("text", x, y - size, max(len(s) * size * 0.55, 40), size * 1.3, text=s, size=size)

    def label_tile(self, cx, cy, w, h, lines, fill=CARD, stroke=INK, size=22, weight=700,
                   r=14, shadow=False, sub=""):
        self.rrect(cx - w / 2, cy - h / 2, w, h, fill=fill, stroke=stroke, r=r, shadow=shadow)
        parts = lines.split("\n")
        lh = size * 1.2
        y0 = cy - (len(parts) - 1) * lh / 2 + size * 0.34
        for i, part in enumerate(parts):
            self.text(cx, y0 + i * lh, part, size=size, weight=weight, anchor="middle")
        if sub:
            self.text(cx, cy + h / 2 + 26, sub, size=17, weight=400, color=MUTED, anchor="middle")

    def line(self, x1, y1, x2, y2, stroke=INK, sw=2.0, dash=None, cap="round"):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.svg.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" '
            f'stroke-width="{sw}"{d} stroke-linecap="{cap}"/>'
        )

    @staticmethod
    def _marker_for(stroke: str) -> str:
        return {
            BLUE: "arrow_blue",
            SAGE: "arrow_sage",
            OCHRE: "arrow_ochre",
            BRICK: "arrow_brick",
            GRAY: "arrow_gray",
            MUTED: "arrow_muted",
        }.get(stroke, "arrow")

    def arrow(self, x1, y1, x2, y2, stroke=INK, sw=2.0, marker=None):
        marker = marker or self._marker_for(stroke)
        self.svg.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" '
            f'stroke-width="{sw}" marker-end="url(#{marker})" stroke-linecap="round"/>'
        )

    def curve(self, x1, y1, x2, y2, bend=0.35, stroke=BLUE, sw=1.8, marker=None):
        """A smooth cubic callout from (x1,y1) to (x2,y2), bowed by `bend`."""
        marker = marker or self._marker_for(stroke)
        dx, dy = x2 - x1, y2 - y1
        # perpendicular offset for the bow
        nx, ny = -dy, dx
        norm = (nx ** 2 + ny ** 2) ** 0.5 or 1
        ox, oy = nx / norm * bend * ((dx ** 2 + dy ** 2) ** 0.5), ny / norm * bend * ((dx ** 2 + dy ** 2) ** 0.5)
        c1 = (x1 + dx * 0.25 + ox * 0.6, y1 + dy * 0.25 + oy * 0.6)
        c2 = (x1 + dx * 0.75 + ox * 0.6, y1 + dy * 0.75 + oy * 0.6)
        self.svg.append(
            f'<path d="M {x1} {y1} C {c1[0]:.1f} {c1[1]:.1f}, {c2[0]:.1f} {c2[1]:.1f}, '
            f'{x2} {y2}" fill="none" stroke="{stroke}" stroke-width="{sw}" '
            f'marker-end="url(#{marker})"/>'
        )

    def bracket(self, x1, y, x2, depth=14, stroke=MUTED, sw=2.0, label="", up=True):
        d = -depth if up else depth
        self.svg.append(
            f'<path d="M {x1} {y} L {x1} {y+d} L {x2} {y+d} L {x2} {y}" fill="none" '
            f'stroke="{stroke}" stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round"/>'
        )
        if label:
            self.text((x1 + x2) / 2, y + d + (-8 if up else 24), label, size=18, weight=700,
                      color=stroke, anchor="middle")

    def section(self, x, y, kicker, sub="", color=INK):
        self.text(x, y, kicker.upper(), size=19, weight=800, color=color, spacing=1.6)
        if sub:
            self.text(x, y + 26, sub, size=18, weight=400, color=MUTED, italic=True)

    # --- output --------------------------------------------------------------
    def save(self, name: str) -> None:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        defs = (
            '<defs>'
            '<marker id="arrow" markerWidth="10" markerHeight="10" refX="7.5" refY="4" '
            f'orient="auto"><path d="M0,0 L9,4 L0,8 z" fill="{INK}"/></marker>'
            '<marker id="arrow_blue" markerWidth="10" markerHeight="10" refX="7.5" refY="4" '
            f'orient="auto"><path d="M0,0 L9,4 L0,8 z" fill="{BLUE}"/></marker>'
            '<marker id="arrow_sage" markerWidth="10" markerHeight="10" refX="7.5" refY="4" '
            f'orient="auto"><path d="M0,0 L9,4 L0,8 z" fill="{SAGE}"/></marker>'
            '<marker id="arrow_ochre" markerWidth="10" markerHeight="10" refX="7.5" refY="4" '
            f'orient="auto"><path d="M0,0 L9,4 L0,8 z" fill="{OCHRE}"/></marker>'
            '<marker id="arrow_brick" markerWidth="10" markerHeight="10" refX="7.5" refY="4" '
            f'orient="auto"><path d="M0,0 L9,4 L0,8 z" fill="{BRICK}"/></marker>'
            '<marker id="arrow_gray" markerWidth="10" markerHeight="10" refX="7.5" refY="4" '
            f'orient="auto"><path d="M0,0 L9,4 L0,8 z" fill="{GRAY}"/></marker>'
            '<marker id="arrow_muted" markerWidth="10" markerHeight="10" refX="7.5" refY="4" '
            f'orient="auto"><path d="M0,0 L9,4 L0,8 z" fill="{MUTED}"/></marker>'
            '</defs>'
        )
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" '
            f'viewBox="0 0 {self.width} {self.height}">\n'
            f'<rect width="100%" height="100%" fill="{PAPER}"/>\n{defs}\n'
            + "\n".join(self.svg) + "\n</svg>\n"
        )
        (FIGURE_DIR / f"{name}.svg").write_text(svg)
        subprocess.run([
            "rsvg-convert", "-w", str(self.width * self.scale), "-h", str(self.height * self.scale),
            str(FIGURE_DIR / f"{name}.svg"), "-o", str(FIGURE_DIR / f"{name}.png"),
        ], check=True)
        print(f"built {name}.svg / .png")
