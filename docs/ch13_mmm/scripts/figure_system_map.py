"""Figure 13.1: the unified measurement system map.

This is a conceptual diagram. It reads no generated data and does not fit any
model, so it can be regenerated on its own in a fraction of a second through
`build_figures.py system-map`.
"""

from __future__ import annotations

from pathlib import Path

C_TITLE = "#111827"
C_TEXT = "#1F2937"
C_SUB = "#6B7280"
C_HEAD = "#9CA3AF"
C_SPINE = "#374151"

# Channel dot colors match CHANNEL_PALETTE in run_analysis.py.
CH_COLOR = {
    "field": "#2F6B9A",
    "email": "#2A9D8F",
    "digital": "#7A68A6",
    "paid media": "#C77D2B",
}


def _tint(hex_color: str, amount: float) -> tuple[float, float, float]:
    """Blend a hex color toward white by `amount` in [0, 1], returned as RGB in [0, 1]."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    r = r + (255 - r) * amount
    g = g + (255 - g) * amount
    b = b + (255 - b) * amount
    return (r / 255, g / 255, b / 255)


def write_system_map(figures_dir: Path) -> None:
    """Draw the evidence-to-budget measurement system map as Figure 13.1."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13.0, 6.3))
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0.05, 6.6)
    ax.axis("off")

    ax.text(6.6, 6.32, "Unified Measurement: From Evidence to Budget",
            ha="center", va="center", fontsize=17, fontweight="bold", color=C_TITLE)

    for x, label in [(1.95, "MEASUREMENT EVIDENCE"), (6.70, "CHANNEL"), (10.6, "DECISION")]:
        ax.text(x, 5.62, label, ha="center", va="center", fontsize=10.5,
                fontweight="bold", color=C_HEAD)

    def pill(xc, yc, w, h, color, title, sub, title_size=13.0, sub_size=9.5):
        x0, y0 = xc - w / 2, yc - h / 2
        ax.add_patch(FancyBboxPatch(
            (x0, y0), w, h, boxstyle="round,pad=0,rounding_size=0.16",
            facecolor=_tint(color, 0.87), edgecolor=color, linewidth=1.8, zorder=3))
        ax.text(xc, yc + h * 0.17 if sub else yc, title, ha="center", va="center",
                fontsize=title_size, fontweight="bold", color=color, zorder=4)
        if sub:
            ax.text(xc, yc - h * 0.24, sub, ha="center", va="center",
                    fontsize=sub_size, color=C_SUB, zorder=4)
        return {"l": (x0, yc), "r": (x0 + w, yc), "t": (xc, y0 + h), "b": (xc, y0)}

    # ── Evidence sources ────────────────────────────────────────────────
    sources = [
        ("Attribution", "fast weekly path signal", "#2F6B9A", 4.75),
        ("Experiments", "causal lift, one action", "#2A9D8F", 3.65),
        ("Natural event", "baseline control", "#C77D2B", 2.55),
        ("MMM", "portfolio response curves", "#7A68A6", 1.45),
    ]
    src = []
    for title, sub, color, yc in sources:
        src.append((pill(1.95, yc, 2.55, 0.92, color, title, sub), color))

    # ── Tributaries funnel into a single reconciled point ───────────────
    conf = (4.55, 3.10)
    for (anchor, color), (_t, _s, _c, yc) in zip(src, sources):
        rad = -(3.10 - yc) * 0.14
        ax.add_patch(FancyArrowPatch(
            anchor["r"], conf, connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-", lw=1.5, color=_tint(color, 0.15), alpha=0.85, zorder=2))

    # ── Channel record spine ────────────────────────────────────────────
    rx0, rw, ry0, rh = 5.25, 3.05, 1.20, 3.75
    rxc = rx0 + rw / 2
    ax.add_patch(FancyBboxPatch(
        (rx0, ry0), rw, rh, boxstyle="round,pad=0,rounding_size=0.14",
        facecolor="#F8FAFC", edgecolor=C_SPINE, linewidth=1.9, zorder=3))
    rows = [
        ("field", "causal-anchored"),
        ("email", "mmm-ready"),
        ("digital", "directional"),
        ("paid media", "directional"),
    ]
    for (name, tier), yy in zip(rows, [3.87, 3.34, 2.81, 2.28]):
        ax.plot(rx0 + 0.40, yy, marker="o", markersize=8, color=CH_COLOR[name], zorder=5)
        ax.text(rx0 + 0.66, yy, name, ha="left", va="center",
                fontsize=9.3, fontweight="bold", color=C_TEXT, zorder=5)
        ax.text(rx0 + rw - 0.16, yy, tier, ha="right", va="center",
                fontsize=8.6, color=C_SUB, zorder=5)

    ax.add_patch(FancyArrowPatch(
        (4.72, 3.10), (rx0 - 0.02, 3.10), arrowstyle="-|>", mutation_scale=17,
        lw=2.3, color=C_SPINE, zorder=4))

    # ── Governed outputs ────────────────────────────────────────────────
    guard = pill(10.55, 3.95, 2.55, 0.98, "#C89433", "Budget move", "")
    ntest = pill(10.55, 1.95, 2.55, 0.98, "#5B7089", "Next test", "")
    ax.add_patch(FancyArrowPatch(
        (rx0 + rw + 0.02, 3.65), guard["l"], connectionstyle="arc3,rad=-0.12",
        arrowstyle="-|>", mutation_scale=16, lw=2.0, color=C_SPINE, zorder=4))
    ax.add_patch(FancyArrowPatch(
        (rx0 + rw + 0.02, 2.55), ntest["l"], connectionstyle="arc3,rad=0.12",
        arrowstyle="-|>", mutation_scale=16, lw=2.0, color=C_SPINE, zorder=4))

    # ── Feedback loop returns via three straight segments ───────────────
    loop_y = 0.62                       # horizontal run below both columns
    lx_start = ntest["b"][0]            # under Next test (Decision column)
    lx_end = 1.95                       # under the measurement-evidence column
    ev_bottom = 1.45 - 0.92 / 2         # bottom edge of the lowest evidence pill
    ax.plot([lx_start, lx_start], [ntest["b"][1] - 0.02, loop_y],
            color=C_HEAD, lw=1.6, zorder=1)
    ax.plot([lx_start, lx_end], [loop_y, loop_y],
            color=C_HEAD, lw=1.6, zorder=1)
    ax.add_patch(FancyArrowPatch(
        (lx_end, loop_y), (lx_end, ev_bottom - 0.02), arrowstyle="-|>",
        mutation_scale=15, lw=1.6, color=C_HEAD, zorder=1))
    ax.text(6.35, 0.34, "measurement loop", ha="center", va="center",
            fontsize=9.5, color=C_HEAD, style="italic", zorder=2)

    for ext in ("png", "svg"):
        fig.savefig(str(figures_dir / f"figure_13_1_measurement_system_map.{ext}"),
                    bbox_inches="tight", pad_inches=0.15, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "assets" / "figures"
    write_system_map(out)
    print(f"Wrote figure_13_1_measurement_system_map to {out}")
