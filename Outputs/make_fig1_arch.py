"""Regenerate Figure 1 (architecture) as a vector diagram.

Two candidates, both matplotlib/vector, no PPTX round-trip:
  Candidate 1 -- "cleaned original": same disease/target two-tower layout as
    the current PPTX export, redrawn with consistent typography, a validated
    colorblind-safe palette, and complete layer labels taken from
    dl_model_def.py (LayerNorm, ELU, dropout rates, the disease-only 64-dim
    ID embedding).
  Candidate 2 -- candidate 1 plus a bottom inference strip showing
    encode-once / score-all-pairs at the actual scale (4,479 targets x
    ~19,000 diseases), and a cold-start marker on the target tower.

Palette (validated via dataviz skill's validate_palette.js, light mode,
3-slot categorical): blue #2a78d6 (disease), aqua #1baf7a (target), violet
#4a3aa7 (shared / output). Fill tints are the same hue at low alpha; borders
and text stay full-strength, so the aqua-vs-surface contrast WARN is covered
by the relief rule (all labels are visible black text, never color-alone).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

HERE = __import__("pathlib").Path(__file__).resolve().parent

BLUE, AQUA, VIOLET = "#2a78d6", "#1baf7a", "#4a3aa7"
INK, MUTED = "#0b0b0b", "#52514e"


def tint(hex_color, alpha=0.10):
    return hex_color + format(int(alpha * 255), "02x")


def box(ax, xy, w, h, title, lines, edge, fill, title_color=None):
    x, y = xy
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.4, edgecolor=edge, facecolor=fill, zorder=2,
    ))
    ax.text(x + 0.15, y + h - 0.28, title, fontsize=11.5, fontweight="bold",
            color=title_color or edge, va="top", ha="left", zorder=3)
    for i, line in enumerate(lines):
        ax.text(x + 0.15, y + h - 0.62 - i * 0.30, line, fontsize=9.3,
                color=INK, va="top", ha="left", zorder=3)


def arrow(ax, p0, p1, color=MUTED, lw=1.6):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=14, linewidth=lw,
        color=color, zorder=1,
    ))


def draw_two_towers(ax, y0):
    """Shared disease/target tower stack; returns (lx, rx, col_w, cx, y0)."""
    col_w, gap = 4.7, 1.0
    lx, rx = 0.5, 0.5 + col_w + gap

    inputs_h, vec_h, tower_h = 2.05, 0.85, 2.05
    inputs_y = y0 + tower_h + 0.5 + vec_h + 0.5
    vec_y = y0 + tower_h + 0.5
    tower_y = y0 + 0.0

    box(ax, (lx, inputs_y), col_w, inputs_h, "Disease inputs",
        ["Name, description, exact synonyms",
         "Ontology parents (EFO terms)",
         "Therapeutic areas, phenotypes",
         "Disease-ID embedding (64-dim)"],
        BLUE, tint(BLUE))
    box(ax, (rx, inputs_y), col_w, inputs_h, "Target inputs",
        ["Gene symbol, name, synonyms",
         "GO terms, Reactome pathways",
         "UniProt function description",
         "Tractability + gnomAD constraint"],
        AQUA, tint(AQUA))

    box(ax, (lx, vec_y), col_w, vec_h, "Text vectorization",
        ["Bag-of-words count encoding"], BLUE, tint(BLUE, 0.06))
    box(ax, (rx, vec_y), col_w, vec_h, "Text vectorization",
        ["Bag-of-words count encoding"], AQUA, tint(AQUA, 0.06))

    box(ax, (lx, tower_y), col_w, tower_h, "Tower (Wide & Deep)",
        ["Wide: Dense 384, linear",
         "Deep: 384→LayerNorm→Dropout .35",
         "  →64→Dropout .15→384, ELU",
         "Residual add → 384-dim embedding"],
        BLUE, tint(BLUE))
    box(ax, (rx, tower_y), col_w, tower_h, "Tower (Wide & Deep)",
        ["Wide: Dense 384, linear",
         "Deep: 384→LayerNorm→Dropout .35",
         "  →64→Dropout .15→384, ELU",
         "Residual add → 384-dim embedding"],
        AQUA, tint(AQUA))

    for x in (lx, rx):
        arrow(ax, (x + col_w / 2, inputs_y), (x + col_w / 2, inputs_y - 0.28))
        arrow(ax, (x + col_w / 2, vec_y), (x + col_w / 2, vec_y - 0.28))

    cx = (lx + rx + col_w) / 2
    cy = tower_y - 1.05
    ax.add_patch(FancyBboxPatch(
        (cx - 1.2, cy - 0.5), 2.4, 1.0, boxstyle="round,pad=0.02,rounding_size=0.5",
        linewidth=1.6, edgecolor=VIOLET, facecolor=tint(VIOLET, 0.12), zorder=2))
    ax.text(cx, cy + 0.22, "Cosine similarity", fontsize=10.3, fontweight="bold",
            ha="center", va="center", color=VIOLET, zorder=3)
    ax.text(cx, cy - 0.15, "(normalized dot product)", fontsize=8.6,
            ha="center", va="center", color=MUTED, zorder=3)

    arrow(ax, (lx + col_w / 2, tower_y), (cx - 1.05, cy + 0.15))
    arrow(ax, (rx + col_w / 2, tower_y), (cx + 1.05, cy + 0.15))

    out_w = 5.2
    ox = cx - out_w / 2
    out_h = 1.15
    out_y = cy - 0.5 - 0.35 - out_h
    box(ax, (ox, out_y), out_w, out_h, "Predicted clinical association",
        ["Sigmoid: P(clinical trial entry)",
         "Linear: OTP evidence score (auxiliary)"],
        VIOLET, tint(VIOLET, 0.08), title_color=VIOLET)
    arrow(ax, (cx, cy - 0.5), (cx, out_y + out_h), color=VIOLET)

    top = inputs_y + inputs_h
    bottom = out_y
    return lx, rx, col_w, cx, top, bottom


CANVAS_W = 11.2


def candidate_1():
    fig, ax = plt.subplots(figsize=(CANVAS_W, 7.3))
    lx, rx, col_w, cx, top, bottom = draw_two_towers(ax, y0=0.3)
    ax.set_xlim(0, CANVAS_W)
    ax.set_ylim(bottom - 0.3, top + 0.75)
    ax.axis("off")
    ax.text(CANVAS_W / 2, top + 0.45, "OTRec two-tower architecture",
            fontsize=14.5, fontweight="bold", ha="center", color=INK)
    for ext in ("pdf", "png"):
        out = HERE / f"Fig1_candidate1.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print("wrote", out)
    plt.close(fig)


def candidate_2():
    fig, ax = plt.subplots(figsize=(CANVAS_W, 9.1))
    y0 = 1.9
    lx, rx, col_w, cx, top, bottom = draw_two_towers(ax, y0=y0)

    # Cold-start marker on the target tower's inputs box.
    inputs_top = top
    tx, ty = rx + col_w - 0.18, inputs_top - 0.18
    ax.add_patch(plt.Circle((tx, ty), 0.15, facecolor=AQUA, edgecolor="white",
                             linewidth=1.1, zorder=4))
    ax.text(tx, ty, "*", fontsize=10, fontweight="bold", color="white",
            ha="center", va="center", zorder=5)
    ax.text(rx + col_w + 0.28, ty, "held-out / unseen targets enter\nthrough annotations alone",
            fontsize=8.2, color=AQUA, va="center", ha="left", style="italic")

    # Inference strip below the towers.
    strip_h = 1.35
    strip_y = bottom - 0.35 - strip_h
    ax.add_patch(FancyBboxPatch(
        (0.3, strip_y), CANVAS_W - 0.6, strip_h,
        boxstyle="round,pad=0.03,rounding_size=0.08",
        linewidth=1.3, edgecolor=MUTED, facecolor=tint(MUTED, 0.05), zorder=1))
    ax.text(0.55, strip_y + strip_h - 0.28, "Inference: encode once, score all pairs",
            fontsize=10.5, fontweight="bold", color=INK, va="top", ha="left")
    ax.text(0.55, strip_y + strip_h - 0.64,
            "Every disease and every target is embedded once; ranking a pair is a single cosine similarity.",
            fontsize=8.8, color=MUTED, va="top", ha="left")
    ax.text(0.55, strip_y + strip_h - 0.98,
            "4,479 druggable targets  ×  ~19,000 OTP diseases  ≈  87M scored pairs at prediction time.",
            fontsize=8.8, color=MUTED, va="top", ha="left")

    arrow(ax, (cx, bottom - 0.35), (cx, strip_y + strip_h), color=MUTED, lw=1.2)

    ax.set_xlim(0, CANVAS_W)
    ax.set_ylim(strip_y - 0.2, top + 0.75)
    ax.axis("off")
    ax.text(CANVAS_W / 2, top + 0.45, "OTRec two-tower architecture",
            fontsize=14.5, fontweight="bold", ha="center", color=INK)
    for ext in ("pdf", "png"):
        out = HERE / f"Fig1_candidate2.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print("wrote", out)
    plt.close(fig)


if __name__ == "__main__":
    candidate_1()
    candidate_2()
