"""
Generate the three methodology diagrams for the thesis chapter:
  1. system_architecture.png   - overall system architecture
  2. audit_workflow.png        - label-independent k-NN auditing workflow
  3. pipeline_integration.png  - integration into the FLIP training pipeline

Output is written to the LaTeX template's figures/ directory so the chapter
can reference them with \includegraphics in the usual template style.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT_DIR = Path(__file__).resolve().parents[2] / "Thesis Template UG" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EDGE = "#333333"
FILL_MAIN = "#e8eef7"
FILL_DEF = "#fdecea"
FILL_DATA = "#eafaf1"
FILL_NEUTRAL = "#f4f4f4"


def box(ax, xy, w, h, text, fill=FILL_MAIN, fontsize=10, bold=False):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.3, edgecolor=EDGE, facecolor=fill,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center", fontsize=fontsize,
        weight="bold" if bold else "normal", wrap=True,
    )
    return (x + w / 2, y + h / 2)


def arrow(ax, p1, p2, text=None, fontsize=8.5):
    a = FancyArrowPatch(
        p1, p2, arrowstyle="-|>", mutation_scale=14,
        linewidth=1.2, color=EDGE, shrinkA=2, shrinkB=2,
    )
    ax.add_patch(a)
    if text:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx, my + 0.12, text, ha="center", va="bottom", fontsize=fontsize)


def base(figsize):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 10)
    ax.axis("off")
    return fig, ax


# --------------------------------------------------------------------------
# 1. System architecture
# --------------------------------------------------------------------------
def system_architecture():
    fig, ax = base((8.0, 6.6))
    ax.set_ylim(0, 12)

    box(ax, (3.2, 11.0), 3.6, 0.9,
        "Training images + assigned labels", fill=FILL_DATA, bold=True)

    box(ax, (0.4, 9.0), 4.2, 1.1,
        "Image branch\n(labels discarded)", fill=FILL_NEUTRAL)
    box(ax, (5.4, 9.0), 4.2, 1.1,
        "Label branch\n(assigned hard labels)", fill=FILL_NEUTRAL)

    box(ax, (0.4, 7.0), 4.2, 1.2,
        "Frozen DINOv2 encoder\n(self-supervised, label-blind)",
        fill=FILL_MAIN, bold=True)
    box(ax, (0.4, 5.2), 4.2, 1.0,
        "L2-normalised feature\nembeddings (cache)", fill=FILL_MAIN)

    box(ax, (2.6, 3.2), 4.8, 1.3,
        "k-NN neighbourhood-consistency\nauditing module",
        fill=FILL_DEF, bold=True)

    box(ax, (5.4, 5.2), 4.2, 1.0,
        "Per-sample suspicion\nscores  s(i)", fill=FILL_DEF)

    box(ax, (0.6, 1.2), 3.8, 1.0,
        "Removal policy\n(auto / budget / fixed-N)", fill=FILL_DEF)
    box(ax, (5.6, 1.2), 3.8, 1.0,
        "Cleaned training set", fill=FILL_DATA, bold=True)

    arrow(ax, (4.4, 11.0), (2.5, 10.1))
    arrow(ax, (5.6, 11.0), (7.5, 10.1))
    arrow(ax, (2.5, 9.0), (2.5, 8.2))
    arrow(ax, (2.5, 7.0), (2.5, 6.2))
    arrow(ax, (2.5, 5.2), (3.8, 4.5))
    arrow(ax, (7.5, 9.0), (7.5, 6.2))
    arrow(ax, (7.5, 5.2), (6.2, 4.5))
    arrow(ax, (4.2, 3.2), (3.0, 2.2))
    arrow(ax, (2.5, 1.2), (5.6, 1.5))

    fig.tight_layout()
    fig.savefig(OUT_DIR / "system_architecture.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# 2. Auditing workflow
# --------------------------------------------------------------------------
def audit_workflow():
    fig, ax = base((6.6, 9.2))
    ax.set_ylim(0, 13)

    steps = [
        ("Input: image x_i and its assigned label y_i", FILL_DATA),
        ("Preprocess: resize 224, centre crop,\nImageNet normalisation", FILL_NEUTRAL),
        ("Encode with frozen DINOv2 -> f_i", FILL_MAIN),
        ("L2-normalise: z_i = f_i / ||f_i||", FILL_MAIN),
        ("Cosine similarity to all samples\nsim(i,j) = z_i . z_j", FILL_MAIN),
        ("Select k nearest neighbours\n(exclude self)", FILL_MAIN),
        ("Score disagreement s(i)\nbetween y_i and neighbour labels", FILL_DEF),
        ("Rank samples by score s(i)", FILL_DEF),
        ("Flag suspicious samples\nvia removal policy", FILL_DEF),
        ("Output: kept / removed indices", FILL_DATA),
    ]

    y = 12.0
    h = 0.92
    gap = 0.32
    centres = []
    for text, fill in steps:
        c = box(ax, (1.4, y), 6.8, h, text, fill=fill,
                bold=(fill in (FILL_DEF, FILL_DATA)))
        centres.append((c[0], y, y + h))
        y -= (h + gap)

    for i in range(len(centres) - 1):
        top = (centres[i][0], centres[i][1])
        bot = (centres[i + 1][0], centres[i + 1][2])
        arrow(ax, top, bot)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "audit_workflow.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# 3. Pipeline integration
# --------------------------------------------------------------------------
def pipeline_integration():
    fig, ax = base((9.0, 5.2))
    ax.set_ylim(0, 8)

    box(ax, (0.2, 6.4), 2.0, 1.0, "train_expert", fill=FILL_NEUTRAL)
    box(ax, (2.6, 6.4), 2.2, 1.0, "generate_labels", fill=FILL_NEUTRAL)
    box(ax, (5.2, 6.4), 1.9, 1.0, "select_flips", fill=FILL_NEUTRAL)
    box(ax, (7.5, 6.4), 2.2, 1.0, "labels.npy /\ntrue.npy",
        fill=FILL_DATA, bold=True)

    arrow(ax, (2.2, 6.9), (2.6, 6.9))
    arrow(ax, (4.8, 6.9), (5.2, 6.9))
    arrow(ax, (7.1, 6.9), (7.5, 6.9))

    box(ax, (2.4, 2.8), 5.4, 2.4,
        "train_user_defense\n\n1. Build user dataset\n2. SSL features (cached)\n"
        "3. k-NN scoring\n4. Apply mode: none / remove\n5. Train user model",
        fill=FILL_DEF, bold=True, fontsize=9.5)

    box(ax, (0.2, 0.2), 2.4, 1.0, "DINOv2 encoder\n(frozen)", fill=FILL_MAIN)
    box(ax, (7.4, 0.2), 2.4, 1.0,
        "CTA / PTA,\nscores, metrics", fill=FILL_DATA, bold=True)

    arrow(ax, (8.6, 6.4), (6.0, 5.2))
    arrow(ax, (1.4, 1.2), (3.2, 2.8))
    arrow(ax, (6.6, 2.8), (8.2, 1.2))

    fig.tight_layout()
    fig.savefig(OUT_DIR / "pipeline_integration.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    system_architecture()
    audit_workflow()
    pipeline_integration()
    print(f"Wrote diagrams to {OUT_DIR}")
