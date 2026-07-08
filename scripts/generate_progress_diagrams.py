"""
Generate diagrams for the supervisor-meeting progress report.

Produces 6 conceptual diagrams via matplotlib + copies 4 existing data plots
from /tmp into `additional docs/figures/`.

Run once before generate_progress_report.py if you want the embedded figures
refreshed.

Usage:
    .venv/bin/python scripts/generate_progress_diagrams.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FIG_DIR = Path("additional docs/figures")

# A small consistent palette
NAVY = "#1F3B6C"
TEAL = "#2A9D8F"
ORANGE = "#E76F51"
SAND = "#F4A261"
LIGHT_GRAY = "#E5E5E5"
MID_GRAY = "#BFBFBF"
DARK = "#222"

DPI = 150


# ---------- low-level helpers ----------

def box(ax, x, y, w, h, *, label, color=LIGHT_GRAY, edge=DARK,
        fontsize=10, bold=False, multiline_color=None):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.05",
        linewidth=1.4, edgecolor=edge, facecolor=color,
    )
    ax.add_patch(p)
    ax.text(
        x + w / 2, y + h / 2, label,
        ha="center", va="center",
        fontsize=fontsize, weight="bold" if bold else "normal",
        color=multiline_color or DARK,
    )


def arrow(ax, x1, y1, x2, y2, *, label=None, color=DARK, style="->", lw=1.4,
          curve=0.0):
    cs = f"arc3,rad={curve}" if curve else "arc3,rad=0"
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, linewidth=lw, color=color,
        mutation_scale=14, connectionstyle=cs,
    )
    ax.add_patch(a)
    if label is not None:
        ax.text(
            (x1 + x2) / 2, (y1 + y2) / 2 + 1.0, label,
            ha="center", va="bottom", fontsize=9, color=color, style="italic"
        )


def setup(ax, xmax=100, ymax=100, title: str | None = None):
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.set_aspect("equal")
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=13, weight="bold", color=NAVY)


def save(fig, name: str) -> None:
    out = FIG_DIR / name
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {out.name}  ({out.stat().st_size / 1024:.0f} KB)")


# ---------- diagram 1: System architecture (3-tier) ----------

def fig_system_architecture():
    fig, ax = plt.subplots(figsize=(11, 7.5))
    setup(ax, xmax=100, ymax=100, title="System architecture — Cloud / Network / Edge")

    # Cloud layer
    box(ax, 5, 72, 90, 20, label="", color="#EDF1F8", edge=NAVY)
    ax.text(50, 89, "CLOUD / SERVER  (AWS or laptop)", ha="center", va="center",
            fontsize=12, weight="bold", color=NAVY)
    box(ax, 10, 75, 23, 10, label="Flower Server\n(FedAvg)", color="white", edge=NAVY)
    box(ax, 38, 75, 24, 10, label="Training Engine\n(Strategy A / B)", color="white", edge=NAVY)
    box(ax, 67, 75, 23, 10, label="Model Repository\n(global weights)", color="white", edge=NAVY)

    # Network
    ax.add_patch(FancyBboxPatch((5, 56), 90, 8,
                                 boxstyle="round,pad=0.05",
                                 facecolor="#FDF4E3", edgecolor=SAND, linewidth=1.4))
    ax.text(50, 60, "NETWORK    MQTT (TLS) over 5 GHz Wi-Fi", ha="center",
            va="center", fontsize=11, weight="bold", color="#A56908")

    # Edge nodes
    box(ax, 5, 18, 42, 32, label="", color="#EAF7F4", edge=TEAL)
    ax.text(26, 47, "EDGE NODE A  (Raspberry Pi 4)", ha="center", va="center",
            fontsize=11, weight="bold", color=TEAL)
    box(ax,  8, 36, 17, 7, label="Camera\n(picamera2)", color="white", edge=TEAL, fontsize=9)
    box(ax, 28, 36, 17, 7, label="I2S Mic", color="white", edge=TEAL, fontsize=9)
    box(ax,  8, 27, 37, 7, label="STFT  +  YOLO  +  Strategy", color="white",
        edge=TEAL, fontsize=9)
    box(ax,  8, 19, 37, 7, label="Local trainer  (Strategy C only)",
        color="white", edge=TEAL, fontsize=9)

    box(ax, 53, 18, 42, 32, label="", color="#EAF7F4", edge=TEAL)
    ax.text(74, 47, "EDGE NODE B  (Raspberry Pi 4)", ha="center", va="center",
            fontsize=11, weight="bold", color=TEAL)
    box(ax, 56, 36, 17, 7, label="Camera", color="white", edge=TEAL, fontsize=9)
    box(ax, 76, 36, 17, 7, label="I2S Mic", color="white", edge=TEAL, fontsize=9)
    box(ax, 56, 27, 37, 7, label="STFT  +  YOLO  +  Strategy", color="white",
        edge=TEAL, fontsize=9)
    box(ax, 56, 19, 37, 7, label="Local trainer", color="white", edge=TEAL,
        fontsize=9)

    # Arrows
    arrow(ax, 26, 50, 26, 56, color=TEAL)
    arrow(ax, 74, 50, 74, 56, color=TEAL)
    arrow(ax, 26, 64, 26, 72, color=NAVY)
    arrow(ax, 74, 64, 74, 72, color=NAVY)
    ax.text(50, 68, "data / weights ⇆ model updates", ha="center",
            fontsize=9, style="italic", color="#666")

    save(fig, "01_system_architecture.png")


# ---------- diagram 2: Two models — YOLO vs Fusion ----------

def fig_two_models():
    fig, ax = plt.subplots(figsize=(11, 6.5))
    setup(ax, xmax=120, ymax=90, title="The two models — never confuse them")

    # YOLO box
    box(ax, 5, 25, 45, 50, label="", color="#EDEDED", edge=MID_GRAY)
    ax.text(27.5, 70, "YOLOv8", ha="center", fontsize=14, weight="bold", color=DARK)
    ax.text(27.5, 65, "THE LABELER", ha="center", fontsize=10, weight="bold", color="#666")
    ax.text(27.5, 56, "Pre-trained on COCO.\nNever updated by us.", ha="center",
            fontsize=10, style="italic", color="#444")
    box(ax,  8, 40, 18, 8, label="image", color="white", edge=MID_GRAY)
    box(ax, 29, 40, 18, 8, label="label\n('car', 'dog', …)", color="white",
        edge=MID_GRAY, fontsize=9)
    arrow(ax, 26, 44, 29, 44, color=DARK)
    ax.text(27.5, 32, "[ frozen — used like a calculator ]", ha="center",
            fontsize=10, color="#888")

    # Fusion model box
    box(ax, 70, 15, 47, 70, label="", color="#FFF0E8", edge=ORANGE)
    ax.text(93.5, 80, "Fusion Model", ha="center", fontsize=14, weight="bold", color=ORANGE)
    ax.text(93.5, 75, "THE THESIS CONTRIBUTION", ha="center", fontsize=10,
            weight="bold", color=ORANGE)
    ax.text(93.5, 67, "Custom: MobileNetV3 +\naudio CNN + fusion head.",
            ha="center", fontsize=10, style="italic", color="#444")
    box(ax, 73, 55, 19, 7, label="image", color="white", edge=ORANGE, fontsize=9)
    box(ax, 95, 55, 19, 7, label="spectrogram", color="white", edge=ORANGE,
        fontsize=9)
    box(ax, 80, 42, 27, 8, label="vision + audio fusion", color="white",
        edge=ORANGE, fontsize=9)
    arrow(ax, 82, 55, 87, 50, color=ORANGE)
    arrow(ax, 105, 55, 100, 50, color=ORANGE)
    box(ax, 80, 28, 27, 8, label="class prediction", color=ORANGE, edge=ORANGE,
        fontsize=10, multiline_color="white", bold=True)
    arrow(ax, 93.5, 42, 93.5, 36, color=ORANGE)
    ax.text(93.5, 20, "[ trained on weakly-labeled data ]", ha="center",
            fontsize=10, color="#A0521B")

    # Connecting arrow: YOLO labels → fusion training
    arrow(ax, 47, 44, 73, 44, color=NAVY, lw=2.0, curve=-0.25)
    ax.text(60, 56, "labels supplied for training",
            ha="center", fontsize=10, style="italic", color=NAVY)

    save(fig, "02_two_models.png")


# ---------- diagram 3: Three strategies side-by-side ----------

def fig_three_strategies():
    fig, ax = plt.subplots(figsize=(12, 8))
    setup(ax, xmax=120, ymax=100,
          title="Three strategies — what runs where, what gets transmitted")

    # Three columns
    cols = [
        ("A — Centralized", "#FCD7CC", ORANGE,
         "YOLOv8 labeling only",
         "Raw audio + label\n(~4 MB / batch)",
         "STFT + full training\n+ broadcast model"),
        ("B — Hybrid STFT", "#FFEDC2", "#C89026",
         "YOLOv8 + STFT",
         "Spectrograms + label\n(~3 MB / batch)",
         "Full training\n+ broadcast model"),
        ("C — Federated", "#CDEAE3", TEAL,
         "YOLOv8 + STFT\n+ LOCAL TRAINING",
         "Model weights only\n(~2 MB / round)",
         "FedAvg averaging\n+ broadcast"),
    ]

    col_w = 34
    gap = 5
    start_x = 5

    for i, (name, fill, edge, edge_step, transmit, server_step) in enumerate(cols):
        x = start_x + i * (col_w + gap)
        # Column header
        box(ax, x, 87, col_w, 7, label=name, color=fill, edge=edge, bold=True,
            fontsize=12)
        # Edge step
        ax.text(x + col_w / 2, 80, "EDGE NODE", ha="center", fontsize=10,
                weight="bold", color="#444")
        box(ax, x, 67, col_w, 11, label=edge_step, color="white", edge=edge,
            fontsize=10)
        # Transmit
        ax.text(x + col_w / 2, 60, "TRANSMITS  v", ha="center", fontsize=10,
                weight="bold", color="#444")
        box(ax, x, 46, col_w, 12, label=transmit, color=fill, edge=edge,
            fontsize=10)
        # Server step
        ax.text(x + col_w / 2, 39, "SERVER", ha="center", fontsize=10,
                weight="bold", color="#444")
        box(ax, x, 24, col_w, 12, label=server_step, color="white", edge=edge,
            fontsize=10)
        # Arrows between steps
        arrow(ax, x + col_w / 2, 67, x + col_w / 2, 60, color=edge)
        arrow(ax, x + col_w / 2, 46, x + col_w / 2, 39, color=edge)

    # Bottom legend
    ax.text(
        60, 12,
        "Lower row of each column = bandwidth / privacy cost.\n"
        "Strategy A: highest bandwidth, lowest privacy.   "
        "Strategy C: lowest bandwidth, highest privacy.",
        ha="center", va="center", fontsize=10, color="#444",
    )

    save(fig, "03_three_strategies.png")


# ---------- diagram 4: Auto-labeling pipeline ----------

def fig_auto_labeling():
    fig, ax = plt.subplots(figsize=(11, 6))
    setup(ax, xmax=120, ymax=80,
          title="Auto-labeling: vision labels audio (weak supervision)")

    # Camera path
    box(ax, 5, 55, 20, 12, label="Camera\n(frame at T)", color="white",
        edge=NAVY, fontsize=10)
    arrow(ax, 25, 61, 35, 61, color=NAVY)
    box(ax, 35, 55, 18, 12, label="YOLOv8\nlabeler", color=LIGHT_GRAY,
        edge=MID_GRAY, fontsize=10, bold=True)
    arrow(ax, 53, 61, 63, 61, color=NAVY)
    box(ax, 63, 55, 18, 12, label="'car'\n(label)", color="white", edge=NAVY,
        fontsize=10, bold=True)

    # Audio path
    box(ax, 5, 25, 20, 12, label="Microphone\n(audio at T)", color="white",
        edge=TEAL, fontsize=10)
    arrow(ax, 25, 31, 35, 31, color=TEAL)
    box(ax, 35, 25, 18, 12, label="STFT\n+ mel-scale", color=LIGHT_GRAY,
        edge=TEAL, fontsize=10, bold=True)
    arrow(ax, 53, 31, 63, 31, color=TEAL)
    box(ax, 63, 25, 18, 12, label="spectrogram\n(51 × 128)", color="white",
        edge=TEAL, fontsize=10, bold=True)

    # Pairing
    box(ax, 90, 38, 25, 18, label="Training pair\n{spectrogram,\n label = 'car'}",
        color="#FFF0E8", edge=ORANGE, fontsize=11, bold=True)
    arrow(ax, 81, 61, 90, 50, color=ORANGE, curve=0.2)
    arrow(ax, 81, 31, 90, 44, color=ORANGE, curve=-0.2)

    # Footer
    ax.text(60, 10,
            "Assumption: the sound at time T is produced by what the camera sees at time T.\n"
            "Noisy but works at scale — measured precision on VGGSound: ~80 %.",
            ha="center", va="center", fontsize=10, style="italic", color="#444")

    save(fig, "04_auto_labeling.png")


# ---------- diagram 5: Fusion model architecture ----------

def fig_fusion_arch():
    fig, ax = plt.subplots(figsize=(11, 9))
    setup(ax, xmax=100, ymax=110,
          title="Fusion model architecture (1.33 M params, 390 K trainable)")

    # Vision branch (left)
    box(ax, 5, 92, 35, 9, label="Image input (224 × 224 × 3)", color=LIGHT_GRAY,
        edge=NAVY, fontsize=10)
    arrow(ax, 22.5, 92, 22.5, 88, color=NAVY)
    box(ax, 5, 75, 35, 12, label="MobileNetV3-Small\n(pre-trained, FROZEN)",
        color="#EDEDED", edge=NAVY, fontsize=11, bold=True)
    arrow(ax, 22.5, 75, 22.5, 71, color=NAVY)
    box(ax, 5, 60, 35, 10, label="Vision features (256-d)", color="white",
        edge=NAVY, fontsize=10)

    # Audio branch (right)
    box(ax, 60, 92, 35, 9, label="Spectrogram input (51 × 128 × 1)",
        color=LIGHT_GRAY, edge=TEAL, fontsize=10)
    arrow(ax, 77.5, 92, 77.5, 88, color=TEAL)
    box(ax, 60, 75, 35, 12, label="Audio CNN\n(3 conv blocks, trained)",
        color="#CDEAE3", edge=TEAL, fontsize=11, bold=True)
    arrow(ax, 77.5, 75, 77.5, 71, color=TEAL)
    box(ax, 60, 60, 35, 10, label="Audio features (128-d)", color="white",
        edge=TEAL, fontsize=10)

    # Fusion head
    arrow(ax, 22.5, 60, 40, 48, color=ORANGE, curve=-0.25)
    arrow(ax, 77.5, 60, 60, 48, color=ORANGE, curve=0.25)
    box(ax, 30, 36, 40, 10, label="Concatenate (384-d)", color="#FFF0E8",
        edge=ORANGE, fontsize=11, bold=True)
    arrow(ax, 50, 36, 50, 32, color=ORANGE)
    box(ax, 30, 21, 40, 10, label="Dense(256) → Dropout → Dense(N)",
        color="#FFF0E8", edge=ORANGE, fontsize=11, bold=True)
    arrow(ax, 50, 21, 50, 17, color=ORANGE)
    box(ax, 30, 6, 40, 9, label="Softmax → class prediction", color=ORANGE,
        edge=ORANGE, fontsize=11, bold=True, multiline_color="white")

    # Labels
    ax.text(22.5, 105, "VISION BRANCH", ha="center", fontsize=10, weight="bold",
            color=NAVY)
    ax.text(77.5, 105, "AUDIO BRANCH", ha="center", fontsize=10, weight="bold",
            color=TEAL)

    save(fig, "05_fusion_architecture.png")


# ---------- diagram 6: Timeline / progress ----------

def fig_timeline():
    fig, ax = plt.subplots(figsize=(11, 5))
    setup(ax, xmax=100, ymax=60, title="Thesis timeline — what's done and what's next")

    rows = [
        ("Bootstrap (YOLO + STFT + AudioCNN)",   5, 32, "#2A9D8F", "DONE"),
        ("Strategy comparison harness (A/B/C)",  37, 18, "#7BC8B5", "NEXT"),
        ("Pi sensor integration",                55, 12, SAND,      "blocked on hardware"),
        ("Thesis experiments",                   67, 18, SAND,      ""),
        ("Thesis writing",                       85, 12, "#F4D1B7", ""),
    ]

    y = 40
    for label, x_start, width, color, status in rows:
        # Bar
        rect = mpatches.Rectangle(
            (x_start, y), width, 6,
            facecolor=color, edgecolor=DARK, linewidth=1.2,
        )
        ax.add_patch(rect)
        ax.text(x_start + width / 2, y + 3, label, ha="center", va="center",
                fontsize=10, weight="bold", color="white" if status == "DONE" else DARK)
        if status:
            ax.text(x_start + width + 1, y + 3, status, va="center",
                    fontsize=9, style="italic", color="#666")
        y -= 7

    # Today marker
    today_x = 32  # just past bootstrap
    ax.axvline(today_x, ymin=0.18, ymax=0.85, linestyle="--", color=ORANGE,
               linewidth=1.6)
    ax.text(today_x, 7, "TODAY", ha="center", fontsize=10, color=ORANGE,
            weight="bold")

    # Axis-style months
    ax.text(50, -2, "Approx. months from project start →", ha="center",
            fontsize=9, color="#666", style="italic")

    save(fig, "06_timeline.png")


# ---------- main ----------

def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating conceptual diagrams...")
    fig_system_architecture()
    fig_two_models()
    fig_three_strategies()
    fig_auto_labeling()
    fig_fusion_arch()
    fig_timeline()

    print("\nCopying data plots from /tmp...")
    copies = [
        ("/tmp/stft_grid.png",        "07_spectrogram_grid.png"),
        ("/tmp/training_curves.png",  "08_training_curves.png"),
        ("/tmp/confusion_matrix.png", "09_confusion_matrix.png"),
    ]
    for src, dest in copies:
        src_p = Path(src)
        if src_p.exists():
            dest_p = FIG_DIR / dest
            shutil.copy2(src_p, dest_p)
            print(f"  copied {src_p.name} → {dest}")
        else:
            print(f"  WARN: {src_p} not present — re-run the relevant test/train "
                  f"script if you need {dest}")

    print(f"\nFigures in {FIG_DIR}:")
    for f in sorted(FIG_DIR.iterdir()):
        print(f"  {f.name:<40} {f.stat().st_size / 1024:>7.1f} KB")

    return 0


if __name__ == "__main__":
    sys.exit(main())
