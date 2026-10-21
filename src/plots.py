"""Small plotting helpers shared by the first notebooks (table snapshots and the cream 'first look' style)."""
import numpy as np
import matplotlib.pyplot as plt

# cream paper, dark teal ink, brick-red accent: the look of the first-look figures
INK, ACCENT, TEAL, PAPER, ROW, GRID, MUTED = "#17323D", "#C0452B", "#4E8C8C", "#FFFDF8", "#F3EBDF", "#E4DCCF", "#6B7378"
CREAM = {
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10,
    "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
}


def table_png(df, title, path, note=None):
    """Draw a small table as an image (used for the file snapshots)."""
    df = df.fillna("").astype(str)
    # column widths follow the longest text in each column; bold headers need a little more room
    chars = np.array([max(1.25 * len(str(c)), df[c].str.len().max()) for c in df.columns], dtype=float) + 2
    fig, ax = plt.subplots(figsize=(0.085 * chars.sum(), 0.3 * (len(df) + 1)))
    ax.axis("off")
    t = ax.table(cellText=df.values, colLabels=df.columns, bbox=[0, 0, 1, 1], cellLoc="left", colLoc="left",
                 colWidths=list(chars / chars.sum()))
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_facecolor(INK)
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor(PAPER if r % 2 else ROW)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", pad=6)
    if note:
        ax.text(0, -0.04, note, transform=ax.transAxes, fontsize=8, color=MUTED, va="top")
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.12)
    plt.show()
    return path
