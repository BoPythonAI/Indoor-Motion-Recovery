"""Build English result figures directly from the frozen evaluation JSON files."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/assets/charts"
STRENGTHS = ["0.0", "0.45", "0.75"]
LABELS = ["No push", "Medium push\n0.45 BW", "Strong push\n0.75 BW"]
METHODS = [
    ("Baseline", "baseline_development.json", "#758697"),
    ("Teacher", "teacher_development.json", "#5575ad"),
    ("H1 BC (selected)", "bc_development.json", "#087f89"),
    ("H1 diffusion", "diffusion_development.json", "#ba7541"),
    ("Nominal-only BC", "nominal_only.json", "#9274aa"),
]
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "text.color": "#173343", "axes.labelcolor": "#173343",
    "xtick.color": "#526776", "ytick.color": "#526776",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.edgecolor": "#cfdae1",
    "svg.fonttype": "none", "svg.hashsalt": "indoor-motion-recovery",
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def read(name):
    return json.loads((ROOT / "evidence" / name).read_text(encoding="utf-8"))


def heading(fig, title, subtitle):
    fig.text(.075, .955, title, fontsize=21, fontweight="bold", va="top")
    fig.text(.075, .895, subtitle, fontsize=11, color="#526776", va="top")


def style(ax, percent=True):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#e6ecf0", linewidth=.8)
    ax.tick_params(axis="both", length=0, pad=9)
    if percent:
        ax.set_ylim(0, 112)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        ax.set_ylabel("Success rate")


def save(fig, name):
    for ext in ["svg", "png"]:
        fig.savefig(OUT / f"{name}.{ext}", dpi=180,
                    metadata={"Date": None} if ext == "svg" else None)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = [read(filename) for _, filename, _ in METHODS]
    rates = np.array([[100 * d["summary"][s]["success"] /
                       d["summary"][s]["total"] for s in STRENGTHS] for d in data])
    with (OUT / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "push_bw", "success", "total", "success_percent"])
        for (name, _, _), d in zip(METHODS, data):
            for s in STRENGTHS:
                r = d["summary"][s]
                writer.writerow([name, s, r["success"], r["total"],
                                 100 * r["success"] / r["total"]])

    fig, axes = plt.subplots(1, 3, figsize=(14, 6), sharey=True)
    fig.subplots_adjust(left=.075, right=.98, top=.70, bottom=.25, wspace=.18)
    heading(fig, "Recovery across push strengths",
            "Development evaluation  /  5 motion sources  /  40 recorded cases per strength")
    for k, ax in enumerate(axes):
        bars = ax.bar(np.arange(5), rates[:, k], color=[m[2] for m in METHODS], width=.68)
        for bar, val, d in zip(bars, rates[:, k], data):
            ax.text(bar.get_x()+bar.get_width()/2, val+2, f"{val:g}%",
                    ha="center", fontsize=10, fontweight="bold")
            ax.text(bar.get_x()+bar.get_width()/2, 5,
                    f'{d["summary"][STRENGTHS[k]]["success"]}/40',
                    ha="center", fontsize=9, color="white", fontweight="bold")
        style(ax)
        ax.set_title(LABELS[k].replace("\n", "  /  "), loc="left", fontsize=12, pad=15)
        ax.set_xticks([])
        if k:
            ax.set_ylabel("")
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=m[2]) for m in METHODS],
               [m[0] for m in METHODS], loc="lower center", bbox_to_anchor=(.52, .13),
               ncol=5, frameon=False, fontsize=10)
    fig.text(.075, .075, "Selected H1 BC clears all original gates. Higher strong-push scores alone do not determine selection.",
             fontsize=10, color="#526776")
    fig.text(.075, .035, "Recorded proportions; development sources were reused. Counts are not independent repeated-seed estimates.",
             fontsize=9, color="#526776")
    save(fig, "controller_comparison")

    fig, ax = plt.subplots(figsize=(11, 5.8))
    fig.subplots_adjust(left=.10, right=.96, top=.76, bottom=.21)
    heading(fig, "Where H1 BC changes the outcome",
            "Selected H1 BC minus same-reference baseline  /  same 40 cases per strength")
    delta = rates[2] - rates[0]
    bars = ax.bar(np.arange(3), delta, width=.5, color=["#ba6547", "#758697", "#087f89"])
    style(ax, percent=False)
    ax.set_ylim(-7, 22)
    ax.set_yticks([-5, 0, 5, 10, 15, 20])
    ax.axhline(0, color="#526776", lw=1)
    ax.set_ylabel("Change in success rate (percentage points)")
    ax.set_xticks(np.arange(3), [s.replace("\n", " / ") for s in LABELS])
    for i, (bar, val) in enumerate(zip(bars, delta)):
        count = data[2]["summary"][STRENGTHS[i]]["success"] - data[0]["summary"][STRENGTHS[i]]["success"]
        ax.text(i, val+1 if val >= 0 else val-1,
                f"{val:+g} pp\n{count:+d} net successes / 40",
                va="bottom" if val >= 0 else "top", ha="center", fontsize=12, fontweight="bold")
    fig.text(.10, .075, "Strong push: 52.5% → 67.5% (+15 pp). Medium push is unchanged; no-push success drops by 2.5 pp.",
             fontsize=10, color="#526776")
    fig.text(.10, .035, "Net change does not show every paired transition. See RESULTS.md for gains, regressions and failure flags.",
             fontsize=9, color="#526776")
    save(fig, "baseline_gain")

    fig, ax = plt.subplots(figsize=(12, 5.8))
    fig.subplots_adjust(left=.085, right=.98, top=.73, bottom=.23)
    heading(fig, "Strong-push recovery, source by source",
            "0.75 BW  /  8 recorded cases per motion source  /  all five development sources")
    sources = list(data[2]["per_source"])
    x = np.arange(len(sources))
    for j, idx in enumerate([0, 2]):
        counts = [data[idx]["per_source"][s]["0.75"]["success"] for s in sources]
        bars = ax.bar(x+(j-.5)*.33, np.array(counts)*100/8, .31,
                      color=METHODS[idx][2], label=METHODS[idx][0])
        for bar, n in zip(bars, counts):
            ax.text(bar.get_x()+bar.get_width()/2, n*100/8+2, f"{n}/8",
                    ha="center", fontsize=11, fontweight="bold")
    style(ax)
    ax.set_xticks(x, sources)
    ax.set_xlabel("Motion source ID", labelpad=12)
    fig.legend(loc="upper right", bbox_to_anchor=(.98, .855), ncol=2, frameon=False, fontsize=10)
    fig.text(.085, .075, "H1 BC improves 4 of 5 source-level strong-push totals and ties on source 015317.",
             fontsize=10, color="#526776")
    fig.text(.085, .035, "Small source-level denominators are shown explicitly. These are development results, not unseen-room tests.",
             fontsize=9, color="#526776")
    save(fig, "source_breakdown")
    print("Generated 3 figures in SVG + PNG, and results.csv from frozen JSON.")


if __name__ == "__main__":
    main()
