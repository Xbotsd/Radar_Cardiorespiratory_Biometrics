"""Generate every manuscript figure from the committed result tables.

Outputs vector PDF (preferred by the journal) and 300 dpi PNG previews in
``figures/``.  Figure widths follow the journal: 85 mm single column,
170 mm double column; fonts stay between 8 and 12 pt at final size.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

MM = 1 / 25.4
W1, W2 = 85 * MM, 170 * MM

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.0,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
})
C_DOG, C_HUMAN, C_GRID, C_INV = "#4C72B0", "#DD8452", "#C44E52", "#55A868"


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=300)
    plt.close(fig)
    print("wrote", name)


def jload(name):
    with open(RES / name) as f:
        return json.load(f)


def panel(ax, letter, title=None):
    """Put the panel letter in front of the title so it can never overlap data."""
    base = title if title is not None else ax.get_title()
    if base:
        ax.set_title(f"({letter}) {base}", loc="left", fontsize=7.2)


# ---------------------------------------------------------------- figure 1 ---
def fig_protocol():
    S = jload("identification_summary.json")["closed_set"]
    labels = ["leaky\n(random window)", "temporal\n(guard gap)", "temporal, 25 min\napart"]
    keys = [("P0", "resp+cardiac", "RF", "mean"), ("P1", "resp+cardiac", "RF", None),
            ("P2", "resp+cardiac", "RF", None)]
    vals, errs = [], []
    p0 = S["P0"]["resp+cardiac"]["RF"]
    vals.append(100 * p0["mean"]); errs.append(100 * p0["sd"])
    vals.append(100 * S["P1"]["resp+cardiac"]["RF"]); errs.append(0)
    vals.append(100 * S["P2"]["resp+cardiac"]["RF"]); errs.append(0)

    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.35))
    ax = axes[0]
    b = ax.bar(range(3), vals, yerr=errs, capsize=2.5, color=[C_GRID, C_DOG, C_HUMAN],
               width=0.62, edgecolor="black", linewidth=0.4)
    for r, v in zip(b, vals):
        ax.text(r.get_x() + r.get_width() / 2, v + 1.6, f"{v:.1f}", ha="center", fontsize=7)
    ax.axhline(100 / 30, color="black", ls=":", lw=0.8)
    ax.text(-0.42, 100 / 30 + 1.6, "chance 3.3 %", ha="left", fontsize=6.5)
    ax.set_xticks(range(3)); ax.set_xticklabels(labels)
    ax.set_ylabel("30-class accuracy (%)")
    ax.set_ylim(0, 68)
    panel(ax, "a", "closed-set identification, same 30 dogs")

    ax = axes[1]
    gaps = sorted(int(k) for k in S["P1_gap_sweep"])
    gv = [100 * S["P1_gap_sweep"][str(g) if str(g) in S["P1_gap_sweep"] else g] for g in gaps]
    ax.plot(gaps, gv, "o-", color=C_DOG, ms=3.5)
    ax.axhline(100 * p0["mean"], color=C_GRID, ls="--", lw=0.9)
    ax.text(gaps[-1], 100 * p0["mean"] - 1.6, "leaky split", ha="right", fontsize=6.5,
            color=C_GRID)
    ax.axhline(100 / 30, color="black", ls=":", lw=0.8)
    ax.set_xlabel("guard gap (windows of 3 s)")
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(0, 60)
    panel(ax, "b", "sensitivity to the guard gap")
    fig.tight_layout(w_pad=2.2)
    save(fig, "fig1_protocol")


# ---------------------------------------------------------------- figure 2 ---
def fig_vitals():
    dog = pd.read_csv(RES / "vitals_dogs.csv")
    vs = pd.read_csv(RES / "vitals_vs120.csv")
    p60 = pd.read_csv(RES / "vitals_p60.csv")
    specs = [
        ("dogs, 7.3 GHz UWB", dog[(dog.level == "recording") & (dog.variant == "full")], "hr_ref", "hr_est", "HR (bpm)", C_DOG),
        ("dogs, breathing rate", dog[(dog.level == "recording") & (dog.variant == "full")], "br_ref", "br_est", "BR (bpm)", C_DOG),
        ("humans, 120 GHz FMCW", vs[vs.level == "recording"], "hr_ref", "hr_est", "HR (bpm)", C_HUMAN),
        ("humans, 60 GHz FMCW", p60[(p60.variant == "full") & p60.condition.str.endswith("/Rest")],
         "hr_ref", "hr_est", "HR (bpm)", C_HUMAN),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(W2, 2.15))
    for ax, (title, df, rk, ek, lab, col) in zip(axes, specs):
        x, y = df[rk].values, df[ek].values
        d = y - x
        lo, hi = np.percentile(np.r_[x, y], [1, 99])
        pad = 0.08 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=0.7)
        ax.scatter(x, y, s=11, color=col, alpha=0.8, edgecolor="none")
        mae = np.abs(d).mean()
        bias, sd = d.mean(), d.std(ddof=1)
        ax.text(0.97, 0.04,
                f"MAE {mae:.2f}\nbias {bias:+.2f}\nLoA {bias - 1.96 * sd:.1f} to {bias + 1.96 * sd:.1f}",
                transform=ax.transAxes, va="bottom", ha="right", fontsize=6.2,
                bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.5))
        ax.set_xlabel(f"reference {lab}")
        ax.set_ylabel(f"radar {lab}")
        ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
        ax.tick_params(labelsize=6.5)
    for ax, l, (title, *_rest) in zip(axes, "abcd", specs):
        panel(ax, l, title)
    fig.tight_layout(w_pad=1.6)
    save(fig, "fig2_vital_signs")


# ---------------------------------------------------------------- figure 3 ---
def fig_open_set():
    S = jload("identification_summary.json")["open_set"]
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.3))
    ax = axes[0]
    fixed = S["fixed_split_anaesth_known"]
    rep = S["random_partitions"]
    names = ["Mahalanobis", "Euclidean", "MaxProb", "Top2Margin"]
    x = np.arange(len(names)); w = 0.2
    ax.bar(x - 0.3, [100 * rep[n]["accept_known"]["mean"] for n in names], w,
           yerr=[100 * rep[n]["accept_known"]["sd"] for n in names], capsize=2,
           label="known accepted (20 partitions)", color=C_DOG, edgecolor="black", linewidth=0.4)
    ax.bar(x - 0.1, [100 * rep[n]["reject_unknown"]["mean"] for n in names], w,
           yerr=[100 * rep[n]["reject_unknown"]["sd"] for n in names], capsize=2,
           label="unknown rejected (20 partitions)", color=C_HUMAN, edgecolor="black", linewidth=0.4)
    ax.bar(x + 0.1, [100 * fixed[n]["accept_known"] for n in names], w,
           label="known accepted (fixed split)", color=C_DOG, alpha=0.35,
           edgecolor="black", linewidth=0.4, hatch="///")
    ax.bar(x + 0.3, [100 * fixed[n]["reject_unknown"] for n in names], w,
           label="unknown rejected (fixed split)", color=C_HUMAN, alpha=0.35,
           edgecolor="black", linewidth=0.4, hatch="///")
    ax.set_xticks(x)
    ax.set_xticklabels(["Mahalanobis", "Euclidean", "MaxProb", "Top-2"],
                       fontsize=6.5, rotation=20, ha="right",
                       rotation_mode="anchor")
    ax.set_ylabel("rate (%)"); ax.set_ylim(0, 104)
    ax.legend(fontsize=5.6, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.20),
              frameon=False, handlelength=1.2, columnspacing=0.8)
    panel(ax, "a", "open-set operating point")

    ax = axes[1]
    aurocs = [rep[n]["auroc"]["mean"] for n in names]
    errs = [rep[n]["auroc"]["sd"] for n in names]
    ax.barh(range(len(names)), aurocs, xerr=errs, capsize=2, color=C_GRID,
            edgecolor="black", linewidth=0.4)
    ax.axvline(0.5, color="black", ls=":", lw=0.9)
    ax.set_ylim(-0.6, 3.75)
    ax.text(0.515, 3.65, "chance", fontsize=6.5, va="top")
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=6.8)
    ax.set_xlabel("AUROC, known vs. unknown")
    ax.set_xlim(0, 1)
    panel(ax, "b", "AUROC, 20 random partitions")
    fig.tight_layout(w_pad=2.4)
    save(fig, "fig3_open_set")


# ---------------------------------------------------------------- figure 4 ---
def fig_species():
    S = jload("cross_species_summary.json")
    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.25))
    ax = axes[0]
    rows = [("species\ndog vs. human", S["B_all110_grouped"]["RF"]),
            ("hardware\nhumans 60 vs. 120 GHz", S["NC1_human60_vs_human120"]["inv16"]),
            ("session state\ndog anaesth. vs. awake", S["NC2_dog_scen1_vs_scen2"]["inv16"]),
            ("physiology\nhuman rest vs. post-ex.", S["NC3_p60_rest_vs_post"]["inv16"])]
    vals = [100 * r[1]["balanced_acc"] for r in rows]
    cols = [C_DOG, C_GRID, C_GRID, C_HUMAN]
    b = ax.bar(range(len(rows)), vals, color=cols, width=0.62, edgecolor="black", linewidth=0.4)
    for r, v in zip(b, vals):
        ax.text(r.get_x() + r.get_width() / 2, v + 1.5, f"{v:.1f}", ha="center", fontsize=6.8)
    ax.axhline(50, color="black", ls=":", lw=0.8)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["dog vs. human", "60 vs. 120 GHz", "anaesth. vs. awake",
                        "rest vs. post-ex."], fontsize=6.2, rotation=90, va="top")
    ax.set_ylabel("balanced accuracy (%)"); ax.set_ylim(0, 108)
    panel(ax, "a", "contrast type")

    ax = axes[1]
    RM = jload("rate_matched_age_summary.json"); rm = RM["rate_matched"]
    labels = ["16\ninvariant", "5 legacy\ngrid-dep.", "21\ncombined"]
    keys = ["inv16", "grid5_only", "inv16+grid5"]
    x = np.arange(3); w = 0.36
    ax.bar(x - w / 2, [100 * rm["dog_native_50Hz"][k]["balanced_acc"] for k in keys], w,
           label="dog at native 50 Hz", color=C_DOG, edgecolor="black", linewidth=0.4)
    ax.bar(x + w / 2, [100 * rm["dog_decimated_10Hz"][k]["balanced_acc"] for k in keys], w,
           label="dog decimated to 10 Hz", color=C_HUMAN, edgecolor="black", linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylabel("balanced accuracy (%)"); ax.set_ylim(0, 108)
    ax.legend(fontsize=6.0, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.30),
              handlelength=1.2)
    panel(ax, "b", "rate-matched control (dog vs. human)")

    ax = axes[2]
    drift = RM["drift_50_to_10Hz"]
    names = ["spectral_centroid", "lf_hf_ratio", "br_spectral_flatness",
             "br_skew", "hr_power_frac", "br_freq_hz", "hrv_rmssd_s"]
    pretty = ["centroid", "LF/HF ratio", "BR flatness", "BR skew",
              "HR power frac.", "BR rate", "HRV RMSSD"]
    vals = [100 * drift[n]["median_rel_change"] for n in names]
    cols = [C_GRID] * 3 + [C_INV] * 4
    ax.barh(range(len(names)), vals, color=cols, edgecolor="black", linewidth=0.4)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(pretty, fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel("median relative change, 50 → 10 Hz (%)")
    panel(ax, "c", "descriptor drift")
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig4_species")


# ---------------------------------------------------------------- figure 5 ---
def fig_effect():
    S = jload("cross_species_summary.json")
    sp = S["G_effect_size_dog_minus_human60"]
    hw = S["G_effect_size_human60_minus_human120"]
    order = sorted(sp, key=lambda k: -abs(sp[k]))
    fig, ax = plt.subplots(figsize=(W2 * 0.72, 2.5))
    y = np.arange(len(order))
    ax.scatter([sp[k] for k in order], y, s=16, label="dog vs. human (species)", color=C_DOG,
               zorder=3)
    ax.scatter([hw[k] for k in order], y, s=16, marker="s",
               label="human 60 GHz vs. human 120 GHz (cross-platform)", color=C_GRID, zorder=3)
    for i, k in enumerate(order):
        ax.plot([hw[k], sp[k]], [i, i], color="grey", lw=0.6, zorder=1)
    ax.axvline(0, color="black", lw=0.6)
    ax.axvspan(-0.2, 0.2, color="grey", alpha=0.12, lw=0)
    ax.set_yticks(y); ax.set_yticklabels(order, fontsize=6.8)
    ax.invert_yaxis()
    ax.set_xlabel("standardised mean difference (Cohen's $d$)")
    ax.legend(fontsize=6.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.26),
              ncol=2, handletextpad=0.4, columnspacing=1.2)
    ax.set_title("species contrast (circles) vs. cross-platform contrast (squares)", fontsize=7)
    fig.tight_layout()
    save(fig, "fig5_effect_sizes")


# ---------------------------------------------------------------- figure 6 ---
def fig_age():
    A = jload("rate_matched_age_summary.json")["age"]
    fig, axes = plt.subplots(1, 2, figsize=(W2 * 0.72, 2.2))
    ax = axes[0]
    bars = {"RF": A["RF"]["mae"], "Ridge": A["Ridge"]["mae"], "mean predictor": A["Mean"]["mae"]}
    b = ax.bar(range(3), list(bars.values()), color=[C_DOG, C_HUMAN, "grey"], width=0.6,
               edgecolor="black", linewidth=0.4)
    for r, v in zip(b, bars.values()):
        ax.text(r.get_x() + r.get_width() / 2, v + 0.25, f"{v:.1f}", ha="center", fontsize=6.8)
    ax.set_xticks(range(3)); ax.set_xticklabels(list(bars), fontsize=6.8)
    ax.set_ylabel("age MAE (years)"); ax.set_ylim(0, 18)
    panel(ax, "a", f"110 subjects, perm. p = {A['ridge_perm_p']:.2f}")
    ax = axes[1]
    vals = [100 * A["RF"]["r2"], 100 * A["Ridge"]["r2"], 100 * A["Mean"]["r2"]]
    ax.bar(range(3), vals, color=[C_DOG, C_HUMAN, "grey"], width=0.6, edgecolor="black",
           linewidth=0.4)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(range(3)); ax.set_xticklabels(["RF", "Ridge", "mean"], fontsize=6.8)
    ax.set_ylabel("$R^2$ (%)")
    panel(ax, "b", "no model beats the mean")
    fig.tight_layout(w_pad=2.5)
    save(fig, "fig6_age")


def fig_pipeline():
    """Schematic of the measurement chain and the three feature layers."""
    fig, ax = plt.subplots(figsize=(W2, 1.9))
    ax.axis("off")
    rows = [("7.3 GHz UWB\n30 dogs · 50 Hz", C_DOG),
            ("60 GHz FMCW\n110 adults · 10 Hz", C_HUMAN),
            ("120 GHz FMCW\n24 adults · 333 Hz", C_HUMAN)]
    for (txt, col), y in zip(rows, [0.78, 0.47, 0.16]):
        ax.add_patch(plt.Rectangle((0.005, y - 0.10), 0.185, 0.20, fc=col, alpha=0.18,
                                   ec=col, lw=0.8))
        ax.text(0.0975, y, txt, ha="center", va="center", fontsize=6.4, linespacing=1.35)
        ax.annotate("", xy=(0.268, 0.5), xytext=(0.192, y),
                    arrowprops=dict(arrowstyle="-|>", lw=0.6, color="grey",
                                    connectionstyle="arc3,rad=0.08"))
    stages = [(0.278, 0.155, "range bin\nor phase", C_INV),
              (0.455, 0.165, "band separation\n+ resampling", C_INV),
              (0.642, 0.150, "feature layers\n19 / 16 / 5", C_HUMAN),
              (0.814, 0.130, "evaluation", C_GRID)]
    for x0, w, t, col in stages:
        ax.add_patch(plt.Rectangle((x0, 0.36), w, 0.28, fc=col, alpha=0.18, ec=col, lw=0.8))
        ax.text(x0 + w / 2, 0.5, t, ha="center", va="center", fontsize=6.4, linespacing=1.35)
    for a, b in [(0.433, 0.455), (0.620, 0.642), (0.792, 0.814)]:
        ax.annotate("", xy=(b, 0.5), xytext=(a, 0.5),
                    arrowprops=dict(arrowstyle="-|>", lw=0.7, color="black"))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    save(fig, "fig7_pipeline")


if __name__ == "__main__":
    fig_protocol()
    fig_vitals()
    fig_open_set()
    fig_species()
    fig_effect()
    fig_age()
    fig_pipeline()
