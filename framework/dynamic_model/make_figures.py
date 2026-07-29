"""Poster figures. Reads the sweep + schedule and emits figs/*.pdf."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figs", exist_ok=True)
plt.rcParams.update({"font.size": 13, "axes.linewidth": 1.1,
                     "font.family": "sans-serif", "pdf.fonttype": 42})

SWEEP = {
    2: dict(p=7.90e-8,  scalar=5.037e4, cls=2.023e6, tsc=3.075e5, tcl=2.731e6),
    3: dict(p=1.97e-5,  scalar=146.3,   cls=1325.0,  tsc=1121.0,  tcl=5668.0),
    4: dict(p=2.09e-4,  scalar=17.53,   cls=62.56,   tsc=133.9,   tcl=420.6),
    6: dict(p=2.08e-3,  scalar=4.454,   cls=7.350,   tsc=15.70,   tcl=29.36),
    8: dict(p=6.74e-3,  scalar=2.663,   cls=3.455,   tsc=6.042,   tcl=8.810),
}
SCHED = np.array([[9.982, 31.043], [2.897, 7.543], [1.947, 3.906],
                  [1.611, 2.711], [1.454, 2.182], [1.381, 1.954],
                  [1.387, 1.960], [1.602, 2.705]])

Ts = sorted(SWEEP)
hrs = [t * 0.25 for t in Ts]
C = {"scalar": "#B0B0B0", "cls": "#4C9F70", "tsc": "#E07A2F", "tcl": "#2E5E9E"}

# ---------------- panel A: collapse with mission length ----------------
fig, ax = plt.subplots(figsize=(7.4, 5.6))
series = [("scalar", "scalar  (1 param)", "o", "--"),
          ("cls",    "+ component  (2)",  "s", "-"),
          ("tsc",    "+ time  (8)",       "^", "-"),
          ("tcl",    "+ both  (16)",      "D", "-")]
for k, lab, mk, ls in series:
    ax.plot(hrs, [SWEEP[t][k] for t in Ts], marker=mk, ls=ls, lw=2.6, ms=9,
            color=C[k], label=lab, zorder=3)
ax.set_yscale("log")
ax.axhline(1.0, color="k", lw=1.0, ls=":", zorder=1)
ax.text(0.55, 1.30, "naive Monte Carlo  (VRF $=1$)", fontsize=11, color="k")
ax.axhspan(3e6, 2e7, color="#2E5E9E", alpha=0.07, zorder=0)
ax.text(0.53, 7.0e6, r"$h$-transform: variance $=0$  (infinite VRF)",
        fontsize=12, color="#2E5E9E", weight="bold")
ax.axvline(0.875, color="#C0392B", lw=1.6, ls="--", zorder=2)
ax.text(0.90, 2.2e5, "crossover", color="#C0392B", fontsize=11.5, weight="bold")
ax.text(0.90, 6.0e4, "component $\\to$ time", color="#C0392B", fontsize=10.5)
ax.set_xlabel("mission length  (hours)")
ax.set_ylabel("exact variance-reduction ceiling")
ax.set_title("Every product-form ceiling collapses with mission length",
             fontsize=14.5, weight="bold", pad=12)
ax.set_ylim(1, 2e7)
ax.legend(loc="upper right", frameon=True, framealpha=0.95, fontsize=12,
          bbox_to_anchor=(0.995, 0.86))
ax.grid(alpha=0.25, which="both", ls=":")
sec = ax.secondary_xaxis("top", functions=(lambda x: x / 0.25, lambda x: x * 0.25))
sec.set_xlabel("mission length  $T$  (steps)", fontsize=12)
fig.tight_layout(); fig.savefig("figs/collapse.pdf"); plt.close(fig)

# ---------------- panel B: the optimal tilt schedule -------------------
fig, ax = plt.subplots(figsize=(7.4, 4.4))
steps = np.arange(len(SCHED))
ax.plot(steps, SCHED[:, 0], marker="o", lw=2.8, ms=9, color="#2E5E9E",
        label="8-GPU nodes")
ax.plot(steps, SCHED[:, 1], marker="s", lw=2.8, ms=9, color="#E07A2F",
        label="4-GPU nodes")
ax.axhline(1.7353, color="#B0B0B0", lw=2.4, ls="--",
           label="best time-homogeneous tilt")
ax.set_yscale("log")
ax.annotate("front-load: damage must\naccumulate within $T$",
            xy=(0.12, 22), xytext=(1.35, 17), fontsize=11,
            arrowprops=dict(arrowstyle="->", lw=1.3))
ax.annotate("last step must land in $F$", xy=(7, 2.75), xytext=(4.15, 5.4),
            fontsize=11, arrowprops=dict(arrowstyle="->", lw=1.3))
ax.set_xlabel("mission step")
ax.set_ylabel("optimal odds tilt  $\\lambda$")
ax.set_title("A flat line is what a time-homogeneous proposal is limited to",
             fontsize=13.5, weight="bold", pad=10)
ax.legend(frameon=True, fontsize=11.5, loc="upper right",
          bbox_to_anchor=(0.995, 0.99))
ax.grid(alpha=0.25, which="both", ls=":")
fig.tight_layout(); fig.savefig("figs/schedule.pdf"); plt.close(fig)
print("wrote figs/collapse.pdf and figs/schedule.pdf")
