#Analysis + plots for Problem 4.2
#Usage (repo root): python3 Python/analyze.py results/final_<jobid> [--img Images]
#Writes summary.csv, summary.md, perf.md into the run folder and the figures into Images/
import argparse
import csv
import math
import os
import statistics as st
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from theory import (A_REF, B, X0, X1, Y0, Y1, HeartArea, n_mixed, sigma_rel,
                    strata_fractions, subbox)

#Tasks from the assignment table
NS = [5760000, 57600000, 576000000]
PS = [16, 36, 64]
TASK = {(16, NS[0]): "A1", (16, NS[1]): "A2", (16, NS[2]): "A3",
        (36, NS[0]): "B1", (36, NS[1]): "B2", (36, NS[2]): "B3",
        (64, NS[0]): "C1", (64, NS[1]): "C2", (64, NS[2]): "C3"}
METHODS = ["m1", "m2v", "m2h", "m3"]
NAME = {"m1": "M1 plain", "m2v": "M2 vertical slices", "m2h": "M2 horizontal slices",
        "m3": "M3 cells"}
SHORT = {"m1": "M1", "m2v": "M2 vert", "m2h": "M2 horiz", "m3": "M3"}

#Colors: validated categorical slots 1-4 in fixed order, one per method everywhere
#Marker shapes double up on identity (slots 3-4 are under 3:1 contrast on white)
COLOR = {"m1": "#2a78d6", "m2v": "#eb6834", "m3": "#1baf7a", "m2h": "#eda100"}
MARK = {"m1": "o", "m2v": "s", "m3": "^", "m2h": "D"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", [SURF, "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
MEAN_ABS = math.sqrt(2.0 / math.pi) #E|Z| for a normal error


def style_axes(ax):
    ax.set_facecolor(SURF)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.grid(True, which="minor", color=GRID, linewidth=0.4, alpha=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    ax.title.set_color(INK)


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def heart_outline(ax):
    #Heart curve as the F = 0 contour
    xs = np.linspace(X0 - 0.02, X1 + 0.02, 900)
    ys = np.linspace(Y0 - 0.02, Y1 + 0.02, 900)
    gx, gy = np.meshgrid(xs, ys)
    F = (gx ** 2 + gy ** 2 - 1) ** 3 - gx ** 2 * gy ** 3
    ax.contour(gx, gy, F, levels=[0.0], colors=INK, linewidths=1.2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="results/final_<jobid> folder with results.csv + ranks.csv")
    ap.add_argument("--img", default="Images")
    args = ap.parse_args()
    os.makedirs(args.img, exist_ok=True)

    R = load(os.path.join(args.run, "results.csv"))
    g = defaultdict(list)
    for r in R:
        g[(int(r["p"]), int(r["N"]), r["method"])].append(r)

    #Exact theory for every stratum layout
    print("computing exact strata areas...")
    ha = HeartArea()
    F = {(m, p): strata_fractions(ha, m, p) for p in [1] + PS for m in METHODS
         if not (p == 1 and m != "m1")}
    q = A_REF / B

    #----- Summary table: 18 + 9 data points -------------------------------------------
    rows = []
    for p in PS:
        for N in NS:
            for m in METHODS:
                rs = g[(p, N, m)]
                err = [float(r["rel_err"]) for r in rs]
                z = [(float(r["A_hat"]) - A_REF) / (float(r["sig_rel"]) * A_REF) for r in rs]
                th = sigma_rel(F[(m, p)], p, N // p)
                rows.append(dict(
                    task=TASK[(p, N)], p=p, n=N // p, N=N, method=m, trials=len(rs),
                    mean_err=st.mean(err), sd_err=st.stdev(err),
                    pred_sig=st.mean(float(r["sig_rel"]) for r in rs),
                    theory_sig=th, expected_mean=th * MEAN_ABS,
                    ratio=st.mean(err) / (th * MEAN_ABS),
                    rms_z=math.sqrt(sum(x * x for x in z) / len(z)),
                    n_mixed=int(rs[0]["n_mixed"]), theory_mixed=n_mixed(F[(m, p)])))
    with open(os.path.join(args.run, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    S = {(r["p"], r["N"], r["method"]): r for r in rows}

    #Calibration over every trial (incl. p = 1)
    z_all = [(float(r["A_hat"]) - A_REF) / (float(r["sig_rel"]) * A_REF) for r in R]
    chi2 = sum(x * x for x in z_all) / len(z_all)

    #----- Q2 / Q3 / Q4 factors ---------------------------------------------------------
    #Error factor vs plain = sigma_M1 / sigma_Mx, three ways:
    #theory (exact), the runs' own sigma_hat, and measured RMS error (errors scaled by sqrt(N))
    def rms_scaled(p, m):
        v = [float(r["rel_err"]) ** 2 * int(r["N"]) for N in NS for r in g[(p, N, m)]]
        return math.sqrt(sum(v) / len(v)), len(v)

    fac = {}
    for p in PS:
        r1, k = rms_scaled(p, "m1")
        for m in METHODS:
            th = sigma_rel(F[("m1", p)], p, 1) / sigma_rel(F[(m, p)], p, 1)
            sh = st.mean(S[(p, N, "m1")]["pred_sig"] / S[(p, N, m)]["pred_sig"] for N in NS)
            rm, _ = rms_scaled(p, m)
            meas = r1 / rm
            unc = meas * math.sqrt(2.0 / (2.0 * k)) if m != "m1" else 0.0 #two RMS from k samples
            fac[(p, m)] = (th, sh, meas, unc)

    #Mean f(1 - f) per layout = variance relative to plain sampling (plain = q(1 - q))
    mf = {(m, p): float(np.mean(F[(m, p)] * (1 - F[(m, p)]))) for (m, p) in F}

    #----- Markdown tables --------------------------------------------------------------
    md = []
    md.append("## Mean ± std of relative error |Â − A| / A over 30 trials\n")
    md.append("| Task | p | n per core | N = p·n | " + " | ".join(NAME[m] for m in METHODS) + " |")
    md.append("|---|---|---|---|" + "---|" * len(METHODS))
    for p in PS:
        for N in NS:
            cells = ["%.2e ± %.2e" % (S[(p, N, m)]["mean_err"], S[(p, N, m)]["sd_err"]) for m in METHODS]
            md.append("| %s%s | %d | %s | %s | %s |" % (
                TASK[(p, N)], " (bonus)" if p == 64 else "", p, f"{N // p:,}", f"{N:,}", " | ".join(cells)))
    md.append("\n## Measured mean error vs theory (exact σ · √(2/π))\n")
    md.append("| Task | " + " | ".join(SHORT[m] for m in METHODS) + " |")
    md.append("|---|" + "---|" * len(METHODS))
    for p in PS:
        for N in NS:
            md.append("| %s | %s |" % (TASK[(p, N)], " | ".join(
                "%.2e / %.2e (%.2f)" % (S[(p, N, m)]["mean_err"], S[(p, N, m)]["expected_mean"],
                                        S[(p, N, m)]["ratio"]) for m in METHODS)))
    md.append("\nAll %d trials: mean z² = %.3f (1 if the predicted σ̂ is right)\n" % (len(z_all), chi2))
    md.append("## Error reduction factor σ_M1 / σ_method (theory / run σ̂ / measured RMS)\n")
    md.append("| p | " + " | ".join(SHORT[m] for m in METHODS[1:]) + " |")
    md.append("|---|" + "---|" * 3)
    for p in PS:
        md.append("| %d | %s |" % (p, " | ".join(
            "%.3f / %.3f / %.2f ± %.2f" % fac[(p, m)] for m in METHODS[1:])))
    md.append("\n## Strata cut by the boundary + mean f(1−f) (plain = q(1−q) = %.4f)\n" % (q * (1 - q)))
    md.append("| p | " + " | ".join(SHORT[m] for m in METHODS[1:]) + " |")
    md.append("|---|" + "---|" * 3)
    for p in PS:
        md.append("| %d | %s |" % (p, " | ".join(
            "%d / %d, %.4f" % (n_mixed(F[(m, p)]), p, mf[(m, p)]) for m in METHODS[1:])))
    with open(os.path.join(args.run, "summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))

    #----- Fig 1: error vs N (report item 1) --------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharey=True)
    fig.patch.set_facecolor(SURF)
    Nline = np.array([NS[0] / 1.6, NS[-1] * 1.6])
    for ax, p in zip(axes, PS):
        style_axes(ax)
        for m in ["m1", "m2v", "m3"]:
            th = sigma_rel(F[(m, p)], p, 1) * MEAN_ABS
            ax.plot(Nline, th * np.sqrt(p / Nline), color=COLOR[m], lw=1.2, ls="--", alpha=0.9)
            y = [S[(p, N, m)]["mean_err"] for N in NS]
            e = [S[(p, N, m)]["sd_err"] / math.sqrt(30) for N in NS]
            ax.errorbar(NS, y, yerr=e, color=COLOR[m], marker=MARK[m], ms=7, lw=2,
                        capsize=3, mec=SURF, mew=1.2, label=NAME[m] if m != "m2v" else "M2 slices (vertical)")
        ref = sigma_rel(F[("m1", p)], p, 1) * MEAN_ABS * 1.8
        ax.plot(Nline, ref * np.sqrt(p / Nline), color=INK2, lw=1, ls=":")
        ax.annotate("slope −½", (Nline[0] * 1.15, ref * math.sqrt(p / (Nline[0] * 1.15))),
                    xytext=(4, 4), textcoords="offset points", fontsize=9, color=INK2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(Nline[0], Nline[1])
        ax.set_xlabel("total samples N = p·n")
        ax.set_title("p = %d  (tasks %s–%s)" % (p, TASK[(p, NS[0])], TASK[(p, NS[-1])]), fontsize=11)
    axes[0].set_ylabel("mean relative error over 30 trials")
    axes[0].legend(loc="lower left", fontsize=9, frameon=False)
    fig.suptitle("Mean relative error vs total samples (points ± standard error, dashed = exact theory)",
                 color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(args.img, "fig1_error_vs_N.png"), dpi=150, facecolor=SURF)
    plt.close(fig)

    #----- Fig 2: strata maps, why slices differ + why cells win -------------------------
    layouts = [("m2v", 16), ("m2h", 16), ("m3", 16), ("m3", 64)]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.3))
    fig.patch.set_facecolor(SURF)
    for ax, (m, p) in zip(axes, layouts):
        ax.set_facecolor(SURF)
        f = F[(m, p)]
        for r in range(p):
            x0, x1, y0, y1 = subbox(m, r, p)
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=SEQ(f[r] * (1 - f[r]) / 0.25),
                                   edgecolor=SURF, linewidth=1.0))
        heart_outline(ax)
        ax.set_xlim(X0, X1)
        ax.set_ylim(Y0, Y1)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title("%s, p = %d\nmean f(1−f) = %.3f  (plain %.3f)" % (NAME[m], p, mf[(m, p)], q * (1 - q)),
                     fontsize=10, color=INK)
    sm = plt.cm.ScalarMappable(cmap=SEQ, norm=plt.Normalize(0, 0.25))
    cb = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.01)
    cb.set_label("stratum variance f(1−f)", color=INK2)
    cb.ax.tick_params(colors=INK2, labelsize=8)
    fig.suptitle("Where the variance comes from: only strata cut by the heart's boundary contribute",
                 color=INK, fontsize=12)
    fig.savefig(os.path.join(args.img, "fig2_strata.png"), dpi=150, facecolor=SURF, bbox_inches="tight")
    plt.close(fig)

    #----- Fig 3: factor vs p + boundary strata (Q2, Q4) ---------------------------------
    print("computing theory curves over p...")
    p_sl = list(range(4, 65, 4))
    p_sq = [k * k for k in range(2, 9)]
    curve = {}
    for m, plist in (("m2v", p_sl), ("m2h", p_sl), ("m3", p_sq)):
        curve[m] = [math.sqrt(q * (1 - q) / float(np.mean(f * (1 - f))))
                    for f in (strata_fractions(ha, m, pp) for pp in plist)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    fig.patch.set_facecolor(SURF)
    style_axes(ax1)
    style_axes(ax2)
    for m, plist in (("m2v", p_sl), ("m2h", p_sl), ("m3", p_sq)):
        ax1.plot(plist, curve[m], color=COLOR[m], lw=1.2, ls="--")
        meas = [fac[(p, m)][2] for p in PS]
        unc = [fac[(p, m)][3] for p in PS]
        ax1.errorbar(PS, meas, yerr=unc, color=COLOR[m], marker=MARK[m], ms=7, lw=0, elinewidth=1.5,
                     capsize=3, mec=SURF, mew=1.2, label=NAME[m])
        ax1.annotate(SHORT[m], (plist[-1], curve[m][-1]), xytext=(6, 0), textcoords="offset points",
                     va="center", fontsize=9, color=INK2)
    ax1.axhline(1.0, color=INK2, lw=1, ls=":")
    ax1.set_xlabel("p (number of strata / cores)")
    ax1.set_ylabel("error reduction factor  σ_M1 / σ")
    ax1.set_title("Error reduction vs plain (points: measured, dashed: exact theory)", fontsize=10)
    ax1.legend(loc="upper left", fontsize=9, frameon=False)
    ax1.set_xlim(0, 72)
    pp = np.array(p_sq)
    ax2.plot(pp, pp, color=COLOR["m2v"], marker=MARK["m2v"], ms=7, lw=2, mec=SURF, label="slices (all p cut)")
    ax2.plot(pp, [n_mixed(strata_fractions(ha, "m3", k)) for k in p_sq], color=COLOR["m3"], marker=MARK["m3"],
             ms=7, lw=2, mec=SURF, label="cells (theory, runs match at p = 16, 36, 64)")
    ax2.plot(pp, 4 * (np.sqrt(pp) - 1), color=INK2, lw=1, ls=":", label="4(√p − 1), exact for even √p")
    ax2.set_xlabel("p (perfect squares 4 to 64)")
    ax2.set_ylabel("strata cut by the boundary")
    ax2.set_title("Strata that add variance", fontsize=10)
    ax2.legend(loc="upper left", fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(args.img, "fig3_factor_vs_p.png"), dpi=150, facecolor=SURF)
    plt.close(fig)

    #----- Fig 4: calibration ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 4))
    fig.patch.set_facecolor(SURF)
    style_axes(ax)
    bins = np.linspace(-4, 4, 33)
    ax.hist(z_all, bins=bins, color=COLOR["m1"], edgecolor=SURF, linewidth=1, density=True,
            label="all %d trials" % len(z_all))
    zz = np.linspace(-4, 4, 300)
    ax.plot(zz, np.exp(-zz ** 2 / 2) / math.sqrt(2 * math.pi), color=INK, lw=1.5, label="N(0, 1)")
    ax.set_xlabel(r"$z = (\hat{A} - A) / \hat{\sigma}$   ($\hat{\sigma}$ from each run's own counts)")
    ax.set_ylabel("density")
    ax.set_title("Predicted error is calibrated: mean z² = %.3f" % chi2, fontsize=10)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(args.img, "fig4_calibration.png"), dpi=150, facecolor=SURF)
    plt.close(fig)

    #----- Performance -------------------------------------------------------------------
    K = load(os.path.join(args.run, "ranks.csv"))
    tr = defaultdict(list)
    for r in K:
        tr[(int(r["p"]), int(r["n"]), r["method"], int(r["trial"]))].append(r)
    #Clock of the slowest rank in each trial, it sets T
    clk = defaultdict(list)
    cyc = defaultdict(list)
    for (p, n, m, t), rs in tr.items():
        slow = max(rs, key=lambda r: float(r["t_comp"]))
        clk[(p, p * n)].append(int(slow["mhz"]))
        cyc[(p, p * n)].extend(float(r["t_comp"]) / n * int(r["mhz"]) * 1e6 for r in rs if int(r["mhz"]) > 0)

    T = {}
    TC = {}
    for (p, N, m), rs in g.items():
        T.setdefault((p, N), []).extend(float(r["t_total"]) for r in rs)
        TC.setdefault((p, N), []).extend(float(r["t_gather"]) for r in rs)
    T = {k: st.median(v) for k, v in T.items()}
    TC = {k: st.median(v) for k, v in TC.items()}
    f1 = st.median(clk[(1, NS[-1])])
    t0 = T[(1, NS[-1])] / NS[-1]

    pm = []
    pm.append("## Timing (median over every trial, all methods)\n")
    pm.append("t0 = %.3f ns per point on one core (%.0f MHz), %.2f cycles per point, %.2f GFLOP/s at 15 flops/point\n"
              % (t0 * 1e9, f1, st.median(cyc[(1, NS[-1])]), 15 / (t0 * 1e9)))
    pm.append("| p | N | T_p (ms) | speedup T1/Tp | efficiency | slowest-rank clock (MHz) | clock limit f_p/f_1 | t_c gather (µs) | overhead h = (P/N)(t_c/t0) | model (N/P)t0 + t_c (ms) |")
    pm.append("|---|---|---|---|---|---|---|---|---|---|")
    perf = {}
    for p in [1] + PS:
        for N in NS:
            Tp = T[(p, N)]
            S_ = T[(1, N)] / Tp
            E = S_ / p
            fp = st.median(clk[(p, N)])
            h = (p / N) * (TC[(p, N)] / t0)
            model = (N / p) * t0 + TC[(p, N)]
            perf[(p, N)] = (Tp, S_, E, fp / f1, h)
            pm.append("| %d | %s | %.3f | %.1f | %.3f | %.0f | %.3f | %.1f | %.4f | %.3f |" % (
                p, f"{N:.2e}", Tp * 1e3, S_, E, fp, fp / f1, TC[(p, N)] * 1e6, h, model * 1e3))
    with open(os.path.join(args.run, "perf.md"), "w") as f:
        f.write("\n".join(pm) + "\n")
    print("\n".join(pm))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    fig.patch.set_facecolor(SURF)
    style_axes(ax1)
    style_axes(ax2)
    pl = [1] + PS
    ncol = {NS[0]: "#86b6ef", NS[1]: "#3987e5", NS[2]: "#184f95"} #Ordinal: one hue, light -> dark
    for N in NS:
        ax1.plot(pl, [perf[(p, N)][1] for p in pl], color=ncol[N], marker="o", ms=7, lw=2, mec=SURF,
                 label="N = %.2e" % N)
        ax2.plot(pl, [perf[(p, N)][2] for p in pl], color=ncol[N], marker="o", ms=7, lw=2, mec=SURF,
                 label="N = %.2e" % N)
    ax1.plot([1, 64], [1, 64], color=INK2, lw=1, ls=":", label="ideal")
    ax2.plot(pl, [st.median(perf[(p, N)][3] for N in NS) for p in pl], color=INK2, lw=1.5, ls="--",
             marker="x", ms=7, label="clock limit f_p / f_1")
    ax1.set_xlabel("p")
    ax1.set_ylabel("speedup T1 / Tp")
    ax1.set_title("Strong scaling (fixed N)", fontsize=10)
    ax1.legend(fontsize=9, frameon=False, loc="upper left")
    ax2.set_xlabel("p")
    ax2.set_ylabel("parallel efficiency")
    ax2.set_ylim(0, 1.08)
    ax2.set_title("Efficiency vs the turbo clock limit", fontsize=10)
    ax2.legend(fontsize=9, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(args.img, "fig5_performance.png"), dpi=150, facecolor=SURF)
    plt.close(fig)
    print("\nfigures in", args.img)


if __name__ == "__main__":
    main()