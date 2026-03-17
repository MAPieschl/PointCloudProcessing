"""
Registration Algorithm Evaluation Framework
============================================
Compares GICP, Semantic-GICP, NDT, Semantic-NDT on:

PART 1 — Proving superiority
  - Empirical CDF with KS dominance test (omnibus)
  - Pairwise Wilcoxon on median (location)
  - Pairwise Brown-Forsythe + Wilcoxon on MAD (scale)
  - Holm-Bonferroni correction, effect sizes throughout

PART 2 — Customer performance specification
  - Bootstrapped confidence intervals on key percentiles
  - Performance envelope table (50th, 75th, 90th, 95th)
  - Summary figure suitable for a technical report

Inputs
------
  R_gt   : (N, 3, 3)    ground-truth rotation matrices
  t_gt   : (N, 3)       ground-truth translation vectors
  R_est  : (4, N, 3, 3) estimated rotations  [GICP, Sem-GICP, NDT, Sem-NDT]
  t_est  : (4, N, 3)    estimated translations
"""

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, ks_2samp
from scipy.stats import fligner
from itertools import combinations, permutations
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

from pose_estimation import (
    geodesic_rotation_error,
    l2_translation_error,
    generate_synthetic_data,
    holm_bonferroni,
)

METHOD_NAMES = ["GICP", "Sem-GICP", "NDT", "Sem-NDT"]
COLOURS      = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

# ──────────────────────────────────────────────────────────────────────────────
# PART 1 — PROVING SUPERIORITY
# ──────────────────────────────────────────────────────────────────────────────

# ── 1a. Descriptive statistics ────────────────────────────────────────────────

def descriptives(errors: np.ndarray, method_names: list[str]) -> pd.DataFrame:
    """
    Median, MAD, IQR, and key percentiles per method.

    MAD = median(|e - median(e)|) — the most robust spread estimator,
    resistant to up to 50% of samples being outliers.
    """
    rows = []
    for i, name in enumerate(method_names):
        e   = errors[i]
        med = np.median(e)
        mad = np.median(np.abs(e - med))
        q25, q75, q90, q95 = np.percentile(e, [25, 75, 90, 95])
        rows.append({
            "Method" : name,
            "Median" : med,
            "MAD"    : mad,
            "IQR"    : q75 - q25,
            "P90"    : q90,
            "P95"    : q95,
            "Max"    : e.max(),
        })
    return pd.DataFrame(rows).set_index("Method")


# ── 1b. KS omnibus — distributional dominance ────────────────────────────────

def ks_omnibus(errors: np.ndarray, method_names: list[str],
               alpha: float = 0.025) -> pd.DataFrame:
    """
    Pairwise two-sample Kolmogorov-Smirnov tests with Holm-Bonferroni.

    The KS statistic is the maximum absolute difference between two empirical
    CDFs — significant result means the distributions differ somewhere.
    Combined with CDF plots this identifies stochastic dominance.
    """
    pairs  = list(combinations(range(len(method_names)), 2))
    raw_p, stats, directions = [], [], []

    for i, j in pairs:
        stat, p = ks_2samp(errors[i], errors[j], alternative="two-sided")
        raw_p.append(p)
        stats.append(stat)
        # Direction: which method has the lower median?
        better = method_names[i] if np.median(errors[i]) < np.median(errors[j]) \
                 else method_names[j]
        directions.append(better)

    adj_p = holm_bonferroni(raw_p)
    rows  = []
    for k, (i, j) in enumerate(pairs):
        rows.append({
            "Method A"    : method_names[i],
            "Method B"    : method_names[j],
            "KS stat"     : stats[k],
            "p (raw)"     : raw_p[k],
            "p (adjusted)": adj_p[k],
            "Significant" : adj_p[k] <= alpha,
            "Lower errors": directions[k],
        })
    return pd.DataFrame(rows)


# ── 1c. Pairwise Wilcoxon on median (location comparison) ────────────────────

def rank_biserial(w: float, n: int) -> float:
    return 1.0 - (2.0 * w) / (n * (n + 1) / 2.0)

def effect_label_r(r: float) -> str:
    a = abs(r)
    return "large" if a >= 0.5 else "medium" if a >= 0.3 else \
           "small" if a >= 0.1 else "negligible"

def pairwise_location(errors: np.ndarray, method_names: list[str],
                      alpha: float = 0.025) -> pd.DataFrame:
    """Pairwise Wilcoxon signed-rank on raw errors (tests location/median)."""
    pairs = list(combinations(range(len(method_names)), 2))
    raw_p, stats, effects = [], [], []

    for i, j in pairs:
        diff = errors[i] - errors[j]
        if np.all(diff == 0):
            stats.append(0.0); raw_p.append(1.0); effects.append(0.0)
        else:
            w, p = wilcoxon(diff, alternative="two-sided")
            stats.append(w); raw_p.append(p)
            effects.append(rank_biserial(w, len(diff)))

    adj_p = holm_bonferroni(raw_p)
    rows  = []
    for k, (i, j) in enumerate(pairs):
        better = method_names[i] \
                 if np.median(errors[i]) < np.median(errors[j]) \
                 else method_names[j]
        rows.append({
            "Method A"    : method_names[i],
            "Method B"    : method_names[j],
            "p (raw)"     : raw_p[k],
            "p (adjusted)": adj_p[k],
            "Significant" : adj_p[k] <= alpha,
            "Lower median": better,
            "Effect r"    : effects[k],
            "Effect size" : effect_label_r(effects[k]),
        })
    return pd.DataFrame(rows)


# ── 1d. Pairwise Brown-Forsythe + Wilcoxon on MAD (scale comparison) ─────────

def variance_ratio_label(vr: float) -> str:
    r = max(vr, 1.0 / vr)
    return "large" if r >= 4.0 else "medium" if r >= 2.0 else \
           "small" if r >= 1.25 else "negligible"

def pairwise_scale(errors: np.ndarray, method_names: list[str],
                   alpha: float = 0.025) -> pd.DataFrame:
    """
    Pairwise Brown-Forsythe + Wilcoxon on absolute median deviations.
    Tests whether methods differ in spread (precision consistency).
    Effect size is the MAD ratio — directly interpretable.
    """
    M     = len(method_names)
    pairs = list(combinations(range(M), 2))

    # Brown-Forsythe transform
    z     = np.array([np.abs(errors[i] - np.median(errors[i])) for i in range(M)])
    raw_p, stats, mad_ratios = [], [], []

    for i, j in pairs:
        diff = z[i] - z[j]
        if np.all(diff == 0):
            stats.append(0.0); raw_p.append(1.0)
        else:
            w, p = wilcoxon(diff, alternative="two-sided")
            stats.append(w); raw_p.append(p)
        # MAD ratio as effect size (more interpretable than variance ratio
        # for heavy-tailed distributions)
        mad_i = np.median(z[i])
        mad_j = np.median(z[j])
        mad_ratios.append(mad_i / mad_j if mad_j != 0 else np.nan)

    adj_p = holm_bonferroni(raw_p)
    rows  = []
    for k, (i, j) in enumerate(pairs):
        mr = mad_ratios[k]
        better = method_names[i] \
                 if np.median(z[i]) < np.median(z[j]) \
                 else method_names[j]
        rows.append({
            "Method A"        : method_names[i],
            "Method B"        : method_names[j],
            "p (raw)"         : raw_p[k],
            "p (adjusted)"    : adj_p[k],
            "Significant"     : adj_p[k] <= alpha,
            "Lower spread"    : better,
            "MAD ratio (A/B)" : mr,
            "Effect size"     : variance_ratio_label(mr) if not np.isnan(mr) else "n/a",
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# PART 2 — CUSTOMER PERFORMANCE SPECIFICATION
# ──────────────────────────────────────────────────────────────────────────────

def bootstrap_percentile_ci(
    errors     : np.ndarray,
    percentiles: list[float] = [50, 75, 90, 95],
    B          : int   = 10_000,
    ci_level   : float = 0.95,
    seed       : int   = 42,
) -> pd.DataFrame:
    """
    Bootstrap confidence intervals on key percentiles.

    For each percentile p and each method, draws B resamples with replacement
    and computes the p-th percentile of each resample. The CI is the
    (1-ci)/2 and (1+ci)/2 quantiles of those B values.

    This answers: "Given that we've observed N samples from the deployment
    population, what is our uncertainty about the true p-th percentile?"

    Parameters
    ----------
    errors      : (M, N)
    percentiles : list of percentile values to evaluate
    B           : number of bootstrap resamples
    ci_level    : confidence level for the interval (default 0.95)

    Returns
    -------
    DataFrame with one row per (method, percentile), columns:
        Estimate, CI_low, CI_high
    """
    rng  = np.random.default_rng(seed)
    M, N = errors.shape
    lo   = (1.0 - ci_level) / 2.0
    hi   = 1.0 - lo
    rows = []

    for i in range(M):
        # Draw all bootstrap samples at once: (B, N)
        idx      = rng.integers(0, N, size=(B, N))
        boot_err = errors[i][idx]                   # (B, N)

        for p in percentiles:
            boot_pct = np.percentile(boot_err, p, axis=1)   # (B,)
            estimate = np.percentile(errors[i], p)
            rows.append({
                "Method"    : METHOD_NAMES[i],
                "Percentile": f"P{int(p)}",
                "Estimate"  : estimate,
                "CI_low"    : np.quantile(boot_pct, lo),
                "CI_high"   : np.quantile(boot_pct, hi),
                "CI_level"  : f"{int(ci_level*100)}%",
            })

    return pd.DataFrame(rows)


def format_spec_table(ci_df: pd.DataFrame, metric_unit: str) -> pd.DataFrame:
    """
    Pivot bootstrapped CI results into a customer-readable specification table.
    Each cell: 'Estimate (CI_low – CI_high)  unit'
    """
    ci_df = ci_df.copy()
    ci_df["Spec"] = ci_df.apply(
        lambda r: f"{r['Estimate']:.3f}  ({r['CI_low']:.3f} – {r['CI_high']:.3f}) {metric_unit}",
        axis=1
    )
    return ci_df.pivot(index="Method", columns="Percentile", values="Spec")


# ──────────────────────────────────────────────────────────────────────────────
# FULL PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def full_evaluation(
    R_gt         : np.ndarray,
    t_gt         : np.ndarray,
    R_est        : np.ndarray,
    t_est        : np.ndarray,
    method_names : list[str]  = METHOD_NAMES,
    alpha_top    : float      = 0.05,
    boot_B       : int        = 10_000,
    boot_ci      : float      = 0.95,
    percentiles  : list[float]= [50, 75, 90, 95],
    save_plot    : str | None = "llm_examples/",
):
    M, N = R_est.shape[:2]
    alpha = alpha_top / 2.0   # Bonferroni over 2 metrics

    rot_errors   = np.array([geodesic_rotation_error(R_gt, R_est[i]) for i in range(M)])
    trans_errors = np.array([l2_translation_error(t_gt, t_est[i])    for i in range(M)])

    print("=" * 72)
    print("REGISTRATION ALGORITHM EVALUATION")
    print(f"  Methods : {', '.join(method_names)}")
    print(f"  Samples : {N}  |  α per metric = {alpha} "
          f"(Bonferroni over 2 metrics from α = {alpha_top})")
    print("=" * 72)

    for metric_name, errors, unit in [
        ("ROTATION",    rot_errors,   "°"),
        ("TRANSLATION", trans_errors, "m"),
    ]:
        print(f"\n{'═'*72}")
        print(f"  METRIC: {metric_name}")
        print(f"{'═'*72}")

        # ── Descriptives ──────────────────────────────────────────────────────
        print("\n── Descriptive Statistics ──────────────────────────────────────────")
        desc = descriptives(errors, method_names)
        print(desc.to_string(float_format=lambda x: f"{x:.4f}"))

        # ── Part 1: KS omnibus ────────────────────────────────────────────────
        print("\n── Part 1a: KS Distributional Dominance Test ───────────────────────")
        ks = ks_omnibus(errors, method_names, alpha=alpha)
        print(ks[["Method A","Method B","KS stat","p (adjusted)","Significant","Lower errors"]]
              .to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        # ── Part 1: Location ──────────────────────────────────────────────────
        print("\n── Part 1b: Pairwise Wilcoxon (Median / Location) ──────────────────")
        loc = pairwise_location(errors, method_names, alpha=alpha)
        print(loc[["Method A","Method B","p (adjusted)","Significant",
                   "Lower median","Effect r","Effect size"]]
              .to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        # ── Part 1: Scale ─────────────────────────────────────────────────────
        print("\n── Part 1c: Pairwise Brown-Forsythe + Wilcoxon (Spread / MAD) ──────")
        scl = pairwise_scale(errors, method_names, alpha=alpha)
        print(scl[["Method A","Method B","p (adjusted)","Significant",
                   "Lower spread","MAD ratio (A/B)","Effect size"]]
              .to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        # ── Part 2: Customer spec ─────────────────────────────────────────────
        print(f"\n── Part 2: Customer Performance Specification "
              f"({int(boot_ci*100)}% CI, B={boot_B:,}) ────")
        ci_df = bootstrap_percentile_ci(
            errors, percentiles=percentiles, B=boot_B, ci_level=boot_ci
        )
        spec  = format_spec_table(ci_df, unit)
        print(spec.to_string())
        print(f"\n  Read as: 'Estimate  (95% CI lower – upper)  {unit}'")
        print(f"  e.g. P90 row: in 90% of deployment scenarios, error < Estimate {unit}")

    # ── Figures ───────────────────────────────────────────────────────────────
    if save_plot:
        _plot_full(rot_errors, trans_errors, method_names, percentiles,
                   boot_B, boot_ci, save_plot)

    return {"rotation": rot_errors, "translation": trans_errors}


# ──────────────────────────────────────────────────────────────────────────────
# VISUALISATION
# ──────────────────────────────────────────────────────────────────────────────

def _plot_full(rot_errors, trans_errors, method_names,
               percentiles, boot_B, boot_ci, save_path):
    fig = plt.figure(figsize=(20, 14))
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.52, wspace=0.38)

    for row, (errors, label, unit) in enumerate([
        (rot_errors,   "Rotation",    "°"),
        (trans_errors, "Translation", "m"),
    ]):
        M, N = errors.shape

        # ── Col 0: Empirical CDF ──────────────────────────────────────────────
        ax = fig.add_subplot(gs[row, 0])
        for i, (name, col) in enumerate(zip(method_names, COLOURS)):
            sorted_e = np.sort(errors[i])
            cdf      = np.arange(1, N + 1) / N
            ax.plot(sorted_e, cdf, color=col, lw=2, label=name)
        ax.set_xlabel(f"{label} error ({unit})", fontsize=9)
        ax.set_ylabel("Cumulative proportion", fontsize=9)
        ax.set_title(f"{label}\nEmpirical CDF", fontsize=10, fontweight="bold")
        ax.legend(fontsize=8); ax.grid(linestyle="--", alpha=0.4)
        # Mark 90th percentile
        ax.axhline(0.90, color="grey", linestyle=":", lw=1)
        ax.text(ax.get_xlim()[1]*0.02, 0.91, "P90", fontsize=7, color="grey")

        # ── Col 1: Box plot ───────────────────────────────────────────────────
        ax = fig.add_subplot(gs[row, 1])
        bp = ax.boxplot(
            [errors[i] for i in range(M)],
            labels=method_names, patch_artist=True,
            medianprops=dict(color="black", lw=2),
            flierprops=dict(marker=".", markersize=3, alpha=0.4),
        )
        for patch, col in zip(bp["boxes"], COLOURS):
            patch.set_facecolor(col); patch.set_alpha(0.75)
        ax.set_ylabel(f"Error ({unit})", fontsize=9)
        ax.set_title(f"{label}\nBox Plot", fontsize=10, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        # ── Col 2: MAD bar chart ──────────────────────────────────────────────
        ax = fig.add_subplot(gs[row, 2])
        mads    = [np.median(np.abs(errors[i] - np.median(errors[i])))
                   for i in range(M)]
        medians = [np.median(errors[i]) for i in range(M)]
        x       = np.arange(M)
        bars    = ax.bar(x, medians, color=COLOURS, alpha=0.5,
                         edgecolor="black", lw=0.8, label="Median")
        ax.bar(x, mads, bottom=medians, color=COLOURS, alpha=0.85,
               edgecolor="black", lw=0.8, hatch="///", label="MAD above median")
        ax.set_xticks(x); ax.set_xticklabels(method_names, fontsize=8)
        ax.set_ylabel(f"Error ({unit})", fontsize=9)
        ax.set_title(f"{label}\nMedian + MAD", fontsize=10, fontweight="bold")
        ax.legend(fontsize=7); ax.grid(axis="y", linestyle="--", alpha=0.4)

        # ── Col 3: Bootstrapped percentile CIs ───────────────────────────────
        ax   = fig.add_subplot(gs[row, 3])
        ci_df = bootstrap_percentile_ci(
            errors, percentiles=percentiles, B=boot_B, ci_level=boot_ci
        )
        pct_labels = [f"P{int(p)}" for p in percentiles]
        x_pct      = np.arange(len(pct_labels))
        width      = 0.18
        offsets    = np.linspace(-(M-1)*width/2, (M-1)*width/2, M)

        for i, (name, col) in enumerate(zip(method_names, COLOURS)):
            sub = ci_df[ci_df["Method"] == name].set_index("Percentile")
            ests = [sub.loc[pl, "Estimate"] for pl in pct_labels]
            los  = [sub.loc[pl, "Estimate"] - sub.loc[pl, "CI_low"]  for pl in pct_labels]
            his  = [sub.loc[pl, "CI_high"]  - sub.loc[pl, "Estimate"] for pl in pct_labels]
            ax.bar(x_pct + offsets[i], ests, width=width,
                   color=col, alpha=0.75, edgecolor="black", lw=0.6, label=name)
            ax.errorbar(x_pct + offsets[i], ests,
                        yerr=[los, his], fmt="none",
                        color="black", capsize=3, lw=1)

        ax.set_xticks(x_pct); ax.set_xticklabels(pct_labels, fontsize=9)
        ax.set_ylabel(f"Error ({unit})", fontsize=9)
        ax.set_title(f"{label}\nBootstrapped Percentile CIs\n"
                     f"({int(boot_ci*100)}% CI, B={boot_B:,})",
                     fontsize=10, fontweight="bold")
        ax.legend(fontsize=7); ax.grid(axis="y", linestyle="--", alpha=0.4)

    # ── Row 2: Joint scatter (mean rotation vs mean translation) ─────────────
    ax_joint = fig.add_subplot(gs[2, :2])
    for i, (name, col) in enumerate(zip(method_names, COLOURS)):
        med_r = np.median(rot_errors[i])
        med_t = np.median(trans_errors[i])
        mad_r = np.median(np.abs(rot_errors[i]   - med_r))
        mad_t = np.median(np.abs(trans_errors[i] - med_t))
        ax_joint.scatter(med_r, med_t, color=col, s=200, zorder=5,
                         label=name, edgecolors="black", lw=0.8)
        ax_joint.errorbar(med_r, med_t, xerr=mad_r, yerr=mad_t,
                          fmt="none", color=col, alpha=0.55, capsize=5, lw=1.5)
    ax_joint.set_xlabel("Median Rotation Error (°)", fontsize=10)
    ax_joint.set_ylabel("Median Translation Error (m)", fontsize=10)
    ax_joint.set_title("Joint Performance Summary\n(Median ± MAD)",
                        fontsize=11, fontweight="bold")
    ax_joint.legend(fontsize=9); ax_joint.grid(linestyle="--", alpha=0.4)

    # ── Row 2: Percentile dominance table ────────────────────────────────────
    ax_tbl = fig.add_subplot(gs[2, 2:])
    ax_tbl.axis("off")

    pct_labels = [f"P{int(p)}" for p in [50, 75, 90, 95]]
    col_labels = ["Method"] + [f"Rot {pl} (°)" for pl in pct_labels] \
                            + [f"Trans {pl} (m)" for pl in pct_labels]
    table_data = []
    for i, name in enumerate(method_names):
        row_vals = [name]
        for errors in [rot_errors, trans_errors]:
            for p in [50, 75, 90, 95]:
                row_vals.append(f"{np.percentile(errors[i], p):.3f}")
        table_data.append(row_vals)

    tbl = ax_tbl.table(
        cellText=table_data, colLabels=col_labels,
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.6)
    # Colour header
    for j in range(len(col_labels)):
        tbl[(0, j)].set_facecolor("#2c3e50")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")
    # Colour best (min) in each column
    for col_idx in range(1, len(col_labels)):
        vals = [float(table_data[r][col_idx]) for r in range(len(method_names))]
        best = int(np.argmin(vals))
        tbl[(best + 1, col_idx)].set_facecolor("#d5f5e3")

    ax_tbl.set_title("Percentile Summary Table  (green = best per column)",
                     fontsize=10, fontweight="bold", pad=12)

    fig.suptitle(
        "Registration Algorithm Evaluation\n"
        "GICP  |  Semantic-GICP  |  NDT  |  Semantic-NDT",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n[Plot] Saved → {save_path}")
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# DEMO
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    R_gt, t_gt, R_est, t_est = generate_synthetic_data(N=200)

    full_evaluation(
        R_gt, t_gt, R_est, t_est,
        method_names = METHOD_NAMES,
        alpha_top    = 0.05,
        boot_B       = 10_000,
        boot_ci      = 0.95,
        percentiles  = [50, 75, 90, 95],
    )