"""
Pose Estimation Statistical Evaluation
=======================================
4-Step test plan:
  1. Compute geodesic rotation error and L2 translation error
  2. Report descriptive statistics (mean ± std, median + IQR)
  3. Omnibus Friedman test per metric (α = 0.025 after top-level Bonferroni)
  4. Post-hoc Wilcoxon signed-rank with Holm-Bonferroni correction + rank-biserial effect size

Input format
------------
Provide ground-truth and per-method poses as numpy arrays:
  R_gt   : (N, 3, 3)  ground-truth rotation matrices
  t_gt   : (N, 3)     ground-truth translation vectors
  R_est  : (M, N, 3, 3)  estimated rotations  for M=4 methods
  t_est  : (M, N, 3)     estimated translations for M=4 methods

A synthetic dataset is generated at the bottom so the script runs out of the box.
"""

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from itertools import combinations
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# 1.  ERROR METRICS
# ──────────────────────────────────────────────────────────────────────────────

def geodesic_rotation_error(R_gt: np.ndarray, R_est: np.ndarray) -> np.ndarray:
    """
    Geodesic angular error on SO(3) in degrees.

    θ_err = arccos( (tr(R_gt^T @ R_est) - 1) / 2 )

    Parameters
    ----------
    R_gt  : (N, 3, 3)
    R_est : (N, 3, 3)

    Returns
    -------
    errors : (N,) in degrees
    """
    # R_rel = R_gt^T @ R_est  — the relative rotation (error rotation)
    R_rel = np.einsum("nij,njk->nik", R_gt.transpose(0, 2, 1), R_est)

    # Clamp trace argument to [-1, 1] to guard against floating-point drift
    trace = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
    cos_theta = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)

    return np.degrees(np.arccos(cos_theta))


def l2_translation_error(t_gt: np.ndarray, t_est: np.ndarray) -> np.ndarray:
    """
    L2 (Euclidean) translation error.

    Parameters
    ----------
    t_gt  : (N, 3)
    t_est : (N, 3)

    Returns
    -------
    errors : (N,)
    """
    return np.linalg.norm(t_gt - t_est, axis=1)


# ──────────────────────────────────────────────────────────────────────────────
# 2.  DESCRIPTIVE STATISTICS
# ──────────────────────────────────────────────────────────────────────────────

def descriptive_stats(errors: np.ndarray, method_names: list[str]) -> pd.DataFrame:
    """
    Mean ± std, median, IQR for each method.

    Parameters
    ----------
    errors       : (M, N)  M methods, N samples
    method_names : list of M strings

    Returns
    -------
    pd.DataFrame with one row per method
    """
    rows = []
    for i, name in enumerate(method_names):
        e = errors[i]
        q25, q75 = np.percentile(e, [25, 75])
        rows.append({
            "Method"   : name,
            "Mean"     : e.mean(),
            "Std"      : e.std(ddof=1),
            "Median"   : np.median(e),
            "IQR"      : q75 - q25,
            "Min"      : e.min(),
            "Max"      : e.max(),
        })
    return pd.DataFrame(rows).set_index("Method")


# ──────────────────────────────────────────────────────────────────────────────
# 3.  OMNIBUS FRIEDMAN TEST
# ──────────────────────────────────────────────────────────────────────────────

def friedman_test(errors: np.ndarray, method_names: list[str],
                  alpha: float = 0.025) -> dict:
    """
    Friedman test across M methods on N paired samples.

    Parameters
    ----------
    errors       : (M, N)
    method_names : list of M strings
    alpha        : significance threshold (0.025 after top-level Bonferroni
                   for 2 metrics)

    Returns
    -------
    dict with statistic, p-value, significance flag, and formatted string
    """
    stat, p = friedmanchisquare(*[errors[i] for i in range(errors.shape[0])])
    result = {
        "statistic"   : stat,
        "p_value"     : p,
        "alpha"       : alpha,
        "significant" : p <= alpha,
    }
    sig_str = "✓ SIGNIFICANT" if result["significant"] else "✗ NOT significant"
    result["summary"] = (
        f"Friedman χ²({len(method_names)-1}) = {stat:.4f},  "
        f"p = {p:.4e}  [{sig_str} at α = {alpha}]"
    )
    return result


# ──────────────────────────────────────────────────────────────────────────────
# 4.  POST-HOC: WILCOXON + HOLM-BONFERRONI + EFFECT SIZE
# ──────────────────────────────────────────────────────────────────────────────

def rank_biserial_r(w_stat: float, n: int) -> float:
    """
    Rank-biserial correlation from a Wilcoxon signed-rank statistic.

    r = 1 - (2W) / (n(n+1)/2)

    |r| ≈ 0.1 small | 0.3 medium | 0.5 large
    """
    return 1.0 - (2.0 * w_stat) / (n * (n + 1) / 2.0)


def effect_size_label(r: float) -> str:
    a = abs(r)
    if a >= 0.5:
        return "large"
    elif a >= 0.3:
        return "medium"
    elif a >= 0.1:
        return "small"
    else:
        return "negligible"


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """
    Holm-Bonferroni correction.  Returns adjusted p-values.
    """
    k = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.zeros(k)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = p_values[idx] * (k - rank)
        running_max = max(running_max, adj)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted.tolist()


def posthoc_wilcoxon(errors: np.ndarray, method_names: list[str],
                     alpha: float = 0.025) -> pd.DataFrame:
    """
    Pairwise Wilcoxon signed-rank tests with Holm-Bonferroni correction
    and rank-biserial effect sizes.

    Parameters
    ----------
    errors       : (M, N)
    method_names : list of M strings
    alpha        : same threshold used for the Friedman test

    Returns
    -------
    pd.DataFrame with one row per pair
    """
    pairs      = list(combinations(range(len(method_names)), 2))
    raw_p      = []
    stats      = []
    effect_r   = []

    for i, j in pairs:
        diff = errors[i] - errors[j]
        # If all differences are zero Wilcoxon is undefined — treat as p=1
        if np.all(diff == 0):
            stats.append(0.0)
            raw_p.append(1.0)
            effect_r.append(0.0)
        else:
            w, p = wilcoxon(diff, alternative="two-sided")
            stats.append(w)
            raw_p.append(p)
            effect_r.append(rank_biserial_r(w, len(diff)))

    adjusted_p = holm_bonferroni(raw_p)

    rows = []
    for k, (i, j) in enumerate(pairs):
        sig = adjusted_p[k] <= alpha
        rows.append({
            "Method A"      : method_names[i],
            "Method B"      : method_names[j],
            "W statistic"   : stats[k],
            "p (raw)"       : raw_p[k],
            "p (adjusted)"  : adjusted_p[k],
            "Significant"   : sig,
            "Effect r"      : effect_r[k],
            "Effect size"   : effect_size_label(effect_r[k]),
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# 5.  FULL PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_poses(
    R_gt   : np.ndarray,
    t_gt   : np.ndarray,
    R_est  : np.ndarray,
    t_est  : np.ndarray,
    method_names : list[str] | None = None,
    alpha_top    : float = 0.05,
    plot         : bool  = True,
    save_plot    : str | None = "pose_evaluation.png",
):
    """
    Full 4-step evaluation pipeline.

    Parameters
    ----------
    R_gt         : (N, 3, 3)
    t_gt         : (N, 3)
    R_est        : (M, N, 3, 3)
    t_est        : (M, N, 3)
    method_names : list of M strings (default: Method 0 … Method M-1)
    alpha_top    : family-wise α before Bonferroni split over 2 metrics (default 0.05)
    plot         : whether to generate the summary figure
    save_plot    : path to save figure (None = display only)
    """
    M, N = R_est.shape[:2]
    if method_names is None:
        method_names = [f"Method {i}" for i in range(M)]

    # Bonferroni correction over 2 metrics
    alpha = alpha_top / 2.0

    # ── Compute per-sample errors ────────────────────────────────────────────
    rot_errors   = np.array([geodesic_rotation_error(R_gt, R_est[i]) for i in range(M)])
    trans_errors = np.array([l2_translation_error(t_gt, t_est[i])    for i in range(M)])

    print("=" * 70)
    print("POSE EVALUATION — STATISTICAL PIPELINE")
    print(f"  {M} methods | {N} samples | α_per_metric = {alpha} "
          f"(Bonferroni over 2 metrics from α = {alpha_top})")
    print("=" * 70)

    for metric_name, errors in [("ROTATION (degrees)", rot_errors),
                                 ("TRANSLATION (L2)",   trans_errors)]:

        print(f"\n{'─'*70}")
        print(f"  METRIC: {metric_name}")
        print(f"{'─'*70}")

        # Step 2 — Descriptive statistics
        print("\n[Step 2] Descriptive Statistics")
        desc = descriptive_stats(errors, method_names)
        print(desc.to_string(float_format=lambda x: f"{x:.4f}"))

        # Step 3 — Friedman omnibus
        print("\n[Step 3] Friedman Omnibus Test")
        friedman = friedman_test(errors, method_names, alpha=alpha)
        print(f"  {friedman['summary']}")

        # Step 4 — Post-hoc (only if Friedman is significant)
        print("\n[Step 4] Post-hoc Pairwise Tests (Wilcoxon + Holm-Bonferroni)")
        if not friedman["significant"]:
            print("  Friedman not significant — post-hoc tests not warranted.")
        else:
            posthoc = posthoc_wilcoxon(errors, method_names, alpha=alpha)
            # Format for readability
            display_cols = ["Method A", "Method B", "p (raw)",
                            "p (adjusted)", "Significant", "Effect r", "Effect size"]
            print(posthoc[display_cols].to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)
            ))

    # ── Summary scatter plot ─────────────────────────────────────────────────
    if plot:
        _plot_summary(rot_errors, trans_errors, method_names, save_plot)

    return {
        "rotation_errors"    : rot_errors,
        "translation_errors" : trans_errors,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 6.  VISUALISATION
# ──────────────────────────────────────────────────────────────────────────────

def _plot_summary(rot_errors, trans_errors, method_names, save_path):
    colours = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    M = len(method_names)

    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax_rot   = fig.add_subplot(gs[0, 0])   # rotation box-plots
    ax_trans = fig.add_subplot(gs[1, 0])   # translation box-plots
    ax_rot_v = fig.add_subplot(gs[0, 1])   # rotation violin
    ax_tr_v  = fig.add_subplot(gs[1, 1])   # translation violin
    ax_joint = fig.add_subplot(gs[:, 2])   # joint scatter

    def _boxplot(ax, errors, title, ylabel):
        bp = ax.boxplot(
            [errors[i] for i in range(M)],
            labels=method_names, patch_artist=True, notch=False,
            medianprops=dict(color="black", linewidth=2),
        )
        for patch, col in zip(bp["boxes"], colours):
            patch.set_facecolor(col)
            patch.set_alpha(0.7)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    def _violin(ax, errors, title, ylabel):
        parts = ax.violinplot(
            [errors[i] for i in range(M)],
            positions=range(M), showmedians=True,
        )
        for pc, col in zip(parts["bodies"], colours):
            pc.set_facecolor(col)
            pc.set_alpha(0.6)
        ax.set_xticks(range(M))
        ax.set_xticklabels(method_names)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    _boxplot(ax_rot,   rot_errors,   "Rotation Error",    "Geodesic error (°)")
    _boxplot(ax_trans, trans_errors, "Translation Error", "L2 error")
    _violin(ax_rot_v,  rot_errors,   "Rotation Error",    "Geodesic error (°)")
    _violin(ax_tr_v,   trans_errors, "Translation Error", "L2 error")

    # Joint scatter — mean per method
    for i, (name, col) in enumerate(zip(method_names, colours)):
        ax_joint.scatter(
            np.mean(rot_errors[i]), np.mean(trans_errors[i]),
            color=col, s=180, zorder=5, label=name, edgecolors="black", linewidths=0.8,
        )
        # Error bars showing ± std
        ax_joint.errorbar(
            np.mean(rot_errors[i]), np.mean(trans_errors[i]),
            xerr=np.std(rot_errors[i], ddof=1),
            yerr=np.std(trans_errors[i], ddof=1),
            fmt="none", color=col, alpha=0.5, capsize=4,
        )
    ax_joint.set_xlabel("Mean Rotation Error (°)", fontsize=10)
    ax_joint.set_ylabel("Mean Translation Error (L2)", fontsize=10)
    ax_joint.set_title("Joint Error Summary\n(mean ± std)", fontsize=11, fontweight="bold")
    ax_joint.legend(fontsize=9)
    ax_joint.grid(linestyle="--", alpha=0.5)

    fig.suptitle("Pose Estimation Evaluation", fontsize=14, fontweight="bold", y=1.01)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n[Plot] Saved to: {save_path}")
    else:
        plt.show()
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# 7.  SYNTHETIC DATA + DEMO
# ──────────────────────────────────────────────────────────────────────────────

def _random_rotation(rng) -> np.ndarray:
    """Uniformly random rotation matrix via QR decomposition."""
    H = rng.standard_normal((3, 3))
    Q, R = np.linalg.qr(H)
    Q *= np.sign(np.diag(R))   # ensure det = +1
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def _perturb_rotation(R: np.ndarray, noise_deg: float, rng) -> np.ndarray:
    """
    Perturb R by a small random rotation of magnitude ~ noise_deg degrees.
    """
    axis  = rng.standard_normal(3)
    axis /= np.linalg.norm(axis)
    angle = np.radians(rng.normal(0, noise_deg))
    K = np.array([
        [0,       -axis[2],  axis[1]],
        [axis[2],  0,       -axis[0]],
        [-axis[1], axis[0],  0      ],
    ])
    R_perturb = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
    return R_perturb @ R


def generate_synthetic_data(N: int = 200, seed: int = 42):
    """
    Generate ground-truth and 4 estimated pose sets with increasing noise.
    """
    rng = np.random.default_rng(seed)

    R_gt = np.array([_random_rotation(rng) for _ in range(N)])
    t_gt = rng.uniform(-5, 5, size=(N, 3))

    # Four methods: progressively noisier, with different characters
    noise_configs = [
        dict(rot_std=2.0,  trans_std=0.05),   # Method 0 — very good
        dict(rot_std=5.0,  trans_std=0.15),   # Method 1 — good
        dict(rot_std=10.0, trans_std=0.30),   # Method 2 — moderate
        dict(rot_std=18.0, trans_std=0.60),   # Method 3 — poor
    ]

    R_est_all = []
    t_est_all = []
    for cfg in noise_configs:
        R_est = np.array([_perturb_rotation(R_gt[n], cfg["rot_std"], rng)
                          for n in range(N)])
        t_est = t_gt + rng.normal(0, cfg["trans_std"], size=(N, 3))
        R_est_all.append(R_est)
        t_est_all.append(t_est)

    return (
        R_gt, t_gt,
        np.stack(R_est_all),   # (4, N, 3, 3)
        np.stack(t_est_all),   # (4, N, 3)
    )


if __name__ == "__main__":
    R_gt, t_gt, R_est, t_est = generate_synthetic_data(N=200)

    evaluate_poses(
        R_gt, t_gt, R_est, t_est,
        method_names=["Method A", "Method B", "Method C", "Method D"],
        alpha_top=0.05,
        plot=True,
        save_plot="/mnt/user-data/outputs/pose_evaluation.png",
    )