"""
Paper outputs from scripts/launch_paper_alpha.sh (runs_paper/paper_alpha):
  Table 1  every decision rule + alpha-potential (Eq. 12) + self-play, both profiles, 3 seeds
  Table 2  unilateral response gains (policy-level, Eq. 15)
  Table 3  self-play deployment: categorical sampling vs argmax
  static 5x3 matrices (heatmaps) and Figures 2 / 3 (data volume, discount factor) via
  summarize_alpha_ablations.

    python scripts/summarize_paper_alpha.py --root analysis/server_results/paper_runs/paper_alpha --out-dir OUT
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import glob
import json

import numpy as np

PROFILES = ("aggressive", "moderate_aggressive")
RULE_ROWS = (("nash-q", "Nash-Q"), ("br-q", "BR-Q"), ("nash-p-ibr", "Potential IBR"), ("nash-p-br", "Potential BR"),
             ("nash-p-adaptive", "Potential Adaptive Match"), ("nash-p-minimax", "Potential Minimax"),
             ("nash-p-argmax", "α-potential (Eq. 12)"))
# selfplay-nc: actor input pi(x, role) as in paper Eq. 13; selfplay: input also carries both players' current choices
SELFPLAY_ROWS = (("selfplay-nc", "Self-play RL, π(x, role) (sampled)"),
                 ("selfplay-nc-argmax", "Self-play RL, π(x, role) (argmax)"),
                 ("selfplay", "Self-play RL, + choice inputs (sampled)"),
                 ("selfplay-argmax", "Self-play RL, + choice inputs (argmax)"))
DEPLOYMENT_ROWS = (("selfplay-nc-argmax", "Deterministic argmax — π(x, role)"),
                   ("selfplay-nc", "Categorical sampling — π(x, role)"),
                   ("selfplay-argmax", "Deterministic argmax — + choice inputs"),
                   ("selfplay", "Categorical sampling — + choice inputs"))
PROFILE_LABELS = {"aggressive": "Aggressive profile", "moderate_aggressive": "Moderate profile"}
# (metric key in summarize_alpha_ablations rows, legend label, line style). The figures count a rally
# unfinished at the step limit as half a point: playing such rallies to the end (diagnose_unfinished_rallies.py)
# gave win rates within 1-5 points of that, while dropping them biased the static minimum by up to 23 points.
WIN_SERIES = (("static_mean_half", "static mean", "-"), ("static_min_half", "static minimum", "--"),
              ("random", "vs random", ":"))
MODEL_SERIES = (("heldout_q1_mse", "held-out value MSE", "-"), ("alpha_p95", "α: p95 |residual|", "--"),
                ("alpha_mean", "mean |residual|", "-."), ("alpha_mean_constant", "mean |residual|, Φ ≡ 0", ":"))


def _load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _static_labels():
    from nash_skills.defend_positions import DEFEND_POSITION_NAMES
    from nash_skills.skills import SKILL_NAMES
    return [f"{a}/{d}" for a in SKILL_NAMES for d in DEFEND_POSITION_NAMES]


def cell_win(cells, prefix):
    hits = [v for k, v in cells.items() if k == prefix or k.startswith(prefix + ":")]
    return hits[0]["win"] if hits else float("nan")


def row_stats(result, played_out=None):
    """Table 1 statistics of one seed; `played_out` replaces cells replayed with every rally played to the end."""
    cells = result["summary"]["cells"]
    static = [(played_out or {}).get(k, cells[k]["win"]) for k in _static_labels() if k in cells]
    return {"static_mean": float(np.mean(static)) if static else float("nan"),
            "static_min": float(np.min(static)) if static else float("nan"),
            "random": cell_win(cells, "random"), "ibr": cell_win(cells, "ibr"),
            "adaptive": cell_win(cells, "nash-p-adaptive"),
            "vs_selfplay": cell_win(cells, "selfplay"), "vs_alpha": cell_win(cells, "nash-p-argmax")}


def ms(values, pct=True):
    v = np.array([x for x in values if x == x], dtype=float)
    if v.size == 0:
        return "—"
    k = 100.0 if pct else 1.0
    return f"{k * v.mean():.1f}" + (f" ± {k * v.std(ddof=1):.1f}" if v.size > 1 else "")


def _indexed_seed_results(root, profile, sub, name):
    """(seed, evaluation JSON) of one method for seeds 0-2 (missing or empty files skipped)."""
    pairs = ((s, _load(os.path.join(root, profile, sub, f"{name}_s{s}.json"))) for s in range(3))
    return [(s, r) for s, r in pairs if r]


def _seed_results(root, profile, sub, name):
    return [r for _s, r in _indexed_seed_results(root, profile, sub, name)]


def played_out_cells(root, profile, name, seed):
    """Win rates of cells replayed with every rally played to the end (diagnose_unfinished_rallies.py),
    read at the longest cutoff; {} when the method has no replay for this seed."""
    rep = _load(os.path.join(root, profile, "diag_unfinished", f"{name}_s{seed}.json"))
    out = {}
    for cell, s in (rep or {}).get("summary", {}).items():
        last = s["by_cutoff"][max(s["by_cutoff"], key=int)]
        if last["wins"] + last["losses"]:
            out[cell] = last["wins"] / (last["wins"] + last["losses"])
    return out


def table1(root, profile):
    """Paper Table 1 columns; IBR as the paper defines it (simulated, H_IBR = 5) and the Q-based variant.
    Static cells replayed with every rally played to the end (played_out_cells) use that win rate."""
    lines = ["| Method | Static mean | Static minimum | Random | IBR (simulated, H=5) | IBR (Q-based) | "
             "Other proposed: vs Potential Adaptive Match |", "|" + "---|" * 7]
    for name, label in RULE_ROWS + SELFPLAY_ROWS:
        seeds = [row_stats(r, played_out_cells(root, profile, name, s))
                 for s, r in _indexed_seed_results(root, profile, "table1", name)]
        if not seeds:
            continue
        sims = [cell_win(r["summary"]["cells"], "ibr-sim") for r in _seed_results(root, profile, "ibr_sim", name)]
        col = lambda k: ms([x[k] for x in seeds])
        lines.append(f"| {label} | {col('static_mean')} | {col('static_min')} | {col('random')} | {ms(sims)} | "
                     f"{col('ibr')} | {col('adaptive')} |")
    return "\n".join(lines)


def head_to_head_table(root, profile):
    """alpha-potential (Eq. 12) against each self-play deployment, from the self-play rows' evaluations."""
    lines = ["| Self-play deployment | Self-play wins | α-potential wins |", "|---|---|---|"]
    for name, label in SELFPLAY_ROWS:
        wins = [w for w in (row_stats(r)["vs_alpha"] for r in _seed_results(root, profile, "table1", name)) if w == w]
        if wins:
            lines.append(f"| {label} | {ms(wins)} | {ms([1.0 - w for w in wins])} |")
    return "\n".join(lines)


def table2(root, profile):
    lines = ["| Method | Player 1 response gain | Player 2 response gain | Profile regret (max) |", "|---|---|---|---|"]
    for name, label in RULE_ROWS + SELFPLAY_ROWS:
        res = [r for r in (_load(os.path.join(root, profile, "gains", f"{name}_s{s}.json")) for s in range(3)) if r]
        if not res:
            continue
        g = lambda p: [r["summary"][f"player{p}"]["gain"] for r in res]
        reg = [r["summary"]["profile_regret"] for r in res]
        lines.append(f"| {label} | {ms(g(1), False)} | {ms(g(2), False)} | {ms(reg, False)} |")
    return "\n".join(lines)


def table3(root, profile):
    lines = ["| Deployment | Static mean | Static minimum | Random | IBR (simulated, H=5) | IBR (Q-based) |",
             "|---|---|---|---|---|---|"]
    for name, label in DEPLOYMENT_ROWS:
        seeds = [row_stats(r) for r in _seed_results(root, profile, "table1", name)]
        sims = [cell_win(r["summary"]["cells"], "ibr-sim") for r in _seed_results(root, profile, "ibr_sim", name)]
        if seeds:
            col = lambda k: ms([x[k] for x in seeds])
            lines.append(f"| {label} | {col('static_mean')} | {col('static_min')} | {col('random')} | {ms(sims)} | {col('ibr')} |")
    return "\n".join(lines)


def model_quality_table(root, profile, run="gamma0.99"):
    """Held-out value error and unilateral residuals |e| (empirical alpha, Eq. 10) of the main models,
    with the trivial potential Phi = 0 in parentheses: the baseline a learned Phi has to beat."""
    reps = [r for r in (_load(os.path.join(root, profile, "phi", f"{run}_s{s}", "heldout_attack_defend.json"))
                        for s in range(3)) if r]
    lines = ["| Seeds | Train rallies | Held-out value MSE | Predict-zero MSE | α max | α p95 | α median | α mean |",
             "|---|---|---|---|---|---|---|---|"]
    if not reps:
        return "\n".join(lines)
    q = np.array([r["heldout_q1_mse"] for r in reps])
    pair = lambda k: (f"{np.mean([r['heldout_residuals'][k] for r in reps]):.3f} "
                      f"({np.mean([r['heldout_residuals_constant_phi'][k] for r in reps]):.3f})")
    q_txt = f"{q.mean():.3f}" + (f" ± {q.std(ddof=1):.3f}" if q.size > 1 else "")
    lines.append(f"| {len(reps)} | {int(np.mean([r['n_train_rallies'] for r in reps]))} | {q_txt} | "
                 f"{np.mean([r['heldout_q1_mse_predict_zero'] for r in reps]):.3f} | {pair('max_abs')} | "
                 f"{pair('p95_abs')} | {pair('median_abs')} | {pair('mean_abs')} |")
    return "\n".join(lines)


def timing_table(root, profile):
    """Mean online decision time per method (Table 1 runs) and of the simulated-IBR opponent."""
    lines = ["| Method | Online decision time (ms) |", "|---|---|"]
    for name, label in RULE_ROWS + SELFPLAY_ROWS:
        res = [r for r in (_load(os.path.join(root, profile, "table1", f"{name}_s{s}.json")) for s in range(3)) if r]
        t = [r["summary"].get("timing", {}).get("method_mean_s") for r in res]
        t = [x for x in t if x is not None]
        if t:
            lines.append(f"| {label} | {1000 * np.mean(t):.1f} |")
    sims = [r["summary"].get("timing", {}).get("by_opponent", {}).get("ibr-sim", {}).get("opponent_mean_s")
            for r in map(_load, sorted(glob.glob(os.path.join(root, profile, "ibr_sim", "*.json")))) if r]
    sims = [x for x in sims if x is not None]
    if sims:
        lines.append(f"\nSimulated IBR opponent (H_IBR = 5, two rounds over the 15 fixed pairs): "
                     f"{np.mean(sims):.1f} s per decision.")
    return "\n".join(lines)


def _short_label(cell):
    return "self-play" if cell.startswith("selfplay") else cell.split(":")[0]


def sample_note(root, profile, name, stall_share=0.2):
    """Evaluation budget behind one Table 1 row: seeds, decided rallies per opponent per seed, and the
    opponents whose rallies often reach the step limit (re-drawn by the evaluator, so they are pooled
    over attempts rather than over rallies)."""
    res = [r for r in (_load(os.path.join(root, profile, "table1", f"{name}_s{s}.json")) for s in range(3)) if r]
    if not res:
        return ""
    decided, truncated = {}, {}
    for r in res:
        for k, c in r["summary"]["cells"].items():
            decided[k] = decided.get(k, 0) + c["decided"]
            truncated[k] = truncated.get(k, 0) + c["truncated"]
    median_decided = int(np.median([c["decided"] for r in res for c in r["summary"]["cells"].values()]))
    shares = sorted(((truncated[k] / (decided[k] + truncated[k]), k) for k in decided if decided[k] + truncated[k]),
                    reverse=True)
    stalled = [f"{_short_label(k)} ({100 * s:.0f}% of attempts)" for s, k in shares if s > stall_share]
    note = (f"{len(res)} seeds × {median_decided} decided rallies per opponent (median; table sides and initial "
            f"roles balanced). Rallies that reach the step limit are re-drawn and excluded from win rates")
    note += ("; frequent stalls vs " + ", ".join(stalled) if stalled else "") + "."
    replayed = sorted({k for s in range(3) for k in played_out_cells(root, profile, name, s)})
    if replayed:
        note += (f" Static columns use win rates with every rally played to the end (20,000-step limit) for "
                 f"{', '.join(replayed)}.")
    return note


def curve(rows, key, x_from_n_train=False):
    """(x, mean, sd) arrays, sorted by x, from summarize_alpha_ablations.table rows; NaN means skipped."""
    pts = sorted(((stats["n_train_rallies"][0] if x_from_n_train else x), stats[key][0], stats[key][1])
                 for x, stats in rows if np.isfinite(stats[key][0]))
    return tuple(np.array(v) for v in zip(*pts)) if pts else None


def plot_curves(path_stem, series, xlabel, keys, ylabel, x_from_n_train=False, logx=False, pct=True):
    """One panel per profile, mean over seeds with a ±1 sd band; writes <stem>.png and <stem>.pdf."""
    curves = {p: [(label, ls, curve(rows, k, x_from_n_train)) for k, label, ls in keys] for p, rows in series.items()}
    if not any(c is not None for items in curves.values() for _l, _s, c in items):
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    fig, axes = plt.subplots(1, len(curves), figsize=(3.3 * len(curves), 2.6), sharey=True, squeeze=False)
    for ax, (profile, items) in zip(axes[0], curves.items()):
        ax.set_title(PROFILE_LABELS.get(profile, profile), fontsize=9)
        for label, ls, c in items:
            if c is not None:
                xs, m, s = c
                line, = ax.plot(xs, m, ls, marker="o", ms=3, lw=1.2, label=label)
                ax.fill_between(xs, m - s, m + s, color=line.get_color(), alpha=0.15, lw=0)
        if logx:
            ax.set_xscale("log")
        if pct:
            ax.axhline(0.5, color="grey", lw=0.7, ls=(0, (4, 3)))
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_xlabel(xlabel, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25, lw=0.5)
    axes[0][0].set_ylabel(ylabel, fontsize=8)
    labelled = [ax for ax in axes[0] if ax.get_legend_handles_labels()[0]]
    labelled[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path_stem}.{ext}", dpi=200)
    plt.close(fig)
    return f"{path_stem}.png"


def static_matrix(root, profile, name):
    """5 x 3 mean win rate over seeds (rows: opponent's attack target, cols: its preparation); cells replayed
    with every rally played to the end use that win rate, as in Table 1."""
    from nash_skills.defend_positions import DEFEND_POSITION_NAMES
    from nash_skills.skills import SKILL_NAMES
    res = _indexed_seed_results(root, profile, "table1", name)
    if not res:
        return None
    overrides = {s: played_out_cells(root, profile, name, s) for s, _r in res}
    win = lambda s, r, k: overrides[s].get(k, r["summary"]["cells"][k]["win"])
    return np.array([[np.mean([win(s, r, f"{a}/{d}") for s, r in res]) for d in DEFEND_POSITION_NAMES]
                     for a in SKILL_NAMES])


def matrix_methods(root):
    """alpha-potential plus the self-play actor to show: the Eq. 13 one when evaluated, else the other."""
    def has(name):
        return any(os.path.exists(os.path.join(root, p, "table1", f"{name}_s{s}.json")) for p in PROFILES for s in range(3))
    nc = has("selfplay-nc") or has("selfplay-nc-argmax")
    tag = "π(x, role)" if nc else "+ choice inputs"
    return (("nash-p-argmax", "α-potential (Eq. 12)"),
            ("selfplay-nc" if nc else "selfplay", f"Self-play, {tag}\n(sampled)"),
            ("selfplay-nc-argmax" if nc else "selfplay-argmax", f"Self-play, {tag}\n(argmax)"))


def plot_matrices(root, out_dir, profiles=PROFILES):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    from nash_skills.defend_positions import DEFEND_POSITION_NAMES
    from nash_skills.skills import SKILL_NAMES
    methods = matrix_methods(root)
    mats = {(p, n): static_matrix(root, p, n) for p in profiles for n, _l in methods}
    if all(m is None for m in mats.values()):
        return None
    fig, axes = plt.subplots(len(profiles), len(methods), figsize=(7.2, 3.3 * len(profiles)), squeeze=False)
    for i, profile in enumerate(profiles):
        for j, (name, label) in enumerate(methods):
            ax = axes[i][j]
            m = mats[(profile, name)]
            ax.set_title(f"{label}\n{PROFILE_LABELS.get(profile, profile)}", fontsize=8)
            if m is None:
                ax.axis("off")
                continue
            ax.imshow(m, vmin=0, vmax=1, cmap="RdYlGn")
            for a in range(m.shape[0]):
                for d in range(m.shape[1]):
                    ax.text(d, a, f"{100 * m[a, d]:.0f}", ha="center", va="center", fontsize=7)
            ax.set_xticks(range(len(DEFEND_POSITION_NAMES)), DEFEND_POSITION_NAMES, fontsize=7)
            ax.set_yticks(range(len(SKILL_NAMES)), SKILL_NAMES, fontsize=7)
    fig.suptitle("Win rate vs each fixed strategy (rows: its attack target, columns: its preparation)", fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"static_matrices.{ext}"), dpi=200)
    plt.close(fig)
    return os.path.join(out_dir, "static_matrices.png")


def main():
    import summarize_alpha_ablations as sa
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--profiles", nargs="+", choices=PROFILES, default=list(PROFILES))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    md = []
    for profile in args.profiles:
        md += [f"\n## {profile}\n", "### Table 1 — point win rates, mean ± sd over seeds\n", table1(args.root, profile),
               "\n" + sample_note(args.root, profile, "nash-p-argmax"),
               "\n### Head to head — α-potential vs self-play (point win rates)\n", head_to_head_table(args.root, profile),
               "\n### Table 2 — unilateral response gains (value units, ±1 per point)\n", table2(args.root, profile),
               "\n### Table 3 — self-play deployment\n", table3(args.root, profile),
               "\n### Value and potential error (γ 0.99, all training rallies; Φ ≡ 0 in parentheses)\n",
               model_quality_table(args.root, profile),
               "\n### Online computation\n", timing_table(args.root, profile)]
    sa.TABLE_DIR = "figevals"
    gamma_series, vol_series = {}, {}
    for profile in args.profiles:
        t, gamma_series[profile] = sa.table(sa.group(args.root, profile, "gamma"), "γ")
        md += [f"\n### Figure 3 data — discount factor ({profile})\n", t]
        t, vol_series[profile] = sa.table(sa.group(args.root, profile, "vol"), "train rallies")
        md += [f"\n### Figure 2 data — training rallies ({profile})\n", t]
        sa.write_csv(os.path.join(args.out_dir, f"fig3_gamma_{profile}.csv"), gamma_series[profile], "gamma")
        sa.write_csv(os.path.join(args.out_dir, f"fig2_volume_{profile}.csv"), vol_series[profile], "train_rallies")
    out = lambda name: os.path.join(args.out_dir, name)
    plot_curves(out("fig2_training_rallies"), vol_series, "training rallies M", WIN_SERIES, "point win rate",
                x_from_n_train=True, logx=True)
    plot_curves(out("fig3_discount_factor"), gamma_series, "discount factor γ", WIN_SERIES, "point win rate")
    plot_curves(out("figS_model_quality_volume"), vol_series, "training rallies M", MODEL_SERIES, "error",
                x_from_n_train=True, logx=True, pct=False)
    plot_matrices(args.root, args.out_dir, args.profiles)
    text = "\n".join(md)
    with open(os.path.join(args.out_dir, "paper_tables.md"), "w") as f:
        f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
