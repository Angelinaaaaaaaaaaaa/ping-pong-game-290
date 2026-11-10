"""scripts/summarize_paper_alpha.py: Table 1 / 2 / 3 rows from evaluation JSONs."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

import summarize_paper_alpha as sp


def _cells(static=0.7, **extra):
    cells = {k: {"win": static} for k in sp._static_labels()}
    cells["left/left"] = {"win": static - 0.2}
    cells.update({k: {"win": v} for k, v in extra.items()})
    return {"summary": {"cells": cells}}


def test_row_stats_reads_each_opponent_column():
    r = sp.row_stats(_cells(random=0.6, ibr=0.5, **{"nash-p-adaptive": 0.55, "selfplay:runs/x.pth": 0.3}))
    assert r["static_min"] == pytest.approx(0.5) and r["random"] == 0.6 and r["ibr"] == 0.5
    assert r["adaptive"] == 0.55 and r["vs_selfplay"] == 0.3
    assert r["static_mean"] == pytest.approx(0.7 - 0.2 / 15)


def test_tables_average_seeds(tmp_path):
    d = tmp_path / "aggressive" / "table1"
    d.mkdir(parents=True)
    for s in range(3):
        (d / f"nash-p-argmax_s{s}.json").write_text(json.dumps(_cells(random=0.5 + 0.1 * s)))
        (d / f"selfplay_s{s}.json").write_text(json.dumps(_cells(**{"nash-p-argmax": 0.6})))
    t1 = sp.table1(str(tmp_path), "aggressive")
    assert "α-potential (Eq. 12)" in t1 and "60.0 ± 10.0" in t1
    t3 = sp.table3(str(tmp_path), "aggressive")
    assert "Categorical sampling" in t3
    g = tmp_path / "aggressive" / "gains"
    g.mkdir()
    (g / "nash-p-argmax_s0.json").write_text(json.dumps({"summary": {
        "player1": {"gain": 0.2}, "player2": {"gain": 0.4}, "profile_regret": 0.4}}))
    assert "0.4" in sp.table2(str(tmp_path), "aggressive")


def test_no_choice_selfplay_rows_and_matrix_fallback(tmp_path):
    d = tmp_path / "aggressive" / "table1"
    d.mkdir(parents=True)
    (d / "selfplay_s0.json").write_text(json.dumps(_cells(**{"nash-p-argmax": 0.6})))
    assert sp.matrix_methods(str(tmp_path))[1][0] == "selfplay"  # falls back to the choice-input actor
    (d / "selfplay-nc_s0.json").write_text(json.dumps(_cells(**{"nash-p-argmax": 0.7})))
    assert sp.matrix_methods(str(tmp_path))[1][0] == "selfplay-nc"
    t1 = sp.table1(str(tmp_path), "aggressive")
    assert len([line for line in t1.splitlines() if line.startswith("| Self-play RL, π(x, role) (sampled)")]) == 1
    assert "π(x, role)" in sp.table3(str(tmp_path), "aggressive")


def test_table1_uses_rallies_played_to_the_end_where_replayed(tmp_path):
    d = tmp_path / "aggressive" / "table1"
    d.mkdir(parents=True)
    (d / "nash-p-argmax_s0.json").write_text(json.dumps(_cells(static=0.7)))  # left/left 0.5 is the minimum
    assert sp.played_out_cells(str(tmp_path), "aggressive", "nash-p-argmax", 0) == {}
    diag = tmp_path / "aggressive" / "diag_unfinished"
    diag.mkdir()
    (diag / "nash-p-argmax_s0.json").write_text(json.dumps({"summary": {"left/left": {"by_cutoff": {
        "2500": {"wins": 1, "losses": 9, "unfinished": 10}, "20000": {"wins": 8, "losses": 12, "unfinished": 0}}}}}))
    assert sp.played_out_cells(str(tmp_path), "aggressive", "nash-p-argmax", 0) == {"left/left": 0.4}
    t1 = sp.table1(str(tmp_path), "aggressive")
    row = [line for line in t1.splitlines() if line.startswith("| α-potential")][0]
    assert "| 40.0 |" in row  # the replayed cell (40%) is now the static minimum


def test_head_to_head_reads_alpha_from_the_selfplay_rows(tmp_path):
    d = tmp_path / "aggressive" / "table1"
    d.mkdir(parents=True)
    for s, w in enumerate((0.6, 0.7)):
        (d / f"selfplay-nc_s{s}.json").write_text(json.dumps(_cells(**{"nash-p-argmax": w})))
    t = sp.head_to_head_table(str(tmp_path), "aggressive")
    row = [line for line in t.splitlines() if "π(x, role) (sampled)" in line]
    assert row == ["| Self-play RL, π(x, role) (sampled) | 65.0 ± 7.1 | 35.0 ± 7.1 |"]
    assert "argmax" not in t.split("|---|---|---|")[1]  # no row without data


def test_plot_matrices_for_one_profile(tmp_path):
    pytest.importorskip("matplotlib")
    d = tmp_path / "root" / "aggressive" / "table1"
    d.mkdir(parents=True)
    (d / "nash-p-argmax_s0.json").write_text(json.dumps(_cells()))
    out = tmp_path / "out"
    out.mkdir()
    path = sp.plot_matrices(str(tmp_path / "root"), str(out), profiles=("aggressive",))
    assert path.endswith("static_matrices.png") and (out / "static_matrices.pdf").exists()
    assert sp.plot_matrices(str(tmp_path / "root"), str(out), profiles=("moderate_aggressive",)) is None


def test_model_quality_table_compares_learned_and_constant_potential(tmp_path):
    for s in range(2):
        d = tmp_path / "aggressive" / "phi" / f"gamma0.99_s{s}"
        d.mkdir(parents=True)
        (d / "heldout_attack_defend.json").write_text(json.dumps({
            "heldout_q1_mse": 0.3 + 0.1 * s, "heldout_q1_mse_predict_zero": 0.9, "n_train_rallies": 14000,
            "heldout_residuals": {"max_abs": 2.0, "p95_abs": 0.8, "median_abs": 0.2, "mean_abs": 0.3},
            "heldout_residuals_constant_phi": {"max_abs": 1.8, "p95_abs": 1.0, "median_abs": 0.3, "mean_abs": 0.4}}))
    t = sp.model_quality_table(str(tmp_path), "aggressive")
    assert "0.350 ± 0.071" in t and "0.900" in t and "0.800 (1.000)" in t and "2.000 (1.800)" in t


def test_timing_table_reads_decision_times(tmp_path):
    for sub, name, timing in (("table1", "nash-p-argmax", {"method_mean_s": 0.004}),
                              ("ibr_sim", "nash-p-argmax", {"method_mean_s": 0.004, "by_opponent": {
                                  "ibr-sim": {"method_mean_s": 0.004, "opponent_mean_s": 30.0}}})):
        d = tmp_path / "aggressive" / sub
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{name}_s0.json").write_text(json.dumps({"summary": {"cells": {}, "timing": timing}}))
    t = sp.timing_table(str(tmp_path), "aggressive")
    assert "| α-potential (Eq. 12) | 4.0 |" in t and "30.0 s" in t


def _stats(mean, n_train=100):
    """One row's stats in summarize_alpha_ablations.table format: key -> (mean, sd, n)."""
    out = {k: (mean, 0.1, 3) for k in ("static_mean", "static_min", "static_mean_half", "static_min_half",
                                       "random", "heldout_q1_mse", "alpha_p95",
                                       "alpha_mean", "alpha_mean_constant")}
    out["n_train_rallies"] = (float(n_train), 0.0, 3)
    return out


def test_curve_sorts_by_x_and_can_use_training_rally_count():
    rows = [(5500.0, _stats(0.7, n_train=5500)), (170.0, _stats(0.5, n_train=170)),
            (99.0, {**_stats(0.6, n_train=14286), "static_min": (float("nan"), float("nan"), 0)})]
    xs, m, s = sp.curve(rows, "static_mean", x_from_n_train=True)
    assert list(xs) == [170, 5500, 14286] and list(m) == [0.5, 0.7, 0.6]
    xs, _m, _s = sp.curve(rows, "static_min")
    assert list(xs) == [170.0, 5500.0]  # NaN means are skipped
    assert sp.curve([], "random") is None


def test_plot_curves_writes_png_and_pdf(tmp_path):
    pytest.importorskip("matplotlib")
    series = {"aggressive": [(0.5, _stats(0.6)), (0.99, _stats(0.7))], "moderate_aggressive": []}
    path = sp.plot_curves(str(tmp_path / "fig3"), series, "discount factor γ", sp.WIN_SERIES, "win rate")
    assert (tmp_path / "fig3.png").exists() and (tmp_path / "fig3.pdf").exists() and path.endswith(".png")
    assert sp.plot_curves(str(tmp_path / "empty"), {"aggressive": []}, "γ", sp.WIN_SERIES, "win rate") is None


def test_sample_note_counts_decided_rallies_and_names_stalled_cells(tmp_path):
    d = tmp_path / "aggressive" / "table1"
    d.mkdir(parents=True)
    cells = {"random": {"decided": 200, "truncated": 0}, "ibr": {"decided": 200, "truncated": 2},
             "center_safe/center": {"decided": 140, "truncated": 960},
             "selfplay:runs_paper/selfplay/x.pth": {"decided": 200, "truncated": 100}}
    for s in range(2):
        (d / f"nash-p-argmax_s{s}.json").write_text(json.dumps({"summary": {"cells": cells}}))
    note = sp.sample_note(str(tmp_path), "aggressive", "nash-p-argmax")
    assert "2 seeds" in note and "200 decided rallies" in note
    assert "center_safe/center (87% of attempts)" in note and "self-play (33% of attempts)" in note
    assert "ibr" not in note.split("re-drawn")[1]  # 1% stalls are not listed
    assert sp.sample_note(str(tmp_path), "aggressive", "missing") == ""
