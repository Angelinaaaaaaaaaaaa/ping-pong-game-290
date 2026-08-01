import json
import pickle

import numpy as np

import scripts.collect_mixed_v2_dataset as mixed
from nash_skills.skills import SKILL_NAMES


def _rally(skill1="left", skill2="right"):
    return {
        "skill1": skill1,
        "skill2": skill2,
        "states": [np.zeros(76, dtype=np.float32)],
        "raw_obs": [np.zeros(116, dtype=np.float32)],
        "winner": 1,
        "skill_pairs": [(skill1, skill2)],
    }


def _install_fake_collect(monkeypatch, extra_discard=False):
    calls = []

    def fake_collect(**kwargs):
        calls.append(kwargs)
        target = kwargs["target_rallies"]
        mode = kwargs["mode"]
        setting = "random_vs_random" if mode == "random" else f"p{kwargs['fixed_player']}_fixed_{kwargs['fixed_skill']}"
        print("Returned successfully by ego 1.0 0.0")
        print("collector warning kept")
        attempts = []
        if extra_discard:
            attempts.append({
                "attempt_id": 0,
                "mode": mode,
                "setting": setting,
                "accepted": False,
                "truncated": True,
                "discard_reason": "discarded-step-cap",
                "steps": 5,
                "net_crossings": 0,
                "recorded_states": 0,
                "winner": 0,
            })
        for i in range(target):
            accepted = i + 1
            if kwargs.get("accepted_progress_callback") is not None:
                kwargs["accepted_progress_callback"]({
                    "mode": mode,
                    "setting": setting,
                    "display_name": f"P{kwargs.get('fixed_player', 1)} fixed {kwargs.get('fixed_skill', 'left')}",
                    "target": target,
                    "accepted": accepted,
                    "attempts": accepted + int(extra_discard),
                    "discarded": int(extra_discard),
                    "elapsed": max(1.0, float(accepted)),
                    "steps_accepted_total": accepted * 10,
                    "net_crossings_accepted_total": accepted,
                })
            attempts.append({
                "attempt_id": len(attempts),
                "mode": mode,
                "setting": setting,
                "accepted": True,
                "truncated": False,
                "discard_reason": "",
                "steps": 10,
                "net_crossings": 1,
                "recorded_states": 1,
                "winner": 1,
            })
        rallies = [_rally() for _ in range(target)]
        output = kwargs["output_path"]
        with open(output, "wb") as f:
            pickle.dump(rallies, f)
        _csv_path, json_path = mixed.collect_data.metadata_paths(output)
        json_path.write_text(json.dumps({"summary": {}, "attempts": attempts}))
        return rallies

    monkeypatch.setattr(mixed.collect_data, "collect", fake_collect)
    return calls


def test_default_style_allocation_is_7000_3000_split():
    plan = mixed.allocate_fixed_random(3000)

    assert len(plan) == 10
    assert sum(item["target"] for item in plan) == 3000
    assert {item["target"] for item in plan} == {300}
    assert [item["fixed_player"] for item in plan[:5]] == [1] * 5
    assert [item["fixed_player"] for item in plan[5:]] == [2] * 5


def test_remainder_distribution_is_deterministic():
    plan = mixed.allocate_fixed_random(13)

    assert [item["target"] for item in plan] == [2, 2, 2, 1, 1, 1, 1, 1, 1, 1]
    assert [item["fixed_skill"] for item in plan[:3]] == SKILL_NAMES[:3]


def test_seed_derivation_is_deterministic_and_distinct():
    first = mixed.derived_seed(7, "fixed_random", 3)
    second = mixed.derived_seed(7, "fixed_random", 3)

    assert first == second
    assert first != mixed.derived_seed(7, "fixed_random", 4)
    assert first != mixed.derived_seed(7, "random", 0)


def test_combined_pickle_integrity_and_collector_delegation(tmp_path, monkeypatch):
    calls = _install_fake_collect(monkeypatch)
    output = tmp_path / "mixed.pkl"
    args = mixed.build_parser().parse_args([
        "--random-rallies", "2",
        "--fixed-random-rallies", "10",
        "--output", str(output),
        "--progress-every", "99",
    ])

    rallies = mixed.run(args)

    assert len(rallies) == 12
    saved = pickle.load(output.open("rb"))
    assert len(saved) == 12
    assert all(set(rally) == {"states", "raw_obs", "winner", "skill_pairs", "skill1", "skill2"} for rally in saved)
    assert all(len(rally["states"]) == len(rally["skill_pairs"]) for rally in saved)
    assert all(rally["winner"] in (1, 2) for rally in saved)
    assert len(calls) == 11
    assert all(call["quiet"] is True for call in calls)
    assert all(call["progress_every"] == 0 for call in calls)
    assert not any("env.step" in name for call in calls for name in call)
    metadata = json.loads((tmp_path / "mixed_metadata.json").read_text())
    assert len(metadata["attempts"]) == 12
    assert metadata["summary"]["random_accepted"] == 2
    assert metadata["summary"]["fixed_random_accepted"] == 10


def test_progress_output_at_requested_interval(tmp_path, monkeypatch, capsys):
    _install_fake_collect(monkeypatch)
    output = tmp_path / "mixed.pkl"
    args = mixed.build_parser().parse_args([
        "--random-rallies", "3",
        "--fixed-random-rallies", "0",
        "--output", str(output),
        "--progress-every", "2",
    ])

    mixed.run(args)

    out = capsys.readouterr().out
    assert "[random_vs_random] accepted=2/3" in out
    assert "[random_vs_random] accepted=3/3" in out
    assert "[random_vs_random] accepted=1/3" not in out


def test_return_messages_suppressed_by_default(tmp_path, monkeypatch, capsys):
    _install_fake_collect(monkeypatch)
    output = tmp_path / "mixed.pkl"
    args = mixed.build_parser().parse_args([
        "--random-rallies", "1",
        "--fixed-random-rallies", "0",
        "--output", str(output),
    ])

    mixed.run(args)

    out = capsys.readouterr().out
    assert "Returned successfully by ego" not in out
    assert "collector warning kept" in out


def test_verbose_returns_restores_return_messages(tmp_path, monkeypatch, capsys):
    _install_fake_collect(monkeypatch)
    output = tmp_path / "mixed.pkl"
    args = mixed.build_parser().parse_args([
        "--random-rallies", "1",
        "--fixed-random-rallies", "0",
        "--output", str(output),
        "--verbose-returns",
    ])

    mixed.run(args)

    assert "Returned successfully by ego" in capsys.readouterr().out


def test_script_does_not_reimplement_rollout_logic():
    source = mixed.Path(mixed.__file__).read_text()

    assert "env.step(" not in source
    assert "model.predict(" not in source
