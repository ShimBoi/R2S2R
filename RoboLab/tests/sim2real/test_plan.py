"""Plan layering, registry and layouts; no simulator needed (pytest RoboLab/tests/sim2real/test_plan.py)."""
import json
import math

import pytest

from robolab.core.sim2real import load_plan, merge, register
from robolab.core.sim2real.layouts import preset_indices, sample_layouts
from robolab.core.sim2real.registry import SECTIONS

BASE = {
    "lights": {"sphere": {"intensity": [0.7, 1.3], "pos_m": 0.3}},
    "material": {"bread": {"static": [0.8, 1.4], "dynamic": [0.6, 1.2]},
                 "plate": {"static": [0.4, 0.8], "dynamic": [0.3, 0.7]}},
    "joints": {"waffle_maker": {"hinge": {"damping": [0.005, 0.02]}}},
}


def test_later_layers_override_key_by_key():
    plan = load_plan(BASE, {"lights": {"sphere": {"pos_m": 0.15}}, "material": {"bread": {"static": [1.0, 1.2]}}})
    assert plan["lights"]["sphere"] == {"intensity": [0.7, 1.3], "pos_m": 0.15}
    assert plan["material"]["bread"] == {"static": [1.0, 1.2], "dynamic": [0.6, 1.2]}
    assert plan["material"]["plate"] == BASE["material"]["plate"]


def test_enabled_false_drops_sections_and_entries():
    plan = load_plan(BASE, {"lights": {"enabled": False}, "material": {"plate": {"enabled": False}}})
    assert "lights" not in plan
    assert list(plan["material"]) == ["bread"]


def test_override_file_from_env(tmp_path, monkeypatch):
    p = tmp_path / "over.json"
    p.write_text(json.dumps({"joints": {"waffle_maker": {"hinge": {"damping": [0.01, 0.01]}}}}))
    monkeypatch.setenv("ROBOLAB_DR_OVERRIDE", str(p))
    assert load_plan(BASE)["joints"]["waffle_maker"]["hinge"]["damping"] == [0.01, 0.01]


def test_layers_do_not_mutate_inputs():
    before = json.dumps(BASE, sort_keys=True)
    merge(BASE, {"lights": {"sphere": {"pos_m": 0.0}}})
    assert json.dumps(BASE, sort_keys=True) == before


def test_register_refuses_silent_replacement():
    @register("test_section")
    def build(cfg):
        return {}

    with pytest.raises(KeyError):
        @register("test_section")
        def build2(cfg):
            return {}

    @register("test_section", replace=True)
    def build3(cfg):
        return {}

    assert SECTIONS.pop("test_section") is build3


def test_layouts_respect_keepouts_and_photo_preset():
    objects = {"bread": {"photo": (0.38, -0.14, -34), "x": (0.3, 0.72), "y": (-0.3, 0.3), "r": 0.085},
               "plate": {"photo": (0.69, 0.045, 0), "x": (0.4, 0.75), "y": (-0.25, 0.3), "r": 0.12},
               "mug": {"photo": (0.536, -0.083, -30), "jitter": 0.02, "r": 0.04}}
    keep = [(0.419, 0.148, 0.19)]
    presets = sample_layouts(50, objects, z=0.05, keepouts=keep, min_dist={("bread", "plate"): 0.24}, jitter_on=True)
    assert len(presets) == 50
    assert presets[0][0][:2] == (0.38, -0.14)
    for p in presets[1:]:
        (bx, by, *_), (px, py, *_), (mx, my, *_) = p
        assert math.hypot(bx - px, by - py) >= 0.24
        assert math.hypot(bx - keep[0][0], by - keep[0][1]) >= 0.19 + 0.085
        assert abs(mx - 0.536) <= 0.02 and abs(my + 0.083) <= 0.02


def test_layouts_without_jitter_keep_photo_poses():
    objects = {"mug": {"photo": (0.5, 0.0, 30), "jitter": 0.02, "r": 0.04},
               "bread": {"photo": (0.3, 0.2, 0), "x": (0.3, 0.6), "y": (-0.3, 0.3), "r": 0.05}}
    for p in sample_layouts(10, objects, z=0.05, jitter_on=False)[1:]:
        assert p[0][:2] == (0.5, 0.0)
        assert p[0][3] == pytest.approx(math.cos(math.radians(15)))


def test_preset_indices_switches(monkeypatch):
    monkeypatch.setenv("ROBOLAB_PHOTO_LAYOUT", "1")
    assert preset_indices(range(1, 5), range(5, 7)) == [0]
    monkeypatch.setenv("ROBOLAB_PHOTO_LAYOUT", "0")
    monkeypatch.setenv("ROBOLAB_EVAL_ONLY", "1")
    assert preset_indices(range(1, 5), range(5, 7)) == [5, 6]


def _task_terms():
    from isaaclab.managers import EventTermCfg

    from robolab.core.sim2real.fixtures import reset_fixture

    def noop(env, env_ids, **kw):
        pass

    return {
        "reset_waffle": EventTermCfg(func=reset_fixture, mode="reset",
                                     params={"name": "waffle_maker", "pose": (0.42, 0.15, 0.05, -14)}),
        "set_background": EventTermCfg(func=noop, mode="reset", params={"which": "task"}),
        "randomize_init_pose": EventTermCfg(func=noop, mode="reset", params={}),
    }


def test_build_events_orders_fixture_resets_last_and_merges_jitter(monkeypatch):
    pytest.importorskip("isaaclab")
    monkeypatch.delenv("ROBOLAB_DR", raising=False)
    from robolab.core.sim2real import build_events

    base = _task_terms()
    ev = build_events(base, {"fixtures": {"waffle_maker": {"pos_m": 0.015, "yaw_deg": 5}},
                             "joints": {"waffle_maker": {"hinge": {"damping": [0.005, 0.02]}}},
                             "custom": {"dr_noop": {"func": "os.path:join", "mode": "startup"}}})
    names = list(ev.__dict__)
    assert names.index("reset_waffle") > names.index("randomize_init_pose")
    assert names.index("reset_waffle") < names.index("dr_joint_gains_waffle_maker_0") < names.index("dr_noop")
    assert ev.reset_waffle.params["pos_m"] == 0.015 and ev.reset_waffle.params["pose"][3] == -14
    assert "pos_m" not in base["reset_waffle"].params


def test_build_events_dr_off_keeps_task_terms_only(monkeypatch):
    pytest.importorskip("isaaclab")
    monkeypatch.setenv("ROBOLAB_DR", "0")
    from robolab.core.sim2real import build_events

    ev = build_events(_task_terms(), {"fixtures": {"waffle_maker": {"pos_m": 0.015}}, "lights": {"sphere": {}}})
    assert set(ev.__dict__) == {"reset_waffle", "set_background", "randomize_init_pose"}
    assert "pos_m" not in ev.reset_waffle.params


def test_build_events_rejects_unknown_sections_and_fixtures(monkeypatch):
    pytest.importorskip("isaaclab")
    monkeypatch.delenv("ROBOLAB_DR", raising=False)
    from robolab.core.sim2real import build_events

    with pytest.raises(KeyError, match="not registered"):
        build_events(_task_terms(), {"lightz": {}})
    with pytest.raises(KeyError, match="no reset_fixture term"):
        build_events(_task_terms(), {"fixtures": {"cabinet": {"pos_m": 0.01}}})
