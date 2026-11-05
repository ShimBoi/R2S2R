"""Load a randomization plan from layers and turn it into a task's event terms.

    events = build_events(base_terms, "lab4_scene/dr.json", {"lights": {"enabled": False}})

Layers (paths, relative to ASSET_DIR or absolute, or dicts) merge left to right, key by key, and the file
named by ROBOLAB_DR_OVERRIDE merges last. {"enabled": false} drops a section or an entry. ROBOLAB_DR=0 turns
the plan off: the task's own terms are returned, fixtures reset to their nominal poses.

Besides the builder sections (builders.py, registry.py) a plan has:
 "extras": ["package.module", ...]   imported first, so they can register sections
 "fixtures": {"<fixture>": {"pos_m": 0.015, "yaw_deg": 5, "joint_pos": {"door": [0, 0.3]}}}
     merged into the task's reset_fixture / place_rigid_fixture term for that fixture
Fixture resets run after every other task reset term (reset_pose_to_presets puts unlisted rigid objects back
to their default poses), then the plan's terms run in section order.
"""
import copy
import importlib
import json
import os

_OFF = {"enabled": False}


def merge(base, over):
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and v.get("enabled") is False:
            out.pop(k, None)
        elif isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _strip(d):
    return {k: _strip(v) if isinstance(v, dict) else v for k, v in d.items() if k != "enabled"}


def _read(layer):
    if isinstance(layer, dict):
        return layer
    path = layer
    if not os.path.isabs(path):
        from robolab.constants import ASSET_DIR

        path = os.path.join(ASSET_DIR, path)
    with open(path) as f:
        return json.load(f)


def load_plan(*layers):
    plan = {}
    extra = os.environ.get("ROBOLAB_DR_OVERRIDE")
    for layer in [*layers, *([extra] if extra else [])]:
        if layer is not None:
            plan = merge(plan, _read(layer))
    return _strip(plan)


def dr_enabled():
    return os.environ.get("ROBOLAB_DR", "1") != "0"


def _fixture_name(term):
    p = term.params or {}
    return p.get("name") or (p.get("entities") or [None])[0]


def _is_fixture_reset(term):
    from robolab.core.sim2real.fixtures import reset_fixture
    from robolab.core.sim2real.randomizers import place_rigid_fixture

    return term.func in (reset_fixture, place_rigid_fixture)


def build_events(base_terms, *layers):
    """The task's terms (dict) with the plan applied, as an events configclass instance."""
    from isaaclab.managers import EventTermCfg
    from isaaclab.utils import configclass

    import robolab.core.sim2real.builders  # noqa: F401
    from robolab.core.sim2real.registry import SECTIONS

    fixtures = {k: copy.copy(v) for k, v in base_terms.items() if _is_fixture_reset(v)}
    terms = {**{k: v for k, v in base_terms.items() if k not in fixtures}, **fixtures}
    if dr_enabled():
        plan = load_plan(*layers)
        for mod in plan.get("extras", []):
            importlib.import_module(mod)
        by_fixture = {_fixture_name(t): k for k, t in fixtures.items()}
        for name, jitter in plan.get("fixtures", {}).items():
            if name not in by_fixture:
                raise KeyError(f"plan fixtures: no reset_fixture term for {name!r} (have {sorted(by_fixture)})")
            t = terms[by_fixture[name]]
            t.params = {**t.params, **jitter}
        unknown = set(plan) - set(SECTIONS) - {"extras", "fixtures"}
        if unknown:
            raise KeyError(f"plan sections {sorted(unknown)} are not registered (have {sorted(SECTIONS)})")
        for section, builder in SECTIONS.items():
            if plan.get(section):
                terms.update(builder(plan[section]))
    cls = type("Sim2RealEvents", (), {"__annotations__": {k: EventTermCfg for k in terms}, **terms})
    return configclass(cls)()
