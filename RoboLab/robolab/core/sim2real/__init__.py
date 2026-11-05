"""Sim-to-real building blocks for reconstructed scenes: a domain-randomization plan (plan.py) built from
registered sections (builders.py, registry.py) over generic randomizers (randomizers.py), articulated fixtures
(articulated_usd.py, fixtures.py) and start layouts (layouts.py). Scene-specific randomizers go in extras/."""
from robolab.core.sim2real.plan import build_events, dr_enabled, load_plan, merge
from robolab.core.sim2real.registry import register

__all__ = ["build_events", "dr_enabled", "load_plan", "merge", "register"]
