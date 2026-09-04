# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Manifest: persistent storage for a scene's discovered subtask tree.

Per PLAN.md section 2: a scene's tree is a flat list of *edge* records -- an edge list for the
tree, not a flat list of independent options. Each edge:

    precondition (exact parent node) --[predicate/predicate_args]--> produces_node (child node)

A **node** is the set of already-satisfied subtask ids (root = the empty set). Nodes are
represented as ``frozenset[str]`` in the Python API; on disk (and in edge records) they're
sorted ``list[str]`` for JSON-friendliness and stable diffs.

The one rule this module exists to enforce mechanically (PLAN.md section 0 is emphatic about
this): **exact-match preconditions, not subset-match**. ``edges_from(node)`` only ever returns
edges whose recorded ``precondition`` is exactly equal to ``node`` -- a superset state must not
match a subset precondition, even though the subset's requirements are technically satisfied.
The reason is the reset-distribution guarantee: a subtask's `reset_states_path` was collected
under its exact precondition; invoking it from a state with additional things also satisfied
is a genuine train/eval distribution mismatch, not just pedantry.

This is plain JSON-file-backed storage, not a database -- appropriate at the scale this is
built for (one scene, a handful to a few dozen edges).
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Iterable, Optional

Node = frozenset  # alias for readability; a node is a frozenset[str] of satisfied subtask ids

_REQUIRED_EDGE_FIELDS = (
    "id",
    "precondition",
    "predicate",
    "predicate_args",
    "instruction",
    "objects_involved",
    "produces_node",
)


def _canon_node(node: Optional[Iterable[str]]) -> tuple:
    """Canonical, hashable, order-independent form of a node."""
    if node is None:
        return ()
    return tuple(sorted(set(node)))


def _node_str_key(node: Optional[Iterable[str]]) -> str:
    """String key for a node, used for the on-disk reset-states-path-for-children map.

    Subtask ids are snake_case identifiers (enforced loosely by convention, not this module),
    so "|" is a safe separator -- it can't collide with a real id.
    """
    return "|".join(_canon_node(node))


class ManifestError(ValueError):
    """Raised on malformed edge records or manifest-integrity violations (e.g. duplicate id)."""


class Manifest:
    """JSON-file-backed store of one scene's edge list.

    Every mutating call re-persists the whole manifest to ``path`` (small enough at this scale
    that there's no reason to build incremental/streaming writes). A ``threading.Lock`` guards
    read-modify-write so this is at least safe to call from multiple threads in the same
    process; it does **not** attempt cross-process file locking.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._edges: list[dict[str, Any]] = []
        self._terminal_nodes: set[tuple] = set()
        self._reset_paths_for_children: dict[str, str] = {}
        if self.path.exists():
            self._load()

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def _load(self) -> None:
        data = json.loads(self.path.read_text())
        self._edges = list(data.get("edges", []))
        self._terminal_nodes = {
            _canon_node(n) for n in data.get("terminal_nodes", [])
        }
        self._reset_paths_for_children = dict(data.get("reset_paths_for_children", {}))

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "edges": self._edges,
            "terminal_nodes": [list(n) for n in sorted(self._terminal_nodes)],
            "reset_paths_for_children": self._reset_paths_for_children,
        }
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(data, indent=2, sort_keys=True))
        tmp_path.replace(self.path)  # atomic on POSIX

    # ------------------------------------------------------------------
    # writes
    # ------------------------------------------------------------------
    def add_edge(self, edge: dict[str, Any]) -> dict[str, Any]:
        """Append one edge record. Returns the stored (normalized) copy.

        Validates required fields and rejects duplicate ids. ``precondition`` and
        ``produces_node`` are stored as sorted lists (canonicalized) regardless of the order
        passed in, so ``edges_from``'s exact-match comparison is well-defined.
        """
        missing = [f for f in _REQUIRED_EDGE_FIELDS if f not in edge]
        if missing:
            raise ManifestError(f"edge record missing required field(s): {missing}")

        with self._lock:
            if any(e["id"] == edge["id"] for e in self._edges):
                raise ManifestError(f"duplicate edge id: {edge['id']!r}")

            stored = dict(edge)
            stored["precondition"] = list(_canon_node(edge["precondition"]))
            stored["produces_node"] = list(_canon_node(edge["produces_node"]))
            stored.setdefault("checkpoint", None)
            stored.setdefault("reset_states_path", None)
            self._edges.append(stored)
            self._save()
        return stored

    def mark_terminal(self, node: Optional[Iterable[str]]) -> None:
        """Record that Phase A found nothing further to propose from ``node`` (a tree leaf)."""
        with self._lock:
            self._terminal_nodes.add(_canon_node(node))
            self._save()

    def set_reset_states_path_for_children(
        self, node: Optional[Iterable[str]], path: Optional[str]
    ) -> None:
        """Record where edges whose ``precondition == node`` should reset from.

        Set once a stage reaches ``node`` and its end-states have been collected (see
        ``orchestrator.train_sequence``); read back by ``get_reset_states_path_for_children``
        when discovering/configuring ``node``'s children. ``path=None`` clears/leaves unset,
        meaning "fall back to the scene's default starting-state pool" (only really meaningful
        at the root).
        """
        with self._lock:
            key = _node_str_key(node)
            if path is None:
                self._reset_paths_for_children.pop(key, None)
            else:
                self._reset_paths_for_children[key] = path
            self._save()

    # ------------------------------------------------------------------
    # reads
    # ------------------------------------------------------------------
    def edges_from(self, node: Optional[Iterable[str]]) -> list[dict[str, Any]]:
        """Edges whose ``precondition`` is EXACTLY ``node`` -- not a subset match.

        This is the concrete enforcement of the exact-match rule (see module docstring / PLAN.md
        section 0): a node that is a strict superset of some edge's precondition does NOT match
        that edge, even though every individual requirement is technically satisfied.
        """
        target = _canon_node(node)
        return [
            dict(e) for e in self._edges if _canon_node(e["precondition"]) == target
        ]

    def get_producing_edge(self, node: Optional[Iterable[str]]) -> Optional[dict[str, Any]]:
        """The single edge whose ``produces_node`` exactly equals ``node``, or ``None``.

        ``None`` both for the root (nothing produces the empty node) and for any node that
        isn't in the manifest yet.
        """
        target = _canon_node(node)
        if not target:
            return None
        for e in self._edges:
            if _canon_node(e["produces_node"]) == target:
                return dict(e)
        return None

    def get_reset_states_path_for_children(
        self, node: Optional[Iterable[str]]
    ) -> Optional[str]:
        """Getter for ``set_reset_states_path_for_children``. ``None`` if never set."""
        return self._reset_paths_for_children.get(_node_str_key(node))

    def is_terminal(self, node: Optional[Iterable[str]]) -> bool:
        return _canon_node(node) in self._terminal_nodes

    def get_edge(self, edge_id: str) -> Optional[dict[str, Any]]:
        for e in self._edges:
            if e["id"] == edge_id:
                return dict(e)
        return None

    def all_edges(self) -> list[dict[str, Any]]:
        return [dict(e) for e in self._edges]

    def __len__(self) -> int:
        return len(self._edges)
