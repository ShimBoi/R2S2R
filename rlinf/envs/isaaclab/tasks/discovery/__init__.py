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

"""Open-ended task discovery & decomposition (VLM-discovered subtask tree).

Implements the offline/online machinery described in ``RLinf/PLAN.md``:

- :mod:`manifest` -- persistent storage for a scene's discovered edge tree.
- :mod:`predicates` -- predicate-menu extraction + Phase A candidate validation.
- :mod:`vlm_client` -- thin, injectable wrapper around whatever VLM API is configured.
- :mod:`phase_a` -- offline discovery prompt + VLM call (PLAN.md section 3).
- :mod:`phase_b` -- eval-time decomposition prompt + VLM call + ``resolve_plan``
  (PLAN.md sections 5, 5.1).
- :mod:`orchestrator` -- ``discover_tree`` / ``train_sequence`` skeleton (PLAN.md section 4).

Deliberately import-light: nothing in this package imports ``torch``, ``isaaclab``, or
``robolab`` at module import time, so unit tests run under plain ``pytest`` (e.g. the repo's
``.venv``), no container / GPU required. Modules that *do* need those (real VLM network calls,
real subprocess training launches, a real scene screenshot) isolate that behind a single,
lazily-imported, monkeypatchable call site.

This package is intentionally separate from ``robolab_task.py`` in the parent directory (that
file, and everything under ``RoboLab/robolab/tasks/benchmark/``, is a different contributor's
territory for this extension -- see the coordinating CLAUDE.md).
"""

from .manifest import Manifest

__all__ = ["Manifest"]
