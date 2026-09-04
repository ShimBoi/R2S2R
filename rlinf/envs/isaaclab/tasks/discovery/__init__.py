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

"""VLM-discovered subtask tree: offline discovery + eval-time decomposition.

- :mod:`manifest` -- persistent storage for a scene's discovered edge tree.
- :mod:`predicates` -- predicate-menu extraction + Phase A candidate validation.
- :mod:`vlm_client` -- thin, injectable wrapper around whatever VLM API is configured.
- :mod:`phase_a` -- offline discovery prompt + VLM call.
- :mod:`phase_b` -- eval-time decomposition prompt + VLM call + ``resolve_plan``.
- :mod:`orchestrator` -- ``discover_tree`` / ``train_sequence`` skeleton.

No ``torch``/``isaaclab``/``robolab`` imports at module load time, so tests run under plain
``pytest`` without a container or GPU. Real VLM calls, subprocess training launches, and scene
screenshots are isolated behind lazily-imported, monkeypatchable call sites.
"""

from .manifest import Manifest

__all__ = ["Manifest"]
