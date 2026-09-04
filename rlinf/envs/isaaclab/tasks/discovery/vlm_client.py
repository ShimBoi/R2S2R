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

"""Thin, injectable wrapper around whatever VLM/LLM API is actually configured.

Backend (Anthropic or OpenAI) is chosen by whichever API key is set, Anthropic preferred if both
are. Both SDK imports are lazy (inside the call functions) so importing this module never
requires either package installed.

``_raw_call`` is the only function that makes a real network call. ``call_vlm_phase_a``/
``call_vlm_phase_b`` take a ``client_fn`` parameter defaulting to it, swappable for a fake in
tests.

Cost-sensitive: callers should avoid making VLM calls speculatively -- each
``discover_tree()``/``resolve_plan()`` run should make only the calls its own logic requires.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Callable, Optional


class VLMCallError(RuntimeError):
    """Raised when the VLM call fails, or its response can't be parsed as JSON."""


@dataclass
class VLMResponse:
    raw_text: str
    parsed: object  # json-decoded response body


def _select_backend() -> str:
    """Which backend to use, decided at call time (not import time) from env vars."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    raise VLMCallError(
        "No VLM backend configured: set ANTHROPIC_API_KEY or OPENAI_API_KEY. "
        "(This is only checked when a real call is actually attempted -- importing this "
        "module, or calling call_vlm_phase_a/b with an injected client_fn, never needs either.)"
    )


def _raw_call_anthropic(prompt: str, image_b64: Optional[str], model: str) -> str:
    import anthropic  # lazy: only needed on the real-network-call path

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    content: list[dict] = []
    if image_b64:
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": image_b64,
                },
            }
        )
    content.append({"type": "text", "text": prompt})
    resp = client.messages.create(
        model=model,
        max_tokens=2048,
        messages=[{"role": "user", "content": content}],
    )
    return "".join(
        block.text for block in resp.content if getattr(block, "type", None) == "text"
    )


def _raw_call_openai(prompt: str, image_b64: Optional[str], model: str) -> str:
    from openai import OpenAI  # lazy

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    content: list[dict] = [{"type": "text", "text": prompt}]
    if image_b64:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_b64}"},
            }
        )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
    )
    return resp.choices[0].message.content


_DEFAULT_MODELS = {"anthropic": "claude-sonnet-4-5", "openai": "gpt-4o"}


def _raw_call(
    prompt: str, image_b64: Optional[str] = None, model: Optional[str] = None
) -> str:
    """The one function that actually hits the network. Monkeypatch/inject this away in tests."""
    backend = _select_backend()
    resolved_model = model or _DEFAULT_MODELS[backend]
    if backend == "anthropic":
        return _raw_call_anthropic(prompt, image_b64, resolved_model)
    return _raw_call_openai(prompt, image_b64, resolved_model)


def call_vlm(
    prompt: str,
    *,
    image_b64: Optional[str] = None,
    model: Optional[str] = None,
    client_fn: Callable[..., str] = _raw_call,
) -> VLMResponse:
    """Call the VLM and parse its response as JSON.

    ``client_fn(prompt, image_b64=..., model=...) -> str`` is the injection point. Production
    callers leave it as the default (``_raw_call``, real network call, backend auto-selected
    from env vars). Tests pass something like
    ``lambda prompt, image_b64=None, model=None: json.dumps([...])``.
    """
    raw_text = client_fn(prompt, image_b64=image_b64, model=model)
    text = raw_text.strip()
    # Tolerate a ```json ... ``` fenced block despite "return ONLY JSON" -- extract via regex
    # search rather than strip("`"), since some models append prose after the closing fence.
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence_match:
        text = fence_match.group(1).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise VLMCallError(f"VLM response was not valid JSON: {raw_text!r}") from exc
    return VLMResponse(raw_text=raw_text, parsed=parsed)
