"""VLM backends for Hemisphere 2 (remote reasoning).

* OpenAICompatAgent - tool-calling vision agent for any OpenAI-compatible chat-completions endpoint.
* NebiusAgent        - NVIDIA Nemotron on Nebius Token Factory (VLM_BACKEND=nebius, the default).
  Another provider, such as NVIDIA's own cloud, is a subclass that only supplies its URL, key and model.
* MockVLM            - deterministic answers for tests and offline development (VLM_BACKEND=mock).
"""
from __future__ import annotations

import base64
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..common.prompts import SYSTEM_PROMPT, user_turn_text
from ..common.schemas import ModelAnswer, NavigateRequest
from .tools import agent_tools, run_tool
from .vision_tools import VISION_TOOL_NAMES, ToolResult, VisionToolbox

log = logging.getLogger(__name__)

MAX_AGENT_TURNS = 5
FALLBACK_SPEECH = "Sorry, I couldn't work that out. Please try again."


@dataclass
class AgentResult:
    answer: ModelAnswer
    model_id: str
    model_ms: float
    tools_used: List[str] = field(default_factory=list)
    stop_reason: str = ""
    trace: List[Dict[str, Any]] = field(default_factory=list)   # one entry per tool call, for observability


def _run_agent_tool(name: str, args: Dict[str, Any], req: NavigateRequest, toolbox: VisionToolbox) -> ToolResult:
    """Runs one tool call and records it in the trace. OpenCV tools may return an image for the model."""
    if name in VISION_TOOL_NAMES:
        return toolbox.call(name, args)
    started = time.perf_counter()
    content, is_error = run_tool(name, args, req)
    result = ToolResult(content, is_error)
    toolbox.record(name, args, result, started)
    return result


class _ToolLoopMixin:
    """Gives every tool-calling backend a per-request OpenCV toolbox and attaches its trace to the result."""

    def run(self, req: NavigateRequest, history: List[Dict[str, Any]]) -> AgentResult:
        toolbox = VisionToolbox(req)
        result = self._run_loop(req, history, toolbox)
        result.trace = toolbox.trace
        return result


class OpenAICompatAgent(_ToolLoopMixin):
    """Vision agent for an OpenAI-compatible chat-completions endpoint.

    Runs a short tool loop (OpenCV tools, transit) and ends when the model calls `final_answer`.
    """

    extra_body: Optional[Dict[str, Any]] = None   # provider-specific request options

    def __init__(self, model_id: str, base_url: str, api_key: str, timeout_s: float = 12.0):
        from openai import OpenAI

        self.model_id = model_id
        self.base_url = base_url
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s, max_retries=1)
        self._tools = [{"type": "function", "function": {
            "name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
            for t in agent_tools()]

    def _run_loop(self, req: NavigateRequest, history: List[Dict[str, Any]], toolbox: VisionToolbox) -> AgentResult:
        messages: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages += [{"role": m["role"], "content": m["content"]} for m in history]
        messages.append({"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{req.image_jpeg_b64}"}},
            {"type": "text", "text": user_turn_text(req)},
        ]})
        tools_used: List[str] = []
        t0 = time.perf_counter()
        stop_reason = ""
        for _ in range(MAX_AGENT_TURNS):
            resp = self.client.chat.completions.create(
                model=self.model_id, messages=messages, tools=self._tools, tool_choice="auto",
                max_tokens=1024, temperature=0.1, extra_body=self.extra_body,
            )
            choice = resp.choices[0]
            stop_reason = choice.finish_reason or ""
            msg = choice.message
            calls = list(msg.tool_calls or [])
            final = next((c for c in calls if c.function.name == "final_answer"), None)
            if final is not None:
                tools_used.append("final_answer")
                ans = ModelAnswer.from_tool_input(_parse_args(final.function.arguments))
                return AgentResult(ans, self.model_id, (time.perf_counter() - t0) * 1000, tools_used, stop_reason)
            if not calls:
                ans = ModelAnswer(speech=(msg.content or "").strip() or FALLBACK_SPEECH)
                return AgentResult(ans, self.model_id, (time.perf_counter() - t0) * 1000, tools_used, stop_reason)
            messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": [
                {"id": c.id, "type": "function", "function": {"name": c.function.name,
                                                              "arguments": c.function.arguments}} for c in calls]})
            images: List[Dict[str, Any]] = []
            for c in calls:  # every tool result goes back before the next model turn
                tools_used.append(c.function.name)
                out = _run_agent_tool(c.function.name, _parse_args(c.function.arguments), req, toolbox)
                messages.append({"role": "tool", "tool_call_id": c.id,
                                 "content": out.content if not out.is_error else f"ERROR: {out.content}"})
                if out.image_b64:   # tool messages are text-only here, so the image follows as a user turn
                    images += [{"type": "text", "text": f"Image returned by {c.function.name}:"},
                               {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{out.image_b64}"}}]
            if images:
                messages.append({"role": "user", "content": images})
        return AgentResult(ModelAnswer(speech=FALLBACK_SPEECH), self.model_id,
                           (time.perf_counter() - t0) * 1000, tools_used, stop_reason or "max_turns")


class NebiusAgent(OpenAICompatAgent):
    """NVIDIA Nemotron on Nebius Token Factory (OpenAI-compatible)."""

    DEFAULT_BASE_URL = "https://api.tokenfactory.us-north1.nebius.com/v1/"
    DEFAULT_MODEL = "nvidia/Nemotron-3.5-Super-VL"

    def __init__(self, model_id: Optional[str] = None, base_url: Optional[str] = None,
                 api_key: Optional[str] = None, timeout_s: float = 12.0, thinking: Optional[bool] = None):
        key = api_key or os.environ.get("NEBIUS_API_KEY")
        if not key:
            raise RuntimeError("NEBIUS_API_KEY is not set. Put it in batvision/.env or the environment.")
        super().__init__(
            model_id=model_id or os.environ.get("NEBIUS_MODEL_ID") or self.DEFAULT_MODEL,
            base_url=base_url or os.environ.get("NEBIUS_BASE_URL") or self.DEFAULT_BASE_URL,
            api_key=key,
            timeout_s=timeout_s,
        )
        if thinking is None:
            thinking = os.environ.get("NEBIUS_THINKING", "").lower() in ("1", "true", "yes")
        # Measured: with thinking on, 4 of 8 answers ran out of tokens and latency was about 3x higher.
        self.extra_body = None if thinking else {"chat_template_kwargs": {"enable_thinking": False}}


def _parse_args(raw: Optional[str]) -> Dict[str, Any]:
    try:
        val = json.loads(raw or "{}")
        return val if isinstance(val, dict) else {}
    except json.JSONDecodeError:
        return {}


class MockVLM:
    """Offline stand-in. It drives the real OpenCV tools, so what they measure decides its answer."""

    model_id = "mock"
    _QUALITY_WORDS = {"too_dark": "too dark", "overexposed": "overexposed", "blurry": "too blurry",
                      "low_contrast": "too washed out"}

    def run(self, req: NavigateRequest, history: List[Dict[str, Any]]) -> AgentResult:
        base64.b64decode(req.image_jpeg_b64[:64] + "==", validate=False)
        toolbox = VisionToolbox(req)
        tools_used: List[str] = []

        def use(name: str, **args: Any) -> Dict[str, Any]:
            tools_used.append(name)
            res = toolbox.call(name, args)
            return {} if res.is_error else json.loads(res.content)   # no OpenCV: behave as before

        q = req.query.lower()
        if "bottle" in q:
            bbox = [600, 450, 680, 620]
            confirmed = use("analyze_region_color", bbox=bbox, expected_color="red").get("matches_expected", True)
            speech = ("The red water bottle is on the table." if confirmed else
                      "I think the red water bottle is on the table, but I could not confirm its colour.")
            ans = ModelAnswer(speech=speech, target_label="red water bottle", target_bbox=bbox,
                              suggested_direction="slightly_right", confidence=0.8 if confirmed else 0.4)
        elif "clear" in q or "walk" in q:
            ans = ModelAnswer(speech="The path ahead is clear.", claims_path_clear=True)
        elif "sign" in q or "read" in q:
            quality = use("assess_image_quality")
            if quality and not quality.get("usable", True):
                why = self._QUALITY_WORDS.get(quality.get("verdict"), "unclear")
                ans = ModelAnswer(speech=f"The photo is {why} to read. Hold still, aim at the sign and ask again.",
                                  needs_better_view=True)
            else:
                regions = use("find_text_regions").get("regions") or []
                if regions:
                    use("zoom_region", bbox=regions[0]["bbox"])
                ans = ModelAnswer(speech="The sign says: Exit, stairs to the left.")
        else:
            ans = ModelAnswer(speech="You are in a room with a table ahead and a chair on your left.")
        return AgentResult(ans, self.model_id, 1.0, tools_used + ["final_answer"], "end_turn", trace=toolbox.trace)


def make_backend(name: Optional[str] = None, timeout_s: float = 12.0):
    name = (name or os.environ.get("VLM_BACKEND") or "nebius").lower()
    if name == "mock":
        return MockVLM()
    if name == "nebius":
        return NebiusAgent(timeout_s=timeout_s)
    raise ValueError(f"unknown VLM_BACKEND {name!r} (use 'nebius' or 'mock')")
