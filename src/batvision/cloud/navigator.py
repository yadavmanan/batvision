"""Hemisphere 2 pipeline, run in process on the device.

Per request: load session memory -> VLM agent (NVIDIA Nemotron on Nebius Token Factory) -> ground the
target with camera geometry and edge depth -> safety verifier -> save memory.

Only the model call leaves the machine. The verifier runs here and again on the device right before speech.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

from ..common.schemas import NavigateRequest, NavigateResponse
from ..common.spatial import guidance_sentence, locate_target
from ..common.verifier import SafetyVerifier
from .memory import SessionMemory
from .vlm import make_backend

log = logging.getLogger(__name__)


class Navigator:
    def __init__(self, backend: Optional[Any] = None, memory: Optional[SessionMemory] = None,
                 timeout_s: float = 12.0):
        self.backend = backend if backend is not None else make_backend(timeout_s=timeout_s)
        self.memory = memory if memory is not None else SessionMemory()
        self.verifier = SafetyVerifier()

    @property
    def model_id(self) -> str:
        return getattr(self.backend, "model_id", "")

    def navigate(self, req: NavigateRequest, request_id: str = "") -> NavigateResponse:
        t0 = time.perf_counter()
        history = self.memory.load(req.session_id) if req.mode == "query" else []
        result = self.backend.run(req, history)
        log.info("agent_trace request_id=%s tools=%s trace=%s", request_id, result.tools_used,
                 json.dumps(result.trace, separators=(",", ":")))
        answer = result.answer
        raw_speech = answer.speech

        target = None
        if answer.target_bbox and answer.target_label:
            target = locate_target(answer.target_label, answer.target_bbox, req.camera.hfov_deg, req.depth_grid)
            if target is not None and answer.speech:
                # Deterministic turn/steps first (from geometry + edge depth), then the model's description.
                answer.speech = f"{guidance_sentence(target)} {answer.speech}".strip()

        verified = self.verifier.verify(answer, req.hazards, req.threats, target)
        total_ms = (time.perf_counter() - t0) * 1000
        resp = NavigateResponse(
            speech=verified.speech if answer.speech else "",
            raw_model_speech=raw_speech,
            target=target,
            safety_override=verified.overridden,
            override_reasons=verified.reasons,
            needs_better_view=answer.needs_better_view,
            tools_used=result.tools_used,
            agent_trace=result.trace,
            model_id=result.model_id,
            latency_ms={"model": round(result.model_ms, 1), "total": round(total_ms, 1)},
            request_id=request_id,
        )
        if req.mode == "query":
            self.memory.save(req.session_id, req.query, resp.speech)
        log.info("navigate request_id=%s model=%s model_ms=%.0f total_ms=%.0f override=%s",
                 request_id, result.model_id, result.model_ms, total_ms, verified.overridden)
        return resp
