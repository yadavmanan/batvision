"""Navigator pipeline, session memory and the edge client, fully offline (mock or scripted backends)."""
import base64
import queue
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from batvision.cloud.memory import SessionMemory
from batvision.cloud.navigator import Navigator
from batvision.cloud.vlm import AgentResult, MockVLM
from batvision.common.schemas import ModelAnswer, NavigateRequest, ValidationError
from batvision.edge.cloud_client import CloudClient

JPEG = base64.b64encode(cv2.imencode(".jpg", np.zeros((48, 64, 3), np.uint8))[1].tobytes()).decode()


def req(**kw) -> NavigateRequest:
    d = {"device_id": "dev", "session_id": "s", "query": "Where is the red water bottle?", "image_jpeg_b64": JPEG,
         "camera": {"width": 640, "height": 480, "hfov_deg": 70},
         "depth_grid": {"rows": 1, "cols": 4, "kind": "metric_m", "values": [3, 3, 2.0, 2.2]}}
    d.update(kw)
    return NavigateRequest.from_dict(d)


def test_find_object_gets_grounded_guidance():
    resp = Navigator(backend=MockVLM()).navigate(req(), "r1")
    assert resp.speech.startswith("Turn ") and "step" in resp.speech
    assert resp.target.clock in (12, 1)
    assert resp.latency_ms["total"] >= 0 and resp.request_id == "r1"


def test_hallucinated_clear_path_is_overridden():
    hz = [{"zone": "CENTER", "level": "TORSO", "proximity": 0.8, "urgency": "CRITICAL", "distance_m": 0.7}]
    resp = Navigator(backend=MockVLM()).navigate(req(query="Can I walk forward?", hazards=hz))
    assert resp.safety_override is True
    assert resp.speech.startswith("Stop.")
    assert resp.raw_model_speech == "The path ahead is clear."


def test_invalid_requests_are_rejected_before_any_model_call():
    for bad in ({"query": ""}, {"image_jpeg_b64": ""}):
        with pytest.raises(ValidationError):
            req(**bad)


def test_backend_failure_propagates_so_the_edge_can_degrade():
    class Boom:
        model_id = "boom"

        def run(self, *a):
            raise RuntimeError("model down")

    with pytest.raises(RuntimeError, match="model down"):
        Navigator(backend=Boom()).navigate(req())


# -------------------------------------------------------------------------------- memory
class Recorder:
    model_id = "recorder"

    def __init__(self):
        self.histories = []

    def run(self, request, history):
        self.histories.append(history)
        return AgentResult(ModelAnswer(speech=f"answer to {request.query}"), "recorder", 1.0, ["final_answer"])


def test_follow_up_questions_get_earlier_turns_as_text_only():
    backend = Recorder()
    nav = Navigator(backend=backend)
    nav.navigate(req(query="What is this?"))
    nav.navigate(req(query="And the left one?"))
    assert backend.histories[0] == []
    assert backend.histories[1] == [{"role": "user", "content": "(Earlier) What is this?"},
                                    {"role": "assistant", "content": "answer to What is this?"}]


def test_ambient_frames_neither_read_nor_write_memory():
    backend = Recorder()
    nav = Navigator(backend=backend)
    nav.navigate(req(query="", mode="ambient"))
    nav.navigate(req(query="Hi"))
    assert backend.histories == [[], []]


def test_memory_is_bounded_per_session_and_expires():
    mem = SessionMemory(max_turns=2, ttl_hours=1.0)
    for i in range(3):
        mem.save("s", f"q{i}", f"a{i}")
    assert [m["content"] for m in mem.load("s")] == ["(Earlier) q1", "a1", "(Earlier) q2", "a2"]
    assert mem.load("other") == []
    mem._turns["s"].appendleft((time.time() - 7200, "old", "old"))
    assert all("old" not in m["content"] for m in mem.load("s"))
    assert SessionMemory(max_turns=0).load("s") == []


# ----------------------------------------------------------------------------- edge client
class Failing:
    def __init__(self, exc):
        self.exc = exc

    def navigate(self, request, request_id):
        raise self.exc


class HttpError(Exception):
    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


def _client(navigator, offline_after=2):
    results = queue.Queue()
    return CloudClient(navigator, results.put, offline_after), results


def test_client_returns_the_navigator_response_off_the_calling_thread():
    client, results = _client(Navigator(backend=MockVLM()))
    client.submit(req())
    res = results.get(timeout=5)
    assert res.error is None and res.response.speech.startswith("Turn ")
    assert client.online and res.round_trip_ms >= 0


def test_repeated_connection_failures_flip_the_client_offline_and_success_recovers():
    navigator = SimpleNamespace(navigate=Failing(ConnectionError("no network")).navigate)
    client, results = _client(navigator)
    for _ in range(2):
        client.submit(req())
        assert results.get(timeout=5).error.startswith("ConnectionError")
    assert client.online is False
    client.navigator = Navigator(backend=MockVLM())
    client.submit(req())
    assert results.get(timeout=5).error is None and client.online is True


def test_a_client_error_status_is_not_treated_as_lost_connectivity():
    client, results = _client(SimpleNamespace(navigate=Failing(HttpError(400)).navigate), offline_after=1)
    client.submit(req())
    assert "HTTP 400" in results.get(timeout=5).error
    assert client.online is True


def test_rate_limits_and_server_errors_count_as_connectivity_failures():
    client, results = _client(SimpleNamespace(navigate=Failing(HttpError(429)).navigate), offline_after=1)
    client.submit(req())
    results.get(timeout=5)
    assert client.online is False
