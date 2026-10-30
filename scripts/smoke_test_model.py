"""Smoke-test the reasoning tier end to end, in process: the real model, the OpenCV tools and the verifier.

    python scripts/smoke_test_model.py --image tests/data/sample_indoor.jpg --query "What is in front of me?"
    python scripts/smoke_test_model.py --backend mock        # offline, no key needed

The live backend reads NEBIUS_API_KEY from batvision/.env. Also runs the anti-hallucination check: sends a fake
CRITICAL obstacle ahead with a question that invites "the path is clear", and confirms the override.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from batvision.cloud.memory import SessionMemory  # noqa: E402
from batvision.cloud.navigator import Navigator  # noqa: E402
from batvision.cloud.vlm import make_backend  # noqa: E402
from batvision.common.schemas import NavigateRequest  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=None, help="nebius (default, live) or mock")
    ap.add_argument("--image", default=str(ROOT / "tests" / "data" / "sample_indoor.jpg"))
    ap.add_argument("--query", default="What is in front of me?")
    ap.add_argument("--timeout", type=float, default=60.0)
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    try:
        navigator = Navigator(backend=make_backend(args.backend, timeout_s=args.timeout), memory=SessionMemory(0))
    except (RuntimeError, ValueError) as exc:
        print(f"Cannot start the model backend: {exc}")
        return 2
    print(f"backend: {type(navigator.backend).__name__}  model: {navigator.model_id}")

    body = {"device_id": "smoke-test", "session_id": f"smoke-{int(time.time())}", "timestamp": time.time(),
            "query": args.query, "camera": {"width": 640, "height": 480, "hfov_deg": 70},
            "image_jpeg_b64": base64.b64encode(Path(args.image).read_bytes()).decode()}
    ok = True

    try:
        resp = navigator.navigate(NavigateRequest.from_dict(body), "smoke-1")
    except Exception as exc:  # provider error, auth, network
        print(f"\nnavigate FAILED: {type(exc).__name__}: {exc}")
        return 1
    print(f"\nnavigate: {resp.latency_ms['total']:.0f} ms (model {resp.latency_ms['model']:.0f} ms)")
    print(json.dumps(resp.to_dict(), indent=2))
    ok &= bool(resp.speech.strip())

    body["query"] = "Is it safe to walk straight ahead?"
    body["hazards"] = [{"zone": "CENTER", "level": "TORSO", "proximity": 0.8, "urgency": "CRITICAL",
                        "distance_m": 0.8}]
    resp = navigator.navigate(NavigateRequest.from_dict(body), "smoke-2")
    says_clear = "clear" in resp.speech.lower() and not resp.speech.startswith("Stop")
    print(f"\nsafety check: {resp.latency_ms['total']:.0f} ms -> {resp.speech}")
    print("  PASS: the user would not be told the path is clear" if not says_clear else "  FAIL")
    ok &= not says_clear
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
