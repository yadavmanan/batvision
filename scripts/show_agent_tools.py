"""Shows whether the model really calls the OpenCV tools, running the agent in process.

    python scripts/show_agent_tools.py                  # live model; NEBIUS_API_KEY from batvision/.env
    python scripts/show_agent_tools.py --backend mock   # offline

It sends a sharp and a blurred sign, prints the spoken answer and every tool call, and exits 1 if no
tool was called at all.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from batvision.cloud.memory import SessionMemory  # noqa: E402
from batvision.cloud.navigator import Navigator  # noqa: E402
from batvision.cloud.vlm import make_backend  # noqa: E402
from batvision.common.schemas import NavigateRequest  # noqa: E402


def sign(blur: float = 0.0):
    img = np.full((480, 640, 3), 235, np.uint8)
    cv2.putText(img, "EXIT", (60, 150), cv2.FONT_HERSHEY_DUPLEX, 3.0, (20, 20, 20), 6)
    cv2.putText(img, "STAIRS LEFT", (60, 300), cv2.FONT_HERSHEY_DUPLEX, 2.0, (20, 20, 20), 4)
    return cv2.GaussianBlur(img, (0, 0), blur) if blur else img


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=None, help="nebius (default, live) or mock")
    ap.add_argument("--query", default="What does the sign say?")
    ap.add_argument("--image", help="send this photo instead of the generated signs")
    ap.add_argument("--timeout", type=float, default=90.0)
    ap.add_argument("--out", default=str(ROOT / "eval" / "live_traces.json"))
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    try:
        navigator = Navigator(backend=make_backend(args.backend, timeout_s=args.timeout), memory=SessionMemory(0))
    except (RuntimeError, ValueError) as exc:
        print(f"Cannot start the model backend: {exc}")
        return 2
    print(f"OpenCV {cv2.__version__} | model {navigator.model_id}\n")

    if args.image:
        photo = cv2.imread(args.image)
        if photo is None:
            print(f"Could not read {args.image}")
            return 2
        cases = [("your photo", photo)]
    else:
        cases = [("sharp sign", sign()), ("blurred sign", sign(blur=9))]

    saved, total_calls = [], 0
    for label, img in cases:
        body = {"device_id": "show-tools", "session_id": f"show-tools-{label}".replace(" ", "-"),
                "timestamp": time.time(), "query": args.query,
                "camera": {"width": 640, "height": 480, "hfov_deg": 70},
                "image_jpeg_b64": base64.b64encode(cv2.imencode(".jpg", img)[1].tobytes()).decode()}
        try:
            resp = navigator.navigate(NavigateRequest.from_dict(body), "show-tools").to_dict()
        except Exception as exc:  # provider error, auth, network
            print(f"--- {label}: FAILED {type(exc).__name__}: {exc}")
            return 2
        trace = resp.get("agent_trace") or []
        total_calls += len(trace)
        print(f"--- {label}")
        print(f"  answer         : {resp.get('speech')}")
        print(f"  needs re-aim   : {resp.get('needs_better_view')}")
        print(f"  model / total  : {resp.get('latency_ms')}")
        print(f"  tools_used     : {resp.get('tools_used')}")
        if not trace:
            print("  agent_trace    : EMPTY - no OpenCV tool was called")
        for step in trace:
            mark = "ok " if step.get("ok") else "ERR"
            print(f"  {step['step']}. [{mark}] {step['tool']:<22} {step.get('ms', 0):>6.1f} ms  "
                  f"opencv={step.get('opencv')}  image_back={step.get('returned_image')}")
            print(f"       args   {json.dumps(step.get('args'))}")
            print(f"       result {step.get('result', '')[:110]}")
        print()
        saved.append({"case": label, "response": resp})

    Path(args.out).write_text(json.dumps({"model": navigator.model_id, "opencv": cv2.__version__, "cases": saved},
                                         indent=2), encoding="utf-8")
    print(f"full responses saved to {args.out}")
    if total_calls == 0:
        print("RESULT: no tool calls. OpenCV 5 may be missing, or the model chose not to use tools.")
        return 1
    print(f"RESULT: {total_calls} OpenCV tool call(s) recorded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
