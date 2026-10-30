"""Evaluate the cloud VLM on VizWiz-VQA validation photos (real blind-user photos).

    python scripts/prepare_vizwiz.py --images val
    python eval/eval_vizwiz_vlm.py --limit 100                 # live Nemotron; NEBIUS_API_KEY from batvision/.env
    python eval/eval_vizwiz_vlm.py --limit 100 --nav-only      # only navigation-relevant questions

Each sample is one paid model call - start with a small --limit.

Metrics
  * accuracy     VizWiz VQA accuracy, min(1, #humans agreeing / 3). Because our
                 answers are spoken sentences, a human answer "agrees" if its
                 normalized text appears in the normalized speech (lenient).
  * answerability  precision/recall of needs_better_view against VizWiz's
                 "unanswerable" labels - this is the false-alarm vs. missed-
                 warning trade-off for "please re-aim the camera".
  * latency      p50 / p90 round trip.
"""
from __future__ import annotations

import argparse
import base64
import json
import random
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
DATA = ROOT / "data" / "vizwiz"

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = re.compile(r"[^\w\s]")


def norm(s: str) -> str:
    return " ".join(_ARTICLES.sub(" ", _PUNCT.sub(" ", s.lower())).split())


def vqa_accuracy(speech: str, answers) -> float:
    sp = f" {norm(speech)} "
    hits = sum(1 for a in answers if (n := norm(a.get("answer", ""))) and len(n) > 1 and f" {n} " in sp)
    return min(1.0, hits / 3.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--nav-only", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from batvision.cloud.vlm import make_backend
    from batvision.common.schemas import NavigateRequest

    recs = [json.loads(l) for l in open(DATA / "vizwiz_val.jsonl", encoding="utf-8")]
    if args.nav_only:
        recs = [r for r in recs if r["nav_relevant"]]
    random.Random(args.seed).shuffle(recs)
    recs = [r for r in recs if (DATA / r["image"]).exists()][: args.limit]
    if not recs:
        raise SystemExit("no images found - run scripts/prepare_vizwiz.py --images val")

    load_dotenv(ROOT / ".env")
    agent = make_backend(timeout_s=60.0)

    rows, lat = [], []
    for i, r in enumerate(recs, 1):
        img = base64.b64encode((DATA / r["image"]).read_bytes()).decode()
        t0 = time.perf_counter()
        try:
            req = NavigateRequest.from_dict({"device_id": "eval", "query": r["question"], "image_jpeg_b64": img})
            res = agent.run(req, [])
            speech, better = res.answer.speech, res.answer.needs_better_view
        except Exception as exc:
            print(f"  [{i}] error: {exc}")
            continue
        lat.append((time.perf_counter() - t0) * 1000)
        acc = vqa_accuracy(speech, r["answers"]) if r["answerable"] else None
        rows.append({"image": r["image"], "question": r["question"], "speech": speech, "needs_better_view": better,
                     "answerable": r["answerable"], "accuracy": acc})
        print(f"  [{i}/{len(recs)}] {r['question'][:50]!r} -> {speech[:70]!r} acc={acc}")

    answerable = [x["accuracy"] for x in rows if x["accuracy"] is not None]
    tp = sum(1 for x in rows if x["needs_better_view"] and not x["answerable"])
    fp = sum(1 for x in rows if x["needs_better_view"] and x["answerable"])
    fn = sum(1 for x in rows if not x["needs_better_view"] and not x["answerable"])
    lat.sort()
    summary = {
        "n": len(rows),
        "accuracy_lenient": round(sum(answerable) / max(1, len(answerable)), 3),
        "unanswerable_precision": round(tp / max(1, tp + fp), 3),
        "unanswerable_recall": round(tp / max(1, tp + fn), 3),
        "latency_p50_ms": round(lat[len(lat) // 2]) if lat else None,
        "latency_p90_ms": round(lat[int(len(lat) * 0.9)]) if lat else None,
    }
    out = ROOT / "eval" / f"vizwiz_results_{int(time.time())}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print("\n", json.dumps(summary, indent=2), f"\nsaved {out}")


if __name__ == "__main__":
    main()
