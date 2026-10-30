"""Multi-turn conversational memory, held in process (text only, bounded, auto-expiring).

Only question/answer text is kept, never images. Turns older than `ttl_hours` are dropped, and each
session keeps at most `max_turns` turns. Nothing is written to disk, so it is gone when the app exits.
"""
from __future__ import annotations

import collections
import threading
import time
from typing import Any, Deque, Dict, List, Tuple


class SessionMemory:
    def __init__(self, max_turns: int = 4, ttl_hours: float = 12.0):
        self.max_turns = max(0, int(max_turns))
        self.ttl_s = float(ttl_hours) * 3600
        self._turns: Dict[str, Deque[Tuple[float, str, str]]] = collections.defaultdict(
            lambda: collections.deque(maxlen=max(1, self.max_turns)))
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.max_turns > 0

    def load(self, session_id: str) -> List[Dict[str, Any]]:
        """Previous turns as alternating user/assistant text messages (oldest first)."""
        if not self.enabled:
            return []
        cutoff = time.time() - self.ttl_s
        with self._lock:
            turns = [t for t in self._turns.get(session_id, ()) if t[0] >= cutoff]
        messages: List[Dict[str, Any]] = []
        for _, question, answer in turns:
            messages.append({"role": "user", "content": f"(Earlier) {question}"})
            messages.append({"role": "assistant", "content": answer})
        return messages

    def save(self, session_id: str, question: str, answer: str) -> None:
        if not self.enabled or not question or not answer:
            return
        with self._lock:
            self._turns[session_id].append((time.time(), question[:500], answer[:1000]))
