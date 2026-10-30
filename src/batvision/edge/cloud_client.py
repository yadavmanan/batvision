"""Edge -> reasoning client. Never blocks the reflex loop.

Requests run on a worker thread and call the in-process Navigator; results come back through a callback.
After ``offline_after_failures`` consecutive failures the client reports itself offline (the app tells the
user once) and keeps probing in the background, so physical safety never depends on connectivity.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Callable, Optional

from ..cloud.navigator import Navigator
from ..common.schemas import NavigateRequest, NavigateResponse

log = logging.getLogger(__name__)


@dataclass
class CloudResult:
    request: NavigateRequest
    response: Optional[NavigateResponse]
    error: Optional[str]
    sent_at: float
    round_trip_ms: float


ResultCallback = Callable[[CloudResult], None]


class CloudClient:
    def __init__(self, navigator: Navigator, on_result: ResultCallback, offline_after_failures: int = 3):
        self.navigator = navigator
        self.on_result = on_result
        self.offline_after = offline_after_failures
        self.failures = 0
        self.in_flight = False
        self._q: "queue.Queue[NavigateRequest]" = queue.Queue(maxsize=2)
        threading.Thread(target=self._run, name="cloud", daemon=True).start()

    @property
    def online(self) -> bool:
        return self.failures < self.offline_after

    def submit(self, req: NavigateRequest) -> bool:
        """Queue a request; drops the oldest if the user fires questions faster than the model answers."""
        try:
            self._q.put_nowait(req)
        except queue.Full:
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            self._q.put_nowait(req)
        return True

    def _run(self) -> None:
        while True:
            req = self._q.get()
            self.in_flight = True
            t0 = time.monotonic()
            resp, err = None, None
            try:
                resp = self.navigator.navigate(req, uuid.uuid4().hex)
                self.failures = 0
            except Exception as exc:  # timeout, DNS, no network, provider error
                err = f"{type(exc).__name__}: {exc}"
                status = getattr(exc, "status_code", None)
                # A 4xx (other than 429) is a request bug, not connectivity; don't flip to offline for it.
                if not (isinstance(status, int) and 400 <= status < 500 and status != 429):
                    self.failures += 1
            finally:
                self.in_flight = False
            rtt = (time.monotonic() - t0) * 1000
            if err:
                log.warning("reasoning request failed after %.0f ms: %s", rtt, err)
            self.on_result(CloudResult(req, resp, err, t0, rtt))
