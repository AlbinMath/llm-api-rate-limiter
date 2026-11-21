import time
import threading
from collections import deque
from typing import Dict, Tuple, TypedDict, Deque, List

Key = Tuple[str, str]
Decision = Tuple[bool, int, int]


class StatsDict(TypedDict):
    strategy: str
    total_allowed: int
    total_rejected: int
    active_keys: int


class RateLimiter:
    def __init__(self, strategy: str = "policy_a"):
        if strategy not in ("policy_a", "policy_b"):
            raise ValueError(f"Unknown strategy: {strategy}")

        self.strategy = strategy
        self._lock = threading.Lock()

        # Generic per-key state store.
        # For policy_a: Key -> deque of timestamp floats
        # For policy_b: Key -> [current_tokens_float, last_refill_time_float]
        self._state: Dict[Key, object] = {}

        self._total_allowed = 0
        self._total_rejected = 0

    def check(
        self,
        client_id: str,
        endpoint: str,
        limit: int,
        window_ms: int,
        capacity: int,
        refill_rate: float,
    ) -> Decision:
        """
        Run the configured policy and return (allowed, remaining, reset_after_ms).

        Policy A uses: limit, window_ms
        Policy B uses: capacity, refill_rate

        remaining:
            Remaining capacity after the current decision.

        reset_after_ms:
            Milliseconds until capacity becomes available again.
        """
        with self._lock:
            if self.strategy == "policy_a":
                result = self.policy_a_limit(client_id, endpoint, limit, window_ms)
            else:
                result = self.policy_b_limit(client_id, endpoint, capacity, refill_rate)

            if result[0]:
                self._total_allowed += 1
            else:
                self._total_rejected += 1

            return result

    def policy_a_limit(
        self,
        client_id: str,
        endpoint: str,
        limit: int,
        window_ms: int,
    ) -> Decision:
        """
        Policy A:
        For each (client_id, endpoint), allow at most `limit` requests during
        the most recent `window_ms` milliseconds.
        """
        now = time.monotonic()
        window_sec = window_ms / 1000.0
        cutoff = now - window_sec

        key = (client_id, endpoint)
        if key not in self._state:
            self._state[key] = deque()

        timestamps: Deque[float] = self._state[key]  # type: ignore

        # Expired requests must be removed before evaluating the current request.
        while timestamps and timestamps[0] <= cutoff:
            timestamps.popleft()

        if len(timestamps) < limit:
            timestamps.append(now)
            remaining = limit - len(timestamps)
            oldest = timestamps[0]
            reset_after_ms = max(0, int((oldest + window_sec - now) * 1000))
            return True, remaining, reset_after_ms
        else:
            remaining = 0
            if timestamps:
                oldest = timestamps[0]
                reset_after_ms = max(0, int((oldest + window_sec - now) * 1000))
            else:
                reset_after_ms = 0
            return False, remaining, reset_after_ms

    def policy_b_limit(
        self,
        client_id: str,
        endpoint: str,
        capacity: int,
        refill_rate: float,
    ) -> Decision:
        """
        Policy B:
        For each (client_id, endpoint), maintain a request budget with maximum
        size `capacity`. The budget increases over time at `refill_rate` units
        per second, up to `capacity`. Each accepted request consumes 1 unit.
        """
        now = time.monotonic()
        key = (client_id, endpoint)

        if key not in self._state:
            self._state[key] = [float(capacity), now]

        bucket: List[float] = self._state[key]  # type: ignore
        tokens, last_time = bucket[0], bucket[1]

        elapsed = max(0.0, now - last_time)
        tokens = max(0.0, min(float(capacity), tokens + elapsed * refill_rate))
        bucket[1] = now

        if tokens >= 1.0:
            tokens -= 1.0
            bucket[0] = tokens
            remaining = int(tokens)
            reset_after_ms = 0
            return True, remaining, reset_after_ms
        else:
            bucket[0] = tokens
            remaining = 0
            if refill_rate > 0:
                needed = 1.0 - tokens
                reset_after_ms = max(0, int((needed / refill_rate) * 1000))
            else:
                reset_after_ms = 0
            return False, remaining, reset_after_ms

    def stats(self) -> StatsDict:
        """
        Return aggregate limiter statistics.

        active_keys counts tracked (client_id, endpoint) entries for the active
        policy state.
        """
        with self._lock:
            return {
                "strategy": self.strategy,
                "total_allowed": self._total_allowed,
                "total_rejected": self._total_rejected,
                "active_keys": len(self._state),
            }