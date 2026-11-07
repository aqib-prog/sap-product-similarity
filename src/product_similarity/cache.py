"""A small cache for search answers that lives inside the service.

It keeps the answers to the most recent questions, so that a question asked again is answered
without searching. When it is full, the answer that was used least recently is dropped. It is
safe when many requests arrive at once. It lives in the memory of one service process: when the
service restarts the cache is empty again, and when there are several copies of the service
each copy has its own cache (a shared cache such as Redis would be the next step).
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Hashable


class SearchCache:
    """Least-recently-used cache with a size limit. A size of 0 turns it off."""

    def __init__(self, max_entries: int = 1024) -> None:
        if max_entries < 0:
            raise ValueError("max_entries cannot be negative")
        self.max_entries = max_entries
        self._data: OrderedDict[Hashable, object] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: Hashable) -> object | None:
        """The saved answer, or None. Counts a hit or a miss."""
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                self.hits += 1
                return self._data[key]
            self.misses += 1
            return None

    def put(self, key: Hashable, value: object) -> None:
        if self.max_entries == 0:
            return
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)

    def clear(self) -> None:
        """Forget every answer and every count."""
        with self._lock:
            self._data.clear()
            self.hits = 0
            self.misses = 0

    def stats(self) -> dict:
        with self._lock:
            total = self.hits + self.misses
            return {
                "enabled": self.max_entries > 0,
                "max_entries": self.max_entries,
                "entries": len(self._data),
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": round(self.hits / total, 4) if total else 0.0,
            }
