"""The numbers the service reports on `/metrics`, in the Prometheus text format.

Prometheus reads this page every few seconds and keeps the numbers over time. Each app has
its own registry, so tests that build several apps do not share counters.
"""

from __future__ import annotations

from collections.abc import Callable

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    ProcessCollector,
    generate_latest,
)

# Search times are mostly 1 to 20 milliseconds, so the buckets are small at the start.
SECONDS_BUCKETS = (0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5)


class Metrics:
    """The counters, timers and gauges of one running service."""

    def __init__(
        self,
        cache_entries: Callable[[], float],
        is_ready: Callable[[], bool],
    ) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "ps_http_requests_total",
            "Requests answered, by endpoint and HTTP status code.",
            ["endpoint", "status"],
            registry=self.registry,
        )
        self.request_seconds = Histogram(
            "ps_http_request_seconds",
            "Time to answer a request, by endpoint.",
            ["endpoint"],
            buckets=SECONDS_BUCKETS,
            registry=self.registry,
        )
        self.searches = Counter(
            "ps_searches_total",
            "Searches answered, by the mode that ran and whether the cache had the answer.",
            ["mode", "cache"],
            registry=self.registry,
        )
        self.search_seconds = Histogram(
            "ps_search_seconds",
            "Time of a search that was really run (not answered from the cache), by mode.",
            ["mode"],
            buckets=SECONDS_BUCKETS,
            registry=self.registry,
        )
        self.errors = Counter(
            "ps_errors_total",
            "Error answers, by error code.",
            ["error"],
            registry=self.registry,
        )
        Gauge(
            "ps_cache_entries",
            "Answers kept in the cache of this process.",
            registry=self.registry,
        ).set_function(cache_entries)
        Gauge(
            "ps_ready",
            "1 when the data is loaded and the service can answer, otherwise 0.",
            registry=self.registry,
        ).set_function(lambda: 1.0 if is_ready() else 0.0)
        # Memory and CPU of the process. This only reports on Linux, which is what the image uses.
        ProcessCollector(namespace="ps", registry=self.registry)

    def render(self) -> tuple[bytes, str]:
        """The page text and its content type."""
        return generate_latest(self.registry), CONTENT_TYPE_LATEST
