"""模型熔断与流式降级的离线回归测试。"""

from __future__ import annotations

import time
import unittest
from unittest.mock import patch

from refmind.config import settings
from refmind.llm import factory


class _StreamingModel:
    def __init__(self, chunks: list[str], error: Exception | None = None) -> None:
        self.chunks = chunks
        self.error = error
        self.calls = 0

    def stream(self, *_args, **_kwargs):
        self.calls += 1
        yield from self.chunks
        if self.error is not None:
            raise self.error


class CircuitBreakerTests(unittest.TestCase):
    def setUp(self) -> None:
        factory._circuit_breaker.force_reset()

    def tearDown(self) -> None:
        factory._circuit_breaker.force_reset()

    def test_half_open_only_allows_one_concurrent_probe(self) -> None:
        breaker = factory._circuit_breaker
        with patch.object(settings, "llm_health_check_interval", 1):
            breaker._transition_to(factory._State.OPEN)
            breaker._opened_at = time.time() - 2

            self.assertTrue(breaker.should_allow_primary())
            self.assertFalse(breaker.should_allow_primary())

            breaker.record_success()
            self.assertTrue(breaker.should_allow_primary())

    def test_partial_primary_stream_is_never_mixed_with_fallback(self) -> None:
        primary = _StreamingModel(["partial"], RuntimeError("mid-stream"))
        fallback = _StreamingModel(["fallback"])

        with (
            patch.object(settings, "llm_circuit_failure_threshold", 1),
            patch.object(factory, "_get_primary_model", return_value=primary),
            patch.object(factory, "_get_fallback_model", return_value=fallback),
        ):
            stream = factory._FallbackLLM(0.0).stream("question")
            self.assertEqual(next(stream), "partial")
            with self.assertRaisesRegex(RuntimeError, "mid-stream"):
                next(stream)

        self.assertEqual(fallback.calls, 0)

    def test_failure_before_first_chunk_can_fall_back(self) -> None:
        primary = _StreamingModel([], RuntimeError("unavailable"))
        fallback = _StreamingModel(["fallback"])

        with (
            patch.object(settings, "llm_circuit_failure_threshold", 1),
            patch.object(factory, "_get_primary_model", return_value=primary),
            patch.object(factory, "_get_fallback_model", return_value=fallback),
        ):
            chunks = list(factory._FallbackLLM(0.0).stream("question"))

        self.assertEqual(chunks, ["fallback"])
        self.assertEqual(fallback.calls, 1)


if __name__ == "__main__":
    unittest.main()
