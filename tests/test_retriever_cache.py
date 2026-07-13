"""检索器缓存失效与并发发布的回归测试。"""

from __future__ import annotations

import threading
import unittest
from unittest.mock import patch

from refmind.rag import retrieval


class RetrieverCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        retrieval.reset_retrievers()

    def tearDown(self) -> None:
        retrieval.reset_retrievers()

    def test_invalidation_waits_for_inflight_build_then_removes_snapshot(self) -> None:
        build_started = threading.Event()
        allow_build = threading.Event()
        built = object()

        def build(_group_id: int):
            build_started.set()
            self.assertTrue(allow_build.wait(timeout=2))
            return built

        with patch.object(retrieval, "build_retriever", side_effect=build):
            getter = threading.Thread(target=retrieval.get_retriever, args=(7,))
            getter.start()
            self.assertTrue(build_started.wait(timeout=2))

            invalidator = threading.Thread(
                target=retrieval.invalidate_retriever, args=(7,)
            )
            invalidator.start()
            allow_build.set()
            getter.join(timeout=2)
            invalidator.join(timeout=2)

        self.assertFalse(getter.is_alive())
        self.assertFalse(invalidator.is_alive())
        self.assertNotIn(7, retrieval._RETRIEVER_CACHE)


if __name__ == "__main__":
    unittest.main()
