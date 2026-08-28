"""Usage tracker unit tests."""
import threading
import unittest
from types import SimpleNamespace

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from utils.usage_tracking import JobUsageTracker, UsageSnapshot, reset_current_tracker, set_current_tracker


class UsageTrackingTests(unittest.TestCase):
    def test_accumulates_tokens_and_rate(self):
        tracker = JobUsageTracker()
        tracker.record_usage(input_tokens=100, output_tokens=20, total_tokens=120, duration_sec=2.0)
        tracker.record_usage(input_tokens=50, output_tokens=30, total_tokens=80, duration_sec=1.0)
        snap = tracker.snapshot()
        self.assertEqual(snap.input_tokens, 150)
        self.assertEqual(snap.output_tokens, 50)
        self.assertEqual(snap.total_tokens, 200)
        self.assertTrue(snap.complete)
        self.assertAlmostEqual(snap.tokens_per_sec, 50 / 3.0)

    def test_missing_usage_marks_incomplete(self):
        tracker = JobUsageTracker()
        tracker.record_usage(input_tokens=10, output_tokens=5, total_tokens=15, duration_sec=1.0)
        tracker.record_usage(duration_sec=0.5, source_complete=False)
        snap = tracker.snapshot()
        self.assertFalse(snap.complete)
        payload = snap.as_payload()
        self.assertIsNone(payload["total_tokens"])
        self.assertIsNone(payload["tokens_per_sec"])

    def test_openai_usage_helper(self):
        tracker = JobUsageTracker()
        usage = SimpleNamespace(prompt_tokens=12, completion_tokens=8, total_tokens=20)
        tracker.record_openai_usage(usage, 0.5)
        snap = tracker.snapshot()
        self.assertEqual(snap.total_tokens, 20)
        self.assertAlmostEqual(snap.tokens_per_sec, 16.0)

    def test_openai_usage_accepts_input_output_fields(self):
        tracker = JobUsageTracker()
        usage = {"input_tokens": 100, "output_tokens": 40, "total_tokens": 140}
        tracker.record_openai_usage(usage, 2.0)
        snap = tracker.snapshot()
        self.assertEqual(snap.total_tokens, 140)
        self.assertAlmostEqual(snap.tokens_per_sec, 20.0)

    def test_chat_model_callback_records_usage(self):
        tracker = JobUsageTracker()
        handler = tracker.callback_handler
        msg = AIMessage(content="ok", usage_metadata={"input_tokens": 30, "output_tokens": 12, "total_tokens": 42})
        result = LLMResult(generations=[[ChatGeneration(message=msg)]])
        handler.on_chat_model_start({}, [[]], run_id="chat-1")
        handler.on_chat_model_end(result, run_id="chat-1")
        self.assertEqual(tracker.snapshot().total_tokens, 42)

    def test_parallel_updates_are_thread_safe(self):
        tracker = JobUsageTracker()
        barrier = threading.Barrier(4)

        def worker(index: int) -> None:
            barrier.wait(timeout=2)
            tracker.record_usage(
                input_tokens=index,
                output_tokens=index,
                total_tokens=index * 2,
                duration_sec=0.1,
            )

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(1, 5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=3)
        snap = tracker.snapshot()
        self.assertEqual(snap.input_tokens, 10)
        self.assertEqual(snap.output_tokens, 10)
        self.assertEqual(snap.total_tokens, 20)

    def test_delta_since(self):
        before = UsageSnapshot(
            input_tokens=100,
            output_tokens=20,
            total_tokens=120,
            tokens_per_sec=10.0,
            complete=True,
            api_duration_sec=2.0,
        )
        after = UsageSnapshot(
            input_tokens=250,
            output_tokens=80,
            total_tokens=330,
            tokens_per_sec=20.0,
            complete=True,
            api_duration_sec=5.0,
        )
        self.assertEqual(after.delta_since(before), {
            "input_tokens": 150,
            "output_tokens": 60,
            "total_tokens": 210,
        })

    def test_context_tracker_is_isolated(self):
        left = JobUsageTracker()
        right = JobUsageTracker()
        token_left = set_current_tracker(left)
        left.record_usage(input_tokens=1, output_tokens=2, total_tokens=3, duration_sec=1.0)
        reset_current_tracker(token_left)
        token_right = set_current_tracker(right)
        right.record_usage(input_tokens=10, output_tokens=20, total_tokens=30, duration_sec=1.0)
        reset_current_tracker(token_right)
        self.assertEqual(left.snapshot().total_tokens, 3)
        self.assertEqual(right.snapshot().total_tokens, 30)


if __name__ == "__main__":
    unittest.main()
