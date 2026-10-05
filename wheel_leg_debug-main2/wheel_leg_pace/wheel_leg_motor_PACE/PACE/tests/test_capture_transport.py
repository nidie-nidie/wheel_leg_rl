import collections
import unittest


SAMPLE_BYTES = 126
SAMPLE_PERIOD_US = 2_000
STATUS_BYTES = 128
STATUS_PERIOD_US = 100_000
RING_CAPACITY_FRAMES = 127


def simulate(consumer_bytes_per_second: float, duration_us: int):
    queue = collections.deque()
    queued_bytes = 0.0
    next_sample = 0
    next_status = 0
    last_time = 0
    overflow = False

    for now in range(0, duration_us + 1, 100):
        budget = consumer_bytes_per_second * (now - last_time) / 1_000_000.0
        last_time = now
        while queue and budget > 0.0:
            sent = min(queue[0], budget)
            queue[0] -= sent
            queued_bytes -= sent
            budget -= sent
            if queue[0] <= 1e-9:
                queue.popleft()

        while next_sample <= now:
            if len(queue) >= RING_CAPACITY_FRAMES:
                overflow = True
                return overflow, len(queue), queued_bytes
            queue.append(float(SAMPLE_BYTES))
            queued_bytes += SAMPLE_BYTES
            next_sample += SAMPLE_PERIOD_US

        while next_status <= now:
            if len(queue) >= RING_CAPACITY_FRAMES:
                overflow = True
                return overflow, len(queue), queued_bytes
            queue.append(float(STATUS_BYTES))
            queued_bytes += STATUS_BYTES
            next_status += STATUS_PERIOD_US

    return overflow, len(queue), queued_bytes


class CaptureTransportTest(unittest.TestCase):
    def test_nominal_uart_capacity_does_not_overflow(self):
        overflow, depth, queued = simulate(92_160.0, 20_000_000)
        self.assertFalse(overflow)
        self.assertLess(depth, 4)
        self.assertLess(queued, 400.0)

    def test_worst_case_stream_matches_documented_budget(self):
        required = SAMPLE_BYTES * 500 + STATUS_BYTES * 10
        self.assertEqual(required, 64_280)
        self.assertLessEqual(required, 92_160 * 0.70)

    def test_underprovisioned_consumer_overflows_deterministically(self):
        overflow, depth, _ = simulate(60_000.0, 20_000_000)
        self.assertTrue(overflow)
        self.assertEqual(depth, RING_CAPACITY_FRAMES)


if __name__ == "__main__":
    unittest.main()
