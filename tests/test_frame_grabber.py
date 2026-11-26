import queue
import threading
import unittest

import numpy as np

from frame_grabber import LatestFrameGrabber


class QueueCapture:
    """テストが push した順にフレームを返し、空の間は read() で待機する。"""

    def __init__(self):
        self._items = queue.Queue()
        self.consumed = threading.Semaphore(0)

    def push(self, value):
        self._items.put((True, np.full((2, 2, 3), value, dtype=np.uint8)))

    def push_failure(self):
        self._items.put((False, None))

    def read(self):
        item = self._items.get()
        self.consumed.release()
        return item


class LatestFrameGrabberTests(unittest.TestCase):
    def start(self, capture, transform=None):
        grabber = LatestFrameGrabber(capture, transform)
        grabber.start()
        self.addCleanup(self.finish, capture, grabber)
        return grabber

    @staticmethod
    def finish(capture, grabber):
        grabber.request_stop()
        capture.push(0)
        grabber.stop()

    def test_skips_stale_frames_and_applies_transform(self):
        capture = QueueCapture()
        grabber = self.start(capture, lambda frame: frame + 100)
        for value in (1, 2, 3):
            capture.push(value)
        for _ in range(3):
            self.assertTrue(capture.consumed.acquire(timeout=2))

        self.assertEqual(int(grabber.read()[0, 0, 0]), 103)

    def test_read_waits_for_a_new_frame_instead_of_repeating_one(self):
        capture = QueueCapture()
        grabber = self.start(capture)
        capture.push(1)
        self.assertEqual(int(grabber.read()[0, 0, 0]), 1)

        results = []
        reader = threading.Thread(target=lambda: results.append(grabber.read()))
        reader.start()
        reader.join(timeout=0.2)
        self.assertTrue(reader.is_alive())

        capture.push(2)
        reader.join(timeout=2)
        self.assertEqual(int(results[0][0, 0, 0]), 2)

    def test_capture_failure_is_raised_to_the_reader(self):
        capture = QueueCapture()
        grabber = self.start(capture)
        capture.push_failure()

        with self.assertRaises(RuntimeError):
            grabber.read()

    def test_stop_wakes_a_waiting_reader_and_ends_the_thread(self):
        capture = QueueCapture()
        grabber = LatestFrameGrabber(capture)
        grabber.start()
        results = []
        reader = threading.Thread(target=lambda: results.append(grabber.read()))
        reader.start()

        grabber.request_stop()
        reader.join(timeout=2)
        capture.push(0)

        self.assertFalse(reader.is_alive())
        self.assertEqual(results, [None])
        self.assertTrue(grabber.stop())


if __name__ == "__main__":
    unittest.main()
