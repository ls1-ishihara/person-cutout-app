import unittest
from unittest.mock import Mock

import numpy as np

from segmentation import TemporalAlphaSmoother, confidence_to_alpha, segment_frame


class ConfidenceToAlphaTests(unittest.TestCase):
    def test_background_pixels_are_fully_transparent(self):
        confidence = np.zeros((8, 8), dtype=np.float32)
        confidence[2:6, 2:6] = 1.0

        alpha = confidence_to_alpha(confidence, (8, 8))

        self.assertTrue(np.all(alpha[:1, :] == 0))
        self.assertTrue(np.all(alpha[:, :1] == 0))
        self.assertEqual(int(alpha[3, 3]), 255)
        self.assertGreater(int(alpha[2, 3]), 0)

    def test_resizes_mask_to_camera_frame_and_clears_invalid_values(self):
        confidence = np.array([[np.nan, 0.0], [1.0, np.inf]], dtype=np.float32)

        alpha = confidence_to_alpha(confidence, (4, 4))

        self.assertEqual(alpha.shape, (4, 4))
        self.assertEqual(alpha.dtype, np.uint8)

    def test_rejects_multi_channel_confidence_mask(self):
        confidence = np.zeros((4, 4, 2), dtype=np.float32)

        with self.assertRaises(ValueError):
            confidence_to_alpha(confidence, (4, 4))


class TemporalAlphaSmootherTests(unittest.TestCase):
    def test_first_frame_and_large_changes_follow_without_a_trail(self):
        smoother = TemporalAlphaSmoother()
        foreground = np.full((8, 8), 255, dtype=np.uint8)
        background = np.zeros_like(foreground)
        np.testing.assert_array_equal(smoother.apply(foreground), foreground)

        np.testing.assert_array_equal(smoother.apply(background), background)
        np.testing.assert_array_equal(smoother.apply(foreground), foreground)

    def test_reduces_alternating_mask_flicker(self):
        smoother = TemporalAlphaSmoother()
        smoother.apply(np.full((8, 8), 128, dtype=np.uint8))
        values = [118, 138] * 10
        filtered = [
            int(smoother.apply(np.full((8, 8), value, dtype=np.uint8))[0, 0])
            for value in values
        ]
        original_variation = max(values[4:]) - min(values[4:])
        filtered_variation = max(filtered[4:]) - min(filtered[4:])
        self.assertLessEqual(filtered_variation, original_variation / 5)

    def test_motion_and_small_noise_are_handled_independently_per_pixel(self):
        smoother = TemporalAlphaSmoother()
        smoother.apply(np.array([[0, 255, 128]], dtype=np.uint8))
        filtered = smoother.apply(np.array([[255, 0, 138]], dtype=np.uint8))
        np.testing.assert_array_equal(filtered, np.array([[255, 0, 130]], dtype=np.uint8))

    def test_small_residual_fades_to_fully_transparent(self):
        smoother = TemporalAlphaSmoother()
        smoother.apply(np.full((8, 8), 15, dtype=np.uint8))
        for _ in range(20):
            filtered = smoother.apply(np.zeros((8, 8), dtype=np.uint8))
        self.assertTrue(np.all(filtered == 0))

    def test_resets_history_when_frame_size_changes(self):
        smoother = TemporalAlphaSmoother()
        smoother.apply(np.full((8, 8), 255, dtype=np.uint8))
        resized_background = np.zeros((4, 6), dtype=np.uint8)
        np.testing.assert_array_equal(smoother.apply(resized_background), resized_background)

    def test_segment_frame_smooths_alpha_but_keeps_latest_colors(self):
        segmenter = Mock()
        mask = Mock()
        segmenter.segment_for_video.return_value.confidence_masks = [mask]
        frame = np.full((8, 8, 3), (10, 20, 30), dtype=np.uint8)
        smoother = TemporalAlphaSmoother()

        mask.numpy_view.return_value = np.full((8, 8), 0.4, dtype=np.float32)
        segment_frame(segmenter, frame, 1, smoother)
        mask.numpy_view.return_value = np.full((8, 8), 0.42, dtype=np.float32)
        latest_frame = np.full_like(frame, (40, 50, 60))
        rgba = segment_frame(segmenter, latest_frame, 2, smoother)

        self.assertEqual(rgba.shape, (8, 8, 4))
        self.assertEqual(rgba.dtype, np.uint8)
        np.testing.assert_array_equal(rgba[:, :, :3], latest_frame[:, :, ::-1])
        unsmoothed = segment_frame(segmenter, latest_frame, 3)
        previous_alpha = confidence_to_alpha(np.full((8, 8), 0.4), (8, 8))
        self.assertTrue(np.all(rgba[:, :, 3] > previous_alpha))
        self.assertTrue(np.all(rgba[:, :, 3] < unsmoothed[:, :, 3]))


if __name__ == "__main__":
    unittest.main()
