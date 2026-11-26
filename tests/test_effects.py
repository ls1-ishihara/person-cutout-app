import unittest

import numpy as np

from effects import (
    DIFF_GRADIENT_MODE,
    DIFF_MODES,
    DIFF_RAINBOW_MODE,
    EffectControls,
    EffectProcessor,
    EffectSettings,
    apply_mosaic,
    apply_negative,
)


def solid_frame(rgb=(10, 20, 30), alpha=255, size=(32, 48)):
    frame = np.empty((*size, 4), dtype=np.uint8)
    frame[:, :, :3] = rgb
    frame[:, :, 3] = alpha
    return frame


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class EffectControlsTests(unittest.TestCase):
    def test_keys_change_only_their_own_effect_and_mosaic_cycles(self):
        controls = EffectControls()
        controls.toggle_fade()
        controls.toggle_negative()
        controls.toggle_diff_blocks()
        self.assertEqual(controls.snapshot(), EffectSettings(True, True, 0, True))

        sizes = []
        for _ in range(5):
            controls.cycle_mosaic()
            sizes.append(controls.snapshot().mosaic_block)
        self.assertEqual(sizes, [4, 8, 16, 0, 4])
        self.assertEqual(controls.snapshot(), EffectSettings(True, True, 4, True))

        controls.toggle_fade()
        self.assertFalse(controls.snapshot().fade)


class ImageEffectTests(unittest.TestCase):
    def test_negative_inverts_colors_and_keeps_alpha(self):
        frame = solid_frame((10, 20, 30), alpha=77)
        apply_negative(frame)
        self.assertEqual(tuple(frame[0, 0]), (245, 235, 225, 77))

    def test_mosaic_makes_each_block_uniform(self):
        frame = solid_frame()
        frame[:, :, 0] = np.arange(48, dtype=np.uint8)[None, :] * 5
        apply_mosaic(frame, 8)
        for x in range(0, 48, 8):
            block = frame[0:8, x : x + 8]
            self.assertTrue(np.all(block == block[0, 0]))
        self.assertNotEqual(int(frame[0, 0, 0]), int(frame[0, 8, 0]))

    def test_mosaic_does_not_mix_transparent_background_color_into_person(self):
        frame = solid_frame((200, 0, 0), alpha=255)
        frame[:, 4:, :3] = (0, 0, 255)
        frame[:, 4:, 3] = 0
        apply_mosaic(frame, 8)
        self.assertEqual(tuple(frame[0, 0, :3]), (200, 0, 0))
        self.assertEqual(int(frame[0, 0, 3]), 128)

    def test_mosaic_handles_sizes_that_are_not_multiples_of_block(self):
        frame = solid_frame(size=(30, 45))
        apply_mosaic(frame, 16)
        self.assertEqual(frame.shape, (30, 45, 4))


class FadeTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.processor = EffectProcessor(self.clock)

    def run_frame(self, fade, seconds):
        self.clock.now += seconds
        frame = solid_frame(alpha=200)
        self.processor.apply(frame, EffectSettings(fade=fade), None)
        return frame

    def test_person_gradually_fades_to_half_and_stops_there(self):
        self.run_frame(True, 0.0)
        halfway = self.run_frame(True, 1.5)
        self.assertGreater(int(halfway[0, 0, 3]), 100)
        self.assertLess(int(halfway[0, 0, 3]), 200)
        final = self.run_frame(True, 10.0)
        self.assertEqual(int(final[0, 0, 3]), 100)
        self.assertEqual(self.processor.opacity, 0.5)

    def test_toggling_off_gradually_restores_full_opacity(self):
        self.run_frame(True, 0.0)
        self.run_frame(True, 10.0)
        restoring = self.run_frame(False, 1.5)
        self.assertGreater(int(restoring[0, 0, 3]), 100)
        self.assertLess(int(restoring[0, 0, 3]), 200)
        self.assertEqual(int(self.run_frame(False, 10.0)[0, 0, 3]), 200)


class DiffBlockTests(unittest.TestCase):
    def setUp(self):
        self.processor = EffectProcessor(FakeClock())
        self.settings = EffectSettings(diff_blocks=True)

    def changed_area(self):
        change = np.zeros((32, 48), dtype=np.float32)
        change[0:8, 8:16] = 200.0
        return change

    def test_only_changed_cells_are_drawn_as_tiles_with_gaps(self):
        frame = solid_frame(rgb=(0, 0, 0), alpha=0)
        self.processor.apply(frame, self.settings, self.changed_area())

        self.assertGreater(int(frame[3, 11, 3]), 150)
        self.assertGreater(int(frame[3, 11, 1]), 200)
        self.assertEqual(int(frame[3, 15, 3]), 0)
        self.assertEqual(int(frame[7, 11, 3]), 0)
        self.assertEqual(int(frame[20, 30, 3]), 0)

    def test_highlight_fades_out_after_the_change_stops(self):
        self.processor.apply(solid_frame(alpha=0), self.settings, self.changed_area())
        alphas = []
        for _ in range(30):
            frame = solid_frame(rgb=(0, 0, 0), alpha=0)
            self.processor.apply(frame, self.settings, np.zeros((32, 48), np.float32))
            alphas.append(int(frame[3, 11, 3]))
        self.assertGreater(alphas[0], alphas[3])
        self.assertEqual(alphas[-1], 0)

    def lit_color(self, mode, at=0.0, change=None):
        clock = FakeClock()
        clock.now += at
        processor = EffectProcessor(clock)
        frame = solid_frame(rgb=(0, 0, 0), alpha=0, size=(32, 48))
        processor.apply(frame, EffectSettings(diff_blocks=mode), change)
        return frame

    def test_f4_cycles_through_colors_gradient_rainbow_then_off(self):
        controls = EffectControls()
        seen = []
        for _ in range(DIFF_MODES + 1):
            controls.toggle_diff_blocks()
            seen.append(controls.snapshot().diff_blocks)
        self.assertEqual(seen, [1, 2, 3, 4, 5, 6, 7, 0, 1])
        self.assertEqual(DIFF_RAINBOW_MODE, 7)

    def test_fixed_color_modes_use_different_colors(self):
        wide = np.full((32, 48), 200.0, dtype=np.float32)
        colors = {tuple(self.lit_color(m, change=wide)[3, 3, :3]) for m in range(1, 6)}
        self.assertEqual(len(colors), 5)

    def test_gradient_color_depends_on_block_position(self):
        wide = np.full((32, 48), 200.0, dtype=np.float32)
        frame = self.lit_color(DIFF_GRADIENT_MODE, change=wide)
        self.assertNotEqual(tuple(frame[3, 3, :3]), tuple(frame[3, 43, :3]))

    def test_rainbow_color_changes_over_time(self):
        wide = np.full((32, 48), 200.0, dtype=np.float32)
        first = self.lit_color(DIFF_RAINBOW_MODE, at=0.0, change=wide)
        later = self.lit_color(DIFF_RAINBOW_MODE, at=1.0, change=wide)
        self.assertNotEqual(tuple(first[3, 3, :3]), tuple(later[3, 3, :3]))
        self.assertEqual(tuple(first[3, 3, :3]), tuple(first[3, 43, :3]))

    def test_small_changes_and_disabled_effect_draw_nothing(self):
        small = np.full((32, 48), 5.0, dtype=np.float32)
        frame = solid_frame(alpha=0)
        self.processor.apply(frame, self.settings, small)
        self.assertTrue(np.all(frame[:, :, 3] == 0))

        frame = solid_frame(alpha=0)
        self.processor.apply(frame, EffectSettings(), self.changed_area())
        self.assertTrue(np.all(frame[:, :, 3] == 0))


class NewEffectTests(unittest.TestCase):
    def apply(self, **kwargs):
        frame = solid_frame((200, 100, 50), alpha=255, size=(32, 48))
        EffectProcessor(FakeClock()).apply(frame, EffectSettings(**kwargs), None)
        return frame

    def test_posterize_limits_levels(self):
        self.assertEqual(tuple(self.apply(posterize=True)[0, 0, :3]), (255, 85, 0))

    def test_tone_gray_and_sepia(self):
        gray = self.apply(tone=1)[0, 0]
        self.assertTrue(gray[0] == gray[1] == gray[2])
        sepia = self.apply(tone=2)[0, 0]
        self.assertTrue(sepia[0] > sepia[1] > sepia[2])

    def test_pixel_art_uses_blocks_limited_palette_binary_alpha_and_outline(self):
        frame = solid_frame((200, 100, 50), alpha=0, size=(60, 60))
        frame[18:42, 18:42, 3] = 255
        frame[18:42, 18:42, :3] = (200, 100, 50)
        EffectProcessor(FakeClock()).apply(frame, EffectSettings(pixel_art=1), None)

        self.assertEqual(tuple(frame[30, 30, :3]), (204, 102, 51))
        self.assertEqual(len(np.unique(frame[24:30, 24:30].reshape(-1, 4), axis=0)), 1)
        self.assertEqual(tuple(frame[15, 30]), (20, 16, 40, 255))
        self.assertEqual(int(frame[2, 2, 3]), 0)
        self.assertTrue(set(np.unique(frame[:, :, 3])) <= {0, 255})

    def test_pixel_art_coarse_mode_uses_larger_blocks(self):
        frame = solid_frame(alpha=255, size=(48, 48))
        frame[:, :, 0] = np.arange(48, dtype=np.uint8)[None, :] * 5
        EffectProcessor(FakeClock()).apply(frame, EffectSettings(pixel_art=2), None)
        self.assertTrue(np.all(frame[0:12, 0:12] == frame[0, 0]))

    def test_silhouette_uses_solid_color(self):
        self.assertEqual(tuple(self.apply(silhouette=1)[5, 5, :3]), (0, 0, 0))
        self.assertEqual(tuple(self.apply(silhouette=2)[5, 5, :3]), (0, 255, 180))

    def test_matrix_draws_green_glyphs_on_person_and_rain_elsewhere(self):
        frame = solid_frame((200, 200, 200), alpha=0, size=(140, 100))
        frame[:, :50, 3] = 255
        processor = EffectProcessor(FakeClock())
        processor.apply(frame, EffectSettings(matrix=True), None)

        person = frame[:, :50]
        self.assertTrue(np.all(person[:, :, 3] >= 89))
        self.assertGreater(int(person[:, :, 1].max()), 100)
        self.assertTrue(np.all(person[:, :, 1] >= person[:, :, 0]))
        glyph_pixels = np.count_nonzero(person[:, :, 3] > 150)
        self.assertTrue(0 < glyph_pixels < person[:, :, 3].size)

    def test_matrix_rain_moves_over_time(self):
        clock = FakeClock()
        processor = EffectProcessor(clock)
        settings = EffectSettings(matrix=True)
        frames = []
        for _ in range(2):
            frame = solid_frame(alpha=0, size=(140, 100))
            processor.apply(frame, settings, None)
            frames.append(frame[:, :, 3].copy())
            clock.now += 0.5
        self.assertTrue(np.any(frames[0] != frames[1]))

    def test_trail_keeps_fading_ghost_after_person_leaves(self):
        processor = EffectProcessor(FakeClock())
        settings = EffectSettings(trail=True)
        processor.apply(solid_frame(alpha=255), settings, None)
        alphas = []
        for _ in range(80):
            frame = solid_frame(alpha=0)
            processor.apply(frame, settings, None)
            alphas.append(int(frame[0, 0, 3]))
        self.assertGreater(alphas[0], 100)
        self.assertGreater(alphas[0], alphas[5])
        self.assertEqual(alphas[-1], 0)

    def test_rgb_shift_separates_channels_at_an_edge(self):
        frame = solid_frame((0, 0, 0), alpha=255, size=(8, 40))
        frame[:, 20:, :3] = 255
        processor = EffectProcessor(FakeClock())
        processor._random.random = lambda: 0.0
        processor.apply(frame, EffectSettings(rgb_shift=1), None)
        self.assertGreater(int(frame[0, 17, 0]), int(frame[0, 17, 2]))

    def test_glitch_only_runs_in_second_rgb_shift_mode(self):
        def render(mode):
            frame = solid_frame(alpha=255, size=(64, 64))
            frame[:, :, 0] = np.arange(64, dtype=np.uint8)[None, :] * 3
            processor = EffectProcessor(FakeClock())
            processor._random.random = lambda: 0.0
            processor.apply(frame, EffectSettings(rgb_shift=mode), None)
            return frame

        self.assertTrue(np.array_equal(render(1), render(1)))
        self.assertFalse(np.array_equal(render(1), render(2)))

    def test_scanlines_darken_every_third_row(self):
        frame = self.apply(scanlines=True)
        self.assertEqual(int(frame[0, 0, 0]), 200)
        self.assertLess(int(frame[2, 0, 0]), 200)

    def test_generic_controls_do_not_reset_other_effects(self):
        controls = EffectControls()
        controls.toggle_fade()
        controls.cycle("pixel_art", 3)
        controls.cycle("tone", 3)
        controls.toggle_negative()
        controls.cycle_mosaic()
        settings = controls.snapshot()
        self.assertTrue(settings.fade and settings.pixel_art == 1 and settings.negative)
        self.assertEqual((settings.tone, settings.mosaic_block), (1, 4))


if __name__ == "__main__":
    unittest.main()
