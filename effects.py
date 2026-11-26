from __future__ import annotations

import math
import threading
import time
import random
from dataclasses import dataclass, replace
from typing import Callable

import cv2
import numpy as np

MOSAIC_BLOCK_SIZES = (0, 4, 8, 16)
FADE_TARGET_OPACITY = 0.5
FADE_SECONDS = 3.0
DIFF_CELL = 8
DIFF_THRESHOLD = 12.0
DIFF_DECAY = 0.8
DIFF_COLORS = tuple(
    np.array(color, dtype=np.float32)
    for color in ((0, 255, 180), (0, 160, 255), (255, 40, 200), (255, 150, 0), (255, 255, 255))
)
DIFF_GRADIENT_MODE = len(DIFF_COLORS) + 1
DIFF_RAINBOW_MODE = DIFF_GRADIENT_MODE + 1
DIFF_MODES = DIFF_RAINBOW_MODE + 1
DIFF_RAINBOW_DEGREES_PER_SECOND = 120.0


@dataclass(frozen=True)
class EffectSettings:
    fade: bool = False
    negative: bool = False
    mosaic_block: int = 0
    diff_blocks: int = 0  # 0=なし 1〜5=固定色 6=位置グラデーション 7=時間で回るレインボー
    posterize: bool = False
    tone: int = 0  # 0=なし 1=グレー 2=セピア
    pixel_art: int = 0  # 0=なし 1=細かい 2=粗い
    matrix: bool = False
    trail: bool = False
    rgb_shift: int = 0  # 0=なし 1=RGBずらしのみ 2=グリッチ追加
    scanlines: bool = False
    silhouette: int = 0  # 0=なし 1=黒 2=ネオン


TONE_MODES = 3
SILHOUETTE_MODES = 3
POSTERIZE_LUT = (((np.arange(256) * 4) // 256) * 255 // 3).astype(np.uint8)
SEPIA_GAIN = np.array([1.0, 0.8, 0.55], dtype=np.float32)
PIXEL_ART_BLOCKS = (0, 6, 12)
PIXEL_ART_LUT = (((np.arange(256) * 6) // 256) * 255 // 5).astype(np.uint8)
PIXEL_ART_OUTLINE = (20, 16, 40, 255)
MATRIX_CHARS = "0123456789ABCDEFGHJKLMNPRSTUVXYZ<>=+*:"
MATRIX_CELL_W = 10
MATRIX_CELL_H = 14
MATRIX_GREEN = np.array([0.0, 255.0, 70.0], dtype=np.float32)
MATRIX_FILL_COLOR = np.array([0.0, 30.0, 8.0], dtype=np.float32)
MATRIX_HEAD_BOOST = np.array([200.0, 0.0, 200.0], dtype=np.float32)
MATRIX_FILL_ALPHA = 90.0
SILHOUETTE_COLORS = ((0, 0, 0), (0, 255, 180))
TRAIL_DECAY = 0.94
TRAIL_DROP = 2
RGB_SHIFT_PIXELS = 6
GLITCH_PROBABILITY = 0.12


def apply_posterize(frame_rgba: np.ndarray) -> None:
    rgb = np.ascontiguousarray(frame_rgba[:, :, :3])
    frame_rgba[:, :, :3] = cv2.LUT(rgb, POSTERIZE_LUT)


def apply_tone(frame_rgba: np.ndarray, tone: int) -> None:
    gray = cv2.cvtColor(frame_rgba, cv2.COLOR_RGBA2GRAY)
    if tone == 1:
        frame_rgba[:, :, :3] = gray[:, :, None]
    else:
        sepia = gray[:, :, None].astype(np.float32) * SEPIA_GAIN
        frame_rgba[:, :, :3] = np.clip(sepia + 0.5, 0, 255).astype(np.uint8)


def apply_silhouette(frame_rgba: np.ndarray, mode: int) -> None:
    frame_rgba[:, :, :3] = SILHOUETTE_COLORS[mode - 1]


def build_glyph_atlas() -> np.ndarray:
    atlas = np.zeros((len(MATRIX_CHARS), MATRIX_CELL_H, MATRIX_CELL_W), dtype=np.float32)
    for index, char in enumerate(MATRIX_CHARS):
        canvas = np.zeros((MATRIX_CELL_H, MATRIX_CELL_W), dtype=np.uint8)
        cv2.putText(canvas, char, (1, MATRIX_CELL_H - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.42, 255, 1, cv2.LINE_AA)
        atlas[index] = canvas / 255.0
    return atlas


class MatrixEffect:
    """人物を緑の文字セルで描き、全面に文字の雨を降らせる。"""

    def __init__(self, rng: random.Random) -> None:
        self._rng = rng
        self._atlas = build_glyph_atlas()
        self._grid: tuple[int, int] | None = None

    def _reset(self, grid: tuple[int, int]) -> None:
        rows, cols = grid
        self._grid = grid
        nprng = np.random.default_rng(self._rng.getrandbits(32))
        self._nprng = nprng
        self._glyphs = nprng.integers(0, len(MATRIX_CHARS), size=grid)
        self._tail = nprng.integers(6, 18, size=cols).astype(np.float32)
        self._speed = nprng.uniform(8.0, 22.0, size=cols).astype(np.float32)
        self._active = nprng.random(cols) < 0.6
        self._cycle = rows + self._tail
        self._head = nprng.random(cols).astype(np.float32) * self._cycle

    def apply(self, frame_rgba: np.ndarray, elapsed: float) -> None:
        height, width = frame_rgba.shape[:2]
        cols = math.ceil(width / MATRIX_CELL_W)
        rows = math.ceil(height / MATRIX_CELL_H)
        if self._grid != (rows, cols):
            self._reset((rows, cols))

        self._head = (self._head + self._speed * elapsed) % self._cycle
        distance = (self._head[None, :] - np.arange(rows, dtype=np.float32)[:, None]) % self._cycle
        rain = np.where(
            (distance < self._tail) & self._active, 1.0 - distance / self._tail, 0.0
        ).astype(np.float32)

        flicker = self._nprng.random((rows, cols)) < 0.04
        flicker |= rain > 0.9
        self._glyphs[flicker] = self._nprng.integers(0, len(MATRIX_CHARS), size=int(flicker.sum()))

        small = cv2.resize(frame_rgba, (cols, rows), interpolation=cv2.INTER_AREA)
        person = small[:, :, 3].astype(np.float32) * (1.0 / 255.0)
        luma = cv2.cvtColor(small, cv2.COLOR_RGBA2GRAY).astype(np.float32) * (1.0 / 255.0)
        cell_level = np.maximum(rain, person * (0.3 + 0.7 * luma))

        tiles = self._atlas[self._glyphs].transpose(0, 2, 1, 3).reshape(
            rows * MATRIX_CELL_H, cols * MATRIX_CELL_W
        )[:height, :width]

        def expand(cells: np.ndarray) -> np.ndarray:
            size = (cols * MATRIX_CELL_W, rows * MATRIX_CELL_H)
            return cv2.resize(cells, size, interpolation=cv2.INTER_NEAREST)[:height, :width]

        level = expand(cell_level)
        head = expand(np.clip((rain - 0.85) * (1.0 / 0.15), 0, 1).astype(np.float32))
        intensity = tiles * level
        fill = frame_rgba[:, :, 3] * np.float32(MATRIX_FILL_ALPHA / 255.0)

        intensity3 = intensity[:, :, None]
        rgb = MATRIX_FILL_COLOR + intensity3 * (MATRIX_GREEN - MATRIX_FILL_COLOR)
        rgb += (tiles * head)[:, :, None] * MATRIX_HEAD_BOOST
        frame_rgba[:, :, :3] = np.clip(rgb + 0.5, 0, 255).astype(np.uint8)
        frame_rgba[:, :, 3] = np.clip(np.maximum(fill, intensity * 255.0) + 0.5, 0, 255).astype(np.uint8)


def apply_scanlines(frame_rgba: np.ndarray) -> None:
    rows = frame_rgba[2::3, :, :3]
    rows[:] = (rows.astype(np.uint16) * 140 >> 8).astype(np.uint8)


class EffectControls:
    """キー操作（GUIスレッド）と画像処理（ワーカースレッド）で共有する効果の設定。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._settings = EffectSettings()
        self._mosaic_index = 0

    def reset(self) -> None:
        with self._lock:
            self._settings = EffectSettings()
            self._mosaic_index = 0

    def snapshot(self) -> EffectSettings:
        with self._lock:
            return self._settings

    def toggle(self, name: str) -> None:
        with self._lock:
            self._settings = replace(self._settings, **{name: not getattr(self._settings, name)})

    def cycle(self, name: str, modes: int) -> None:
        with self._lock:
            self._settings = replace(
                self._settings, **{name: (getattr(self._settings, name) + 1) % modes}
            )

    def toggle_fade(self) -> None:
        self.toggle("fade")

    def toggle_negative(self) -> None:
        self.toggle("negative")

    def cycle_mosaic(self) -> None:
        with self._lock:
            self._mosaic_index = (self._mosaic_index + 1) % len(MOSAIC_BLOCK_SIZES)
            self._settings = replace(
                self._settings, mosaic_block=MOSAIC_BLOCK_SIZES[self._mosaic_index]
            )

    def toggle_diff_blocks(self) -> None:
        self.cycle("diff_blocks", DIFF_MODES)

def apply_negative(frame_rgba: np.ndarray) -> None:
    rgb = frame_rgba[:, :, :3]
    np.subtract(255, rgb, out=rgb)


def block_average(frame_rgba: np.ndarray, block: int) -> np.ndarray:
    """RGBA を block 四方で平均した縮小画像を返す。色は透過度で重み付けし、背景色が人物に混ざらないようにする。"""
    height, width = frame_rgba.shape[:2]
    grid = (math.ceil(width / block), math.ceil(height / block))
    alpha = frame_rgba[:, :, 3:4].astype(np.float32)
    premultiplied = np.empty((height, width, 4), dtype=np.float32)
    np.multiply(frame_rgba[:, :, :3], alpha * (1.0 / 255.0), out=premultiplied[:, :, :3])
    premultiplied[:, :, 3:4] = alpha
    small = cv2.resize(premultiplied, grid, interpolation=cv2.INTER_AREA)
    small_alpha = small[:, :, 3:4]
    small_rgb = small[:, :, :3] * (255.0 / np.maximum(small_alpha, 1e-3))
    small_rgba = np.concatenate((small_rgb, small_alpha), axis=2)
    return np.clip(small_rgba + 0.5, 0, 255).astype(np.uint8)


def apply_mosaic(frame_rgba: np.ndarray, block: int) -> None:
    height, width = frame_rgba.shape[:2]
    small = block_average(frame_rgba, block)
    frame_rgba[:] = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)


def apply_pixel_art(frame_rgba: np.ndarray, block: int) -> None:
    """低解像度化・限定パレット・くっきりした輪郭・暗い縁取りでドット絵風にする。"""
    height, width = frame_rgba.shape[:2]
    small = block_average(frame_rgba, block)
    small[:, :, :3] = cv2.LUT(np.ascontiguousarray(small[:, :, :3]), PIXEL_ART_LUT)
    inside = (small[:, :, 3] > 127).astype(np.uint8)
    border = cv2.dilate(inside, np.ones((3, 3), np.uint8)) & (1 - inside)
    small[:, :, 3] = inside * 255
    small[border.astype(bool)] = PIXEL_ART_OUTLINE
    frame_rgba[:] = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)


class EffectProcessor:
    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._last_time: float | None = None
        self.opacity = 1.0
        self._heat = np.empty(0, dtype=np.float32)
        self._cell_pattern: tuple[tuple[int, int], np.ndarray] | None = None
        self._trail: np.ndarray | None = None
        self._random = random.Random()
        self._gradient: tuple[tuple[int, int], np.ndarray] | None = None
        self._matrix = MatrixEffect(self._random)

    def apply(
        self,
        frame_rgba: np.ndarray,
        settings: EffectSettings,
        alpha_change: np.ndarray | None,
    ) -> np.ndarray:
        """frame_rgba を直接書き換えて返す。alpha_change は平滑化前後の透過度の差。"""
        now = self._clock()
        elapsed = 0.0 if self._last_time is None else max(now - self._last_time, 0.0)
        self._last_time = now

        if settings.posterize:
            apply_posterize(frame_rgba)
        if settings.tone:
            apply_tone(frame_rgba, settings.tone)
        if settings.silhouette:
            apply_silhouette(frame_rgba, settings.silhouette)
        if settings.negative:
            apply_negative(frame_rgba)
        if settings.mosaic_block:
            apply_mosaic(frame_rgba, settings.mosaic_block)
        if settings.pixel_art:
            apply_pixel_art(frame_rgba, PIXEL_ART_BLOCKS[settings.pixel_art])
        if settings.matrix:
            self._matrix.apply(frame_rgba, elapsed)
        if settings.rgb_shift:
            self._apply_rgb_shift(frame_rgba, glitch=settings.rgb_shift >= 2)
        self._apply_trail(frame_rgba, settings.trail)
        if settings.scanlines:
            apply_scanlines(frame_rgba)
        self._apply_fade(frame_rgba, settings.fade, elapsed)
        self._apply_diff_blocks(frame_rgba, settings.diff_blocks, alpha_change, now)
        return frame_rgba

    def _apply_rgb_shift(self, frame_rgba: np.ndarray, glitch: bool) -> None:
        shift = RGB_SHIFT_PIXELS
        source = frame_rgba.copy()
        frame_rgba[:, :, 0] = np.roll(source[:, :, 0], -shift, axis=1)
        frame_rgba[:, :, 2] = np.roll(source[:, :, 2], shift, axis=1)
        np.maximum(frame_rgba[:, :, 3], np.roll(source[:, :, 3], -shift, axis=1), out=frame_rgba[:, :, 3])
        np.maximum(frame_rgba[:, :, 3], np.roll(source[:, :, 3], shift, axis=1), out=frame_rgba[:, :, 3])
        if glitch and self._random.random() < GLITCH_PROBABILITY:
            height = frame_rgba.shape[0]
            top = self._random.randrange(height)
            bottom = min(height, top + self._random.randrange(4, 40))
            offset = self._random.choice((-1, 1)) * self._random.randrange(10, 60)
            frame_rgba[top:bottom] = np.roll(frame_rgba[top:bottom], offset, axis=1)

    def _apply_trail(self, frame_rgba: np.ndarray, enabled: bool) -> None:
        if not enabled:
            self._trail = None
            return
        if self._trail is not None and self._trail.shape == frame_rgba.shape:
            older = self._trail[:, :, 3] > frame_rgba[:, :, 3]
            frame_rgba[older] = self._trail[older]
        faded = frame_rgba.copy()
        alpha = cv2.convertScaleAbs(faded[:, :, 3], alpha=TRAIL_DECAY)
        faded[:, :, 3] = np.maximum(alpha, TRAIL_DROP) - TRAIL_DROP
        self._trail = faded

    def _apply_fade(self, frame_rgba: np.ndarray, enabled: bool, elapsed: float) -> None:
        target = FADE_TARGET_OPACITY if enabled else 1.0
        step = (1.0 - FADE_TARGET_OPACITY) / FADE_SECONDS * elapsed
        if self.opacity < target:
            self.opacity = min(self.opacity + step, target)
        else:
            self.opacity = max(self.opacity - step, target)
        if self.opacity < 1.0:
            frame_rgba[:, :, 3] = cv2.convertScaleAbs(frame_rgba[:, :, 3], alpha=self.opacity)

    def _apply_diff_blocks(
        self, frame_rgba: np.ndarray, mode: int, alpha_change: np.ndarray | None, now: float
    ) -> None:
        height, width = frame_rgba.shape[:2]
        grid = (math.ceil(height / DIFF_CELL), math.ceil(width / DIFF_CELL))
        if self._heat.shape != grid:
            self._heat = np.zeros(grid, dtype=np.float32)
        self._heat *= DIFF_DECAY
        if not mode:
            self._heat[:] = 0.0
            return
        if alpha_change is not None and alpha_change.shape == (height, width):
            cell_change = cv2.resize(alpha_change, (grid[1], grid[0]), interpolation=cv2.INTER_AREA)
            np.maximum(self._heat, cell_change >= DIFF_THRESHOLD, out=self._heat)
        if self._heat.max() < 0.02:
            return

        cells = cv2.resize(self._heat, (grid[1] * DIFF_CELL, grid[0] * DIFF_CELL), interpolation=cv2.INTER_NEAREST)
        intensity = cells[:height, :width] * self._grid_pattern(height, width)
        lit = intensity > 0.01
        weight = intensity[lit][:, None]
        pixels = frame_rgba[lit]
        color = self._diff_color(mode, grid, (height, width), lit, now)
        pixels[:, :3] = (pixels[:, :3] * (1.0 - weight) + color * weight + 0.5).astype(np.uint8)
        pixels[:, 3] = np.maximum(pixels[:, 3], (weight[:, 0] * 220.0).astype(np.uint8))
        frame_rgba[lit] = pixels

    def _diff_color(
        self, mode: int, grid: tuple[int, int], size: tuple[int, int], lit: np.ndarray, now: float
    ) -> np.ndarray:
        """固定色、位置によるグラデーション（形状は点灯ブロック単位）、時間で回るレインボーのいずれかを返す。"""
        if mode <= len(DIFF_COLORS):
            return DIFF_COLORS[mode - 1]
        if mode == DIFF_GRADIENT_MODE:
            if self._gradient is None or self._gradient[0] != grid:
                rows, cols = np.mgrid[0 : grid[0], 0 : grid[1]].astype(np.float32)
                hue = ((cols / grid[1]) * 0.75 + (rows / grid[0]) * 0.25) * 179.0
                hsv = np.stack((hue, np.full_like(hue, 255), np.full_like(hue, 255)), axis=2)
                rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
                self._gradient = (grid, rgb)
            full = cv2.resize(
                self._gradient[1],
                (grid[1] * DIFF_CELL, grid[0] * DIFF_CELL),
                interpolation=cv2.INTER_NEAREST,
            )
            return full[: size[0], : size[1]][lit]
        hue = np.uint8(int(now * DIFF_RAINBOW_DEGREES_PER_SECOND / 2.0) % 180)
        pixel = cv2.cvtColor(np.array([[[hue, 255, 255]]], dtype=np.uint8), cv2.COLOR_HSV2RGB)
        return pixel[0, 0].astype(np.float32)

    def _grid_pattern(self, height: int, width: int) -> np.ndarray:
        """ブロック間に1pxの隙間を空け、デジタルなタイル状に見せる。"""
        if self._cell_pattern is None or self._cell_pattern[0] != (height, width):
            rows = (np.arange(height) % DIFF_CELL) < DIFF_CELL - 1
            cols = (np.arange(width) % DIFF_CELL) < DIFF_CELL - 1
            pattern = (rows[:, None] & cols[None, :]).astype(np.float32)
            self._cell_pattern = ((height, width), pattern)
        return self._cell_pattern[1]
