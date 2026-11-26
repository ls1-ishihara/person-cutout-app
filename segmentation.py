from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np


class TemporalAlphaSmoother:
    def __init__(self) -> None:
        self._filtered: np.ndarray | None = None
        self._current = np.empty(0, dtype=np.float32)
        self._delta = np.empty(0, dtype=np.float32)
        self._weight = np.empty(0, dtype=np.float32)
        self._output = np.empty(0, dtype=np.uint8)
        self._change = np.empty(0, dtype=np.float32)
        # 直前の平滑化結果との差（0〜255）。最初のフレームは None。次の apply まで有効。
        self.change: np.ndarray | None = None

    def apply(self, alpha: np.ndarray) -> np.ndarray:
        """Returns a buffer that is overwritten by the next call."""
        if self._filtered is None or self._filtered.shape != alpha.shape:
            self._filtered = alpha.astype(np.float32)
            self._current = np.empty_like(self._filtered)
            self._delta = np.empty_like(self._filtered)
            self._weight = np.empty_like(self._filtered)
            self._change = np.empty_like(self._filtered)
            self._output = np.empty(alpha.shape, dtype=np.uint8)
            self.change = None
        else:
            current, delta, weight = self._current, self._delta, self._weight
            np.copyto(current, alpha)
            cv2.subtract(current, self._filtered, dst=delta)
            cv2.absdiff(current, self._filtered, dst=self._change)
            self.change = self._change
            np.copyto(weight, self._change)
            # Small alpha changes retain history; changes of 96/255 or more follow immediately.
            weight -= 16.0
            weight *= 1.0 / 80.0
            np.clip(weight, 0.0, 1.0, out=weight)
            weight *= 0.8
            weight += 0.2
            delta *= weight
            self._filtered += delta
        cv2.convertScaleAbs(self._filtered, dst=self._output)
        return self._output


def confidence_to_alpha(confidence_mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    mask = np.asarray(confidence_mask, dtype=np.float32)
    if mask.ndim == 3 and mask.shape[2] == 1:
        mask = mask[:, :, 0]
    if mask.ndim != 2:
        raise ValueError(f"Expected a single-channel confidence mask; got shape {mask.shape}.")

    width, height = size
    # Clean the small model-resolution mask before enlarging it.
    mask = np.nan_to_num(mask, nan=0.0, posinf=1.0, neginf=0.0)
    np.clip(mask, 0.0, 1.0, out=mask)
    if mask.shape != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
    mask = cv2.GaussianBlur(mask, (5, 5), sigmaX=0.8)
    mask -= 0.1
    mask *= 255.0 / 0.7
    np.clip(mask, 0.0, 255.0, out=mask)
    return cv2.convertScaleAbs(mask)


def segment_frame(
    segmenter: mp.tasks.vision.ImageSegmenter,
    frame_bgr: np.ndarray,
    timestamp_ms: int,
    alpha_smoother: TemporalAlphaSmoother | None = None,
) -> np.ndarray:
    height, width = frame_bgr.shape[:2]
    frame_rgba = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGBA)
    image = mp.Image(image_format=mp.ImageFormat.SRGBA, data=frame_rgba)
    result = segmenter.segment_for_video(image, timestamp_ms)
    if not result.confidence_masks:
        raise RuntimeError("人物検出モデルからマスクが返されませんでした。")

    person_confidence = result.confidence_masks[0].numpy_view()
    alpha = confidence_to_alpha(person_confidence, (width, height))
    if alpha_smoother is not None:
        alpha = alpha_smoother.apply(alpha)
    frame_rgba[:, :, 3] = alpha
    return frame_rgba
