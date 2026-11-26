from __future__ import annotations

import threading
from typing import Callable, Protocol

import numpy as np


class FrameCapture(Protocol):
    def read(self) -> tuple[bool, np.ndarray | None]: ...


class LatestFrameGrabber:
    """カメラ取得を別スレッドで続け、推論側には常に最新の未処理フレームだけを渡す。"""

    def __init__(
        self,
        capture: FrameCapture,
        transform: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        self._capture = capture
        self._transform = transform
        self._condition = threading.Condition()
        self._frame: np.ndarray | None = None
        self._sequence = 0
        self._delivered = 0
        self._error: str | None = None
        self._stopped = False
        self._thread = threading.Thread(target=self._run, name="camera-grabber", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def request_stop(self) -> None:
        with self._condition:
            self._stopped = True
            self._condition.notify_all()

    def stop(self) -> bool:
        """取得スレッドの終了を待つ。終了したら True。"""
        self.request_stop()
        if self._thread.is_alive():
            self._thread.join(timeout=3.0)
        return not self._thread.is_alive()

    def read(self) -> np.ndarray | None:
        """未処理の最新フレームを返す。停止済みなら None、取得に失敗したら RuntimeError。"""
        with self._condition:
            self._condition.wait_for(
                lambda: self._stopped or self._error is not None or self._sequence > self._delivered
            )
            if self._stopped:
                return None
            if self._error is not None:
                raise RuntimeError(self._error)
            self._delivered = self._sequence
            return self._frame

    def _run(self) -> None:
        while True:
            with self._condition:
                if self._stopped:
                    return
            try:
                success, frame = self._capture.read()
                if success and frame is not None and self._transform is not None:
                    frame = self._transform(frame)
            except Exception as error:
                self._publish(None, str(error))
                return
            if not success or frame is None:
                self._publish(None, "カメラから映像を取得できませんでした。")
                return
            self._publish(frame, None)

    def _publish(self, frame: np.ndarray | None, error: str | None) -> None:
        with self._condition:
            if error is not None:
                self._error = error
            else:
                self._frame = frame
                self._sequence += 1
            self._condition.notify_all()
