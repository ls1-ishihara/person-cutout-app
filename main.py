from __future__ import annotations

import sys
import threading
import time
from functools import partial
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from PySide6.QtCore import QByteArray, QPoint, QSettings, QThread, QTimer, Qt, Signal
from PySide6.QtGui import QColor, QImage, QKeySequence, QMouseEvent, QPainter, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSizePolicy,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
)

from effects import (
    PIXEL_ART_BLOCKS,
    SILHOUETTE_MODES,
    TONE_MODES,
    EffectControls,
    EffectProcessor,
)
from frame_grabber import LatestFrameGrabber
from segmentation import TemporalAlphaSmoother, segment_frame


def resource_path(relative_path: str) -> Path:
    bundle_root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return bundle_root / relative_path


class CameraWorker(QThread):
    frame_ready = Signal(object)
    status_changed = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, camera_index: int, model_path: Path, controls: EffectControls) -> None:
        super().__init__()
        self.camera_index = camera_index
        self.model_path = model_path
        self.controls = controls
        self._stop_event = threading.Event()
        self._grabber: LatestFrameGrabber | None = None

    def stop(self) -> None:
        self._stop_event.set()
        grabber = self._grabber
        if grabber is not None:
            grabber.request_stop()

    def run(self) -> None:
        capture = None
        grabber = None
        segmenter = None
        try:
            if not self.model_path.is_file():
                raise FileNotFoundError(f"人物検出モデルが見つかりません: {self.model_path}")

            capture = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            if not capture.isOpened():
                capture.release()
                capture = cv2.VideoCapture(self.camera_index)
            if not capture.isOpened():
                raise RuntimeError(f"カメラ {self.camera_index} を開けません。")

            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            options = mp.tasks.vision.ImageSegmenterOptions(
                base_options=mp.tasks.BaseOptions(model_asset_buffer=self.model_path.read_bytes()),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                output_confidence_masks=True,
                output_category_mask=False,
            )
            segmenter = mp.tasks.vision.ImageSegmenter.create_from_options(options)
            self.status_changed.emit("カメラ起動中・背景を透過しています")

            # 推論中も次のフレームを取得しておき、常に最新フレームだけを処理する
            grabber = LatestFrameGrabber(capture, lambda frame: cv2.flip(frame, 1))
            self._grabber = grabber
            grabber.start()
            if self._stop_event.is_set():
                return

            last_timestamp_ms = 0
            alpha_smoother = TemporalAlphaSmoother()
            effects = EffectProcessor()
            while not self._stop_event.is_set():
                frame_bgr = grabber.read()
                if frame_bgr is None:
                    break

                timestamp_ms = max(time.monotonic_ns() // 1_000_000, last_timestamp_ms + 1)
                last_timestamp_ms = timestamp_ms
                frame_rgba = segment_frame(segmenter, frame_bgr, timestamp_ms, alpha_smoother)
                self.frame_ready.emit(
                    effects.apply(frame_rgba, self.controls.snapshot(), alpha_smoother.change)
                )
        except Exception as error:
            self.error_occurred.emit(str(error))
        finally:
            released = True
            if grabber is not None:
                released = grabber.stop()
            self._grabber = None
            # 取得スレッドが読み取り中のままカメラを解放しない
            if capture is not None and released:
                capture.release()
            if segmenter is not None:
                segmenter.close()


class PreviewLabel(QLabel):
    def __init__(self) -> None:
        super().__init__()
        self._frame: np.ndarray | None = None
        self._image: QImage | None = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(120, 90)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setStyleSheet("background: transparent;")

    def set_frame(self, frame_rgba: np.ndarray) -> None:
        height, width, _ = frame_rgba.shape
        # QImageは配列を所有しないため、描画中は _frame で配列を保持する
        self._frame = frame_rgba
        self._image = QImage(
            frame_rgba.data, width, height, frame_rgba.strides[0], QImage.Format.Format_RGBA8888
        )
        self._update_pixmap()

    def clear_image(self) -> None:
        self._image = None
        self._frame = None
        self.clear()


    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_pixmap()

    def _update_pixmap(self) -> None:
        if self._image is None or self.size().isEmpty():
            return
        self.setPixmap(
            QPixmap.fromImage(self._image).scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )


class OutlineRoot(QWidget):
    """選択中のみウィンドウの輪郭を描画する。
    ウィンドウ内のドラッグで移動、縁のドラッグでサイズ変更を行う。"""

    EDGE = 10

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)

    def _edges(self, pos: QPoint) -> Qt.Edge:
        edges = Qt.Edge(0)
        if pos.x() <= self.EDGE:
            edges |= Qt.Edge.LeftEdge
        elif pos.x() >= self.width() - self.EDGE:
            edges |= Qt.Edge.RightEdge
        if pos.y() <= self.EDGE:
            edges |= Qt.Edge.TopEdge
        elif pos.y() >= self.height() - self.EDGE:
            edges |= Qt.Edge.BottomEdge
        return edges

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        edges = self._edges(event.position().toPoint())
        horizontal = Qt.Edge.LeftEdge | Qt.Edge.RightEdge
        vertical = Qt.Edge.TopEdge | Qt.Edge.BottomEdge
        if edges in (Qt.Edge.LeftEdge | Qt.Edge.TopEdge, Qt.Edge.RightEdge | Qt.Edge.BottomEdge):
            cursor = Qt.CursorShape.SizeFDiagCursor
        elif edges in (Qt.Edge.RightEdge | Qt.Edge.TopEdge, Qt.Edge.LeftEdge | Qt.Edge.BottomEdge):
            cursor = Qt.CursorShape.SizeBDiagCursor
        elif edges & horizontal:
            cursor = Qt.CursorShape.SizeHorCursor
        elif edges & vertical:
            cursor = Qt.CursorShape.SizeVerCursor
        else:
            cursor = Qt.CursorShape.SizeAllCursor
        self.setCursor(cursor)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        handle = self.window().windowHandle()
        if handle is None:
            return
        edges = self._edges(event.position().toPoint())
        if edges:
            handle.startSystemResize(edges)
        else:
            handle.startSystemMove()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
        painter.fillRect(self.rect(), Qt.GlobalColor.transparent)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        active = self.window().isActiveWindow()
        # ほぼ透明な塗りで、透過部分でもマウス操作を受け付ける
        painter.setBrush(QColor(255, 255, 255, 3))
        if active:
            pen = QPen(QColor(59, 130, 246, 255))
            pen.setWidth(3)
            painter.setPen(pen)
        else:
            painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(self.rect().adjusted(2, 2, -2, -2), 12, 12)


class PersonCutoutWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self._worker: CameraWorker | None = None
        self._started = False
        self.effect_controls = EffectControls()
        self.setWindowTitle("人物透過カメラ")
        self.setMinimumSize(240, 180)
        self.resize(800, 600)
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setStyleSheet(
            """
            QMainWindow { background: transparent; }
            QLabel#idleHint {
                color: #f5f6fa;
                background-color: rgba(27, 30, 38, 224);
                border: 1px solid rgba(255, 255, 255, 28);
                border-radius: 10px;
                padding: 14px 20px;
                font-size: 13px;
            }
            """
        )

        root = OutlineRoot(self)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(14, 14, 14, 14)

        preview_container = QWidget()
        preview_container.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        preview_layout = QStackedLayout(preview_container)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setStackingMode(QStackedLayout.StackingMode.StackAll)

        self.preview = PreviewLabel()
        preview_layout.addWidget(self.preview)

        self.idle_hint = QLabel("カメラを起動しています…")
        self.idle_hint.setObjectName("idleHint")
        self.idle_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.idle_hint.setWordWrap(True)
        self.idle_hint.setMaximumWidth(360)
        preview_layout.addWidget(self.idle_hint)
        layout.addWidget(preview_container, 1)
        self.setCentralWidget(root)
        QShortcut(QKeySequence(Qt.Key.Key_Escape), self, activated=self.close)
        QShortcut(QKeySequence(Qt.Key.Key_Backspace), self, activated=self.effect_controls.reset)
        for key, action in (
            (Qt.Key.Key_F1, self.effect_controls.toggle_fade),
            (Qt.Key.Key_F2, self.effect_controls.toggle_negative),
            (Qt.Key.Key_F3, self.effect_controls.cycle_mosaic),
            (Qt.Key.Key_F4, self.effect_controls.toggle_diff_blocks),
        ):
            QShortcut(QKeySequence(key), self, activated=action)
        for key, name, modes in (
            (Qt.Key.Key_F5, "posterize", 0),
            (Qt.Key.Key_F6, "tone", TONE_MODES),
            (Qt.Key.Key_F7, "pixel_art", len(PIXEL_ART_BLOCKS)),
            (Qt.Key.Key_F8, "matrix", 0),
            (Qt.Key.Key_F9, "trail", 0),
            (Qt.Key.Key_F10, "rgb_shift", 3),
            (Qt.Key.Key_F11, "scanlines", 0),
            (Qt.Key.Key_F12, "silhouette", SILHOUETTE_MODES),
        ):
            action = (
                partial(self.effect_controls.cycle, name, modes)
                if modes
                else partial(self.effect_controls.toggle, name)
            )
            QShortcut(QKeySequence(key), self, activated=action)

        self._settings = QSettings("PersonCutout", "PersonCutout")
        geometry = self._settings.value("window/geometry")
        if self._settings.status() != QSettings.Status.NoError:
            QMessageBox.warning(self, "設定の読み込みエラー", "前回の位置とサイズを読み込めませんでした。")
        elif geometry is not None:
            if not isinstance(geometry, QByteArray) or not self.restoreGeometry(geometry):
                QMessageBox.warning(self, "設定の読み込みエラー", "保存された位置とサイズが不正です。")

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._started:
            self._started = True
            QTimer.singleShot(0, self.start_camera)

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() == event.Type.ActivationChange:
            self.centralWidget().update()

    def start_camera(self) -> None:
        if self._worker is not None:
            return
        worker = CameraWorker(
            0, resource_path("models/selfie_segmenter.tflite"), self.effect_controls
        )
        worker.frame_ready.connect(self._show_frame)
        worker.error_occurred.connect(self._show_error)
        worker.finished.connect(self._camera_finished)
        worker.finished.connect(worker.deleteLater)
        self._worker = worker
        worker.start()

    def _show_frame(self, frame_rgba: np.ndarray) -> None:
        self.preview.set_frame(frame_rgba)
        self.idle_hint.hide()

    def _show_error(self, message: str) -> None:
        self.idle_hint.setText(message)
        self.idle_hint.show()

    def _camera_finished(self) -> None:
        self._worker = None
        self.preview.clear_image()

    def closeEvent(self, event) -> None:
        if self._worker is not None:
            worker = self._worker
            worker.stop()
            if not worker.wait(3000):
                event.ignore()
                return
            self._worker = None
        self._settings.setValue("window/geometry", self.saveGeometry())
        self._settings.sync()
        if self._settings.status() != QSettings.Status.NoError:
            QMessageBox.warning(self, "設定の保存エラー", "ウィンドウの位置とサイズを保存できませんでした。")
        event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("人物透過カメラ")
    window = PersonCutoutWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
