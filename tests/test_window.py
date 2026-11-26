import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QByteArray, QPoint, QSettings
from PySide6.QtWidgets import QApplication

from main import PersonCutoutWindow
from effects import EffectSettings


class WindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.settings_path = str(Path(self.directory.name) / "settings.ini")
        settings_patch = patch(
            "main.QSettings",
            side_effect=lambda *args: QSettings(self.settings_path, QSettings.Format.IniFormat),
        )
        self.settings_mock = settings_patch.start()
        self.settings_mock.Status = QSettings.Status
        self.addCleanup(settings_patch.stop)
        camera_patch = patch.object(PersonCutoutWindow, "start_camera")
        camera_patch.start()
        self.addCleanup(camera_patch.stop)

    def create_window(self):
        window = PersonCutoutWindow()
        self.addCleanup(window.close)
        window.show()
        self.app.processEvents()
        return window

    def test_position_and_size_survive_closing_and_reopening(self):
        window = self.create_window()
        available = window.screen().availableGeometry()
        window.resize(320, 240)
        window.move(available.topLeft() + QPoint(40, 50))
        self.app.processEvents()
        expected = window.geometry()
        self.assertTrue(window.close())

        saved = QSettings(self.settings_path, QSettings.Format.IniFormat)
        self.assertIsInstance(saved.value("window/geometry"), QByteArray)
        restored = self.create_window()
        self.assertEqual(restored.geometry(), expected)

    def test_ignored_close_does_not_save_geometry(self):
        window = self.create_window()
        worker = Mock()
        worker.wait.return_value = False
        window._worker = worker
        self.addCleanup(setattr, window, "_worker", None)

        self.assertFalse(window.close())
        worker.stop.assert_called_once()
        saved = QSettings(self.settings_path, QSettings.Format.IniFormat)
        self.assertFalse(saved.contains("window/geometry"))

    def test_function_keys_drive_effects(self):
        from PySide6.QtCore import Qt
        from PySide6.QtTest import QTest

        window = self.create_window()
        window.activateWindow()
        for key in (Qt.Key.Key_F1, Qt.Key.Key_F2, Qt.Key.Key_F3, Qt.Key.Key_F4):
            QTest.keyClick(window, key)
        for key in range(Qt.Key.Key_F5, Qt.Key.Key_F12 + 1):
            QTest.keyClick(window, Qt.Key(key))
        settings = window.effect_controls.snapshot()
        self.assertTrue(
            settings.posterize and settings.pixel_art == 1 and settings.matrix and settings.trail
            and settings.rgb_shift and settings.scanlines
        )
        self.assertEqual((settings.tone, settings.silhouette), (1, 1))

        QTest.keyClick(window, Qt.Key.Key_Backspace)
        self.assertEqual(window.effect_controls.snapshot(), EffectSettings())
        QTest.keyClick(window, Qt.Key.Key_F3)
        self.assertEqual(window.effect_controls.snapshot().mosaic_block, 4)
        self.assertTrue(settings.fade)
        self.assertTrue(settings.negative)
        self.assertEqual(settings.mosaic_block, 4)
        self.assertTrue(settings.diff_blocks)

    def test_outline_is_blue_only_when_active(self):
        window = self.create_window()
        window.idle_hint.hide()
        root = window.centralWidget()
        for active in (True, False):
            with self.subTest(active=active):
                with patch.object(window, "isActiveWindow", return_value=active):
                    image = root.grab().toImage()
                edge = image.pixelColor(2, image.height() // 2)
                if active:
                    self.assertEqual(edge.getRgb(), (59, 130, 246, 255))
                else:
                    self.assertLessEqual(edge.alpha(), 3)

    def test_invalid_saved_geometry_is_reported(self):
        saved = QSettings(self.settings_path, QSettings.Format.IniFormat)
        saved.setValue("window/geometry", QByteArray(b"invalid"))
        saved.sync()
        with patch("main.QMessageBox.warning") as warning:
            self.create_window()
        warning.assert_called_once()


if __name__ == "__main__":
    unittest.main()
