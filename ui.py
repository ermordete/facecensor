#!/usr/bin/python3
"""
ui.py – Hauptfenster (vereinfacht)
=====================================
Nur noch Kernfunktionen:
  1. Gesicht-Blur (Echtzeit)
  2. Stärke-Regler
  3. Aufnahme starten / stoppen
  4. Screenshot

Alle Presets, Emoji-Features, Effektauswahl und sonstigen Extras entfernt.
"""

import cv2
import numpy as np
import os
import time
from collections import deque

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QSlider, QFrame, QSizePolicy,
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

from camera import CameraThread
from detector import FaceDetector
from effects import BlurProcessor
from recorder import Recorder


# ═══════════════════════════════════════════════════════════════════════════════
#  FARBEN
# ═══════════════════════════════════════════════════════════════════════════════
C_BG         = "#1F2A36"   # Navy – Hintergrund
C_PANEL      = "#263545"   # Dunkel – Cards / Panels
C_PANEL2     = "#2C3E50"   # Etwas heller – Header / Stats
C_BUTTON     = "#E1DACA"   # Chalk Beige – Buttons
C_BUTTON_HOV = "#EDE9E0"   # Beige hell – Hover
C_BUTTON_ACT = "#CAC6B6"   # Beige dunkel – Pressed
C_TEXT       = "#CBCCBE"   # Sage – Text
C_TEXT_DARK  = "#1F2A36"   # Navy – Text auf Beige-Buttons
C_TEXT_DIM   = "#7A8490"   # Gedimmt – Labels, Hints
C_ACCENT     = "#4A9EBF"   # Blau – aktive Zustände
C_DANGER     = "#BF4A4A"   # Rot – Aufnahme stoppen
C_SUCCESS    = "#4A9E72"   # Grün – Aufnahme starten
C_BORDER     = "#33495C"   # Rahmen
C_VIDEO_BG   = "#0D1720"   # Fast-Schwarz – Kamera-Hintergrund

FONT = '"Noto Sans", "DejaVu Sans", "Liberation Sans", Arial, sans-serif'

# ═══════════════════════════════════════════════════════════════════════════════
#  PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
DETECT_EVERY_N = 3    # Detection nur jeden N-ten Frame → flüssigerer Feed
FPS_WINDOW     = 30   # Gleitender FPS-Durchschnitt über N Frames

# ═══════════════════════════════════════════════════════════════════════════════
#  ZENTRALES STYLESHEET
# ═══════════════════════════════════════════════════════════════════════════════
STYLE = f"""
QMainWindow, QWidget {{
    background-color: {C_BG};
    color: {C_TEXT};
    font-family: {FONT};
    font-size: 13px;
}}
QPushButton {{
    background-color: {C_BUTTON};
    color: {C_TEXT_DARK};
    border: none;
    border-radius: 8px;
    padding: 10px 16px;
    font-family: {FONT};
    font-size: 12px;
    font-weight: 500;
    text-align: left;
}}
QPushButton:hover   {{ background-color: {C_BUTTON_HOV}; }}
QPushButton:pressed {{ background-color: {C_BUTTON_ACT}; }}
QPushButton:disabled {{ background-color: {C_PANEL}; color: {C_TEXT_DIM}; }}
QLabel {{
    background: transparent;
    color: {C_TEXT};
    font-family: {FONT};
}}
QSlider::groove:horizontal {{
    height: 3px;
    background: {C_BORDER};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {C_BUTTON};
    width: 14px; height: 14px;
    margin: -6px 0;
    border-radius: 7px;
    border: none;
}}
QSlider::sub-page:horizontal {{
    background: {C_ACCENT};
    border-radius: 2px;
}}
"""


# ═══════════════════════════════════════════════════════════════════════════════
#  HILFSFUNKTIONEN
# ═══════════════════════════════════════════════════════════════════════════════

def _section_label(text: str) -> QLabel:
    lbl = QLabel(text.upper())
    lbl.setStyleSheet(
        f"color: {C_TEXT_DIM}; font-size: 10px; letter-spacing: 1.5px; "
        f"font-weight: 600; font-family: {FONT}; background: transparent;"
    )
    return lbl


def _card() -> QFrame:
    f = QFrame()
    f.setStyleSheet(
        f"QFrame {{ background-color: {C_PANEL}; border-radius: 10px; "
        f"border: 1px solid {C_BORDER}; }}"
    )
    return f


def _vline() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.VLine)
    f.setFixedWidth(1)
    f.setStyleSheet(f"background: {C_BORDER}; border: none;")
    return f


def _btn_ss(bg: str, fg: str = "#FFFFFF", hover: str = "") -> str:
    h = hover if hover else bg
    return (
        f"QPushButton {{ background-color: {bg}; color: {fg}; border: none; "
        f"border-radius: 8px; padding: 10px 14px; font-family: {FONT}; "
        f"font-size: 12px; font-weight: 600; text-align: left; }} "
        f"QPushButton:hover {{ background-color: {h}; }}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  VERARBEITUNGS-THREAD
# ═══════════════════════════════════════════════════════════════════════════════

class ProcessingThread(QThread):
    """
    Kamera-Frame → Detection → Blur → Signal an UI.

    Performance:
    - Detection nur jeden DETECT_EVERY_N-ten Frame
    - Letzte bekannte Faces dazwischen wiederverwenden
    - Gleitender FPS-Durchschnitt über FPS_WINDOW Frames
    """
    frame_ready = pyqtSignal(np.ndarray, int, float)  # frame, faces, fps

    def __init__(
        self,
        camera: CameraThread,
        detector: FaceDetector,
        blur: BlurProcessor,
        recorder: Recorder,
    ):
        super().__init__()
        self.camera   = camera
        self.detector = detector
        self.blur     = blur
        self.recorder = recorder

        self._running       = True
        self._detect_active = True
        self._frame_count   = 0
        self._last_faces    = []
        self._ts: deque     = deque(maxlen=FPS_WINDOW)

    def run(self):
        while self._running:
            frame = self.camera.get_frame()
            if frame is None:
                time.sleep(0.005)
                continue

            if self._detect_active:
                self._frame_count += 1
                if self._frame_count % DETECT_EVERY_N == 0:
                    self._last_faces = self.detector.detect(frame)
                frame = self.blur.apply(frame, self._last_faces)
                face_count = len(self._last_faces)
            else:
                face_count = 0
                self._last_faces = []

            if self.recorder.is_recording:
                self.recorder.write_frame(frame)

            # Gleitender FPS-Durchschnitt
            now = time.monotonic()
            self._ts.append(now)
            if len(self._ts) >= 2:
                fps = (len(self._ts) - 1) / (self._ts[-1] - self._ts[0])
            else:
                fps = 0.0

            self.frame_ready.emit(frame, face_count, fps)

    def set_detection(self, enabled: bool):
        self._detect_active = enabled
        if not enabled:
            self._last_faces = []

    def stop(self):
        self._running = False
        self.wait()


# ═══════════════════════════════════════════════════════════════════════════════
#  HAUPTFENSTER
# ═══════════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    """
    FaceCensor Pro – vereinfachtes Hauptfenster.
    Drei Panels: Links (Blur-Stärke) | Mitte (Kamera) | Rechts (Aktionen)
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("FaceCensor Pro")
        self.setMinimumSize(900, 600)
        self.resize(1080, 680)
        self.setStyleSheet(STYLE)

        self.camera   = CameraThread(use_csi=True)
        self.detector = FaceDetector()
        self.blur     = BlurProcessor()
        self.recorder = Recorder()

        self._build_ui()

        self.proc = ProcessingThread(self.camera, self.detector, self.blur, self.recorder)
        self.proc.frame_ready.connect(self._on_frame)
        self.camera.start()
        self.proc.start()

        self._status_timer = QTimer(self)
        self._status_timer.timeout.connect(self._update_status)
        self._status_timer.start(1000)

        self._rec_blink = False
        self._rec_timer = QTimer(self)
        self._rec_timer.timeout.connect(self._blink_rec)

    # ── UI-Aufbau ────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        vbox = QVBoxLayout(root)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)

        vbox.addWidget(self._build_header())

        body = QWidget()
        hbox = QHBoxLayout(body)
        hbox.setContentsMargins(14, 14, 14, 10)
        hbox.setSpacing(12)
        hbox.addWidget(self._build_left_panel(), 0)
        hbox.addWidget(self._build_video_area(), 1)
        hbox.addWidget(self._build_right_panel(), 0)
        vbox.addWidget(body, 1)

        vbox.addWidget(self._build_statusbar())

    # ── Header ───────────────────────────────────────────────────────────────

    def _build_header(self) -> QWidget:
        w = QWidget()
        w.setFixedHeight(50)
        w.setStyleSheet(
            f"background-color: {C_PANEL2}; border-bottom: 1px solid {C_BORDER};"
        )
        hbox = QHBoxLayout(w)
        hbox.setContentsMargins(20, 0, 20, 0)

        title = QLabel("FaceCensor Pro")
        title.setStyleSheet(
            f"color: {C_BUTTON}; font-size: 16px; font-weight: 700; "
            f"font-family: {FONT}; background: transparent;"
        )
        sub = QLabel("  ·  Jetson Nano Edition")
        sub.setStyleSheet(
            f"color: {C_TEXT_DIM}; font-size: 12px; font-family: {FONT}; background: transparent;"
        )

        hbox.addWidget(title)
        hbox.addWidget(sub)
        hbox.addStretch()

        self.rec_indicator = QLabel("● AUFNAHME")
        self.rec_indicator.setStyleSheet(
            f"color: {C_DANGER}; font-size: 11px; font-weight: 700; "
            f"font-family: {FONT}; background: transparent;"
        )
        self.rec_indicator.setVisible(False)
        hbox.addWidget(self.rec_indicator)
        return w

    # ── Linkes Panel: Blur-Stärke ─────────────────────────────────────────────

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(200)
        vbox = QVBoxLayout(panel)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(10)

        # ── Blur-Stärke ──
        card = _card()
        cv = QVBoxLayout(card)
        cv.setContentsMargins(14, 14, 14, 16)
        cv.setSpacing(10)

        cv.addWidget(_section_label("Blur-Stärke"))

        self.strength_slider = QSlider(Qt.Horizontal)
        self.strength_slider.setRange(1, 100)
        self.strength_slider.setValue(50)
        self.strength_slider.valueChanged.connect(self._on_strength)

        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(self.strength_slider)

        self.strength_val = QLabel("50")
        self.strength_val.setFixedWidth(26)
        self.strength_val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.strength_val.setStyleSheet(
            f"color: {C_TEXT}; font-size: 12px; font-weight: 600; "
            f"font-family: {FONT}; background: transparent;"
        )
        row.addWidget(self.strength_val)
        cv.addLayout(row)

        # Beschriftung Skala
        scale_row = QHBoxLayout()
        for txt in ("Sanft", "Stark"):
            lbl = QLabel(txt)
            lbl.setStyleSheet(
                f"color: {C_TEXT_DIM}; font-size: 9px; font-family: {FONT}; background: transparent;"
            )
            scale_row.addWidget(lbl)
            if txt == "Sanft":
                scale_row.addStretch()
        cv.addLayout(scale_row)

        vbox.addWidget(card)

        # ── Erkennung ein/aus ──
        det_card = _card()
        dv = QVBoxLayout(det_card)
        dv.setContentsMargins(14, 14, 14, 14)
        dv.setSpacing(6)
        dv.addWidget(_section_label("Erkennung"))

        self.detection_btn = QPushButton("Aktiv")
        self.detection_btn.setCheckable(True)
        self.detection_btn.setChecked(True)
        self.detection_btn.setStyleSheet(
            _btn_ss(C_ACCENT, "#FFFFFF", "#5BAECE")
        )
        self.detection_btn.toggled.connect(self._on_detection_toggled)
        dv.addWidget(self.detection_btn)

        vbox.addWidget(det_card)
        vbox.addStretch()
        return panel

    # ── Video-Bereich ─────────────────────────────────────────────────────────

    def _build_video_area(self) -> QWidget:
        w = QWidget()
        vbox = QVBoxLayout(w)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(8)

        self.video_label = QLabel("Kamera wird gestartet …")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(480, 360)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setStyleSheet(
            f"background-color: {C_VIDEO_BG}; border-radius: 10px; "
            f"border: 1px solid {C_BORDER}; color: {C_TEXT_DIM}; font-size: 13px;"
        )
        vbox.addWidget(self.video_label, 1)
        vbox.addWidget(self._build_stats_bar())
        return w

    def _build_stats_bar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(58)
        bar.setStyleSheet(
            f"background-color: {C_PANEL2}; border-radius: 10px; border: 1px solid {C_BORDER};"
        )
        hbox = QHBoxLayout(bar)
        hbox.setContentsMargins(20, 6, 20, 6)
        hbox.setSpacing(0)

        def stat(label_txt: str, default: str):
            cell = QWidget()
            cell.setStyleSheet("background: transparent;")
            v = QVBoxLayout(cell)
            v.setContentsMargins(0, 0, 0, 0)
            v.setSpacing(1)
            val = QLabel(default)
            val.setAlignment(Qt.AlignCenter)
            val.setStyleSheet(
                f"color: #FFFFFF; font-size: 18px; font-weight: 700; "
                f"font-family: {FONT}; background: transparent;"
            )
            lbl = QLabel(label_txt)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet(
                f"color: {C_TEXT_DIM}; font-size: 9px; letter-spacing: 1.2px; "
                f"font-family: {FONT}; background: transparent;"
            )
            v.addWidget(val)
            v.addWidget(lbl)
            return cell, val

        fps_w,  self.fps_label        = stat("BILDRATE",  "—")
        face_w, self.face_count_label = stat("GESICHTER", "0")

        hbox.addWidget(fps_w)
        hbox.addWidget(_vline())
        hbox.addWidget(face_w)
        return bar

    # ── Rechtes Panel: Aktionen ───────────────────────────────────────────────

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(200)
        vbox = QVBoxLayout(panel)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(10)

        card = _card()
        cv = QVBoxLayout(card)
        cv.setContentsMargins(14, 14, 14, 14)
        cv.setSpacing(8)
        cv.addWidget(_section_label("Aktionen"))

        # Screenshot
        self.screenshot_btn = QPushButton("Screenshot")
        self.screenshot_btn.setStyleSheet(
            _btn_ss(C_BUTTON, C_TEXT_DARK, C_BUTTON_HOV)
        )
        self.screenshot_btn.clicked.connect(self._take_screenshot)
        cv.addWidget(self.screenshot_btn)

        # Aufnahme
        self.record_btn = QPushButton("Aufnahme starten")
        self.record_btn.setStyleSheet(_btn_ss(C_SUCCESS, "#FFFFFF", "#5AAD80"))
        self.record_btn.clicked.connect(self._toggle_recording)
        cv.addWidget(self.record_btn)

        vbox.addWidget(card)
        vbox.addStretch()
        return panel

    # ── Statusleiste ──────────────────────────────────────────────────────────

    def _build_statusbar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(28)
        bar.setStyleSheet(
            f"background-color: {C_PANEL2}; border-top: 1px solid {C_BORDER};"
        )
        hbox = QHBoxLayout(bar)
        hbox.setContentsMargins(16, 0, 16, 0)

        self.status_label = QLabel("Bereit  ·  Kamera wird initialisiert …")
        self.status_label.setStyleSheet(
            f"color: {C_TEXT_DIM}; font-size: 11px; font-family: {FONT}; background: transparent;"
        )
        hint = QLabel("ESC = Beenden   S = Screenshot   R = Aufnahme   Leertaste = Erkennung")
        hint.setStyleSheet(
            f"color: {C_TEXT_DIM}; font-size: 10px; font-family: {FONT}; background: transparent;"
        )
        hbox.addWidget(self.status_label)
        hbox.addStretch()
        hbox.addWidget(hint)
        return bar

    # ── Event Handler ─────────────────────────────────────────────────────────

    def _on_frame(self, frame: np.ndarray, face_count: int, fps: float):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        pixmap = QPixmap.fromImage(
            QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        ).scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.FastTransformation)
        self.video_label.setPixmap(pixmap)

        self.fps_label.setText(f"{fps:.1f}" if fps > 0.5 else "—")
        self.face_count_label.setText(str(face_count))

    def _on_strength(self, value: int):
        self.strength_val.setText(str(value))
        self.blur.set_strength(value)

    def _on_detection_toggled(self, enabled: bool):
        self.proc.set_detection(enabled)
        self.detection_btn.setText("Aktiv" if enabled else "Deaktiviert")
        self.detection_btn.setStyleSheet(
            _btn_ss(C_ACCENT, "#FFFFFF", "#5BAECE") if enabled
            else _btn_ss(C_BUTTON, C_TEXT_DARK, C_BUTTON_HOV)
        )
        self.status_label.setText(
            "Gesichtserkennung aktiviert" if enabled else "Gesichtserkennung deaktiviert"
        )

    def _take_screenshot(self):
        frame = self.camera.get_frame()
        if frame is not None:
            faces = self.detector.detect(frame)
            processed = self.blur.apply(frame.copy(), faces)
            filename = self.recorder.save_screenshot(processed)
            self.status_label.setText(f"Screenshot: {os.path.basename(filename)}")
        else:
            self.status_label.setText("Kein Bild verfügbar")

    def _toggle_recording(self):
        if not self.recorder.is_recording:
            frame = self.camera.get_frame()
            if frame is not None:
                self.recorder.start_recording(frame.shape)
                self.record_btn.setText("Aufnahme stoppen")
                self.record_btn.setStyleSheet(_btn_ss(C_DANGER, "#FFFFFF", "#CC5555"))
                self.rec_indicator.setVisible(True)
                self._rec_timer.start(600)
                self.status_label.setText(
                    f"Aufnahme läuft: {os.path.basename(self.recorder.current_file)}"
                )
        else:
            saved = self.recorder.stop_recording()
            self.record_btn.setText("Aufnahme starten")
            self.record_btn.setStyleSheet(_btn_ss(C_SUCCESS, "#FFFFFF", "#5AAD80"))
            self._rec_timer.stop()
            self.rec_indicator.setVisible(False)
            self.status_label.setText(f"Gespeichert: {os.path.basename(saved)}")

    def _blink_rec(self):
        self._rec_blink = not self._rec_blink
        self.rec_indicator.setVisible(self._rec_blink)

    def _update_status(self):
        err = self.camera.get_error()
        if err:
            self.status_label.setText(f"Kamera-Fehler: {err}")
        elif self.camera.is_running():
            txt = self.status_label.text()
            if any(x in txt for x in ("Bereit", "initialisiert")):
                self.status_label.setText("Kamera aktiv")

    # ── Tastaturkürzel ────────────────────────────────────────────────────────

    def keyPressEvent(self, event):
        k = event.key()
        if k == Qt.Key_Escape:
            self.close()
        elif k == Qt.Key_S:
            self._take_screenshot()
        elif k == Qt.Key_R:
            self._toggle_recording()
        elif k == Qt.Key_Space:
            self.detection_btn.setChecked(not self.detection_btn.isChecked())
        super().keyPressEvent(event)

    # ── Shutdown ──────────────────────────────────────────────────────────────

    def closeEvent(self, event):
        if self.recorder.is_recording:
            self.recorder.stop_recording()
        self.proc.stop()
        self.camera.stop()
        self.camera.join(timeout=2.0)
        event.accept()
