#!/usr/bin/python3
"""
ui.py - Hauptfenster und GUI
==============================
Änderungen v2:
- Alle Emojis aus Button-Beschriftungen und Labels entfernt
- Farben korrigiert: alle Control-Buttons/Panels in Chalk Beige (#E1DACA)
- Entfernte Effekte (Gaussian/Strong Blur) aus UI entfernt
- Performance: Detection nur alle N Frames (DETECT_EVERY_N_FRAMES)
- Performance: Frame-Signal gedrosselt, kein Signal wenn kein neuer Frame
- Performance: FastTransformation statt SmoothTransformation für Pixmap-Scaling
"""

import cv2
import numpy as np
import os

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QSlider, QFrame,
    QSizePolicy, QButtonGroup,
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

from camera import CameraThread
from detector import FaceDetector
from effects import EffectsProcessor, EFFECTS, EMOJIS, PRESETS, DEFAULT_EFFECT
from recorder import Recorder


# ─── Farb-Konstanten ─────────────────────────────────────────────────────────
C_BG         = "#1F2A36"   # Navy Blue - Hintergrund
C_PANEL      = "#263545"   # Panel-Hintergrund (etwas heller)
C_BUTTON     = "#E1DACA"   # Chalk Beige - alle Control-Buttons
C_BUTTON_HOV = "#F0EDE6"   # Heller Beige - Hover
C_BUTTON_ACT = "#C8C4B4"   # Dunkler Beige - Pressed
C_TEXT       = "#CBCCBE"   # Sage - Haupttext
C_TEXT_DIM   = "#8A9099"   # Gedimmter Text
C_ACCENT     = "#4A9EBF"   # Akzentblau (aktiver Effekt, Slider)
C_DANGER     = "#BF4A4A"   # Rot - Stopp/Aufnahme
C_SUCCESS    = "#4ABF7E"   # Grün - Aufnahme starten
C_BORDER     = "#2E3F50"   # Dezente Border

# ─── Performance-Einstellung ─────────────────────────────────────────────────
# Detection nur jeden N-ten Frame ausführen.
# Zwischen den Detection-Frames werden die letzten bekannten Bounding Boxes
# wiederverwendet → deutlich flüssigerer Feed bei gleicher Erkennungsqualität.
# Wert 3 = Detection ~10x/Sek bei 30 FPS. Wert 2 für schnellere Reaktion.
DETECT_EVERY_N_FRAMES = 3

# ─── Stylesheet ──────────────────────────────────────────────────────────────
# Alle Control-Elemente in Chalk Beige, Hintergrund Navy Blue, Text Sage.
GLOBAL_STYLE = f"""
QMainWindow, QWidget {{
    background-color: {C_BG};
    color: {C_TEXT};
    font-family: "Palatino Linotype", "Palatino", "Book Antiqua", Georgia, serif;
    font-size: 13px;
}}

/* Standard-Button: Chalk Beige */
QPushButton {{
    background-color: {C_BUTTON};
    color: #1F2A36;
    border: none;
    border-radius: 10px;
    padding: 9px 14px;
    font-weight: 600;
    font-family: "Palatino Linotype", "Palatino", "Book Antiqua", Georgia, serif;
    font-size: 12px;
    text-align: left;
}}
QPushButton:hover  {{ background-color: {C_BUTTON_HOV}; }}
QPushButton:pressed {{ background-color: {C_BUTTON_ACT}; padding-top: 10px; padding-bottom: 8px; }}
QPushButton:disabled {{ background-color: #3A4858; color: {C_TEXT_DIM}; }}

/* Labels */
QLabel {{ color: {C_TEXT}; background: transparent; }}

/* Slider */
QSlider::groove:horizontal {{
    height: 4px; background: {C_BORDER}; border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {C_BUTTON}; border: none;
    width: 14px; height: 14px; margin: -5px 0; border-radius: 7px;
}}
QSlider::sub-page:horizontal {{ background: {C_ACCENT}; border-radius: 2px; }}

/* Scrollbar */
QScrollBar:vertical {{ background: {C_BG}; width: 6px; border-radius: 3px; }}
QScrollBar::handle:vertical {{ background: {C_BORDER}; border-radius: 3px; min-height: 30px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; }}
"""


# ─── Hilfsfunktionen ─────────────────────────────────────────────────────────

def make_section_label(text: str) -> QLabel:
    lbl = QLabel(text.upper())
    lbl.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 10px; letter-spacing: 2px; background: transparent;")
    return lbl


def panel_style() -> str:
    return f"background: {C_PANEL}; border-radius: 12px; border: 1px solid {C_BORDER};"


def effect_btn_style(checked: bool = False) -> str:
    """Chalk Beige Effekt-Button. Aktiv: Akzentblau."""
    if checked:
        return f"""
            QPushButton {{
                background-color: {C_ACCENT};
                color: #FFFFFF;
                border: none;
                border-radius: 8px;
                padding: 8px 12px;
                text-align: left;
                font-family: "Palatino Linotype", Georgia, serif;
                font-size: 12px;
                font-weight: 600;
            }}
        """
    return f"""
        QPushButton {{
            background-color: {C_BUTTON};
            color: #1F2A36;
            border: none;
            border-radius: 8px;
            padding: 8px 12px;
            text-align: left;
            font-family: "Palatino Linotype", Georgia, serif;
            font-size: 12px;
            font-weight: 500;
        }}
        QPushButton:hover {{ background-color: {C_BUTTON_HOV}; }}
        QPushButton:pressed {{ background-color: {C_BUTTON_ACT}; }}
    """


# ─── Verarbeitungs-Thread ────────────────────────────────────────────────────

class ProcessingThread(QThread):
    """
    Thread für Frame-Verarbeitung (Detection + Effekte).

    Performance-Optimierungen v2:
    A) Detection nur alle DETECT_EVERY_N_FRAMES Frames → entlastet CPU/GPU
    B) Letzte bekannte Faces werden zwischen Detection-Frames wiederverwendet
    C) Kein Signal wenn kein neuer Frame von der Kamera → verhindert leere Updates
    """
    frame_ready = pyqtSignal(np.ndarray, int, float)

    def __init__(
        self,
        camera: CameraThread,
        detector: FaceDetector,
        effects: EffectsProcessor,
        recorder: Recorder,
    ):
        super().__init__()
        self.camera = camera
        self.detector = detector
        self.effects = effects
        self.recorder = recorder
        self._running = True
        self._detection_enabled = True
        self._frame_counter = 0
        # Letzte bekannte Gesichter für Frame-Interpolation
        self._last_faces = []

    def run(self):
        import time
        while self._running:
            frame = self.camera.get_frame()

            # Kein neuer Frame → kurz warten, kein Signal senden
            if frame is None:
                time.sleep(0.005)
                continue

            if self._detection_enabled:
                self._frame_counter += 1

                # PERF: Detection nur jeden N-ten Frame
                if self._frame_counter % DETECT_EVERY_N_FRAMES == 0:
                    self._last_faces = self.detector.detect(frame)

                # Zwischen Detection-Frames: letzte bekannte Faces verwenden
                faces = self._last_faces
                frame = self.effects.apply(frame, faces)
                face_count = len(faces)
            else:
                face_count = 0
                self._last_faces = []

            if self.recorder.is_recording:
                self.recorder.write_frame(frame)

            fps = self.camera.get_fps()
            self.frame_ready.emit(frame, face_count, fps)

    def set_detection_enabled(self, enabled: bool):
        self._detection_enabled = enabled
        if not enabled:
            self._last_faces = []

    def stop(self):
        self._running = False
        self.wait()


# ─── Hauptfenster ────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    """
    Hauptfenster von FaceCensor Pro.
    Layout: [Effekte-Panel] | [Video-Feed] | [Presets/Aktionen-Panel]
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("FaceCensor Pro  ·  Jetson Edition")
        self.setMinimumSize(1080, 680)
        self.resize(1200, 750)
        self.setStyleSheet(GLOBAL_STYLE)

        self.camera = CameraThread(use_csi=True)
        self.detector = FaceDetector()
        self.effects = EffectsProcessor()
        self.recorder = Recorder()

        # Referenzen auf Effekt-Buttons für programmatischen Check-Update
        self._effect_buttons: dict = {}

        self._build_ui()

        self.proc_thread = ProcessingThread(
            self.camera, self.detector, self.effects, self.recorder
        )
        self.proc_thread.frame_ready.connect(self._on_frame_ready)

        self.camera.start()
        self.proc_thread.start()

        self._status_timer = QTimer()
        self._status_timer.timeout.connect(self._check_camera_status)
        self._status_timer.start(1000)

        self._rec_blink = False
        self._rec_timer = QTimer()
        self._rec_timer.timeout.connect(self._blink_rec_indicator)

    # ─── UI-Aufbau ───────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        main_layout.addWidget(self._build_header())

        content = QWidget()
        cl = QHBoxLayout(content)
        cl.setContentsMargins(12, 12, 12, 8)
        cl.setSpacing(12)
        cl.addWidget(self._build_left_panel(), 0)
        cl.addWidget(self._build_video_area(), 1)
        cl.addWidget(self._build_right_panel(), 0)
        main_layout.addWidget(content, 1)

        main_layout.addWidget(self._build_status_bar())

    def _build_header(self) -> QWidget:
        header = QWidget()
        header.setFixedHeight(56)
        header.setStyleSheet(f"background: {C_PANEL}; border-bottom: 1px solid {C_BORDER};")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(20, 0, 20, 0)

        title = QLabel("FaceCensor Pro")
        title.setStyleSheet(f"""
            color: {C_BUTTON};
            font-size: 18px; font-weight: bold;
            font-family: "Palatino Linotype", Georgia, serif;
            letter-spacing: 1px;
        """)
        subtitle = QLabel("Content Creator Edition  ·  Jetson Nano")
        subtitle.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 11px; padding-left: 10px;")

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addStretch()

        self.rec_indicator = QLabel("REC")
        self.rec_indicator.setStyleSheet(
            f"color: {C_DANGER}; font-size: 12px; font-weight: bold; padding-right: 4px;"
        )
        self.rec_indicator.setVisible(False)
        layout.addWidget(self.rec_indicator)
        return header

    def _build_video_area(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.video_label = QLabel("Kamera wird gestartet...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(480, 360)
        self.video_label.setStyleSheet(f"""
            background-color: #0D1720;
            border-radius: 12px;
            border: 1px solid {C_BORDER};
            color: {C_TEXT_DIM};
            font-size: 14px;
        """)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.video_label, 1)
        layout.addWidget(self._build_stats_row())
        return container

    def _build_stats_row(self) -> QWidget:
        row = QWidget()
        row.setFixedHeight(64)
        row.setStyleSheet(f"background: {C_PANEL}; border-radius: 10px; border: 1px solid {C_BORDER};")
        layout = QHBoxLayout(row)
        layout.setContentsMargins(16, 8, 16, 8)
        layout.setSpacing(0)

        def stat_widget(label: str, default: str):
            w = QWidget()
            v = QVBoxLayout(w)
            v.setContentsMargins(0, 0, 0, 0)
            v.setSpacing(1)
            val = QLabel(default)
            val.setStyleSheet("color: #FFFFFF; font-size: 20px; font-weight: bold; background: transparent;")
            val.setAlignment(Qt.AlignCenter)
            lbl = QLabel(label)
            lbl.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 10px; letter-spacing: 1px; background: transparent;")
            lbl.setAlignment(Qt.AlignCenter)
            v.addWidget(val)
            v.addWidget(lbl)
            return w, val

        def vline():
            l = QFrame()
            l.setFrameShape(QFrame.VLine)
            l.setStyleSheet(f"background: {C_BORDER}; max-width: 1px;")
            return l

        fps_w, self.fps_label = stat_widget("FPS", "—")
        face_w, self.face_count_label = stat_widget("GESICHTER", "0")
        mode_w, self.mode_label = stat_widget("MODUS", "—")

        layout.addWidget(fps_w)
        layout.addWidget(vline())
        layout.addWidget(face_w)
        layout.addWidget(vline())
        layout.addWidget(mode_w)
        return row

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(210)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # ── Effekte ──
        effects_frame = QFrame()
        effects_frame.setStyleSheet(panel_style())
        ef_layout = QVBoxLayout(effects_frame)
        ef_layout.setContentsMargins(12, 12, 12, 12)
        ef_layout.setSpacing(5)
        ef_layout.addWidget(make_section_label("Effekte"))

        self.effect_btn_group = QButtonGroup()
        self.effect_btn_group.setExclusive(True)

        first_btn = None
        for effect_id, info in EFFECTS.items():
            btn = QPushButton(info["name"])
            btn.setCheckable(True)
            # Standard-Effekt initial aktivieren
            is_default = (effect_id == DEFAULT_EFFECT)
            btn.setChecked(is_default)
            btn.setStyleSheet(effect_btn_style(checked=is_default))
            btn.clicked.connect(
                lambda checked, eid=effect_id, b=btn: self._on_effect_clicked(eid, b)
            )
            self.effect_btn_group.addButton(btn)
            self._effect_buttons[effect_id] = btn
            ef_layout.addWidget(btn)
            if first_btn is None:
                first_btn = btn

        layout.addWidget(effects_frame)

        # ── Stärke-Slider ──
        strength_frame = QFrame()
        strength_frame.setStyleSheet(panel_style())
        sl_layout = QVBoxLayout(strength_frame)
        sl_layout.setContentsMargins(12, 12, 12, 12)
        sl_layout.setSpacing(6)
        sl_layout.addWidget(make_section_label("Staerke"))

        slider_row = QHBoxLayout()
        self.strength_slider = QSlider(Qt.Horizontal)
        self.strength_slider.setRange(10, 100)
        self.strength_slider.setValue(50)
        self.strength_slider.valueChanged.connect(self._on_strength_changed)

        self.strength_value_label = QLabel("50")
        self.strength_value_label.setFixedWidth(28)
        self.strength_value_label.setStyleSheet(
            f"color: {C_TEXT}; font-size: 12px; font-weight: bold;"
        )

        slider_row.addWidget(self.strength_slider)
        slider_row.addWidget(self.strength_value_label)
        sl_layout.addLayout(slider_row)
        layout.addWidget(strength_frame)

        layout.addStretch()
        return panel

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(210)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # ── Presets ──
        preset_frame = QFrame()
        preset_frame.setStyleSheet(panel_style())
        pr_layout = QVBoxLayout(preset_frame)
        pr_layout.setContentsMargins(12, 12, 12, 12)
        pr_layout.setSpacing(5)
        pr_layout.addWidget(make_section_label("Presets"))

        self.preset_btn_group = QButtonGroup()
        self.preset_btn_group.setExclusive(True)

        for pid, pinfo in PRESETS.items():
            btn = QPushButton(pinfo["name"])
            btn.setCheckable(True)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {C_BUTTON};
                    color: #1F2A36;
                    border: none;
                    border-radius: 8px;
                    padding: 8px 12px;
                    font-family: "Palatino Linotype", Georgia, serif;
                    font-size: 12px;
                    font-weight: 500;
                    text-align: left;
                }}
                QPushButton:hover {{ background-color: {C_BUTTON_HOV}; }}
                QPushButton:pressed {{ background-color: {C_BUTTON_ACT}; }}
                QPushButton:checked {{
                    background-color: {C_SUCCESS};
                    color: #FFFFFF;
                    font-weight: 600;
                }}
            """)
            btn.clicked.connect(lambda checked, p=pid: self._apply_preset(p))
            self.preset_btn_group.addButton(btn)
            pr_layout.addWidget(btn)

        layout.addWidget(preset_frame)

        # ── Emoji-Auswahl ──
        emoji_frame = QFrame()
        emoji_frame.setStyleSheet(panel_style())
        em_layout = QVBoxLayout(emoji_frame)
        em_layout.setContentsMargins(12, 12, 12, 12)
        em_layout.setSpacing(5)
        em_layout.addWidget(make_section_label("Emoji Overlay"))

        self.emoji_btn_group = QButtonGroup()
        self.emoji_btn_group.setExclusive(True)

        emoji_items = list(EMOJIS.items())
        row1 = QHBoxLayout()
        row1.setSpacing(4)
        row2 = QHBoxLayout()
        row2.setSpacing(4)

        for i, (eid, einfo) in enumerate(emoji_items):
            # Keine Emojis im Button-Label: nur der Name (Cool, Laugh, …)
            btn = QPushButton(einfo["name"])
            btn.setCheckable(True)
            btn.setFixedHeight(32)
            btn.setToolTip(einfo["name"])
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {C_BUTTON};
                    color: #1F2A36;
                    border: none;
                    border-radius: 6px;
                    padding: 4px 6px;
                    font-size: 10px;
                    font-weight: 500;
                    text-align: center;
                    font-family: "Palatino Linotype", Georgia, serif;
                }}
                QPushButton:hover {{ background-color: {C_BUTTON_HOV}; }}
                QPushButton:checked {{
                    background-color: {C_ACCENT};
                    color: #FFFFFF;
                }}
            """)
            btn.clicked.connect(lambda checked, e=eid: self.effects.set_emoji(e))
            self.emoji_btn_group.addButton(btn)
            if i < 4:
                row1.addWidget(btn)
            else:
                row2.addWidget(btn)

        em_layout.addLayout(row1)
        em_layout.addLayout(row2)
        layout.addWidget(emoji_frame)

        # ── Aktionen ──
        action_frame = QFrame()
        action_frame.setStyleSheet(panel_style())
        ac_layout = QVBoxLayout(action_frame)
        ac_layout.setContentsMargins(12, 12, 12, 12)
        ac_layout.setSpacing(8)
        ac_layout.addWidget(make_section_label("Aktionen"))

        screenshot_btn = QPushButton("Screenshot")
        screenshot_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {C_BUTTON};
                color: #1F2A36;
                border: none; border-radius: 8px;
                padding: 9px 12px; font-weight: 600;
                font-family: "Palatino Linotype", Georgia, serif;
                font-size: 12px; text-align: left;
            }}
            QPushButton:hover {{ background-color: {C_BUTTON_HOV}; }}
            QPushButton:pressed {{ background-color: {C_BUTTON_ACT}; }}
        """)
        screenshot_btn.clicked.connect(self._take_screenshot)
        ac_layout.addWidget(screenshot_btn)

        self.record_btn = QPushButton("Aufnahme starten")
        self._set_record_btn_style(recording=False)
        self.record_btn.clicked.connect(self._toggle_recording)
        ac_layout.addWidget(self.record_btn)

        layout.addWidget(action_frame)

        # ── Erkennung ──
        detect_frame = QFrame()
        detect_frame.setStyleSheet(panel_style())
        dt_layout = QVBoxLayout(detect_frame)
        dt_layout.setContentsMargins(12, 12, 12, 12)
        dt_layout.setSpacing(6)
        dt_layout.addWidget(make_section_label("Erkennung"))

        self.detection_btn = QPushButton("Aktiv")
        self.detection_btn.setCheckable(True)
        self.detection_btn.setChecked(True)
        self._set_detection_btn_style(active=True)
        self.detection_btn.toggled.connect(self._on_detection_toggled)
        dt_layout.addWidget(self.detection_btn)

        layout.addWidget(detect_frame)
        layout.addStretch()
        return panel

    def _build_status_bar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(30)
        bar.setStyleSheet(f"background: {C_PANEL}; border-top: 1px solid {C_BORDER};")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(16, 0, 16, 0)

        self.status_label = QLabel("Bereit  ·  Kamera wird initialisiert...")
        self.status_label.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 11px;")

        hint = QLabel("ESC = Beenden   S = Screenshot   R = Aufnahme   Leertaste = Erkennung")
        hint.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 10px;")

        layout.addWidget(self.status_label)
        layout.addStretch()
        layout.addWidget(hint)
        return bar

    # ─── Style-Helfer ────────────────────────────────────────────────────────

    def _set_record_btn_style(self, recording: bool):
        if recording:
            self.record_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {C_DANGER};
                    color: #FFFFFF; border: none; border-radius: 8px;
                    padding: 9px 12px; font-weight: 700;
                    font-family: "Palatino Linotype", Georgia, serif;
                    font-size: 12px; text-align: left;
                }}
                QPushButton:hover {{ background-color: #CF5A5A; }}
            """)
        else:
            self.record_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {C_SUCCESS};
                    color: #FFFFFF; border: none; border-radius: 8px;
                    padding: 9px 12px; font-weight: 700;
                    font-family: "Palatino Linotype", Georgia, serif;
                    font-size: 12px; text-align: left;
                }}
                QPushButton:hover {{ background-color: #5ACFA0; }}
            """)

    def _set_detection_btn_style(self, active: bool):
        if active:
            self.detection_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {C_ACCENT};
                    color: #FFFFFF; border: none; border-radius: 8px;
                    padding: 8px 12px; font-weight: 600;
                    font-family: "Palatino Linotype", Georgia, serif;
                    font-size: 12px; text-align: left;
                }}
                QPushButton:hover {{ background-color: #5AB0CF; }}
            """)
        else:
            self.detection_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {C_BUTTON};
                    color: #1F2A36; border: none; border-radius: 8px;
                    padding: 8px 12px; font-weight: 500;
                    font-family: "Palatino Linotype", Georgia, serif;
                    font-size: 12px; text-align: left;
                }}
                QPushButton:hover {{ background-color: {C_BUTTON_HOV}; }}
            """)

    # ─── Event Handler ────────────────────────────────────────────────────────

    def _on_effect_clicked(self, effect_id: str, clicked_btn: QPushButton):
        """Setzt Effekt und aktualisiert alle Button-Stile."""
        self.effects.set_effect(effect_id)
        for eid, btn in self._effect_buttons.items():
            btn.setStyleSheet(effect_btn_style(checked=(eid == effect_id)))

    def _on_frame_ready(self, frame: np.ndarray, face_count: int, fps: float):
        """
        Empfängt verarbeiteten Frame und zeigt ihn im Video-Label an.

        PERF: FastTransformation statt SmoothTransformation — auf Jetson Nano
        deutlich schneller bei minimalem Qualitätsunterschied im Live-Feed.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        q_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(q_img)
        scaled = pixmap.scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.FastTransformation,   # PERF: schneller als SmoothTransformation
        )
        self.video_label.setPixmap(scaled)

        self.fps_label.setText(f"{fps:.0f}")
        self.face_count_label.setText(str(face_count))
        self.mode_label.setText(self.effects.get_effect_name())

    def _on_strength_changed(self, value: int):
        self.strength_value_label.setText(str(value))
        self.effects.set_strength(value)

    def _apply_preset(self, preset_id: str):
        self.effects.apply_preset(preset_id)
        # Stärke-Slider synchronisieren
        preset = PRESETS.get(preset_id, {})
        if "strength" in preset:
            self.strength_slider.setValue(preset["strength"])
        # Effekt-Button-Stile aktualisieren
        active_effect = self.effects.current_effect
        for eid, btn in self._effect_buttons.items():
            btn.setStyleSheet(effect_btn_style(checked=(eid == active_effect)))
        self.status_label.setText(f"Preset aktiviert: {preset.get('name', preset_id)}")

    def _on_detection_toggled(self, enabled: bool):
        self.proc_thread.set_detection_enabled(enabled)
        self.detection_btn.setText("Aktiv" if enabled else "Deaktiviert")
        self._set_detection_btn_style(active=enabled)
        self.status_label.setText(
            f"Gesichtserkennung {'aktiviert' if enabled else 'deaktiviert'}"
        )

    def _take_screenshot(self):
        frame = self.camera.get_frame()
        if frame is not None:
            faces = self.detector.detect(frame)
            processed = self.effects.apply(frame.copy(), faces)
            filename = self.recorder.save_screenshot(processed)
            self.status_label.setText(f"Screenshot gespeichert: {os.path.basename(filename)}")
        else:
            self.status_label.setText("Kein Frame fuer Screenshot verfuegbar")

    def _toggle_recording(self):
        if not self.recorder.is_recording:
            frame = self.camera.get_frame()
            if frame is not None:
                self.recorder.start_recording(frame.shape)
                self.record_btn.setText("Aufnahme stoppen")
                self._set_record_btn_style(recording=True)
                self.rec_indicator.setVisible(True)
                self._rec_timer.start(600)
                self.status_label.setText(
                    f"Aufnahme laeuft: {os.path.basename(self.recorder.current_file)}"
                )
        else:
            saved_file = self.recorder.stop_recording()
            self.record_btn.setText("Aufnahme starten")
            self._set_record_btn_style(recording=False)
            self._rec_timer.stop()
            self.rec_indicator.setVisible(False)
            self.status_label.setText(
                f"Aufnahme gespeichert: {os.path.basename(saved_file)}"
            )

    def _blink_rec_indicator(self):
        self._rec_blink = not self._rec_blink
        self.rec_indicator.setVisible(self._rec_blink)

    def _check_camera_status(self):
        if self.camera.get_error():
            self.status_label.setText(f"Kamera-Fehler: {self.camera.get_error()}")
        elif self.camera.is_running():
            if any(x in self.status_label.text() for x in ("Bereit", "initialisiert")):
                self.status_label.setText(f"Kamera aktiv  ·  {self.camera.get_fps():.0f} FPS")

    # ─── Keyboard Shortcuts ───────────────────────────────────────────────────

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Escape:
            self.close()
        elif key == Qt.Key_S:
            self._take_screenshot()
        elif key == Qt.Key_R:
            self._toggle_recording()
        elif key == Qt.Key_Space:
            self.detection_btn.setChecked(not self.detection_btn.isChecked())
        super().keyPressEvent(event)

    # ─── Shutdown ─────────────────────────────────────────────────────────────

    def closeEvent(self, event):
        print("FaceCensor Pro wird beendet...")
        if self.recorder.is_recording:
            self.recorder.stop_recording()
        self.proc_thread.stop()
        self.camera.stop()
        self.camera.join(timeout=2.0)
        print("Shutdown abgeschlossen")
        event.accept()
