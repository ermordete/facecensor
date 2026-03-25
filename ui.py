#!/usr/bin/python3
"""
ui.py – Hauptfenster und GUI
==============================
Version 3 – Finale Überarbeitung:
- Vollständig deutsche Oberfläche
- Farben konsequent über zentrales QSS durchgesetzt
  (Navy #1F2A36 · Beige #E1DACA · Sage #CBCCBE)
- Schrift: Noto Sans / DejaVu Sans / Liberation Sans (sauber, modern, lesbar)
- FPS: gleitender Durchschnitt über 30 Frames im ProcessingThread
- Kein Palatino / keine handschriftlichen Fonts mehr
- Alle inline-Styles auf das zentrale GLOBAL_STYLE-QSS reduziert
- Ästhetik: clean, minimalistisch, professionell
"""

import cv2
import numpy as np
import os
import time
from collections import deque

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QSlider, QFrame,
    QSizePolicy, QButtonGroup,
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFontDatabase, QFont

from camera import CameraThread
from detector import FaceDetector
from effects import EffectsProcessor, EFFECTS, EMOJIS, PRESETS, DEFAULT_EFFECT
from recorder import Recorder


# ═══════════════════════════════════════════════════════════════════════════════
#  FARB-KONSTANTEN  (einmalig definiert, überall referenziert)
# ═══════════════════════════════════════════════════════════════════════════════
C_BG         = "#1F2A36"   # Navy Blue     – App-Hintergrund
C_PANEL      = "#263545"   # Dunkel-Panel  – Seitenleisten, Cards
C_PANEL2     = "#2C3E50"   # Etwas heller  – Stats-Bar, Header
C_BUTTON     = "#E1DACA"   # Chalk Beige   – alle Bedien-Buttons
C_BUTTON_HOV = "#EDE9E0"   # Beige hell    – Hover-Zustand
C_BUTTON_ACT = "#CAC6B6"   # Beige dunkel  – Pressed-Zustand
C_TEXT       = "#CBCCBE"   # Sage          – Fließtext, Labels
C_TEXT_DIM   = "#7A8490"   # Gedimmt       – Unter-Labels, Hints
C_TEXT_DARK  = "#1F2A36"   # Navy          – Text auf Beige-Buttons
C_ACCENT     = "#4A9EBF"   # Blau          – aktiver Effekt, Slider
C_DANGER     = "#BF4A4A"   # Rot           – Aufnahme stoppen
C_SUCCESS    = "#4A9E72"   # Grün          – Aufnahme starten
C_BORDER     = "#33495C"   # Border        – Panel-Rahmen
C_VIDEO_BG   = "#0D1720"   # Fast-Schwarz  – Kamera-Hintergrund

# ═══════════════════════════════════════════════════════════════════════════════
#  SCHRIFTART  – robuste Linux-Fallback-Kette, kein Palatino
# ═══════════════════════════════════════════════════════════════════════════════
FONT_STACK = '"Noto Sans", "DejaVu Sans", "Liberation Sans", "Ubuntu", Arial, sans-serif'

# ═══════════════════════════════════════════════════════════════════════════════
#  PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
DETECT_EVERY_N_FRAMES = 3   # Detection nur jeden 3. Frame → flüssigerer Feed
FPS_WINDOW = 30             # Gleitender FPS-Durchschnitt über N Frames


# ═══════════════════════════════════════════════════════════════════════════════
#  ZENTRALES STYLESHEET  (QSS)
#  Alle Farben, Fonts und Abstände sind hier definiert.
#  Einzelne Widgets überschreiben NUR was wirklich abweicht.
# ═══════════════════════════════════════════════════════════════════════════════
GLOBAL_STYLE = f"""

/* ── Basis ── */
QMainWindow, QWidget {{
    background-color: {C_BG};
    color: {C_TEXT};
    font-family: {FONT_STACK};
    font-size: 13px;
}}

/* ── Alle Buttons: Chalk Beige Hintergrund, Sage Text ── */
QPushButton {{
    background-color: {C_BUTTON};
    color: {C_TEXT_DARK};
    border: none;
    border-radius: 8px;
    padding: 9px 14px;
    font-family: {FONT_STACK};
    font-size: 12px;
    font-weight: 500;
    text-align: left;
    outline: none;
}}
QPushButton:hover {{
    background-color: {C_BUTTON_HOV};
}}
QPushButton:pressed {{
    background-color: {C_BUTTON_ACT};
}}
QPushButton:disabled {{
    background-color: {C_PANEL};
    color: {C_TEXT_DIM};
}}

/* ── Aktiver Effekt-Button ── */
QPushButton[active="true"] {{
    background-color: {C_ACCENT};
    color: #FFFFFF;
    font-weight: 600;
}}

/* ── Aufnahme-Button (Rot) ── */
QPushButton[danger="true"] {{
    background-color: {C_DANGER};
    color: #FFFFFF;
    font-weight: 600;
}}
QPushButton[danger="true"]:hover {{
    background-color: #CC5555;
}}

/* ── Start-Button (Grün) ── */
QPushButton[success="true"] {{
    background-color: {C_SUCCESS};
    color: #FFFFFF;
    font-weight: 600;
}}
QPushButton[success="true"]:hover {{
    background-color: #5AAD80;
}}

/* ── Erkennung aktiv ── */
QPushButton[detection="active"] {{
    background-color: {C_ACCENT};
    color: #FFFFFF;
    font-weight: 600;
}}
QPushButton[detection="inactive"] {{
    background-color: {C_BUTTON};
    color: {C_TEXT_DARK};
}}

/* ── Labels ── */
QLabel {{
    background: transparent;
    color: {C_TEXT};
    font-family: {FONT_STACK};
}}

/* ── Abschnitts-Überschriften ── */
QLabel[role="section"] {{
    color: {C_TEXT_DIM};
    font-size: 10px;
    letter-spacing: 1.5px;
    font-weight: 600;
    text-transform: uppercase;
}}

/* ── Stat-Wert (groß) ── */
QLabel[role="stat-value"] {{
    color: #FFFFFF;
    font-size: 20px;
    font-weight: 700;
    font-family: {FONT_STACK};
}}

/* ── Stat-Label (klein, gedimmt) ── */
QLabel[role="stat-label"] {{
    color: {C_TEXT_DIM};
    font-size: 10px;
    letter-spacing: 1px;
}}

/* ── Panel / Card ── */
QFrame[role="panel"] {{
    background-color: {C_PANEL};
    border-radius: 10px;
    border: 1px solid {C_BORDER};
}}

/* ── Slider ── */
QSlider::groove:horizontal {{
    height: 3px;
    background: {C_BORDER};
    border-radius: 2px;
    margin: 0px;
}}
QSlider::handle:horizontal {{
    background: {C_BUTTON};
    width: 14px;
    height: 14px;
    margin: -6px 0;
    border-radius: 7px;
    border: none;
}}
QSlider::sub-page:horizontal {{
    background: {C_ACCENT};
    border-radius: 2px;
}}

/* ── Scrollbar (schmal, dezent) ── */
QScrollBar:vertical {{
    background: transparent;
    width: 5px;
}}
QScrollBar::handle:vertical {{
    background: {C_BORDER};
    border-radius: 2px;
    min-height: 20px;
}}
QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar::add-page:vertical,
QScrollBar::sub-page:vertical {{ background: transparent; }}
"""


# ═══════════════════════════════════════════════════════════════════════════════
#  HILFSFUNKTIONEN
# ═══════════════════════════════════════════════════════════════════════════════

def section_label(text: str) -> QLabel:
    """Abschnitts-Überschrift im gedimmten Stil."""
    lbl = QLabel(text.upper())
    lbl.setProperty("role", "section")
    # setProperty allein reicht nicht zum Neu-Rendern → explizit Style nochmal setzen
    lbl.setStyleSheet(
        f"color: {C_TEXT_DIM}; font-size: 10px; letter-spacing: 1.5px; "
        f"font-weight: 600; font-family: {FONT_STACK}; background: transparent;"
    )
    return lbl


def set_btn_prop(btn: QPushButton, prop: str, value: str):
    """Setzt eine Qt-Property und erzwingt Stylesheet-Neuberechnung."""
    btn.setProperty(prop, value)
    btn.style().unpolish(btn)
    btn.style().polish(btn)
    btn.update()


def h_divider() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setFixedHeight(1)
    line.setStyleSheet(f"background: {C_BORDER}; border: none;")
    return line


def v_divider() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.VLine)
    line.setFixedWidth(1)
    line.setStyleSheet(f"background: {C_BORDER}; border: none;")
    return line


# ═══════════════════════════════════════════════════════════════════════════════
#  VERARBEITUNGS-THREAD
# ═══════════════════════════════════════════════════════════════════════════════

class ProcessingThread(QThread):
    """
    Verarbeitung (Detection + Effekte) im eigenen Thread.

    FPS-Berechnung:
    - Gleitender Durchschnitt über FPS_WINDOW Frames
    - Misst tatsächliche Verarbeitungsrate (nicht Kamera-Rohrate)
    - Stabile Anzeige ohne Sprünge
    """
    # frame, Gesichtsanzahl, FPS (geglättet)
    frame_ready = pyqtSignal(np.ndarray, int, float)

    def __init__(
        self,
        camera: CameraThread,
        detector: FaceDetector,
        effects: EffectsProcessor,
        recorder: Recorder,
    ):
        super().__init__()
        self.camera   = camera
        self.detector = detector
        self.effects  = effects
        self.recorder = recorder

        self._running           = True
        self._detection_enabled = True
        self._frame_counter     = 0
        self._last_faces        = []

        # Gleitender FPS-Puffer: speichert Zeitstempel der letzten N Frames
        self._ts_buffer: deque = deque(maxlen=FPS_WINDOW)

    def run(self):
        while self._running:
            frame = self.camera.get_frame()
            if frame is None:
                time.sleep(0.005)
                continue

            # ── Gesichtserkennung (gedrosselt) ──
            if self._detection_enabled:
                self._frame_counter += 1
                if self._frame_counter % DETECT_EVERY_N_FRAMES == 0:
                    self._last_faces = self.detector.detect(frame)
                faces = self._last_faces
                frame = self.effects.apply(frame, faces)
                face_count = len(faces)
            else:
                face_count = 0
                self._last_faces = []

            # ── Aufnahme ──
            if self.recorder.is_recording:
                self.recorder.write_frame(frame)

            # ── FPS: gleitender Durchschnitt ──
            now = time.monotonic()
            self._ts_buffer.append(now)
            if len(self._ts_buffer) >= 2:
                elapsed = self._ts_buffer[-1] - self._ts_buffer[0]
                smooth_fps = (len(self._ts_buffer) - 1) / elapsed if elapsed > 0 else 0.0
            else:
                smooth_fps = 0.0

            self.frame_ready.emit(frame, face_count, smooth_fps)

    def set_detection_enabled(self, enabled: bool):
        self._detection_enabled = enabled
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
    FaceCensor Pro – Hauptfenster.
    Layout: [Effekte-Panel links] | [Kamera-Feed] | [Aktions-Panel rechts]
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("FaceCensor Pro")
        self.setMinimumSize(1060, 660)
        self.resize(1180, 730)

        # Zentrales Stylesheet einmalig setzen – gilt für ALLE Kind-Widgets
        self.setStyleSheet(GLOBAL_STYLE)

        # Kern-Objekte
        self.camera   = CameraThread(use_csi=True)
        self.detector = FaceDetector()
        self.effects  = EffectsProcessor()
        self.recorder = Recorder()

        # Button-Referenzen für programmatische Updates
        self._effect_buttons: dict = {}

        self._build_ui()

        # Verarbeitungsthread
        self.proc_thread = ProcessingThread(
            self.camera, self.detector, self.effects, self.recorder
        )
        self.proc_thread.frame_ready.connect(self._on_frame_ready)

        self.camera.start()
        self.proc_thread.start()

        # Status-Prüf-Timer (1 Sek.)
        self._status_timer = QTimer(self)
        self._status_timer.timeout.connect(self._check_camera_status)
        self._status_timer.start(1000)

        # REC-Blink-Timer
        self._rec_blink = False
        self._rec_timer = QTimer(self)
        self._rec_timer.timeout.connect(self._blink_rec_indicator)

    # ── UI-Aufbau ────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        vbox = QVBoxLayout(root)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)

        vbox.addWidget(self._build_header())

        # Inhaltsbereich
        content = QWidget()
        hbox = QHBoxLayout(content)
        hbox.setContentsMargins(14, 14, 14, 10)
        hbox.setSpacing(12)
        hbox.addWidget(self._build_left_panel(), 0)
        hbox.addWidget(self._build_video_area(), 1)
        hbox.addWidget(self._build_right_panel(), 0)
        vbox.addWidget(content, 1)

        vbox.addWidget(self._build_statusbar())

    # ── Header ───────────────────────────────────────────────────────────────

    def _build_header(self) -> QWidget:
        w = QWidget()
        w.setFixedHeight(52)
        w.setStyleSheet(
            f"background-color: {C_PANEL2}; border-bottom: 1px solid {C_BORDER};"
        )
        hbox = QHBoxLayout(w)
        hbox.setContentsMargins(20, 0, 20, 0)
        hbox.setSpacing(0)

        title = QLabel("FaceCensor Pro")
        title.setStyleSheet(
            f"color: {C_BUTTON}; font-size: 16px; font-weight: 700; "
            f"font-family: {FONT_STACK}; background: transparent;"
        )

        sub = QLabel("  ·  Jetson Nano Edition")
        sub.setStyleSheet(
            f"color: {C_TEXT_DIM}; font-size: 12px; font-family: {FONT_STACK}; background: transparent;"
        )

        hbox.addWidget(title)
        hbox.addWidget(sub)
        hbox.addStretch()

        self.rec_indicator = QLabel("● AUFNAHME")
        self.rec_indicator.setStyleSheet(
            f"color: {C_DANGER}; font-size: 11px; font-weight: 700; "
            f"font-family: {FONT_STACK}; background: transparent;"
        )
        self.rec_indicator.setVisible(False)
        hbox.addWidget(self.rec_indicator)
        return w

    # ── Video-Bereich ─────────────────────────────────────────────────────────

    def _build_video_area(self) -> QWidget:
        w = QWidget()
        vbox = QVBoxLayout(w)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(8)

        # Kamera-Vorschau
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
        """Drei Kennzahlen: Bildrate / Gesichter / Effekt."""
        bar = QWidget()
        bar.setFixedHeight(62)
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
                f"color: #FFFFFF; font-size: 19px; font-weight: 700; "
                f"font-family: {FONT_STACK}; background: transparent;"
            )

            lbl = QLabel(label_txt)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet(
                f"color: {C_TEXT_DIM}; font-size: 9px; letter-spacing: 1.2px; "
                f"font-family: {FONT_STACK}; background: transparent;"
            )

            v.addWidget(val)
            v.addWidget(lbl)
            return cell, val

        fps_w,  self.fps_label        = stat("BILDRATE",  "—")
        face_w, self.face_count_label = stat("GESICHTER", "0")
        mode_w, self.mode_label       = stat("EFFEKT",    "—")

        hbox.addWidget(fps_w)
        hbox.addWidget(v_divider())
        hbox.addWidget(face_w)
        hbox.addWidget(v_divider())
        hbox.addWidget(mode_w)
        return bar

    # ── Linkes Panel (Effekte + Stärke) ──────────────────────────────────────

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(205)
        vbox = QVBoxLayout(panel)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(10)

        # ── Effekte ──
        ef_card = self._make_card()
        ef_vbox = QVBoxLayout(ef_card)
        ef_vbox.setContentsMargins(12, 12, 12, 12)
        ef_vbox.setSpacing(5)
        ef_vbox.addWidget(section_label("Effekt"))

        self.effect_btn_group = QButtonGroup(self)
        self.effect_btn_group.setExclusive(True)

        for effect_id, info in EFFECTS.items():
            btn = QPushButton(info["name"])
            btn.setCheckable(True)
            btn.setProperty("active", "false")

            if effect_id == DEFAULT_EFFECT:
                btn.setProperty("active", "true")
                btn.setStyleSheet(
                    f"QPushButton {{ background-color: {C_ACCENT}; color: #FFF; "
                    f"border: none; border-radius: 8px; padding: 9px 12px; "
                    f"font-family: {FONT_STACK}; font-size: 12px; font-weight: 600; text-align: left; }}"
                )
            else:
                btn.setStyleSheet(self._default_btn_ss())

            btn.clicked.connect(
                lambda _checked, eid=effect_id: self._on_effect_clicked(eid)
            )
            self.effect_btn_group.addButton(btn)
            self._effect_buttons[effect_id] = btn
            ef_vbox.addWidget(btn)

        vbox.addWidget(ef_card)

        # ── Stärke ──
        st_card = self._make_card()
        st_vbox = QVBoxLayout(st_card)
        st_vbox.setContentsMargins(12, 12, 12, 14)
        st_vbox.setSpacing(8)
        st_vbox.addWidget(section_label("Stärke"))

        slider_row = QHBoxLayout()
        slider_row.setSpacing(8)

        self.strength_slider = QSlider(Qt.Horizontal)
        self.strength_slider.setRange(10, 100)
        self.strength_slider.setValue(50)
        self.strength_slider.valueChanged.connect(self._on_strength_changed)

        self.strength_value_label = QLabel("50")
        self.strength_value_label.setFixedWidth(26)
        self.strength_value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.strength_value_label.setStyleSheet(
            f"color: {C_TEXT}; font-size: 12px; font-weight: 600; "
            f"font-family: {FONT_STACK}; background: transparent;"
        )

        slider_row.addWidget(self.strength_slider)
        slider_row.addWidget(self.strength_value_label)
        st_vbox.addLayout(slider_row)
        vbox.addWidget(st_card)

        vbox.addStretch()
        return panel

    # ── Rechtes Panel (Presets / Emoji / Aktionen / Erkennung) ───────────────

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(205)
        vbox = QVBoxLayout(panel)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(10)

        # ── Schnellwahl ──
        pr_card = self._make_card()
        pr_vbox = QVBoxLayout(pr_card)
        pr_vbox.setContentsMargins(12, 12, 12, 12)
        pr_vbox.setSpacing(5)
        pr_vbox.addWidget(section_label("Schnellwahl"))

        self.preset_btn_group = QButtonGroup(self)
        self.preset_btn_group.setExclusive(True)

        for pid, pinfo in PRESETS.items():
            btn = QPushButton(pinfo["name"])
            btn.setCheckable(True)
            btn.setStyleSheet(self._preset_btn_ss(active=False))
            btn.clicked.connect(lambda _c, p=pid: self._apply_preset(p))
            self.preset_btn_group.addButton(btn)
            pr_vbox.addWidget(btn)

        vbox.addWidget(pr_card)

        # ── Überlagerung ──
        em_card = self._make_card()
        em_vbox = QVBoxLayout(em_card)
        em_vbox.setContentsMargins(12, 12, 12, 12)
        em_vbox.setSpacing(6)
        em_vbox.addWidget(section_label("Überlagerung"))

        self.emoji_btn_group = QButtonGroup(self)
        self.emoji_btn_group.setExclusive(True)

        emoji_items = list(EMOJIS.items())
        row1 = QHBoxLayout()
        row1.setSpacing(4)
        row2 = QHBoxLayout()
        row2.setSpacing(4)

        for i, (eid, einfo) in enumerate(emoji_items):
            btn = QPushButton(einfo["name"])
            btn.setCheckable(True)
            btn.setFixedHeight(30)
            btn.setStyleSheet(self._emoji_btn_ss(active=False))
            btn.clicked.connect(lambda _c, e=eid: self._on_emoji_clicked(e))
            self.emoji_btn_group.addButton(btn)
            (row1 if i < 4 else row2).addWidget(btn)

        em_vbox.addLayout(row1)
        em_vbox.addLayout(row2)
        vbox.addWidget(em_card)

        # ── Aktionen ──
        ac_card = self._make_card()
        ac_vbox = QVBoxLayout(ac_card)
        ac_vbox.setContentsMargins(12, 12, 12, 12)
        ac_vbox.setSpacing(6)
        ac_vbox.addWidget(section_label("Aktionen"))

        screenshot_btn = QPushButton("Screenshot")
        screenshot_btn.setStyleSheet(self._default_btn_ss())
        screenshot_btn.clicked.connect(self._take_screenshot)
        ac_vbox.addWidget(screenshot_btn)

        self.record_btn = QPushButton("Aufnahme starten")
        self.record_btn.setStyleSheet(self._success_btn_ss())
        self.record_btn.clicked.connect(self._toggle_recording)
        ac_vbox.addWidget(self.record_btn)

        vbox.addWidget(ac_card)

        # ── Gesichtserkennung ──
        det_card = self._make_card()
        det_vbox = QVBoxLayout(det_card)
        det_vbox.setContentsMargins(12, 12, 12, 12)
        det_vbox.setSpacing(6)
        det_vbox.addWidget(section_label("Erkennung"))

        self.detection_btn = QPushButton("Aktiv")
        self.detection_btn.setCheckable(True)
        self.detection_btn.setChecked(True)
        self.detection_btn.setStyleSheet(self._accent_btn_ss())
        self.detection_btn.toggled.connect(self._on_detection_toggled)
        det_vbox.addWidget(self.detection_btn)

        vbox.addWidget(det_card)
        vbox.addStretch()
        return panel

    # ── Statuszeile ───────────────────────────────────────────────────────────

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
            f"color: {C_TEXT_DIM}; font-size: 11px; font-family: {FONT_STACK}; background: transparent;"
        )

        hint = QLabel("ESC  ·  S = Screenshot  ·  R = Aufnahme  ·  Leertaste = Erkennung")
        hint.setStyleSheet(
            f"color: {C_TEXT_DIM}; font-size: 10px; font-family: {FONT_STACK}; background: transparent;"
        )

        hbox.addWidget(self.status_label)
        hbox.addStretch()
        hbox.addWidget(hint)
        return bar

    # ── Style-Helfer (inline SS für spezifische Zustände) ─────────────────────

    def _make_card(self) -> QFrame:
        card = QFrame()
        card.setStyleSheet(
            f"QFrame {{ background-color: {C_PANEL}; border-radius: 10px; border: 1px solid {C_BORDER}; }}"
        )
        return card

    def _default_btn_ss(self) -> str:
        return (
            f"QPushButton {{ background-color: {C_BUTTON}; color: {C_TEXT_DARK}; border: none; "
            f"border-radius: 8px; padding: 9px 12px; font-family: {FONT_STACK}; "
            f"font-size: 12px; font-weight: 500; text-align: left; }} "
            f"QPushButton:hover {{ background-color: {C_BUTTON_HOV}; }} "
            f"QPushButton:pressed {{ background-color: {C_BUTTON_ACT}; }}"
        )

    def _active_effect_btn_ss(self) -> str:
        return (
            f"QPushButton {{ background-color: {C_ACCENT}; color: #FFFFFF; border: none; "
            f"border-radius: 8px; padding: 9px 12px; font-family: {FONT_STACK}; "
            f"font-size: 12px; font-weight: 600; text-align: left; }}"
        )

    def _accent_btn_ss(self) -> str:
        return (
            f"QPushButton {{ background-color: {C_ACCENT}; color: #FFFFFF; border: none; "
            f"border-radius: 8px; padding: 9px 12px; font-family: {FONT_STACK}; "
            f"font-size: 12px; font-weight: 600; text-align: left; }} "
            f"QPushButton:hover {{ background-color: #5BAECE; }}"
        )

    def _success_btn_ss(self) -> str:
        return (
            f"QPushButton {{ background-color: {C_SUCCESS}; color: #FFFFFF; border: none; "
            f"border-radius: 8px; padding: 9px 12px; font-family: {FONT_STACK}; "
            f"font-size: 12px; font-weight: 600; text-align: left; }} "
            f"QPushButton:hover {{ background-color: #5AAD80; }}"
        )

    def _danger_btn_ss(self) -> str:
        return (
            f"QPushButton {{ background-color: {C_DANGER}; color: #FFFFFF; border: none; "
            f"border-radius: 8px; padding: 9px 12px; font-family: {FONT_STACK}; "
            f"font-size: 12px; font-weight: 600; text-align: left; }} "
            f"QPushButton:hover {{ background-color: #CC5555; }}"
        )

    def _preset_btn_ss(self, active: bool) -> str:
        if active:
            return (
                f"QPushButton {{ background-color: {C_SUCCESS}; color: #FFFFFF; border: none; "
                f"border-radius: 8px; padding: 9px 12px; font-family: {FONT_STACK}; "
                f"font-size: 12px; font-weight: 600; text-align: left; }}"
            )
        return self._default_btn_ss()

    def _emoji_btn_ss(self, active: bool) -> str:
        if active:
            return (
                f"QPushButton {{ background-color: {C_ACCENT}; color: #FFFFFF; border: none; "
                f"border-radius: 6px; padding: 4px 6px; font-family: {FONT_STACK}; "
                f"font-size: 10px; font-weight: 600; text-align: center; }}"
            )
        return (
            f"QPushButton {{ background-color: {C_BUTTON}; color: {C_TEXT_DARK}; border: none; "
            f"border-radius: 6px; padding: 4px 6px; font-family: {FONT_STACK}; "
            f"font-size: 10px; font-weight: 400; text-align: center; }} "
            f"QPushButton:hover {{ background-color: {C_BUTTON_HOV}; }} "
            f"QPushButton:pressed {{ background-color: {C_BUTTON_ACT}; }}"
        )

    # ── Event-Handler ─────────────────────────────────────────────────────────

    def _on_frame_ready(self, frame: np.ndarray, face_count: int, fps: float):
        """Frame anzeigen + Stats aktualisieren."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        q_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img).scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.FastTransformation,
        )
        self.video_label.setPixmap(pixmap)

        # FPS nur anzeigen wenn sinnvoll (> 0)
        self.fps_label.setText(f"{fps:.1f}" if fps > 0.5 else "—")
        self.face_count_label.setText(str(face_count))
        self.mode_label.setText(self.effects.get_effect_name())

    def _on_effect_clicked(self, effect_id: str):
        self.effects.set_effect(effect_id)
        for eid, btn in self._effect_buttons.items():
            btn.setStyleSheet(
                self._active_effect_btn_ss() if eid == effect_id
                else self._default_btn_ss()
            )

    def _on_strength_changed(self, value: int):
        self.strength_value_label.setText(str(value))
        self.effects.set_strength(value)

    def _apply_preset(self, preset_id: str):
        self.effects.apply_preset(preset_id)
        preset = PRESETS.get(preset_id, {})
        if "strength" in preset:
            self.strength_slider.setValue(preset["strength"])
        # Effekt-Buttons synchronisieren
        active_eff = self.effects.current_effect
        for eid, btn in self._effect_buttons.items():
            btn.setStyleSheet(
                self._active_effect_btn_ss() if eid == active_eff
                else self._default_btn_ss()
            )
        # Preset-Buttons: aktiven hervorheben
        for btn in self.preset_btn_group.buttons():
            is_active = (btn.text() == preset.get("name", ""))
            btn.setStyleSheet(self._preset_btn_ss(active=is_active))
        self.status_label.setText(f"Schnellwahl: {preset.get('name', preset_id)}")

    def _on_emoji_clicked(self, emoji_id: str):
        self.effects.set_emoji(emoji_id)
        # Emoji-Buttons: aktiven hervorheben, alle anderen zurücksetzen
        from effects import EMOJIS as _EMOJIS
        for btn in self.emoji_btn_group.buttons():
            name_match = btn.text() == _EMOJIS.get(emoji_id, {}).get("name", "")
            btn.setStyleSheet(self._emoji_btn_ss(active=name_match))
        # Effekt-Button für Emoji-Overlay aktivieren
        for eid, btn in self._effect_buttons.items():
            btn.setStyleSheet(
                self._active_effect_btn_ss() if eid == "emoji"
                else self._default_btn_ss()
            )

    def _on_detection_toggled(self, enabled: bool):
        self.proc_thread.set_detection_enabled(enabled)
        self.detection_btn.setText("Aktiv" if enabled else "Deaktiviert")
        self.detection_btn.setStyleSheet(
            self._accent_btn_ss() if enabled else self._default_btn_ss()
        )
        self.status_label.setText(
            f"Gesichtserkennung {'aktiviert' if enabled else 'deaktiviert'}"
        )

    def _take_screenshot(self):
        frame = self.camera.get_frame()
        if frame is not None:
            faces = self.detector.detect(frame)
            processed = self.effects.apply(frame.copy(), faces)
            filename = self.recorder.save_screenshot(processed)
            self.status_label.setText(
                f"Screenshot gespeichert: {os.path.basename(filename)}"
            )
        else:
            self.status_label.setText("Kein Bild verfügbar")

    def _toggle_recording(self):
        if not self.recorder.is_recording:
            frame = self.camera.get_frame()
            if frame is not None:
                self.recorder.start_recording(frame.shape)
                self.record_btn.setText("Aufnahme stoppen")
                self.record_btn.setStyleSheet(self._danger_btn_ss())
                self.rec_indicator.setVisible(True)
                self._rec_timer.start(600)
                self.status_label.setText(
                    f"Aufnahme läuft: {os.path.basename(self.recorder.current_file)}"
                )
        else:
            saved = self.recorder.stop_recording()
            self.record_btn.setText("Aufnahme starten")
            self.record_btn.setStyleSheet(self._success_btn_ss())
            self._rec_timer.stop()
            self.rec_indicator.setVisible(False)
            self.status_label.setText(
                f"Gespeichert: {os.path.basename(saved)}"
            )

    def _blink_rec_indicator(self):
        self._rec_blink = not self._rec_blink
        self.rec_indicator.setVisible(self._rec_blink)

    def _check_camera_status(self):
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

    # ── Sauberer Shutdown ─────────────────────────────────────────────────────

    def closeEvent(self, event):
        if self.recorder.is_recording:
            self.recorder.stop_recording()
        self.proc_thread.stop()
        self.camera.stop()
        self.camera.join(timeout=2.0)
        event.accept()
