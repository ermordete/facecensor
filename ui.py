#!/usr/bin/python3
"""
ui.py - Hauptfenster und GUI
==============================
PyQt5-basierte Oberfläche für FaceCensor Pro.

Design-Konzept:
- Navy Blue Background (#1F2A36)
- Chalk Beige Buttons (#E1DACA)
- Sage Labels (#CBCCBE)
- Modernes Creator-Tool-Look
- 3-Spalten-Layout: Controls | Kamerafeed | Controls
"""

import cv2
import numpy as np
import os
from datetime import datetime

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QSlider, QFrame, QScrollArea,
    QSizePolicy, QSpacerItem, QButtonGroup, QStackedWidget,
    QProgressBar,
)
from PyQt5.QtCore import (
    Qt, QTimer, QThread, pyqtSignal, QPropertyAnimation,
    QEasingCurve, QSize, QRect,
)
from PyQt5.QtGui import (
    QImage, QPixmap, QFont, QColor, QPalette, QIcon,
    QPainter, QBrush, QPen, QLinearGradient, QFontDatabase,
)

from camera import CameraThread
from detector import FaceDetector
from effects import EffectsProcessor, EFFECTS, EMOJIS, PRESETS
from recorder import Recorder


# ─── Farb-Konstanten ─────────────────────────────────────────────────────────
C_BG          = "#1F2A36"   # Navy Blue - Hintergrund
C_PANEL       = "#263545"   # Etwas heller - Panel-Hintergrund
C_BUTTON      = "#E1DACA"   # Chalk Beige - Buttons
C_BUTTON_HOV  = "#F0EDE6"   # Heller Beige - Hover
C_BUTTON_ACT  = "#C8C4B4"   # Dunkler Beige - Aktiv/Pressed
C_TEXT        = "#CBCCBE"   # Sage - Haupttext
C_TEXT_DIM    = "#8A9099"   # Gedimmter Text
C_ACCENT      = "#4A9EBF"   # Akzentblau für Recording
C_DANGER      = "#BF4A4A"   # Rot für Stopp/Warnung
C_SUCCESS     = "#4ABF7E"   # Grün für Erfolg
C_BORDER      = "#2E3F50"   # Dezente Border


# ─── Stylesheet ──────────────────────────────────────────────────────────────
GLOBAL_STYLE = f"""
QMainWindow, QWidget {{
    background-color: {C_BG};
    color: {C_TEXT};
    font-family: "Palatino Linotype", "Palatino", "Book Antiqua", Georgia, serif;
    font-size: 13px;
}}

/* ─── Buttons ─── */
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

QPushButton:hover {{
    background-color: {C_BUTTON_HOV};
    /* Subtle lift effect via shadow - approximated */
}}

QPushButton:pressed {{
    background-color: {C_BUTTON_ACT};
    padding-top: 10px;
    padding-bottom: 8px;
}}

QPushButton:checked {{
    background-color: #3A7FA0;
    color: #FFFFFF;
}}

QPushButton:disabled {{
    background-color: #3A4858;
    color: {C_TEXT_DIM};
}}

/* ─── Effect Buttons (Sidebar) ─── */
QPushButton.effect-btn {{
    background-color: {C_PANEL};
    color: {C_TEXT};
    border: 1px solid {C_BORDER};
    border-radius: 8px;
    padding: 8px 12px;
    text-align: left;
}}

QPushButton.effect-btn:hover {{
    background-color: #2E4055;
    border-color: #4A9EBF44;
}}

QPushButton.effect-btn:checked {{
    background-color: #2D5B78;
    border-color: #4A9EBF;
    color: #FFFFFF;
}}

/* ─── Preset Buttons ─── */
QPushButton.preset-btn {{
    background-color: {C_PANEL};
    color: {C_TEXT};
    border: 1px solid {C_BORDER};
    border-radius: 8px;
    font-size: 11px;
    padding: 7px 10px;
    text-align: center;
}}

QPushButton.preset-btn:hover {{
    background-color: #2E4055;
}}

QPushButton.preset-btn:checked {{
    background-color: #3A5E2E;
    border-color: #4ABF7E;
    color: #FFFFFF;
}}

/* ─── Action Buttons ─── */
QPushButton.action-btn {{
    background-color: {C_ACCENT};
    color: #FFFFFF;
    font-size: 12px;
    font-weight: 700;
    text-align: center;
}}

QPushButton.action-btn:hover {{
    background-color: #5AB0CF;
}}

QPushButton.action-btn:pressed {{
    background-color: #3A8EAF;
}}

QPushButton.action-btn.recording {{
    background-color: {C_DANGER};
}}

QPushButton.action-btn.recording:hover {{
    background-color: #CF5A5A;
}}

/* ─── Labels ─── */
QLabel {{
    color: {C_TEXT};
    background: transparent;
}}

QLabel.section-title {{
    color: {C_TEXT_DIM};
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 4px 0px;
}}

QLabel.stat-value {{
    color: #FFFFFF;
    font-size: 22px;
    font-weight: bold;
}}

QLabel.stat-label {{
    color: {C_TEXT_DIM};
    font-size: 10px;
    letter-spacing: 1px;
}}

/* ─── Slider ─── */
QSlider::groove:horizontal {{
    height: 4px;
    background: {C_BORDER};
    border-radius: 2px;
}}

QSlider::handle:horizontal {{
    background: {C_BUTTON};
    border: none;
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}}

QSlider::sub-page:horizontal {{
    background: {C_ACCENT};
    border-radius: 2px;
}}

/* ─── Frame / Panel ─── */
QFrame.panel {{
    background-color: {C_PANEL};
    border-radius: 12px;
    border: 1px solid {C_BORDER};
}}

/* ─── Video-Bereich ─── */
QLabel.video-label {{
    background-color: #0D1720;
    border-radius: 12px;
    border: 1px solid {C_BORDER};
}}

/* ─── Status Bar ─── */
QLabel.status-bar {{
    background-color: {C_PANEL};
    border-top: 1px solid {C_BORDER};
    color: {C_TEXT_DIM};
    font-size: 11px;
    padding: 6px 12px;
    border-radius: 0px;
}}

/* ─── Scrollbar ─── */
QScrollBar:vertical {{
    background: {C_BG};
    width: 6px;
    border-radius: 3px;
}}
QScrollBar::handle:vertical {{
    background: {C_BORDER};
    border-radius: 3px;
    min-height: 30px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}
"""


# ─── Utility Widgets ─────────────────────────────────────────────────────────

def make_section_label(text: str) -> QLabel:
    lbl = QLabel(text.upper())
    lbl.setProperty("class", "section-title")
    lbl.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 10px; letter-spacing: 2px;")
    return lbl


def make_divider() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setStyleSheet(f"color: {C_BORDER}; background: {C_BORDER}; max-height: 1px;")
    return line


class StyledButton(QPushButton):
    """Erweiterter Button mit animiertem Hover-Glow."""
    
    def __init__(self, text: str, icon: str = "", css_class: str = ""):
        display = f"{icon}  {text}" if icon else text
        super().__init__(display)
        if css_class:
            self.setProperty("class", css_class)


# ─── Verarbeitungs-Thread ────────────────────────────────────────────────────

class ProcessingThread(QThread):
    """
    Thread für Frame-Verarbeitung (Detection + Effekte).
    Entkoppelt schwere Rechenoperationen vom UI-Thread.
    Sendet fertigen Frame per Signal.
    """
    frame_ready = pyqtSignal(np.ndarray, int, float)  # frame, face_count, fps
    
    def __init__(self, camera: CameraThread, detector: FaceDetector, effects: EffectsProcessor, recorder: Recorder):
        super().__init__()
        self.camera = camera
        self.detector = detector
        self.effects = effects
        self.recorder = recorder
        self._running = True
        self._detection_enabled = True
    
    def run(self):
        import time
        while self._running:
            frame = self.camera.get_frame()
            
            if frame is None:
                time.sleep(0.005)
                continue
            
            if self._detection_enabled:
                faces = self.detector.detect(frame)
                frame = self.effects.apply(frame, faces)
                face_count = len(faces)
            else:
                face_count = 0
            
            # Aufnahme-Frame schreiben
            if self.recorder.is_recording:
                self.recorder.write_frame(frame)
            
            fps = self.camera.get_fps()
            self.frame_ready.emit(frame, face_count, fps)
    
    def set_detection_enabled(self, enabled: bool):
        self._detection_enabled = enabled
    
    def stop(self):
        self._running = False
        self.wait()


# ─── Hauptfenster ────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    """
    Hauptfenster von FaceCensor Pro.
    Layout: [Left Panel] | [Video Feed] | [Right Panel]
    """
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FaceCensor Pro  ·  Jetson Edition")
        self.setMinimumSize(1080, 680)
        self.resize(1200, 750)
        self.setStyleSheet(GLOBAL_STYLE)
        
        # Kernkomponenten
        self.camera = CameraThread(use_csi=True)
        self.detector = FaceDetector()
        self.effects = EffectsProcessor()
        self.recorder = Recorder()
        
        # UI aufbauen
        self._build_ui()
        
        # Verarbeitungsthread starten
        self.proc_thread = ProcessingThread(
            self.camera, self.detector, self.effects, self.recorder
        )
        self.proc_thread.frame_ready.connect(self._on_frame_ready)
        
        # Kamera starten
        self.camera.start()
        self.proc_thread.start()
        
        # Status-Prüftimer
        self._status_timer = QTimer()
        self._status_timer.timeout.connect(self._check_camera_status)
        self._status_timer.start(1000)
        
        # Aufnahme-Blink-Timer
        self._rec_blink = False
        self._rec_timer = QTimer()
        self._rec_timer.timeout.connect(self._blink_rec_indicator)
        
        self._last_face_count = 0
    
    # ─── UI Construction ───────────────────────────────────────────────────
    
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Header
        header = self._build_header()
        main_layout.addWidget(header)
        
        # Mittelbereich: 3 Spalten
        content = QWidget()
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(12, 12, 12, 8)
        content_layout.setSpacing(12)
        
        left_panel = self._build_left_panel()
        video_area = self._build_video_area()
        right_panel = self._build_right_panel()
        
        content_layout.addWidget(left_panel, 0)
        content_layout.addWidget(video_area, 1)
        content_layout.addWidget(right_panel, 0)
        
        main_layout.addWidget(content, 1)
        
        # Status-Bar
        status_bar = self._build_status_bar()
        main_layout.addWidget(status_bar)
    
    def _build_header(self) -> QWidget:
        header = QWidget()
        header.setFixedHeight(56)
        header.setStyleSheet(f"background: {C_PANEL}; border-bottom: 1px solid {C_BORDER};")
        
        layout = QHBoxLayout(header)
        layout.setContentsMargins(20, 0, 20, 0)
        
        # Logo / Titel
        title = QLabel("✦  FaceCensor Pro")
        title.setStyleSheet(f"""
            color: {C_BUTTON};
            font-size: 18px;
            font-weight: bold;
            font-family: "Palatino Linotype", Georgia, serif;
            letter-spacing: 1px;
        """)
        
        subtitle = QLabel("Content Creator Edition · Jetson Nano")
        subtitle.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 11px; padding-left: 8px;")
        
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addStretch()
        
        # Recording-Indikator
        self.rec_indicator = QLabel("⬤  REC")
        self.rec_indicator.setStyleSheet(f"color: {C_DANGER}; font-size: 12px; font-weight: bold;")
        self.rec_indicator.setVisible(False)
        layout.addWidget(self.rec_indicator)
        
        return header
    
    def _build_video_area(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        
        # Kamera-Feed-Label
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(480, 360)
        self.video_label.setText("● Kamera wird gestartet...")
        self.video_label.setStyleSheet(f"""
            background-color: #0D1720;
            border-radius: 12px;
            border: 1px solid {C_BORDER};
            color: {C_TEXT_DIM};
            font-size: 14px;
        """)
        self.video_label.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        layout.addWidget(self.video_label, 1)
        
        # Stats-Zeile unter dem Video
        stats_row = self._build_stats_row()
        layout.addWidget(stats_row)
        
        return container
    
    def _build_stats_row(self) -> QWidget:
        row = QWidget()
        row.setFixedHeight(64)
        row.setStyleSheet(f"background: {C_PANEL}; border-radius: 10px; border: 1px solid {C_BORDER};")
        
        layout = QHBoxLayout(row)
        layout.setContentsMargins(16, 8, 16, 8)
        layout.setSpacing(0)
        
        def stat_widget(label: str, default: str) -> tuple:
            w = QWidget()
            v = QVBoxLayout(w)
            v.setContentsMargins(0, 0, 0, 0)
            v.setSpacing(1)
            val = QLabel(default)
            val.setStyleSheet(f"color: #FFFFFF; font-size: 20px; font-weight: bold; background: transparent;")
            val.setAlignment(Qt.AlignCenter)
            lbl = QLabel(label)
            lbl.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 10px; letter-spacing: 1px; background: transparent;")
            lbl.setAlignment(Qt.AlignCenter)
            v.addWidget(val)
            v.addWidget(lbl)
            return w, val
        
        fps_w, self.fps_label = stat_widget("FPS", "—")
        face_w, self.face_count_label = stat_widget("GESICHTER", "0")
        mode_w, self.mode_label = stat_widget("MODUS", "—")
        
        # Trennlinien
        def vline():
            l = QFrame()
            l.setFrameShape(QFrame.VLine)
            l.setStyleSheet(f"color: {C_BORDER}; background: {C_BORDER};")
            l.setFixedWidth(1)
            return l
        
        layout.addWidget(fps_w)
        layout.addWidget(vline())
        layout.addWidget(face_w)
        layout.addWidget(vline())
        layout.addWidget(mode_w)
        
        return row
    
    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(220)
        
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        
        # ── Effekte ──
        effects_frame = QFrame()
        effects_frame.setStyleSheet(f"""
            QFrame {{
                background: {C_PANEL};
                border-radius: 12px;
                border: 1px solid {C_BORDER};
            }}
        """)
        ef_layout = QVBoxLayout(effects_frame)
        ef_layout.setContentsMargins(12, 12, 12, 12)
        ef_layout.setSpacing(6)
        
        ef_layout.addWidget(make_section_label("Effekte"))
        
        self.effect_btn_group = QButtonGroup()
        self.effect_btn_group.setExclusive(True)
        
        for effect_id, info in EFFECTS.items():
            btn = QPushButton(f"{info['icon']}  {info['name']}")
            btn.setCheckable(True)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {C_BG};
                    color: {C_TEXT};
                    border: 1px solid {C_BORDER};
                    border-radius: 8px;
                    padding: 8px 10px;
                    text-align: left;
                    font-family: "Palatino Linotype", Georgia, serif;
                    font-size: 12px;
                }}
                QPushButton:hover {{
                    background-color: #2E4055;
                    border-color: #4A9EBF55;
                }}
                QPushButton:checked {{
                    background-color: #2D5B78;
                    border-color: #4A9EBF;
                    color: #FFFFFF;
                }}
            """)
            btn.clicked.connect(lambda checked, eid=effect_id: self.effects.set_effect(eid))
            if effect_id == "blur_gaussian":
                btn.setChecked(True)
            self.effect_btn_group.addButton(btn)
            ef_layout.addWidget(btn)
        
        layout.addWidget(effects_frame)
        
        # ── Stärke-Slider ──
        strength_frame = QFrame()
        strength_frame.setStyleSheet(f"""
            QFrame {{
                background: {C_PANEL};
                border-radius: 12px;
                border: 1px solid {C_BORDER};
            }}
        """)
        sl_layout = QVBoxLayout(strength_frame)
        sl_layout.setContentsMargins(12, 12, 12, 12)
        sl_layout.setSpacing(6)
        
        sl_layout.addWidget(make_section_label("Stärke"))
        
        slider_row = QHBoxLayout()
        self.strength_slider = QSlider(Qt.Horizontal)
        self.strength_slider.setRange(10, 100)
        self.strength_slider.setValue(50)
        self.strength_slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                height: 4px;
                background: {C_BORDER};
                border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                background: {C_BUTTON};
                width: 14px; height: 14px;
                margin: -5px 0;
                border-radius: 7px;
                border: none;
            }}
            QSlider::sub-page:horizontal {{
                background: {C_ACCENT};
                border-radius: 2px;
            }}
        """)
        self.strength_slider.valueChanged.connect(self._on_strength_changed)
        
        self.strength_value_label = QLabel("50")
        self.strength_value_label.setFixedWidth(28)
        self.strength_value_label.setStyleSheet(f"color: {C_TEXT}; font-size: 12px; font-weight: bold;")
        
        slider_row.addWidget(self.strength_slider)
        slider_row.addWidget(self.strength_value_label)
        sl_layout.addLayout(slider_row)
        
        layout.addWidget(strength_frame)
        layout.addStretch()
        
        return panel
    
    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(220)
        
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        
        # ── Presets ──
        preset_frame = QFrame()
        preset_frame.setStyleSheet(f"""
            QFrame {{
                background: {C_PANEL};
                border-radius: 12px;
                border: 1px solid {C_BORDER};
            }}
        """)
        pr_layout = QVBoxLayout(preset_frame)
        pr_layout.setContentsMargins(12, 12, 12, 12)
        pr_layout.setSpacing(6)
        
        pr_layout.addWidget(make_section_label("Presets"))
        
        preset_grid = QHBoxLayout()
        preset_grid.setSpacing(6)
        
        self.preset_btn_group = QButtonGroup()
        self.preset_btn_group.setExclusive(True)
        
        for i, (pid, pinfo) in enumerate(PRESETS.items()):
            btn = QPushButton(f"{pinfo['icon']}\n{pinfo['name']}")
            btn.setCheckable(True)
            btn.setFixedHeight(52)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {C_BG};
                    color: {C_TEXT};
                    border: 1px solid {C_BORDER};
                    border-radius: 8px;
                    padding: 4px;
                    font-size: 11px;
                    text-align: center;
                    font-family: "Palatino Linotype", Georgia, serif;
                }}
                QPushButton:hover {{ background: #2E4055; }}
                QPushButton:checked {{
                    background: #3A5E2E;
                    border-color: #4ABF7E;
                    color: #FFFFFF;
                }}
            """)
            btn.clicked.connect(lambda checked, p=pid: self._apply_preset(p))
            self.preset_btn_group.addButton(btn)
            preset_grid.addWidget(btn)
        
        pr_layout.addLayout(preset_grid)
        layout.addWidget(preset_frame)
        
        # ── Emoji-Auswahl ──
        emoji_frame = QFrame()
        emoji_frame.setStyleSheet(f"""
            QFrame {{
                background: {C_PANEL};
                border-radius: 12px;
                border: 1px solid {C_BORDER};
            }}
        """)
        em_layout = QVBoxLayout(emoji_frame)
        em_layout.setContentsMargins(12, 12, 12, 12)
        em_layout.setSpacing(6)
        
        em_layout.addWidget(make_section_label("Emoji Overlay"))
        
        self.emoji_btn_group = QButtonGroup()
        self.emoji_btn_group.setExclusive(True)
        
        emoji_grid = QHBoxLayout()
        emoji_grid.setSpacing(4)
        emoji_line2 = QHBoxLayout()
        emoji_line2.setSpacing(4)
        
        emoji_items = list(EMOJIS.items())
        
        for i, (eid, einfo) in enumerate(emoji_items):
            name_parts = einfo["name"].split(" ")
            icon_char = name_parts[0] if name_parts else "?"
            
            btn = QPushButton(icon_char)
            btn.setCheckable(True)
            btn.setFixedSize(40, 40)
            btn.setToolTip(einfo["name"])
            btn.setStyleSheet(f"""
                QPushButton {{
                    background: {C_BG};
                    border: 1px solid {C_BORDER};
                    border-radius: 8px;
                    font-size: 18px;
                    padding: 0;
                }}
                QPushButton:hover {{ background: #2E4055; }}
                QPushButton:checked {{
                    background: #3A5E78;
                    border-color: #4A9EBF;
                }}
            """)
            btn.clicked.connect(lambda checked, e=eid: self.effects.set_emoji(e))
            self.emoji_btn_group.addButton(btn)
            
            if i < 4:
                emoji_grid.addWidget(btn)
            else:
                emoji_line2.addWidget(btn)
        
        # Restliche Slots auffüllen
        for layout_row in (emoji_grid, emoji_line2):
            while layout_row.count() < 4:
                layout_row.addStretch()
        
        em_layout.addLayout(emoji_grid)
        em_layout.addLayout(emoji_line2)
        layout.addWidget(emoji_frame)
        
        # ── Aktionen ──
        action_frame = QFrame()
        action_frame.setStyleSheet(f"""
            QFrame {{
                background: {C_PANEL};
                border-radius: 12px;
                border: 1px solid {C_BORDER};
            }}
        """)
        ac_layout = QVBoxLayout(action_frame)
        ac_layout.setContentsMargins(12, 12, 12, 12)
        ac_layout.setSpacing(8)
        
        ac_layout.addWidget(make_section_label("Aktionen"))
        
        # Screenshot Button
        screenshot_btn = QPushButton("📸   Screenshot")
        screenshot_btn.setStyleSheet(f"""
            QPushButton {{
                background: {C_BUTTON};
                color: #1F2A36;
                border: none;
                border-radius: 8px;
                padding: 9px 12px;
                font-weight: 600;
                font-family: "Palatino Linotype", Georgia, serif;
                font-size: 12px;
                text-align: left;
            }}
            QPushButton:hover {{ background: {C_BUTTON_HOV}; }}
            QPushButton:pressed {{ background: {C_BUTTON_ACT}; }}
        """)
        screenshot_btn.clicked.connect(self._take_screenshot)
        ac_layout.addWidget(screenshot_btn)
        
        # Record Button
        self.record_btn = QPushButton("⏺   Aufnahme starten")
        self.record_btn.setStyleSheet(f"""
            QPushButton {{
                background: #4ABF7E;
                color: #FFFFFF;
                border: none;
                border-radius: 8px;
                padding: 9px 12px;
                font-weight: 700;
                font-family: "Palatino Linotype", Georgia, serif;
                font-size: 12px;
                text-align: left;
            }}
            QPushButton:hover {{ background: #5ACFA0; }}
            QPushButton:pressed {{ background: #3AAF6E; }}
        """)
        self.record_btn.clicked.connect(self._toggle_recording)
        ac_layout.addWidget(self.record_btn)
        
        layout.addWidget(action_frame)
        
        # ── Detection Toggle ──
        detect_frame = QFrame()
        detect_frame.setStyleSheet(f"""
            QFrame {{
                background: {C_PANEL};
                border-radius: 12px;
                border: 1px solid {C_BORDER};
            }}
        """)
        dt_layout = QVBoxLayout(detect_frame)
        dt_layout.setContentsMargins(12, 12, 12, 12)
        dt_layout.setSpacing(6)
        
        dt_layout.addWidget(make_section_label("Erkennung"))
        
        self.detection_btn = QPushButton("⚡   Aktiv")
        self.detection_btn.setCheckable(True)
        self.detection_btn.setChecked(True)
        self.detection_btn.setStyleSheet(f"""
            QPushButton {{
                background: #2D5B78;
                color: #FFFFFF;
                border: 1px solid #4A9EBF;
                border-radius: 8px;
                padding: 8px 12px;
                font-family: "Palatino Linotype", Georgia, serif;
                font-size: 12px;
                text-align: left;
            }}
            QPushButton:checked {{
                background: #2D5B78;
                border-color: #4A9EBF;
            }}
            QPushButton:!checked {{
                background: {C_BG};
                border-color: {C_BORDER};
                color: {C_TEXT_DIM};
            }}
            QPushButton:hover {{ background: #3A6B88; }}
        """)
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
        
        self.status_label = QLabel("● Bereit · Kamera wird initialisiert...")
        self.status_label.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 11px;")
        
        hint = QLabel("ESC = Beenden  ·  S = Screenshot  ·  R = Aufnahme")
        hint.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 11px; opacity: 0.6;")
        
        layout.addWidget(self.status_label)
        layout.addStretch()
        layout.addWidget(hint)
        
        return bar
    
    # ─── Event Handler ────────────────────────────────────────────────────────
    
    def _on_frame_ready(self, frame: np.ndarray, face_count: int, fps: float):
        """Empfängt verarbeiteten Frame und zeigt ihn im Video-Label an."""
        # Frame konvertieren (BGR → RGB → QImage)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Auf Label-Größe skalieren (erhalte Seitenverhältnis)
        pixmap = QPixmap.fromImage(q_img)
        scaled = pixmap.scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.video_label.setPixmap(scaled)
        
        # Stats aktualisieren
        self.fps_label.setText(f"{fps:.0f}")
        self.face_count_label.setText(str(face_count))
        self.mode_label.setText(self.effects.get_effect_name())
    
    def _on_strength_changed(self, value: int):
        self.strength_value_label.setText(str(value))
        self.effects.set_strength(value)
    
    def _apply_preset(self, preset_id: str):
        self.effects.apply_preset(preset_id)
        # Stärke-Slider synchronisieren
        from effects import PRESETS
        preset = PRESETS.get(preset_id, {})
        if "strength" in preset:
            self.strength_slider.setValue(preset["strength"])
        self.status_label.setText(f"● Preset aktiviert: {preset.get('name', preset_id)}")
    
    def _on_detection_toggled(self, enabled: bool):
        self.proc_thread.set_detection_enabled(enabled)
        self.detection_btn.setText("⚡   Aktiv" if enabled else "○   Deaktiviert")
        self.status_label.setText(f"● Gesichtserkennung {'aktiviert' if enabled else 'deaktiviert'}")
    
    def _take_screenshot(self):
        frame = self.camera.get_frame()
        if frame is not None:
            # Effekte anwenden
            faces = self.detector.detect(frame)
            processed = self.effects.apply(frame.copy(), faces)
            filename = self.recorder.save_screenshot(processed)
            self.status_label.setText(f"📸 Screenshot gespeichert: {os.path.basename(filename)}")
        else:
            self.status_label.setText("⚠️  Kein Frame für Screenshot verfügbar")
    
    def _toggle_recording(self):
        if not self.recorder.is_recording:
            # Starten
            frame = self.camera.get_frame()
            if frame is not None:
                self.recorder.start_recording(frame.shape)
                self.record_btn.setText("⏹   Aufnahme stoppen")
                self.record_btn.setStyleSheet(f"""
                    QPushButton {{
                        background: {C_DANGER};
                        color: #FFFFFF;
                        border: none;
                        border-radius: 8px;
                        padding: 9px 12px;
                        font-weight: 700;
                        font-family: "Palatino Linotype", Georgia, serif;
                        font-size: 12px;
                        text-align: left;
                    }}
                    QPushButton:hover {{ background: #CF5A5A; }}
                """)
                self.rec_indicator.setVisible(True)
                self._rec_timer.start(600)
                self.status_label.setText(f"🔴 Aufnahme läuft: {self.recorder.current_file}")
        else:
            # Stoppen
            saved_file = self.recorder.stop_recording()
            self.record_btn.setText("⏺   Aufnahme starten")
            self.record_btn.setStyleSheet(f"""
                QPushButton {{
                    background: #4ABF7E;
                    color: #FFFFFF;
                    border: none;
                    border-radius: 8px;
                    padding: 9px 12px;
                    font-weight: 700;
                    font-family: "Palatino Linotype", Georgia, serif;
                    font-size: 12px;
                    text-align: left;
                }}
                QPushButton:hover {{ background: #5ACFA0; }}
            """)
            self._rec_timer.stop()
            self.rec_indicator.setVisible(False)
            self.status_label.setText(f"✅ Aufnahme gespeichert: {os.path.basename(saved_file)}")
    
    def _blink_rec_indicator(self):
        """Lässt den REC-Indikator blinken."""
        self._rec_blink = not self._rec_blink
        self.rec_indicator.setVisible(self._rec_blink)
    
    def _check_camera_status(self):
        """Prüft ob Kamera läuft und zeigt Fehler an."""
        if self.camera.get_error():
            self.status_label.setText(f"❌ Kamera-Fehler: {self.camera.get_error()}")
        elif self.camera.is_running():
            # Nur updaten wenn kein anderer Status angezeigt wird
            if "Bereit" in self.status_label.text() or "initialisiert" in self.status_label.text():
                self.status_label.setText(f"● Kamera aktiv · {self.camera.get_fps():.0f} FPS")
    
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
            # Toggle Detection
            self.detection_btn.setChecked(not self.detection_btn.isChecked())
        
        super().keyPressEvent(event)
    
    # ─── Shutdown ─────────────────────────────────────────────────────────────
    
    def closeEvent(self, event):
        """Sauberer Shutdown: Threads stoppen, Kamera freigeben."""
        print("\n👋 FaceCensor Pro wird beendet...")
        
        # Aufnahme sicherstellen
        if self.recorder.is_recording:
            self.recorder.stop_recording()
        
        # Threads stoppen
        self.proc_thread.stop()
        self.camera.stop()
        self.camera.join(timeout=2.0)
        
        print("✅ Shutdown abgeschlossen")
        event.accept()
