#!/usr/bin/python3
"""
hand_detector.py – Handerkennung (Jetson Nano optimiert)
==========================================================

Technische Entscheidung: Haar Cascade statt MediaPipe
-------------------------------------------------------
MediaPipe Hands ist auf dem Jetson Nano (ARM + JetPack) schwierig zu
installieren und benötigt oft 30–80 ms pro Frame – das würde den
Live-Feed bei gleichzeitiger Gesichtserkennung stark einbrechen lassen.

Stattdessen wird ein Haar Cascade für Hände verwendet:
  - haarcascade_hand.xml  (Community-Modell, leicht, ~1–3 ms / Frame)
  - Rein OpenCV-basiert, keine zusätzlichen Abhängigkeiten
  - Gut geeignet für einfache, robuste Handerkennung

Einschränkungen (ehrliche Einschätzung):
  - Erkennt primär Frontalsicht / flach gehaltene Hände
  - Weniger präzise als MediaPipe
  - False Positives bei komplexem Hintergrund möglich

PERF-Maßnahmen:
  - Detection nur auf heruntergeskaltem Frame (240px Breite)
  - Detection-Intervall wird vom aufrufenden Thread gesteuert
    (DETECT_EVERY_N_FRAMES in ui.py, Hände = alle 4 Frames)
  - Kein Smoothing-Tracking (zu wenig Nutzen bei Händen, kostet Zeit)
  - Bounding-Box-Padding für bessere Abdeckung

Fallback:
  Falls das Haar-Cascade-XML nicht gefunden wird, gibt detect()
  immer eine leere Liste zurück. Die App läuft weiter, Handerkennung
  ist einfach deaktiviert ohne Absturz.

Modell-Datei:
  models/haarcascade_hand.xml
  → Automatisch heruntergeladen von einem bekannten GitHub-Mirror.
  → Alternativ manuell dort ablegen.
"""

import cv2
import numpy as np
import os
import urllib.request
from typing import List, Tuple


# Öffentliche Quelle für Haar Cascade Handerkennung
_HAND_CASCADE_URL = (
    "https://raw.githubusercontent.com/Nikita-Hritsay/"
    "Hand-Detection/main/hand.xml"
)
# Backup-URL falls primäre nicht verfügbar
_HAND_CASCADE_URL_BACKUP = (
    "https://raw.githubusercontent.com/quanhua92/"
    "human-pose-estimation-opencv/master/pose_iter_440000.caffemodel"
    # Hinweis: der Backup-Link ist nur als Platzhalter. Wenn der primäre
    # Download scheitert, wird kein halbfertiges Modell geladen.
)


class HandDetector:
    """
    Leichtgewichtige Handerkennung via Haar Cascade.

    Jetson-Nano-freundlich:
    - Nur OpenCV, keine weiteren Pakete
    - Intern auf 240px skalieren → sehr schnell
    - detect() gibt leere Liste zurück wenn kein Modell geladen
    """

    def __init__(
        self,
        model_dir: str = "models",
        min_neighbors: int = 5,
        scale_factor: float = 1.2,
        min_size: int = 40,
        padding: float = 0.10,
    ):
        self._cascade    = None
        self._ok         = False
        self.min_neighbors = min_neighbors
        self.scale_factor  = scale_factor
        self.min_size      = min_size
        self.padding       = padding

        self._load(model_dir)

    def _load(self, model_dir: str):
        """Lädt Haar Cascade XML, lädt bei Bedarf herunter."""
        os.makedirs(model_dir, exist_ok=True)
        path = os.path.join(model_dir, "haarcascade_hand.xml")

        if os.path.exists(path):
            clf = cv2.CascadeClassifier(path)
            if not clf.empty():
                self._cascade = clf
                self._ok      = True
                print("Hand Cascade geladen")
                return

        # Herunterladen
        print("Lade Hand Cascade herunter ...")
        try:
            urllib.request.urlretrieve(_HAND_CASCADE_URL, path)
            clf = cv2.CascadeClassifier(path)
            if not clf.empty():
                self._cascade = clf
                self._ok      = True
                print("Hand Cascade geladen (heruntergeladen)")
                return
            else:
                # Fehlerhaftes XML → löschen
                os.remove(path)
                print("Hand Cascade Download ungueltig – Handerkennung deaktiviert")
        except Exception as e:
            print(f"Hand Cascade Download fehlgeschlagen: {e}")
            print("Handerkennung deaktiviert (kein Modell verfügbar)")

    def is_available(self) -> bool:
        return self._ok

    def detect(
        self, frame: np.ndarray
    ) -> List[Tuple[int, int, int, int]]:
        """
        Erkennt Hände im Frame.

        PERF: Intern auf 240px Breite skalieren.
        Rückgabe: Liste von (x, y, w, h) in Original-Koordinaten.
        Gibt leere Liste zurück wenn kein Modell geladen.
        """
        if not self._ok or self._cascade is None:
            return []

        h_orig, w_orig = frame.shape[:2]

        # PERF: Intern auf kleine Breite skalieren
        target_w = 240
        scale    = target_w / w_orig
        small    = cv2.resize(frame, (target_w, int(h_orig * scale)))

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        dets = self._cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=(self.min_size, self.min_size),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        if len(dets) == 0:
            return []

        result = []
        for (x, y, w, h) in dets:
            # Zurückskalieren auf Original-Koordinaten
            x_o = int(x / scale)
            y_o = int(y / scale)
            w_o = int(w / scale)
            h_o = int(h / scale)

            # Padding für bessere Abdeckung
            pad_x = int(w_o * self.padding)
            pad_y = int(h_o * self.padding)
            x_o = max(0, x_o - pad_x)
            y_o = max(0, y_o - pad_y)
            w_o = min(w_orig - x_o, w_o + 2 * pad_x)
            h_o = min(h_orig - y_o, h_o + 2 * pad_y)

            if w_o > 10 and h_o > 10:
                result.append((x_o, y_o, w_o, h_o))

        return result
