#!/usr/bin/python3
"""
hand_detector.py – Handerkennung (Jetson Nano)
================================================

Strategie: zwei Ebenen, automatischer Fallback

EBENE 1 – MediaPipe HandLandmarker (primär, ~9 MB Einmal-Download)
--------------------------------------------------------------------
- Erkennt Hände zuverlässig anhand von 21 Landmarks pro Hand
- Leichtestes Modell (float16) läuft auf Jetson Nano in ~15–30 ms
- Wird einmalig nach models/hand_landmarker.task heruntergeladen
- Erkennt bis zu 4 Hände gleichzeitig
- Bounding Box wird aus Min/Max der Landmarks berechnet + Padding

EBENE 2 – Hautfarb-Detektion (Fallback, kein Download)
-------------------------------------------------------
- Funktioniert sofort, ohne jedes Modell
- HSV-Farbraum: erkennt Hautfarbregionen im Bild
- Morphologie-Filter entfernen Rauschen
- Konturen werden zu Bounding Boxes umgewandelt
- ~10 ms pro Frame auf Jetson Nano
- Weniger präzise als MediaPipe, aber besser als nichts
- Reagiert auf Beleuchtung / Hautton (kann fehlschlagen bei extremen Bedingungen)

PERF-Maßnahmen:
- MediaPipe: LIVE_STREAM RunningMode → non-blocking, Ergebnisse per Callback
- Hautfarbe: Intern auf 320px skalieren → schnell
- Detection-Intervall wird vom ProcessingThread gesteuert (alle 4 Frames)
"""

import cv2
import numpy as np
import os
import threading
import urllib.request
from typing import List, Tuple, Optional


# ─── Modell-Download ─────────────────────────────────────────────────────────

_MODEL_URL      = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
_MODEL_FILENAME = "hand_landmarker.task"
_MODEL_SIZE_MIN = 1_000_000   # Mindestgröße: 1 MB (Plausibilitätsprüfung)


def _download_model(model_dir: str) -> Optional[str]:
    """
    Lädt hand_landmarker.task herunter.
    Gibt Pfad zurück wenn erfolgreich, sonst None.
    """
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, _MODEL_FILENAME)

    if os.path.exists(path) and os.path.getsize(path) >= _MODEL_SIZE_MIN:
        return path

    print(f"Lade hand_landmarker.task herunter (~9 MB) ...")
    print(f"Quelle: {_MODEL_URL}")
    try:
        urllib.request.urlretrieve(_MODEL_URL, path)
        if os.path.getsize(path) >= _MODEL_SIZE_MIN:
            print(f"hand_landmarker.task geladen ({os.path.getsize(path)//1024} KB)")
            return path
        else:
            os.remove(path)
            print("Download unvollständig – Hautfarb-Fallback wird verwendet")
            return None
    except Exception as e:
        print(f"Download fehlgeschlagen: {e}")
        print("Hautfarb-Fallback wird verwendet")
        if os.path.exists(path):
            os.remove(path)
        return None


# ─── MediaPipe Landmarker ────────────────────────────────────────────────────

class _MediaPipeHands:
    """
    Wrapper für MediaPipe HandLandmarker (neue Tasks-API, MediaPipe 0.10+).
    Läuft im LIVE_STREAM Modus: nicht-blockierend, Ergebnis per Callback.
    """

    def __init__(self, model_path: str, max_hands: int = 4):
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.vision import RunningMode

        self._lock        = threading.Lock()
        self._last_result = None
        self._frame_ts    = 0

        def _callback(result, output_image, timestamp_ms):
            with self._lock:
                self._last_result = result

        options = vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.LIVE_STREAM,
            num_hands=max_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            result_callback=_callback,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._mp_image_cls = mp_tasks.vision.Image
        self._mp_format    = mp_tasks.ImageFormat.SRGB
        print("MediaPipe HandLandmarker bereit")

    def process(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Sendet Frame an MediaPipe (nicht-blockierend).
        Gibt letztes bekanntes Ergebnis als Bounding Boxes zurück.
        """
        from mediapipe.tasks.python.vision import RunningMode

        # Frame zu MediaPipe Image konvertieren (RGB)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = self._mp_image_cls(
            image_format=self._mp_format, data=rgb
        )

        # Timestamp muss strikt monoton steigen (ms)
        self._frame_ts += 33
        self._landmarker.detect_async(mp_img, self._frame_ts)

        # Letztes Callback-Ergebnis auslesen
        with self._lock:
            result = self._last_result

        if result is None or not result.hand_landmarks:
            return []

        h, w = frame.shape[:2]
        boxes = []
        for landmarks in result.hand_landmarks:
            xs = [lm.x * w for lm in landmarks]
            ys = [lm.y * h for lm in landmarks]
            x1, y1 = int(min(xs)), int(min(ys))
            x2, y2 = int(max(xs)), int(max(ys))

            # Padding: 20% für bessere Blur-Abdeckung
            pad_x = int((x2 - x1) * 0.20)
            pad_y = int((y2 - y1) * 0.20)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)

            bw, bh = x2 - x1, y2 - y1
            if bw > 10 and bh > 10:
                boxes.append((x1, y1, bw, bh))

        return boxes

    def close(self):
        try:
            self._landmarker.close()
        except Exception:
            pass


# ─── Hautfarb-Fallback ────────────────────────────────────────────────────────

class _SkinColorDetector:
    """
    Einfache Hautfarb-Detektion als Fallback wenn kein Modell vorhanden.
    Kein Download, läuft sofort.

    Erkennt Hautfarbregionen im HSV-Farbraum und gibt Bounding Boxes zurück.
    Weniger präzise als MediaPipe, aber besser als nichts.
    """

    def process(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        h_orig, w_orig = frame.shape[:2]

        # PERF: Intern auf 320px skalieren
        scale = 320 / w_orig
        small = cv2.resize(frame, (320, int(h_orig * scale)))

        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

        # Hautfarb-Bereich (deckt helle bis dunkle Hauttöne ab)
        lower1 = np.array([0,  20,  70], dtype=np.uint8)
        upper1 = np.array([20, 255, 255], dtype=np.uint8)
        lower2 = np.array([170, 20,  70], dtype=np.uint8)
        upper2 = np.array([180, 255, 255], dtype=np.uint8)

        mask = cv2.bitwise_or(
            cv2.inRange(hsv, lower1, upper1),
            cv2.inRange(hsv, lower2, upper2),
        )

        # Morphologie: Rauschen entfernen, Lücken schließen
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.dilate(mask, k, iterations=2)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        result = []
        for cnt in contours:
            if cv2.contourArea(cnt) < 1500:   # Zu klein ignorieren
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            # Seitenverhältnis filtern
            aspect = w / h if h > 0 else 0
            if not (0.2 < aspect < 4.0):
                continue
            # Zurück auf Original-Koordinaten
            x_o = max(0, int(x / scale))
            y_o = max(0, int(y / scale))
            w_o = min(w_orig - x_o, int(w / scale))
            h_o = min(h_orig - y_o, int(h / scale))
            if w_o > 20 and h_o > 20:
                result.append((x_o, y_o, w_o, h_o))

        return result


# ─── Öffentliche Klasse ───────────────────────────────────────────────────────

class HandDetector:
    """
    Handerkennung mit automatischem Fallback.

    Versucht MediaPipe HandLandmarker zu laden (einmaliger Download ~9 MB).
    Fällt automatisch auf Hautfarb-Detektion zurück wenn MediaPipe nicht
    verfügbar ist.

    detect() gibt immer eine Liste von (x, y, w, h) zurück.
    """

    def __init__(self, model_dir: str = "models", max_hands: int = 4):
        self._mediapipe: Optional[_MediaPipeHands]  = None
        self._skin                                   = _SkinColorDetector()
        self._mode                                   = "skin"   # "mediapipe" | "skin"

        # MediaPipe laden (mit Fallback)
        model_path = _download_model(model_dir)
        if model_path:
            try:
                self._mediapipe = _MediaPipeHands(model_path, max_hands)
                self._mode = "mediapipe"
            except Exception as e:
                print(f"MediaPipe HandLandmarker konnte nicht initialisiert werden: {e}")
                print("Fallback: Hautfarb-Detektion")

        print(f"Handerkennung Modus: {self._mode}")

    def is_available(self) -> bool:
        """Immer True – Hautfarb-Fallback ist immer verfügbar."""
        return True

    def get_mode(self) -> str:
        """Gibt 'mediapipe' oder 'skin' zurück."""
        return self._mode

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Erkennt Hände im Frame.
        Rückgabe: Liste von (x, y, w, h) in Original-Koordinaten.
        """
        if self._mode == "mediapipe" and self._mediapipe is not None:
            return self._mediapipe.process(frame)
        return self._skin.process(frame)

    def close(self):
        if self._mediapipe:
            self._mediapipe.close()
