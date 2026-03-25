#!/usr/bin/python3
"""
detector.py - Gesichtserkennung Modul
=======================================
Verbesserte Gesichtserkennung mit OpenCV DNN Face Detector
und temporalem Smoothing gegen Flackern.

Warum DNN statt Haar Cascade?
- Deutlich weniger False Positives
- Robuster bei verschiedenen Winkeln / Beleuchtung
- Ähnliche Performance auf Jetson Nano
- Moderner und genauer
"""

import cv2
import numpy as np
import os
import urllib.request
from typing import List, Tuple
from dataclasses import dataclass, field


@dataclass
class TrackedFace:
    """Repräsentiert ein tracktes Gesicht mit Glättungs-Historie."""
    x: float
    y: float
    w: float
    h: float
    confidence: float
    missed_frames: int = 0
    # Letzten N Positionen für Glättung
    history: List[Tuple[float, float, float, float]] = field(default_factory=list)
    
    def to_int_rect(self) -> Tuple[int, int, int, int]:
        return (int(self.x), int(self.y), int(self.w), int(self.h))


class FaceDetector:
    """
    Gesichtsdetektor mit OpenCV DNN und temporalem Smoothing.
    
    Temporal Smoothing:
    - Positionen werden über N Frames gemittelt → kein Springen
    - Gesichter die kurz "verschwinden" werden noch kurz gehalten
    - IoU-basiertes Tracking ordnet Detektionen vorhandenen Tracks zu
    """
    
    # DNN Modell URLs (Caffe ResNet-SSD)
    MODEL_URL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
    CONFIG_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
    
    def __init__(
        self,
        confidence_threshold: float = 0.5,
        smooth_frames: int = 5,
        max_missed_frames: int = 8,
        face_padding: float = 0.15,
        model_dir: str = "models",
    ):
        self.confidence_threshold = confidence_threshold
        self.smooth_frames = smooth_frames
        self.max_missed_frames = max_missed_frames
        self.face_padding = face_padding  # Prozentuale Vergrößerung der Bounding Box
        
        self._tracked_faces: List[TrackedFace] = []
        self._net = None
        self._use_dnn = False
        self._haar_cascade = None
        
        # Modell laden
        self._load_detector(model_dir)
    
    def _load_detector(self, model_dir: str):
        """Lädt DNN-Modell, fällt auf Haar Cascade zurück."""
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, "face_detector.caffemodel")
        config_path = os.path.join(model_dir, "deploy.prototxt")
        
        # Versuche DNN-Modell zu laden
        if os.path.exists(model_path) and os.path.exists(config_path):
            try:
                self._net = cv2.dnn.readNetFromCaffe(config_path, model_path)
                self._use_dnn = True
                print("✅ DNN Face Detector geladen")
                return
            except Exception as e:
                print(f"⚠️  DNN-Modell Fehler: {e}")
        
        # DNN-Modell herunterladen
        print("📥 Lade DNN Face Detector Modell herunter...")
        try:
            urllib.request.urlretrieve(self.MODEL_URL, model_path)
            urllib.request.urlretrieve(self.CONFIG_URL, config_path)
            self._net = cv2.dnn.readNetFromCaffe(config_path, model_path)
            self._use_dnn = True
            print("✅ DNN Face Detector geladen (heruntergeladen)")
            return
        except Exception as e:
            print(f"⚠️  Download fehlgeschlagen: {e}")
        
        # Fallback: Haar Cascade
        print("↩️  Fallback: Haar Cascade")
        self._load_haar_cascade()
    
    def _load_haar_cascade(self):
        """Lädt Haar Cascade als Fallback."""
        paths = [
            '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
            '/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml',
            '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
        ]
        for p in paths:
            if os.path.exists(p):
                self._haar_cascade = cv2.CascadeClassifier(p)
                if not self._haar_cascade.empty():
                    print(f"✅ Haar Cascade geladen: {p}")
                    return
        print("❌ Kein Detektor verfügbar!")
    
    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Erkennt Gesichter und gibt geglättete Bounding Boxes zurück.
        
        Intern:
        1. Neue Detektionen ermitteln
        2. Mit bestehenden Tracks assoziieren (IoU)
        3. Positionen glätten
        4. Padding anwenden
        
        Returns: Liste von (x, y, w, h) Tuples
        """
        # Für Detection intern kleiner skalieren → schneller
        detect_frame, scale = self._prepare_detect_frame(frame)
        
        # Rohe Detektionen
        raw_detections = self._raw_detect(detect_frame)
        
        # Zurück auf Originalmaßstab skalieren
        scaled_detections = [
            (int(x / scale), int(y / scale), int(w / scale), int(h / scale))
            for (x, y, w, h) in raw_detections
        ]
        
        # Tracks aktualisieren
        self._update_tracks(scaled_detections, frame.shape)
        
        # Geglättete Positionen mit Padding zurückgeben
        result = []
        for face in self._tracked_faces:
            if face.missed_frames == 0:  # Nur aktive Faces
                x, y, w, h = self._apply_padding(face, frame.shape)
                result.append((x, y, w, h))
        
        return result
    
    def _prepare_detect_frame(
        self, frame: np.ndarray, target_width: int = 300
    ) -> Tuple[np.ndarray, float]:
        """
        Skaliert Frame für schnelle Detektion herunter.
        Gibt (skalierter_frame, scale_faktor) zurück.
        """
        h, w = frame.shape[:2]
        scale = target_width / w
        new_h = int(h * scale)
        small = cv2.resize(frame, (target_width, new_h))
        return small, scale
    
    def _raw_detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Führt eigentliche Detektion durch (DNN oder Haar Cascade)."""
        if self._use_dnn and self._net is not None:
            return self._dnn_detect(frame)
        elif self._haar_cascade is not None:
            return self._haar_detect(frame)
        return []
    
    def _dnn_detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """OpenCV DNN Gesichtserkennung (ResNet-SSD)."""
        h, w = frame.shape[:2]
        
        # Blob erstellen (normalisiert für das Modell)
        blob = cv2.dnn.blobFromImage(
            frame, 1.0, (300, 300),
            (104.0, 177.0, 123.0),
            swapRB=False
        )
        self._net.setInput(blob)
        detections = self._net.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < self.confidence_threshold:
                continue
            
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            
            # Clamp auf Frame-Grenzen
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            fw, fh = x2 - x1, y2 - y1
            if fw > 10 and fh > 10:
                faces.append((x1, y1, fw, fh))
        
        return faces
    
    def _haar_detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Haar Cascade Fallback."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected = self._haar_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        if len(detected) == 0:
            return []
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in detected]
    
    def _update_tracks(
        self,
        detections: List[Tuple[int, int, int, int]],
        frame_shape: Tuple[int, ...],
    ):
        """
        Assoziiert neue Detektionen mit bestehenden Tracks via IoU.
        Nicht gematchte Tracks werden als 'missed' markiert.
        """
        matched_track_ids = set()
        matched_det_ids = set()
        
        # Für jede Detektion: besten passenden Track suchen
        for det_id, det in enumerate(detections):
            best_iou = 0.3  # Minimum IoU-Schwelle
            best_track_id = -1
            
            for track_id, face in enumerate(self._tracked_faces):
                if track_id in matched_track_ids:
                    continue
                iou = self._compute_iou(det, face.to_int_rect())
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = track_id
            
            if best_track_id >= 0:
                # Track aktualisieren
                face = self._tracked_faces[best_track_id]
                face.missed_frames = 0
                face.confidence = best_iou
                
                # Zur History hinzufügen
                face.history.append(det)
                if len(face.history) > self.smooth_frames:
                    face.history.pop(0)
                
                # Geglättete Position berechnen
                avg = np.mean(face.history, axis=0)
                face.x, face.y, face.w, face.h = avg
                
                matched_track_ids.add(best_track_id)
                matched_det_ids.add(det_id)
        
        # Nicht gematchte Tracks: missed_frames erhöhen
        for track_id, face in enumerate(self._tracked_faces):
            if track_id not in matched_track_ids:
                face.missed_frames += 1
        
        # Neue Tracks für ungematchte Detektionen erstellen
        for det_id, det in enumerate(detections):
            if det_id not in matched_det_ids:
                x, y, w, h = det
                new_face = TrackedFace(
                    x=float(x), y=float(y),
                    w=float(w), h=float(h),
                    confidence=1.0,
                    history=[det],
                )
                self._tracked_faces.append(new_face)
        
        # Abgelaufene Tracks entfernen
        self._tracked_faces = [
            f for f in self._tracked_faces
            if f.missed_frames <= self.max_missed_frames
        ]
    
    def _apply_padding(
        self,
        face: TrackedFace,
        frame_shape: Tuple[int, ...],
    ) -> Tuple[int, int, int, int]:
        """Vergrößert Bounding Box um padding_factor für bessere Abdeckung."""
        pad_x = int(face.w * self.face_padding)
        pad_y = int(face.h * self.face_padding)
        
        x = max(0, int(face.x) - pad_x)
        y = max(0, int(face.y) - pad_y)
        w = min(frame_shape[1] - x, int(face.w) + 2 * pad_x)
        h = min(frame_shape[0] - y, int(face.h) + 2 * pad_y)
        
        return (x, y, w, h)
    
    @staticmethod
    def _compute_iou(
        a: Tuple[int, int, int, int],
        b: Tuple[int, int, int, int],
    ) -> float:
        """Berechnet Intersection over Union zweier Bounding Boxes."""
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        
        ix1 = max(ax, bx)
        iy1 = max(ay, by)
        ix2 = min(ax + aw, bx + bw)
        iy2 = min(ay + ah, by + bh)
        
        if ix2 < ix1 or iy2 < iy1:
            return 0.0
        
        intersection = (ix2 - ix1) * (iy2 - iy1)
        area_a = aw * ah
        area_b = bw * bh
        union = area_a + area_b - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def set_confidence_threshold(self, value: float):
        self.confidence_threshold = max(0.1, min(1.0, value))
    
    def reset_tracks(self):
        """Alle Tracks zurücksetzen."""
        self._tracked_faces.clear()
