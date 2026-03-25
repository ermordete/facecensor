#!/usr/bin/python3
"""
effects.py - Zensur-Effekte Modul
====================================
Alle verfügbaren Anonymisierungs-Effekte für Gesichtsregionen.
Leicht erweiterbar: neuen Effekt einfach als Methode hinzufügen
und in EFFECTS_REGISTRY registrieren.
"""

import cv2
import numpy as np
import os
from typing import Dict, Callable, List, Optional, Tuple


# ─── Effekt-Implementierungen ────────────────────────────────────────────────

def apply_gaussian_blur(region: np.ndarray, strength: int = 51) -> np.ndarray:
    """Weicher Gaussian Blur."""
    k = _odd_kernel(strength, 15, 101)
    return cv2.GaussianBlur(region, (k, k), 0)


def apply_strong_blur(region: np.ndarray, strength: int = 51) -> np.ndarray:
    """Starker Gaussian Blur mit höherem Sigma."""
    k = _odd_kernel(strength, 21, 101)
    return cv2.GaussianBlur(region, (k, k), 30)


def apply_box_blur(region: np.ndarray, strength: int = 25) -> np.ndarray:
    """Gleichmäßiger Box Blur."""
    k = _odd_kernel(strength, 9, 51)
    return cv2.blur(region, (k, k))


def apply_pixelate_light(region: np.ndarray, strength: int = 15) -> np.ndarray:
    """Leichte Pixelierung."""
    return _pixelate(region, max(4, strength // 4))


def apply_pixelate_heavy(region: np.ndarray, strength: int = 15) -> np.ndarray:
    """Starke Pixelierung."""
    return _pixelate(region, max(2, strength // 8))


def apply_black_bar(region: np.ndarray, strength: int = 50) -> np.ndarray:
    """Schwarzer Balken (klassische TV-Zensur)."""
    result = region.copy()
    result[:, :] = (0, 0, 0)
    return result


def apply_oval_blur(region: np.ndarray, strength: int = 51) -> np.ndarray:
    """
    Weicher Blur mit ovaler Maske für natürlicheres Aussehen.
    Bereich außerhalb der Ellipse bleibt unverändert.
    """
    h, w = region.shape[:2]
    
    # Blur des gesamten Bereichs
    k = _odd_kernel(strength, 15, 101)
    blurred = cv2.GaussianBlur(region, (k, k), 20)
    
    # Elliptische Maske erstellen
    mask = np.zeros((h, w), dtype=np.float32)
    cx, cy = w // 2, h // 2
    cv2.ellipse(mask, (cx, cy), (cx, cy), 0, 0, 360, 1.0, -1)
    
    # Maske weich machen
    mask = cv2.GaussianBlur(mask, (21, 21), 10)
    mask = mask[:, :, np.newaxis]  # Channel-Dimension für Broadcasting
    
    # Blend: Original × (1 - Maske) + Blur × Maske
    return (region * (1 - mask) + blurred * mask).astype(np.uint8)


def apply_emoji_overlay(
    frame: np.ndarray,
    x: int, y: int, w: int, h: int,
    emoji_img: Optional[np.ndarray],
    strength: int = 50,
) -> np.ndarray:
    """
    Legt ein Emoji-Bild über die Gesichtsregion.
    Wird direkt auf dem ganzen Frame aufgerufen (nicht nur Region).
    
    emoji_img: BGRA-Bild (mit Alpha-Kanal)
    """
    if emoji_img is None:
        # Fallback: farbiger Block
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (50, 205, 50), -1)
        return cv2.addWeighted(frame, 0.3, overlay, 0.7, 0)
    
    # Emoji auf Gesichtsgröße skalieren
    resized = cv2.resize(emoji_img, (w, h), interpolation=cv2.INTER_AREA)
    
    # Sicherstellen, dass die Koordinaten im Frame-Bereich liegen
    fh, fw = frame.shape[:2]
    x_end = min(x + w, fw)
    y_end = min(y + h, fh)
    x_start = max(0, x)
    y_start = max(0, y)
    
    emoji_w = x_end - x_start
    emoji_h = y_end - y_start
    
    if emoji_w <= 0 or emoji_h <= 0:
        return frame
    
    emoji_crop = resized[:emoji_h, :emoji_w]
    
    if emoji_crop.shape[2] == 4:
        # Mit Alpha-Kanal blenden
        alpha = emoji_crop[:, :, 3:4].astype(float) / 255.0
        emoji_bgr = emoji_crop[:, :, :3]
        roi = frame[y_start:y_end, x_start:x_end].astype(float)
        blended = roi * (1 - alpha) + emoji_bgr.astype(float) * alpha
        frame[y_start:y_end, x_start:x_end] = blended.astype(np.uint8)
    else:
        frame[y_start:y_end, x_start:x_end] = emoji_crop
    
    return frame


# ─── Hilfsfunktionen ─────────────────────────────────────────────────────────

def _odd_kernel(strength: int, min_k: int, max_k: int) -> int:
    """Berechnet eine ungerade Kernel-Größe aus einem Stärkewert."""
    k = int(min_k + (max_k - min_k) * (strength / 100.0))
    k = max(min_k, min(max_k, k))
    return k if k % 2 == 1 else k - 1


def _pixelate(region: np.ndarray, pixel_size: int) -> np.ndarray:
    """Pixeliert einen Bildbereich."""
    h, w = region.shape[:2]
    pixel_size = max(1, pixel_size)
    
    small_w = max(1, w // pixel_size)
    small_h = max(1, h // pixel_size)
    
    temp = cv2.resize(region, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)


# ─── Effekt-Registry ─────────────────────────────────────────────────────────

# Effekt-Definitionen: id → (Anzeigename, Icon, Funktion)
EFFECTS: Dict[str, Dict] = {
    "blur_gaussian":  {"name": "Gaussian Blur",   "icon": "🌫️",  "fn": apply_gaussian_blur},
    "blur_strong":    {"name": "Strong Blur",      "icon": "💨",  "fn": apply_strong_blur},
    "blur_box":       {"name": "Box Blur",         "icon": "📦",  "fn": apply_box_blur},
    "pixelate_light": {"name": "Pixel (leicht)",   "icon": "🔲",  "fn": apply_pixelate_light},
    "pixelate_heavy": {"name": "Pixel (stark)",    "icon": "🔳",  "fn": apply_pixelate_heavy},
    "black_bar":      {"name": "Black Bar",        "icon": "▬",   "fn": apply_black_bar},
    "oval_blur":      {"name": "Oval Blur",        "icon": "🫧",  "fn": apply_oval_blur},
    "emoji":          {"name": "Emoji",            "icon": "😎",  "fn": None},  # Sonderfall
}

# Emoji-Definitionen: id → (Anzeigename, Dateipfad oder None für Fallback)
EMOJIS: Dict[str, Dict] = {
    "sunglasses": {"name": "😎 Cool",   "file": "assets/emoji_sunglasses.png"},
    "laugh":      {"name": "😂 Laugh",  "file": "assets/emoji_laugh.png"},
    "ghost":      {"name": "👻 Ghost",  "file": "assets/emoji_ghost.png"},
    "robot":      {"name": "🤖 Robot",  "file": "assets/emoji_robot.png"},
    "cat":        {"name": "🐱 Cat",    "file": "assets/emoji_cat.png"},
    "fire":       {"name": "🔥 Fire",   "file": "assets/emoji_fire.png"},
    "heart":      {"name": "❤️  Heart", "file": "assets/emoji_heart.png"},
    "star":       {"name": "⭐ Star",   "file": "assets/emoji_star.png"},
}

# Preset-Definitionen für schnellen Moduswechsel
PRESETS: Dict[str, Dict] = {
    "interview":  {"name": "Interview",  "icon": "🎙️",  "effect": "blur_gaussian", "strength": 70},
    "street":     {"name": "Street",     "icon": "🏙️",  "effect": "pixelate_heavy","strength": 80},
    "streaming":  {"name": "Streaming",  "icon": "📺",  "effect": "blur_strong",   "strength": 60},
    "funny":      {"name": "Funny",      "icon": "😂",  "effect": "emoji",         "strength": 50},
}


class EffectsProcessor:
    """
    Zentrale Klasse für alle Zensur-Effekte.
    Verwaltet aktuellen Effekt, Stärke, Emoji-Auswahl.
    """
    
    def __init__(self, assets_dir: str = "assets"):
        self.current_effect = "blur_gaussian"
        self.strength = 50  # 0–100
        self.current_emoji = "sunglasses"
        self.assets_dir = assets_dir
        
        # Emoji-Bilder cachen
        self._emoji_cache: Dict[str, Optional[np.ndarray]] = {}
        self._load_emoji_assets()
    
    def _load_emoji_assets(self):
        """Lädt alle Emoji-Bilder (BGRA) in den Cache."""
        os.makedirs(self.assets_dir, exist_ok=True)
        
        for key, info in EMOJIS.items():
            path = info["file"]
            if os.path.exists(path):
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                self._emoji_cache[key] = img
                if img is not None:
                    print(f"✅ Emoji geladen: {key}")
                else:
                    self._emoji_cache[key] = None
                    print(f"⚠️  Emoji konnte nicht geladen werden: {path}")
            else:
                self._emoji_cache[key] = None
                # Erstelle Platzhalter-Emoji als farbiges Rechteck
                self._emoji_cache[key] = self._create_placeholder_emoji(key)
    
    def _create_placeholder_emoji(self, key: str) -> np.ndarray:
        """
        Erstellt ein einfaches Platzhalter-Emoji als farbiges Kreissymbol.
        Wird verwendet wenn keine PNG-Datei vorhanden ist.
        """
        size = 64
        img = np.zeros((size, size, 4), dtype=np.uint8)
        
        # Verschiedene Farben je Emoji
        colors = {
            "sunglasses": (50, 50, 200, 255),
            "laugh":      (50, 200, 200, 255),
            "ghost":      (200, 200, 200, 255),
            "robot":      (100, 100, 150, 255),
            "cat":        (100, 180, 255, 255),
            "fire":       (50, 100, 255, 255),
            "heart":      (50, 50, 255, 255),
            "star":       (50, 200, 255, 255),
        }
        
        color = colors.get(key, (150, 150, 150, 255))
        cv2.circle(img, (size // 2, size // 2), size // 2 - 2, color, -1)
        
        return img
    
    def apply(
        self,
        frame: np.ndarray,
        faces: List[Tuple[int, int, int, int]],
    ) -> np.ndarray:
        """
        Wendet den aktuellen Effekt auf alle erkannten Gesichter an.
        
        Returns: Modifiziertes Frame
        """
        for (x, y, w, h) in faces:
            frame = self._apply_to_face(frame, x, y, w, h)
        return frame
    
    def _apply_to_face(
        self,
        frame: np.ndarray,
        x: int, y: int, w: int, h: int,
    ) -> np.ndarray:
        """Wendet Effekt auf eine einzelne Gesichtsregion an."""
        # Grenzen sicherstellen
        fh, fw = frame.shape[:2]
        x = max(0, x)
        y = max(0, y)
        w = min(fw - x, w)
        h = min(fh - y, h)
        
        if w <= 0 or h <= 0:
            return frame
        
        # Emoji: direkter Frame-Overlay
        if self.current_effect == "emoji":
            emoji_img = self._emoji_cache.get(self.current_emoji)
            return apply_emoji_overlay(frame, x, y, w, h, emoji_img, self.strength)
        
        # Alle anderen Effekte: Region extrahieren, bearbeiten, einsetzen
        effect_info = EFFECTS.get(self.current_effect)
        if effect_info is None or effect_info["fn"] is None:
            return frame
        
        region = frame[y:y + h, x:x + w].copy()
        if region.size == 0:
            return frame
        
        processed = effect_info["fn"](region, self.strength)
        frame[y:y + h, x:x + w] = processed
        
        return frame
    
    def apply_preset(self, preset_id: str):
        """Aktiviert ein Preset."""
        preset = PRESETS.get(preset_id)
        if preset:
            self.current_effect = preset["effect"]
            self.strength = preset["strength"]
    
    def set_effect(self, effect_id: str):
        """Setzt den aktuellen Effekt."""
        if effect_id in EFFECTS:
            self.current_effect = effect_id
    
    def set_strength(self, value: int):
        """Setzt die Effektstärke (0–100)."""
        self.strength = max(0, min(100, value))
    
    def set_emoji(self, emoji_id: str):
        """Setzt das aktuelle Emoji."""
        if emoji_id in EMOJIS:
            self.current_emoji = emoji_id
            self.current_effect = "emoji"
    
    def get_effect_name(self) -> str:
        return EFFECTS.get(self.current_effect, {}).get("name", "Unknown")
    
    def get_effect_icon(self) -> str:
        return EFFECTS.get(self.current_effect, {}).get("icon", "?")
