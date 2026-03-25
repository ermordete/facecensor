#!/usr/bin/python3
"""
effects.py - Zensur-Effekte Modul
====================================
Änderungen v2:
- "Gaussian Blur" und "Strong Blur" vollständig entfernt
- Pixel-Logik korrigiert: leicht = feinere Blöcke, stark = grobe Blöcke
- Emoji-Icons aus allen Namen/Labels entfernt
- Emoji-Overlay-Bug behoben: robustes Clipping + sicheres Alpha-Blending
- Presets bereinigt (keine Referenzen auf entfernte Effekte)
"""

import cv2
import numpy as np
import os
from typing import Dict, List, Optional, Tuple


# ─── Effekt-Implementierungen ────────────────────────────────────────────────

def apply_box_blur(region: np.ndarray, strength: int = 25) -> np.ndarray:
    """Gleichmäßiger Box Blur."""
    k = _odd_kernel(strength, 9, 51)
    return cv2.blur(region, (k, k))


def apply_pixelate_light(region: np.ndarray, strength: int = 50) -> np.ndarray:
    """
    Leichte Pixelierung.

    KORREKTUR: "leicht" = feinere, weniger grobe Pixel-Blöcke.
    Kleinerer pixel_size → mehr Blöcke → weniger auffällig.
    Bei strength=50: pixel_size ~8
    """
    pixel_size = max(4, int(4 + strength * 0.08))   # Range: ~4–12
    return _pixelate(region, pixel_size)


def apply_pixelate_heavy(region: np.ndarray, strength: int = 50) -> np.ndarray:
    """
    Starke Pixelierung.

    KORREKTUR: "stark" = sehr grobe, deutlich sichtbare Pixel-Blöcke.
    Größerer pixel_size → weniger Blöcke → maximal auffällig.
    Bei strength=50: pixel_size ~25
    """
    pixel_size = max(10, int(10 + strength * 0.3))  # Range: ~10–40
    return _pixelate(region, pixel_size)


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
    k = _odd_kernel(strength, 15, 101)
    blurred = cv2.GaussianBlur(region, (k, k), 20)

    mask = np.zeros((h, w), dtype=np.float32)
    cx, cy = w // 2, h // 2
    cv2.ellipse(mask, (cx, cy), (cx, cy), 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (21, 21), 10)
    mask = mask[:, :, np.newaxis]

    return (region * (1 - mask) + blurred * mask).astype(np.uint8)


def apply_emoji_overlay(
    frame: np.ndarray,
    x: int, y: int, w: int, h: int,
    emoji_img: Optional[np.ndarray],
) -> np.ndarray:
    """
    Legt ein Emoji-Bild zuverlässig über die Gesichtsregion.

    Bugfixes v2:
    - Koordinaten werden VOR der Skalierung auf Frame-Grenzen geclampt
    - Emoji wird auf die tatsächliche (geclamnte) Zielgröße skaliert
    - float32-Alpha-Blending für stabile Ergebnisse bei allen Gesichtsgrößen
    - Robuster Fallback (farbige Box) wenn kein Bild vorhanden
    """
    fh, fw = frame.shape[:2]

    # Koordinaten auf Frame-Grenzen clampen
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(fw, x + w)
    y2 = min(fh, y + h)

    draw_w = x2 - x1
    draw_h = y2 - y1

    if draw_w <= 0 or draw_h <= 0:
        return frame

    # Fallback: solide Farbe wenn kein Emoji-Asset vorhanden
    if emoji_img is None or emoji_img.size == 0:
        frame[y1:y2, x1:x2] = (80, 60, 200)
        return frame

    # Emoji auf tatsächliche Zielgröße skalieren (nach Clamp)
    resized = cv2.resize(emoji_img, (draw_w, draw_h), interpolation=cv2.INTER_AREA)

    roi = frame[y1:y2, x1:x2]

    if resized.ndim == 3 and resized.shape[2] == 4:
        alpha = resized[:, :, 3:4].astype(np.float32) / 255.0
        emoji_bgr = resized[:, :, :3].astype(np.float32)
        roi_f = roi.astype(np.float32)
        blended = roi_f * (1.0 - alpha) + emoji_bgr * alpha
        frame[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
    else:
        frame[y1:y2, x1:x2] = resized[:, :, :3]

    return frame


# ─── Hilfsfunktionen ─────────────────────────────────────────────────────────

def _odd_kernel(strength: int, min_k: int, max_k: int) -> int:
    """Berechnet eine ungerade Kernel-Größe aus einem Stärkewert."""
    k = int(min_k + (max_k - min_k) * (strength / 100.0))
    k = max(min_k, min(max_k, k))
    return k if k % 2 == 1 else k - 1


def _pixelate(region: np.ndarray, pixel_size: int) -> np.ndarray:
    """Pixeliert einen Bildbereich. Größerer pixel_size = gröbere Pixel."""
    h, w = region.shape[:2]
    pixel_size = max(1, pixel_size)
    small_w = max(1, w // pixel_size)
    small_h = max(1, h // pixel_size)
    temp = cv2.resize(region, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)


# ─── Effekt-Registry ─────────────────────────────────────────────────────────
# Gaussian Blur und Strong Blur vollständig entfernt.
# Keine Emoji-Icons in den Namen.

EFFECTS: Dict[str, Dict] = {
    "blur_box":       {"name": "Box Blur",       "fn": apply_box_blur},
    "pixelate_light": {"name": "Pixel (leicht)", "fn": apply_pixelate_light},
    "pixelate_heavy": {"name": "Pixel (stark)",  "fn": apply_pixelate_heavy},
    "black_bar":      {"name": "Black Bar",      "fn": apply_black_bar},
    "oval_blur":      {"name": "Oval Blur",      "fn": apply_oval_blur},
    "emoji":          {"name": "Emoji Overlay",  "fn": None},  # Sonderfall
}

DEFAULT_EFFECT = "blur_box"

# Keine Emoji-Zeichen in den Namen
EMOJIS: Dict[str, Dict] = {
    "sunglasses": {"name": "Cool",  "file": "assets/emoji_sunglasses.png"},
    "laugh":      {"name": "Laugh", "file": "assets/emoji_laugh.png"},
    "ghost":      {"name": "Ghost", "file": "assets/emoji_ghost.png"},
    "robot":      {"name": "Robot", "file": "assets/emoji_robot.png"},
    "cat":        {"name": "Cat",   "file": "assets/emoji_cat.png"},
    "fire":       {"name": "Fire",  "file": "assets/emoji_fire.png"},
    "heart":      {"name": "Heart", "file": "assets/emoji_heart.png"},
    "star":       {"name": "Star",  "file": "assets/emoji_star.png"},
}

# Presets: keine entfernten Effekte, keine Emoji-Icons in Namen
PRESETS: Dict[str, Dict] = {
    "interview": {"name": "Interview", "effect": "oval_blur",      "strength": 70},
    "street":    {"name": "Street",    "effect": "pixelate_heavy", "strength": 80},
    "streaming": {"name": "Streaming", "effect": "blur_box",       "strength": 60},
    "funny":     {"name": "Funny",     "effect": "emoji",          "strength": 50},
}


class EffectsProcessor:
    """
    Zentrale Klasse für alle Zensur-Effekte.
    Verwaltet aktuellen Effekt, Stärke, Emoji-Auswahl.
    """

    def __init__(self, assets_dir: str = "assets"):
        self.current_effect = DEFAULT_EFFECT
        self.strength = 50
        self.current_emoji = "sunglasses"
        self.assets_dir = assets_dir
        self._emoji_cache: Dict[str, Optional[np.ndarray]] = {}
        self._load_emoji_assets()

    def _load_emoji_assets(self):
        """Lädt alle Emoji-Bilder (BGRA) in den Cache."""
        os.makedirs(self.assets_dir, exist_ok=True)

        for key, info in EMOJIS.items():
            path = info["file"]
            if os.path.exists(path):
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    # Immer auf 4 Kanäle normalisieren
                    if img.ndim == 2:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
                    elif img.shape[2] == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                    self._emoji_cache[key] = img
                    print(f"Emoji geladen: {key}")
                else:
                    self._emoji_cache[key] = self._create_placeholder_emoji(key)
            else:
                self._emoji_cache[key] = self._create_placeholder_emoji(key)

    def _create_placeholder_emoji(self, key: str) -> np.ndarray:
        """Erstellt ein robustes BGRA-Platzhalter-Bild (immer 4 Kanäle)."""
        size = 128
        img = np.zeros((size, size, 4), dtype=np.uint8)
        colors = {
            "sunglasses": (50,  50,  200, 255),
            "laugh":      (50,  200, 200, 255),
            "ghost":      (200, 200, 200, 255),
            "robot":      (100, 100, 150, 255),
            "cat":        (100, 180, 255, 255),
            "fire":       (50,  100, 255, 255),
            "heart":      (50,  50,  255, 255),
            "star":       (50,  200, 255, 255),
        }
        color = colors.get(key, (150, 150, 150, 255))
        cv2.circle(img, (size // 2, size // 2), size // 2 - 4, color, -1)
        cv2.circle(img, (size // 2, size // 2), size // 2 - 4, (255, 255, 255, 180), 3)
        return img

    def apply(
        self,
        frame: np.ndarray,
        faces: List[Tuple[int, int, int, int]],
    ) -> np.ndarray:
        """Wendet den aktuellen Effekt auf alle erkannten Gesichter an."""
        for (x, y, w, h) in faces:
            frame = self._apply_to_face(frame, x, y, w, h)
        return frame

    def _apply_to_face(
        self,
        frame: np.ndarray,
        x: int, y: int, w: int, h: int,
    ) -> np.ndarray:
        """Wendet Effekt auf eine einzelne Gesichtsregion an."""
        fh, fw = frame.shape[:2]
        x = max(0, x)
        y = max(0, y)
        w = min(fw - x, w)
        h = min(fh - y, h)

        if w <= 0 or h <= 0:
            return frame

        if self.current_effect == "emoji":
            emoji_img = self._emoji_cache.get(self.current_emoji)
            return apply_emoji_overlay(frame, x, y, w, h, emoji_img)

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
        preset = PRESETS.get(preset_id)
        if preset:
            self.current_effect = preset["effect"]
            self.strength = preset["strength"]

    def set_effect(self, effect_id: str):
        if effect_id in EFFECTS:
            self.current_effect = effect_id

    def set_strength(self, value: int):
        self.strength = max(0, min(100, value))

    def set_emoji(self, emoji_id: str):
        if emoji_id in EMOJIS:
            self.current_emoji = emoji_id
            self.current_effect = "emoji"

    def get_effect_name(self) -> str:
        return EFFECTS.get(self.current_effect, {}).get("name", "Unknown")
