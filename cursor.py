import cv2
import sys
import time
from typing import Optional

import numpy as np
import handtrackingModule as htm
import pyautogui
import math
from pycaw.pycaw import AudioUtilities

try:
    import screen_brightness_control as sbc
except ImportError:
    sbc = None

try:
    import winsound
except ImportError:
    winsound = None

wCam, hCam = 640, 480
frameR = 40
smoothening = 2.5
moveDeadZone = 2.0

wasPinching = False
mode = "cursor"
modes = ["cursor", "volume", "brightness", "keyboard", "drawing"]
exitHoldSeconds = 0.8
exitGestureStart = None
switchHoldSeconds = 1.25
# Ignore mode-switch gesture while thumb and index are close (same pose as volume/brightness pinch).
switchMinThumbIndexDist = 90
switchGestureStart = None
scrollThreshold = 7
scrollAmount = 90
scrollCooldown = 0.015
lastScrollY = None
lastScrollTime = 0.0
scrollAccumulator = 0.0
keyboardPinchThreshold = 32
keyboard_pinch_prev = False

# Drawing mode setup.
canvas = np.zeros((hCam, wCam, 3), dtype=np.uint8)
# penColor: last palette pick; drawing uses mirrored coords (drawX) to match on-screen hand.
penColor = (0, 255, 255)
drawPrevX, drawPrevY = None, None
clearLatch = False
cursorLocked = False
cursorToggleLatch = False
drawEraserMode = False
eraserToggleLatch = False
palettePinchLatch = False
# Basic BGR swatches (red, green, blue, yellow); pinch thumb+index on a swatch to select.
DRAW_PALETTE_COLORS = [
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
    (0, 255, 255),
]
DRAW_PALETTE_SW = 56
DRAW_PALETTE_SH = 30
DRAW_PALETTE_GAP = 6
DRAW_PALETTE_TOP = 34
DRAW_PALETTE_PREVIEW_W = 36
DRAW_PALETTE_PREVIEW_GAP = 14
# Looser than keyboard typing so color pick registers more reliably.
DRAW_PALETTE_PINCH_THRESH = 48
# Brief flash on preview after a successful pick.
palette_pick_flash_until = 0.0
# Set each frame in drawing mode for palette UI (hover / hints).
draw_palette_hover_idx = None
draw_palette_wants_pinch = False
# Last brush size label for drawing UI (kept when hand briefly lost).
drawBrushThicknessDisplay = 6
# Fingertip cursor drawn after canvas blend so it stays fully visible (not dimmed by addWeighted).
draw_overlay_tip_show = False
draw_overlay_tip_x = 0
draw_overlay_tip_y = 0
draw_overlay_tip_fill = (0, 255, 255)
draw_overlay_tip_hover = False

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, wCam)
cap.set(4, hCam)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

detector = htm.handDetector(maxHands=1)

wScr, hScr = pyautogui.size()

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0
pyautogui.MINIMUM_DURATION = 0

cv2.namedWindow("Virtual Mouse", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Virtual Mouse", cv2.WND_PROP_TOPMOST, 1)

drawOverlay = False

# System volume setup.
device = AudioUtilities.GetSpeakers()
volume = device.EndpointVolume
minVol, maxVol, _ = volume.GetVolumeRange()
lastVol = volume.GetMasterVolumeLevel()
lastVolScalar = volume.GetMasterVolumeLevelScalar()
lastVolUpdate = 0.0
volumeLocked = False
volumeToggleLatch = False
brightnessLocked = False
brightnessToggleLatch = False
# Stops trying after repeated set failures (e.g. no WMI/DDC display). Never conflate with "pip missing".
brightnessGiveUp = False
brightnessValue = 50
lastBrightnessUpdate = 0.0
brightnessApplyFailures = 0


def _read_system_brightness():
    """Return first valid brightness reading, or None if unavailable."""
    if sbc is None:
        return None
    try:
        for v in sbc.get_brightness():
            if v is not None:
                return int(v)
    except Exception:
        pass
    for idx in range(4):
        try:
            vals = sbc.get_brightness(display=idx)
            for v in vals or []:
                if v is not None:
                    return int(v)
        except Exception:
            continue
    return None


def _set_system_brightness(level):
    """Apply brightness; prefer all controllable displays, then display=0. Returns True on success."""
    if sbc is None:
        return False
    level = int(max(0, min(100, level)))
    try:
        sbc.set_brightness(level)
        return True
    except Exception:
        pass
    try:
        sbc.set_brightness(level, display=0)
        return True
    except Exception:
        return False


if sbc is not None:
    got = _read_system_brightness()
    if got is not None:
        brightnessValue = got


class SimpleKeyboard:
    """Draws QWERTY + SPACE/BACK/ENTER on the main camera frame only (no extra window)."""

    def __init__(self, frame_width: int, frame_height: int):
        self.frame_w = frame_width
        self.frame_h = frame_height
        self.keys = [
            ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
            ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
            ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"],
            ["SPACE", "BACKSPACE", "DELETE", "ENTER"],
        ]
        self.key_rects = []
        self.last_press_time = 0.0
        self.press_cooldown = 0.12
        self._typed_preview = ""
        self._press_key = None
        self._press_until = 0.0
        self.KEY_W = 45
        self.KEY_H = 40
        self.KEY_GAP = 5
        # Fits 4 rows in lower part of 480p frame (tutorial-style bottom band).
        self.KEY_START_Y = 305
        self._bg_alpha = 0.5
        self._face_default = np.array([210, 210, 210], dtype=np.float32)
        self._face_hover = np.array([180, 230, 255], dtype=np.float32)
        self._face_press = np.array([100, 220, 255], dtype=np.float32)
        self._face_glow = np.array([120, 200, 255], dtype=np.float32)
        self._border = (255, 255, 255)
        self._text_color = (0, 0, 0)
        # Performance optimization
        self._last_draw_time = 0.0
        self._keyboard_fps_limit = 30.0
        self._min_draw_interval = 1.0 / self._keyboard_fps_limit
        self._cached_background = None
        self._cached_key_surfaces = {}
        self._enable_haptic = True
        self._last_hovered_key = None
        self._hover_start_time = 0.0
        # Performance metrics
        self._keyboard_fps = 0.0
        self._last_fps_update = 0.0
        self._fps_frame_count = 0
        self._performance_mode = False
        
        # Temporal smoothing and stability
        self._smoothed_x = 0.0
        self._smoothed_y = 0.0
        self._smoothing_alpha = 0.3  # EMA smoothing factor
        self._hover_debounce_time = 0.2  # 200ms to confirm hover
        self._hover_candidate_key = None
        self._hover_candidate_start = 0.0
        self._confirmed_hovered_key = None
        self._pinch_stable_time = 0.1  # 100ms stable pinch before click
        self._pinch_candidate_start = 0.0
        self._pinch_confirmed = False
        self._key_dead_zone = 8  # Extra margin around keys
        self._last_drawn_hovered_key = None
        self._finger_moved_threshold = 5.0  # Minimum movement to trigger redraw
        self._build_rects()
        self._pre_render_keys()

    def _row_width_letters(self, n_keys: int) -> int:
        return n_keys * self.KEY_W + (n_keys - 1) * self.KEY_GAP

    def _build_rects(self):
        self.key_rects = []
        y = self.KEY_START_Y
        for ri, row in enumerate(self.keys):
            if ri < 3:
                rw = self._row_width_letters(len(row))
                x = (self.frame_w - rw) // 2
                for k in row:
                    self.key_rects.append((k, x, y, self.KEY_W, self.KEY_H))
                    x += self.KEY_W + self.KEY_GAP
            else:
                w_space = 5 * self.KEY_W + 4 * self.KEY_GAP
                w_bs = 2 * self.KEY_W + self.KEY_GAP
                w_del = 2 * self.KEY_W + self.KEY_GAP
                w_ent = 2 * self.KEY_W + self.KEY_GAP
                rw = w_space + w_bs + w_del + w_ent + 3 * self.KEY_GAP
                x = (self.frame_w - rw) // 2
                for k, kw in [("SPACE", w_space), ("BACKSPACE", w_bs), ("DELETE", w_del), ("ENTER", w_ent)]:
                    self.key_rects.append((k, x, y, kw, self.KEY_H))
                    x += kw + self.KEY_GAP
            y += self.KEY_H + self.KEY_GAP

    def _label_for_draw(self, key: str) -> str:
        if key == "BACKSPACE":
            return "\u232b"
        if key == "DELETE":
            return "Del"
        if key == "ENTER":
            return "\u21b5"
        if key == "SPACE":
            return "Space"
        return key

    def key_at(self, x_pos: float, y_pos: float) -> Optional[str]:
        for item in self.key_rects:
            k, x, y, w, h = item
            # Add dead zone around keys for hysteresis
            margin = self._key_dead_zone
            if x - margin <= x_pos <= x + w + margin and y - margin <= y_pos <= y + h + margin:
                return k
        return None

    def _smooth_finger_position(self, raw_x: float, raw_y: float) -> tuple[float, float]:
        """Apply exponential moving average to smooth finger position."""
        if self._smoothed_x == 0.0 and self._smoothed_y == 0.0:
            # Initialize on first call
            self._smoothed_x = raw_x
            self._smoothed_y = raw_y
        else:
            # Apply EMA smoothing
            self._smoothed_x = self._smoothing_alpha * raw_x + (1 - self._smoothing_alpha) * self._smoothed_x
            self._smoothed_y = self._smoothing_alpha * raw_y + (1 - self._smoothing_alpha) * self._smoothed_y
        return self._smoothed_x, self._smoothed_y

    def _update_stable_hover(self, candidate_key: Optional[str], now: float) -> Optional[str]:
        """Update stable hover state with debouncing."""
        if candidate_key != self._hover_candidate_key:
            # New candidate, reset timer
            self._hover_candidate_key = candidate_key
            self._hover_candidate_start = now
        else:
            # Same candidate, check if debounce time passed
            if candidate_key and now - self._hover_candidate_start >= self._hover_debounce_time:
                # Confirm this as the hovered key
                if candidate_key != self._confirmed_hovered_key:
                    self._confirmed_hovered_key = candidate_key
                    self._hover_start_time = now  # Reset for preview timing
                return candidate_key
            elif not candidate_key:
                # No candidate, clear confirmed hover
                self._confirmed_hovered_key = None
        
        return self._confirmed_hovered_key

    def _should_trigger_pinch(self, is_pinching: bool, now: float) -> bool:
        """Determine if pinch should trigger with stability check."""
        if is_pinching:
            if not self._pinch_confirmed:
                # Start or continue pinch candidate
                if self._pinch_candidate_start == 0.0:
                    self._pinch_candidate_start = now
                # Check if pinch is stable for required time
                if now - self._pinch_candidate_start >= self._pinch_stable_time:
                    self._pinch_confirmed = True
                    return True
        else:
            # Reset pinch state
            self._pinch_candidate_start = 0.0
            self._pinch_confirmed = False
        
        return False

    def _finger_moved_significantly(self, x: float, y: float) -> bool:
        """Check if finger moved enough to warrant redraw."""
        if not hasattr(self, '_last_finger_x'):
            self._last_finger_x = x
            self._last_finger_y = y
            return True
        
        distance = math.hypot(x - self._last_finger_x, y - self._last_finger_y)
        if distance >= self._finger_moved_threshold:
            self._last_finger_x = x
            self._last_finger_y = y
            return True
        return False

    def _keyboard_roi_bounds(self):
        if not self.key_rects:
            return None
        min_y = min(r[2] for r in self.key_rects)
        max_y = max(r[2] + r[4] for r in self.key_rects)
        min_x = min(r[1] for r in self.key_rects)
        max_x = max(r[1] + r[3] for r in self.key_rects)
        pad = 6
        return (
            max(0, min_x - pad),
            max(0, min_y - pad),
            min(self.frame_w, max_x + pad),
            min(self.frame_h, max_y + pad),
        )

    def _pre_render_keys(self):
        """Pre-render key surfaces for better performance."""
        self._cached_key_surfaces = {}
        for key_name in ["default", "hover", "press"]:
            color_map = {
                "default": self._face_default,
                "hover": self._face_hover,
                "press": self._face_press
            }
            self._cached_key_surfaces[key_name] = color_map[key_name]

    def _draw_key_with_effects(self, img: np.ndarray, x: int, y: int, w: int, h: int, 
                              face_color: np.ndarray, is_hovered: bool, is_pressed: bool, now: float):
        """Draw a single key with visual effects."""
        # Key press animation (shrink effect)
        if is_pressed:
            shrink = 2
            x += shrink
            y += shrink
            w -= shrink * 2
            h -= shrink * 2
        
        # Draw key face with rounded corners effect
        cv2.rectangle(img, (x, y), (x + w, y + h), 
                     tuple(int(c) for c in face_color), cv2.FILLED, cv2.LINE_AA)
        
        # Draw border with glow effect on hover
        border_color = self._border
        border_thickness = 1
        if is_hovered:
            border_color = (150, 200, 255)  # Glow effect
            border_thickness = 2
            # Add subtle glow
            glow_overlay = img[max(0, y-2):min(self.frame_h, y+h+2), 
                              max(0, x-2):min(self.frame_w, x+w+2)].astype(np.float32)
            glow_color = np.full_like(glow_overlay, (120, 200, 255))
            alpha = 0.3
            blended = glow_overlay * (1 - alpha) + glow_color * alpha
            img[max(0, y-2):min(self.frame_h, y+h+2), 
                max(0, x-2):min(self.frame_w, x+w+2)] = blended.astype(np.uint8)
        
        cv2.rectangle(img, (x, y), (x + w, y + h), border_color, border_thickness, cv2.LINE_AA)

    def _draw_key_preview(self, img: np.ndarray, key: str, finger_x: float, finger_y: float):
        """Draw a preview popup above the finger when hovering."""
        if key and self._last_hovered_key == key:
            hover_duration = time.time() - self._hover_start_time
            if hover_duration > 0.3:  # Show preview after 300ms of hover
                preview_size = 30
                preview_x = int(finger_x - preview_size // 2)
                preview_y = int(finger_y - preview_size - 20)
                
                # Ensure preview stays within bounds
                preview_x = max(5, min(preview_x, self.frame_w - preview_size - 5))
                preview_y = max(5, preview_y)
                
                # Draw preview background
                cv2.rectangle(img, (preview_x, preview_y), 
                            (preview_x + preview_size, preview_y + preview_size),
                            (50, 50, 60), cv2.FILLED, cv2.LINE_AA)
                cv2.rectangle(img, (preview_x, preview_y), 
                            (preview_x + preview_size, preview_y + preview_size),
                            (200, 200, 255), 2, cv2.LINE_AA)
                
                # Draw preview text
                label = self._label_for_draw(key)
                if len(label) > 3:
                    label = label[:3]  # Truncate long labels
                fs = 0.6
                tw = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)[0][0]
                tx = preview_x + (preview_size - tw) // 2
                ty = preview_y + int(preview_size * 0.7)
                cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 
                          fs, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_cooldown_indicator(self, img: np.ndarray, now: float):
        """Draw a visual indicator for typing cooldown status."""
        if now - self.last_press_time < self.press_cooldown:
            cooldown_progress = (now - self.last_press_time) / self.press_cooldown
            bar_width = 100
            bar_height = 6
            bar_x = self.frame_w - bar_width - 10
            bar_y = 10
            
            # Background
            cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                        (40, 40, 50), cv2.FILLED)
            
            # Progress
            progress_width = int(bar_width * cooldown_progress)
            cv2.rectangle(img, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height),
                        (0, 255, 100), cv2.FILLED)
            
            # Border
            cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                        (100, 100, 120), 1)

    def draw(self, img: np.ndarray, finger_x: float, finger_y: float, now: float) -> Optional[str]:
        # Apply temporal smoothing to finger position
        smoothed_x, smoothed_y = self._smooth_finger_position(finger_x, finger_y)
        
        # Check if finger moved significantly (for draw optimization)
        finger_moved = self._finger_moved_significantly(smoothed_x, smoothed_y)
        
        # Update performance metrics
        self._fps_frame_count += 1
        if now - self._last_fps_update >= 1.0:
            self._keyboard_fps = self._fps_frame_count / (now - self._last_fps_update)
            self._last_fps_update = now
            self._fps_frame_count = 0
            
            # Auto-adjust quality based on performance
            if self._keyboard_fps < 25.0 and not self._performance_mode:
                self._performance_mode = True
                self._bg_alpha = 0.8  # Less transparency = less blending work
                self._keyboard_fps_limit = 20.0  # Lower FPS limit
                self._min_draw_interval = 1.0 / self._keyboard_fps_limit
            elif self._keyboard_fps > 28.0 and self._performance_mode:
                self._performance_mode = False
                self._bg_alpha = 0.5  # Restore transparency
                self._keyboard_fps_limit = 30.0  # Restore FPS limit
                self._min_draw_interval = 1.0 / self._keyboard_fps_limit
        
        # FPS limiting for keyboard mode
        if now - self._last_draw_time < self._min_draw_interval:
            return self._confirmed_hovered_key
        
        self._last_draw_time = now
        
        # Get candidate key from smoothed position
        candidate_key = self.key_at(smoothed_x, smoothed_y)
        
        # Update stable hover state with debouncing
        hovered = self._update_stable_hover(candidate_key, now)
        
        # Only redraw if hover state changed or finger moved significantly
        if hovered != self._last_drawn_hovered_key or finger_moved:
            self._last_drawn_hovered_key = hovered
        
        # Draw keyboard background
        bounds = self._keyboard_roi_bounds()
        if bounds:
            x0, y0, x1, y1 = bounds
            roi = img[y0:y1, x0:x1].astype(np.float32)
            overlay = np.full_like(roi, (40.0, 42.0, 48.0))
            blended = roi * (1.0 - self._bg_alpha) + overlay * self._bg_alpha
            img[y0:y1, x0:x1] = blended.astype(np.uint8)

        # Draw keys with effects (skip some effects in performance mode)
        for item in self.key_rects:
            k, x, y, w, h = item
            face = self._face_default.copy()
            is_hovered = (k == hovered)
            is_pressed = (self._press_key == k and now < self._press_until)
            
            if is_hovered:
                face = self._face_hover.copy()
            if is_pressed:
                face = self._face_press.copy()
                
            if self._performance_mode:
                # Simplified drawing in performance mode
                cv2.rectangle(img, (x, y), (x + w, y + h), 
                             tuple(int(c) for c in face), cv2.FILLED)
                cv2.rectangle(img, (x, y), (x + w, y + h), self._border, 1)
            else:
                self._draw_key_with_effects(img, x, y, w, h, face, is_hovered, is_pressed, now)
            
            # Draw key label
            label = self._label_for_draw(k)
            fs = 0.48 if k in ("SPACE", "BACKSPACE", "DELETE", "ENTER") else 0.55
            tw = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)[0][0]
            tx = x + max(0, (w - tw) // 2)
            ty = y + int(h * 0.65)
            cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs,
                       self._text_color, 1, cv2.LINE_AA)

        # Only draw extra effects in normal mode
        if not self._performance_mode:
            # Draw key preview popup using smoothed position
            self._draw_key_preview(img, hovered, smoothed_x, smoothed_y)
        
        # Draw cooldown indicator
        self._draw_cooldown_indicator(img, now)

        # Draw typed preview
        preview_y = self.KEY_START_Y - 10
        if self._typed_preview:
            preview = self._typed_preview[-10:]
            pad_x = 8
            tw = cv2.getTextSize(preview, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0]
            bx1 = min(self.frame_w - 4, pad_x + tw + 16)
            cv2.rectangle(img, (pad_x, preview_y - 26), (bx1, preview_y + 4), 
                        (235, 235, 240), cv2.FILLED, cv2.LINE_AA)
            cv2.rectangle(img, (pad_x, preview_y - 26), (bx1, preview_y + 4), 
                        (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img, preview, (pad_x + 8, preview_y - 4), cv2.FONT_HERSHEY_SIMPLEX,
                       0.55, (20, 20, 20), 1, cv2.LINE_AA)
        
        # Draw FPS indicator in performance mode
        if self._performance_mode:
            cv2.putText(img, f"KB FPS: {int(self._keyboard_fps)}", (10, self.frame_h - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 100), 1)
        
        return hovered

    def press_key(self, key: Optional[str], now: float) -> bool:
        if not key:
            return False
        if now - self.last_press_time < self.press_cooldown:
            return False
        self.last_press_time = now
        self._press_key = key
        self._press_until = now + 0.12

        # Haptic feedback
        if self._enable_haptic and winsound:
            try:
                winsound.Beep(1000, 20)  # 1000Hz for 20ms
            except:
                pass  # Fail silently if sound doesn't work

        if key == "SPACE":
            pyautogui.press("space")
            self._typed_preview += " "
        elif key == "BACKSPACE":
            pyautogui.press("backspace")
            if self._typed_preview:
                self._typed_preview = self._typed_preview[:-1]
        elif key == "DELETE":
            pyautogui.press("delete")
        elif key == "ENTER":
            pyautogui.press("enter")
            self._typed_preview += " "
        else:
            ch = key.lower() if len(key) == 1 and key.isalpha() else key
            pyautogui.write(ch, interval=0)
            self._typed_preview += ch if len(ch) == 1 else ""
        self._typed_preview = self._typed_preview[-40:]
        return True


simple_kb = SimpleKeyboard(wCam, hCam)


def _drawing_palette_layout():
    """Return (start_x, top, sw_w, sh, gap, colors, preview_x)."""
    colors = DRAW_PALETTE_COLORS
    n = len(colors)
    sw, sh, gap = DRAW_PALETTE_SW, DRAW_PALETTE_SH, DRAW_PALETTE_GAP
    total_w = n * sw + (n - 1) * gap
    start_x = (wCam - total_w) // 2
    top = DRAW_PALETTE_TOP
    preview_x = start_x + total_w + DRAW_PALETTE_PREVIEW_GAP
    return start_x, top, sw, sh, gap, colors, preview_x


def _drawing_palette_sw_index(px, py):
    """Return swatch index if (px, py) lies inside the palette bar; px must be mirrored X (same as drawX)."""
    start_x, top, sw, sh, gap, colors, _ = _drawing_palette_layout()
    if py < top or py > top + sh:
        return None
    for i in range(len(colors)):
        sx = start_x + i * (sw + gap)
        if sx <= px <= sx + sw:
            return i
    return None


def _render_drawing_ui(img, thickness_display, now_t):
    """Palette, current-color preview, thickness — call after canvas blend so UI stays crisp."""
    start_x, top, sw, sh, gap, colors, preview_x = _drawing_palette_layout()
    bar_pad = 8
    bar_x0 = max(4, start_x - bar_pad)
    bar_y0 = 8
    bar_x1 = min(wCam - 4, preview_x + DRAW_PALETTE_PREVIEW_W + bar_pad)
    bar_y1 = top + sh + bar_pad + 52
    cv2.rectangle(img, (bar_x0, bar_y0), (bar_x1, bar_y1), (25, 25, 35), cv2.FILLED)
    cv2.rectangle(img, (bar_x0, bar_y0), (bar_x1, bar_y1), (90, 90, 110), 1)

    cv2.putText(img, "COLORS: touch a square with index tip, then pinch thumb+index to select", (bar_x0 + 4, bar_y0 + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 245, 255), 1)

    for i, c in enumerate(colors):
        sx = start_x + i * (sw + gap)
        cv2.rectangle(img, (sx, top), (sx + sw, top + sh), c, cv2.FILLED)
        border = (220, 220, 220)
        thick = 1
        if draw_palette_hover_idx == i:
            border = (80, 255, 120) if not draw_palette_wants_pinch else (0, 255, 255)
            thick = 2
        cv2.rectangle(img, (sx, top), (sx + sw, top + sh), border, thick)

    if drawEraserMode:
        preview_c = (55, 55, 62)
        cv2.rectangle(img, (preview_x, top), (preview_x + DRAW_PALETTE_PREVIEW_W, top + sh),
                      preview_c, cv2.FILLED)
        cv2.line(img, (preview_x + 4, top + 4), (preview_x + DRAW_PALETTE_PREVIEW_W - 4, top + sh - 4),
                 (200, 200, 210), 2)
        cv2.line(img, (preview_x + DRAW_PALETTE_PREVIEW_W - 4, top + 4), (preview_x + 4, top + sh - 4),
                 (200, 200, 210), 2)
    else:
        preview_c = penColor
        cv2.rectangle(img, (preview_x, top), (preview_x + DRAW_PALETTE_PREVIEW_W, top + sh),
                      preview_c, cv2.FILLED)
    pv_flash = now_t < palette_pick_flash_until
    preview_border = (0, 255, 128) if pv_flash else (255, 255, 255)
    cv2.rectangle(img, (preview_x, top), (preview_x + DRAW_PALETTE_PREVIEW_W, top + sh),
                  preview_border, 2 if pv_flash else 1)

    cv2.putText(img, "current", (preview_x, top + sh + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 210), 1)

    hint_y = top + sh + 36
    if draw_palette_hover_idx is not None and draw_palette_wants_pinch:
        cv2.putText(img, ">>> PINCH now (bring thumb to index) <<<", (start_x, hint_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2)
    elif draw_palette_hover_idx is not None:
        cv2.putText(img, "On color — pinch thumb + index to select", (start_x, hint_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 255, 200), 1)

    cv2.putText(img, f"Brush: {thickness_display}px", (start_x, hint_y + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if drawEraserMode:
        cv2.putText(img, "ERASER", (preview_x + DRAW_PALETTE_PREVIEW_W + 10, top + sh - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 230, 255), 2)


def _draw_drawing_fingertip_overlay(img):
    """High-contrast fingertip marker on top of blended feed (not multiplied by addWeighted)."""
    if not draw_overlay_tip_show:
        return
    x, y = int(draw_overlay_tip_x), int(draw_overlay_tip_y)
    fill = draw_overlay_tip_fill
    # Outer white ring, thin black ring, filled core — stays readable on any background.
    cv2.circle(img, (x, y), 16, (255, 255, 255), 2)
    cv2.circle(img, (x, y), 14, (0, 0, 0), 2)
    cv2.circle(img, (x, y), 11, fill, cv2.FILLED)
    cv2.circle(img, (x, y), 11, (255, 255, 255), 1)
    if draw_overlay_tip_hover:
        cv2.circle(img, (x, y), 24, (0, 255, 100), 2)


while True:
    success, img = cap.read()
    if not success:
        continue

    draw_overlay_tip_show = False

    # Disable heavy landmark drawing in hot path for lower latency.
    img = detector.findHands(img, draw=False)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) == 0 and mode == "drawing":
        draw_palette_hover_idx = None
        draw_palette_wants_pinch = False

    if mode != "keyboard":
        keyboard_pinch_prev = False

    if len(lmList) != 0:
        ix, iy = lmList[8][1], lmList[8][2]  # Index tip
        mx, my = lmList[12][1], lmList[12][2]  # Middle tip
        tx, ty = lmList[4][1], lmList[4][2]  # Thumb tip
        thumb_index_dist = math.hypot(tx - ix, ty - iy)
        fingers = detector.fingersUp()
        now = time.time()

        # Exit gesture: close all 5 fingers (fist) for a short hold.
        if fingers == [0, 0, 0, 0, 0]:
            if exitGestureStart is None:
                exitGestureStart = now
            holdLeft = max(0.0, exitHoldSeconds - (now - exitGestureStart))
            cv2.putText(img, f'Exit in {holdLeft:.1f}s', (20, 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if now - exitGestureStart >= exitHoldSeconds:
                break
        else:
            exitGestureStart = None

        # Mode switch gesture (hold): index + middle up -> [0,1,1,0,0]
        # Require thumb far from index so this does not fire during brightness/volume pinch.
        if fingers == [0, 1, 1, 0, 0] and thumb_index_dist >= switchMinThumbIndexDist:
            if switchGestureStart is None:
                switchGestureStart = now
            switchLeft = max(0.0, switchHoldSeconds - (now - switchGestureStart))
            cv2.putText(img, f'Switch mode in {switchLeft:.1f}s', (20, 145),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            if now - switchGestureStart >= switchHoldSeconds:
                mode = modes[(modes.index(mode) + 1) % len(modes)]
                switchGestureStart = None
                wasPinching = True  # debounce accidental click right after mode change
        else:
            switchGestureStart = None

        if drawOverlay:
            cv2.rectangle(img, (frameR, frameR), (wCam-frameR, hCam-frameR),
                          (255, 0, 255), 2)

        if mode == "cursor":
            x1 = min(max(ix, frameR), wCam - frameR)
            y1 = min(max(iy, frameR), hCam - frameR)
            x3 = (x1 - frameR) * wScr / (wCam - 2 * frameR)
            y3 = (y1 - frameR) * hScr / (hCam - 2 * frameR)

            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            # Cursor lock gesture: pinky-only up -> [0,0,0,0,1]
            if fingers == [0, 0, 0, 0, 1]:
                if not cursorToggleLatch:
                    cursorLocked = not cursorLocked
                    cursorToggleLatch = True
            else:
                cursorToggleLatch = False

            # Scroll gesture: thumb + index + middle up -> [1,1,1,0,0]
            scrollGesture = (fingers[0] == 1 and fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0)
            if scrollGesture and not cursorLocked:
                currentScrollY = (iy + my) / 2.0
                if lastScrollY is not None:
                    # Up movement => positive accumulator => scroll up
                    scrollAccumulator += (lastScrollY - currentScrollY)

                if (abs(scrollAccumulator) >= scrollThreshold) and ((now - lastScrollTime) > scrollCooldown):
                    steps = int(scrollAccumulator / scrollThreshold)
                    # Limit burst to keep behavior smooth.
                    steps = max(-3, min(3, steps))
                    if steps != 0:
                        pyautogui.scroll(steps * scrollAmount)
                        scrollAccumulator -= steps * scrollThreshold
                        lastScrollTime = now

                lastScrollY = currentScrollY
            else:
                lastScrollY = None
                scrollAccumulator = 0.0

            if fingers[1] == 1 and not scrollGesture and not cursorLocked:
                if abs(clocX - plocX) > moveDeadZone or abs(clocY - plocY) > moveDeadZone:
                    pyautogui.moveTo(wScr - clocX, clocY)

            plocX, plocY = clocX, clocY

            # Click gesture: all 5 fingers up (single click per gesture hold).
            if fingers == [1, 1, 1, 1, 1]:
                if not wasPinching and not cursorLocked:
                    pyautogui.click()
                    wasPinching = True
                    cv2.circle(img, (ix, iy), 12, (0, 255, 0), cv2.FILLED)
            else:
                wasPinching = False

            cv2.circle(img, (ix, iy), 8, (255, 0, 255), cv2.FILLED)
            lockText = "CURSOR: LOCKED" if cursorLocked else "CURSOR: LIVE"
            lockColor = (0, 255, 0) if cursorLocked else (0, 200, 255)
            cv2.putText(img, lockText, (20, 175),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, lockColor, 2)
        elif mode == "volume":
            # Volume mode: control master volume with thumb-index distance.
            lastScrollY = None
            scrollAccumulator = 0.0
            # 5 fingers up toggles lock/unlock for stable volume hold.
            if fingers == [1, 1, 1, 1, 1]:
                if not volumeToggleLatch:
                    volumeLocked = not volumeLocked
                    volumeToggleLatch = True
            else:
                volumeToggleLatch = False

            if not volumeLocked:
                length = math.hypot(tx - ix, ty - iy)
                volScalar = float(np.interp(length, [45, 200], [0.0, 1.0]))
                volScalar = max(0.0, min(1.0, volScalar))

                # Throttle expensive volume API calls to keep loop fast.
                if (abs(volScalar - lastVolScalar) > 0.01) and (now - lastVolUpdate > 0.03):
                    volume.SetMasterVolumeLevelScalar(volScalar, None)
                    lastVolScalar = volScalar
                    # Keep dB value synced for internal consistency if needed elsewhere.
                    lastVol = np.interp(lastVolScalar, [0.0, 1.0], [minVol, maxVol])
                    lastVolUpdate = now

            # Display from scalar to match Windows system volume percentage.
            volBar = int(np.interp(lastVolScalar, [0.0, 1.0], [400, 150]))
            volPer = int(lastVolScalar * 100)
            cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 2)
            cv2.rectangle(img, (50, volBar), (85, 400), (255, 0, 0), cv2.FILLED)
            cv2.putText(img, f'{volPer} %', (40, 440),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
            lockText = "VOL: LOCKED" if volumeLocked else "VOL: LIVE"
            lockColor = (0, 255, 0) if volumeLocked else (0, 200, 255)
            cv2.putText(img, lockText, (35, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, lockColor, 2)
            cv2.circle(img, (tx, ty), 8, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (ix, iy), 8, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (tx, ty), (ix, iy), (255, 0, 255), 2)
        elif mode == "brightness":
            # Brightness mode: control screen brightness with thumb-index distance.
            lastScrollY = None
            scrollAccumulator = 0.0

            # 5 fingers up toggles lock/unlock for stable brightness hold.
            if fingers == [1, 1, 1, 1, 1]:
                if not brightnessToggleLatch:
                    brightnessLocked = not brightnessLocked
                    brightnessToggleLatch = True
            else:
                brightnessToggleLatch = False

            if sbc is not None and not brightnessGiveUp and not brightnessLocked:
                length = thumb_index_dist
                brightnessTarget = int(np.interp(length, [40, 220], [0, 100]))
                brightnessTarget = max(0, min(100, brightnessTarget))

                if (abs(brightnessTarget - brightnessValue) > 0) and (now - lastBrightnessUpdate > 0.06):
                    if _set_system_brightness(brightnessTarget):
                        brightnessApplyFailures = 0
                        confirmed = _read_system_brightness()
                        brightnessValue = confirmed if confirmed is not None else brightnessTarget
                    else:
                        brightnessApplyFailures += 1
                        if brightnessApplyFailures >= 8:
                            brightnessGiveUp = True
                    lastBrightnessUpdate = now

            briBar = int(np.interp(brightnessValue, [0, 100], [400, 150]))
            cv2.rectangle(img, (110, 150), (145, 400), (0, 220, 255), 2)
            cv2.rectangle(img, (110, briBar), (145, 400), (0, 220, 255), cv2.FILLED)
            cv2.putText(img, f'{brightnessValue} %', (100, 440),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 220, 255), 2)
            lockText = "BRT: LOCKED" if brightnessLocked else "BRT: LIVE"
            lockColor = (0, 255, 0) if brightnessLocked else (0, 200, 255)
            cv2.putText(img, lockText, (95, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, lockColor, 2)
            if sbc is None:
                cv2.putText(img, "pip install screen-brightness-control", (180, 205),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
                pyexe = sys.executable
                if len(pyexe) > 52:
                    pyexe = "..." + pyexe[-49:]
                cv2.putText(img, f"for: {pyexe}", (180, 228),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1)
            elif brightnessGiveUp:
                cv2.putText(img, "Brightness API failed - external monitors often unsupported", (20, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 255), 2)
                cv2.putText(img, "cannot be dimmed; try the laptop screen", (20, 222),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 255), 2)

            cv2.circle(img, (tx, ty), 8, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (ix, iy), 8, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (tx, ty), (ix, iy), (255, 0, 255), 2)
        elif mode == "keyboard":
            # Keyboard mode: no OS cursor motion — raw index tip (ix, iy) drives overlay only.
            lastScrollY = None
            scrollAccumulator = 0.0

            active_key = simple_kb.draw(img, ix, iy, now)
            cv2.circle(img, (ix, iy), 8, (255, 0, 255), cv2.FILLED)

            pinch_down = thumb_index_dist < keyboardPinchThreshold and fingers[1] == 1
            
            # Use stable pinch detection to prevent accidental clicks
            should_click = simple_kb._should_trigger_pinch(pinch_down, now)
            
            if should_click and active_key:
                simple_kb.press_key(active_key, now)
        else:
            # Drawing mode: mirror X so ink tracks the fingertip as shown (same intuition as cursor wScr flip).
            lastScrollY = None
            scrollAccumulator = 0.0
            x1 = min(max(ix, frameR), wCam - frameR)
            y1 = min(max(iy, frameR), hCam - frameR)
            drawX = wCam - x1
            # Fingertip dot aligns with palette using mirrored X (drawX) and raw Y in the top band.
            pointer_y = iy if iy <= frameR else y1

            # Thumb–index distance sets brush size (pen 2–20px, eraser 15–35px).
            t_pen = int(np.clip(np.interp(thumb_index_dist, [30, 150], [2, 20]), 2, 20))
            t_eraser = int(np.clip(np.interp(thumb_index_dist, [30, 150], [15, 35]), 15, 35))
            drawBrushThicknessDisplay = t_eraser if drawEraserMode else t_pen

            # Eraser toggle: middle finger only up [thumb, index, middle, ring, pinky].
            eraser_gesture = fingers == [0, 0, 1, 0, 0]
            if eraser_gesture:
                if not eraserToggleLatch:
                    drawEraserMode = not drawEraserMode
                    eraserToggleLatch = True
            else:
                eraserToggleLatch = False

            # Palette hit uses same mirrored X as drawing so the dot sits on the swatch you mean.
            hover_idx = _drawing_palette_sw_index(drawX, iy)
            draw_palette_hover_idx = hover_idx
            draw_palette_wants_pinch = (
                hover_idx is not None and thumb_index_dist > DRAW_PALETTE_PINCH_THRESH
            )

            # Palette: looser pinch threshold than keyboard; index must be up.
            palette_pinch = (
                thumb_index_dist < DRAW_PALETTE_PINCH_THRESH
                and fingers[1] == 1
                and fingers != [1, 1, 1, 1, 1]
            )
            if palette_pinch:
                if hover_idx is not None and not palettePinchLatch:
                    penColor = DRAW_PALETTE_COLORS[hover_idx]
                    drawEraserMode = False
                    palettePinchLatch = True
                    palette_pick_flash_until = now + 0.35
            else:
                palettePinchLatch = False

            # Draw state gestures.
            drawGesture = (fingers[1] == 1 and fingers[2] == 0)
            hoverGesture = (fingers[1] == 1 and fingers[2] == 1)

            # Eraser removes ink by painting canvas black (0,0,0); blend layer uses zero ink there — not white paint.
            CANVAS_CLEAR = (0, 0, 0)
            stroke_color = CANVAS_CLEAR if drawEraserMode else penColor
            stroke_thick = t_eraser if drawEraserMode else t_pen
            stroke_y = pointer_y

            if drawGesture:
                if drawPrevX is None:
                    drawPrevX, drawPrevY = drawX, stroke_y
                cv2.line(
                    canvas,
                    (drawPrevX, drawPrevY),
                    (drawX, stroke_y),
                    stroke_color,
                    stroke_thick,
                    cv2.LINE_8,
                )
                drawPrevX, drawPrevY = drawX, stroke_y
            else:
                drawPrevX, drawPrevY = None, None

            # Clear canvas on 5-finger gesture (latching avoids repeated clears).
            if fingers == [1, 1, 1, 1, 1]:
                if not clearLatch:
                    canvas[:] = 0
                    clearLatch = True
            else:
                clearLatch = False

            # Fingertip marker drawn after addWeighted so it is not dimmed.
            draw_overlay_tip_show = True
            draw_overlay_tip_x = drawX
            draw_overlay_tip_y = pointer_y
            if drawEraserMode:
                draw_overlay_tip_fill = (0, 140, 255) if drawGesture else (120, 180, 255)
            elif drawGesture:
                draw_overlay_tip_fill = (0, 255, 255)
            else:
                draw_overlay_tip_fill = (255, 80, 255)
            draw_overlay_tip_hover = hoverGesture

    # Blend drawing canvas on top of camera frame.
    if mode == "drawing":
        img = cv2.addWeighted(img, 0.7, canvas, 1.0, 0.0)
        _render_drawing_ui(img, drawBrushThicknessDisplay, time.time())
        _draw_drawing_fingertip_overlay(img)

    cTime = time.time()
    fps = 1/(cTime - pTime) if (cTime - pTime) != 0 else 0
    pTime = cTime

    modeText = "CURSOR MODE [cursor]"
    if mode == "volume":
        modeText = "VOLUME MODE [speaker]"
    elif mode == "brightness":
        modeText = "BRIGHTNESS MODE [sun]"
    elif mode == "keyboard":
        modeText = "KEYBOARD MODE [typing]"
    elif mode == "drawing":
        modeText = "DRAWING MODE [fun]"
    cv2.putText(img, modeText, (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(img, "Scroll: thumb+index+middle | Switch: index+middle hold ~1.2s (thumb spread) | Exit: fist", (20, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 255, 200), 1)
    if mode == "cursor":
        cv2.putText(img, "Cursor lock: pinky only (toggle)", (20, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 255, 180), 1)
    elif mode == "volume":
        cv2.putText(img, "Volume lock: 5 fingers (toggle)", (20, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 255, 180), 1)
    elif mode == "brightness":
        cv2.putText(img, "Brightness: thumb-index distance | Lock: 5 fingers", (20, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 255, 180), 1)
    elif mode == "keyboard":
        cv2.putText(img, "Keyboard: pinch on key to type", (20, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 255, 180), 1)
    elif mode == "drawing":
        cv2.putText(img, "Draw: index | Hover: index+mid | Clear: 5 | Palette: pinch swatch | Eraser: mid only", (20, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 255, 180), 1)
    cv2.putText(img, f'FPS: {int(fps)}', (20, 50),
                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)

    cv2.imshow("Virtual Mouse", img)
    # Keep window visible/topmost; user can still minimize/maximize manually.
    cv2.setWindowProperty("Virtual Mouse", cv2.WND_PROP_TOPMOST, 1)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('m'):
        mode = modes[(modes.index(mode) + 1) % len(modes)]
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()