"""
detector.py
===========
Garbage-bin detector for the Skyscouter CV assessment.

Primary strategy: HSV colour segmentation (blue bin in grey warehouse)
----------------------------------------------------------------------
The bin is a distinctive bright blue against a neutral grey/white background.
This is the most stable signal in the scene:
  • Works from frame 0 — no background-model warmup required
  • Unaffected by motion blur (colour survives blur)
  • Unaffected by MOG2 absorption of static objects
  • Survives partial occlusion (blue pixels visible even when person adjacent)
  • Works when person carries the bin (bin body remains blue)

Secondary strategy: MOG2 background subtraction
------------------------------------------------
MOG2 detects motion-based foreground and confirms the colour blob is moving
or recently placed. OR'd with the colour mask so EITHER signal can trigger.

Person masking
--------------
YOLO detects people. Person pixels are erased from the combined mask before
contour finding, with zero padding on the side facing the last known bin
position so adjacent bin pixels are never accidentally removed.

Scoring
-------
Blobs that survive masking are scored by:
  1. Blueness score  — fraction of blob pixels that pass the colour filter
  2. Aspect score    — h/w ratio vs. known bin shape
  3. Proximity score — distance from last known position (for continuity)

Final score = 0.50 * blue_s + 0.30 * aspect_s + 0.20 * prox_s
"""

import cv2
import numpy as np
from ultralytics import YOLO

# COCO class ids
_PERSON_CLS = 0

# Bin shape prior (original assessment dimensions)
_BIN_ASPECT    = 0.65 / 0.40   # ≈ 1.625  (height / width)
_ASPECT_TOL    = 0.60           # ±60 % — generous; blob is often partial
_MIN_BLOB_AREA = 800            # px² — small threshold, colour filter does the work

# HSV colour range for the large blue industrial bin
# Tuned for a medium-bright blue in indoor warehouse lighting
# H in OpenCV is 0-180; blue ≈ 100-130
_HSV_LOWER = np.array([95,  80,  60],  dtype=np.uint8)
_HSV_UPPER = np.array([135, 255, 230], dtype=np.uint8)

# Minimum fraction of blob pixels that must be blue for it to score well
_MIN_BLUE_FRAC = 0.04   # fraction OR absolute count — see scoring loop


class BinDetector:
    """
    Colour + motion bin detector.

    Parameters
    ----------
    use_gpu : bool
        If True YOLO runs on CUDA.
    yolo_weights : str
        YOLOv8 weights — used ONLY for person detection.
    conf_thresh : float
        Minimum YOLO person-detection confidence.
    """

    def __init__(self, use_gpu=False, yolo_weights="yolov8n.pt", conf_thresh=0.30):
        self._device = "cuda" if use_gpu else "cpu"
        self.model   = YOLO(yolo_weights)
        self.model.to(self._device)

        # MOG2 — secondary motion channel
        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=40, detectShadows=True
        )
        self._k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self._k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

        self._conf_thresh = conf_thresh
        self._last_box    = None
        self._frame_count = 0

        # Colour mask from previous frame (for smoothing)
        self._prev_colour_mask = None

        # ── Debug hooks consumed by track_bin.py ──────────────────
        # track_bin.py reads these four attributes after each detect()
        # call.  This detector does not produce a histogram-refined bbox
        # or per-channel stats, so last_hist_mask / last_hist_diag /
        # last_vground_hist stay None.  last_vground_hsv is set to the
        # v_ground returned by detect() on success so the diagnostic
        # print line ("vgnd=hsv:...") has a value to show.
        self.last_hist_mask    = None
        self.last_hist_diag    = None
        self.last_vground_hist = None
        self.last_vground_hsv  = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> tuple | None:
        self._frame_count += 1

        # 1. Person detection first (needed for masking)
        yolo_res     = self._run_yolo(frame)
        person_boxes = self._get_person_boxes(yolo_res)

        # 2. Combined foreground mask (colour OR motion)
        mask = self._combined_mask(frame, person_boxes)

        # 3. Find candidate blobs from COLOUR MASK ONLY
        # The combined (colour OR MOG2) mask creates blobs that extend
        # far below the bin because MOG2 foreground includes floor/legs,
        # pushing the centroid too low and causing depth underestimation.
        # Using only the colour mask keeps blobs tight around the actual
        # blue bin body + reflection.  MOG2 is used as a score boost below.
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        blue_mask_full = cv2.inRange(hsv, _HSV_LOWER, _HSV_UPPER)

        # Morphological clean-up of colour mask
        blue_clean = cv2.morphologyEx(blue_mask_full, cv2.MORPH_OPEN,
                                       self._k_open, iterations=2)
        blue_clean = cv2.morphologyEx(blue_clean, cv2.MORPH_CLOSE,
                                       self._k_close, iterations=3)

        cnts, _ = cv2.findContours(blue_clean, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        best_score, best_box = 0.0, None

        # Vertical band: bin base must be in this row range
        V_MIN, V_MAX = 250, 920

        for c in cnts:
            if cv2.contourArea(c) < _MIN_BLOB_AREA:
                continue
            x, y, w, h = cv2.boundingRect(c)
            x1, y1, x2, y2 = x, y, x + w, y + h

            if not (V_MIN <= y2 <= V_MAX):
                continue

            # ── Blueness score ──────────────────────────────────────
            roi = blue_mask_full[y1:y2, x1:x2]
            blue_frac = float(roi.sum() / 255) / max(w * h, 1)
            blue_s = min(blue_frac / 0.30, 1.0)  # saturates at 30% blue

            # Reject blobs with too few blue pixels.
            # Use EITHER fraction OR absolute count so that large MOG2
            # blobs (which dilute the fraction) still pass if they contain
            # enough actual blue pixels (>= 800px² absolute).
            blue_px_abs = float(roi.sum() / 255)
            if blue_frac < _MIN_BLUE_FRAC and blue_px_abs < 800:
                continue

            # ── Aspect score ────────────────────────────────────────
            aspect_s = self._aspect_score(x1, y1, x2, y2)

            # ── Proximity score ─────────────────────────────────────
            prox_s = self._proximity_score(x1, y1, x2, y2)

            # ── MOG2 motion score (bonus) ────────────────────────────
            # Check overlap between colour blob and MOG2 foreground
            mog2_roi = mask[y1:y2, x1:x2]
            mog2_s   = float(mog2_roi.sum() / 255) / max(w * h, 1)
            mog2_s   = min(mog2_s / 0.3, 1.0)

            score = 0.45 * blue_s + 0.25 * aspect_s + 0.15 * prox_s + 0.15 * mog2_s

            if score > best_score:
                best_score = score
                best_box   = (x1, y1, x2, y2)

        # ── Occlusion recovery ───────────────────────────────────────
        if best_box is None and self._last_box is not None:
            best_box   = self._colour_recovery(blue_mask_full)
            best_score = 0.20 if best_box is not None else 0.0
            if best_box is not None:
                # Recovery: no gap analysis possible, use lower third
                x1,y1,x2,y2 = best_box
                v_ground = y1 + (y2-y1)*2//3
                self._last_box = best_box
                self.last_vground_hsv = int(v_ground)
                self._populate_debug(frame, best_box, blue_clean)
                return (x1, y1, x2, y2, round(best_score, 3), int(v_ground))

        if best_box is not None:
            self._last_box = best_box
            x1, y1, x2, y2 = best_box

            # ── Find ground contact row (v_ground) ───────────────────
            # Instead of "topmost blue row" (sensitive to 1-pixel flicker
            # at the top of the blob), we use a peak-fraction threshold:
            # a row qualifies as "bin edge" only if its blue-pixel count
            # is at least 30% of the densest row's count.  This rejects
            # stray pixels above/below the bin body + reflection.
            #
            #   row_counts[i] = blue pixels in row (y1+i)
            #   peak          = max(row_counts)
            #   stable_rows   = rows where row_counts >= 0.30 * peak
            #   y1_b = first stable row  (tight bin top)
            #   y2_b = last  stable row  (tight reflection bottom)
            roi_full    = blue_mask_full[y1:y2+1, max(x1,0):x2+1]
            row_counts  = (roi_full > 0).sum(axis=1)

            if row_counts.size > 0 and row_counts.max() >= 10:
                peak        = int(row_counts.max())
                stable_th   = max(int(0.30 * peak), 5)
                stable_rows = np.where(row_counts >= stable_th)[0]
                if len(stable_rows) >= 5:
                    y1_b = y1 + int(stable_rows[0])   # tight top
                    y2_b = y1 + int(stable_rows[-1])  # tight bottom
                else:
                    # Fallback: wider threshold
                    rows_blue = np.where(row_counts > 3)[0]
                    y1_b = y1 + int(rows_blue[0])  if len(rows_blue) else y1
                    y2_b = y1 + int(rows_blue[-1]) if len(rows_blue) else y2
            else:
                y1_b, y2_b = y1, y2   # nothing usable — keep raw bbox

            # ── Mirror-midpoint ground contact ──────────────────────
            # The polished floor creates a mirror reflection of the bin.
            # The tight blue bounding box spans:
            #   y1_b ≈ blue body top (below dark lid)
            #   y2_b ≈ bottom of the rim's floor reflection
            # Midpoint = ground contact pixel → cast to z_plane=-0.41m
            # (calibrated for this scene's dark-lid + rim-reflection asymmetry)
            v_ground = (y1_b + y2_b) // 2
            v_ground = int(np.clip(v_ground, y1_b, y2_b))

            # Return the TIGHTENED bbox (y1_b, y2_b) instead of the raw
            # contour bbox.  The localizer uses `h_px = v_ground - y1`,
            # so a cleaner y1 directly improves depth stability.
            best_box = (x1, y1_b, x2, y2_b)

            self.last_vground_hsv = int(v_ground)
            self._populate_debug(frame, best_box, blue_clean)
            self._last_box = best_box

            return (x1, y1_b, x2, y2_b, round(best_score, 3), v_ground)

        return None

    # ------------------------------------------------------------------
    # Debug overlays for track_bin.py (purely visualisation, no logic change)
    # ------------------------------------------------------------------

    def _populate_debug(self, frame: np.ndarray, bbox: tuple,
                         blue_mask: np.ndarray) -> None:
        """
        Set `last_hist_mask` and `last_hist_diag` for the annotated output.

        last_hist_mask : 2-D uint8 (0/255) — the cleaned blue mask cropped
                         to the bbox.  _draw_detection paints cyan on the
                         255 pixels and red on the 0 pixels, producing the
                         "kept vs background" segmentation overlay.

        last_hist_diag : list of per-channel dicts carrying the BGR median
                         ("peak") and σ ("std") of the bbox crop, used by
                         _draw_detection in the bbox label.

        This function does not touch detection, scoring, or the return
        tuple — it only writes to the two debug attributes.
        """
        x1, y1, x2, y2 = bbox
        H, W = frame.shape[:2]
        fx1, fy1 = max(x1, 0), max(y1, 0)
        fx2, fy2 = min(x2, W), min(y2, H)
        if fx2 <= fx1 or fy2 <= fy1:
            self.last_hist_mask = None
            self.last_hist_diag = None
            return

        # Blue mask crop — what the HSV classifier kept inside the bbox.
        self.last_hist_mask = blue_mask[fy1:fy2, fx1:fx2].copy()

        # Per-channel BGR stats over the raw crop.
        crop = frame[fy1:fy2, fx1:fx2]
        pixels = crop.reshape(-1, 3).astype(np.float32)
        if pixels.size == 0:
            self.last_hist_diag = None
            return

        median = np.median(pixels, axis=0)
        std    = np.std(pixels, axis=0)
        self.last_hist_diag = [
            {"ch": "B", "peak": float(median[0]), "std": float(std[0])},
            {"ch": "G", "peak": float(median[1]), "std": float(std[1])},
            {"ch": "R", "peak": float(median[2]), "std": float(std[2])},
        ]

    # ------------------------------------------------------------------
    # Combined foreground mask
    # ------------------------------------------------------------------

    def _combined_mask(self, frame: np.ndarray,
                        person_boxes: list[tuple]) -> np.ndarray:
        """
        Colour mask OR MOG2 mask, with person regions erased.

        Using OR means a blob is detected if EITHER:
          (a) it is the right blue colour, OR
          (b) it is moving/new foreground

        This covers:
          - Static bin (MOG2 absorbed): found by colour
          - Fast-moving bin (motion blur reduces colour fidelity): found by MOG2
          - Early frames (MOG2 not warmed up): found by colour
          - Person carrying bin: colour finds the blue bin body
        """
        # Colour channel
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        colour_mask = cv2.inRange(hsv, _HSV_LOWER, _HSV_UPPER)

        # Dilate colour mask slightly to fill small gaps in the bin body
        colour_mask = cv2.morphologyEx(colour_mask, cv2.MORPH_CLOSE,
                                        self._k_close, iterations=2)

        # MOG2 channel
        lr   = 0.003 if self._frame_count < 30 else -1
        fg   = self._bg.apply(frame, learningRate=lr)
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        fg   = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  self._k_open,  iterations=2)
        fg   = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self._k_close, iterations=2)

        # Combined
        mask = cv2.bitwise_or(colour_mask, fg)

        # Erase person regions (direction-aware padding)
        self._erase_persons(mask, person_boxes, frame.shape)

        # Final clean-up
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self._k_open,  iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._k_close, iterations=2)
        return mask

    def _erase_persons(self, mask: np.ndarray,
                        person_boxes: list[tuple], shape: tuple):
        """Erase person pixels; use zero pad on the side facing the bin."""
        h_frame, w_frame = shape[:2]
        PAD_V, PAD_H = 4, 6

        last_cx = ((self._last_box[0] + self._last_box[2]) / 2.0
                   if self._last_box is not None else None)

        for (px1, py1, px2, py2) in person_boxes:
            # If person overlaps last known bin heavily → skip (carrying)
            if self._last_box is not None:
                bx1, by1, bx2, by2 = self._last_box
                ix = max(0, min(px2, bx2) - max(px1, bx1))
                iy = max(0, min(py2, by2) - max(py1, by1))
                if ix * iy / max((bx2-bx1)*(by2-by1), 1) > 0.40:
                    continue   # person is on top of bin → let colour find it

            pcx = (px1 + px2) / 2.0
            if last_cx is not None and last_cx < pcx:
                lp, rp = 0, PAD_H
            elif last_cx is not None and last_cx > pcx:
                lp, rp = PAD_H, 0
            else:
                lp, rp = PAD_H, PAD_H

            ex1 = max(px1 - lp,    0)
            ey1 = max(py1 - PAD_V, 0)
            ex2 = min(px2 + rp,    w_frame - 1)
            ey2 = min(py2 + PAD_V, h_frame - 1)
            mask[ey1:ey2, ex1:ex2] = 0

    # ------------------------------------------------------------------
    # Colour-only recovery (when no blob passes all filters)
    # ------------------------------------------------------------------

    def _colour_recovery(self, blue_mask: np.ndarray) -> tuple | None:
        """
        When normal scoring finds nothing, find the largest blue blob
        near the last known position.
        """
        if self._last_box is None:
            return None

        lx1, ly1, lx2, ly2 = self._last_box
        lcx = (lx1 + lx2) / 2.0
        lcy = (ly1 + ly2) / 2.0

        cnts, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        best, best_d = None, float("inf")
        for c in cnts:
            if cv2.contourArea(c) < 400:
                continue
            x, y, w, h = cv2.boundingRect(c)
            cx = x + w / 2.0
            cy = y + h / 2.0
            d  = np.hypot(cx - lcx, cy - lcy)
            if d < best_d:
                best_d = d
                best   = (x, y, x + w, y + h)

        # Only accept if reasonably close
        diag = np.hypot(lx2 - lx1, ly2 - ly1)
        if best is not None and best_d < 3.0 * diag:
            return best
        return None

    # ------------------------------------------------------------------
    # YOLO (person detection only)
    # ------------------------------------------------------------------

    def _run_yolo(self, frame: np.ndarray):
        return self.model(frame, verbose=False,
                          conf=self._conf_thresh, imgsz=640)

    def _get_person_boxes(self, yolo_res) -> list[tuple]:
        boxes = []
        for r in yolo_res:
            if r.boxes is None:
                continue
            for det in r.boxes:
                if int(det.cls[0]) == _PERSON_CLS:
                    x1, y1, x2, y2 = [int(v) for v in det.xyxy[0].tolist()]
                    boxes.append((x1, y1, x2, y2))
        return boxes

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def _aspect_score(self, x1, y1, x2, y2) -> float:
        w = max(x2 - x1, 1)
        h = max(y2 - y1, 1)
        diff = abs(h / w - _BIN_ASPECT) / _BIN_ASPECT
        return float(max(0.0, 1.0 - diff / _ASPECT_TOL))

    def _proximity_score(self, x1, y1, x2, y2) -> float:
        if self._last_box is None:
            return 0.5   # neutral when no history
        lx1, ly1, lx2, ly2 = self._last_box
        lcx = (lx1 + lx2) / 2.0
        lcy = (ly1 + ly2) / 2.0
        bcx = (x1 + x2) / 2.0
        bcy = (y1 + y2) / 2.0
        diag = max(np.hypot(lx2 - lx1, ly2 - ly1), 1.0)
        dist = np.hypot(bcx - lcx, bcy - lcy)
        return float(max(0.0, 1.0 - dist / (2.5 * diag)))