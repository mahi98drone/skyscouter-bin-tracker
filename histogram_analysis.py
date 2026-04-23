"""
histogram_analysis.py
=====================
Standalone tool to visualise the v2 histogram semi-segmentation on
individual frames from the video.

It processes every N-th frame and saves a 4-panel debug image for each:
  Panel 1: full frame with bounding box
  Panel 2: raw BGR crop (original resolution)
  Panel 3: crop colourised by the histogram mask (cyan = keep, red = prune)
  Panel 4: the binary mask itself

It also prints per-channel histogram diagnostics (peak, std, lo, hi,
coverage) for each detected frame so you can compare:
  - Frames where the bbox fits tightly
  - Frames where the crop contains background (bin far away / partially visible)
  - Near vs. far bin

Usage
-----
    python histogram_analysis.py --video input.mp4 --calib calib.json
    python histogram_analysis.py --video input.mp4 --calib calib.json \
        --every 30 --out-dir hist_debug --n-sigma 2.0

Outputs
-------
  <out_dir>/frame_XXXX.png   — 4-panel debug images
  <out_dir>/hist_stats.csv   — per-frame channel statistics
"""

import argparse
import csv
import json
import os

import cv2
import numpy as np

from detector import BinDetector


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_panel(crop: np.ndarray, mask: np.ndarray,
                diag: list[dict], frame_id: int,
                full_frame: np.ndarray,
                x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """
    Build a 4-panel debug image:
      [full frame + bbox]  |  [raw crop]
      [mask overlay]       |  [binary mask]
    All panels are resized to the same height for easy comparison.
    """
    PANEL_H = 320   # target height for each panel

    def _fit(img, h=PANEL_H):
        """Resize img so height == h, preserving aspect ratio."""
        ih, iw = img.shape[:2]
        nw = max(1, int(iw * h / max(ih, 1)))
        return cv2.resize(img, (nw, h))

    # ── Panel 1: full frame with bounding box ─────────────────────────
    p1 = full_frame.copy()
    cv2.rectangle(p1, (x1, y1), (x2, y2), (0, 255, 255), 3)
    cv2.putText(p1, f"frame {frame_id}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    p1 = _fit(p1)

    # ── Panel 2: raw crop ─────────────────────────────────────────────
    p2 = _fit(crop.copy())
    cv2.putText(p2, "raw crop", (4, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # ── Panel 3: mask overlay (cyan=keep, red=prune) ──────────────────
    p3_src  = crop.copy()
    mask_rs = cv2.resize(mask, (crop.shape[1], crop.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    keep  = (mask_rs == 255)
    prune = (mask_rs == 0)
    p3_src[keep]  = (p3_src[keep].astype(np.float32) * 0.5
                     + np.array([180, 220, 0],   np.float32) * 0.5).astype(np.uint8)
    p3_src[prune] = (p3_src[prune].astype(np.float32) * 0.5
                     + np.array([0, 0, 200], np.float32) * 0.5).astype(np.uint8)
    p3 = _fit(p3_src)

    # Annotate with diag text
    for i, d in enumerate(diag):
        txt = (f"{d['ch']} pk={d['peak']:.0f} "
               f"±{d['std']:.0f} "
               f"[{d['lo']:.0f},{d['hi']:.0f}] "
               f"cov={d['coverage']:.2f}")
        cv2.putText(p3, txt, (4, 22 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

    # ── Panel 4: binary mask (greyscale → BGR) ────────────────────────
    mask_vis = cv2.cvtColor(mask_rs, cv2.COLOR_GRAY2BGR)
    p4 = _fit(mask_vis)
    cv2.putText(p4, "hist mask", (4, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 255, 180), 1)

    # ── Stack top row and bottom row ──────────────────────────────────
    # Pad all panels to the same width before stacking
    max_w = max(p1.shape[1], p2.shape[1], p3.shape[1], p4.shape[1])

    def _pad(img, w):
        pad = w - img.shape[1]
        if pad > 0:
            img = np.concatenate(
                [img, np.zeros((img.shape[0], pad, 3), dtype=np.uint8)], axis=1
            )
        return img

    top = np.concatenate([_pad(p1, max_w), _pad(p2, max_w)], axis=1)
    bot = np.concatenate([_pad(p3, max_w), _pad(p4, max_w)], axis=1)
    return np.concatenate([top, bot], axis=0)


def _draw_histograms(crop: np.ndarray, diag: list[dict]) -> np.ndarray:
    """
    Draw per-channel histograms with peak and ±2σ markers.
    Returns a BGR image (256 px wide per channel, stacked side by side).
    """
    HIST_H = 150
    HIST_W = 256
    MARGIN = 10
    colors = {"B": (200, 80, 30), "G": (30, 180, 30), "R": (30, 80, 200)}
    ch_idx = {"B": 0, "G": 1, "R": 2}

    panels = []
    for d in diag:
        ch  = d["ch"]
        img = np.zeros((HIST_H + MARGIN * 2, HIST_W + MARGIN * 2, 3), dtype=np.uint8)
        plane = crop[:, :, ch_idx[ch]].astype(np.float32).ravel()
        hist, _ = np.histogram(plane, bins=256, range=(0, 256))
        hist_norm = (hist / (hist.max() + 1e-6) * HIST_H).astype(int)

        col = colors[ch]
        for x, h in enumerate(hist_norm):
            cv2.line(img,
                     (x + MARGIN, HIST_H + MARGIN),
                     (x + MARGIN, HIST_H + MARGIN - h),
                     col, 1)

        # Peak line
        pk = int(d["peak"])
        cv2.line(img, (pk + MARGIN, MARGIN),
                 (pk + MARGIN, HIST_H + MARGIN), (255, 255, 255), 2)
        # ±std lines
        lo = max(0, int(d["lo"]))
        hi = min(255, int(d["hi"]))
        cv2.line(img, (lo + MARGIN, MARGIN),
                 (lo + MARGIN, HIST_H + MARGIN), (100, 100, 255), 1)
        cv2.line(img, (hi + MARGIN, MARGIN),
                 (hi + MARGIN, HIST_H + MARGIN), (100, 100, 255), 1)

        cv2.putText(img, f"ch={ch}  pk={pk}  std={d['std']:.0f}",
                    (2, MARGIN - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (200, 200, 200), 1)
        panels.append(img)

    return np.concatenate(panels, axis=1)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Histogram segmentation analyser")
    ap.add_argument("--video",   required=True)
    ap.add_argument("--calib",   required=True)
    ap.add_argument("--every",   type=int,   default=15,
                    help="Process every N-th frame (default 15)")
    ap.add_argument("--out-dir", default="hist_debug")
    ap.add_argument("--n-sigma", type=float, default=2.0,
                    help="Std-dev multiplier for histogram window (default 2.0)")
    ap.add_argument("--gpu",     action="store_true")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="Stop after this many total frames (for quick testing)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    detector = BinDetector(use_gpu=args.gpu, hist_n_sigma=args.n_sigma)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {args.video}")

    csv_path = os.path.join(args.out_dir, "hist_stats.csv")
    csv_file = open(csv_path, "w", newline="")
    writer   = csv.writer(csv_file)
    writer.writerow(["frame", "x1", "y1", "x2", "y2", "conf", "vgnd_hist", "vgnd_hsv", "vgnd_delta_px",
                     "B_peak", "B_std", "B_lo", "B_hi", "B_cov",
                     "G_peak", "G_std", "G_lo", "G_hi", "G_cov",
                     "R_peak", "R_std", "R_lo", "R_hi", "R_cov",
                     "mask_fill"])

    frame_id   = 0
    saved      = 0

    print(f"Analysing '{args.video}' — saving debug images to '{args.out_dir}/'")
    print(f"n_sigma={args.n_sigma}  every={args.every} frames")
    print("-" * 70)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if args.max_frames is not None and frame_id >= args.max_frames:
            break

        det = detector.detect(frame)

        if det is not None and frame_id % args.every == 0:
            x1, y1, x2, y2, conf, _ = det
            mask = detector.last_hist_mask
            diag = detector.last_hist_diag
            crop = detector.last_crop

            if mask is None or crop is None:
                frame_id += 1
                continue

            # ── Console output ────────────────────────────────────────
            mask_fill  = float((mask == 255).sum()) / max(mask.size, 1)
            vg_hist    = detector.last_vground_hist
            vg_hsv     = detector.last_vground_hsv
            vg_delta   = (vg_hist - vg_hsv) if (vg_hist and vg_hsv) else None
            vg_str     = (f"vgnd hist={vg_hist} hsv={vg_hsv} delta={vg_delta:+d}px"
                          if vg_delta is not None else f"vgnd hsv={vg_hsv} (mask fallback)")
            print(f"[frame {frame_id:04d}]  bbox=({x1},{y1},{x2},{y2})  "
                  f"conf={conf:.2f}  mask_fill={mask_fill:.2%}  {vg_str}")
            for d in diag:
                print(f"    {d['ch']}:  peak={d['peak']:.1f}  "
                      f"std={d['std']:.1f}  "
                      f"window=[{d['lo']:.1f},{d['hi']:.1f}]  "
                      f"coverage={d['coverage']:.1%}")

            # ── CSV row ───────────────────────────────────────────────
            vg_hist = detector.last_vground_hist
            vg_hsv  = detector.last_vground_hsv
            vg_delta = (vg_hist - vg_hsv) if (vg_hist is not None and vg_hsv is not None) else None
            row = [frame_id, x1, y1, x2, y2, f"{conf:.3f}",
                   vg_hist or "", vg_hsv or "", vg_delta if vg_delta is not None else ""]
            for d in diag:
                row += [d["peak"], d["std"], d["lo"], d["hi"], d["coverage"]]
            row.append(f"{mask_fill:.4f}")
            writer.writerow(row)
            csv_file.flush()

            # ── Debug image (4-panel + histogram strip) ───────────────
            panels = _make_panel(crop, mask, diag, frame_id,
                                  frame, x1, y1, x2, y2)
            hists  = _draw_histograms(crop, diag)

            # Pad histogram strip to match panels width
            pw = panels.shape[1]
            hw = hists.shape[1]
            if hw < pw:
                hists = np.concatenate(
                    [hists,
                     np.zeros((hists.shape[0], pw - hw, 3), dtype=np.uint8)],
                    axis=1
                )
            else:
                hists = hists[:, :pw]

            full_debug = np.concatenate([panels, hists], axis=0)
            out_path = os.path.join(args.out_dir, f"frame_{frame_id:04d}.png")
            cv2.imwrite(out_path, full_debug)
            saved += 1

        frame_id += 1

    cap.release()
    csv_file.close()
    print(f"\nSaved {saved} debug images → {args.out_dir}/")
    print(f"Stats CSV → {csv_path}")


if __name__ == "__main__":
    main()