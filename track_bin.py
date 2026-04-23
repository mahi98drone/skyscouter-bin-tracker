"""
track_bin.py
============
Skyscouter CV Assessment — main tracking pipeline.

Usage
-----
    python track_bin.py --video input.mp4 --calib calib.json
    python track_bin.py --video input.mp4 --calib calib.json --kalman
    python track_bin.py --video input.mp4 --calib calib.json --kalman --out-video results/out.mp4

Outputs
-------
  results/output.csv       — per-frame world coordinates  (task 3b)
  results/output.mp4       — annotated video with bounding box + HUD overlay
  trajectory.png           — top-down XY trajectory + waypoint overlays (bonus 2d)
  stdout stream            — real-time position line every frame (task 3a)
"""

import argparse
import json
import os
import time

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from detector      import BinDetector
from localizer     import Localizer, BIN_HEIGHT_M, BIN_DIAMETER_M
from kalman_filter import PositionKalman


def reconstruct_bbox(xyz_cam: np.ndarray, det_box: tuple, K: np.ndarray,
                     localizer=None, xyz_world: np.ndarray = None) -> tuple:
    """
    Reconstruct the correct bounding box by projecting the 8 corners of
    the bin's 3D bounding volume from world space into the image.

    Assessment-specified dimensions
    --------------------------------
    BIN_DIAMETER_M = 0.40 m  (front-facing width)
    BIN_HEIGHT_M   = 0.65 m  (ground to rim)

    Bin volume in world frame (+X forward, +Y left, +Z up)
    -------------------------------------------------------
    The bin base centre is at (Xw, Yw, 0).
    The bin occupies:
      X: [Xw - BIN_DIAMETER_M/2,  Xw + BIN_DIAMETER_M/2]  (depth extent)
      Y: [Yw - BIN_DIAMETER_M/2,  Yw + BIN_DIAMETER_M/2]  (lateral extent)
      Z: [0,  BIN_HEIGHT_M]                                 (vertical)

    All 8 corners are projected through the full camera model:
      P_cam = R^T @ (P_world - t)
      u = fx * X_cam / Z_cam + cx
      v = fy * Y_cam / Z_cam + cy

    The bounding box is the min/max envelope of all projected corners.
    This correctly handles perspective foreshortening — the base appears
    wider and lower than the rim at close range.
    """
    fx = K[0, 0];  fy = K[1, 1]
    cx = K[0, 2];  cy = K[1, 2]

    # Fallback: single-Z projection when no world position available
    if localizer is None or xyz_world is None:
        X, Y, Z = float(xyz_cam[0]), float(xyz_cam[1]), max(float(xyz_cam[2]), 0.5)
        uc   = fx * X / Z + cx
        vc   = fy * Y / Z + cy
        h_px = fy * BIN_HEIGHT_M   / Z
        w_px = fx * BIN_DIAMETER_M / Z
        return (int(uc - w_px/2), int(vc - h_px/2),
                int(uc + w_px/2), int(vc + h_px/2))

    Xw = float(xyz_world[0])
    Yw = float(xyz_world[1])
    hd = BIN_DIAMETER_M / 2.0   # half the footprint dimension

    # 8 corners of the bin's axis-aligned bounding box in world space
    world_corners = [
        np.array([Xw - hd, Yw - hd, 0.0          ]),  # base corners (4)
        np.array([Xw - hd, Yw + hd, 0.0          ]),
        np.array([Xw + hd, Yw - hd, 0.0          ]),
        np.array([Xw + hd, Yw + hd, 0.0          ]),
        np.array([Xw - hd, Yw - hd, BIN_HEIGHT_M ]),  # rim corners (4)
        np.array([Xw - hd, Yw + hd, BIN_HEIGHT_M ]),
        np.array([Xw + hd, Yw - hd, BIN_HEIGHT_M ]),
        np.array([Xw + hd, Yw + hd, BIN_HEIGHT_M ]),
    ]

    R_t = localizer.R.T
    t   = localizer.t
    us, vs = [], []
    for pw in world_corners:
        pc = R_t @ (pw - t)
        if pc[2] < 0.1:          # behind the camera — skip
            continue
        us.append(fx * pc[0] / pc[2] + cx)
        vs.append(fy * pc[1] / pc[2] + cy)

    if len(us) < 2:
        # All corners behind camera — single-Z fallback
        X, Y, Z = float(xyz_cam[0]), float(xyz_cam[1]), max(float(xyz_cam[2]), 0.5)
        uc = fx*X/Z+cx; vc = fy*Y/Z+cy
        h_px = fy*BIN_HEIGHT_M/Z; w_px = fx*BIN_DIAMETER_M/Z
        return (int(uc-w_px/2), int(vc-h_px/2), int(uc+w_px/2), int(vc+h_px/2))

    return (int(min(us)), int(min(vs)), int(max(us)), int(max(vs)))


# ── Calibration ───────────────────────────────────────────────────────────────

def load_calib(path: str):
    with open(path) as f:
        c = json.load(f)
    K        = np.array(c["K"],           dtype=np.float64)
    D        = np.array(c["dist_coeffs"], dtype=np.float64)
    cam_h    = float(c["camera_height_m"])
    tilt_rad = float(np.deg2rad(c["camera_tilt_deg"]))
    return K, D, cam_h, tilt_rad


# ── Video annotation helpers ──────────────────────────────────────────────────

def _draw_detection(frame, x1, y1, x2, y2, xyz_world, conf, dt_ms, frame_id,
                     hist_mask=None, hist_diag=None):
    """Draw bounding box, crosshair, histogram mask overlay and HUD (in-place)."""
    xw, yw, zw = xyz_world

    # ── v2 Histogram mask overlay ─────────────────────────────────────────
    # Colour the KEPT pixels (mask=255) with a cyan tint, and the PRUNED
    # pixels (mask=0, i.e. identified as background) with a red tint.
    # Both overlays are semi-transparent so the original image shows through.
    if hist_mask is not None:
        fy1 = max(y1, 0);  fy2 = min(y2, frame.shape[0])
        fx1 = max(x1, 0);  fx2 = min(x2, frame.shape[1])
        mh  = fy2 - fy1;   mw  = fx2 - fx1
        if mh > 0 and mw > 0:
            # Resize mask to match the actual clipped crop size (handles
            # bboxes that partially extend outside the frame boundary)
            mask_crop = cv2.resize(hist_mask, (mw, mh),
                                   interpolation=cv2.INTER_NEAREST)
            roi = frame[fy1:fy2, fx1:fx2]

            # Cyan highlight on kept (bin) pixels
            keep = (mask_crop == 255)
            roi[keep] = (roi[keep].astype(np.float32) * 0.55
                         + np.array([180, 220, 0], np.float32) * 0.45).astype(np.uint8)

            # Red tint on pruned (background) pixels
            prune = (mask_crop == 0)
            roi[prune] = (roi[prune].astype(np.float32) * 0.65
                          + np.array([0, 0, 200], np.float32) * 0.35).astype(np.uint8)

    # ── Bounding box ──────────────────────────────────────────────────────
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)

    # Centroid crosshair
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    cv2.drawMarker(frame, (cx, cy), (0, 220, 0),
                   cv2.MARKER_CROSS, 16, 2)

    # Label above box — include histogram coverage if available
    if hist_diag:
        # Overall mask coverage = fraction of crop pixels kept
        b_d = next((d for d in hist_diag if d["ch"] == "B"), {})
        g_d = next((d for d in hist_diag if d["ch"] == "G"), {})
        r_d = next((d for d in hist_diag if d["ch"] == "R"), {})
        label = (f"bin  conf={conf:.2f}  "
                 f"B={b_d.get('peak',0):.0f}±{b_d.get('std',0):.0f}  "
                 f"G={g_d.get('peak',0):.0f}±{g_d.get('std',0):.0f}  "
                 f"R={r_d.get('peak',0):.0f}±{r_d.get('std',0):.0f}")
    else:
        label = f"bin  conf={conf:.2f}"

    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
    cv2.rectangle(frame, (x1, y1 - lh - 8), (x1 + lw + 4, y1), (0, 140, 0), -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 255, 220), 1, cv2.LINE_AA)

    # World coords below box
    coord_txt = f"({xw:+.2f}, {yw:+.2f}, {zw:+.2f}) m"
    cv2.putText(frame, coord_txt, (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 220, 0), 1, cv2.LINE_AA)

    # HUD top-left
    _draw_hud(frame, frame_id, dt_ms, xw, yw, zw, conf, occluded=False)


def _draw_occluded(frame, predicted, occ_age, dt_ms, frame_id):
    """Draw occlusion banner + HUD when bin is not detected."""
    h, w = frame.shape[:2]

    # Red semi-transparent banner
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 48), (0, 0, 180), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    msg = f"OCCLUDED  age={occ_age}fr"
    cv2.putText(frame, msg, (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    if predicted is not None:
        xw, yw, zw = predicted
        _draw_hud(frame, frame_id, dt_ms, xw, yw, zw, conf=None, occluded=True)


def _draw_hud(frame, frame_id, dt_ms, xw, yw, zw, conf, occluded):
    """Small info panel in the bottom-left corner."""
    h, w = frame.shape[:2]
    lines = [
        f"frame : {frame_id:04d}",
        f"dt    : {dt_ms} ms",
        f"X     : {xw:+.3f} m",
        f"Y     : {yw:+.3f} m",
        f"Z     : {zw:+.3f} m",
        f"conf  : {conf:.2f}" if conf is not None else "conf  : (kalman)",
    ]
    pad    = 8
    lh     = 20
    box_h  = len(lines) * lh + pad * 2
    box_w  = 190
    y0     = h - box_h - 10

    overlay = frame.copy()
    cv2.rectangle(overlay, (6, y0), (6 + box_w, y0 + box_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    col = (180, 180, 180) if occluded else (200, 255, 200)
    for i, line in enumerate(lines):
        cv2.putText(frame, line,
                    (12, y0 + pad + (i + 1) * lh),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.47, col, 1, cv2.LINE_AA)


# ── Trajectory plot ───────────────────────────────────────────────────────────

def save_trajectory_plot(trajectory, waypoints_path, localizer,
                          out_path="trajectory.png"):
    if not trajectory:
        print("No trajectory data — skipping plot.")
        return

    xs = [p[0] for p in trajectory]
    ys = [p[1] for p in trajectory]
    n  = len(xs)

    fig, ax = plt.subplots(figsize=(11, 8))
    cmap = plt.cm.viridis

    for i in range(n - 1):
        ax.plot(xs[i:i+2], ys[i:i+2],
                color=cmap(i / max(n - 1, 1)), linewidth=1.8)

    ax.scatter(xs[0],  ys[0],  s=150, color="lime",    zorder=6,
               marker="^", label="start", edgecolors="black", linewidths=0.6)
    ax.scatter(xs[-1], ys[-1], s=150, color="crimson",  zorder=6,
               marker="v", label="end",   edgecolors="black", linewidths=0.6)

    if waypoints_path and os.path.exists(waypoints_path):
        with open(waypoints_path) as f:
            wp = json.load(f)
        color_map = {"green": "limegreen", "orange": "darkorange", "red": "crimson"}
        print("\nEstimated world-frame waypoint positions (ray-cast to ground):")
        for m in wp["markers"]:
            pw = localizer.pixel_to_world_ground(float(m["pixel_u"]), float(m["pixel_v"]))
            if pw is None:
                continue
            wx, wy = pw[0], pw[1]
            c = color_map.get(m["color"], "gold")
            ax.scatter(wx, wy, s=250, marker="*", zorder=7,
                       color=c, edgecolors="black", linewidths=0.6,
                       label=f"Stop {m['name']} ({m['color']})")
            ax.annotate(f"  {m['name']}  ({wx:.2f}, {wy:.2f}) m",
                        (wx, wy), fontsize=9, fontweight="bold")
            print(f"  Stop {m['name']:>2s} [{m['color']:>6s}]  "
                  f"pixel=({m['pixel_u']}, {m['pixel_v']})  ->  "
                  f"world = ({wx:.3f}, {wy:.3f}, 0.000) m")

    ax.set_xlabel("X world [m]  (forward from camera pole)", fontsize=11)
    ax.set_ylabel("Y world [m]  (left from camera pole)",    fontsize=11)
    ax.set_title("Garbage bin trajectory — world XY plane (top-down view)", fontsize=13)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_aspect("equal")
    ax.grid(True, linewidth=0.4, alpha=0.5)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=n))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Frame index", shrink=0.7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"trajectory.png saved -> {out_path}")


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Skyscouter bin tracker")
    parser.add_argument("--video",     required=True)
    parser.add_argument("--calib",     required=True)
    parser.add_argument("--output",    default="results/output.csv")
    parser.add_argument("--out-video", default="results/output.mp4",
                        help="Annotated output video  (default: results/output.mp4)")
    parser.add_argument("--waypoints", default="waypoints.json")
    parser.add_argument("--gpu",       action="store_true")
    parser.add_argument("--kalman",    action="store_true")
    args = parser.parse_args()

    K, D, cam_h, tilt_rad = load_calib(args.calib)
    localizer = Localizer(K, D, cam_h, tilt_rad)
    detector  = BinDetector(use_gpu=args.gpu)
    kf        = PositionKalman() if args.kalman else None

    os.makedirs(os.path.dirname(args.output)    or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_video) or ".", exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    # ── Video writer setup ────────────────────────────────────────────
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out_video, fourcc, fps, (width, height))
    print(f"Output video : {args.out_video}  ({width}x{height} @ {fps:.0f}fps)")

    trajectory = []
    last_pos   = None
    occ_age    = 0

    print(f"Camera : height={cam_h:.2f} m  tilt={np.degrees(tilt_rad):.1f} deg")
    print(f"Kalman : {'ENABLED' if kf else 'disabled'}")
    print(f"GPU    : {'ENABLED' if args.gpu else 'disabled (CPU only)'}")
    print("-" * 70)

    with open(args.output, "w") as csv_f:
        csv_f.write("frame_id,timestamp_ms,"
                    "x_cam,y_cam,z_cam,"
                    "x_world,y_world,z_world,conf,detected\n")

        frame_id = 0
        while True:
            t0  = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                break

            ts_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            det   = detector.detect(frame)

            if det is not None:
                x1, y1, x2, y2, conf, v_gnd = det
                xyz_cam   = localizer.estimate_cam((x1, y1, x2, y2), v_ground=v_gnd)

                # ── Depth plausibility gate ──────────────────────────────────
                # Bin should be 3–12 m from the camera.
                # If Z_cam is outside this range the blob is noise, a person,
                # or a background object — discard and treat as occluded.
                # (At 3 m: h_px ≈ 304 px — nearly fills the frame vertically,
                #  almost certainly the person.  At 12 m: h_px ≈ 76 px.)
                Z_MIN, Z_MAX = 2.5, 13.0   # slightly wider to tolerate imperfect bbox edges
                if not (Z_MIN <= xyz_cam[2] <= Z_MAX):
                    occ_age += 1
                    predicted = kf.predict() if kf is not None else last_pos
                    dt_ms = int((time.perf_counter() - t0) * 1000)
                    pos_str = (f"({predicted[0]:+.2f}, {predicted[1]:+.2f}, {predicted[2]:+.2f})"
                               if predicted is not None else "unknown")
                    print(f"[frame {frame_id:04d}] "
                          f"DEPTH-REJECT Z={xyz_cam[2]:.1f}m - last known {pos_str} m  "
                          f"dt={dt_ms:3d}ms")
                    _draw_occluded(frame, predicted, occ_age, dt_ms, frame_id)
                    writer.write(frame)
                    frame_id += 1
                    continue
                # ────────────────────────────────────────────────────────────

                xyz_world = localizer.cam_to_world(xyz_cam)

                # ── Kalman world-space gate ──────────────────────────────────
                # A bin pushed by a person moves at most ~1.5 m/s.
                # At 30 fps that is 0.05 m/frame.  We allow 10× that (0.5 m)
                # to handle fast pushes and Kalman lag, but anything larger
                # is almost certainly a false positive (sleeve, wall, etc.).
                # When the gate fires we discard the detection and let the
                # Kalman filter extrapolate instead.
                MAX_WORLD_JUMP = 0.5   # metres per frame
                if last_pos is not None:
                    jump = float(np.linalg.norm(
                        np.array(xyz_world[:2]) - np.array(last_pos[:2])
                    ))
                    if jump > MAX_WORLD_JUMP:
                        occ_age += 1
                        predicted = kf.predict() if kf is not None else last_pos
                        dt_ms = int((time.perf_counter() - t0) * 1000)
                        pos_str = (f"({predicted[0]:+.2f}, {predicted[1]:+.2f}, {predicted[2]:+.2f})"
                                   if predicted is not None else "unknown")
                        print(f"[frame {frame_id:04d}] "
                              f"KALMAN-GATE jump={jump:.2f}m > {MAX_WORLD_JUMP}m - last known {pos_str} m  "
                              f"dt={dt_ms:3d}ms")
                        _draw_occluded(frame, predicted, occ_age, dt_ms, frame_id)
                        writer.write(frame)
                        frame_id += 1
                        continue
                # ─────────────────────────────────────────────────────────────

                if kf is not None:
                    xyz_world = kf.update(xyz_world)

                last_pos = xyz_world
                occ_age  = 0
                xw, yw, zw = xyz_world
                dt_ms = int((time.perf_counter() - t0) * 1000)

                vg_hist = detector.last_vground_hist
                vg_hsv  = detector.last_vground_hsv
                vg_str  = (f"vgnd=hist:{vg_hist}/hsv:{vg_hsv}"
                           if vg_hist is not None else f"vgnd=hsv:{vg_hsv}")
                print(f"[frame {frame_id:04d}] "
                      f"bin @ world ({xw:+.2f}, {yw:+.2f}, {zw:+.2f}) m  "
                      f"conf={conf:.2f}  {vg_str}  dt={dt_ms:3d}ms")

                csv_f.write(f"{frame_id},{ts_ms},"
                            f"{xyz_cam[0]:.4f},{xyz_cam[1]:.4f},{xyz_cam[2]:.4f},"
                            f"{xw:.4f},{yw:.4f},{zw:.4f},{conf:.3f},1\n")
                csv_f.flush()
                trajectory.append((xw, yw))

                # Draw bbox with v2 histogram mask overlay.
                # detector.last_hist_mask and last_hist_diag are set as
                # side-effects of detector.detect() above.
                _draw_detection(frame, x1, y1, x2, y2, xyz_world, conf, dt_ms,
                                frame_id,
                                hist_mask=detector.last_hist_mask,
                                hist_diag=detector.last_hist_diag)

            else:
                occ_age += 1
                if kf is not None:
                    predicted = kf.predict()
                elif last_pos is not None:
                    predicted = last_pos
                else:
                    predicted = None

                # When lost for 50 frames (~1.7s) reset MOG2 so a
                # static bin re-emerges as new foreground
                dt_ms = int((time.perf_counter() - t0) * 1000)
                pos_str = (f"({predicted[0]:+.2f}, {predicted[1]:+.2f}, {predicted[2]:+.2f})"
                           if predicted is not None else "unknown")
                print(f"[frame {frame_id:04d}] "
                      f"OCCLUDED - last known {pos_str} m  "
                      f"age={occ_age}fr  dt={dt_ms:3d}ms")

                # After 15 frames with no detection the bin is almost
                # certainly static — kill the Kalman velocity so it stops
                # drifting and holds the last known position.
                if kf is not None and occ_age == 15 and kf.x is not None:
                    kf.x[3:] = 0.0   # zero vx, vy, vz

                # Write Kalman estimate to CSV even when occluded — the
                # assessment requires per-frame output. detected=0 flags
                # these as extrapolated, not directly observed.
                if predicted is not None:
                    pw = predicted
                    csv_f.write(f"{frame_id},{ts_ms},"
                                f"nan,nan,nan,"
                                f"{pw[0]:.4f},{pw[1]:.4f},{pw[2]:.4f},"
                                f"0.000,0\n")
                    csv_f.flush()

                _draw_occluded(frame, predicted, occ_age, dt_ms, frame_id)

            writer.write(frame)
            frame_id += 1

    cap.release()
    writer.release()
    print(f"\n{'─'*70}")
    print(f"Processed {frame_id} frames")
    print(f"CSV      -> {args.output}")
    print(f"Video    -> {args.out_video}")
    save_trajectory_plot(trajectory, args.waypoints, localizer)


if __name__ == "__main__":
    main()