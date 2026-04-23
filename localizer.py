"""
localizer.py
============
3-D localisation for the Skyscouter CV assessment.

All mathematical derivations are reproduced as inline comments here and
expanded with numbers in README.md §2.

Camera coordinate system (OpenCV convention)
  +X : right
  +Y : down
  +Z : forward (optical axis)

World coordinate system (pole base origin)
  +X : forward (optical axis projected to ground plane)
  +Y : left
  +Z : up

Camera mount: height 1.35 m, pitched 15° downward (tilt_deg = −15.0).
"""

import cv2
import numpy as np

# Known bin dimensions
BIN_DIAMETER_M = 0.40   # m  — as specified in assessment
BIN_HEIGHT_M   = 0.65   # m  — as specified in assessment


class Localizer:
    """
    Convert a 2-D bounding box → 3-D position in camera frame →
    3-D position in world frame.

    Parameters
    ----------
    K        : (3,3) camera intrinsic matrix
    D        : (5,)  distortion coefficients [k1, k2, p1, p2, k3]
    cam_h    : camera height above ground [m]
    tilt_rad : downward tilt angle [rad]  (negative = downward)
    """

    def __init__(
        self,
        K: np.ndarray,
        D: np.ndarray,
        cam_h: float,
        tilt_rad: float,
        h_px_alpha: float = 0.25,
    ):
        self.K      = K
        self.D      = D
        self.cam_h  = cam_h
        self.tilt   = tilt_rad
        self.R, self.t = self._build_extrinsic()

        # ── Temporal smoothing of the bin-height measurement ──────────
        # h_px = v_ground - y1 varies by a few pixels frame-to-frame from
        # bbox/segmentation noise, even though the bin's PHYSICAL height
        # is constant at 0.65 m.  An EMA over h_px removes that jitter
        # before Z = fy·H/h_px amplifies it.  At α = 0.25 the EMA settles
        # to within 1% of a step change in ~15 frames (0.5 s at 30 fps).
        self._h_px_ema   = None
        self._h_px_alpha = float(h_px_alpha)

    # ------------------------------------------------------------------
    # Extrinsic calibration
    # ------------------------------------------------------------------

    def _build_extrinsic(self):
        """
        Build rotation R and translation t such that:

            P_world = R @ P_cam + t

        ── Step 1: Axis Swap (level camera, tilt = 0) ──────────────────
        Map camera axes to world axes assuming the camera is perfectly level.

          cam +Z (forward) → world +X    row = [0,  0,  1]
          cam +X (right)   → world −Y    row = [-1, 0,  0]   (−Y: world Y is left)
          cam +Y (down)    → world −Z    row = [0,  -1, 0]   (−Z: world Z is up)

          axis_swap = [[ 0,  0,  1],
                       [-1,  0,  0],
                       [ 0, -1,  0]]

        ── Step 2: Pitch rotation (15° downward tilt) ──────────────────
        After axis_swap the frame matches world for a level camera.
        The camera is ACTUALLY pitched downward by |tilt| = 15°.
        In world frame this is a rotation of +|tilt| about the world +Y axis
        (the left/right axis = the pitch axis for a forward-facing camera).

        A rotation of angle α about world +Y:

            Ry(α) = [[ cos α,  0,  sin α],
                     [     0,  1,      0],
                     [-sin α,  0,  cos α]]

        We want the result of Ry(α) @ [1,0,0] (the post-swap forward direction)
        to tilt downward, i.e. to become [cos(15°), 0, −sin(15°)]:

            Ry(α) @ [1,0,0]^T = [cos α, 0, −sin α]

        Setting α = +15° = |tilt_rad| = −tilt_rad gives cos α = 0.9659,
        −sin α = −0.2588 which correctly points forward-and-DOWN.

        NOTE: tilt_rad is negative (−0.2618), so we use alpha = −tilt_rad.

        ── Step 3: Combine ─────────────────────────────────────────────

            R = Ry(alpha) @ axis_swap

        Numerically (alpha = 0.2618 rad = 15°, c = 0.9659, s = 0.2588):

            R = [[ 0,      −0.2588,   0.9659],
                 [−1,       0,         0    ],
                 [ 0,      −0.9659,  −0.2588]]

        Column check (each column = where that camera axis points in world):
          col 0 (cam +X right)   → [ 0, −1,  0]  = world −Y  ✓  (right = −left)
          col 1 (cam +Y down)    → [−0.2588, 0, −0.9659]     ✓  (down, slightly fwd)
          col 2 (cam +Z forward) → [ 0.9659, 0, −0.2588]     ✓  (forward + down 15°)

        ── Step 4: Translation ─────────────────────────────────────────
        The camera optical centre is at world position (0, 0, cam_h):

            t = [0, 0, cam_h] = [0, 0, 1.35]

        Full transform:   P_world = R @ P_cam + t
        """
        # Axis swap matrix
        axis_swap = np.array([
            [ 0,  0,  1],
            [-1,  0,  0],
            [ 0, -1,  0],
        ], dtype=np.float64)

        # Pitch rotation: alpha = −tilt_rad (positive for downward tilt)
        alpha = -self.tilt          # tilt is negative → alpha is positive
        c, s  = np.cos(alpha), np.sin(alpha)
        Ry    = np.array([
            [ c,  0,  s],
            [ 0,  1,  0],
            [-s,  0,  c],
        ], dtype=np.float64)

        R = Ry @ axis_swap
        t = np.array([0.0, 0.0, self.cam_h])
        return R, t

    # ------------------------------------------------------------------
    # Depth estimation  (task 2a)
    # ------------------------------------------------------------------

    def estimate_cam(self, bbox: tuple, v_ground: int | None = None) -> np.ndarray:
        """
        Estimate bin centroid in the CAMERA frame.

        ── Primary method: bin-height pinhole depth ────────────────────
        The pinhole camera projects a physical height H to h_px pixels at
        depth Z:

            Z = fy · H / h_px                    (README §2a)

        The detector's bounding box spans the bin BODY plus its mirror
        reflection on the polished floor, and `v_ground` is the midpoint
        of that blue region — i.e. the bin's actual ground-contact row.
        So the bin-body height in pixels is:

            h_px = v_ground - y1

        where y1 is the top of the detection bbox (≈ bin top).  This gives
        exactly the physical bin height 0.65 m without double-counting
        the reflection.

        The centroid sits at y_mid = (y1 + v_ground) / 2 and projects to
            X = (u_c - cx) · Z / fx
            Y = (v_mid - cy) · Z / fy

        All pixels are undistorted first via cv2.undistortPoints.

        ── Fallback: mirror-plane ray casting (original method) ────────
        If h_px is non-positive or Z comes out outside a sane range
        (0.5 m to 50 m), we fall back to ray-casting v_ground to the
        calibrated z_world = -0.41 m plane and lifting by BIN_HEIGHT_M/2
        + 0.41 m.  This is the method the original colour-based pipeline
        used — retained for robustness when bbox geometry is degenerate.

        Parameters
        ----------
        bbox     : (x1, y1, x2, y2) in raw (distorted) pixel coordinates.
        v_ground : pixel row of the bin's ground-contact (blue-region midpoint).
                   Required for height-based depth; without it we skip
                   straight to the fallback.

        Returns
        -------
        xyz_cam : (3,) ndarray — [X_c, Y_c, Z_c] in metres, camera frame.
        """
        x1, y1, x2, y2 = bbox
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]

        # ── Primary: height-based pinhole depth ────────────────────
        # Requires v_ground.  h_px is measured from bbox top (bin top)
        # down to v_ground (bin base), which is the true bin-body height.
        # This never double-counts the floor reflection that sits below
        # v_ground in the original bbox.
        if v_ground is not None:
            u_c     = (x1 + x2) / 2.0
            v_top   = float(y1)
            v_base  = float(v_ground)
            v_mid   = 0.5 * (v_top + v_base)

            # Undistort the three pixels we need
            raw = np.array(
                [[u_c, v_top], [u_c, v_base], [u_c, v_mid]],
                dtype=np.float32,
            ).reshape(-1, 1, 2)
            und = cv2.undistortPoints(raw, self.K, self.D, P=self.K)
            _, uy_top  = und[0, 0]
            uu_c, uy_b = und[1, 0]
            _, uy_mid  = und[2, 0]

            h_px = uy_b - uy_top
            if h_px > 1.0:
                # ── Temporal smoothing of h_px ──────────────────────
                # Cheap EMA in-place: h_ema = α·h_new + (1-α)·h_prev.
                # First measurement bootstraps the state directly.  If
                # the new h_px differs from the EMA by more than 50 %
                # (fast-moving bin, or one frame's bbox blew up), we
                # trust the new value more — blend at 0.60 instead of
                # 0.25 so a real motion is caught quickly.
                if self._h_px_ema is None:
                    self._h_px_ema = float(h_px)
                else:
                    change = abs(h_px - self._h_px_ema) / self._h_px_ema
                    a = 0.60 if change > 0.50 else self._h_px_alpha
                    self._h_px_ema = a * float(h_px) + (1.0 - a) * self._h_px_ema
                h_px_use = self._h_px_ema

                Z = float(fy * BIN_HEIGHT_M / h_px_use)
                if 0.5 < Z < 50.0:
                    X = float((uu_c  - cx) * Z / fx)
                    Y = float((uy_mid - cy) * Z / fy)
                    return np.array([X, Y, Z])

        # ── Fallback: mirror-plane ray-cast at z_world = -0.41 m ───
        # Identical to the original localizer's primary method.  Handles
        # degenerate h_px (e.g. bbox clamped to image edge, missing top).
        u_c      = (x1 + x2) / 2.0
        v_anchor = float(v_ground) if v_ground is not None else (y1 + y2) / 2.0
        Z_CAST   = -0.41
        P_mirror = self._pixel_to_world_plane(u_c, v_anchor, Z_CAST)
        if P_mirror is not None:
            offset = BIN_HEIGHT_M / 2.0 - Z_CAST
            P_ctr  = P_mirror + np.array([0.0, 0.0, offset])
            return self.R.T @ (P_ctr - self.t)

        # ── Final fallback: width-based pinhole depth ──────────────
        # Only reached when ray-cast is behind camera / parallel.
        raw_pts = np.array(
            [[x1, y1], [x2, y2], [(x1 + x2) / 2.0, (y1 + y2) / 2.0]],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        und_pts = cv2.undistortPoints(raw_pts, self.K, self.D, P=self.K)
        ux1, _   = und_pts[0, 0]
        ux2, _   = und_pts[1, 0]
        uc,  vc  = und_pts[2, 0]
        w_px = max(ux2 - ux1, 1.0)
        Z = fx * BIN_DIAMETER_M / w_px
        X = (uc - cx) * Z / fx
        Y = (vc - cy) * Z / fy
        return np.array([X, Y, Z])

    # ------------------------------------------------------------------
    # Coordinate transforms  (tasks 2b, 2c)
    # ------------------------------------------------------------------

    def cam_to_world(self, xyz_cam: np.ndarray) -> np.ndarray:
        """
        Transform a camera-frame position to the world frame.

            P_world = R @ P_cam + t

        where R and t are built in _build_extrinsic().
        """
        return self.R @ xyz_cam + self.t

    # ------------------------------------------------------------------
    # Waypoint projection  (bonus 2d)
    # ------------------------------------------------------------------

    def _pixel_to_world_plane(self, u: float, v: float,
                               z_plane: float) -> np.ndarray | None:
        """
        Ray-cast pixel (u, v) to the horizontal world plane at
        z_world = z_plane.

        Used to intersect:
          z_plane = 0              → ground contact (base of bin)
          z_plane = BIN_HEIGHT_M   → rim plane (top of bin)

        Derivation
        ----------
        Undistort (u,v) → normalised ray in camera frame:
            r_cam = [(u−cx)/fx, (v−cy)/fy, 1]
        World ray: r_w = R @ r_cam
        Point on ray: P(λ) = t + λ · r_w
        Set z_world = z_plane:
            t[2] + λ · r_w[2] = z_plane
            λ = (z_plane − t[2]) / r_w[2]
              = (z_plane − cam_h) / r_w[2]

        Returns None if ray is nearly parallel to the plane.
        """
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]

        pt     = np.array([[[u, v]]], dtype=np.float32)
        pt_und = cv2.undistortPoints(pt, self.K, self.D, P=self.K)
        u2, v2 = pt_und[0, 0]

        r_cam = np.array([(u2 - cx) / fx, (v2 - cy) / fy, 1.0])
        r_w   = self.R @ r_cam

        if abs(r_w[2]) < 1e-6:
            return None

        lam     = (z_plane - self.t[2]) / r_w[2]
        if lam < 0:
            return None          # intersection behind camera
        return self.t + lam * r_w

    def pixel_to_world_ground(self, u: float, v: float) -> np.ndarray | None:
        """Project pixel (u,v) to the ground plane (z_world=0)."""
        return self._pixel_to_world_plane(u, v, z_plane=0.0)