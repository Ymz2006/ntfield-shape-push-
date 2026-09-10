"""arm/camera calibration -- pair robot poses with the table coordinate frame.

Live RealSense window with ArUco detection.  Three markers, **all ID 0**, sit flat on the
table at three corners of a known rectangle (``aruco.field_rect_cm``, default 94 x 66 cm):

* the **top-left** one is the origin ``(0, 0)``
* the one **straight down** from it is at ``(0, -66)``
* the **diagonal** one is at ``(+94, -66)`` -- same y as the straight-down one

so the table frame is: origin at the top-left marker, ``+x`` to the right, ``+y`` up (the
diagonal marker is therefore at positive x, negative y).

A small **ID-1** marker is tracked live: whenever it and the origin marker are both in
view its position relative to the origin is shown on the GUI (and saved with each sample),
``x`` and ``y`` in centimetres.

Press **c** to take a calibration sample.  Each sample is appended to the output file and
holds:

* the robot's reported joint angles (rad) and TCP pose ``(x,y,z,rx,ry,rz)`` in the base frame
* the field markers' pixel centres and their 3-D camera-frame points (from the aligned
  RealSense depth), and the **origin-to-diagonal difference** expressed in the table
  frame, ``x`` and ``y`` in centimetres
* the same difference measured a second, independent way -- ``solvePnP`` on each marker's
  four corners with its known physical side length (``aruco.aruco0_len_cm``), which needs
  no depth at all
* and a third way -- a 2-D Procrustes fit of all three markers onto the known rectangle
* ``marker1`` -- the tracked ID-1 marker's position relative to the origin, in cm (or null)

Three measurements of the same separation, so they cross-check each other:

* **depth**      -- deproject each marker centre through the aligned RealSense depth frame
* **pnp**        -- solve each marker's pose from its corners + its known side length
* **procrustes** -- the three markers are coplanar and their rectangle is known, so fit
  the plane they span, drop the points into it, and Kabsch/Umeyama-align them onto the
  rectangle.  This one measures the table's *own* axes instead of assuming the camera
  looks straight down, and the fitted ``scale`` is a direct readout of the metric error
  in whichever source fed it (marker size for pnp, depth scale for depth).

``--measure N`` skips the GUI entirely: it grabs N frames, prints all three methods'
``x``/``y`` separation with the spread over those frames, and exits.

    sudo docker run \\
      --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \\
      --volume="/home/jeffrey/ntrlshape_arm:/workspace" \\
      --privileged --volume="/dev:/dev" --network=host \\
      --runtime=nvidia -ti --rm ntrlshapelocal
    python camera_calibrate.py                       # robot at 192.168.10.2
    python camera_calibrate.py --ip 192.168.10.2 --dict DICT_4X4_50
    python camera_calibrate.py --no-robot            # camera only, robot fields null
    python camera_calibrate.py --measure 30          # headless: report depth vs pnp, then exit

Keys
----
    c         take a calibration sample -> append to --out (+ save a .png)
    s         save the current frame(s) only
    q / ESC   quit

The ID-1 marker is tracked and shown continuously; no key needed.
"""

import argparse
import json
import os
import time
from datetime import datetime

import cv2
import numpy as np

from camera_intrinsics import CameraIntrinsics
from real_world_params import PARAMS

WINDOW = "arm/camera calibration"
DEPTH_WINDOW = "arm/camera calibration -- depth"
ROBOT_IP = PARAMS.robot_ip
MARKER_ID = PARAMS.field_marker_id   # all three field markers carry this id
TRACK_ID = PARAMS.track_marker_id    # the small marker tracked live, relative to origin
ARUCO0_LEN_CM = PARAMS.aruco0_len_cm  # physical side length of the ID-0 field markers
FIELD_RECT_CM = PARAMS.field_rect_cm  # (x, y) cm of the rectangle the three markers sit on

DICTS = [
    "DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250", "DICT_4X4_1000",
    "DICT_5X5_50", "DICT_5X5_100", "DICT_5X5_250", "DICT_5X5_1000",
    "DICT_6X6_50", "DICT_6X6_100", "DICT_6X6_250", "DICT_6X6_1000",
    "DICT_7X7_50", "DICT_7X7_100", "DICT_7X7_250", "DICT_7X7_1000",
    "DICT_ARUCO_ORIGINAL",
    "DICT_APRILTAG_16h5", "DICT_APRILTAG_25h9",
    "DICT_APRILTAG_36h10", "DICT_APRILTAG_36h11",
]


# ======================================================================================
# RealSense: colour + aligned depth, plus pixel -> camera-frame-metres deprojection
# ======================================================================================
def _pick_fps(rs, serial, stream_type, fmt, width, height, want):
    """Highest fps <= want that the camera offers for this stream/format/resolution."""
    ctx = rs.context()
    devs = list(ctx.query_devices())
    if serial:
        devs = [d for d in devs
                if d.get_info(rs.camera_info.serial_number) == str(serial)]
    if not devs:
        return want                                  # let pipeline.start raise a clear error
    rates = []
    for s in devs[0].query_sensors():
        for p in s.get_stream_profiles():
            try:
                v = p.as_video_stream_profile()
            except Exception:
                continue
            if (p.stream_type() == stream_type and v.format() == fmt
                    and v.width() == width and v.height() == height):
                rates.append(p.fps())
    if not rates:
        return want
    ok = [r for r in rates if r <= want]
    return max(ok) if ok else min(rates)


class RealSenseCamera:
    """Colour stream with depth aligned to it, through ``pyrealsense2``."""

    def __init__(self, serial, width, height, fps=30):
        import pyrealsense2 as rs

        self.rs = rs
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        if serial:
            cfg.enable_device(str(serial))
        # A D415 on a USB-2 link can't do 1280x720@30, so clamp each stream to a rate the
        # camera actually offers for this resolution ("Couldn't resolve requests" otherwise).
        c_fps = _pick_fps(rs, serial, rs.stream.color, rs.format.bgr8, width, height, fps)
        d_fps = _pick_fps(rs, serial, rs.stream.depth, rs.format.z16, width, height, fps)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, c_fps)
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, d_fps)
        self.align = rs.align(rs.stream.color)

        try:
            profile = self.pipeline.start(cfg)
        except RuntimeError as exc:
            raise SystemExit(
                f"RealSense start failed: {exc}\n"
                "Plugged in and visible to the container?  Run with --privileged -v /dev:/dev.\n"
                "If the resolution is unsupported on this USB link, try --width 640 --height 480."
            )
        dev = profile.get_device()
        self.desc = (f"realsense {dev.get_info(rs.camera_info.name)} "
                     f"({dev.get_info(rs.camera_info.serial_number)}): "
                     f"{width}x{height} colour@{c_fps} + depth@{d_fps}")

        # Use the checkerboard calibration from calibrate_camera.py when it is present:
        # every colour frame is undistorted and pixels deproject with that pinhole model.
        # Without it, fall back to the RealSense factory intrinsics.
        self.calib = CameraIntrinsics.load()
        if self.calib is not None and not self.calib.matches(width, height):
            print(f"note: {self.calib.path} is {self.calib.width}x{self.calib.height}, "
                  f"camera is {width}x{height} -- ignoring it (re-run calibrate_camera.py "
                  f"at this resolution)")
            self.calib = None
        if self.calib is not None:
            self._calib_intrin = self.calib.rs_intrinsics(rs)
            print(f"using checkerboard calibration: {self.calib}")
        else:
            self._calib_intrin = None
            print("note: no camera intrinsics.json -- using RealSense factory intrinsics "
                  "(run  python calibrate_camera.py  to calibrate)")

        self.depth_vis = None
        self._depth_frame = None
        self._intrin = None

    def read(self):
        frames = self.align.process(self.pipeline.wait_for_frames())
        color = frames.get_color_frame()
        depth = frames.get_depth_frame()
        if not color:
            return False, None
        bgr = np.asanyarray(color.get_data())

        self._depth_frame = depth or None
        if depth:
            if self._calib_intrin is not None:
                self._intrin = self._calib_intrin
            else:
                self._intrin = depth.profile.as_video_stream_profile().intrinsics
            d = np.asanyarray(depth.get_data())
            self.depth_vis = cv2.applyColorMap(
                cv2.convertScaleAbs(d, alpha=0.03), cv2.COLORMAP_JET)

        if self.calib is not None:
            bgr = self.calib.undistort(bgr)
        return True, bgr

    def pinhole(self):
        """``(K, dist)`` matching the frames ``read`` hands back, or None before frame 1.

        With a checkerboard calibration loaded the frames are already undistorted, so the
        model is the ideal pinhole ``new_camera_matrix`` with zero distortion; otherwise it
        is the RealSense factory colour intrinsics and the frames are still distorted.
        """
        if self.calib is not None:
            return self.calib.new_camera_matrix(), np.zeros(5)
        if self._intrin is None:
            return None
        it = self._intrin
        K = np.array([[it.fx, 0.0, it.ppx],
                      [0.0, it.fy, it.ppy],
                      [0.0, 0.0, 1.0]], dtype=float)
        return K, np.asarray(it.coeffs, dtype=float).reshape(-1)

    def deproject(self, px, py, patch=4):
        """Camera-frame point ``[X, Y, Z]`` in metres for a colour pixel, or None.

        RealSense camera frame: ``+X`` right, ``+Y`` down, ``+Z`` into the scene.  Depth is
        sampled over a small window and the median of the valid (non-zero) hits is used.
        """
        if self._depth_frame is None or self._intrin is None:
            return None
        px, py = int(round(px)), int(round(py))
        w, h = self._intrin.width, self._intrin.height
        # (px, py) is in the undistorted colour frame when a calibration is loaded, but the
        # aligned depth frame is still in raw pixels -- sample depth at the raw pixel.
        if self.calib is not None:
            sx, sy = self.calib.raw_pixel(px, py)
            sx, sy = int(round(sx)), int(round(sy))
        else:
            sx, sy = px, py
        ds = []
        for y in range(max(0, sy - patch), min(h, sy + patch + 1)):
            for x in range(max(0, sx - patch), min(w, sx + patch + 1)):
                dist = self._depth_frame.get_distance(x, y)
                if dist > 0:
                    ds.append(dist)
        if not ds:
            return None
        z = float(np.median(ds))
        return self.rs.rs2_deproject_pixel_to_point(self._intrin, [float(px), float(py)], z)

    def close(self):
        self.pipeline.stop()


def realsense_present():
    try:
        import pyrealsense2 as rs
    except ImportError:
        return False
    try:
        return len(rs.context().query_devices()) > 0
    except Exception:
        return False


# ======================================================================================
# robot: reported state only, no motion
# ======================================================================================
class Robot:
    """Read-only RTDE receive connection to the UR arm."""

    def __init__(self, ip):
        from rtde_receive import RTDEReceiveInterface

        self.ip = ip
        print(f"connecting to robot {ip} ...")
        self.r = RTDEReceiveInterface(ip)
        print("  connected")

    def sample(self):
        return {
            "ip": self.ip,
            "connected": True,
            "joints_rad": [float(v) for v in self.r.getActualQ()],
            "tcp_pose": [float(v) for v in self.r.getActualTCPPose()],
        }

    def close(self):
        try:
            self.r.disconnect()
        except Exception:
            pass


def null_robot_sample():
    return {"ip": None, "connected": False, "joints_rad": None, "tcp_pose": None}


# ======================================================================================
# aruco
# ======================================================================================
def make_detector(dict_name):
    """Return ``detect(gray) -> (corners, ids, rejected)`` for the given dictionary."""
    if not hasattr(cv2, "aruco"):
        raise SystemExit("cv2.aruco is missing -- install opencv-contrib-python.")
    if not hasattr(cv2.aruco, dict_name):
        raise SystemExit(f"unknown dictionary {dict_name!r}; choose one of:\n  "
                         + "  ".join(DICTS))
    key = getattr(cv2.aruco, dict_name)

    # Subpixel corner refinement: the default integer corners are good enough for a
    # centre, but solvePnP reads the marker's *size in pixels* as its distance, so half a
    # pixel of corner error is centimetres of range error at table distance.
    if hasattr(cv2.aruco, "ArucoDetector"):                       # OpenCV >= 4.7
        dictionary = cv2.aruco.getPredefinedDictionary(key)
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        detector = cv2.aruco.ArucoDetector(dictionary, params)
        return detector.detectMarkers

    dictionary = cv2.aruco.Dictionary_get(key)                    # OpenCV < 4.7
    params = cv2.aruco.DetectorParameters_create()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return lambda gray: cv2.aruco.detectMarkers(gray, dictionary, parameters=params)


def field_quads(corners, ids, target=MARKER_ID):
    """Every ``target``-id marker's corner quad in view, as (4, 2) float arrays."""
    if ids is None:
        return []
    return [np.asarray(c, dtype=np.float64).reshape(4, 2)
            for c, i in zip(corners, ids.flatten()) if int(i) == target]


def field_triple(corners, ids, rect_cm=FIELD_RECT_CM):
    """The three ID-0 quads as ``(origin, ydown, diag)``, or ``None``.

    The markers sit on three corners of the ``rect_cm`` rectangle -- ``origin`` top left,
    ``ydown`` below it at ``(0, -rect_y)``, ``diag`` across at ``(+rect_x, -rect_y)`` --
    so they are told apart by their geometry alone, with no assumption about how the
    camera is rotated over the table:

    * the two markers furthest apart in the image are the rectangle's *diagonal*, so the
      third one is the right-angle corner, ``ydown``;
    * the two legs meeting at ``ydown`` have known, different lengths: ``rect_y`` back to
      ``origin`` and ``rect_x`` on to ``diag``.  Whichever leg the rectangle says is
      shorter is the one that identifies ``origin``.

    The legs do have to be distinguishable in the image for that second step; with the
    default 94 x 66 cm rectangle they differ by 42%, far more than perspective can
    distort.  A square-ish rectangle would make the two indistinguishable.
    """
    quads = field_quads(corners, ids)
    if len(quads) != 3:
        return None
    c = [q.mean(axis=0) for q in quads]
    i, j = max(((0, 1), (0, 2), (1, 2)),
               key=lambda p: float(np.linalg.norm(c[p[0]] - c[p[1]])))
    k = 3 - i - j                                     # the remaining, right-angle corner
    # Order the diagonal pair by their pixel distance to that corner, then hand the near
    # end to whichever of (origin, diag) the rectangle says sits on the shorter leg.
    near, far = sorted((i, j), key=lambda m: float(np.linalg.norm(c[m] - c[k])))
    origin, diag = (near, far) if rect_cm[1] < rect_cm[0] else (far, near)
    return quads[origin], quads[k], quads[diag]


def field_markers(corners, ids, rect_cm=FIELD_RECT_CM):
    """The origin and diagonal ID-0 quads, ordered ``(origin, xaxis)``, or ``None``.

    This is the pair the depth and solvePnP methods measure -- the rectangle's diagonal,
    ``(+rect_x, -rect_y)`` apart.  With three markers in view they come from
    ``field_triple``; with the older two-marker layout the top-left one (smallest pixel
    ``x + y``) is the origin and the other is the ``+x, -y`` corner.
    """
    quads = field_quads(corners, ids)
    if len(quads) == 3:
        origin, _, diag = field_triple(corners, ids, rect_cm)
        return origin, diag
    if len(quads) != 2:
        return None
    a, b = sorted(quads, key=lambda q: float(q.mean(axis=0).sum()))
    return a, b


def quad_centres(quads):
    """A sequence of corner quads -> their pixel centres, in the same order."""
    return tuple(q.mean(axis=0) for q in quads)


def marker_center(corners, ids, target):
    """Pixel centre (float x, y) of the first marker with id ``target``, or None."""
    if ids is None:
        return None
    for c, i in zip(corners, ids.flatten()):
        if int(i) == target:
            return c.reshape(4, 2).mean(axis=0)
    return None


def marker_edge_cm(cam, corners, ids, target=MARKER_ID):
    """Mean physical edge length [cm] of an id-``target`` marker, from deprojected corners.

    A depth-scale sanity check: this should read back close to ``ARUCO0_LEN_CM``.  Returns
    ``None`` when no such marker has valid depth on all four corners.
    """
    if ids is None:
        return None
    for c, i in zip(corners, ids.flatten()):
        if int(i) != target:
            continue
        quad = c.reshape(4, 2)
        pts = [cam.deproject(*p) for p in quad]
        if any(p is None for p in pts):
            return None
        pts = [np.asarray(p, dtype=float) for p in pts]
        edges = [np.linalg.norm(pts[(k + 1) % 4] - pts[k]) for k in range(4)]
        return float(np.mean(edges) * 100.0)
    return None


# --------------------------------------------------------------------------------------
# method 2: solvePnP -- pose of each marker from its 4 corners + its known side length.
# Uses no depth at all; the physical marker size is the only metric scale.
# --------------------------------------------------------------------------------------
def marker_object_points(side_cm):
    """The 4 corners of a marker in its own frame, metres, in ArUco's corner order.

    ArUco hands corners back clockwise from the marker's top-left, so the object points
    run (-s/2, +s/2), (+s/2, +s/2), (+s/2, -s/2), (-s/2, -s/2) with z = 0 -- the same
    convention ``cv2.aruco.estimatePoseSingleMarkers`` used.
    """
    h = side_cm / 200.0                                  # cm -> m, half a side
    return np.array([[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]],
                    dtype=np.float64)


def solve_marker_pose(quad, K, dist, side_cm):
    """``(rvec, tvec)`` of one marker from its corner quad, or None if the solve fails.

    ``tvec`` is the marker centre in the camera frame, metres.  IPPE_SQUARE is the
    planar-square solver: it is exact for 4 coplanar corners and far steadier than the
    iterative solver at the near-fronto-parallel view a top-down table camera gives.
    """
    flags = getattr(cv2, "SOLVEPNP_IPPE_SQUARE", cv2.SOLVEPNP_ITERATIVE)
    ok, rvec, tvec = cv2.solvePnP(marker_object_points(side_cm),
                                  np.ascontiguousarray(quad, dtype=np.float64),
                                  K, dist, flags=flags)
    if not ok:
        return None
    return rvec.reshape(3), tvec.reshape(3)


def table_cm(d):
    """Camera-frame delta [m] -> table frame (x, y, z) [cm].

    Table frame: ``+x`` = camera right, ``+y`` = camera up, so the RealSense ``+Y`` (down)
    axis flips sign.  Assumes the camera looks straight down at the table.
    """
    return float(d[0] * 100.0), float(-d[1] * 100.0), float(d[2] * 100.0)


def relative_to_origin_cm(cam, origin_px, point_px):
    """Table-frame position of ``point_px`` relative to ``origin_px``, in cm, or None."""
    p_o = cam.deproject(*origin_px)
    p_p = cam.deproject(*point_px)
    if p_o is None or p_p is None:
        return None
    d = np.array(p_p) - np.array(p_o)
    x_cm, y_cm, z_cm = table_cm(d)
    return {
        "px": [float(point_px[0]), float(point_px[1])],
        "xyz_cam_m": [float(v) for v in p_p],
        "x_cm": x_cm,
        "y_cm": y_cm,
        "z_cm": z_cm,
        "distance_cm": float(np.linalg.norm(d) * 100.0),
    }


def table_frame_diff(cam, origin_px, xaxis_px):
    """Marker-B-minus-marker-A in the table frame: dict with x_cm, y_cm, ... or None.

    Table frame: origin at the top-left marker, ``+x`` = camera right, ``+y`` = camera up.
    """
    rel = relative_to_origin_cm(cam, origin_px, xaxis_px)
    if rel is None:
        return None
    p_o = cam.deproject(*origin_px)
    return {
        "origin_px": [float(origin_px[0]), float(origin_px[1])],
        "xaxis_px": rel["px"],
        "origin_xyz_cam_m": [float(v) for v in p_o],
        "xaxis_xyz_cam_m": rel["xyz_cam_m"],
        "x_cm": rel["x_cm"],                          # camera +X (right) -> table +x
        "y_cm": rel["y_cm"],                          # camera +Y (down)  -> table -y
        "z_cm": rel["z_cm"],                          # depth difference, informational
        "distance_cm": rel["distance_cm"],
    }


def pnp_frame_diff(cam, pair, side_cm=ARUCO0_LEN_CM):
    """Marker-B-minus-marker-A in the table frame from solvePnP alone: dict, or None.

    Same table frame and same sign convention as ``table_frame_diff``, so the two dicts
    are directly comparable -- that is the whole point of measuring it twice.
    """
    model = cam.pinhole()
    if model is None:
        return None
    K, dist = model
    poses = [solve_marker_pose(q, K, dist, side_cm) for q in pair]
    if any(p is None for p in poses):
        return None
    (_, t_o), (_, t_x) = poses
    d = t_x - t_o
    x_cm, y_cm, z_cm = table_cm(d)
    return {
        "origin_px": [float(v) for v in pair[0].mean(axis=0)],
        "xaxis_px": [float(v) for v in pair[1].mean(axis=0)],
        "origin_xyz_cam_m": [float(v) for v in t_o],
        "xaxis_xyz_cam_m": [float(v) for v in t_x],
        "origin_range_cm": float(np.linalg.norm(t_o) * 100.0),
        "xaxis_range_cm": float(np.linalg.norm(t_x) * 100.0),
        "x_cm": x_cm,
        "y_cm": y_cm,
        "z_cm": z_cm,
        "distance_cm": float(np.linalg.norm(d) * 100.0),
        "marker_len_cm": float(side_cm),
    }


# --------------------------------------------------------------------------------------
# method 3: 2-D Procrustes -- fit the three markers onto the rectangle they are known to
# sit on.  The markers lie flat on the table, so they are coplanar, and their layout is
# known: origin (0, 0), straight-down (0, -rect_y), diagonal (+rect_x, -rect_y).  So
#
#   1. take the three marker centres as 3-D camera-frame points -- from solvePnP (no depth
#      at all) or from the aligned depth frame; both work and the two disagreeing is
#      itself the interesting signal;
#   2. fit the plane they span and re-express them as 2-D coordinates inside it, which
#      *measures* the camera's tilt out instead of assuming it away;
#   3. Kabsch/Umeyama-align those three 2-D points onto the known rectangle.
#
# What this gives that the other two methods cannot:
#
# * ``x_cm`` / ``y_cm`` in the table's *own* axes rather than in camera right/up.  Camera
#   right/up is only the table frame when the camera looks exactly straight down --
#   methods 1 and 2 assume that (see ``table_cm``); this one measures the error in it.
# * ``scale`` -- three points give 6 numbers and the similarity fit has 4 free parameters
#   (rotation, scale, 2-D translation), so with the rectangle as ground truth the fitted
#   scale reads off the metric error of whichever source fed it: for pnp that is the error
#   in ``aruco0_len_cm`` (or in the focal length), for depth it is the depth scale.
# * ``rms_cm`` -- the 2 degrees of freedom the fit cannot absorb, namely the leg-length
#   ratio and the corner angle.  A small residual means the three markers really are laid
#   out the way ``field_rect_cm`` claims.
# --------------------------------------------------------------------------------------
def rect_model_cm(rect_cm=FIELD_RECT_CM):
    """The three marker centres in the table frame, cm: origin, straight-down, diagonal."""
    x, y = rect_cm
    return np.array([[0.0, 0.0], [0.0, -y], [x, -y]], dtype=np.float64)


def plane_coords(points_cam):
    """3-D camera-frame points -> ``(uv, normal)``, their 2-D coordinates in their own plane.

    The basis comes from the SVD of the centred points; the normal is flipped to face the
    camera and the in-plane axes are ordered so ``(e1, e2, n)`` is right-handed.  That
    matches the table frame's own handedness -- table ``+x`` cross ``+y`` points up out of
    the table, toward the camera -- so the alignment that follows is a pure rotation and a
    mirrored point set cannot pass as a good fit.
    """
    P = np.asarray(points_cam, dtype=np.float64)
    c = P.mean(axis=0)
    _, _, vt = np.linalg.svd(P - c)
    e1, n = vt[0], vt[2]
    if float(n @ c) > 0.0:                    # camera sits at the origin looking down +Z
        n = -n
    e2 = np.cross(n, e1)
    return np.stack([(P - c) @ e1, (P - c) @ e2], axis=1), n


def procrustes_2d(src, dst):
    """Umeyama similarity ``src -> dst``: ``(scale, R, t, rms)``, ``R`` a proper rotation.

    Minimises ``sum ||scale * R @ src_i + t - dst_i||^2``.  Reflections are excluded (the
    ``d`` term), so a mirrored point set shows up as a large ``rms`` instead of a fit.
    """
    S = np.asarray(src, dtype=np.float64)
    D = np.asarray(dst, dtype=np.float64)
    cs, cd = S.mean(axis=0), D.mean(axis=0)
    S0, D0 = S - cs, D - cd
    U, sig, Vt = np.linalg.svd(S0.T @ D0)
    d = 1.0 if np.linalg.det(Vt.T @ U.T) > 0.0 else -1.0
    R = Vt.T @ np.diag([1.0, d]) @ U.T
    var = float((S0 ** 2).sum())
    scale = float((sig[0] + d * sig[1]) / var) if var > 0.0 else 1.0
    t = cd - scale * (R @ cs)
    resid = (scale * (R @ S.T).T + t) - D
    return scale, R, t, float(np.sqrt((resid ** 2).sum() / len(S)))


def marker_points_cam(cam, triple, side_cm, source):
    """The three marker centres as camera-frame metres, (3, 3), or None.

    ``source="pnp"`` solves each marker's pose from its four corners and its known side
    length (no depth at all); ``source="depth"`` deprojects each centre through the
    aligned depth frame.
    """
    if source == "depth":
        pts = [cam.deproject(*c) for c in quad_centres(triple)]
        if any(p is None for p in pts):
            return None
        return np.asarray(pts, dtype=np.float64)
    model = cam.pinhole()
    if model is None:
        return None
    K, dist = model
    poses = [solve_marker_pose(q, K, dist, side_cm) for q in triple]
    if any(p is None for p in poses):
        return None
    return np.asarray([t for _, t in poses], dtype=np.float64)


def procrustes_frame_diff(cam, triple, side_cm=ARUCO0_LEN_CM, rect_cm=FIELD_RECT_CM,
                          source="pnp"):
    """The three-marker fit, as a dict comparable with the other two methods, or None.

    ``x_cm`` / ``y_cm`` are the diagonal marker minus the origin marker in the rectangle's
    own axes and deliberately *unscaled* -- the fitted ``scale`` is reported separately
    rather than folded in, so these stay an honest measurement of the same quantity
    ``table_frame_diff`` and ``pnp_frame_diff`` report and should read
    ``(+rect_x, -rect_y)``.
    """
    if triple is None:
        return None
    pts = marker_points_cam(cam, triple, side_cm, source)
    if pts is None:
        return None

    uv, normal = plane_coords(pts)
    uv_cm = uv * 100.0                                              # metres -> cm
    scale, R, _, rms = procrustes_2d(uv_cm, rect_model_cm(rect_cm))
    a, b, c = (R @ uv_cm.T).T             # rotated into the table's axes, still unscaled
    d_diag, d_down = c - a, b - a

    # Methods 1 and 2 take camera right/up as the table axes, i.e. they assume the table
    # normal is the camera's -Z.  This is the angle between the two.
    tilt_deg = float(np.degrees(np.arccos(np.clip(-normal[2], -1.0, 1.0))))
    v1, v2 = a - b, c - b
    corner_deg = float(np.degrees(np.arccos(np.clip(
        float(v1 @ v2) / float(np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0))))

    return {
        "source": source,
        "origin_px": [float(v) for v in triple[0].mean(axis=0)],
        "ydown_px": [float(v) for v in triple[1].mean(axis=0)],
        "diag_px": [float(v) for v in triple[2].mean(axis=0)],
        "xyz_cam_m": [[float(v) for v in p] for p in pts],
        "x_cm": float(d_diag[0]),                   # diagonal marker - origin, table axes
        "y_cm": float(d_diag[1]),
        "distance_cm": float(np.linalg.norm(d_diag)),
        "ydown_x_cm": float(d_down[0]),             # straight-down marker - origin
        "ydown_y_cm": float(d_down[1]),
        "leg_y_cm": float(np.linalg.norm(d_down)),  # expect rect_y
        "leg_x_cm": float(np.linalg.norm(c - b)),   # expect rect_x
        "corner_deg": corner_deg,                   # expect 90
        "scale": float(scale),                      # expect 1; the metric error of `source`
        "rms_cm": float(rms),
        "tilt_deg": tilt_deg,                       # 0 = camera looking straight down
        "rect_cm": [float(rect_cm[0]), float(rect_cm[1])],
        "marker_len_cm": float(side_cm),
    }


def depth_ranges_cm(cam, pair):
    """Straight-line camera-to-marker range [cm] of each field marker, from depth."""
    out = []
    for centre in quad_centres(pair):
        p = cam.deproject(*centre)
        out.append(None if p is None else float(np.linalg.norm(p) * 100.0))
    return out


# ======================================================================================
# drawing
# ======================================================================================
def draw_markers(frame, corners, ids, pair, triple=None):
    cv2.aruco.drawDetectedMarkers(frame, corners, ids, (0, 200, 0))
    if triple is not None:
        o, d, g = (tuple(np.round(c).astype(int)) for c in quad_centres(triple))
        # The rectangle's fourth corner is implied by the other three.  Only exact under
        # an affine camera, so this outline is a sighting aid, not a measurement.
        fourth = tuple(np.array(o) + np.array(g) - np.array(d))
        cv2.polylines(frame, [np.array([o, d, g, fourth], np.int32)], True,
                      (90, 90, 255), 1, cv2.LINE_AA)
        cv2.circle(frame, d, 6, (255, 200, 0), -1, cv2.LINE_AA)
        for c, t in (((0, 0, 0), 4), ((255, 200, 0), 2)):
            cv2.putText(frame, "0,-y", (d[0] + 8, d[1] - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, c, t, cv2.LINE_AA)
    if pair is None:
        return
    origin_px, xaxis_px = quad_centres(pair)
    o = tuple(np.round(origin_px).astype(int))
    x = tuple(np.round(xaxis_px).astype(int))
    cv2.arrowedLine(frame, o, x, (0, 165, 255), 2, cv2.LINE_AA, tipLength=0.05)
    for pt, label, color in ((o, "origin (0,0)", (0, 255, 0)),
                             (x, "+x,-y", (255, 0, 255))):
        cv2.circle(frame, pt, 6, color, -1, cv2.LINE_AA)
        for c, t in (((0, 0, 0), 4), (color, 2)):
            cv2.putText(frame, label, (pt[0] + 8, pt[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, t, cv2.LINE_AA)


def draw_tracked(frame, origin_px, point_px, rel):
    """Line from the origin to the tracked marker, labelled with its cm position."""
    o = tuple(np.round(origin_px).astype(int))
    p = tuple(np.round(point_px).astype(int))
    cv2.line(frame, o, p, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.circle(frame, p, 7, (255, 255, 0), 2, cv2.LINE_AA)
    label = f"m{TRACK_ID}  x={rel['x_cm']:+.1f} y={rel['y_cm']:+.1f} cm"
    for c, t in (((0, 0, 0), 4), ((255, 255, 0), 2)):
        cv2.putText(frame, label, (p[0] + 10, p[1] + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, t, cv2.LINE_AA)


def hud(frame, lines):
    for row, text in enumerate(lines):
        y = 24 + 22 * row
        for c, t in (((0, 0, 0), 4), ((255, 255, 255), 1)):
            cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, t,
                        cv2.LINE_AA)


# ======================================================================================
# headless measurement: depth vs solvePnP, over N frames
# ======================================================================================
def _stats(values):
    """(median, min, max, std) of a list of floats, or None when it is empty."""
    if not values:
        return None
    a = np.asarray(values, dtype=float)
    return float(np.median(a)), float(a.min()), float(a.max()), float(a.std())


def _fmt(stat, unit="cm", dec=2):
    if stat is None:
        return "     --"
    med, lo, hi, sd = stat
    return (f"{med:+8.{dec}f} {unit}   [{lo:+.{dec}f} .. {hi:+.{dec}f}]  "
            f"sd {sd:.{dec}f}")


def measure(cam, detect, n_frames, side_cm, rect_cm=FIELD_RECT_CM, save_png=None,
            warmup=15):
    """Grab ``n_frames`` frames and report the marker separation all three ways."""
    for _ in range(warmup):                    # let auto-exposure and the depth filters settle
        cam.read()
    print(f"\nmeasuring over {n_frames} frame(s); "
          f"marker side = {side_cm:.2f} cm (aruco.aruco0_len_cm), "
          f"rectangle = {rect_cm[0]:.1f} x {rect_cm[1]:.1f} cm (aruco.field_rect_cm)\n")
    cols = {k: [] for k in ("d_x", "d_y", "d_dist", "d_r0", "d_r1", "d_edge",
                            "p_x", "p_y", "p_dist", "p_r0", "p_r1",
                            "q_x", "q_y", "q_dist", "q_scale", "q_rms", "q_tilt",
                            "q_leg_x", "q_leg_y", "q_corner",
                            "qd_scale", "qd_rms")}
    seen, last_frame = 0, None
    for i in range(n_frames):
        ok, frame = cam.read()
        if not ok:
            continue
        last_frame = frame
        corners, ids, _ = detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        pair = field_markers(corners, ids, rect_cm)
        triple = field_triple(corners, ids, rect_cm)
        if pair is None:
            n = 0 if ids is None else int(np.sum(ids.flatten() == MARKER_ID))
            print(f"  frame {i:3d}: need 3 ID-{MARKER_ID} markers, have {n}")
            continue
        seen += 1
        depth = table_frame_diff(cam, *quad_centres(pair))
        pnp = pnp_frame_diff(cam, pair, side_cm)
        proc = procrustes_frame_diff(cam, triple, side_cm, rect_cm, "pnp")
        proc_d = procrustes_frame_diff(cam, triple, side_cm, rect_cm, "depth")
        if depth is not None:
            r0, r1 = depth_ranges_cm(cam, pair)
            cols["d_x"].append(depth["x_cm"])
            cols["d_y"].append(depth["y_cm"])
            cols["d_dist"].append(depth["distance_cm"])
            if r0 is not None:
                cols["d_r0"].append(r0)
            if r1 is not None:
                cols["d_r1"].append(r1)
            edge = marker_edge_cm(cam, corners, ids, MARKER_ID)
            if edge is not None:
                cols["d_edge"].append(edge)
        if pnp is not None:
            cols["p_x"].append(pnp["x_cm"])
            cols["p_y"].append(pnp["y_cm"])
            cols["p_dist"].append(pnp["distance_cm"])
            cols["p_r0"].append(pnp["origin_range_cm"])
            cols["p_r1"].append(pnp["xaxis_range_cm"])
        if proc is not None:
            cols["q_x"].append(proc["x_cm"])
            cols["q_y"].append(proc["y_cm"])
            cols["q_dist"].append(proc["distance_cm"])
            cols["q_scale"].append(proc["scale"])
            cols["q_rms"].append(proc["rms_cm"])
            cols["q_tilt"].append(proc["tilt_deg"])
            cols["q_leg_x"].append(proc["leg_x_cm"])
            cols["q_leg_y"].append(proc["leg_y_cm"])
            cols["q_corner"].append(proc["corner_deg"])
        if proc_d is not None:
            cols["qd_scale"].append(proc_d["scale"])
            cols["qd_rms"].append(proc_d["rms_cm"])
        d_txt = (f"depth x={depth['x_cm']:+7.2f} y={depth['y_cm']:+7.2f}"
                 if depth else "depth      --          --   ")
        p_txt = (f"pnp x={pnp['x_cm']:+7.2f} y={pnp['y_cm']:+7.2f}"
                 if pnp else "pnp      --          --   ")
        q_txt = (f"proc x={proc['x_cm']:+7.2f} y={proc['y_cm']:+7.2f} s={proc['scale']:.4f}"
                 if proc else "proc      --          --            --")
        print(f"  frame {i:3d}: {d_txt}  | {p_txt}  | {q_txt}")

    st = {k: _stats(v) for k, v in cols.items()}
    rx, ry = float(rect_cm[0]), float(rect_cm[1])
    diag_cm = float(np.hypot(rx, ry))
    print("\n" + "=" * 78)
    print(f"marker separation (diagonal marker - top-left origin), "
          f"{seen}/{n_frames} usable frames")
    print("depth/pnp table frame: +x = camera right, +y = camera up")
    print(f"procrustes table frame: the rectangle's own axes "
          f"(expect x={rx:+.2f} y={-ry:+.2f} |d|={diag_cm:.2f} cm)\n")
    print(f"  {'':16s}{'median':>11s}      {'[min .. max]':<24s}{'spread'}")
    for label, dk, pk, qk in (("x", "d_x", "p_x", "q_x"),
                              ("y", "d_y", "p_y", "q_y"),
                              ("separation |d|", "d_dist", "p_dist", "q_dist")):
        print(f"  {label}")
        print(f"    depth         {_fmt(st[dk])}")
        print(f"    pnp           {_fmt(st[pk])}")
        print(f"    procrustes    {_fmt(st[qk])}")
    print("\n  camera -> marker range")
    print(f"    depth origin  {_fmt(st['d_r0'])}")
    print(f"    pnp   origin  {_fmt(st['p_r0'])}")
    print(f"    depth +x,-y   {_fmt(st['d_r1'])}")
    print(f"    pnp   +x,-y   {_fmt(st['p_r1'])}")

    print(f"\n  three-marker fit (rectangle {rx:.1f} x {ry:.1f} cm, "
          f"aruco.field_rect_cm)")
    print(f"    scale  pnp    {_fmt(st['q_scale'], 'x ', 4)}")
    print(f"    scale  depth  {_fmt(st['qd_scale'], 'x ', 4)}")
    print(f"    rms    pnp    {_fmt(st['q_rms'])}")
    print(f"    rms    depth  {_fmt(st['qd_rms'])}")
    print(f"    leg -y        {_fmt(st['q_leg_y'])}   expect {ry:.2f}")
    print(f"    leg +x        {_fmt(st['q_leg_x'])}   expect {rx:.2f}")
    print(f"    corner angle  {_fmt(st['q_corner'], 'deg')}   expect 90.00")
    print(f"    camera tilt   {_fmt(st['q_tilt'], 'deg')}   0 = looking straight down")

    if st["d_edge"]:
        med = st["d_edge"][0]
        print(f"\n  depth scale check: ID-{MARKER_ID} edge reads {med:.2f} cm "
              f"vs {side_cm:.2f} cm printed ({100.0 * med / side_cm - 100.0:+.1f}%)")
    if st["q_scale"]:
        s = st["q_scale"][0]
        print(f"  pnp scale check:   the rectangle wants {s:.4f}x the pnp geometry "
              f"({100.0 * s - 100.0:+.2f}%), i.e. aruco0_len_cm reads "
              f"{side_cm * s:.2f} cm rather than {side_cm:.2f} cm")
    for label, ak, bk in (("pnp - depth", "p", "d"), ("procrustes - pnp", "q", "p"),
                          ("procrustes - depth", "q", "d")):
        if not (st[f"{ak}_x"] and st[f"{bk}_x"]):
            continue
        dx = st[f"{ak}_x"][0] - st[f"{bk}_x"][0]
        dy = st[f"{ak}_y"][0] - st[f"{bk}_y"][0]
        dd = st[f"{ak}_dist"][0] - st[f"{bk}_dist"][0]
        print(f"\n  {label}:  dx={dx:+.2f} cm  dy={dy:+.2f} cm  d|d|={dd:+.2f} cm")
    print("=" * 78)

    if save_png and last_frame is not None:
        cv2.imwrite(save_png, last_frame)
        print(f"wrote {save_png}")
    return seen


# ======================================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ip", default=ROBOT_IP, help="robot IP (default %(default)s)")
    ap.add_argument("--no-robot", action="store_true",
                    help="do not connect to the arm; robot fields are logged as null")
    ap.add_argument("--serial", default=None, help="RealSense serial (default: first found)")
    ap.add_argument("--dict", default=PARAMS.aruco_dict, metavar="NAME",
                    help="ArUco dictionary the field markers are from (default %(default)s)")
    ap.add_argument("--width", type=int, default=PARAMS.camera_width)
    ap.add_argument("--height", type=int, default=PARAMS.camera_height)
    ap.add_argument("--marker-len", type=float, default=ARUCO0_LEN_CM, metavar="CM",
                    help="physical side length of the ID-%d markers, cm, used by the "
                         "solvePnP method (default %%(default)s from real_world_params.json)"
                         % MARKER_ID)
    ap.add_argument("--rect-x", type=float, default=FIELD_RECT_CM[0], metavar="CM",
                    help="table-frame x of the diagonal marker, cm -- the long leg of the "
                         "three-marker rectangle (default %(default)s)")
    ap.add_argument("--rect-y", type=float, default=FIELD_RECT_CM[1], metavar="CM",
                    help="how far straight down the second marker is, cm -- the short leg "
                         "of the three-marker rectangle (default %(default)s)")
    ap.add_argument("--measure", type=int, default=0, metavar="N",
                    help="headless: read N frames, report the depth, solvePnP and "
                         "Procrustes marker separation, then exit (no GUI, no robot)")
    ap.add_argument("--measure-png", default=None, metavar="PATH",
                    help="with --measure, also write the last frame here")
    ap.add_argument("--out", default="calibration/arm_camera_calib.jsonl",
                    help="samples are appended here, one JSON object per line")
    args = ap.parse_args()

    if not realsense_present():
        raise SystemExit("no RealSense found -- this tool needs depth for the cm scale.")

    detect = make_detector(args.dict)
    rect_cm = (args.rect_x, args.rect_y)
    cam = RealSenseCamera(args.serial, args.width, args.height)
    print(cam.desc)

    if args.measure:
        try:
            seen = measure(cam, detect, args.measure, args.marker_len, rect_cm,
                           args.measure_png)
        finally:
            cam.close()
        raise SystemExit(0 if seen else 1)

    robot = None
    if not args.no_robot:
        robot = Robot(args.ip)
    else:
        print("--no-robot: robot fields will be null")

    try:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    except cv2.error as exc:
        cam.close()
        if robot:
            robot.close()
        raise SystemExit(f"cv2 has no GUI support ({exc}); need $DISPLAY + /tmp/.X11-unix.")

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    n_saved = 0
    if os.path.exists(out):
        with open(out) as fh:
            n_saved = sum(1 for line in fh if line.strip())
    if n_saved:
        print(f"{out} already has {n_saved} sample(s); appending")

    message = ""
    msg_until = 0.0
    fps, last = 0.0, time.time()

    def set_msg(text):
        nonlocal message, msg_until
        message = text
        msg_until = time.time() + 4.0
        print(text)

    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                blank = np.zeros((240, 640, 3), np.uint8)
                hud(blank, ["no frame from camera -- retrying"])
                cv2.imshow(WINDOW, blank)
                if (cv2.waitKey(200) & 0xFF) in (27, ord("q")):
                    break
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detect(gray)
            pair = field_markers(corners, ids, rect_cm)
            triple = field_triple(corners, ids, rect_cm)
            draw_markers(frame, corners, ids, pair, triple)

            live = table_frame_diff(cam, *quad_centres(pair)) if pair else None
            pnp = pnp_frame_diff(cam, pair, args.marker_len) if pair else None
            proc = procrustes_frame_diff(cam, triple, args.marker_len, rect_cm, "pnp")

            # Track the small ID-1 marker relative to the origin (needs the origin marker).
            track = None
            if pair is not None:
                origin_px = quad_centres(pair)[0]
                m1_px = marker_center(corners, ids, TRACK_ID)
                if m1_px is not None:
                    track = relative_to_origin_cm(cam, origin_px, m1_px)
                    if track is not None:
                        draw_tracked(frame, origin_px, m1_px, track)

            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
            last = now

            n_ids0 = 0 if ids is None else int(np.sum(ids.flatten() == MARKER_ID))
            status = (f"depth: x={live['x_cm']:+.1f}cm  y={live['y_cm']:+.1f}cm  "
                      f"d={live['distance_cm']:.1f}cm" if live
                      else f"need 3 ID-{MARKER_ID} markers with depth (have {n_ids0})")
            pnp_line = (f"pnp:   x={pnp['x_cm']:+.1f}cm  y={pnp['y_cm']:+.1f}cm  "
                        f"d={pnp['distance_cm']:.1f}cm  "
                        f"(range {pnp['origin_range_cm']:.0f}/{pnp['xaxis_range_cm']:.0f}cm)"
                        if pnp else
                        f"pnp:   need 2 ID-{MARKER_ID} markers (have {n_ids0})")
            proc_line = (f"proc:  x={proc['x_cm']:+.1f}cm  y={proc['y_cm']:+.1f}cm  "
                         f"scale={proc['scale']:.4f}  rms={proc['rms_cm']:.2f}cm  "
                         f"tilt={proc['tilt_deg']:.1f}deg"
                         if proc else
                         f"proc:  need 3 ID-{MARKER_ID} markers (have {n_ids0})")
            if track is not None:
                marker1 = f"m{TRACK_ID}:   x={track['x_cm']:+.1f}cm  y={track['y_cm']:+.1f}cm"
            elif marker_center(corners, ids, TRACK_ID) is not None:
                marker1 = f"m{TRACK_ID}:   visible, no depth"
            else:
                marker1 = f"m{TRACK_ID}:   not visible"
            edge_cm = marker_edge_cm(cam, corners, ids, MARKER_ID)
            scale = (f"m{MARKER_ID} edge: {edge_cm:.1f}cm  (expect {ARUCO0_LEN_CM:.1f}cm, "
                     f"{100.0 * edge_cm / ARUCO0_LEN_CM - 100.0:+.0f}%)"
                     if edge_cm else f"m{MARKER_ID} edge: no depth")
            head = [f"{fps:4.1f} fps  |  samples={n_saved}  |  c = sample, q = quit",
                    status, pnp_line, proc_line, marker1, scale]
            if message and now < msg_until:
                head.append(message)
            hud(frame, head)

            cv2.imshow(WINDOW, frame)
            if cam.depth_vis is not None:
                cv2.imshow(DEPTH_WINDOW, cam.depth_vis)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

            if key == ord("s"):
                cv2.imwrite(f"arm_camera_calib_{n_saved}_view.png", frame)
                set_msg(f"saved arm_camera_calib_{n_saved}_view.png")

            if key == ord("c"):
                if live is None:
                    set_msg(f"cannot sample: need both ID-{MARKER_ID} markers with valid depth")
                    continue
                try:
                    rob = robot.sample() if robot else null_robot_sample()
                except Exception as exc:                       # noqa: BLE001
                    set_msg(f"robot read failed: {exc}")
                    continue

                record = {
                    "index": n_saved,
                    "time": datetime.now().isoformat(timespec="seconds"),
                    "coord_convention": (
                        "table frame: origin = top-left ID-0 marker, +x = camera right, "
                        "+y = camera up; field x/y is (diagonal marker - origin) in cm; "
                        "markers_procrustes uses the three-marker rectangle's own axes "
                        "instead of camera right/up; "
                        f"marker{TRACK_ID} x/y is (ID-{TRACK_ID} marker - origin) in cm"
                    ),
                    "robot": rob,
                    "robot_tcp_xyz_cm": ([round(v * 100.0, 3) for v in rob["tcp_pose"][:3]]
                                         if rob["tcp_pose"] else None),
                    "markers": {"dict": args.dict, **live},
                    "markers_pnp": pnp,
                    "markers_procrustes": proc,
                    f"marker{TRACK_ID}": track,
                }
                with open(out, "a") as fh:
                    fh.write(json.dumps(record) + "\n")
                cv2.imwrite(f"arm_camera_calib_{n_saved}_view.png", frame)
                n_saved += 1
                set_msg(f"sample {record['index']}: x={live['x_cm']:+.1f}cm "
                        f"y={live['y_cm']:+.1f}cm -> {out}")
    finally:
        cam.close()
        if robot:
            robot.close()
        cv2.destroyAllWindows()
        print(f"{n_saved} sample(s) in {out}")


if __name__ == "__main__":
    main()
