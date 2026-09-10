"""camera_test_id10 -- the table frame from the **single ID-10 corner marker**.

One **DICT_ARUCO_ORIGINAL** marker, id **10**, sits flat on a corner of the table and is
the only marker on it.  It *is* the table frame:

    id 10 ....... length ....... (+L, 0)   origin (0, 0) at the ID-10 marker centre
     origin (0,0)                |            +x runs along the length, 79 cm
      |                          |            +y runs along the width,  63 cm
    width                      width          so the table occupies x >= 0, y >= 0
      |                          |
    (0, +W) ..... length ..... (+L, +W)    +z comes out of the table toward the camera

In the four-marker layout ``camera_test_table.py`` measures, this is the corner the ids
0, 20 and 30 used to bracket: ``10 -> 0`` is the length and is now **+x**, ``10 -> 20`` is
the width and is now **+y**.  Those two markers are gone -- nothing here detects, draws or
solves for any id but 10 -- so the rectangle is no longer *measured*, it is the nominal
``--length`` x ``--width`` (79 x 63 cm, unchanged) drawn out from the one marker.

Which way the axes point therefore comes from the marker's own printed orientation: lay it
down with its **+x edge (ArUco corner 0 -> corner 1) pointing along the length** and its
**+y edge pointing along the width**, or measure how far it ended up rotated and pass
``--yaw-offset`` (degrees CCW from table +x to the marker's own +x edge); the table axes
are the marker's spun back by that angle.

The pose is found two independent ways, and everything is reported for both:

* **pnp**   -- ``solvePnP`` on the marker's four corners.  No depth at all; the printed
  ``--marker-len`` is the only metric scale, so a wrong one scales every range by that
  ratio (it does *not* rotate the frame, so the table-frame directions survive it).
* **depth** -- the four corners deprojected through the aligned RealSense depth frame, and
  the frame read straight off them: ``+x`` from the two horizontal edges, ``+y`` from the
  two vertical ones, ``+z`` their cross product.  Independent of the marker size -- and
  because it *measures* the marker's side, it is also the only scale check left once the
  79 x 63 rectangle is no longer being observed.

What one marker cannot do is average anything: a four-marker board votes on the origin and
the plane four times over, and here every number rests on one 5.5 cm square.  So the two
diagnostics that matter most are printed prominently:

* **lever arm** -- the frame's noise, propagated out to each nominal table corner.  An
  angular wobble that is invisible at the marker is ``79/5.5 ~ 14x`` larger at the far
  corner, and this says how many centimetres that is.
* **pnp vs depth** -- how far the two independent frames disagree, in origin (cm) and in
  axis direction (deg).  With no redundancy inside either source, this is the cross-check.

A separate **T-shape marker -- id 1, DICT_4X4_50** (``--tee-id`` / ``--tee-dict``) is
tracked whenever it is in view and its position reported in that table frame, four ways:

* **plane**       -- its pixel ray intersected with the table plane from the pnp pose.
  Needs neither the T marker's physical size nor depth; quote this one.
  ``--tee-height`` lifts the plane if the marker sits on top of something thick.
* **plane_depth** -- the same ray, through the depth-built frame: how much the choice of
  frame alone moves the answer.
* **pnp**         -- ``solvePnP`` on the T marker alone, scaled by ``--tee-len``.
* **depth**       -- its centre deprojected through the aligned depth frame.

``yaw`` is the marker's own rotation on the table, degrees CCW from table ``+x``.

Colour frames are undistorted with ``calibration/camera/intrinsics.json`` when it exists
(run ``calibrate_camera.py`` to make one); without it the RealSense factory intrinsics are
used and every method gets noticeably worse.

    sudo docker run \\
      --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \\
      --volume="/home/jeffrey/ntrlshape_arm:/workspace" \\
      --privileged --volume="/dev:/dev" --network=host \\
      --runtime=nvidia -ti --rm ntrlshapelocal
    python camera_test_id10.py                       # live window
    python camera_test_id10.py --measure 30          # headless: report both frames, exit
    python camera_test_id10.py --length 79 --width 63 --marker-len 5.5
    python camera_test_id10.py --yaw-offset -3.5     # marker glued down 3.5 deg clockwise

Keys
----
    s         save the current frame as a .png
    q / ESC   quit
"""

import argparse
import json
import time

import cv2
import numpy as np

from camera_test import (
    DICTS,
    RealSenseCamera,
    _fmt as fmt_stat,
    _stats as stats,
    hud,
    make_detector,
    marker_object_points,
    realsense_present,
    solve_marker_pose,
)
# The frame conversions live in frame_conversions now -- imported rather than defined so
# this file and locate_functions cannot drift apart on what "table cm" means.  The two
# that stayed behind, ray_to_table_cm and project_table, are the two that need the lens
# model (undistort / project), which is cv2 work and belongs next to ``cam``.
from frame_conversions import table_yaw_deg, tilt_deg, to_cam_m, to_table_cm
from real_world_params import PARAMS

WINDOW = "table frame -- single id 10 marker"
DEPTH_WINDOW = "table frame -- depth"

ARUCO_DICT = "DICT_ARUCO_ORIGINAL"
ORIGIN_ID = 10                     # the only table marker; its centre is (0, 0)
MARKER_LEN_CM = 5.5
# Heading of the marker's own +x edge, deg CCW from table +x -- measured, so the frame
# does not depend on having glued the marker down square.  See real_world_params.json.
YAW_OFFSET_DEG = PARAMS.origin_marker_yaw_deg
LENGTH_CM = 79.0                   # +x extent of the table, 10 -> (old) id 0
WIDTH_CM = 63.0                    # +y extent of the table, 10 -> (old) id 20

TEE_DICT = "DICT_4X4_50"           # the T's markers are from a different dictionary
# The shape carries FOUR markers of different ids, so one going under the arm is not a lost
# frame -- ``frame_conversions.TEE_MARKER_IDS`` is the set and TEE_MARKER_POS_CM holds
# where each one sits on the shape.  This file is the single-marker DIAGNOSTIC, so it
# looks at whichever one ``--tee-id`` names; the multi-marker read is
# ``locate_functions.locate_shape``.
TEE_ID = 1                         # which one this diagnostic inspects by default
TEE_LEN_CM = 5.5                   # only the pnp estimate of the T depends on this

SOURCES = ("pnp", "depth")
AXIS_LEN_CM = 15.0                 # length of the drawn frame arrows

# The marker's four corners, in ArUco's order, named by the edge each pair spans.  Used
# for the depth source's axes and for its measured-side / squareness diagnostics.
EDGES = ((0, 1, "x"), (3, 2, "x"), (3, 0, "y"), (2, 1, "y"))
DIAGONALS = ((0, 2), (1, 3))

# How the T marker is placed, and which of its coordinates each method can give.
TEE_METHODS = (("plane", ("x", "y", "yaw")), ("plane_depth", ("x", "y", "yaw")),
               ("pnp", ("x", "y", "z", "yaw")), ("depth", ("x", "y", "z")))


# ======================================================================================
# the one marker
# ======================================================================================
def table_corners_cm(length_cm, width_cm):
    """The four nominal table corners in the table frame, cm, as ``(name, (x, y))``.

    Only the first one is observed -- it is the marker itself.  The other three are where
    the ``--length`` x ``--width`` rectangle says the table ends, carried out from the one
    marker, and they are what the lever-arm numbers are evaluated at.
    """
    return (("origin", (0.0, 0.0)), ("+x", (length_cm, 0.0)),
            ("far", (length_cm, width_cm)), ("+y", (0.0, width_cm)))


def find_quad(corners, ids, target):
    """The first ``target``-id marker's ``(4, 2)`` corner quad, or None.

    A duplicate id -- a stray print of the same marker in frame -- keeps the first seen.
    """
    if ids is None:
        return None
    for c, i in zip(corners, ids.flatten()):
        if int(i) == target:
            return np.asarray(c, dtype=np.float64).reshape(4, 2)
    return None


def corner_points(cam, quad, patch=2):
    """``(4, 3)`` camera-frame metres for the marker's corners, or None if any misses.

    Corners rather than the centre: four samples spread over the square are what give the
    depth source a *frame* and not just a point.
    """
    if quad is None:
        return None
    pts = [cam.deproject(u, v, patch=patch) for u, v in quad]
    if any(p is None for p in pts):
        return None
    return np.asarray(pts, dtype=np.float64)


# --------------------------------------------------------------------------------------
# the table frame itself: origin at the ID-10 marker, +x along the length, +y the width
# --------------------------------------------------------------------------------------
def apply_yaw(R, yaw_offset_deg):
    """Table rotation from the marker's own, spun back by the marker's heading.

    ``yaw_offset_deg`` is where the marker's printed ``+x`` edge points, in degrees CCW
    from table ``+x``, so the table axes are the marker's rotated by ``-yaw`` about the
    ``+z`` the two share.  0 means the marker was laid down square to the table.
    """
    th = np.radians(-float(yaw_offset_deg))
    c, s = np.cos(th), np.sin(th)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return np.asarray(R, dtype=np.float64).reshape(3, 3) @ Rz


def pnp_frame(cam, quad, side_cm, yaw_offset_deg):
    """``{R, t, rms}`` -- the table pose from ``solvePnP`` on the marker alone, or None.

    ``R``'s columns are the table axes in the camera frame and ``t`` is the origin, metres.
    ``rms`` is the reprojection error of the four corners: the only internal consistency
    check a four-point planar solve has, so it catches a mis-detected corner but *not* a
    wrong ``--marker-len`` (that slides the marker along the ray and reprojects perfectly).
    """
    if quad is None:
        return None
    model = cam.pinhole()
    if model is None:
        return None
    K, dist = model
    pose = solve_marker_pose(quad, K, dist, side_cm)
    if pose is None:
        return None
    rvec, tvec = pose
    proj, _ = cv2.projectPoints(marker_object_points(side_cm), rvec, tvec, K, dist)
    rms = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - quad) ** 2, axis=1))))
    return {"source": "pnp",
            "R": apply_yaw(cv2.Rodrigues(rvec)[0], yaw_offset_deg),
            "t": np.asarray(tvec, dtype=np.float64).reshape(3),
            "rms": rms, "rms_unit": "px"}


def depth_frame(cam, quad, yaw_offset_deg, pts=None):
    """``{R, t, rms}`` -- the table pose read off the four deprojected corners, or None.

    ``+x`` is the mean of the two horizontal edges and ``+y`` the mean of the two vertical
    ones (averaging the pair cancels the depth noise on any single corner), then ``+y`` is
    orthogonalised against ``+x`` and ``+z = +x cross +y`` comes out of the table toward
    the camera -- the same handedness the pnp frame has.  ``rms`` is how far the four
    corners sit off the plane they define, in cm: the depth source's noise, directly.

    No marker size enters, so this frame's scale is the depth sensor's own.
    """
    P = corner_points(cam, quad) if pts is None else pts
    if P is None:
        return None
    ex = (P[1] - P[0]) + (P[2] - P[3])
    ey = (P[0] - P[3]) + (P[1] - P[2])
    nx = float(np.linalg.norm(ex))
    if nx < 1e-9:
        return None
    ex = ex / nx
    ey = ey - float(ey @ ex) * ex
    ny = float(np.linalg.norm(ey))
    if ny < 1e-9:
        return None
    ey = ey / ny
    ez = np.cross(ex, ey)
    t = P.mean(axis=0)
    rms = float(np.sqrt(np.mean(((P - t) @ ez) ** 2)) * 100.0)
    return {"source": "depth", "R": apply_yaw(np.column_stack([ex, ey, ez]), yaw_offset_deg),
            "t": t, "rms": rms, "rms_unit": "cm"}


def ray_to_table_cm(cam, pose, uv, height_cm):
    """Table-frame ``(x, y)`` [cm] where the ray through pixel ``uv`` meets the table.

    The table frame's ``+z`` faces the camera, so a marker sitting ``height_cm`` above the
    table is simply the plane ``z = height_cm``.  No depth and no T-marker size involved --
    the scale comes entirely from the frame, which is why this is the best of the three.
    """
    model = cam.pinhole()
    if model is None:
        return None
    K, dist = model
    n = cv2.undistortPoints(np.asarray([[uv]], dtype=np.float64), K, dist).reshape(2)
    d_b = pose["R"].T @ np.array([n[0], n[1], 1.0])
    o_b = pose["R"].T @ (-pose["t"])
    if abs(float(d_b[2])) < 1e-9:
        return None
    step = (height_cm / 100.0 - float(o_b[2])) / float(d_b[2])
    if step <= 0.0:                                    # plane is behind the camera
        return None
    b = o_b + step * d_b
    return (float(b[0] * 100.0), float(b[1] * 100.0))


def project_table(cam, pose, pts_cm, z_cm=0.0):
    """Table-frame points [cm] -> pixels, or None without a camera model."""
    model = cam.pinhole()
    if model is None:
        return None
    K, dist = model
    obj = np.array([[x / 100.0, y / 100.0, z_cm / 100.0] for x, y in pts_cm],
                   dtype=np.float64)
    rvec = cv2.Rodrigues(pose["R"])[0]
    proj, _ = cv2.projectPoints(obj, rvec, pose["t"], K, dist)
    return proj.reshape(-1, 2)


# --------------------------------------------------------------------------------------
# what one marker can and cannot tell you
# --------------------------------------------------------------------------------------
def marker_geometry(P, side_cm):
    """The marker's own measured shape from its deprojected corners, or None.

    With the rectangle gone this is the only thing left that *measures* rather than
    assumes a length, so it carries the whole scale check:

    * ``side_cm``     -- each of the four edges, and their mean.  Against ``--marker-len``
      this is the depth scale error, the single-marker stand-in for the 79 x 63 fit.
    * ``diag_ratio``  -- diagonal / (side * sqrt2).  A square seen in 3-D has both
      diagonals equal; a skewed one says the corner depths disagree.
    * ``angle_deg``   -- the interior angle at each corner, 90 when the depth is clean.
    """
    if P is None:
        return None
    sides = {(a, b): float(np.linalg.norm(P[b] - P[a]) * 100.0) for a, b, _ in EDGES}
    diags = {(a, b): float(np.linalg.norm(P[b] - P[a]) * 100.0) for a, b in DIAGONALS}
    mean_side = float(np.mean(list(sides.values())))
    angles = {}
    for k in range(4):
        u, v = P[(k - 1) % 4] - P[k], P[(k + 1) % 4] - P[k]
        cos = float(u @ v) / float(np.linalg.norm(u) * np.linalg.norm(v))
        angles[k] = float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
    return {
        "side_cm": sides,
        "mean_side_cm": mean_side,
        "diag_cm": diags,
        "diag_ratio": {k: v / (mean_side * np.sqrt(2.0)) for k, v in diags.items()},
        "angle_deg": angles,
        # what the printed marker says the depth geometry should be multiplied by
        "scale": float(side_cm / mean_side) if mean_side > 1e-9 else float("nan"),
    }


def frame_diff(poses):
    """How far the pnp and depth frames disagree, or None without both.

    No redundancy lives inside either frame -- one marker, four corners, an exact planar
    solve -- so the only honest error bar on the frame is the other frame.  ``origin_cm``
    is dominated by the metric scale (``--marker-len`` vs the depth scale) and the two
    angles by corner noise; the angles are the ones that matter, because they are what the
    lever arm multiplies out to the far side of the table.
    """
    a, b = poses.get("pnp"), poses.get("depth")
    if a is None or b is None:
        return None

    def ang(u, v):
        return float(np.degrees(np.arccos(np.clip(float(u @ v), -1.0, 1.0))))

    return {
        "origin_cm": float(np.linalg.norm(a["t"] - b["t"]) * 100.0),
        "normal_deg": ang(a["R"][:, 2], b["R"][:, 2]),
        "xaxis_deg": ang(a["R"][:, 0], b["R"][:, 0]),
    }


def tee_in_table(cam, quad, poses, side_cm, height_cm):
    """Where the T marker is on the table, all four ways, or None without a table frame.

    Each entry is ``{x_cm, y_cm, ...}`` in the table frame; any of them may be missing (no
    depth on the marker, no camera model) while the others are fine.  ``plane`` /
    ``plane_depth`` are the same pixel ray through the two different frames, so the gap
    between them is the frame's contribution to the answer rather than the T marker's.
    """
    pose = poses.get("pnp") or poses.get("depth")
    if quad is None or pose is None:
        return None
    centre = quad.mean(axis=0)
    out = {"px": [float(centre[0]), float(centre[1])], "height_cm": float(height_cm)}

    for name, p in (("plane", pose), ("plane_depth", poses.get("depth"))):
        if p is None or (name == "plane_depth" and p is pose):
            continue
        plane = ray_to_table_cm(cam, p, centre, height_cm)
        if plane is None:
            continue
        # Yaw without the marker size: drop two corners onto the plane and read the
        # marker's own +x edge (ArUco corner 0 -> corner 1) off the result.
        c0 = ray_to_table_cm(cam, p, quad[0], height_cm)
        c1 = ray_to_table_cm(cam, p, quad[1], height_cm)
        yaw = (None if c0 is None or c1 is None
               else float(np.degrees(np.arctan2(c1[1] - c0[1], c1[0] - c0[0]))))
        out[name] = {"x_cm": plane[0], "y_cm": plane[1], "yaw_deg": yaw}

    model = cam.pinhole()
    if model is not None:
        K, dist = model
        p = solve_marker_pose(quad, K, dist, side_cm)
        if p is not None:
            x, y, z = to_table_cm(pose, p[1])
            out["pnp"] = {"x_cm": x, "y_cm": y, "z_cm": z,
                          "yaw_deg": table_yaw_deg(pose, cv2.Rodrigues(p[0])[0][:, 0]),
                          "marker_len_cm": float(side_cm)}

    p = cam.deproject(*centre)
    if p is not None:
        x, y, z = to_table_cm(pose, p)
        out["depth"] = {"x_cm": x, "y_cm": y, "z_cm": z}
    return out


# ======================================================================================
# everything one frame has to say
# ======================================================================================
def measure_frame(cam, quad, side_cm, length_cm, width_cm, yaw_offset_deg=0.0,
                  tee_q=None, tee_len_cm=TEE_LEN_CM, tee_height_cm=0.0):
    """``{"sources": {source: dict}, "diff": dict, "tee": dict or None}`` -- this frame."""
    P = corner_points(cam, quad)
    poses = {"pnp": pnp_frame(cam, quad, side_cm, yaw_offset_deg),
             "depth": depth_frame(cam, quad, yaw_offset_deg, P)}
    poses = {s: p for s, p in poses.items() if p is not None}

    corners = table_corners_cm(length_cm, width_cm)
    out = {}
    for source, pose in poses.items():
        px = project_table(cam, pose, [xy for _, xy in corners])
        out[source] = {
            "source": source,
            "pose": pose,
            "origin_cam_m": [float(v) for v in pose["t"]],
            "range_cm": float(np.linalg.norm(pose["t"]) * 100.0),
            "tilt_deg": tilt_deg(pose),
            "rms": pose["rms"],
            "rms_unit": pose["rms_unit"],
            # every nominal table corner, in the camera frame and on screen -- the lever
            # arm the one marker is being asked to carry
            "corner_cam_m": {n: [float(v) for v in to_cam_m(pose, xy)] for n, xy in corners},
            "corner_range_cm": {n: float(np.linalg.norm(to_cam_m(pose, xy)) * 100.0)
                                for n, xy in corners},
            "corner_px": (None if px is None else
                          {n: [float(px[k][0]), float(px[k][1])]
                           for k, (n, _) in enumerate(corners)}),
        }
    return {"sources": out,
            "geometry": marker_geometry(P, side_cm),
            "diff": frame_diff(poses),
            "tee": tee_in_table(cam, tee_q, poses, tee_len_cm, tee_height_cm)}


# ======================================================================================
# drawing
# ======================================================================================
def _label(frame, text, pt, color, scale=0.55):
    for c, t in (((0, 0, 0), 4), (color, 1)):
        cv2.putText(frame, text, (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, c, t, cv2.LINE_AA)


def _pt(p):
    return tuple(np.round(np.asarray(p, dtype=np.float64)).astype(int))


def draw_tee(frame, quad, tee_q, tee):
    """Mark the T marker and label it with its table-frame position."""
    if tee_q is None:
        return
    c = tee_q.mean(axis=0)
    cv2.polylines(frame, [np.round(tee_q).astype(np.int32)], True, (255, 255, 0), 2,
                  cv2.LINE_AA)
    cv2.circle(frame, _pt(c), 7, (255, 255, 0), 2, cv2.LINE_AA)
    best = (tee or {}).get("plane") or (tee or {}).get("pnp") or (tee or {}).get("depth")
    if best is None:
        _label(frame, f"T id {TEE_ID}: no table frame", (c[0] + 10, c[1] + 6), (255, 255, 0))
        return
    if quad is not None:
        cv2.line(frame, _pt(quad.mean(axis=0)), _pt(c), (255, 255, 0), 1, cv2.LINE_AA)
    yaw = best.get("yaw_deg")
    text = f"T id {TEE_ID}: x={best['x_cm']:+.1f} y={best['y_cm']:+.1f} cm"
    if yaw is not None:
        text += f"  yaw={yaw:+.1f}deg"
    _label(frame, text, (c[0] + 10, c[1] + 6), (255, 255, 0), 0.6)


def draw_table(frame, cam, quad, meas, length_cm, width_cm):
    """Outline the marker, draw the frame's axes, and project the nominal table rectangle."""
    if quad is None:
        return
    cv2.polylines(frame, [np.round(quad).astype(np.int32)], True, (0, 200, 0), 2,
                  cv2.LINE_AA)
    centre = quad.mean(axis=0)
    cv2.circle(frame, _pt(centre), 6, (0, 255, 0), -1, cv2.LINE_AA)
    _label(frame, f"id {ORIGIN_ID} (origin 0,0)", (centre[0] + 9, centre[1] - 9),
           (0, 255, 0), 0.6)

    pose = (meas or {}).get("pose")
    if pose is None:
        return

    # The nominal rectangle carried out from the one marker: if the frame is right, this
    # lands on the physical table edges.
    corners = table_corners_cm(length_cm, width_cm)
    px = project_table(cam, pose, [xy for _, xy in corners])
    if px is not None and np.all(np.isfinite(px)):
        cv2.polylines(frame, [np.round(px).astype(np.int32)], True, (90, 90, 255), 1,
                      cv2.LINE_AA)
        for k, (name, (x, y)) in enumerate(corners):
            if name == "origin":
                continue
            cv2.circle(frame, _pt(px[k]), 5, (255, 0, 255), 1, cv2.LINE_AA)
            _label(frame, f"{x:+.0f},{y:+.0f}cm", (px[k][0] + 8, px[k][1] - 8),
                   (255, 0, 255))

    # The table frame's own axes, drawn out of the marker.
    ax = project_table(cam, pose, [(0.0, 0.0), (AXIS_LEN_CM, 0.0), (0.0, AXIS_LEN_CM)])
    if ax is not None and np.all(np.isfinite(ax)):
        for k, (name, colour) in enumerate((("+x", (60, 60, 255)), ("+y", (255, 200, 0))),
                                           start=1):
            cv2.arrowedLine(frame, _pt(ax[0]), _pt(ax[k]), colour, 2, cv2.LINE_AA,
                            tipLength=0.2)
            _label(frame, name, (ax[k][0] + 4, ax[k][1] - 4), colour, 0.6)


def hud_lines(out, quad, length_cm, width_cm, extra=()):
    """The per-source summary block shown on the GUI."""
    seen = "seen" if quad is not None else "NOT VISIBLE"
    lines = [f"id {ORIGIN_ID} {seen}  |  nominal table {length_cm:.1f} x {width_cm:.1f} cm"
             f"  |  s = save, q = quit"]
    by_source = out.get("sources") or {}
    for source in SOURCES:
        meas = by_source.get(source)
        if meas is None:
            lines.append(f"{source:<5s} frame  --")
            continue
        lines.append(f"{source:<5s} frame  range {meas['range_cm']:7.2f} cm  "
                     f"tilt {meas['tilt_deg']:5.2f} deg  "
                     f"rms {meas['rms']:.3f}{meas['rms_unit']}  "
                     f"far corner {meas['corner_range_cm']['far']:7.2f} cm")
    geo = out.get("geometry")
    if geo:
        sides = "/".join(f"{geo['side_cm'][(a, b)]:.2f}" for a, b, _ in EDGES)
        lines.append(f"depth marker  sides {sides} cm (mean {geo['mean_side_cm']:.2f})  "
                     f"scale {geo['scale']:.4f}")
    d = out.get("diff")
    if d:
        lines.append(f"pnp vs depth  origin {d['origin_cm']:6.2f} cm  "
                     f"normal {d['normal_deg']:5.2f} deg  +x {d['xaxis_deg']:5.2f} deg")
    tee = out.get("tee")
    if tee is None:
        lines.append(f"T id {TEE_ID}  not visible (or no table frame yet)")
    else:
        for method, _ in TEE_METHODS:
            t = tee.get(method)
            if t is None:
                lines.append(f"T {method:<11s} --")
                continue
            yaw = t.get("yaw_deg")
            lines.append(f"T {method:<11s} x {t['x_cm']:+7.2f}  y {t['y_cm']:+7.2f} cm"
                         + (f"  yaw {yaw:+7.2f} deg" if yaw is not None else ""))
    return lines + list(extra)


# ======================================================================================
# headless measurement over N frames
# ======================================================================================
def _sd_cm(cols):
    """3-D wobble [cm] of a camera-frame point sampled over the run, or None.

    ``sqrt(sdx^2 + sdy^2 + sdz^2)`` -- one number for how much a point moved frame to
    frame.  Evaluated at each nominal table corner this *is* the lever arm.
    """
    if not cols["x"]:
        return None
    return float(np.sqrt(sum(float(np.std(cols[ax])) ** 2 for ax in ("x", "y", "z")))
                 * 100.0)


def measure(cam, detect, detect_tee, n_frames, side_cm, length_cm, width_cm,
            yaw_offset_deg=0.0, tee_id=TEE_ID, tee_len_cm=TEE_LEN_CM, tee_height_cm=0.0,
            json_out=None, save_png=None, warmup=15):
    """Grab ``n_frames`` frames and report the single-marker table frame, pnp vs depth."""
    for _ in range(warmup):                 # let auto-exposure and the depth filters settle
        cam.read()
    corners = table_corners_cm(length_cm, width_cm)
    print(f"\nmeasuring over {n_frames} frame(s);  marker id {ORIGIN_ID} from {ARUCO_DICT}, "
          f"{side_cm:.2f} cm a side\ntable frame: origin at that marker, +x {length_cm:.1f} "
          f"cm along the length, +y {width_cm:.1f} cm along the width"
          + (f"\nmarker yaw offset {yaw_offset_deg:+.2f} deg\n" if yaw_offset_deg else "\n"))

    cols = {s: {"range": [], "tilt": [], "rms": [],
                "origin": {ax: [] for ax in "xyz"},
                "corner": {n: {ax: [] for ax in "xyz"} for n, _ in corners},
                "corner_range": {n: [] for n, _ in corners},
                "corner_px": {n: {"u": [], "v": []} for n, _ in corners}}
            for s in SOURCES}
    geo_cols = {"side": {(a, b): [] for a, b, _ in EDGES}, "mean_side": [],
                "diag_ratio": {d: [] for d in DIAGONALS},
                "angle": {k: [] for k in range(4)}, "scale": []}
    diff_cols = {k: [] for k in ("origin_cm", "normal_deg", "xaxis_deg")}
    tee_cols = {m: {f: [] for f in fields} for m, fields in TEE_METHODS}
    seen, tee_seen, last_frame = 0, 0, None

    for f in range(n_frames):
        ok, frame = cam.read()
        if not ok:
            continue
        last_frame = frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        c, i, _ = detect(gray)
        quad = find_quad(c, i, ORIGIN_ID)
        if quad is None:
            print(f"  frame {f:3d}: marker id {ORIGIN_ID} not detected")
            continue
        seen += 1
        tc, ti, _ = detect_tee(gray)
        out = measure_frame(cam, quad, side_cm, length_cm, width_cm, yaw_offset_deg,
                            find_quad(tc, ti, tee_id), tee_len_cm, tee_height_cm)
        by_source, geo, diff, tee = (out["sources"], out["geometry"], out["diff"],
                                     out["tee"])
        if geo:
            for key in geo_cols["side"]:
                geo_cols["side"][key].append(geo["side_cm"][key])
            for key in geo_cols["diag_ratio"]:
                geo_cols["diag_ratio"][key].append(geo["diag_ratio"][key])
            for k in range(4):
                geo_cols["angle"][k].append(geo["angle_deg"][k])
            geo_cols["mean_side"].append(geo["mean_side_cm"])
            geo_cols["scale"].append(geo["scale"])
        if diff:
            for key in diff_cols:
                diff_cols[key].append(diff[key])
        if tee is not None:
            tee_seen += 1
            for method, fields in TEE_METHODS:
                t = tee.get(method)
                if t is None:
                    continue
                for fld in fields:
                    v = t.get("yaw_deg" if fld == "yaw" else f"{fld}_cm")
                    if v is not None:
                        tee_cols[method][fld].append(v)
        texts = []
        for source in SOURCES:
            meas, col = by_source.get(source), cols[source]
            if meas is None:
                texts.append(f"{source} frame     --   ")
                continue
            col["range"].append(meas["range_cm"])
            col["tilt"].append(meas["tilt_deg"])
            col["rms"].append(meas["rms"])
            for ax, v in zip("xyz", meas["origin_cam_m"]):
                col["origin"][ax].append(v)
            for name, _ in corners:
                for ax, v in zip("xyz", meas["corner_cam_m"][name]):
                    col["corner"][name][ax].append(v)
                col["corner_range"][name].append(meas["corner_range_cm"][name])
                if meas["corner_px"]:
                    u, v = meas["corner_px"][name]
                    col["corner_px"][name]["u"].append(u)
                    col["corner_px"][name]["v"].append(v)
            texts.append(f"{source} range={meas['range_cm']:7.2f} tilt={meas['tilt_deg']:5.2f}")
        best = (tee or {}).get("plane")
        texts.append(f"T x={best['x_cm']:+7.2f} y={best['y_cm']:+7.2f}"
                     if best else "T      --        --  ")
        print(f"  frame {f:3d}: " + "  |  ".join(texts))

    st = {s: {"range": stats(cols[s]["range"]), "tilt": stats(cols[s]["tilt"]),
              "rms": stats(cols[s]["rms"]),
              "origin_sd_cm": _sd_cm(cols[s]["origin"]),
              "corner_sd_cm": {n: _sd_cm(cols[s]["corner"][n]) for n, _ in corners},
              "corner_range": {n: stats(cols[s]["corner_range"][n]) for n, _ in corners},
              "corner_px": {n: {ax: stats(cols[s]["corner_px"][n][ax]) for ax in "uv"}
                            for n, _ in corners}}
          for s in SOURCES}
    geo_st = {"side": {k: stats(v) for k, v in geo_cols["side"].items()},
              "mean_side": stats(geo_cols["mean_side"]),
              "diag_ratio": {k: stats(v) for k, v in geo_cols["diag_ratio"].items()},
              "angle": {k: stats(v) for k, v in geo_cols["angle"].items()},
              "scale": stats(geo_cols["scale"])}
    diff_st = {k: stats(v) for k, v in diff_cols.items()}
    tee_st = {m: {f: stats(v) for f, v in d.items()} for m, d in tee_cols.items()}

    print("\n" + "=" * 88)
    print(f"table frame from the single ID-{ORIGIN_ID} marker, {seen}/{n_frames} usable frames")
    print("pnp = solvePnP on the marker corners (no depth); "
          "depth = RealSense aligned depth (no marker size)")

    print("\n>> THE FRAME -- origin at the marker centre, +x along the length, "
          "+y along the width")
    print("   Everything below rests on one marker, so there is no internal redundancy:")
    print("   the residual only catches a bad corner detection, not a bad scale.\n")
    print(f"  {'':22s}{'median':>11s}      {'[min .. max]':<24s}{'spread'}")
    print("  camera -> origin range")
    for source in SOURCES:
        print(f"    {source:<18s}{fmt_stat(st[source]['range'])}")
    print("  camera tilt (0 = looking straight down)")
    for source in SOURCES:
        print(f"    {source:<18s}{fmt_stat(st[source]['tilt'], 'deg')}")
    print("  frame residual (pnp = corner reprojection, depth = corner planarity)")
    for source, unit in (("pnp", "px"), ("depth", "cm")):
        print(f"    {source:<18s}{fmt_stat(st[source]['rms'], unit, 3)}")

    print(f"\n>> LEVER ARM -- the frame's own noise carried out to each nominal table "
          f"corner")
    print(f"   The frame is fixed by one {side_cm:.2f} cm square, so an angular wobble too")
    print(f"   small to see at the marker is ~{length_cm / side_cm:.0f}x larger at the far "
          f"corner.  These are the")
    print("   frame-to-frame standard deviations of each corner's position, in cm --")
    print("   the price of running on a single marker, measured.\n")
    for name, (x, y) in corners:
        print(f"    corner {name:<7s} ({x:+6.1f}, {y:+6.1f}) cm")
        for source in SOURCES:
            sd = st[source]["corner_sd_cm"][name]
            rng = st[source]["corner_range"][name]
            print(f"      {source:<16s}" + ("     --" if sd is None else
                  f"sd {sd:6.3f} cm   range {'--' if rng is None else f'{rng[0]:7.2f} cm'}"))
    print("    (the origin row is the marker itself -- everything else is extrapolation)")

    print("\n  where those corners land on screen, px (median) -- overlay this on the")
    print("  real table edge to see whether the frame is pointing the right way")
    for source in SOURCES:
        cells = []
        for name, _ in corners:
            p = st[source]["corner_px"][name]
            cells.append(f"{name} " + ("--" if p["u"] is None else
                                       f"({p['u'][0]:.0f},{p['v'][0]:.0f})"))
        print(f"    {source:<10s}" + "  ".join(cells))

    print(f"\n>> MARKER GEOMETRY from depth -- the only thing still being measured")
    print(f"   With the four-marker rectangle gone, the marker's own square is the whole")
    print(f"   scale check: the depth source measures it, --marker-len asserts it.\n")
    for a, b, axis in EDGES:
        print(f"    side {a}-{b} ({axis})   {fmt_stat(geo_st['side'][(a, b)])}")
    print(f"    mean side       {fmt_stat(geo_st['mean_side'])}"
          + ("" if geo_st["mean_side"] is None else
             f"   {geo_st['mean_side'][0] - side_cm:+.3f} cm vs --marker-len "
             f"({100.0 * geo_st['mean_side'][0] / side_cm - 100.0:+.2f}%)"))
    for a, b in DIAGONALS:
        print(f"    diag {a}-{b} ratio  {fmt_stat(geo_st['diag_ratio'][(a, b)], '  ', 4)}"
              "   (1.0000 = square)")
    for k in range(4):
        print(f"    angle at {k}      {fmt_stat(geo_st['angle'][k], 'deg')}"
              + ("   (expect 90)" if k == 0 else ""))
    if geo_st["scale"]:
        s = geo_st["scale"][0]
        print(f"    depth scale check: the printed marker wants {s:.4f}x the depth "
              f"geometry ({100.0 * s - 100.0:+.2f}% depth-scale error)")

    print("\n>> PNP vs DEPTH -- two independent frames off the same four corners")
    print("   With no redundancy inside either one, this is the error bar.  The origin")
    print("   gap is mostly metric scale (--marker-len against the depth scale); the two")
    print("   angles are what the lever arm above multiplies.\n")
    print("    origin gap        " + fmt_stat(diff_st["origin_cm"]))
    print("    normal angle      " + fmt_stat(diff_st["normal_deg"], "deg", 3))
    print("    +x axis angle     " + fmt_stat(diff_st["xaxis_deg"], "deg", 3))
    if diff_st["xaxis_deg"]:
        drift = np.radians(diff_st["xaxis_deg"][0]) * np.hypot(length_cm, width_cm)
        print(f"    ^ {diff_st['xaxis_deg'][0]:.3f} deg of +x disagreement is {drift:.2f} cm "
              f"at the far corner")

    print(f"\n>> T MARKER -- id {tee_id} from {TEE_DICT}, position in the table frame")
    print(f"   seen in {tee_seen}/{seen} of the usable frames; "
          f"origin = ID {ORIGIN_ID}, +x along the length, +y along the width")
    print("   plane       = pixel ray x table plane from the pnp frame -- no depth, no T")
    print(f"                 marker size, plane lifted by --tee-height "
          f"{tee_height_cm:.2f} cm; quote this one")
    print("   plane_depth = the same ray through the depth-built frame")
    print(f"   pnp         = solvePnP on the T marker alone (--tee-len {tee_len_cm:.2f} cm)")
    print("   depth       = its centre deprojected through the aligned depth frame\n")
    if not tee_seen:
        print(f"    T marker id {tee_id} never seen together with a table frame")
    for fld, unit, label in (("x", "cm", "x  (along the length, 0 .. %.1f on the table)"
                              % length_cm),
                             ("y", "cm", "y  (along the width,  0 .. %.1f on the table)"
                              % width_cm),
                             ("z", "cm", "z  (height above the table plane)"),
                             ("yaw", "deg", "yaw (CCW from table +x)")):
        rows = [(m, tee_st[m].get(fld)) for m, fields in TEE_METHODS if fld in fields]
        if not any(v for _, v in rows):
            continue
        print(f"  {label}")
        for method, st_v in rows:
            print(f"    {method:<18s}{fmt_stat(st_v, unit)}")
    print("=" * 88)

    if json_out:
        payload = {
            "dict": ARUCO_DICT, "id": ORIGIN_ID, "marker_len_cm": side_cm,
            "yaw_offset_deg": yaw_offset_deg,
            "nominal_cm": {"length": length_cm, "width": width_cm,
                           "corners": {n: list(xy) for n, xy in corners}},
            "frames_requested": n_frames, "frames_used": seen,
            "frame": {s: {"range_cm": st[s]["range"], "tilt_deg": st[s]["tilt"],
                          "rms": st[s]["rms"], "origin_sd_cm": st[s]["origin_sd_cm"],
                          "corner_sd_cm": st[s]["corner_sd_cm"],
                          "corner_range_cm": st[s]["corner_range"],
                          "corner_px": st[s]["corner_px"]} for s in SOURCES},
            "marker_geometry": {
                "side_cm": {f"{a}-{b}": geo_st["side"][(a, b)] for a, b, _ in EDGES},
                "mean_side_cm": geo_st["mean_side"],
                "diag_ratio": {f"{a}-{b}": geo_st["diag_ratio"][(a, b)]
                               for a, b in DIAGONALS},
                "angle_deg": {str(k): geo_st["angle"][k] for k in range(4)},
                "scale": geo_st["scale"]},
            "pnp_vs_depth": diff_st,
            "tee": {"id": tee_id, "dict": TEE_DICT, "marker_len_cm": tee_len_cm,
                    "height_cm": tee_height_cm, "frames_seen": tee_seen,
                    "table_cm": tee_st},
            "stat_format": ["median", "min", "max", "std"],
        }
        with open(json_out, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"wrote {json_out}")
    if save_png and last_frame is not None:
        cv2.imwrite(save_png, last_frame)
        print(f"wrote {save_png}")
    return seen


# ======================================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--serial", default=None, help="RealSense serial (default: first found)")
    ap.add_argument("--dict", default=ARUCO_DICT, metavar="NAME",
                    help="ArUco dictionary the corner marker is from "
                         "(default %(default)s); one of:\n  " + "  ".join(DICTS))
    ap.add_argument("--width-px", type=int, default=PARAMS.camera_width, metavar="PX",
                    help="colour stream width (default %(default)s)")
    ap.add_argument("--height-px", type=int, default=PARAMS.camera_height, metavar="PX",
                    help="colour stream height (default %(default)s)")
    ap.add_argument("--marker-len", type=float, default=MARKER_LEN_CM, metavar="CM",
                    help="physical side length of the corner marker, cm -- the solvePnP "
                         "frame's only metric scale (default %(default)s)")
    ap.add_argument("--length", type=float, default=LENGTH_CM, metavar="CM",
                    help="nominal table extent along +x, from the marker "
                         "(default %(default)s)")
    ap.add_argument("--width", type=float, default=WIDTH_CM, metavar="CM",
                    help="nominal table extent along +y, from the marker "
                         "(default %(default)s)")
    ap.add_argument("--yaw-offset", type=float, default=YAW_OFFSET_DEG, metavar="DEG",
                    help="heading of the marker's own +x edge, deg CCW from table +x; the "
                         "table axes are the marker's spun back by this (default %(default)s)")
    # One id at a time here on purpose: this is the diagnostic that shows every placement
    # method for ONE marker.  locate_functions.py is what falls back across all four.
    ap.add_argument("--tee-id", type=int, default=TEE_ID, metavar="ID",
                    help="id of the T-shape marker (default %(default)s)")
    ap.add_argument("--tee-dict", default=TEE_DICT, metavar="NAME",
                    help="dictionary the T marker is from -- a different one from the "
                         "corner marker (default %(default)s)")
    ap.add_argument("--tee-len", type=float, default=TEE_LEN_CM, metavar="CM",
                    help="physical side length of the T marker, cm; only its pnp estimate "
                         "depends on this (default %(default)s)")
    ap.add_argument("--tee-height", type=float, default=0.0, metavar="CM",
                    help="how far the T marker sits above the table plane, cm -- lifts the "
                         "plane the ray is intersected with (default %(default)s)")
    ap.add_argument("--measure", type=int, default=0, metavar="N",
                    help="headless: read N frames, report both frames, then exit (no GUI)")
    ap.add_argument("--measure-json", default=None, metavar="PATH",
                    help="with --measure, also write the summary here as JSON")
    ap.add_argument("--measure-png", default=None, metavar="PATH",
                    help="with --measure, also write the last frame here")
    args = ap.parse_args()

    if not realsense_present():
        raise SystemExit("no RealSense found -- the depth estimate needs one "
                         "(and pnp needs the colour stream).")

    detect = make_detector(args.dict)
    detect_tee = make_detector(args.tee_dict)
    cam = RealSenseCamera(args.serial, args.width_px, args.height_px)
    print(cam.desc)
    print(f"table marker: id {ORIGIN_ID} from {args.dict};  "
          f"T marker: id {args.tee_id} from {args.tee_dict}")

    if args.measure:
        try:
            seen = measure(cam, detect, detect_tee, args.measure, args.marker_len,
                           args.length, args.width, args.yaw_offset, args.tee_id,
                           args.tee_len, args.tee_height, args.measure_json,
                           args.measure_png)
        finally:
            cam.close()
        raise SystemExit(0 if seen else 1)

    try:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    except cv2.error as exc:
        cam.close()
        raise SystemExit(f"cv2 has no GUI support ({exc}); need $DISPLAY + /tmp/.X11-unix.")

    n_saved = 0
    message, msg_until = "", 0.0
    fps, last = 0.0, time.time()
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
            tee_corners, tee_ids, _ = detect_tee(gray)
            quad = find_quad(corners, ids, ORIGIN_ID)
            tee_q = find_quad(tee_corners, tee_ids, args.tee_id)
            out = measure_frame(cam, quad, args.marker_len, args.length, args.width,
                                args.yaw_offset, tee_q, args.tee_len, args.tee_height)
            draw_table(frame, cam, quad,
                       out["sources"].get("pnp") or out["sources"].get("depth"),
                       args.length, args.width)
            draw_tee(frame, quad, tee_q, out["tee"])

            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
            last = now
            extra = [message] if message and now < msg_until else []
            lines = hud_lines(out, quad, args.length, args.width, extra)
            lines[0] = f"{fps:4.1f} fps  |  " + lines[0]
            hud(frame, lines)

            cv2.imshow(WINDOW, frame)
            if cam.depth_vis is not None:
                cv2.imshow(DEPTH_WINDOW, cam.depth_vis)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("s"):
                path = f"table_id10_{n_saved}.png"
                cv2.imwrite(path, frame)
                n_saved += 1
                message, msg_until = f"saved {path}", now + 4.0
                print(message)
    finally:
        cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
