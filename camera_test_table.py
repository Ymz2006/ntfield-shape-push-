"""camera_test_table -- measure the table rectangle from its four corner ArUco markers.

Four **DICT_ARUCO_ORIGINAL** markers, ids **0, 10, 20, 30**, sit flat on the four corners
of the table.  Walking around the rectangle they go ``20 -> 30 -> 0 -> 10 -> 20``:

    id 20 ....... length ....... id 30       table frame, origin at the **ID-20** marker:
     ^ origin (0, 0)              |             +x runs 20 -> 30, along the length
    width                       width           -y runs 20 -> 10, along the width
      |                           |             so the table occupies x >= 0, y <= 0
    id 10 ....... length ....... id 0           and ID 0 is the far corner (+L, -W)

so ``0-10`` and ``20-30`` are the two lengths, ``0-30`` and ``10-20`` the two widths.
Going along the width from the origin is **-y**, so ID 10 is at ``(0, -W)``, ID 30 at
``(+L, 0)`` and ID 0 at ``(+L, -W)`` -- the same "+x right, +y up, table below the
origin" convention ``camera_test.py`` uses.
Centre to centre the rectangle is ``--length`` x ``--width`` cm (default 79 x 63) and each
marker is ``--marker-len`` cm on a side (default 5.5).

Two independent **sources** of 3-D geometry are used, and everything is reported for both:

* **pnp**   -- ``solvePnP`` on marker corners.  No depth at all; the printed marker size is
  the only metric scale, so a wrong ``--marker-len`` scales every length by that ratio.
* **depth** -- the aligned RealSense depth frame, deprojected.  Independent of the marker
  size; carries the depth sensor's own scale and noise instead.

and each source measures the rectangle two ways:

* **sides**  -- the four marker centres placed independently, then differenced pairwise.
  Every side is one measurement off two points, so it inherits both points' noise, and
  the pnp version leans on the 5.5 cm marker as its whole baseline.
* **joint**  -- *one* fit of the whole board, which is where the known layout pays off.
  The four markers are not four free points: they are coplanar, they sit on a rectangle
  (right angles, opposite sides equal), and each is a square of known side.  So the only
  unknowns are the board's pose (6), each marker's own yaw on the table (4), and the two
  dimensions **L** and **W** -- 12 parameters against 32 measurements (16 corners x 2).
  Levenberg-Marquardt drives all 16 corners at once, so the baseline is the whole 79 cm
  board rather than one 5.5 cm marker, and the rectangle constraint folds four
  measurements of each dimension into one.  ``ratio`` = L/W is better still: it is a pure
  shape number, so for **pnp** it does not depend on the marker size at all.

There is also a **nominal fit**: a 4-point Procrustes onto the ``--length x --width``
rectangle you typed in.  Unlike the two above it *assumes* the answer, so it is a check,
not a measurement -- ``scale`` is the metric error of that source (1.000 = agrees),
``rms`` is how un-rectangular the four centres landed, ``tilt`` is the angle between the
table normal and the camera axis (0 = looking straight down).

A separate **T-shape marker -- id 1, DICT_4X4_50** (``--tee-id`` / ``--tee-dict``) is
tracked whenever it is in view and its position is reported in that table frame, three
ways:

* **plane** -- its pixel ray intersected with the table plane taken from the joint board
  fit.  Needs neither the T marker's physical size nor depth, and the plane comes from
  all 16 board corners, so this is the one to quote.  ``--tee-height`` lifts the plane if
  the marker sits on top of something thick.
* **pnp**   -- ``solvePnP`` on the T marker alone, scaled by ``--tee-len``.
* **depth** -- its centre deprojected through the aligned depth frame.

``yaw`` is the marker's own rotation on the table, degrees CCW from table ``+x``.

Colour frames are undistorted with ``calibration/camera/intrinsics.json`` when it exists
(run ``calibrate_camera.py`` to make one); without it the RealSense factory intrinsics are
used and every method gets noticeably worse.

    sudo docker run \\
      --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \\
      --volume="/home/jeffrey/ntrlshape_arm:/workspace" \\
      --privileged --volume="/dev:/dev" --network=host \\
      --runtime=nvidia -ti --rm ntrlshapelocal
    python camera_test_table.py                      # live window
    python camera_test_table.py --measure 30         # headless: report both methods, exit
    python camera_test_table.py --length 79 --width 63 --marker-len 5.5

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
    plane_coords,
    procrustes_2d,
    realsense_present,
    solve_marker_pose,
)
from real_world_params import PARAMS

WINDOW = "table rectangle -- 4 aruco corners"
DEPTH_WINDOW = "table rectangle -- depth"

ARUCO_DICT = "DICT_ARUCO_ORIGINAL"
ORIGIN_ID = 20                     # the table-frame origin, (0, 0)
XAXIS_ID = 30                      # +x from the origin -- the length
YAXIS_ID = 10                      # +y from the origin -- the width
FAR_ID = 0                         # diagonally opposite the origin
CORNER_IDS = (ORIGIN_ID, XAXIS_ID, FAR_ID, YAXIS_ID)   # in order around the rectangle
# Going along the width, away from the origin, is -y -- so the +y axis points *away* from
# the ID-10 marker and the whole table sits at y <= 0.
YAXIS_SIGN = -1.0                  # ID 10 is at y = YAXIS_SIGN * width
MARKER_LEN_CM = 5.5
LENGTH_CM = 79.0                   # 0-10 and 20-30, centre to centre
WIDTH_CM = 63.0                    # 0-30 and 10-20, centre to centre

TEE_DICT = "DICT_4X4_50"           # the T marker is from a different dictionary
TEE_ID = 1
TEE_LEN_CM = 5.5                   # only the pnp estimate of the T depends on this

# The named sides, as (id_a, id_b, kind).  ``kind`` picks which nominal each is compared to.
EDGES = ((0, 10, "length"), (20, 30, "length"), (0, 30, "width"), (10, 20, "width"))
DIAGONALS = ((0, 20), (10, 30))
SOURCES = ("pnp", "depth")
FRAME_IDS = (ORIGIN_ID, XAXIS_ID, YAXIS_ID)   # the ids the table frame is built from


# ======================================================================================
# the four markers
# ======================================================================================
def rect_model_cm(length_cm, width_cm):
    """The four marker centres in the table frame, cm, keyed by id.

    Origin at ID 20, ``+x`` along the length toward ID 30, and ``-y`` along the width
    toward ID 10 -- the frame the whole program reports in.
    """
    w = YAXIS_SIGN * width_cm
    return {ORIGIN_ID: (0.0, 0.0), XAXIS_ID: (length_cm, 0.0),
            FAR_ID: (length_cm, w), YAXIS_ID: (0.0, w)}


def corner_quads(corners, ids):
    """``{id: (4, 2) corner quad}`` for whichever of the four corner markers are in view.

    Each id is unique here (unlike the all-ID-0 field markers ``camera_test.py`` uses), so
    the markers identify themselves and no geometric disambiguation is needed.  A
    duplicate id -- a stray print of the same marker in frame -- keeps the first one seen.
    """
    out = {}
    if ids is None:
        return out
    for quad, i in zip(corners, ids.flatten()):
        i = int(i)
        if i in CORNER_IDS and i not in out:
            out[i] = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    return out


def marker_poses(cam, quads, side_cm):
    """``{id: (rvec, tvec)}`` from a per-marker ``solvePnP``, one marker at a time.

    ``tvec`` is the marker centre in the camera frame, metres.  This is the *independent*
    solve -- each marker's 5.5 cm square is the entire baseline for its own range, which
    is exactly the weakness ``joint_fit`` removes.
    """
    model = cam.pinhole()
    if model is None:
        return {}
    K, dist = model
    out = {}
    for i, quad in quads.items():
        pose = solve_marker_pose(quad, K, dist, side_cm)
        if pose is not None:
            out[i] = (np.asarray(pose[0], dtype=np.float64),
                      np.asarray(pose[1], dtype=np.float64))
    return out


def depth_points(cam, quads, patch=3):
    """``{id: [X, Y, Z] m}`` for the marker centres, deprojected through aligned depth."""
    out = {}
    for i, quad in quads.items():
        p = cam.deproject(*quad.mean(axis=0), patch=patch)
        if p is not None:
            out[i] = np.asarray(p, dtype=np.float64)
    return out


def depth_corner_points(cam, quads, patch=2):
    """``{id: (4, 3) or None}`` -- every marker corner deprojected, ``None`` if any misses.

    The joint depth fit wants corners, not just centres: 16 depth samples spread over the
    board constrain the plane far better than 4 do.
    """
    out = {}
    for i, quad in quads.items():
        pts = [cam.deproject(u, v, patch=patch) for u, v in quad]
        out[i] = (None if any(p is None for p in pts)
                  else np.asarray(pts, dtype=np.float64))
    return out


def spans_cm(pts):
    """Centre-to-centre distance [cm] of every named side and diagonal we have both ends of."""
    out = {}
    for a, b, _ in EDGES:
        if a in pts and b in pts:
            out[(a, b)] = float(np.linalg.norm(pts[b] - pts[a]) * 100.0)
    for a, b in DIAGONALS:
        if a in pts and b in pts:
            out[(a, b)] = float(np.linalg.norm(pts[b] - pts[a]) * 100.0)
    return out


# --------------------------------------------------------------------------------------
# the table frame itself: origin at ID 0, +x toward ID 10, +y toward ID 30
# --------------------------------------------------------------------------------------
def table_basis(pts):
    """``(origin, ex, ey, ez)`` of the table frame from the marker centres, or None.

    ``+x`` is the ORIGIN -> XAXIS direction exactly; ``+y`` is ``YAXIS_SIGN`` times
    ORIGIN -> YAXIS with the ``+x`` component projected out, so the axes come out
    orthonormal even when the corners are a little off square -- and, with
    ``YAXIS_SIGN = -1``, ``+y`` points away from the table.  ``ez = ex x ey`` then points
    out of the table, toward the camera or away from it depending on which way round the
    four ids were laid down.
    """
    if not set(FRAME_IDS) <= set(pts):
        return None
    o = pts[ORIGIN_ID]
    ex = pts[XAXIS_ID] - o
    nx = float(np.linalg.norm(ex))
    if nx < 1e-9:
        return None
    ex = ex / nx
    ey = YAXIS_SIGN * (pts[YAXIS_ID] - o)
    ey = ey - float(ey @ ex) * ex
    ny = float(np.linalg.norm(ey))
    if ny < 1e-9:
        return None
    ey = ey / ny
    return o, ex, ey, np.cross(ex, ey)


def table_coords_cm(pts):
    """Each marker's ``(x, y, z)`` in the table frame, cm, or None without ``FRAME_IDS``.

    ``z`` is the out-of-plane residual.  It is 0 for the three ``FRAME_IDS`` markers by
    construction, so only ID 0's ``z`` says anything: how far the fourth corner sits out
    of the plane the other three define.
    """
    basis = table_basis(pts)
    if basis is None:
        return None
    o, ex, ey, ez = basis
    return {i: tuple(float(v * 100.0) for v in ((p - o) @ ex, (p - o) @ ey, (p - o) @ ez))
            for i, p in sorted(pts.items())}


def corner_angles_deg(pts):
    """Interior angle [deg] at each corner marker whose two neighbours are also in view."""
    ring = list(CORNER_IDS)
    out = {}
    for k, i in enumerate(ring):
        a, b = ring[k - 1], ring[(k + 1) % 4]
        if not {a, i, b} <= set(pts):
            continue
        u, v = pts[a] - pts[i], pts[b] - pts[i]
        cos = float(u @ v) / float(np.linalg.norm(u) * np.linalg.norm(v))
        out[i] = float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
    return out


def nominal_fit(pts, length_cm, width_cm):
    """Procrustes the four measured centres onto the rectangle *you typed in*, or None.

    A check rather than a measurement -- it assumes the nominal dimensions and reports how
    far off the source is:

    * ``scale`` -- what the nominal rectangle says this source's geometry should be
      multiplied by.  For ``pnp`` that is the error in ``--marker-len`` (or in the focal
      length); for ``depth`` it is the depth scale.
    * ``rms``   -- residual after scale, rotation and translation are absorbed, i.e. how
      un-rectangular the four measured centres are.

    ``procrustes_2d`` excludes reflections, and whether ``0 -> 10 -> 20 -> 30`` runs
    clockwise or anticlockwise in the image depends on how the markers were laid out, so
    both chiralities are fitted and the better one is kept (``mirrored``).
    """
    if not set(CORNER_IDS) <= set(pts):
        return None
    P = np.asarray([pts[i] for i in CORNER_IDS], dtype=np.float64)
    uv, normal = plane_coords(P)
    uv_cm = uv * 100.0
    model = rect_model_cm(length_cm, width_cm)
    D = np.asarray([model[i] for i in CORNER_IDS], dtype=np.float64)

    best = None
    for mirrored in (False, True):
        scale, _, _, rms = procrustes_2d(uv_cm, D * ([1.0, -1.0] if mirrored else [1.0, 1.0]))
        if best is None or rms < best[0]:
            best = (rms, scale, mirrored)
    rms, scale, mirrored = best
    return {
        "scale": float(scale),
        "rms_cm": float(rms),
        "mirrored": bool(mirrored),
        "tilt_deg": float(np.degrees(np.arccos(np.clip(-normal[2], -1.0, 1.0)))),
    }


# ======================================================================================
# the joint fit: solve the whole board at once, with only L and W free
# ======================================================================================
# Four markers on a rectangle are not 12 free numbers.  Up to the board's rigid pose they
# are two: the length and the width.  So instead of placing each marker independently and
# differencing, fit one model to every corner at once --
#
#   parameters   rvec, tvec        board -> camera pose            6
#                L, W              the two dimensions we want      2
#                yaw_i             each marker's own spin on the   4
#                                  table (they are glued down by
#                                  hand, not aligned to the axes)
#   measurements 16 marker corners                              32 (pnp, pixels)
#                                                            or 48 (depth, metres)
#
# and the rectangle constraint -- right angles, opposite sides equal, all four coplanar --
# is baked into the model rather than checked afterwards.  Two wins over differencing
# centres: every corner of every marker constrains L and W (so the baseline is the whole
# board, not one 5.5 cm marker), and the four redundant side measurements are averaged
# optimally instead of reported separately.
#
# ``ratio = L / W`` is the strongest number of all for the pnp source: it is scale-free,
# so unlike L and W themselves it does not depend on ``--marker-len`` being right.
# --------------------------------------------------------------------------------------
def _numeric_jacobian(residual, x, r0):
    """Forward-difference Jacobian.  numpy only -- scipy is not in the image."""
    J = np.empty((r0.size, x.size), dtype=np.float64)
    for k in range(x.size):
        step = 1e-6 * max(1.0, abs(float(x[k])))
        xk = x.copy()
        xk[k] += step
        J[:, k] = (residual(xk) - r0) / step
    return J


def levmar(residual, x0, iters=40, lam=1e-3, tol=1e-10):
    """Levenberg-Marquardt least squares: ``(x, rms)``.

    Small enough problem (12 parameters, ~40 residuals) that a numeric Jacobian and a
    dense solve are cheaper than any dependency, and the initial guess comes from the
    per-marker solves so it converges in a handful of iterations.
    """
    x = np.asarray(x0, dtype=np.float64).copy()
    r = residual(x)
    cost = float(r @ r)
    for _ in range(iters):
        J = _numeric_jacobian(residual, x, r)
        JtJ = J.T @ J
        Jtr = J.T @ r
        stepped = False
        for _ in range(10):
            damp = lam * np.diag(np.maximum(np.diag(JtJ), 1e-12))
            try:
                dx = np.linalg.solve(JtJ + damp, -Jtr)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            r_new = residual(x + dx)
            cost_new = float(r_new @ r_new)
            if cost_new < cost:
                gain = cost - cost_new
                x, r, cost = x + dx, r_new, cost_new
                lam = max(lam * 0.3, 1e-9)
                stepped = True
                break
            lam *= 10.0
        if not stepped or gain <= tol * max(cost, 1e-12):
            break
    return x, float(np.sqrt(cost / max(r.size, 1)))


def board_points(L, W, chi, yaws, ids, side_m):
    """``(4n, 3)`` model corner points in the board frame, metres, in ``ids`` order.

    The board frame is right-handed with ``+z`` out of the table toward the camera, so it
    shares the marker frame's handedness and ArUco's corner order carries over unchanged.
    ``chi`` is +1 or -1 -- which way round the ids run when seen from the camera, fixed at
    initialisation and not optimised.
    """
    return marker_grid({ORIGIN_ID: (0.0, 0.0), XAXIS_ID: (L, 0.0),
                        FAR_ID: (L, chi * W), YAXIS_ID: (0.0, chi * W)},
                       yaws, ids, side_m)


def marker_grid(centres, yaws, ids, side_m):
    """``(4n, 3)`` board-frame corner points from explicit marker centres and yaws.

    Shared by both fits: the rectangle fit feeds it centres derived from ``L`` and ``W``,
    the free fit feeds it centres that are themselves free parameters.
    """
    h = side_m / 2.0
    base = np.array([[-h, h], [h, h], [h, -h], [-h, -h]], dtype=np.float64)  # ArUco order
    out = []
    for i, yaw in zip(ids, yaws):
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s], [s, c]], dtype=np.float64)
        xy = (R @ base.T).T + np.asarray(centres[i], dtype=np.float64)
        out.append(np.column_stack([xy, np.zeros(4)]))
    return np.vstack(out)


def board_init(pts, poses, ids):
    """Initial ``(rvec, tvec, L, W, chi, yaws)`` for the joint fit, or None.

    Built from the independent per-marker solves: the table basis gives the pose, the two
    spans from the origin marker give L and W, and each marker's own rotation gives its
    yaw.  It only has to be close enough for LM to take over.
    """
    basis = table_basis(pts)
    if basis is None:
        return None
    o, ex, ey, ez = basis
    if float(ez @ o) > 0.0:          # camera sits at the origin: make +z face it
        ey, ez = -ey, -ez
    chi = 1.0 if float((pts[YAXIS_ID] - o) @ ey) > 0.0 else -1.0
    R_bc = np.column_stack([ex, ey, ez])
    rvec = cv2.Rodrigues(R_bc)[0].reshape(3)
    L = float(np.linalg.norm(pts[XAXIS_ID] - o))
    W = abs(float((pts[YAXIS_ID] - o) @ ey))

    yaws = []
    for i in ids:
        if i in poses:
            # the marker's own x axis, expressed in the board frame
            v = R_bc.T @ cv2.Rodrigues(poses[i][0])[0][:, 0]
            yaws.append(float(np.arctan2(v[1], v[0])))
        else:
            yaws.append(0.0)
    return rvec, o.copy(), L, W, chi, np.asarray(yaws, dtype=np.float64)


def _unpack(x, n):
    return x[:3], x[3:6], float(x[6]), float(x[7]), x[8:8 + n]


def residual_maker(cam, quads, ids, source, corner_depth=None):
    """``(make, unit)``, or None.  ``make(points_of)`` -> an LM residual over ``x``.

    ``points_of(x)`` builds the ``(4n, 3)`` board-frame model for a parameter vector whose
    first six entries are always ``rvec, tvec``.  Both fits share this so they are scored
    on exactly the same measurements -- reprojection error in pixels for ``"pnp"``, 3-D
    distance in cm for ``"depth"`` -- and only their model differs.
    """
    if source == "pnp":
        model = cam.pinhole()
        if model is None:
            return None
        K, dist = model
        obs = np.vstack([quads[i] for i in ids])

        def make(points_of):
            def residual(x):
                proj, _ = cv2.projectPoints(points_of(x), x[:3], x[3:6], K, dist)
                return (proj.reshape(-1, 2) - obs).ravel()
            return residual

        return make, "px"

    cd = corner_depth if corner_depth is not None else depth_corner_points(cam, quads)
    rows, obs = [], []
    for k, i in enumerate(ids):
        if cd.get(i) is None:
            continue
        rows.extend(range(4 * k, 4 * k + 4))
        obs.append(cd[i])
    if len(obs) < 3:                          # fewer than 3 markers cannot pin the board
        return None
    rows, obs = np.asarray(rows, dtype=int), np.vstack(obs)

    def make(points_of):
        def residual(x):
            R = cv2.Rodrigues(x[:3])[0]
            return (((R @ points_of(x)[rows].T).T + x[3:6] - obs) * 100.0).ravel()
        return residual

    return make, "cm"


def board_table_cm(pose_k, centres_b):
    """Board-frame marker centres [m] -> table-frame ``{id: (x, y)}`` [cm]."""
    return {i: (float(c[0] * 100.0), float(pose_k * c[1] * 100.0))
            for i, c in centres_b.items()}


def spans_from_table(pos_cm):
    """Side and diagonal lengths [cm] from 2-D table-frame marker positions."""
    out = {}
    for a, b in [(a, b) for a, b, _ in EDGES] + list(DIAGONALS):
        if a in pos_cm and b in pos_cm:
            out[(a, b)] = float(np.hypot(pos_cm[b][0] - pos_cm[a][0],
                                         pos_cm[b][1] - pos_cm[a][1]))
    return out


def rotation_average(Rs):
    """The Frobenius-mean rotation: the arithmetic mean projected back onto SO(3).

    Rotation matrices do not form a vector space, so a plain element-wise mean is not a
    rotation.  The SVD projection is the closest one that is, and for a tight cluster of
    estimates -- which four views of one flat board are -- it matches the geodesic mean.
    """
    U, _, Vt = np.linalg.svd(np.mean(np.stack(Rs), axis=0))
    d = float(np.sign(np.linalg.det(U @ Vt)))
    return U @ np.diag([1.0, 1.0, d]) @ Vt


def pose_average_fit(cam, quads, poses, init_pts, length_cm, width_cm):
    """Solve each marker's pose separately, then average the four into one board pose.

    Each marker already gives a full 6-DoF pose, so with the layout known each one is an
    independent vote on where the whole board is.  Two things have to happen before those
    votes can be averaged, though:

    * the markers are glued down by hand at arbitrary yaws, so ``R_i`` is not the board's
      orientation.  Each is de-yawed first -- the yaw is read off against a provisional
      board frame -- leaving four estimates of the same board orientation to average;
    * a pose at the marker is not a pose at the origin, so each is walked back to ID %d
      through that marker's *nominal* position in the ID-20 frame, which is why this
      method (like the nominal fit) assumes ``length_cm`` x ``width_cm`` rather than
      measuring it.

    What it buys is a better **frame**: the origin comes from four votes instead of one, so
    its noise drops by about a factor of two, and the plane normal likewise.  What it does
    *not* buy is better inter-marker geometry -- the spans it reports are identical to the
    independent ones, because the distance between two points does not care what frame you
    express it in.  ``origin_spread_cm`` and ``normal_spread_deg``, how far the four votes
    disagree, are the diagnostic this method really adds.

    pnp only: the depth source gives no per-marker rotation to average.
    """
    ids = [i for i in CORNER_IDS if i in quads and i in poses]
    if len(ids) < 2:
        return None
    init = board_init(init_pts, poses, ids)
    if init is None:
        return None
    rvec0, _, _, _, chi, _ = init
    R0 = cv2.Rodrigues(rvec0)[0]
    k = YAXIS_SIGN * chi
    model = rect_model_cm(length_cm, width_cm)

    Rs, origins, yaws = [], [], {}
    for i in ids:
        R_i = cv2.Rodrigues(poses[i][0])[0]
        v = R0.T @ R_i[:, 0]                       # the marker's own x axis, board frame
        yaw = float(np.arctan2(v[1], v[0]))
        yaws[i] = yaw
        c, sn = np.cos(-yaw), np.sin(-yaw)
        R_b = R_i @ np.array([[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]])
        mx, my = model[i]
        Rs.append(R_b)
        origins.append(poses[i][1] - R_b @ np.array([mx / 100.0, k * my / 100.0, 0.0]))

    R_avg = rotation_average(Rs)
    t_avg = np.mean(np.stack(origins), axis=0)
    n_avg = R_avg[:, 2]
    pose = board_pose(R_avg, t_avg, chi)
    table = {i: to_table_cm(pose, poses[i][1])[:2] for i in ids}
    return {
        "source": "pnp",
        "ids": ids,
        "pose": pose,
        "table_cm": table,
        "spans_cm": spans_from_table(table),
        "yaw_deg": {i: float(np.degrees(y)) for i, y in yaws.items()},
        # how far the four independent votes disagree -- the point of the method
        "origin_spread_cm": float(np.sqrt(np.mean(
            [float(np.sum((t - t_avg) ** 2)) for t in origins])) * 100.0),
        "normal_spread_deg": float(np.sqrt(np.mean(
            [np.degrees(np.arccos(np.clip(float(R[:, 2] @ n_avg), -1.0, 1.0))) ** 2
             for R in Rs]))),
        "origin_range_cm": float(np.linalg.norm(t_avg) * 100.0),
        "tilt_deg": float(np.degrees(np.arccos(np.clip(abs(float(n_avg[2])), -1.0, 1.0)))),
    }


def free_fit(cam, quads, poses, init_pts, side_cm, source, corner_depth=None):
    """Where each marker actually is: one board fit with the positions left free.

    This is the honest answer to "where are the four markers", and the middle rung between
    the other two:

    * **independent** -- each marker solved alone.  For ``pnp`` that means a 5.5 cm
      baseline per marker, so its range (and hence its position) is the weakest number in
      the program; for ``depth`` it is one noisy depth sample per centre.
    * **free** (this) -- *one* pose and *one* plane for the whole board, fitted to all 16
      corners at once, with every marker's ``(x, y)`` and yaw free.  It uses the structure
      that is genuinely known -- the markers are coplanar, each is a square of known side,
      and one rigid transform relates them all to the camera -- without assuming the thing
      being measured.  So the positions it reports are measurements, and they are much
      better conditioned than the independent ones.
    * **joint** -- the rectangle imposed, ``L`` and ``W`` the only free dimensions.  Its
      marker positions are ``(0,0), (L,0), (L,-W), (0,-W)`` *by construction*, so they
      would carry no information; only ``L`` and ``W`` come out of it.

    The gauge is the reported table frame itself: ID 20 pinned at ``(0, 0)`` and ID 30
    pinned to ``y = 0``, which fixes the three degrees of freedom a free plane would
    otherwise slide along.  With four markers that leaves 15 parameters against 32
    measurements.
    """
    ids = [i for i in CORNER_IDS if i in quads]
    if not set(FRAME_IDS) <= set(ids):
        return None
    init = board_init(init_pts, poses, ids)
    if init is None:
        return None
    rvec0, tvec0, L0, W0, chi0, yaws0 = init
    made = residual_maker(cam, quads, ids, source, corner_depth)
    if made is None:
        return None
    make, unit = made

    side_m = side_cm / 100.0
    n = len(ids)
    free = [i for i in ids if i not in (ORIGIN_ID, XAXIS_ID)]

    def centres_of(x):
        c = {ORIGIN_ID: (0.0, 0.0), XAXIS_ID: (float(x[6]), 0.0)}
        for j, i in enumerate(free):
            c[i] = (float(x[7 + 2 * j]), float(x[8 + 2 * j]))
        return c

    def points_of(x):
        return marker_grid(centres_of(x), x[7 + 2 * len(free):], ids, side_m)

    # Seed the positions from the independent solves rather than from the rectangle, so
    # the fit is not nudged toward the answer it is meant to be checking.
    R_bc = cv2.Rodrigues(rvec0)[0]
    model0 = rect_model_cm(L0 * 100.0, W0 * 100.0)
    pos0 = []
    for i in free:
        if i in init_pts:
            b = R_bc.T @ (init_pts[i] - tvec0)
            pos0.extend([float(b[0]), float(b[1])])
        else:
            mx, my = model0[i]
            pos0.extend([mx / 100.0, chi0 * abs(my) / 100.0 * np.sign(my or -1.0)])
    x0 = np.concatenate([rvec0, tvec0, [L0], pos0, yaws0])
    x, rms = levmar(make(points_of), x0)

    centres = centres_of(x)
    yaws = x[7 + 2 * len(free):]
    chi = 1.0 if centres[YAXIS_ID][1] > 0.0 else -1.0
    pose = board_pose(cv2.Rodrigues(x[:3])[0], x[3:6], chi)
    table = board_table_cm(pose["k"], centres)
    return {
        "source": source,
        "ids": ids,
        "pose": pose,
        "table_cm": table,
        "spans_cm": spans_from_table(table),
        "yaw_deg": {i: float(np.degrees(y)) for i, y in zip(ids, yaws)},
        "rms": float(rms),
        "rms_unit": unit,
    }


def joint_fit(cam, quads, poses, init_pts, side_cm, source, corner_depth=None):
    """One fit of the whole board -> ``L`` and ``W`` in cm, or None.

    Needs ids 20, 30 and 10 (the origin and both axes) to start from; ID 0 joins in when
    it is in view, which is the usual case and what makes the fit over-determined.
    ``source`` picks what the model is fitted *to*: reprojection error in pixels
    (``"pnp"``, no depth involved) or 3-D distance in cm (``"depth"``, no marker size
    involved beyond the corner layout).
    """
    ids = [i for i in CORNER_IDS if i in quads]
    if not set(FRAME_IDS) <= set(ids):
        return None
    init = board_init(init_pts, poses, ids)
    if init is None:
        return None
    rvec0, tvec0, L0, W0, chi, yaws0 = init
    made = residual_maker(cam, quads, ids, source, corner_depth)
    if made is None:
        return None
    make, unit = made
    side_m = side_cm / 100.0
    n = len(ids)

    def points_of(x):
        _, _, L, W, yaws = _unpack(x, n)
        return board_points(L, W, chi, yaws, ids, side_m)

    x0 = np.concatenate([rvec0, tvec0, [L0, W0], yaws0])
    x, rms = levmar(make(points_of), x0)
    rvec, tvec, L, W, yaws = _unpack(x, n)
    L, W = abs(L) * 100.0, abs(W) * 100.0
    R = cv2.Rodrigues(rvec)[0]
    normal = R[:, 2]
    return {
        "source": source,
        "ids": ids,
        "pose": board_pose(R, tvec, chi),
        "length_cm": float(L),
        "width_cm": float(W),
        "ratio": float(L / W) if W > 1e-9 else float("nan"),
        "rms": float(rms),
        "rms_unit": unit,
        "yaw_deg": {i: float(np.degrees(y)) for i, y in zip(ids, yaws)},
        "origin_range_cm": float(np.linalg.norm(tvec) * 100.0),
        "tilt_deg": float(np.degrees(np.arccos(np.clip(abs(float(normal[2])), -1.0, 1.0)))),
    }


# ======================================================================================
# the T marker: id 1, from a second (4x4) dictionary
# ======================================================================================
# The board frame the joint fit works in is right-handed with +z toward the camera, and
# ``chi`` says which way round the ids run in it -- neither is the frame the user asked
# for.  The reported table frame shares the origin and the +x axis and differs only in the
# sign of y (and hence of z, to stay right-handed), so one factor ``k`` converts between
# them: ID 10 sits at board y = chi*W and must report as y = YAXIS_SIGN*W.
def board_pose(R, t, chi):
    """``{R, t, k}``: board -> camera rotation and origin, plus the board -> table y sign."""
    return {"R": np.asarray(R, dtype=np.float64).reshape(3, 3),
            "t": np.asarray(t, dtype=np.float64).reshape(3),
            "k": float(YAXIS_SIGN * chi)}


def to_table_cm(pose, p_cam):
    """Camera-frame point [m] -> table-frame ``(x, y, z)`` [cm]."""
    b = pose["R"].T @ (np.asarray(p_cam, dtype=np.float64).reshape(3) - pose["t"])
    k = pose["k"]
    return (float(b[0] * 100.0), float(k * b[1] * 100.0), float(k * b[2] * 100.0))


def table_yaw_deg(pose, v_cam):
    """Camera-frame direction -> its heading in the table plane, deg CCW from table +x."""
    b = pose["R"].T @ np.asarray(v_cam, dtype=np.float64).reshape(3)
    return float(np.degrees(np.arctan2(pose["k"] * b[1], b[0])))


def ray_to_table_cm(cam, pose, uv, height_cm):
    """Table-frame ``(x, y)`` [cm] where the ray through pixel ``uv`` meets the table.

    The board frame's ``+z`` faces the camera, so a marker sitting ``height_cm`` above the
    table is simply the plane ``z = height_cm``.  No depth and no marker size involved --
    the scale comes entirely from the board, which is why this is the best of the three.
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
    return (float(b[0] * 100.0), float(pose["k"] * b[1] * 100.0))


def find_quad(corners, ids, target):
    """The first ``target``-id marker's ``(4, 2)`` corner quad, or None."""
    if ids is None:
        return None
    for c, i in zip(corners, ids.flatten()):
        if int(i) == target:
            return np.asarray(c, dtype=np.float64).reshape(4, 2)
    return None


def tee_in_table(cam, quad, pose, side_cm, height_cm, pose_avg=None):
    """Where the T marker is on the table, all three ways, or None without a board pose.

    Each entry is ``{x_cm, y_cm, ...}`` in the table frame; any of them may be missing
    (no depth on the marker, no camera model) while the others are fine.
    """
    if quad is None or pose is None:
        return None
    centre = quad.mean(axis=0)
    out = {"px": [float(centre[0]), float(centre[1])], "height_cm": float(height_cm)}

    plane = ray_to_table_cm(cam, pose, centre, height_cm)
    if plane is not None:
        # Yaw without the marker size: drop two corners onto the plane and read the
        # marker's own +x edge (ArUco corner 0 -> corner 1) off the result.
        c0 = ray_to_table_cm(cam, pose, quad[0], height_cm)
        c1 = ray_to_table_cm(cam, pose, quad[1], height_cm)
        yaw = (None if c0 is None or c1 is None
               else float(np.degrees(np.arctan2(c1[1] - c0[1], c1[0] - c0[0]))))
        out["plane"] = {"x_cm": plane[0], "y_cm": plane[1], "yaw_deg": yaw}

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

    # The same ray, but through the averaged-pose frame -- how much the frame choice alone
    # moves the answer.
    if pose_avg is not None:
        plane = ray_to_table_cm(cam, pose_avg, centre, height_cm)
        if plane is not None:
            c0 = ray_to_table_cm(cam, pose_avg, quad[0], height_cm)
            c1 = ray_to_table_cm(cam, pose_avg, quad[1], height_cm)
            out["plane_avg"] = {
                "x_cm": plane[0], "y_cm": plane[1],
                "yaw_deg": (None if c0 is None or c1 is None else
                            float(np.degrees(np.arctan2(c1[1] - c0[1], c1[0] - c0[0])))),
            }
    return out


# ======================================================================================
# everything one frame has to say
# ======================================================================================
def measure_frame(cam, quads, side_cm, length_cm, width_cm, tee_q=None,
                  tee_len_cm=TEE_LEN_CM, tee_height_cm=0.0):
    """``{"sources": {source: dict}, "tee": dict or None}`` -- this frame, in full."""
    poses = marker_poses(cam, quads, side_cm)
    pts = {"pnp": {i: t for i, (_, t) in poses.items()},
           "depth": depth_points(cam, quads)}
    corner_depth = depth_corner_points(cam, quads)
    # The joint fit only needs *an* initial guess; pnp gives a cleaner one, so it seeds
    # both fits whenever it is available.
    init_pts = pts["pnp"] if set(FRAME_IDS) <= set(pts["pnp"]) else pts["depth"]

    out = {}
    for source in SOURCES:
        p = pts[source]
        out[source] = {
            "source": source,
            "ids": sorted(p),
            "xyz_cam_m": {i: [float(v) for v in q] for i, q in sorted(p.items())},
            "range_cm": {i: float(np.linalg.norm(q) * 100.0) for i, q in sorted(p.items())},
            "spans_cm": spans_cm(p),
            "table_cm": table_coords_cm(p),
            "corner_deg": corner_angles_deg(p),
            "nominal_fit": nominal_fit(p, length_cm, width_cm),
            "free": free_fit(cam, quads, poses, init_pts, side_cm, source, corner_depth),
            "joint": joint_fit(cam, quads, poses, init_pts, side_cm, source, corner_depth),
        }

    # The T is placed in one frame -- the best board pose available -- so its three
    # estimates differ only in how the T itself was located, not in where the table is.
    # The free fit's pose is preferred: same shared plane and baseline as the joint fit,
    # but it has not been forced onto a rectangle the table may not exactly be.
    pose = next((out[s][kind]["pose"] for kind in ("free", "joint") for s in SOURCES
                 if out[s][kind]), None)
    if pose is None:
        init = board_init(init_pts, poses, [i for i in CORNER_IDS if i in quads])
        if init is not None:
            rvec0, tvec0, _, _, chi, _ = init
            pose = board_pose(cv2.Rodrigues(rvec0)[0], tvec0, chi)
    avg = pose_average_fit(cam, quads, poses, init_pts, length_cm, width_cm)
    return {"sources": out, "avg": avg,
            "tee": tee_in_table(cam, tee_q, pose, tee_len_cm, tee_height_cm,
                                (avg or {}).get("pose"))}


# ======================================================================================
# drawing
# ======================================================================================
def _label(frame, text, pt, color, scale=0.55):
    for c, t in (((0, 0, 0), 4), (color, 1)):
        cv2.putText(frame, text, (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, c, t, cv2.LINE_AA)


def draw_tee(frame, quads, tee_q, tee):
    """Mark the T marker and label it with its table-frame position."""
    if tee_q is None:
        return
    c = tee_q.mean(axis=0)
    cv2.polylines(frame, [np.round(tee_q).astype(np.int32)], True, (255, 255, 0), 2,
                  cv2.LINE_AA)
    cv2.circle(frame, tuple(np.round(c).astype(int)), 7, (255, 255, 0), 2, cv2.LINE_AA)
    best = (tee or {}).get("plane") or (tee or {}).get("pnp") or (tee or {}).get("depth")
    if best is None:
        _label(frame, f"T id {TEE_ID}: no table frame", (c[0] + 10, c[1] + 6), (255, 255, 0))
        return
    if ORIGIN_ID in quads:
        cv2.line(frame, tuple(np.round(quads[ORIGIN_ID].mean(axis=0)).astype(int)),
                 tuple(np.round(c).astype(int)), (255, 255, 0), 1, cv2.LINE_AA)
    yaw = best.get("yaw_deg")
    text = f"T id {TEE_ID}: x={best['x_cm']:+.1f} y={best['y_cm']:+.1f} cm"
    if yaw is not None:
        text += f"  yaw={yaw:+.1f}deg"
    _label(frame, text, (c[0] + 10, c[1] + 6), (255, 255, 0), 0.6)


def draw_table(frame, corners, ids, quads, meas):
    """Outline the rectangle, mark the origin, and label each side with its pnp length."""
    cv2.aruco.drawDetectedMarkers(frame, corners, ids, (0, 200, 0))
    centres = {i: q.mean(axis=0) for i, q in quads.items()}

    ring = [centres[i] for i in CORNER_IDS if i in centres]
    if len(ring) == 4:
        cv2.polylines(frame, [np.round(ring).astype(np.int32)], True, (90, 90, 255), 1,
                      cv2.LINE_AA)

    spans = (meas or {}).get("spans_cm") or {}
    for a, b, kind in EDGES:
        if a not in centres or b not in centres:
            continue
        cv2.line(frame, tuple(np.round(centres[a]).astype(int)),
                 tuple(np.round(centres[b]).astype(int)), (0, 165, 255), 1, cv2.LINE_AA)
        if (a, b) in spans:
            mid = (centres[a] + centres[b]) / 2.0
            _label(frame, f"{kind} {a}-{b}: {spans[(a, b)]:.1f}cm",
                   (mid[0] - 60, mid[1] - 8), (0, 200, 255))

    free = (meas or {}).get("free") or {}
    table = free.get("table_cm") or (meas or {}).get("table_cm") or {}
    for i, c in centres.items():
        colour = (0, 255, 0) if i == ORIGIN_ID else (255, 0, 255)
        cv2.circle(frame, tuple(np.round(c).astype(int)), 6, colour, -1, cv2.LINE_AA)
        text = f"id {i}" + (" (origin 0,0)" if i == ORIGIN_ID else "")
        if i in table:
            x, y = table[i][0], table[i][1]
            text += f"  {x:+.1f},{y:+.1f}cm"
        _label(frame, text, (c[0] + 9, c[1] - 9), colour, 0.6)

    # The table frame's own axes, drawn from the origin marker toward XAXIS_ID and YAXIS_ID.
    if ORIGIN_ID in centres:
        for i, name, colour in ((XAXIS_ID, "+x", (60, 60, 255)),
                                (YAXIS_ID, "+y", (255, 200, 0))):
            if i not in centres:
                continue
            d = centres[i] - centres[ORIGIN_ID]
            n = float(np.linalg.norm(d))
            if n < 1e-6:
                continue
            tip = centres[ORIGIN_ID] + d / n * min(90.0, n * 0.35)
            cv2.arrowedLine(frame, tuple(np.round(centres[ORIGIN_ID]).astype(int)),
                            tuple(np.round(tip).astype(int)), colour, 2, cv2.LINE_AA,
                            tipLength=0.2)
            _label(frame, name, (tip[0] + 4, tip[1] - 4), colour, 0.6)


def hud_lines(by_source, quads, length_cm, width_cm, avg=None, tee=None, extra=()):
    """The per-source summary block shown on the GUI."""
    have = ",".join(str(i) for i in CORNER_IDS if i in quads) or "none"
    lines = [f"ids {have} ({len(quads)}/4)  |  nominal {length_cm:.1f} x {width_cm:.1f} cm"
             f"  |  s = save, q = quit"]
    for source in SOURCES:
        meas = by_source.get(source) or {}
        spans = meas.get("spans_cm") or {}
        cell = []
        for a, b, _ in EDGES:
            v = spans.get((a, b))
            cell.append("  --  " if v is None else f"{v:6.2f}")
        lines.append(f"{source:<5s} sides  L {cell[0]} {cell[1]}   W {cell[2]} {cell[3]} cm")
        fr = meas.get("free")
        if fr:
            sp = fr["spans_cm"]
            cell = [f"{sp[(a, b)]:6.2f}" if (a, b) in sp else "  --  " for a, b, _ in EDGES]
            lines.append(f"{source:<5s} free   L {cell[0]} {cell[1]}   "
                         f"W {cell[2]} {cell[3]} cm  rms {fr['rms']:.2f}{fr['rms_unit']}")
        j = meas.get("joint")
        lines.append(f"{source:<5s} joint  L {j['length_cm']:6.2f}  W {j['width_cm']:6.2f}  "
                     f"ratio {j['ratio']:.4f}  rms {j['rms']:.2f}{j['rms_unit']}  "
                     f"({len(j['ids'])} markers)"
                     if j else f"{source:<5s} joint  need ids {FRAME_IDS}")
    if avg:
        lines.append(f"avg   pose   origin spread {avg['origin_spread_cm']:5.2f} cm  "
                     f"normal spread {avg['normal_spread_deg']:5.2f} deg  "
                     f"tilt {avg['tilt_deg']:5.1f} deg  ({len(avg['ids'])} poses)")
    if tee is None:
        lines.append(f"T id {TEE_ID}  not visible (or no board pose yet)")
    else:
        for method in ("plane", "plane_avg", "pnp", "depth"):
            t = tee.get(method)
            if t is None:
                lines.append(f"T {method:<9s} --")
                continue
            yaw = t.get("yaw_deg")
            lines.append(f"T {method:<9s} x {t['x_cm']:+7.2f}  y {t['y_cm']:+7.2f} cm"
                         + (f"  yaw {yaw:+7.2f} deg" if yaw is not None else ""))
    fits = {s: (by_source.get(s) or {}).get("nominal_fit") for s in SOURCES}
    if any(fits.values()):
        bits = [f"{s} scale {fits[s]['scale']:.4f} rms {fits[s]['rms_cm']:.2f}cm"
                for s in SOURCES if fits[s]]
        tilt = next((f["tilt_deg"] for f in fits.values() if f), None)
        lines.append("nominal fit  " + "  |  ".join(bits) + f"  |  tilt {tilt:.1f}deg")
    return lines + list(extra)


# ======================================================================================
# headless measurement over N frames
# ======================================================================================
def _span_keys():
    return [((a, b), kind) for a, b, kind in EDGES] + [(d, "diagonal") for d in DIAGONALS]


TEE_METHODS = (("plane", ("x", "y", "yaw")), ("plane_avg", ("x", "y", "yaw")),
               ("pnp", ("x", "y", "z", "yaw")), ("depth", ("x", "y", "z")))


def measure(cam, detect, detect_tee, n_frames, side_cm, length_cm, width_cm,
            tee_id=TEE_ID, tee_len_cm=TEE_LEN_CM, tee_height_cm=0.0, json_out=None,
            save_png=None, warmup=15):
    """Grab ``n_frames`` frames and report every rectangle dimension, pnp vs depth."""
    for _ in range(warmup):                 # let auto-exposure and the depth filters settle
        cam.read()
    nominal = {"length": length_cm, "width": width_cm,
               "diagonal": float(np.hypot(length_cm, width_cm))}
    ratio = length_cm / width_cm
    print(f"\nmeasuring over {n_frames} frame(s);  markers {CORNER_IDS} from {ARUCO_DICT}, "
          f"{side_cm:.2f} cm a side\nnominal rectangle {length_cm:.1f} x {width_cm:.1f} cm "
          f"(diagonal {nominal['diagonal']:.2f} cm, ratio {ratio:.4f})\n")

    cols = {s: {"span": {k: [] for k, _ in _span_keys()},
                "range": {i: [] for i in CORNER_IDS},
                "corner": {i: [] for i in CORNER_IDS},
                "table": {i: {"x": [], "y": [], "z": []} for i in CORNER_IDS},
                "scale": [], "rms": [], "tilt": [],
                "jL": [], "jW": [], "jratio": [], "jrms": [], "jtilt": [],
                "fspan": {k: [] for k, _ in _span_keys()},
                "ftable": {i: {"x": [], "y": []} for i in CORNER_IDS},
                "frms": []}
            for s in SOURCES}
    tee_cols = {m: {f: [] for f in fields} for m, fields in TEE_METHODS}
    avg_cols = {"table": {i: {"x": [], "y": []} for i in CORNER_IDS},
                "span": {k: [] for k, _ in _span_keys()},
                "origin_spread": [], "normal_spread": [], "tilt": []}
    seen, tee_seen, last_frame = 0, 0, None

    for f in range(n_frames):
        ok, frame = cam.read()
        if not ok:
            continue
        last_frame = frame
        corners, ids, _ = detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        quads = corner_quads(corners, ids)
        if len(quads) < 4:
            missing = [i for i in CORNER_IDS if i not in quads]
            print(f"  frame {f:3d}: only {len(quads)}/4 corner markers -- missing {missing}")
            continue
        seen += 1
        tc, ti, _ = detect_tee(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        frame_out = measure_frame(cam, quads, side_cm, length_cm, width_cm,
                                  find_quad(tc, ti, tee_id), tee_len_cm, tee_height_cm)
        by_source, tee, avg = frame_out["sources"], frame_out["tee"], frame_out["avg"]
        if avg:
            for i, (x, y) in avg["table_cm"].items():
                avg_cols["table"][i]["x"].append(x)
                avg_cols["table"][i]["y"].append(y)
            for key, _ in _span_keys():
                if key in avg["spans_cm"]:
                    avg_cols["span"][key].append(avg["spans_cm"][key])
            avg_cols["origin_spread"].append(avg["origin_spread_cm"])
            avg_cols["normal_spread"].append(avg["normal_spread_deg"])
            avg_cols["tilt"].append(avg["tilt_deg"])
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
            meas, col = by_source[source], cols[source]
            for key, _ in _span_keys():
                if key in meas["spans_cm"]:
                    col["span"][key].append(meas["spans_cm"][key])
            for i, r in meas["range_cm"].items():
                col["range"][i].append(r)
            for i, a in meas["corner_deg"].items():
                col["corner"][i].append(a)
            for i, (x, y, z) in (meas["table_cm"] or {}).items():
                col["table"][i]["x"].append(x)
                col["table"][i]["y"].append(y)
                col["table"][i]["z"].append(z)
            fit = meas["nominal_fit"]
            if fit:
                col["scale"].append(fit["scale"])
                col["rms"].append(fit["rms_cm"])
                col["tilt"].append(fit["tilt_deg"])
            fr = meas["free"]
            if fr:
                col["frms"].append(fr["rms"])
                for key, _ in _span_keys():
                    if key in fr["spans_cm"]:
                        col["fspan"][key].append(fr["spans_cm"][key])
                for i, (x, y) in fr["table_cm"].items():
                    col["ftable"][i]["x"].append(x)
                    col["ftable"][i]["y"].append(y)
            j = meas["joint"]
            if j:
                col["jL"].append(j["length_cm"])
                col["jW"].append(j["width_cm"])
                col["jratio"].append(j["ratio"])
                col["jrms"].append(j["rms"])
                col["jtilt"].append(j["tilt_deg"])
            texts.append(f"{source} joint L={j['length_cm']:6.2f} W={j['width_cm']:6.2f}"
                         if j else f"{source} joint     --        --  ")
        best = (tee or {}).get("plane")
        texts.append(f"T x={best['x_cm']:+7.2f} y={best['y_cm']:+7.2f}"
                     if best else "T      --        --  ")
        print(f"  frame {f:3d}: " + "  |  ".join(texts))

    st = {s: {"span": {k: stats(v) for k, v in cols[s]["span"].items()},
              "range": {i: stats(v) for i, v in cols[s]["range"].items()},
              "corner": {i: stats(v) for i, v in cols[s]["corner"].items()},
              "table": {i: {ax: stats(v) for ax, v in d.items()}
                        for i, d in cols[s]["table"].items()},
              "fspan": {k: stats(v) for k, v in cols[s]["fspan"].items()},
              "ftable": {i: {ax: stats(v) for ax, v in d.items()}
                         for i, d in cols[s]["ftable"].items()},
              "frms": stats(cols[s]["frms"]),
              **{k: stats(cols[s][k]) for k in
                 ("scale", "rms", "tilt", "jL", "jW", "jratio", "jrms", "jtilt")}}
          for s in SOURCES}
    tee_st = {m: {f: stats(v) for f, v in d.items()} for m, d in tee_cols.items()}
    avg_st = {"table": {i: {ax: stats(v) for ax, v in d.items()}
                        for i, d in avg_cols["table"].items()},
              "span": {k: stats(v) for k, v in avg_cols["span"].items()},
              **{k: stats(avg_cols[k]) for k in ("origin_spread", "normal_spread", "tilt")}}

    print("\n" + "=" * 88)
    print(f"table rectangle from markers {CORNER_IDS}, {seen}/{n_frames} usable frames")
    print("pnp = solvePnP on the marker corners (no depth); "
          "depth = RealSense aligned depth (no marker size)")

    print(f"\n>> JOINT FIT -- one 12-parameter fit of the whole board, L and W the only")
    print("   free dimensions (rectangle + coplanarity + known marker squares assumed)."
          "\n   This is the estimate to quote.\n")
    print(f"  {'':22s}{'median':>11s}      {'[min .. max]':<24s}{'spread':<10s}{'vs nominal'}")
    for label, key, want in (("length (0-10, 20-30)", "jL", length_cm),
                             ("width  (0-30, 10-20)", "jW", width_cm)):
        print(f"  {label}  (expect {want:6.2f} cm)")
        for source in SOURCES:
            s = st[source][key]
            err = "" if s is None else (f"   {s[0] - want:+.2f} cm  "
                                        f"({100.0 * s[0] / want - 100.0:+.2f}%)")
            print(f"    {source:<18s}{fmt_stat(s)}{err}")
    print(f"  ratio L/W  (expect {ratio:.4f}) -- scale-free, so the pnp row does not "
          f"depend on --marker-len")
    for source in SOURCES:
        s = st[source]["jratio"]
        err = "" if s is None else f"   {100.0 * s[0] / ratio - 100.0:+.2f}%"
        print(f"    {source:<18s}{fmt_stat(s, '  ', 4)}{err}")
    print("  fit residual")
    for source, unit in (("pnp", "px"), ("depth", "cm")):
        print(f"    {source:<18s}{fmt_stat(st[source]['jrms'], unit)}")
    print("  camera tilt (0 = looking straight down)")
    for source in SOURCES:
        print(f"    {source:<18s}{fmt_stat(st[source]['jtilt'], 'deg')}")

    print("\n>> INDEPENDENT SIDES -- each marker placed on its own, then differenced.")
    print("   Noisier, but it assumes nothing about the layout, so it is the check that")
    print("   the four markers really do form the rectangle the joint fit imposes.\n")
    for key, kind in _span_keys():
        a, b = key
        print(f"  {kind:<8s} {a:>2d}-{b:<2d}  (expect {nominal[kind]:6.2f} cm)")
        for source in SOURCES:
            s = st[source]["span"][key]
            err = "" if s is None else (f"   {s[0] - nominal[kind]:+.2f} cm  "
                                        f"({100.0 * s[0] / nominal[kind] - 100.0:+.2f}%)")
            print(f"    {source:<18s}{fmt_stat(s)}{err}")

    print("\n  interior angle at each corner (expect 90.00 deg)")
    for i in CORNER_IDS:
        for source in SOURCES:
            print(f"    id {i:<2d} {source:<13s}{fmt_stat(st[source]['corner'][i], 'deg')}")

    print("\n  camera -> marker range")
    for i in CORNER_IDS:
        for source in SOURCES:
            print(f"    id {i:<2d} {source:<13s}{fmt_stat(st[source]['range'][i])}")

    print(f"\n>> MARKER POSITIONS in the table frame -- origin = ID {ORIGIN_ID}, "
          f"+x toward ID {XAXIS_ID}, -y toward ID {YAXIS_ID}")
    print("   free  = one board fit: one plane and one pose driven by all 16 corners, but")
    print("           every marker's (x, y) left free.  The best measurement of the layout.")
    print("   avg   = each marker's own pose, de-yawed, walked back to the origin through")
    print("           the nominal layout, and the four averaged.  Same per-marker points as")
    print("           indep, expressed in a four-vote frame -- so it moves the frame, not")
    print("           the geometry (its spans below are identical to the indep ones).")
    print(f"   indep = each marker solved on its own -- for pnp that is a {side_cm:.2f} cm")
    print("           baseline per marker, which is why it is the noisiest thing here.")
    print("   The joint fit is absent here by design: it pins the centres onto the")
    print("   rectangle, so its positions are (0,0), (L,0), (L,-W), (0,-W) exactly and")
    print("   would only be restating L and W.\n")
    model = rect_model_cm(length_cm, width_cm)
    for i in CORNER_IDS:
        mx, my = model[i]
        print(f"    id {i:<2d} (nominal {mx:+7.2f}, {my:+7.2f} cm)")
        a = avg_st["table"][i]
        if a["x"] is not None and a["y"] is not None:
            print(f"      {'avg pnp':<16s}x={a['x'][0]:+8.2f}  y={a['y'][0]:+8.2f} cm   "
                  f"(dx={a['x'][0] - mx:+6.2f}, dy={a['y'][0] - my:+6.2f}, "
                  f"sd {a['x'][3]:.2f}/{a['y'][3]:.2f})")
        for kind, key in (("free", "ftable"), ("indep", "table")):
            for source in SOURCES:
                t = st[source][key][i]
                if t["x"] is None or t["y"] is None:
                    print(f"      {kind + ' ' + source:<16s}    --")
                    continue
                x, y = t["x"][0], t["y"][0]
                print(f"      {kind + ' ' + source:<16s}x={x:+8.2f}  y={y:+8.2f} cm   "
                      f"(dx={x - mx:+6.2f}, dy={y - my:+6.2f}, "
                      f"sd {t['x'][3]:.2f}/{t['y'][3]:.2f})")

    print("\n  sides implied by those free positions -- measured, not constrained,")
    print("  so these are what say whether the table really is a rectangle")
    for key, kind in _span_keys():
        a, b = key
        print(f"    {kind:<8s} {a:>2d}-{b:<2d}  (expect {nominal[kind]:6.2f} cm)")
        for source in SOURCES:
            v = st[source]["fspan"][key]
            err = "" if v is None else f"   {v[0] - nominal[kind]:+.2f} cm"
            print(f"      {source:<16s}{fmt_stat(v)}{err}")
    print("  free-fit residual")
    for source, unit in (("pnp", "px"), ("depth", "cm")):
        print(f"    {source:<18s}{fmt_stat(st[source]['frms'], unit)}")

    print(f"\n>> POSE AVERAGE -- solvePnP each marker on its own, de-yaw it, walk it back")
    print(f"   to ID {ORIGIN_ID} through the nominal {length_cm:.1f} x {width_cm:.1f} cm "
          f"layout, then average the four")
    print("   rotations (SVD projection onto SO(3)) and the four origins.  pnp only --")
    print("   depth gives no per-marker rotation to average.  Assumes the nominal layout,")
    print("   so it is a frame estimator, not a dimension estimator.\n")
    print("    origin spread     " + fmt_stat(avg_st["origin_spread"]))
    print("    normal spread     " + fmt_stat(avg_st["normal_spread"], "deg"))
    print("    camera tilt       " + fmt_stat(avg_st["tilt"], "deg"))
    print("    ^ the spreads are how far the four independent votes disagree: small means")
    print("      the poses and the nominal layout are consistent, large means one marker")
    print("      is mis-measured, mis-placed, or seen at too shallow an angle to trust.")
    print("\n  sides from the averaged frame (identical to indep by construction --")
    print("  averaging poses cannot change a distance between two fixed points)")
    for key, kind in _span_keys():
        a, b = key
        v = avg_st["span"][key]
        err = "" if v is None else f"   {v[0] - nominal[kind]:+.2f} cm"
        print(f"    {kind:<8s} {a:>2d}-{b:<2d}  {fmt_stat(v)}{err}")

    print(f"\n>> NOMINAL FIT -- Procrustes onto the {length_cm:.1f} x {width_cm:.1f} cm "
          f"rectangle you typed in (a check, not a measurement)")
    for source in SOURCES:
        print(f"    scale {source:<13s}{fmt_stat(st[source]['scale'], 'x ', 4)}")
    for source in SOURCES:
        print(f"    rms   {source:<13s}{fmt_stat(st[source]['rms'])}")
    if st["pnp"]["scale"]:
        s = st["pnp"]["scale"][0]
        print(f"    pnp scale check:   the rectangle wants {s:.4f}x the pnp geometry, i.e. "
              f"the markers measure {side_cm * s:.2f} cm rather than {side_cm:.2f} cm")
    if st["depth"]["scale"]:
        s = st["depth"]["scale"][0]
        print(f"    depth scale check: the rectangle wants {s:.4f}x the depth geometry "
              f"({100.0 * s - 100.0:+.2f}% depth-scale error)")

    print(f"\n>> T MARKER -- id {tee_id} from {TEE_DICT}, position in the table frame")
    print(f"   seen in {tee_seen}/{seen} of the usable frames; "
          f"origin = ID {ORIGIN_ID}, +x along the length, -y along the width")
    print("   plane = pixel ray x table plane from the joint board fit -- no depth, no T")
    print(f"           marker size, plane lifted by --tee-height {tee_height_cm:.2f} cm; "
          f"quote this one")
    print(f"   pnp   = solvePnP on the T marker alone (--tee-len {tee_len_cm:.2f} cm)")
    print("   depth = its centre deprojected through the aligned depth frame\n")
    if not tee_seen:
        print(f"    T marker id {tee_id} never seen together with a board pose")
    for field, unit, label in (("x", "cm", "x  (along the length, >= 0 on the table)"),
                               ("y", "cm", "y  (along the width, <= 0 on the table)"),
                               ("z", "cm", "z  (height above the table plane)"),
                               ("yaw", "deg", "yaw (CCW from table +x)")):
        rows = [(m, tee_st[m].get(field)) for m, fields in TEE_METHODS if field in fields]
        if not any(v for _, v in rows):
            continue
        print(f"  {label}")
        for method, st_v in rows:
            print(f"    {method:<18s}{fmt_stat(st_v, unit)}")

    print("\n>> pnp - depth")
    for label, key in (("joint length", "jL"), ("joint width", "jW")):
        p, d = st["pnp"][key], st["depth"][key]
        if p and d:
            print(f"    {label:<16s}{p[0] - d[0]:+7.2f} cm")
    for key, kind in _span_keys():
        p, d = st["pnp"]["span"][key], st["depth"]["span"][key]
        if p and d:
            print(f"    {kind + f' {key[0]}-{key[1]}':<16s}{p[0] - d[0]:+7.2f} cm")
    print("=" * 88)

    if json_out:
        payload = {
            "dict": ARUCO_DICT, "ids": list(CORNER_IDS), "marker_len_cm": side_cm,
            "nominal_cm": {"length": length_cm, "width": width_cm,
                           "diagonal": nominal["diagonal"], "ratio": ratio},
            "frames_requested": n_frames, "frames_used": seen,
            "joint": {s: {"length_cm": st[s]["jL"], "width_cm": st[s]["jW"],
                          "ratio": st[s]["jratio"], "rms": st[s]["jrms"],
                          "tilt_deg": st[s]["jtilt"]} for s in SOURCES},
            "spans_cm": {s: {f"{a}-{b}": st[s]["span"][(a, b)]
                             for (a, b), _ in _span_keys()} for s in SOURCES},
            "free": {s: {"table_cm": st[s]["ftable"], "rms": st[s]["frms"],
                         "spans_cm": {f"{a}-{b}": st[s]["fspan"][(a, b)]
                                      for (a, b), _ in _span_keys()}} for s in SOURCES},
            "nominal_fit": {s: {"scale": st[s]["scale"], "rms_cm": st[s]["rms"]}
                            for s in SOURCES},
            "pose_average": {"table_cm": avg_st["table"],
                             "spans_cm": {f"{a}-{b}": avg_st["span"][(a, b)]
                                          for (a, b), _ in _span_keys()},
                             "origin_spread_cm": avg_st["origin_spread"],
                             "normal_spread_deg": avg_st["normal_spread"]},
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
                    help="ArUco dictionary the corner markers are from "
                         "(default %(default)s); one of:\n  " + "  ".join(DICTS))
    ap.add_argument("--width-px", type=int, default=PARAMS.camera_width, metavar="PX",
                    help="colour stream width (default %(default)s)")
    ap.add_argument("--height-px", type=int, default=PARAMS.camera_height, metavar="PX",
                    help="colour stream height (default %(default)s)")
    ap.add_argument("--marker-len", type=float, default=MARKER_LEN_CM, metavar="CM",
                    help="physical side length of each corner marker, cm -- the solvePnP "
                         "method's only metric scale (default %(default)s)")
    ap.add_argument("--length", type=float, default=LENGTH_CM, metavar="CM",
                    help="nominal centre-to-centre length, ids 0-10 and 20-30 "
                         "(default %(default)s)")
    ap.add_argument("--width", type=float, default=WIDTH_CM, metavar="CM",
                    help="nominal centre-to-centre width, ids 0-30 and 10-20 "
                         "(default %(default)s)")
    ap.add_argument("--tee-id", type=int, default=TEE_ID, metavar="ID",
                    help="id of the T-shape marker (default %(default)s)")
    ap.add_argument("--tee-dict", default=TEE_DICT, metavar="NAME",
                    help="dictionary the T marker is from -- a different one from the "
                         "corner markers (default %(default)s)")
    ap.add_argument("--tee-len", type=float, default=TEE_LEN_CM, metavar="CM",
                    help="physical side length of the T marker, cm; only its pnp estimate "
                         "depends on this (default %(default)s)")
    ap.add_argument("--tee-height", type=float, default=0.0, metavar="CM",
                    help="how far the T marker sits above the table plane, cm -- lifts the "
                         "plane the ray is intersected with (default %(default)s)")
    ap.add_argument("--measure", type=int, default=0, metavar="N",
                    help="headless: read N frames, report both methods, then exit (no GUI)")
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
    print(f"corner markers: ids {CORNER_IDS} from {args.dict};  "
          f"T marker: id {args.tee_id} from {args.tee_dict}")

    if args.measure:
        try:
            seen = measure(cam, detect, detect_tee, args.measure, args.marker_len,
                           args.length, args.width, args.tee_id, args.tee_len,
                           args.tee_height, args.measure_json, args.measure_png)
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
            quads = corner_quads(corners, ids)
            tee_q = find_quad(tee_corners, tee_ids, args.tee_id)
            out = measure_frame(cam, quads, args.marker_len, args.length, args.width,
                                tee_q, args.tee_len, args.tee_height)
            by_source, tee, avg = out["sources"], out["tee"], out["avg"]
            draw_table(frame, corners, ids, quads, by_source["pnp"])
            draw_tee(frame, quads, tee_q, tee)

            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
            last = now
            extra = [message] if message and now < msg_until else []
            lines = hud_lines(by_source, quads, args.length, args.width, avg, tee, extra)
            lines[0] = f"{fps:4.1f} fps  |  " + lines[0]
            hud(frame, lines)

            cv2.imshow(WINDOW, frame)
            if cam.depth_vis is not None:
                cv2.imshow(DEPTH_WINDOW, cam.depth_vis)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("s"):
                path = f"table_rect_{n_saved}.png"
                cv2.imwrite(path, frame)
                n_saved += 1
                message, msg_until = f"saved {path}", now + 4.0
                print(message)
    finally:
        cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
