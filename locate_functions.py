"""locate_functions -- where the T shape is on the table, live, in table centimetres.

Three functions, in the order they depend on each other:

``aruco_to_center``
    None of the ArUco markers is at the middle of the T.  The T points **down** -- crossbar
    along the top, stem hanging below -- inside a 12 x 12 cm bounding box, and each marker
    sits somewhere else on it: id 1 horizontally centred, 1.2 cm from the top edge, the rest
    wherever they were glued.  Given where a *marker* is, which one it is, and which way the
    shape is turned, this returns where the *shape* is.  Where each marker sits is the one
    per-marker constant, ``frame_conversions.TEE_MARKER_POS_CM``.

``marker_fix``
    One marker's worth of that: its quad -> the shape pose it implies, or a reason it could
    not.  The same measurement for every id, which is what makes the markers swappable.

``locate_shape``
    One observation, off **four markers**.  The shape carries four ids on four different
    parts of it, so the marker the arm parks on top of no longer costs a frame: every id is
    looked for in one detection pass and the ones in view are tried in **increasing id
    order**, the first clean read winning.  Not visible at all is the only real failure --
    the shape is out of frame, or the arm is covering all four at once.

    Then, per marker, split exactly the way the hardware is good at:

    * **x, y from depth** -- the RealSense's aligned depth frame is deprojected at the T
      marker's pixel centre, giving a metric 3-D point that owes nothing to the marker's
      printed size.
    * **theta from pnp** -- ``solvePnP`` on the same marker's four corners.  A small
      marker's *position* from pnp is weak (its range rides on a 5.5 cm baseline) but its
      *in-plane rotation* is the one thing it measures well, and rotation needs no scale
      at all.

    Lowest visible id rather than an average of the visible ones: the markers sit at
    different distances from the shape's centre, so their errors are not the same size, and
    one sliding part-covered out of frame would drag a mean without ever failing outright.
    Lowest-first also means the answer keeps coming off the same marker while several are
    in view, instead of dithering between them frame to frame.

    Both are expressed in the table frame of ``camera_test_id10`` -- origin at the **ID-10
    corner marker**, ``+x`` along the length, ``+y`` along the width -- which comes from
    that one marker's own ``solvePnP`` pose, redone every frame.  Then ``aruco_to_center``
    converts marker -> shape and the function reports ``x, y, theta``.

    The frame used to come from a board fit over four corner markers; now one 5.5 cm
    square carries the whole table, so its *direction* is the marker's own printed
    orientation.  Lay the marker with its ``+x`` edge along the length, or measure how far
    it ended up rotated and pass ``--yaw-offset``; ``camera_test_id10.py --measure`` prints
    what that costs at the far corner if you want the number.

``load_env``
    The environment ``push_t_demo_sim`` actually pushes against -- ``2denv4.obj``'s
    footprint -- loaded through ``push_t_demo_realworld.load_geometry`` and expressed in
    table centimetres.  The sim builds its collision world with
    ``obstacles = list(env_polys)``, "wall ring + interior blocks, all solid", so that
    footprint *is* the obstacle set.

``visualize``
    A viser scene: the environment and the table drawn once, the T redrawn continuously
    from ``locate_shape``.  Three things are drawn in three colours, because mistaking one
    for another is the alignment bug that is hardest to see: **grey** the nominal
    ``--length`` x ``--width`` table rectangle carried out from the ID-10 marker, **green**
    the box ``real_world_params.json`` normalizes to, and **dark** the real env geometry
    with its obstacles.  The grey rectangle is now drawn, not measured -- with one marker
    there is nothing left observing the far corners.

    python locate_functions.py                      # open http://localhost:8080
    python locate_functions.py --center bbox        # if you mean the bounding-box middle
    python locate_functions.py --tee-ids 1 3        # only trust these two markers
    python locate_functions.py --marker-from-top 3.95
    python locate_functions.py --yaw-offset -3.5    # marker glued down 3.5 deg clockwise

**Measuring a marker in.**  Hold the T the way it reads in text -- crossbar on top, stem
hanging down -- and measure each marker's centre **from the top-left corner** of its 12 x 12
bounding box: ``x`` to the right along the top edge, ``y`` DOWNWARD and therefore negative.
Those two numbers are the whole entry, ``(x_cm, y_cm)`` in
``frame_conversions.TEE_MARKER_POS_CM``; the centroid arithmetic and the mount turn are done
for you, and a point that does not land on the shape is rejected with the reason rather than
believed, so a dropped minus sign shows up immediately.  Only id 1 is filled in.  An id left
as ``None`` is skipped rather than guessed at, so the stack runs on however many are
measured -- one is enough to work, four is what makes it robust.

**One assumption worth checking.** For id 1, "1.2 cm from the top" is read literally as the
marker's *centre* being 1.2 cm below the T's top edge.  If you meant the marker's top *edge*
is 1.2 cm down, its centre is 1.2 + tee_len/2 further in -- pass ``--marker-from-top 3.95``
for a 5.5 cm marker.  Nothing else changes; it is one constant, and it only moves id 1.

``--center`` picks what "the centre" means, and the two differ by 3.2 cm here:
``centroid`` (default) is the T's area centroid, which is the origin the sim and
``push_t_demo_realworld`` use for the body; ``bbox`` is the middle of the 12 x 12 box.
"""

import argparse
import math
import os
import time

import cv2
import numpy as np

import frame_conversions as FC
from frame_conversions import (
    CENTER_MODES,
    DEFAULT_TEE_MARKER_ID,
    MARKER_FROM_TOP_CM,
    MARKER_YAW_ON_TEE_DEG,
    TEE_BAR_CM,
    TEE_MARKER_IDS,
    TEE_MARKER_POS_CM,
    TEE_SIZE_CM,
    TEE_STEM_CM,
    aruco_to_center,
    marker_to_center_offset_cm,
    tee_body_outline_cm,
    tee_center_local_cm,
    tee_marker_configured,
    tee_marker_ids,
    tee_marker_offset,
    tee_marker_pos,
    tee_theta_from_marker,
    tee_topleft_to_body_cm,
    to_sim_pose,
)

from camera_test_id10 import (
    ARUCO_DICT,
    LENGTH_CM,
    MARKER_LEN_CM,
    ORIGIN_ID,
    TEE_DICT,
    TEE_LEN_CM,
    WIDTH_CM,
    YAW_OFFSET_DEG,
    RealSenseCamera,
    depth_frame,
    find_quad,
    make_detector,
    pnp_frame,
    realsense_present,
    table_corners_cm,
    tee_in_table,
    to_cam_m,
)

# The T's proportions, the marker mount angle and the marker -> shape offset all live in
# ``frame_conversions`` now and are imported above; ``sim_tee_dims`` below still reads the
# real mesh, which is what keeps them honest.
# Absolute, so the shape is found no matter where the program is run from.
HERE = os.path.dirname(os.path.abspath(__file__))
SIM_SHAPE_ZUP = os.path.join(HERE, "datasets", "3dshape", "Tshape3d_zup.obj")
SIM_SHAPE = os.path.join(HERE, "datasets", "3dshape", "Tshape3d.obj")
SIM_ENV = os.path.join(HERE, "datasets", "3dshape", "2denv4.obj")

# Straight from the sim so the two scenes read as the same world: COL_ENV / COL_TEE /
# COL_FLOOR are pymunk_viser_push's, COL_SIM_ENV is push_t_demo_sim's COL_GHOST green.
COL_ENV = (110, 118, 132)          # pymunk_viser_push.COL_ENV -- the obstacle geometry
COL_TEE = (232, 138, 62)           # pymunk_viser_push.COL_TEE -- the shape
COL_FLOOR = (232, 232, 228)        # pymunk_viser_push.COL_FLOOR -- the table top
COL_MARKER = (60, 200, 90)
COL_HEADING = (250, 250, 250)
COL_SIM_ENV = (150, 210, 150)      # push_t_demo_sim.COL_GHOST -- the normalization box
COL_ENV_EDGE = (150, 150, 175)
COL_TABLE = (150, 150, 165)        # push_t_demo_sim.COL_STANDOFF -- the table rectangle
COL_FIX = (240, 210, 90)           # one marker's own answer for the shape centre

# How far two markers' answers for the shape centre may sit apart before ``markers_markdown``
# calls it a mis-measured position rather than noise.  The depth return at a marker centre
# jitters a couple of millimetres and the plane heading a degree or so, which at these lever
# arms is a few mm of disagreement; a centimetre is not noise, it is a wrong number in
# TEE_MARKER_POS_CM.  Display only -- nothing is rejected on it, since a real run should
# report a slightly-off pose rather than none at all.
MARKER_AGREE_WARN_CM = 1.0


# ======================================================================================
# the T's own geometry, in its body frame
# ======================================================================================
def _obj_verts(path):
    """``(N, 3)`` of an .obj's ``v`` lines, or None if it cannot be read."""
    try:
        return np.array([[float(t) for t in ln.split()[1:4]]
                         for ln in open(path) if ln.startswith("v ")], dtype=np.float64)
    except (OSError, ValueError):
        return None


def sim_tee_dims(size_cm=TEE_SIZE_CM, path=SIM_SHAPE_ZUP):
    """``(bar_cm, stem_cm)`` for a ``size_cm`` T, read off the sim's own mesh, or None.

    The planner was trained on one particular T, so its proportions are the ground truth
    and guessing them is how the reported centre drifts.  The z-up mesh is the planner
    frame (``push_t_demo_realworld.load_geometry`` reads the same file for
    ``bbox_to_centroid``): footprint in x/y, thickness in z, T pointing down.

    Returns None -- caller falls back to the module defaults -- if the file is missing or
    does not look like a downward T.
    """
    V = _obj_verts(path)
    if V is None or len(V) < 8:
        return None
    xy = V[:, :2]
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    span = hi - lo
    if min(span) <= 0.0:
        return None
    scale = size_cm / float(max(span))            # the mesh bbox maps onto size_cm
    # The crossbar's underside is the second-highest distinct y; the stem's width is the
    # x extent of the vertices sitting on the bottom edge.
    ys = np.unique(np.round(xy[:, 1], 6))
    if len(ys) < 3:
        return None
    bar = float(ys[-1] - ys[-2])
    bottom = xy[np.isclose(xy[:, 1], ys[0])]
    stem = float(bottom[:, 0].max() - bottom[:, 0].min())
    if bar <= 0.0 or stem <= 0.0:
        return None
    return bar * scale, stem * scale


def sim_tee_thickness_cm(size_cm=TEE_SIZE_CM, path=SIM_SHAPE_ZUP):
    """How thick the sim's T is, scaled to a ``size_cm`` shape, or None.

    Same mesh and the same footprint-to-``size_cm`` scale ``sim_tee_dims`` uses, read off
    the z extent instead -- so the drawn prism is as tall, relative to its footprint, as
    the one ``push_t_demo_sim`` renders (10 units on a 60 unit box: one sixth).
    """
    V = _obj_verts(path)
    if V is None or len(V) < 8:
        return None
    span = V[:, :2].max(axis=0) - V[:, :2].min(axis=0)
    thick = float(V[:, 2].max() - V[:, 2].min())
    if min(span) <= 0.0 or thick <= 0.0:
        return None
    return thick * (size_cm / float(max(span)))


# ======================================================================================
# the bridge to push_t_demo_sim
# ======================================================================================
# ``frame_conversions.to_sim_pose`` / ``sim_bridge`` do the crossing; what is left here is
# the *check* that the two ends line up, which needs this module's mesh readings.
#
# Two things had to be made to line up, and both are checked by ``sim_alignment``:
#   * the body origin.  ``load_geometry`` builds ``tee_poly`` about the footprint centroid
#     and the sim world SE(2) pose is about that point, so ``mode="centroid"``.
#   * theta = 0.  ``Tshape3d_zup.obj`` is authored pointing down, and ``tee_outline_cm``
#     is built pointing down, so the two zeros agree; ``real_to_sim_pose`` then adds
#     ``rotation_deg`` on top.
def sim_alignment(size_cm=TEE_SIZE_CM, bar_cm=TEE_BAR_CM, stem_cm=None, mode="centroid"):
    """What the sim expects vs what this module is using -- a printable dict.

    ``main`` prints it at startup so a mismatch is visible before any data is taken,
    rather than showing up later as a shape that is offset by a couple of centimetres.
    """
    out = {"mode": mode, "bar_cm": bar_cm, "stem_cm": bar_cm if stem_cm is None else stem_cm}
    dims = sim_tee_dims(size_cm)
    out["sim_bar_cm"], out["sim_stem_cm"] = dims if dims else (None, None)
    out["centroid_local_cm"] = tee_center_local_cm(size_cm, bar_cm, stem_cm, mode).tolist()
    rw = FC.sim_bridge()
    if rw is None:
        out["bridge"] = "push_t_demo_realworld unavailable -- table frame only"
        return out
    out["bridge"] = "push_t_demo_realworld"
    out["env_cm"] = [float(rw.ENV_X), float(rw.ENV_Y)]
    out["env_center_cm"] = [float(rw.CALIB_OFFSET_X), float(rw.CALIB_OFFSET_Y)]
    out["rotation_deg"] = float(rw.CALIB_ROTATION_DEG)
    out["sim_units_per_cm"] = float(rw.sim_units_per_cm())
    return out


def env_box_cm():
    """The planner's normalization square in table cm as ``(4, 2)``.

    Not the environment itself -- this is the field ``frame_conversions`` maps onto
    ``[-0.5, 0.5]^2``.  Worth drawing next to the real geometry, because the env sitting
    anywhere other than inside it is the calibration bug that is hardest to see otherwise.
    Straight from ``frame_conversions``, so it needs no sim stack and cannot disagree with
    the pose conversion the planner is actually fed.
    """
    return FC.field_corners_cm()


def load_env(env_obj=SIM_ENV, shape_obj=SIM_SHAPE, shape_zup=SIM_SHAPE_ZUP):
    """The real sim environment, in table centimetres.  ``dict`` or ``None``.

    Loads exactly what ``push_t_demo_sim`` pushes against.  ``push_t_demo_sim`` builds its
    collision world with ``obstacles = list(env_polys)`` -- "wall ring + interior blocks,
    all solid" -- and ``env_polys`` is the vertical footprint of ``2denv4.obj``.  So the
    footprint *is* the obstacle set, and drawing it is drawing the obstacles.

    Returns ``{"rings", "vertices", "faces", "n_polys", "n_holes"}``:

    * ``rings``    -- every polygon boundary, exterior and hole alike, as ``(N, 2)`` loops
      in table cm, straight from ``Geometry.env_rings_real``.
    * ``vertices`` / ``faces`` -- the env mesh itself, flattened onto the table and pushed
      through the same ``sim -> real`` transform, so the fill is the true geometry rather
      than a re-triangulation of it.  The prism's side walls collapse to zero area in
      plan view and simply do not render.

    Paths default to absolute so this works from any working directory.  Returns None if
    the sim stack, trimesh or the .obj files are unavailable -- the caller falls back to
    the plain normalization box.
    """
    rw = FC.sim_bridge()
    if rw is None:
        load_env.last_error = ("push_t_demo_realworld unavailable -- it needs pymunk, "
                               "trimesh and shapely")
        return None
    try:
        geo = rw.load_geometry(rw.make_args(env=env_obj, shape=shape_obj,
                                            shape_zup=shape_zup))
        rings = [np.asarray(r, dtype=np.float64) for r in geo.env_rings_real()]
        V = np.asarray(geo.env_mesh.vertices, dtype=np.float64)
        xy = np.asarray(rw.path_sim_to_real(V[:, :2]), dtype=np.float64)
        load_env.last_error = None
        return {
            "rings": rings,
            "vertices": np.column_stack([xy, np.zeros(len(xy))]).astype(np.float32),
            "faces": np.asarray(geo.env_mesh.faces, dtype=np.uint32),
            "n_polys": len(geo.env_polys),
            "n_holes": sum(len(p.interiors) for p in geo.env_polys),
        }
    except Exception as exc:                              # noqa: BLE001
        load_env.last_error = f"{type(exc).__name__}: {exc}"
        return None


load_env.last_error = None


# ======================================================================================
# one observation
# ======================================================================================
# A further quarter-turn on top of ``yaw_offset_deg``, measured off the live scene: with
# the offset alone a point read (x, y) in the table frame where it should have read
# (-y, x), i.e. the frame sat a quarter-turn clockwise of the table.  Positive is CCW and
# adds to the offset, because the table-frame reading of a camera point is the marker-frame
# one spun by +yaw (``camera_test_id10.apply_yaw`` builds ``R`` as the marker's rotated by
# -yaw, and reading a point applies ``R.T``).  Kept here, and not folded into
# ``real_world_params.json``'s ``origin_marker_yaw_deg``, on purpose: that value is shared
# with ``camera_test_id10`` and with the arm calibration -- ``offset_to_origin_cm`` and p0
# were both measured in the un-turned frame -- so turning it there would silently move
# every one of those.
ORIGIN_EXTRA_YAW_DEG = 90.0


def table_pose(cam, gray, detect_table, marker_len_cm=MARKER_LEN_CM,
               yaw_offset_deg=YAW_OFFSET_DEG, extra_yaw_deg=ORIGIN_EXTRA_YAW_DEG):
    """The table frame for this frame: ``(pose, info)``, or ``(None, info)``.

    One marker, so there is nothing to fit and nothing to average: ``solvePnP`` on the
    ID-10 square gives the origin and all three axes outright.  The depth-built frame is
    the fallback for the frames before the camera model is up.  Recomputed every call
    rather than cached, so nudging the camera mid-run costs one frame instead of silently
    biasing every reading after it.

    What is gone with the other three markers is the redundancy: ``rms`` here is the
    marker's own corner reprojection, which catches a bad detection but says nothing about
    whether the frame is *aimed* right -- that now rests entirely on how squarely the
    marker was laid down (``yaw_offset_deg`` plus ``extra_yaw_deg``, the measured
    quarter-turn -- see ``ORIGIN_EXTRA_YAW_DEG``; the two are added and only their sum
    reaches the pose, which ``info["yaw_total_deg"]`` reports).
    """
    corners, ids, _ = detect_table(gray)
    quad = find_quad(corners, ids, ORIGIN_ID)
    yaw_total_deg = float(yaw_offset_deg) + float(extra_yaw_deg)
    info = {"table_id": ORIGIN_ID, "table_seen": quad is not None,
            "yaw_total_deg": yaw_total_deg}
    if quad is None:
        return None, info
    pose = (pnp_frame(cam, quad, marker_len_cm, yaw_total_deg)
            or depth_frame(cam, quad, yaw_total_deg))
    if pose is None:
        return None, info
    info["frame_source"] = pose["source"]
    info["frame_rms"] = pose["rms"]
    info["frame_rms_unit"] = pose["rms_unit"]
    info["table_px"] = [float(v) for v in quad.mean(axis=0)]
    # the origin itself, in the camera frame -- the table frame pins (0, 0, 0) here
    info["origin_cam_cm"] = [float(v * 100.0) for v in pose["t"]]
    return pose, info


def marker_fix(cam, pose, quad, marker_id, offset, tee_len_cm=TEE_LEN_CM,
               tee_height_cm=0.0):
    """What ONE shape marker says about the shape: ``(fix, None)`` or ``(None, reason)``.

    Split out of ``locate_shape`` because with four markers this is the part that runs per
    id.  The measurement is identical for every marker -- depth at its centre for ``x, y``,
    its own ``+x`` edge on the table plane for ``theta`` -- and only ``offset``, where that
    marker sits on the shape, differs.  That is what makes the markers interchangeable: a
    clean read off any one of them lands the same shape pose as a clean read off any other,
    so the one the arm is standing on can simply be skipped.

    ``offset`` is ``(dx_cm, dy_cm, yaw_deg)`` for this marker, from
    ``frame_conversions.tee_marker_offset`` -- looked up once by the caller so the position
    and the heading cannot end up using two different rows.

    ``reason`` is a short phrase for the miss list, so a caller can say *which* marker
    failed and *how* rather than just "no fix".
    """
    tee = tee_in_table(cam, quad, {pose["source"]: pose}, tee_len_cm, tee_height_cm)
    if tee is None:
        return None, "no camera model yet"
    plane, pnp = tee.get("plane"), tee.get("pnp")

    # ---- x, y: the depth frame, deprojected at the marker centre ----
    d = tee.get("depth")
    if d is None:
        return None, "no valid depth on it"
    mx, my, mz = d["x_cm"], d["y_cm"], d["z_cm"]

    # ---- theta: the marker's own +x edge, dropped onto the table plane ----
    # pnp is kept only as the cross-check -- it is what this replaced.
    if plane is not None and plane.get("yaw_deg") is not None:
        marker_theta_deg, theta_source = plane["yaw_deg"], "plane"
    elif pnp is not None:
        marker_theta_deg, theta_source = pnp["yaw_deg"], "pnp"
    else:
        return None, "no heading off it (plane and pnp both failed)"

    # Everything above is the MARKER's heading.  The marker is mounted turned, so the
    # shape's own heading -- what gets reported, drawn and handed to the planner -- is that
    # less the mount angle.  aruco_to_center gets handed the same ``offset`` and applies the
    # same turn internally, so the position and the orientation cannot drift apart.
    marker_theta = math.radians(marker_theta_deg)
    theta = tee_theta_from_marker(marker_theta, offset=offset)
    x_cm, y_cm = aruco_to_center(mx, my, marker_theta, offset=offset)
    centre_px = quad.mean(axis=0)
    return {
        "x_cm": x_cm,
        "y_cm": y_cm,
        "theta_rad": theta,
        "theta_deg": math.degrees(theta),
        "theta_source": theta_source,
        # which marker this fix came off, and the row of the table it used
        "tee_id": int(marker_id),
        "marker_offset_cm": [float(offset[0]), float(offset[1])],
        # the raw marker heading, before the mount turn -- what each method measured
        "marker_theta_deg": marker_theta_deg,
        "marker_yaw_on_tee_deg": float(offset[2]),
        "theta_plane_deg": None if plane is None else plane.get("yaw_deg"),
        "theta_pnp_deg": None if pnp is None else pnp.get("yaw_deg"),
        # the plane method's own x, y -- an independent check on the depth ones above
        "plane_x_cm": None if plane is None else plane["x_cm"],
        "plane_y_cm": None if plane is None else plane["y_cm"],
        "marker_x_cm": mx,
        "marker_y_cm": my,
        "marker_z_cm": mz,          # height above the table plane; a depth sanity check
        "marker_px": [float(centre_px[0]), float(centre_px[1])],
    }, None


def locate_shape(cam, detect_table, detect_tee, marker_len_cm=MARKER_LEN_CM,
                 yaw_offset_deg=YAW_OFFSET_DEG, tee_ids=None, tee_len_cm=TEE_LEN_CM,
                 tee_height_cm=None, size_cm=TEE_SIZE_CM, bar_cm=TEE_BAR_CM, stem_cm=None,
                 positions=None, mode="centroid", with_sim_pose=True,
                 prefer_id=None, compare=False):
    """Locate the T on the table: ``{x_cm, y_cm, theta_rad, ...}``, or ``None``.

    **Four markers, not one.**  The shape carries four ArUco markers of different ids on
    different parts of it, and each one alone fixes the whole pose, so the marker the arm is
    parked on top of is not a lost frame -- the other three are still there.  Every id in
    ``tee_ids`` is looked for in the one detection pass, and the ones actually in view are
    tried **in increasing id order**, the first that measures cleanly winning.  Lowest-first
    rather than best-of or averaged, so while several markers are visible the answer keeps
    coming off the same one and cannot dither between them; when that one disappears the
    next id up takes over.  ``tee_id`` in the result says which marker the reported pose
    came off, and ``tee_ids_seen`` / ``tee_ids_blocked`` say what the frame had to offer.

    Averaging the visible markers is deliberately *not* done: the ids differ in how far they
    are from the shape's centre, so their errors are not the same size, and a marker sliding
    out of view part-covered would drag the mean without ever failing outright.

    Two arguments exist for checking that claim against the real table, and are what
    ``visualize``'s marker panel drives:

    ``prefer_id``
        Read this marker if it is usable, falling back to the ascending walk only if it is
        not.  Pointing it at each id in turn is how you confirm they all agree; the result
        says whether the preference was honoured (``tee_id_preferred`` vs ``tee_id``, and
        ``fix_from_preferred``).
    ``compare``
        Do not stop at the winner -- measure **every** visible marker and return them all in
        ``fixes``, keyed by id, each a full fix dict.  The chosen one is still what the
        top-level ``x_cm`` / ``y_cm`` / ``theta_rad`` report, so turning this on changes
        nothing about the answer; it only puts the others alongside it.  ``fix_spread`` then
        measures how far apart they landed, which is the number that says whether the
        positions in ``TEE_MARKER_POS_CM`` are right.

    ``x`` and ``y`` come from **depth**, the heading from the **plane** method, each from the
    source that measures it best, and both are then put in the table frame before that
    marker's own offset walks out to the shape's centre.  ``marker_fix`` above does one
    marker's worth of that; the offset comes from
    ``frame_conversions.tee_marker_offset``, which turns the marker's **measured position**
    -- ``TEE_MARKER_POS_CM``, ``(x_cm, y_cm)`` from the T's top-left corner, ``+x`` right and
    ``-y`` down -- into a vector to the centre.  That table is where a newly stuck-on marker
    gets measured in, and it is the only per-marker thing there is to know.  An id still
    ``None`` there is skipped: an unmeasured marker would give a confidently wrong pose,
    which is worse than a missed frame.

    ``theta_rad`` / ``theta_deg`` are the **shape's** heading, not the marker's: the markers
    are mounted turned by ``marker_yaw_on_tee_deg`` (one value, shared by all four, since
    they are all glued on the same way round), and that turn comes off before anything is
    reported, so the pose here is one the sim and the planner can use directly.  The raw
    marker heading each method measured is kept as ``marker_theta_deg`` /
    ``theta_plane_deg`` / ``theta_pnp_deg``.

    **theta = plane, not pnp.**  ``solvePnP`` on a 5.5 cm square has to infer the marker's
    tilt from how far its outline departs from a perfect square, and near fronto-parallel
    -- exactly how a top-down table camera sees it -- that departure is a pixel or two, so
    the rotation it recovers is noisy and prone to flipping between near-degenerate
    solutions.  The plane method never solves the marker's pose at all: it drops corner 0
    and corner 1 onto the *table* plane, which the ID-10 marker has already fixed, and
    reads the heading off that chord.  The tilt pnp had to guess is now given, and no
    marker size enters.

    It does need ``tee_height_cm``, how far the shape markers ride above the table, to put
    that plane at the right height; ``None`` takes the shape's own thickness, since the
    markers sit on top of the T.  ``theta_source`` says which method actually produced the
    answer and ``theta_pnp_deg`` carries what pnp would have said, so the disagreement is
    visible rather than assumed.

    Returns ``None`` when the ID-10 marker is not in view, when **not one** shape marker is
    in view, or when every visible one failed to measure -- the reason, per id, is in
    ``locate_shape.last_miss`` so a caller can show it without a second detection pass.

    It never takes the table's nominal dimensions: nothing here depends on the table really
    being 79 x 63 cm, and ``--length`` / ``--width`` only draw the rectangle in the viser
    scene.  What it does depend on, and the four-corner-marker version did not, is the
    ID-10 marker's own heading -- see ``table_pose``.
    """
    ids_wanted = (tee_marker_ids(positions) if tee_ids is None
                  else tuple(sorted({int(i) for i in tee_ids})))
    if not ids_wanted:
        locate_shape.last_miss = ("no shape marker has a position measured -- fill in "
                                  "frame_conversions.TEE_MARKER_POS_CM")
        return None

    ok, frame = cam.read()
    if not ok:
        locate_shape.last_miss = "no frame from the camera"
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    pose, info = table_pose(cam, gray, detect_table, marker_len_cm, yaw_offset_deg)
    if pose is None:
        locate_shape.last_miss = (f"table marker id {ORIGIN_ID} not visible"
                                  if not info["table_seen"] else
                                  f"no pose from table marker id {ORIGIN_ID} "
                                  f"(no camera model and no depth on its corners)")
        return None

    # One detection pass for all four ids -- they are in the same dictionary, so finding
    # them is one call and only the quad lookup is per id.
    tee_corners, tee_found, _ = detect_tee(gray)
    quads = {mid: find_quad(tee_corners, tee_found, mid) for mid in ids_wanted}
    seen = tuple(mid for mid in ids_wanted if quads[mid] is not None)
    blocked = tuple(mid for mid in ids_wanted if quads[mid] is None)
    if not seen:
        locate_shape.last_miss = (
            f"none of the shape markers {list(ids_wanted)} is visible -- the shape is out "
            f"of frame, or every marker on it is covered")
        return None

    if tee_height_cm is None:
        tee_height_cm = sim_tee_thickness_cm(size_cm) or 0.0

    # Increasing id order, first clean read wins.  Anything tried and rejected before it is
    # kept in ``misses`` -- a marker that is visible but never usable (a position that is
    # off the shape, no depth return where it sits) is worth seeing rather than silently
    # skipping it every frame.
    # ``prefer_id`` jumps one marker to the front of the walk; everything after it stays in
    # ascending order, so a preferred marker that is blocked degrades to the normal rule
    # rather than to no fix at all.
    order = list(seen)
    want = None if prefer_id is None else int(prefer_id)
    if want in seen:
        order = [want] + [m for m in seen if m != want]

    fix, fixes = None, {}
    misses = [f"id {mid} not visible" for mid in blocked]
    for mid in order:
        try:
            offset = tee_marker_offset(mid, positions, mode, size_cm, bar_cm, stem_cm)
        except (KeyError, ValueError) as exc:
            misses.append(f"id {mid}: {exc}")
            continue
        cand, why = marker_fix(cam, pose, quads[mid], mid, offset, tee_len_cm,
                               tee_height_cm)
        if cand is None:
            misses.append(f"id {mid}: {why}")
            continue
        fixes[mid] = cand
        if fix is None:
            fix = cand              # the first success in ``order`` is the one reported
        if not compare:
            break

    if fix is None:
        locate_shape.last_miss = ("no usable shape marker -- " + "; ".join(misses))
        return None

    locate_shape.last_miss = None
    return {
        **fix,
        # the same pose in push_t_demo_sim's world frame, ready for the planner
        "sim_pose": (to_sim_pose(fix["x_cm"], fix["y_cm"], fix["theta_rad"])
                     if with_sim_pose else None),
        "tee_height_cm": float(tee_height_cm),
        # what the frame had to offer: which ids were looked for, which were in view, which
        # were not, and why the ones tried before the winner were passed over
        "tee_ids_wanted": list(ids_wanted),
        "tee_ids_seen": list(seen),
        "tee_ids_blocked": list(blocked),
        "tee_marker_misses": misses,
        # Every marker actually measured this frame, keyed by id.  With ``compare`` off this
        # is just the winner; with it on it is the whole side-by-side.
        "fixes": fixes,
        "tee_ids_fixed": sorted(fixes),
        "tee_id_preferred": want,
        # False when ``prefer_id`` was asked for but could not be read, so the pose came off
        # a different marker than the one requested -- worth saying rather than hiding
        "fix_from_preferred": want is None or fix["tee_id"] == want,
        "fix_spread": fix_spread(fixes),
        "center_mode": mode,
        # the table frame itself, so a caller can carry any table point -- the shape's
        # origin, the field's -- back into the camera frame without re-detecting
        "pose": pose,
        **info,
    }


locate_shape.last_miss = None


# ======================================================================================
# viser
# ======================================================================================
def _loop_segments(xy, z=0.0):
    """``(N, 2)`` polygon -> ``(N, 2, 3)`` closed line segments."""
    p = np.asarray(xy, dtype=float)
    if len(p) and not np.allclose(p[0], p[-1]):
        p = np.vstack([p, p[:1]])
    p3 = np.column_stack([p, np.full(len(p), z)]).astype(np.float32)
    return np.stack([p3[:-1], p3[1:]], axis=1)


def _square(centre, side, z=0.0):
    h = side / 2.0
    cx, cy = float(centre[0]), float(centre[1])
    return _loop_segments([(cx - h, cy - h), (cx + h, cy - h),
                           (cx + h, cy + h), (cx - h, cy + h)], z)


def tee_mesh_cm(size_cm=TEE_SIZE_CM, bar_cm=TEE_BAR_CM, stem_cm=None, mode="centroid",
                z=0.0, thickness_cm=None):
    """``(vertices, faces)`` for the T as a solid prism standing on ``z``.

    The shape sits *on* the table -- ``z = 0`` is its underside, and it is extruded up by
    ``thickness_cm`` -- so it reads as the same object ``push_t_demo_sim`` draws (which
    renders ``Tshape3d_zup.obj`` itself, 60 x 60 x 10 units, i.e. one sixth of the box
    thick) rather than as a flat decal on the table.

    The outline is concave, so the footprint is built from its two rectangles rather than
    fanned from one vertex, which would spill triangles outside the shape.  Each rectangle
    becomes a box: 2 caps + 4 walls.  The two boxes overlap where the stem meets the
    crossbar, which is invisible in a solid render and saves triangulating a concave prism.
    """
    stem_cm = bar_cm if stem_cm is None else stem_cm
    if thickness_cm is None:
        thickness_cm = sim_tee_thickness_cm(size_cm) or size_cm / 6.0
    h, b = size_cm / 2.0, stem_cm / 2.0
    y_bar = h - bar_cm
    cx, cy = tee_center_local_cm(size_cm, bar_cm, stem_cm, mode)
    z0, z1 = float(z), float(z) + float(thickness_cm)
    quads = [[(-h, y_bar), (h, y_bar), (h, h), (-h, h)],        # crossbar
             [(-b, -h), (b, -h), (b, y_bar), (-b, y_bar)]]      # stem
    verts, faces = [], []
    for q in quads:
        k = len(verts)
        verts.extend([(x - cx, y - cy, z0) for x, y in q])       # 0..3 bottom
        verts.extend([(x - cx, y - cy, z1) for x, y in q])       # 4..7 top
        faces.extend([(k, k + 2, k + 1), (k, k + 3, k + 2),                  # bottom
                      (k + 4, k + 5, k + 6), (k + 4, k + 6, k + 7)])         # top
        for j in range(4):                                                   # walls
            a, bb = k + j, k + (j + 1) % 4
            faces.extend([(a, bb, bb + 4), (a, bb + 4, a + 4)])
    return np.asarray(verts, dtype=np.float32), np.asarray(faces, dtype=np.uint32)


def _dtheta_deg(a_deg, b_deg):
    """Signed smallest difference between two headings in degrees, in ``(-180, 180]``."""
    return (float(a_deg) - float(b_deg) + 180.0) % 360.0 - 180.0


def fix_spread(fixes):
    """How far apart the markers' answers landed: ``{n, xy_cm, theta_deg, worst}``.

    **This is the number that says whether the marker positions are right.**  Every marker
    on the shape claims to see the same rigid body, so their implied shape poses have to
    agree; what they actually disagree by is measurement noise plus whatever is wrong in
    ``TEE_MARKER_POS_CM``.  A centimetre or two of spread is one of the positions being
    mis-measured (or a marker glued on turned differently from the rest, which shows up in
    ``theta_deg`` while ``xy_cm`` stays small).  A couple of millimetres is the depth return
    and the plane heading jittering, which is as good as this hardware gets.

    Worst pair rather than a standard deviation, and ``worst`` names the two ids: with four
    markers the useful question is *which one disagrees with the others*, and a spread
    summarized to one number cannot answer it.  ``None`` for ``xy_cm`` / ``theta_deg`` when
    fewer than two markers were measured -- one marker never disagrees with itself, and
    reporting 0 there would read as "checked and fine" when nothing was checked.
    """
    ids = sorted(fixes)
    out = {"n": len(ids), "xy_cm": None, "theta_deg": None, "worst": None}
    if len(ids) < 2:
        return out
    worst_xy, worst_pair, worst_th = -1.0, None, 0.0
    for k, a in enumerate(ids):
        for b in ids[k + 1:]:
            fa, fb = fixes[a], fixes[b]
            d = math.hypot(fa["x_cm"] - fb["x_cm"], fa["y_cm"] - fb["y_cm"])
            worst_th = max(worst_th,
                           abs(_dtheta_deg(fa["theta_deg"], fb["theta_deg"])))
            if d > worst_xy:
                worst_xy, worst_pair = d, (a, b)
    out.update(xy_cm=worst_xy, theta_deg=worst_th, worst=worst_pair)
    return out


def markers_markdown(obs):
    """Every measured marker's own answer for the shape, side by side with the chosen one.

    One row per marker: where the camera saw the marker, what shape pose that implies, and
    how far that is from the row being used.  The deltas are the point -- absolute numbers
    all look plausible, and a mis-measured position only shows up as one row sitting a
    centimetre off the others.
    """
    fixes = obs.get("fixes") or {}
    chosen = obs.get("tee_id")
    lines = ["**markers** -- each one's own answer, and how far it is from the one in use",
             "```",
             f"{'id':<4s}{'marker x, y cm':<20s}{'-> shape x, y cm':<22s}"
             f"{'theta':<10s}{'d_xy':<8s}d_th"]
    ref = fixes.get(chosen)
    for mid in sorted(fixes):
        f = fixes[mid]
        if ref is None or mid == chosen:
            dxy, dth = "  --", "  --"
        else:
            dxy = f"{math.hypot(f['x_cm'] - ref['x_cm'], f['y_cm'] - ref['y_cm']):.2f}"
            dth = f"{_dtheta_deg(f['theta_deg'], ref['theta_deg']):+.2f}"
        mark = "*" if mid == chosen else " "
        lines.append(
            f"{mark}{mid:<3d}({f['marker_x_cm']:+7.2f}, {f['marker_y_cm']:+7.2f})   "
            f"({f['x_cm']:+7.2f}, {f['y_cm']:+7.2f})     "
            f"{f['theta_deg']:+8.2f}  {dxy:>6s}  {dth:>7s}")
    blocked = obs.get("tee_ids_blocked") or []
    for mid in blocked:
        lines.append(f" {mid:<3d}not visible")
    lines.append("```")
    lines.append(f"`*` = the marker the pose above came off (id {chosen})")

    sp = obs.get("fix_spread") or {}
    if sp.get("xy_cm") is None:
        lines.append(f"\nspread: **not checked** -- only {sp.get('n', 0)} marker measured. "
                     f"Turn on _compare every visible marker_, or get a second one in view.")
    else:
        a, b = sp["worst"]
        lines.append(f"\nspread: worst pair **id {a} vs id {b}**, "
                     f"**{sp['xy_cm']:.2f} cm** apart, "
                     f"heading up to **{sp['theta_deg']:.2f} deg** apart "
                     f"(over {sp['n']} markers)")
        if sp["xy_cm"] > MARKER_AGREE_WARN_CM:
            lines.append(f"\n**> {MARKER_AGREE_WARN_CM:.1f} cm apart** -- one of these two "
                         f"positions in `TEE_MARKER_POS_CM` is wrong.  Force each id in turn "
                         f"with the dropdown and watch which one moves the shape off the "
                         f"real thing; a heading gap with a small `d_xy` means that marker "
                         f"is glued on turned differently from the others.")
    if not obs.get("fix_from_preferred", True):
        lines.append(f"\n_id {obs.get('tee_id_preferred')} was requested but could not be "
                     f"read this frame -- fell back to id {chosen}_")
    return "\n".join(lines) + "\n\n"


def theta_markdown(obs):
    """Which method gave ``theta``, and how far the one it replaced disagrees.

    pnp on a near-fronto-parallel 5.5 cm square is the noisy estimate this stack moved off
    of, so it stays on screen as the contrast rather than disappearing: a large gap here
    is pnp being unreliable, not the plane drifting.
    """
    src = obs.get("theta_source", "?")
    mount = obs.get("marker_yaw_on_tee_deg", 0.0)
    line = (f"theta from **{src}**: marker reads "
            f"{obs.get('marker_theta_deg', float('nan')):+.2f}°, less {mount:+.2f}° for "
            f"the mount = {obs.get('theta_deg', float('nan')):+.2f}° for the shape "
            f"(plane height {obs.get('tee_height_cm', 0.0):.2f} cm)")
    pnp, plane = obs.get("theta_pnp_deg"), obs.get("theta_plane_deg")
    if pnp is not None and plane is not None:
        gap = (plane - pnp + 180.0) % 360.0 - 180.0
        line += f" -- pnp would say {pnp:+.2f}° ({gap:+.2f}° away)"
    elif pnp is None:
        line += " -- pnp gave nothing to compare against"
    px, py = obs.get("plane_x_cm"), obs.get("plane_y_cm")
    if px is not None:
        line += (f"\n\nplane also places the marker at ({px:+.2f}, {py:+.2f}) cm, vs "
                 f"depth's ({obs['marker_x_cm']:+.2f}, {obs['marker_y_cm']:+.2f})")
    return line + "\n\n"


def origins_cm(obs):
    """The three origins, each as ``(label, (x, y, z) table cm, (x, y, z) camera cm)``.

    All three are z = 0: they sit on the table plane, which is what the table frame's
    ``+z = 0`` means.  The camera-frame column is the one that carries information -- in
    the table frame the first is ``(0, 0, 0)`` by definition and the third is the measured
    constant ``FIELD_CENTER_CM`` -- and it comes from the pose ``locate_shape`` returns,
    so it is this frame's, not a stale one.
    """
    pose = (obs or {}).get("pose")
    rows = [(f"table (id {ORIGIN_ID})", (0.0, 0.0)),
            ("shape", (obs["x_cm"], obs["y_cm"])),
            ("sim env", FC.FIELD_CENTER_CM)]
    out = []
    for label, (x, y) in rows:
        cam_cm = (None if pose is None else
                  tuple(float(v * 100.0) for v in to_cam_m(pose, (x, y), 0.0)))
        out.append((label, (float(x), float(y), 0.0), cam_cm))
    return out


def origins_markdown(obs):
    """The three origins as a GUI table: where each one is, and where the camera sees it."""
    lines = ["**origins** -- all on the table plane, so z = 0 in table cm", "```",
             f"{'':<15s}{'table cm (x, y, z)':<26s}camera cm (x, y, z)"]
    for label, (x, y, z), cam in origins_cm(obs):
        c = ("--" if cam is None else
             f"({cam[0]:+7.1f}, {cam[1]:+7.1f}, {cam[2]:+7.1f})")
        lines.append(f"{label:<15s}({x:+7.2f}, {y:+7.2f}, {z:+5.2f})   {c}")
    lines.append("```")
    return "\n".join(lines) + "\n\n"


def _origin_axes(server, name, xy_cm, length_cm, label, z_cm=0.0):
    """An RGB axes triad at a table-frame point, with a floating label above it.

    One helper for all three origins so they are drawn identically and cannot drift apart
    in size or convention: red ``+x``, green ``+y``, blue ``+z`` out of the table.
    """
    node = server.scene.add_frame(name, show_axes=True, axes_length=length_cm,
                                  axes_radius=length_cm / 60.0,
                                  origin_radius=length_cm / 20.0,
                                  position=(float(xy_cm[0]), float(xy_cm[1]), float(z_cm)))
    server.scene.add_label(f"{name}/label", label,
                           position=(0.0, 0.0, length_cm * 0.45))
    return node


def visualize(cam, detect_table, detect_tee, port=8080, rate_hz=15.0, **kw):
    """A live viser scene: the table drawn once, the T re-posed from ``locate_shape``.

    The T is a viser *frame* with the mesh and outline as children, so each update is one
    position and one quaternion rather than a re-uploaded mesh.  When an observation is
    missed the last good pose stays on screen and the reason is shown in the GUI, which is
    far easier to work with than a shape that blinks out.

    **The marker panel is how the four-marker setup gets checked against the real table.**
    Every marker on the shape is an independent measurement of the same rigid body, so they
    all have to give the same shape pose; whether they actually do is the one thing that says
    the positions in ``TEE_MARKER_POS_CM`` are right.  The panel makes that visible three
    ways at once:

    * **a triad per marker** at the shape centre *that marker alone* reports (``/fix<id>``,
      plus its square at ``/tee_marker<id>``).  Four triads in a pile means four correct
      positions; one sitting off the pile is that marker's row being wrong, and you can read
      which way and by how much straight off the scene.
    * **the table of numbers** -- ``markers_markdown``: every marker's own answer with its
      ``d_xy`` / ``d_th`` from the one in use, and ``fix_spread``'s worst pair called out by
      id when it exceeds ``MARKER_AGREE_WARN_CM``.
    * **the "pose from" dropdown** -- force one id and the whole scene is driven by that
      marker alone, which is the direct test: pick each in turn and the drawn T should not
      move.  Cover the forced marker with your hand and it falls back to the ascending walk,
      with the readout saying it did, which is the blocked-marker case being exercised
      deliberately rather than waited for.

    Turning *measure every visible marker* off drops back to exactly what a run does -- one
    marker measured, the rest not touched -- so the comparison costs nothing when not wanted.
    """
    import viser

    mode = kw.get("mode", "centroid")
    size_cm, bar_cm = kw.get("size_cm", TEE_SIZE_CM), kw.get("bar_cm", TEE_BAR_CM)
    stem_cm = kw.get("stem_cm", None)
    length_cm, width_cm = kw.get("length_cm", LENGTH_CM), kw.get("width_cm", WIDTH_CM)
    thick_cm = sim_tee_thickness_cm(size_cm) or size_cm / 6.0
    top_z = thick_cm + 0.5                       # what rides on top of the shape

    server = viser.ViserServer(port=port)
    server.scene.set_up_direction("+z")
    server.scene.world_axes.visible = False

    # ---- the environment ----
    # Three different things, in three colours, because confusing them is exactly the
    # alignment bug this is meant to catch:
    #   grey   the nominal table rectangle -- --length x --width, carried out from the
    #          one ID-10 marker.  Drawn, not measured: with a single marker nothing is
    #          observing the far corners any more.
    #   green  the normalization box -- what real_world_params.json claims the env is
    #   dark   the actual env geometry -- the walls and blocks push_t_demo_sim collides
    #          against, loaded from the same .obj and put through the same transform
    env = kw.get("env")
    if env is not None:
        server.scene.add_mesh_simple("/env/fill", env["vertices"], env["faces"],
                                     color=COL_ENV, flat_shading=True)
        for i, ring in enumerate(env["rings"]):
            server.scene.add_line_segments(f"/env/ring{i}", _loop_segments(ring, 0.01),
                                           colors=COL_ENV_EDGE, thickness=3.0)
    box = env_box_cm()
    server.scene.add_line_segments("/env/norm_box", _loop_segments(box, 0.03),
                                   colors=COL_SIM_ENV, thickness=2.0)
    ring = [xy for _, xy in table_corners_cm(length_cm, width_cm)]
    server.scene.add_line_segments("/env/table", _loop_segments(ring), colors=COL_TABLE,
                                   thickness=4.0)
    # only one marker is on the table now, and it is the origin
    server.scene.add_line_segments(f"/env/marker{ORIGIN_ID}",
                                   _square((0.0, 0.0), kw.get("marker_len_cm",
                                                              MARKER_LEN_CM)),
                                   colors=COL_MARKER, thickness=2.5)
    # Framing, the way run_viser does it: the bounds of everything drawn, padded by
    # 1.15, with the floor and the grid filling that square.
    pts = [np.asarray(box, dtype=float), np.asarray(ring, dtype=float)]
    if env is not None and env["rings"]:
        pts.append(np.vstack(env["rings"]))
    allpts = np.vstack(pts)
    lo, hi = allpts.min(axis=0), allpts.max(axis=0)
    cx, cy = float((lo[0] + hi[0]) / 2.0), float((lo[1] + hi[1]) / 2.0)
    span = float(max(hi - lo) * 1.15)
    server.scene.add_box("/floor", color=COL_FLOOR, dimensions=(span, span, 2.0),
                         position=(cx, cy, -1.0))
    server.scene.add_grid("/grid", width=span, height=span, plane="xy",
                          cell_size=10.0, section_size=50.0, position=(cx, cy, 0.02))
    server.scene.add_light_directional("/sun", color=(255, 255, 255), intensity=2.0,
                                       position=(cx + 100, cy - 150, 400))

    # ---- the three origins ----
    # Every frame in play, drawn where it actually is, because "which way is +x" and
    # "what is this pose measured from" are the two questions the numbers alone never
    # answer.  Each gets an RGB triad (red +x, green +y, blue +z) and a label:
    #   table    the ID-10 marker, (0, 0) by definition -- what the camera reports in
    #   sim env  the field centre, which is normalized (0, 0) -- what the planner sees
    #   shape    the T's own body origin, a child of /tee so it rides with the shape
    _origin_axes(server, "/origin", (0.0, 0.0), size_cm,
                 f"table origin - id {ORIGIN_ID}")
    _origin_axes(server, "/env/origin", FC.FIELD_CENTER_CM, size_cm * 1.25,
                 "sim env origin - normalized (0, 0)")

    # ---- the shape ----
    verts, faces = tee_mesh_cm(size_cm, bar_cm, stem_cm, mode, z=0.0,
                               thickness_cm=thick_cm)
    tee = server.scene.add_frame("/tee", show_axes=False, position=(cx, cy, 0.0))
    server.scene.add_mesh_simple("/tee/mesh", verts, faces, color=COL_TEE,
                                 flat_shading=True)
    server.scene.add_line_segments("/tee/outline",
                                   _loop_segments(tee_body_outline_cm(size_cm, bar_cm,
                                                                      stem_cm, mode),
                                                  top_z),
                                   colors=(255, 255, 255), thickness=2.0)
    # a stub along the T's own +y, so the heading is readable at a glance
    server.scene.add_line_segments(
        "/tee/heading", np.array([[[0.0, 0.0, top_z], [0.0, size_cm * 0.75, top_z]]],
                                 dtype=np.float32), colors=COL_HEADING, thickness=3.0)
    # The shape's own origin -- the centroid push_t_demo_sim poses about -- as a child of
    # /tee, so the triad turns with the T and shows what theta is measured about.  Its
    # (x, y) is the origin exactly; only z is lifted to the prism's top face, or the triad
    # would be buried inside the solid shape.
    _origin_axes(server, "/tee/origin", (0.0, 0.0), size_cm * 0.8,
                 f"shape origin - {mode}", z_cm=top_z)
    # ---- one set of nodes per marker on the shape ----
    # Every usable id gets its own square (where the camera saw that marker) and its own
    # little triad (where THAT marker alone says the shape centre is).  All the triads
    # landing on top of each other is the whole test: they are four independent measurements
    # of one rigid body, so if a position in TEE_MARKER_POS_CM is wrong, that marker's triad
    # sits visibly off the pile while the others agree.
    #
    # Positions are updated on frames rather than by rebuilding line geometry, which is the
    # pattern the rest of this scene already uses -- one position and one quaternion per
    # node per frame.
    ids_usable = list(kw.get("tee_ids") or tee_marker_ids(kw.get("positions")))
    m_nodes, fix_nodes = {}, {}
    for mid in ids_usable:
        m_nodes[mid] = server.scene.add_frame(f"/tee_marker{mid}", show_axes=False,
                                              position=(cx, cy, 0.0), visible=False)
        server.scene.add_line_segments(
            f"/tee_marker{mid}/sq",
            _square((0, 0), kw.get("tee_len_cm", TEE_LEN_CM), top_z),
            colors=COL_MARKER, thickness=2.0)
        server.scene.add_label(f"/tee_marker{mid}/label", f"id {mid}",
                               position=(0.0, 0.0, top_z + 1.0))
        fix_nodes[mid] = _origin_axes(server, f"/fix{mid}", (cx, cy), size_cm * 0.45,
                                      f"id {mid} -> centre", z_cm=top_z + 0.5)
        fix_nodes[mid].visible = False

    @server.on_client_connect
    def _(client):
        client.camera.position = (cx, cy - 1e-3, span)
        client.camera.look_at = (cx, cy, 0.0)
        client.camera.up = (0.0, 1.0, 0.0)

    server.gui.add_markdown(
        f"**T shape on the table** -- table centimetres, origin at the ID-{ORIGIN_ID} "
        f"marker, `+x` along the length, `+y` along the width.\n\n"
        f"`x, y` from RealSense **depth**, `theta` from the **plane** method, centre = "
        f"`{mode}` (the body origin `push_t_demo_sim` poses about).\n\n"
        f"The shape carries **{len(ids_usable)} usable markers** ({ids_usable}); the lowest "
        f"id in view is the one read, so one going under the arm is not a lost frame. Use "
        f"the **markers** panel to check they agree -- see the per-marker triads and the "
        f"`d_xy` column.\n\n"
        f"The frame comes from that one marker, so its direction is the marker's own "
        f"heading -- `--yaw-offset` is "
        f"{kw.get('yaw_offset_deg', YAW_OFFSET_DEG):+.2f} deg, plus the measured "
        f"{ORIGIN_EXTRA_YAW_DEG:+.2f} deg quarter-turn `table_pose` adds. "
        f"The RGB axes at the "
        f"origin are that frame: red `+x`, green `+y`, blue `+z` out of the table.\n\n"
        f"Grey: the nominal {length_cm:.0f} x {width_cm:.0f} cm table rectangle, drawn "
        f"out from that marker rather than measured. "
        f"Green: the box `real_world_params.json` normalizes to. "
        f"Dark: the real env geometry -- the walls and blocks `push_t_demo_sim` "
        f"collides against."
        + (f"\n\n_env: {env['n_polys']} polygons, {env['n_holes']} holes_"
           if env is not None else
           f"\n\n_env geometry not loaded: {load_env.last_error}_"))
    txt = server.gui.add_markdown("_waiting for the table marker and a shape marker_")

    # ---- the marker panel: what this scene is for testing ----
    # AUTO is what a real run does (lowest visible id).  Forcing one id is how the positions
    # in TEE_MARKER_POS_CM get checked: pick each in turn and watch whether the drawn shape
    # stays on the real one.  A forced id that is blocked falls back rather than blanking, and
    # the readout says so, so covering a marker on purpose is a valid thing to try.
    AUTO = "auto - lowest id in view"
    with server.gui.add_folder("markers"):
        pick = server.gui.add_dropdown("pose from", (AUTO, *(f"id {i}" for i in ids_usable)),
                                       initial_value=AUTO)
        compare = server.gui.add_checkbox("measure every visible marker", True)
        show_marker = server.gui.add_checkbox("show the marker squares", True)
        show_fixes = server.gui.add_checkbox("show each marker's own centre", True)
        server.gui.add_markdown(
            "_Each marker's triad is the shape centre **it alone** reports. They should sit "
            "on top of each other; one sitting off the pile is that marker's `(x_cm, y_cm)` "
            "in `TEE_MARKER_POS_CM` being wrong. Cover one with your hand to watch the "
            "fallback._")

    print(f"viser on http://localhost:{port}   (ctrl-c to stop)")
    period = 1.0 / max(rate_hz, 1e-6)
    kw_locate = {k: v for k, v in kw.items()
                 if k not in ("length_cm", "width_cm", "env")}
    n_seen, n_miss = 0, 0
    try:
        while True:
            t0 = time.time()
            # The two GUI controls are read fresh every frame rather than through an
            # on_update callback: locate_shape is called from here anyway, so passing them
            # straight in keeps one path into the measurement and no state to fall out of
            # sync with the widgets.
            forced = None if pick.value == AUTO else int(pick.value.split()[-1])
            try:
                obs = locate_shape(cam, detect_table, detect_tee, prefer_id=forced,
                                   compare=bool(compare.value), **kw_locate)
            except Exception as exc:                       # noqa: BLE001 -- keep the view up
                obs, locate_shape.last_miss = None, f"{type(exc).__name__}: {exc}"

            if obs is None:
                n_miss += 1
                for mid in ids_usable:
                    m_nodes[mid].visible = False
                    fix_nodes[mid].visible = False
                txt.content = (f"**no fix** -- {locate_shape.last_miss}\n\n"
                               f"{n_seen} located / {n_miss} missed")
            else:
                n_seen += 1
                th = obs["theta_rad"]
                tee.position = (obs["x_cm"], obs["y_cm"], 0.0)
                tee.wxyz = (math.cos(th / 2.0), 0.0, 0.0, math.sin(th / 2.0))
                # One square and one triad per marker that was actually measured; the rest
                # are hidden, so the scene shows exactly what the camera had this frame.
                for mid in ids_usable:
                    f = obs["fixes"].get(mid)
                    if f is None:
                        m_nodes[mid].visible = False
                        fix_nodes[mid].visible = False
                        continue
                    mth = math.radians(f["marker_theta_deg"])
                    m_nodes[mid].position = (f["marker_x_cm"], f["marker_y_cm"], 0.0)
                    m_nodes[mid].wxyz = (math.cos(mth / 2.0), 0.0, 0.0,
                                         math.sin(mth / 2.0))
                    m_nodes[mid].visible = bool(show_marker.value)
                    fix_nodes[mid].position = (f["x_cm"], f["y_cm"], 0.0)
                    fix_nodes[mid].visible = bool(show_fixes.value)
                other = [i for i in obs["tee_ids_seen"] if i != obs["tee_id"]]
                blocked = obs["tee_ids_blocked"]
                nx, ny, nth = FC.table_to_sim(obs["x_cm"], obs["y_cm"], obs["theta_rad"])
                txt.content = (
                    f"**shape**  x = {obs['x_cm']:+.2f} cm   y = {obs['y_cm']:+.2f} cm   "
                    f"theta = {obs['theta_deg']:+.2f}°   (z = 0, on the table)\n\n"
                    f"marker  **id {obs['tee_id']}** at "
                    f"({obs['marker_x_cm']:+.2f}, {obs['marker_y_cm']:+.2f}) cm, "
                    f"z = {obs['marker_z_cm']:+.2f} cm, offset "
                    f"({obs['marker_offset_cm'][0]:+.2f}, "
                    f"{obs['marker_offset_cm'][1]:+.2f}) cm in the T's frame\n\n"
                    f"also in view {other or 'none'}; "
                    f"not visible {blocked or 'none'} "
                    f"(of {obs['tee_ids_wanted']})\n\n"
                    + markers_markdown(obs)
                    + theta_markdown(obs)
                    + origins_markdown(obs) +
                    f"normalized  ({nx:+.3f}, {ny:+.3f}) -- the planner's "
                    f"[-0.5, 0.5] field\n\n"
                    + (f"sim  ({obs['sim_pose'][0]:+.3f}, {obs['sim_pose'][1]:+.3f}, "
                       f"{obs['sim_pose'][2]:+.3f} rad)\n\n" if obs.get("sim_pose")
                       else "") +
                    (f"frame  id {ORIGIN_ID} {obs.get('frame_source', '?')}, "
                     f"rms {obs.get('frame_rms', float('nan')):.3f} "
                     f"{obs.get('frame_rms_unit', '')}\n\n"
                     f"{n_seen} located / {n_miss} missed"))
            time.sleep(max(0.0, period - (time.time() - t0)))
    except KeyboardInterrupt:
        print(f"\nstopped: {n_seen} located, {n_miss} missed")


# ======================================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--serial", default=None, help="RealSense serial (default: first found)")
    ap.add_argument("--dict", default=ARUCO_DICT, metavar="NAME",
                    help="dictionary of the ID-%d table marker (default %%(default)s)"
                         % ORIGIN_ID)
    ap.add_argument("--tee-dict", default=TEE_DICT, metavar="NAME",
                    help="dictionary of the T marker (default %(default)s)")
    ap.add_argument("--tee-ids", type=int, nargs="+", default=None, metavar="ID",
                    help="ids of the markers on the T, tried in increasing order until one "
                         "reads cleanly (default: every id with a position filled in in "
                         "frame_conversions.TEE_MARKER_POS_CM, i.e. "
                         f"{list(tee_marker_ids())} of {list(TEE_MARKER_IDS)})")
    ap.add_argument("--marker-len", type=float, default=MARKER_LEN_CM, metavar="CM",
                    help="side length of the ID-%d table marker -- the whole table "
                         "frame's metric scale (default %%(default)s)" % ORIGIN_ID)
    ap.add_argument("--yaw-offset", type=float, default=YAW_OFFSET_DEG, metavar="DEG",
                    help="heading of the table marker's own +x edge, deg CCW from table "
                         "+x; the table axes are spun back by it, so this is what keeps "
                         "x and y from coming out swapped (default %(default)s, from "
                         "real_world_params.json)")
    ap.add_argument("--tee-len", type=float, default=TEE_LEN_CM, metavar="CM",
                    help="side length of the markers on the T (all the same); only the pnp "
                         "cross-check depends on it now that theta comes from the plane "
                         "(default %(default)s)")
    ap.add_argument("--tee-height", type=float, default=None, metavar="CM",
                    help="how far the T's markers ride above the table, cm -- the height "
                         "the plane theta is read at (default: the shape's own thickness)")
    ap.add_argument("--length", type=float, default=LENGTH_CM, metavar="CM",
                    help="table extent along +x from the marker; drawing only "
                         "(default %(default)s)")
    ap.add_argument("--width", type=float, default=WIDTH_CM, metavar="CM",
                    help="table extent along +y from the marker; drawing only "
                         "(default %(default)s)")
    ap.add_argument("--tee-size", type=float, default=TEE_SIZE_CM, metavar="CM",
                    help="the T's bounding box, both ways (default %(default)s)")
    ap.add_argument("--tee-bar", type=float, default=None, metavar="CM",
                    help="crossbar thickness; default is read from the sim mesh "
                         f"({SIM_SHAPE_ZUP}) so the shape matches what the planner "
                         f"was trained on, else {TEE_BAR_CM}")
    ap.add_argument("--tee-stem", type=float, default=None, metavar="CM",
                    help="stem width; same source as --tee-bar")
    ap.add_argument("--marker-from-top", type=float, default=MARKER_FROM_TOP_CM,
                    metavar="CM",
                    help=f"id-{DEFAULT_TEE_MARKER_ID} marker centre, measured down from "
                         f"the T's top edge; overrides THAT id's y only -- the other "
                         f"markers are hand-placed, so theirs come from "
                         f"TEE_MARKER_POS_CM (default %(default)s)")
    ap.add_argument("--marker-yaw-on-tee", type=float, default=MARKER_YAW_ON_TEE_DEG,
                    metavar="DEG",
                    help="heading of a T marker's own +x edge in the T's body frame, "
                         "deg CCW -- the turn taken off the measured heading to get the "
                         "shape's, and the turn the offset is rotated by.  One value for "
                         "ALL the markers, since they are glued on the same way round "
                         "(default %(default)s)")
    ap.add_argument("--center", choices=CENTER_MODES, default="centroid",
                    help="what 'the centre' means: the area centroid, as the sim uses, or "
                         "the middle of the bounding box (default %(default)s)")
    ap.add_argument("--width-px", type=int, default=1280, metavar="PX")
    ap.add_argument("--height-px", type=int, default=720, metavar="PX")
    ap.add_argument("--env", default=SIM_ENV, metavar="OBJ",
                    help="the environment push_t_demo_sim pushes against; its footprint "
                         "is the obstacle set (default %(default)s)")
    ap.add_argument("--no-env", action="store_true",
                    help="skip loading the env geometry (draw only the normalization box)")
    ap.add_argument("--port", type=int, default=8080, help="viser port (default %(default)s)")
    ap.add_argument("--rate", type=float, default=15.0, metavar="HZ",
                    help="how often to locate the shape (default %(default)s)")
    args = ap.parse_args()

    # The planner was trained on one particular T, so take its proportions from that mesh
    # unless the user overrides them.
    dims = sim_tee_dims(args.tee_size)
    if dims is None:
        bar, stem = TEE_BAR_CM, TEE_STEM_CM
        print(f"note: {SIM_SHAPE_ZUP} not readable -- falling back to bar {bar} / "
              f"stem {stem} cm; check this matches the printed T")
    else:
        bar, stem = dims
        print(f"T proportions from {SIM_SHAPE_ZUP}: bar {bar:.3f} cm, stem {stem:.3f} cm")
    bar = args.tee_bar if args.tee_bar is not None else bar
    stem = args.tee_stem if args.tee_stem is not None else stem

    info = sim_alignment(args.tee_size, bar, stem, args.center)
    print(f"T {args.tee_size:.1f} x {args.tee_size:.1f} cm, bar {bar:.2f}, stem "
          f"{stem:.2f} cm, centre = {args.center} "
          f"at ({info['centroid_local_cm'][0]:+.3f}, {info['centroid_local_cm'][1]:+.3f}) "
          f"cm from the bbox middle")

    # ---- the markers on the shape ----
    # The table in frame_conversions is the source; the two CLI flags are overrides on top
    # of it, kept separate so `python locate_functions.py` with no flags is exactly what
    # push_t_realworld_run sees.  --marker-yaw-on-tee moves every row (one mount angle for
    # all four markers), --marker-from-top only id 1's y, the one row that comes from the
    # geometry rather than from a ruler.
    positions = dict(TEE_MARKER_POS_CM)
    if args.marker_yaw_on_tee != MARKER_YAW_ON_TEE_DEG:
        positions = {i: (None if v is None else (v[0], v[1], args.marker_yaw_on_tee))
                     for i, v in positions.items()}
    if args.marker_from_top != MARKER_FROM_TOP_CM:
        row = positions.get(DEFAULT_TEE_MARKER_ID) or (args.tee_size / 2.0, 0.0)
        positions[DEFAULT_TEE_MARKER_ID] = (row[0], -abs(args.marker_from_top),
                                            args.marker_yaw_on_tee)
        print(f"--marker-from-top {args.marker_from_top:.2f} cm: id "
              f"{DEFAULT_TEE_MARKER_ID} moved to "
              f"({row[0]:+.2f}, {-abs(args.marker_from_top):+.2f}) cm")
    ready = tee_marker_ids(positions)
    wanted = tuple(sorted({int(i) for i in args.tee_ids})) if args.tee_ids else ready
    missing = [i for i in wanted if i not in ready]
    if missing:
        raise SystemExit(f"--tee-ids names {missing}, which have no position measured -- "
                         f"fill in frame_conversions.TEE_MARKER_POS_CM first")
    if not wanted:
        raise SystemExit("no shape marker is usable -- measure each marker's centre from "
                         "the T's top-left corner (+x right, -y DOWN) into "
                         "frame_conversions.TEE_MARKER_POS_CM")
    print(f"markers on the shape -- position measured from the T's TOP-LEFT corner "
          f"(+x right, -y down), then the offset out to the {args.center}:")
    bad = []
    for mid in sorted(positions):
        if mid not in ready:
            print(f"  id {mid}   NOT MEASURED -- put (x_cm, y_cm) from the top-left "
                  f"corner in frame_conversions.TEE_MARKER_POS_CM[{mid}] to use it")
            continue
        px, py, _ = tee_marker_pos(mid, positions)
        note = "" if mid in wanted else "   [excluded by --tee-ids]"
        try:
            dx, dy, yaw = tee_marker_offset(mid, positions, args.center, args.tee_size,
                                            bar, stem)
        except ValueError as exc:
            bad.append(mid)
            print(f"  id {mid}   ({px:+6.2f}, {py:+6.2f}) cm   REJECTED -- {exc}")
            continue
        bx, by = tee_topleft_to_body_cm(px, py, args.tee_size)
        print(f"  id {mid}   ({px:+6.2f}, {py:+6.2f}) cm  ->  body "
              f"({bx:+6.2f}, {by:+6.2f})  ->  offset ({dx:+6.2f}, {dy:+6.2f}) cm, "
              f"mount {yaw:+.1f} deg{note}")
    if [i for i in wanted if i in bad]:
        raise SystemExit(f"shape marker ids {[i for i in wanted if i in bad]} are measured "
                         f"to a point that is not on the T -- fix them above first")
    print(f"reading them in increasing id order {list(wanted)}: the lowest one in view "
          f"wins, so a marker under the arm costs nothing.  The offset is turned by "
          f"{-args.marker_yaw_on_tee:+.1f} deg for the mount before it is applied "
          f"(shape heading = marker heading {-args.marker_yaw_on_tee:+.1f} deg)")
    if len(wanted) < 2:
        print(f"NOTE: only {len(wanted)} marker is usable, so a blocked marker is still a "
              f"lost frame.  Measure the rest in to get the redundancy.")
    if info["bridge"] == "push_t_demo_realworld":
        ex, ey = info["env_cm"]
        cx, cy = info["env_center_cm"]
        print(f"sim bridge: env {ex:.1f} x {ey:.1f} cm centred at ({cx:+.1f}, {cy:+.1f}) "
              f"cm, rotation {info['rotation_deg']:+.1f} deg, "
              f"{info['sim_units_per_cm']:.4f} sim units/cm")
        # The env has to sit inside the measured table, or the planner is normalizing to a
        # box that is partly off the table.
        lo = (cx - ex / 2.0, cy - ey / 2.0)
        hi = (cx + ex / 2.0, cy + ey / 2.0)
        if lo[0] < 0.0 or hi[0] > args.length or lo[1] < 0.0 or hi[1] > args.width:
            print(f"WARNING: the env box spans x [{lo[0]:+.1f}, {hi[0]:+.1f}] "
                  f"y [{lo[1]:+.1f}, {hi[1]:+.1f}] cm, which is not inside the "
                  f"{args.length:.0f} x {args.width:.0f} cm table "
                  f"(x [0, {args.length:.0f}], y [0, {args.width:.0f}]).\n"
                  f"         Set environment.offset_to_origin_cm in "
                  f"real_world_params.json to the env centre in THIS frame -- "
                  f"({args.length / 2:.1f}, {args.width / 2:.1f}) puts it mid-table.\n"
                  f"         NOTE the origin moved to the ID-{ORIGIN_ID} corner and +y "
                  f"flipped: a value measured against the old four-marker frame becomes "
                  f"(x, y + {args.width:.1f}) here.")
    else:
        print(f"sim bridge: {info['bridge']}")

    env = None if args.no_env else load_env(args.env)
    if env is not None:
        allpts = np.vstack(env["rings"]) if env["rings"] else np.zeros((1, 2))
        lo, hi = allpts.min(axis=0), allpts.max(axis=0)
        print(f"env {os.path.basename(args.env)}: {env['n_polys']} polygon(s), "
              f"{env['n_holes']} hole(s), {len(env['faces'])} faces; spans "
              f"x [{lo[0]:+.1f}, {hi[0]:+.1f}]  y [{lo[1]:+.1f}, {hi[1]:+.1f}] cm "
              f"in the table frame")
    elif not args.no_env:
        print(f"env geometry not loaded ({load_env.last_error}) -- "
              f"drawing the normalization box only")

    if not realsense_present():
        raise SystemExit("no RealSense found -- locate_shape needs depth for x, y.")
    cam = RealSenseCamera(args.serial, args.width_px, args.height_px)
    print(cam.desc)
    try:
        visualize(cam, make_detector(args.dict), make_detector(args.tee_dict),
                  port=args.port, rate_hz=args.rate,
                  marker_len_cm=args.marker_len, yaw_offset_deg=args.yaw_offset,
                  tee_ids=wanted, positions=positions,
                  tee_len_cm=args.tee_len, tee_height_cm=args.tee_height,
                  length_cm=args.length, width_cm=args.width,
                  size_cm=args.tee_size, bar_cm=bar, stem_cm=stem,
                  mode=args.center, env=env)
    finally:
        cam.close()


if __name__ == "__main__":
    main()
