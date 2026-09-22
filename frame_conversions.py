"""frame_conversions -- every frame in the rig, and the conversions between them.

The whole chain, camera to arm, lives here so that each hop is one function with one
place to fix it::

    camera m  --to_table_cm-->  table cm  --aruco_to_center-->  shape cm
              --table_to_sim-->  normalized  --sim_to_robot-->  UR base m

The pieces that need a *camera model* (undistorting a pixel, projecting back to one) stay
in ``camera_test_id10``, and the pieces that need the *sim stack* stay behind
:func:`sim_bridge`; everything here is math on numbers that have already been measured, so
this module imports nothing heavier than numpy and can be used with no OpenCV, no torch
and no pymunk installed.

The frames
----------
camera      RealSense optical frame, metres.
table       origin at the **ID-10** corner marker, ``+x`` along the length, ``+y`` along
            the width, centimetres, ``theta`` CCW from ``+x`` -- what ``camera_test_id10``
            and ``locate_functions`` report in.
shape       the T's body frame: what a *marker* on it says, less that marker's mount
            offset.  Four markers carry it, on different parts of the shape, so one being
            covered by the arm is not a lost frame -- see ``TEE_MARKER_POS_CM``.
sim         the planner's **normalized** frame, the ``[-0.5, 0.5]^2`` square the network
            was trained on.
base        the UR base frame ``(x, y, z, rx, ry, rz)``, metres and rotation vector.

The core pair, inverses of each other:

``table_to_sim(x_cm, y_cm, theta_rad)``
    where the camera says something is -> where the planner thinks it is
``sim_to_table(x, y, theta_rad)``
    the way back, for turning a planned pose or path into something the arm can reach

The field is not centred on the marker -- the marker is a table corner, the field is out in
the middle of the table -- and the sim's axes do not point the same way as the table's, so
the conversion is centre, rotate, divide::

    delta = (x_cm, y_cm) - FIELD_CENTER_CM        # measured: 39.8 right, 32.5 up from ID 10
    norm  = R(ROTATION_DEG) @ delta / FIELD_CM    # FIELD_CM = 70 cm across
    theta_sim = theta + radians(ROTATION_DEG)     # ROTATION_DEG = 180

**The rotation is 180 degrees, about the field centre.**  The sim runs its axes the other
way round: ``sim +x`` is ``table -x`` and ``sim +y`` is ``table -y``, so a point 35 cm to
the *right* of the field centre on the table is at ``-0.5``, not ``+0.5``, for the planner.
It is a rotation and not a mirror, so the frame stays right-handed and a heading is simply
turned by ``pi``.  Note the order: the centre is subtracted **first** and the rotation acts
on that centred vector, which is what "rotate about the field centre" means -- rotating
about the ID-10 marker instead would move the field somewhere else entirely.

180 is its own inverse, which is a small mercy: unlike any other angle there is no CW/CCW
sign to get backwards here.

**The divisor is the whole 70 cm, not 35.**  The normalized square runs ``-0.5 .. +0.5``,
so it is *half* the field -- 35 cm, the centre-to-edge distance -- that has to land on
``0.5``, and ``35 / 70 = 0.5``.  Dividing by 35 would put the field edges at ``+-1.0`` and
every position the planner sees would be twice as far off centre as it really is.

So the field's own corners are the corners of the normalized square:

===================  ==================  ==============================
table cm             normalized
===================  ==================  ==============================
``( 39.8,  32.5)``   ``( 0.0,  0.0)``    the field centre
``(  4.8,  -2.5)``   ``(+0.5, +0.5)``    bottom-left on the table ...
``( 74.8,  67.5)``   ``(-0.5, -0.5)``    ... is top-right for the planner
===================  ==================  ==============================

Every number comes from ``real_world_params.json`` (``environment.offset_to_origin_cm``,
``env_len_cm`` / ``env_x_cm`` / ``env_y_cm``, ``rotation_deg``) so there is one place to
edit after re-measuring, and nothing here is hard-coded.  ``push_t_demo_realworld``
performs the same centre-rotate-divide from the same constants and then carries on into
pymunk mesh units (``* ENV_SCALE + ENV_CENTER``); :func:`to_sim_pose` is the one hop that
goes that far, and it reaches it through :func:`sim_bridge` so this module still imports
nothing but numpy.

    python frame_conversions.py            # print the constants and check every frame
"""

import json
import math
import os

import numpy as np

import shapes
from real_world_params import PARAMS

HERE = os.path.dirname(os.path.abspath(__file__))

ORIGIN_ID = 10                        # the marker the table frame is pinned to

# Table-frame centimetres of the FIELD CENTRE, measured from the ID-10 marker:
# +x to the right along the length, +y up along the width.
FIELD_CENTER_CM = PARAMS.offset_to_origin_cm

# Full width / height of the field on the table, cm -- the box that normalizes to
# ``[-0.5, 0.5]^2``.  Square in the sim (2denv4 is 350 x 350 mesh units), so keep them equal.
FIELD_X_CM = PARAMS.env_x_cm
FIELD_Y_CM = PARAMS.env_y_cm

# Rotation from the table axes into the sim axes, CCW degrees, about the field centre.
# 180 here: the sim's +x / +y run opposite the table's.
ROTATION_DEG = PARAMS.rotation_deg

NORM_HALF = 0.5                       # the normalized frame runs -NORM_HALF .. +NORM_HALF

# Full field width, cm -- the scalar behind FIELD_X_CM / FIELD_Y_CM, and the multiplier
# that turns a normalized coordinate into centimetres for the arm.
FIELD_CM = PARAMS.env_len_cm

# Sim -> base axis signs: the base axes run OPPOSITE the sim ones.  A 180 deg turn about
# p0, which is why p0 itself is still sim (0, 0): a rotation fixes its own centre.
# Measured, so it lives here rather than being folded into sim_to_robot's arithmetic
# where it could not be found again.
#
# Note what this cancels.  table_to_sim already turns table -> sim by ROTATION_DEG (180),
# so table -> sim -> base turns by 360: **base +x ends up running the same way as table
# +x**, and table -> base is a pure translation by p0.  The two rotations are kept
# separate rather than cancelled in the source because they are two different
# measurements, and folding them together would hide which one is wrong when the arm
# lands in the wrong place.
SIM_TO_BASE = np.array([-1.0, -1.0])

# Where arm_calibrate.py writes p0, relative to the repo.
STARTPOS_JSON = PARAMS.startpos_json

# ======================================================================================
# the shape's own geometry -- whichever shape the rig is running
# ======================================================================================
# The rig pushes ONE shape per process, chosen with ``--shape-name`` and resolved by
# ``shapes.py``; ``SHAPE`` is that choice, read once here so every default argument below
# is bound to it.  Selecting a different shape after this module is imported raises
# rather than half-applying -- see ``shapes.select``.
#
# The proportions are NOT constants any more.  They are read off the shape's own mesh,
# which is the shape the planner was trained on, and scaled by the field calibration
# (2denv4 is 350 mesh units across a 70 cm field, so one mesh unit is 2 mm).  The T comes
# out 12 x 12 cm with a 2 cm crossbar and a 2 cm stem -- exactly the numbers that used to
# be written here -- but now it cannot drift from the mesh, and a V or a rectangle needs
# no new constants at all.
# Run as a script (``python frame_conversions.py --shape-name V``) the flag has to be
# honoured HERE, before the constants below bind: by the time ``main()`` runs, every
# default argument in this file is already bound to whatever shape was active.  Imported
# as a library it is the entry point's job, which is why this is guarded -- a library
# reading sys.argv behind its caller's back is how two modules end up on two shapes.
if __name__ == "__main__":
    shapes.select_from_argv()

SHAPE = shapes.active()

# ``(width_cm, height_cm)`` of the shape's bounding box on the table.
SHAPE_SIZE_CM = SHAPE.size_cm()
# The single-number size, kept because the T is square and most of the rig says "the 12 cm
# T".  For a shape whose box is not square this is the LONGER side; anything that needs
# the box per axis uses SHAPE_SIZE_CM.
TEE_SIZE_CM = float(max(SHAPE_SIZE_CM))
# The T's crossbar / stem widths, centimetres.  Meaningful for the T alone -- they are
# what its outline used to be rebuilt from -- so they are measured back off the mesh for
# it and left None for any other shape.  Nothing in the measurement path reads them any
# more; the outline itself does that job now.
TEE_BAR_CM, TEE_STEM_CM = (2.0, 2.0) if SHAPE.name == 'T' else (None, None)
# The nominal placement of the lowest-id marker: horizontally centred, this far down from
# the shape's top edge.  Only the T has one -- a designed spot its measured row can be held
# against -- so it is None for a shape whose markers were simply put where they fit, and
# the self-test below then has nothing to compare and says so instead of crying slip.
MARKER_FROM_TOP_CM = SHAPE.marker_nominal_from_top_cm
# Heading of a shape marker's own +x edge measured in the shape's BODY frame, degrees CCW.
# The markers are stuck on turned: at 180 the +x edge runs along the shape's -x, so the
# body heading is the marker heading less 180, and the marker -> centre offset has to be
# turned by that same -180 before it can be added to a table-frame position.  0 would mean
# the two frames agree.  Measured off the live scene.
#
# **This one number covers every marker on the shape**, because they are all glued on the
# same way round, so it is NOT part of the per-marker table -- see TEE_MARKER_POS_CM,
# which holds only where each marker sits.  Nothing but the position differs per marker.
MARKER_YAW_ON_TEE_DEG = float(SHAPE.marker_yaw_deg)
CENTER_MODES = shapes.CENTER_MODES
# How far outside the shape's outline a measured marker centre may land before
# ``tee_marker_offset`` calls it a mistake.  Not zero: a marker glued right up to an edge
# measures a couple of millimetres over it, and a ruler read to the nearest millimetre on a
# 2 cm stem is easily that far out.  Big enough to allow a real placement, far too small to
# let a sign slip through -- a positive y, or x and y swapped, misses by centimetres.
MARKER_ON_TEE_TOL_CM = 0.6
# How far ID 1 may sit from the nominal "horizontally centred, MARKER_FROM_TOP_CM down"
# placement before the self-test calls it a measuring slip rather than a real measurement.
# Generous, because once ID 1 is measured with a ruler like the other three it is EXPECTED to
# differ from the nominal by a few millimetres -- the nominal is only where the marker was
# meant to go.  What this still catches is a row that is off by a whole marker width.
MARKER_NOMINAL_TOL_CM = 2.0


def _shape(shape=None):
    """The :class:`shapes.ShapeSpec` in play -- the module's, or a caller's override."""
    return SHAPE if shape is None else shapes.resolve(shape)


def _center():
    return np.array(FIELD_CENTER_CM, dtype=float)


def _field_cm():
    """``[FIELD_X_CM, FIELD_Y_CM]`` -- the divisor, i.e. the full field, not the half."""
    f = np.array([float(FIELD_X_CM), float(FIELD_Y_CM)], dtype=float)
    if not np.all(f > 0.0):
        raise ValueError(f"field size must be positive centimetres, got {tuple(f)}")
    return f


def _rot(deg):
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s], [s, c]], dtype=float)


def _wrap(theta):
    """Fold a heading into ``(-pi, pi]``.

    With ``ROTATION_DEG = 180`` every heading shifts by ``pi``, so without this half of
    them would come out beyond ``pi`` -- the same angle, but a number that surprises
    anything normalizing by ``2*pi``.  ``push_t_demo_realworld.real_to_sim_pose`` adds the
    offset without folding, so its answer can differ from this one by exactly ``2*pi``;
    that is the same heading, and this is the friendlier of the two forms.
    """
    return -((-np.asarray(theta, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi)


def _se2(x, y, theta):
    """Broadcast ``(x, y, theta)`` into one ``(..., 3)`` array of poses."""
    return np.stack(np.broadcast_arrays(np.asarray(x, dtype=float),
                                        np.asarray(y, dtype=float),
                                        np.asarray(theta, dtype=float)), axis=-1)


# ======================================================================================
# camera <-> table
# ======================================================================================
# ``pose`` throughout is what ``camera_test_id10``'s ``pnp_frame`` / ``depth_frame``
# return: a dict with ``R`` (3x3, table axes as columns in camera coordinates) and ``t``
# (the table origin in camera metres).  Rotating by ``R.T`` and subtracting ``t`` is the
# whole of camera -> table; the only reason the two directions are separate functions is
# the unit change, metres out there and centimetres in here.
#
# What is NOT here: anything that needs the camera's intrinsics -- undistorting a pixel
# into a ray (``ray_to_table_cm``) or projecting table points back onto the image
# (``project_table``).  Those need the lens model and OpenCV, so they stay in
# ``camera_test_id10`` next to the code that builds ``cam``.
def to_table_cm(pose, p_cam):
    """Camera-frame point [m] -> table-frame ``(x, y, z)`` [cm]."""
    b = pose["R"].T @ (np.asarray(p_cam, dtype=np.float64).reshape(3) - pose["t"])
    return (float(b[0] * 100.0), float(b[1] * 100.0), float(b[2] * 100.0))


def to_cam_m(pose, xy_cm, z_cm=0.0):
    """Table-frame ``(x, y)`` [cm] -> the camera-frame point [m] it sits at."""
    v = np.array([xy_cm[0] / 100.0, xy_cm[1] / 100.0, z_cm / 100.0], dtype=np.float64)
    return pose["R"] @ v + pose["t"]


def table_yaw_deg(pose, v_cam):
    """Camera-frame direction -> its heading in the table plane, deg CCW from table +x."""
    b = pose["R"].T @ np.asarray(v_cam, dtype=np.float64).reshape(3)
    return float(np.degrees(np.arctan2(b[1], b[0])))


def tilt_deg(pose):
    """Angle between the table normal and the camera axis, deg.  0 = looking straight down."""
    return float(np.degrees(np.arccos(np.clip(abs(float(pose["R"][2, 2])), -1.0, 1.0))))


# ======================================================================================
# the shape's body geometry -- what the marker -> shape offset is built out of
# ======================================================================================
# All four of these used to rebuild the T analytically from TEE_SIZE_CM / TEE_BAR_CM /
# TEE_STEM_CM.  They now read the ACTIVE SHAPE'S MESH instead (``shapes.ShapeSpec``), so
# the polygon the marker positions are checked against, the centroid the camera reports
# and the outline the viewers draw are all the same shape the checkpoint was trained on,
# for a V and a rectangle as much as for the T.  The names are unchanged because the
# whole rig calls them.
def tee_outline_cm(shape=None, mode="bbox"):
    """The shape's outline as an ``(N, 2)`` loop in centimetres, CCW.

    ``mode="bbox"`` (the default, and what the old T-only version returned) puts the
    origin at the middle of the bounding box; ``"centroid"`` puts it at the area centroid.

    For the T this is the same eight-sided polygon as before -- crossbar along the top,
    stem hanging to the bottom edge, ``+y`` toward the top of the shape and ``+x`` to its
    right, the orientation ``Tshape3d_zup.obj`` is authored in -- so a sim pose of
    ``theta = 0`` and a table pose of ``theta = 0`` still mean the same thing.
    """
    return _shape(shape).outline_cm(mode)


def polygon_centroid(pts):
    """Area centroid of a simple polygon, by the shoelace formula."""
    return shapes.polygon_centroid(pts)


def tee_center_local_cm(shape=None, mode="centroid"):
    """Where "the centre" is, in bounding-box coordinates.

    ``bbox`` -> ``(0, 0)`` by definition.  ``centroid`` -> the area centroid, which for the
    12 x 12 T sits 2.27 cm *above* the box middle, because the crossbar carries more area
    than the stem.  ``load_geometry`` builds ``tee_poly`` about the centroid and the sim
    world SE(2) pose is about that same point, so ``centroid`` is what aligns with
    ``push_t_demo_sim`` and is the default here.
    """
    return _shape(shape).center_local_cm(mode)


def tee_body_outline_cm(shape=None, mode="centroid"):
    """The outline about whichever centre ``mode`` names -- what ``visualize`` draws."""
    return _shape(shape).outline_cm(mode)


def marker_to_center_offset_cm(shape=None, from_top_cm=MARKER_FROM_TOP_CM,
                               mode="centroid"):
    """Vector from a horizontally centred marker's centre to the shape centre, cm.

    Such a marker is on the shape's own ``+y`` axis, so for a symmetric shape this comes
    out purely along ``-y``: from the marker, down toward the middle of the shape.  It is
    the *derivation* of ID 1's row of ``TEE_MARKER_POS_CM`` -- ``(width / 2, -from_top_cm)``
    in the measuring frame -- kept so that row can be checked rather than trusted;
    ``frame_conversions.py`` run as a script compares the two.  The other markers are
    hand-placed anywhere on the shape, so nothing here predicts them: they are measured
    straight into that table.
    """
    spec = _shape(shape)
    if from_top_cm is None:
        raise ValueError(f"{spec.name} has no nominal marker placement to derive -- set "
                         f"shapes.SHAPES[{spec.name!r}].marker_nominal_from_top_cm, or "
                         f"pass from_top_cm, if one of its markers has a designed spot")
    _w, h = spec.marker_box_cm()
    marker_local = np.array([0.0, h / 2.0 - float(from_top_cm)])
    return spec.marker_center_cm(mode) - marker_local


# ======================================================================================
# where each marker sits on the shape -- measured in the TOP-LEFT frame
# ======================================================================================
# FOUR ArUco markers, four different ids, glued to four different parts of the shape.  Any
# one of them fixes the whole shape pose by itself, so a marker going under the arm's own
# body -- the failure that used to drop the track outright -- costs nothing as long as one
# of the others is in view.  ``locate_functions.locate_shape`` walks the ids in
# INCREASING order and reports off the first one it can actually measure.
#
# What each marker needs is where it sits on the shape.  That is measured in the frame you
# can actually put a ruler in -- **hold the shape the way it reads in text** (for the T:
# crossbar along the top, stem hanging down):
#
#         (0, 0)                                    ORIGIN: the TOP-LEFT corner of the
#           +------------------------------+        shape's bounding box.
#           |                              |        +x  to the RIGHT
#           |          crossbar            |        -y  DOWNWARD, so every point on the
#           +---------+          +---------+            shape has y <= 0.
#                     |          |
#                     |  stem    |                  Read x off the top edge, y down the
#                     |          |                  left edge, both to the CENTRE of the
#                     |          |                  marker's square.  Nothing else.
#                     +----------+
#                (6, -12)
#
# So a marker whose centre is 4 cm in from the left edge and 3 cm down from the top is
# ``(4.0, -3.0)``.  The shape's centre of area, its arms, its notches -- none of that comes
# into it; ``tee_marker_offset`` converts and does the centroid arithmetic itself, and it
# checks the point actually lands on the shape's outline so a dropped minus sign is caught
# rather than believed.
#
# **The table is per shape and lives in ``shapes.SHAPES[...].markers``**, not here: a
# position measured on the T means nothing on a V.  This name is the ACTIVE shape's rows,
# so everything that already read ``TEE_MARKER_POS_CM`` keeps working and automatically
# follows ``--shape-name``.  Fill in a new shape's rows there.
#
# An id whose row is ``None`` is **skipped, not guessed at**: ``locate_shape`` falls
# through to the next one rather than reporting a pose off a position nobody measured.
#
# The mount angle is NOT in the table because it is the same for every marker on a shape:
# they are all glued on the same way round, so one ``marker_yaw_deg`` covers them all.  (A
# marker deliberately stuck on turned differently can carry its own as an optional third
# element, ``(x_cm, y_cm, yaw_deg)`` -- but if they all match, leave it off.)
TEE_MARKER_POS_CM = dict(SHAPE.markers)
TEE_MARKER_IDS = tuple(sorted(TEE_MARKER_POS_CM))
DEFAULT_TEE_MARKER_ID = TEE_MARKER_IDS[0] if TEE_MARKER_IDS else 1


def tee_topleft_to_body_cm(x_cm, y_cm, shape=None):
    """A point measured from the shape's TOP-LEFT corner -> its bbox-centred body frame.

    The measuring frame has its origin at that corner with ``-y`` running down the shape;
    the body frame ``tee_outline_cm`` is drawn in is centred on the bounding box with
    ``+y`` running up it.  Same axis directions, so this is a pure translation by half the
    box -- PER AXIS, since a rectangle's box is not square: ``(0, 0)`` -> ``(-6, +6)`` and
    ``(6, -12)`` -> ``(0, -6)`` for the 12 x 12 T.
    """
    return _shape(shape).topleft_to_body_cm(x_cm, y_cm)


def tee_body_to_topleft_cm(x_cm, y_cm, shape=None):
    """Inverse of ``tee_topleft_to_body_cm`` -- a body-frame point back onto the ruler."""
    return _shape(shape).body_to_topleft_cm(x_cm, y_cm)


def on_tee(x_cm, y_cm, shape=None, tol_cm=0.0):
    """Is this BODY-frame point on the shape's material?

    A crossing-number test against the mesh outline, so it is right for a V or an L as
    well as for the T's two rectangles.  ``tol_cm`` grows the shape, for a marker centre
    measured a hair outside a real edge.
    """
    return _shape(shape).on_shape(x_cm, y_cm, tol_cm)


def _positions(positions=None):
    """The position table in play -- the module's, or a caller's override of it."""
    return TEE_MARKER_POS_CM if positions is None else positions


def tee_marker_configured(marker_id, positions=None):
    """True when ``marker_id`` has a position filled in, i.e. can carry a fix."""
    return _positions(positions).get(int(marker_id)) is not None


def tee_marker_ids(positions=None):
    """Every usable marker id, ASCENDING -- the order ``locate_shape`` prefers them in.

    Ascending because the ids have to be tried in *some* fixed order, and lowest-first is
    the one that is obvious from outside: whenever id 1 is visible it is the one used, so
    the reported pose does not quietly change which marker it came from frame to frame
    while all four are in view.  Ids still ``None`` in the table are left out -- their
    position is unknown, so a fix taken off them would be wrong rather than merely noisy.
    """
    return tuple(sorted(i for i, v in _positions(positions).items() if v is not None))


def tee_marker_pos(marker_id, positions=None):
    """``(x_cm, y_cm, yaw_deg)`` for one marker, as measured from the TOP-LEFT corner.

    Raises ``KeyError`` for an id that is not in the table at all and ``ValueError`` for one
    that is there but still ``None``, because they are different mistakes: a mistyped id
    versus a marker whose position has not been measured yet.
    """
    key = int(marker_id)
    table = _positions(positions)
    if key not in table:
        raise KeyError(f"marker id {key} is not in the shape's marker table (have "
                       f"{sorted(table)}) -- add it to TEE_MARKER_POS_CM")
    entry = table[key]
    if entry is None:
        raise ValueError(
            f"marker id {key} has no position yet: set shapes.SHAPES[{SHAPE.name!r}]"
            f".markers[{key}] to that marker's centre measured from the shape's TOP-LEFT "
            f"corner, (x_cm, y_cm) with +x right and -y DOWN -- so y is negative")
    yaw = float(entry[2]) if len(entry) > 2 else MARKER_YAW_ON_TEE_DEG
    return (float(entry[0]), float(entry[1]), yaw)


def tee_marker_offset(marker_id, positions=None, mode="centroid", shape=None):
    """``(dx_cm, dy_cm, yaw_deg)`` -- marker centre -> shape centre, in its body frame.

    This is the bridge between the two frames: the table is measured off the top-left
    corner, everything downstream wants a vector to the shape's own origin, and doing the
    conversion in one place is what keeps the measuring simple.  ``mode`` picks which centre
    that origin is (``centroid``, what the sim poses about, or the bbox middle), so the
    measured numbers never have to change when that choice does.

    The point is checked against the shape's own outline first: an ``x`` past an edge or a
    ``y`` entered positive puts the marker off the shape, which is a measuring slip rather
    than a strange but valid placement, so it raises here instead of quietly biasing every
    frame.
    """
    spec = _shape(shape)
    x_cm, y_cm, yaw = tee_marker_pos(marker_id, positions)
    local = spec.topleft_to_body_cm(x_cm, y_cm)
    if not spec.on_shape(local[0], local[1], tol_cm=MARKER_ON_TEE_TOL_CM):
        w, h = spec.size_cm()
        raise ValueError(
            f"marker id {int(marker_id)} at ({x_cm:+.2f}, {y_cm:+.2f}) cm is not on the "
            f"{spec.name}. Measure from the TOP-LEFT corner with +x right and -y DOWN, to "
            f"the marker's centre: x in [0, {w:.1f}] and y in [-{h:.1f}, 0], and on the "
            f"shape's material rather than in a notch beside it. A positive y is the usual "
            f"slip.")
    dx, dy = spec.center_local_cm(mode) - local
    return (float(dx), float(dy), yaw)


def _resolve_offset(marker_id, offset, positions, mode, shape=None):
    """One place where "which offset am I using" is decided.

    An explicit ``offset`` wins -- that is the path ``locate_shape`` takes, having looked
    the marker up once already -- otherwise it is ``marker_id``'s row in the table.
    """
    if offset is None:
        return tee_marker_offset(marker_id, positions, mode, shape)
    return (float(offset[0]), float(offset[1]),
            float(offset[2]) if len(offset) > 2 else MARKER_YAW_ON_TEE_DEG)


# ======================================================================================
# marker <-> shape
# ======================================================================================
def tee_theta_from_marker(marker_theta_rad, marker_id=DEFAULT_TEE_MARKER_ID, offset=None,
                          positions=None, shape=None):
    """The T's own heading, from the heading the camera measures off one of its markers.

    The markers are mounted turned, so the two headings differ by a fixed amount: the
    camera reads a marker's ``+x`` edge, the shape's frame is that marker's ``yaw_deg``
    behind it.  All four share one ``yaw_deg``, so which marker was read does not change
    the answer -- ``marker_id`` is here so the table stays the single source of the number,
    not because the arithmetic depends on it.
    """
    yaw = _resolve_offset(marker_id, offset, positions, "centroid", shape)[2]
    return float(_wrap(float(marker_theta_rad) - math.radians(yaw)))


def marker_theta_from_tee(tee_theta_rad, marker_id=DEFAULT_TEE_MARKER_ID, offset=None,
                          positions=None, shape=None):
    """Inverse of ``tee_theta_from_marker``: the T's heading -> that marker's."""
    yaw = _resolve_offset(marker_id, offset, positions, "centroid", shape)[2]
    return float(_wrap(float(tee_theta_rad) + math.radians(yaw)))


def aruco_to_center(marker_x_cm, marker_y_cm, marker_theta_rad,
                    marker_id=DEFAULT_TEE_MARKER_ID, offset=None, positions=None,
                    mode="centroid", shape=None):
    """One marker's pose in the table frame -> the T **shape's** centre, cm.

    Two turns, in this order:

    1. **the mount**, ``-yaw_deg`` -- the marker is glued on turned, so the heading the
       camera reads off it is not the shape's.  Take that off first, and what is left is the
       T's own heading in the table frame.
    2. **the pose**, that heading -- the offset is fixed in the T's *own* frame, so it has
       to be turned into table axes before it can be added to a table-frame position, which
       is why this needs a heading at all and cannot be a constant shift.

    ``marker_theta_rad`` is therefore the **marker's** heading, straight from
    ``locate_shape``'s plane method; the T's own is ``tee_theta_from_marker`` of it, and a
    shape heading of 0 means the T points down in table terms -- top edge toward table
    ``+y``, stem toward ``-y``.

    Because every marker's offset lands on the same point, all four ids give the same
    ``(x, y)`` for a given shape pose; that is the whole reason a blocked marker can simply
    be swapped for another one.  Pass ``offset`` when the row has already been looked up.

    Returns ``(x_cm, y_cm)``.  ``center_to_aruco`` is the exact inverse.
    """
    dx, dy, yaw = _resolve_offset(marker_id, offset, positions, mode, shape)
    th = tee_theta_from_marker(marker_theta_rad, offset=(dx, dy, yaw), shape=shape)
    c, s = math.cos(th), math.sin(th)
    return (float(marker_x_cm) + c * dx - s * dy,
            float(marker_y_cm) + s * dx + c * dy)


def center_to_aruco(x_cm, y_cm, tee_theta_rad, marker_id=DEFAULT_TEE_MARKER_ID,
                    offset=None, positions=None, mode="centroid", shape=None):
    """The T shape's centre -> where one of its markers sits, cm.  Inverse of
    ``aruco_to_center``.

    Note which heading each one takes: ``aruco_to_center`` is fed what the camera measured,
    so it takes the **marker's** heading, while this is fed a shape pose -- a planned one,
    a goal, something out of the sim -- so it takes the **T's**.  Each takes the heading of
    the thing it is handed, and the mount turn is applied in whichever direction closes the
    loop, so ``aruco_to_center(*center_to_aruco(x, y, th), ...)`` returns ``(x, y)``.

    Unlike ``aruco_to_center`` the answer here DOES depend on ``marker_id``: one shape pose
    puts four different markers in four different places, which is what makes this useful
    for predicting where each one should appear (and hence which are worth looking for).
    """
    dx, dy, _ = _resolve_offset(marker_id, offset, positions, mode, shape)
    c, s = math.cos(float(tee_theta_rad)), math.sin(float(tee_theta_rad))
    return (float(x_cm) - (c * dx - s * dy),
            float(y_cm) - (s * dx + c * dy))


# ======================================================================================
# the two conversions
# ======================================================================================
def table_to_sim(x_cm, y_cm, theta_rad=0.0):
    """Table-frame SE(2) [cm, rad] -> the planner's normalized SE(2).

    Subtract the field centre, rotate into the sim axes, divide by the full field size.
    The result is in the ``[-0.5, 0.5]`` square the network was trained on -- outside that
    range means the pose is off the field, which is worth checking rather than clipping.

    ``theta`` is only offset by the frame rotation, never divided: it stays in radians CCW
    from the sim ``+x``, folded back into ``(-pi, pi]``.  The ``/ 2*pi`` the network wants
    happens further downstream, in ``push_t_demo_sim.world_to_planner``, not here.

    Scalars or matching array-likes; returns ``(3,)`` for scalars, ``(..., 3)`` for arrays,
    the last axis being ``(x, y, theta)``.  For points alone, pass ``theta_rad=0.0`` and
    take ``[..., :2]``.
    """
    pose = _se2(x_cm, y_cm, theta_rad)
    norm = ((pose[..., :2] - _center()) @ _rot(ROTATION_DEG).T) / _field_cm()
    theta = _wrap(pose[..., 2] + math.radians(ROTATION_DEG))
    return np.concatenate([norm, theta[..., None]], axis=-1)


def sim_to_table(x, y, theta_rad=0.0):
    """The planner's normalized SE(2) -> table-frame SE(2) [cm, rad].

    Exact inverse of ``table_to_sim`` for headings already in ``(-pi, pi]`` -- one outside
    it comes back folded into that range, the same angle by a multiple of ``2*pi``.
    Multiply by the full field size, rotate back into the table axes, add the field centre.  ``(0, 0)`` comes back as the field centre --
    ``FIELD_CENTER_CM``, not the ID-10 marker -- and ``(+-0.5, +-0.5)`` as the field
    corners.  Same calling convention.
    """
    pose = _se2(x, y, theta_rad)
    real = (pose[..., :2] * _field_cm()) @ _rot(ROTATION_DEG) + _center()
    theta = _wrap(pose[..., 2] - math.radians(ROTATION_DEG))
    return np.concatenate([real, theta[..., None]], axis=-1)


# ======================================================================================
# sim <-> the UR base frame
# ======================================================================================
# Anchored on **p0**, the point ``arm_calibrate.py`` records: the TCP pose the UR reports
# with the end effector placed physically on the FIELD CENTRE -- sim (0, 0), not the ID-10
# marker.  Its x, y are that centre in base coordinates, its z is the height every target
# is held at, and its rotation vector is the tool orientation reproduced at every target
# (a cylindrical pusher is rotationally symmetric, so the wrist never turns).
def load_origin_pose(path=None):
    """The p0 TCP pose out of ``arm_calibrate.py``'s JSON -- the robot at the field centre.

    Returns ``(x, y, z, rx, ry, rz)`` in the UR base frame, metres / rotation vector.  A
    relative path is resolved against the repo directory rather than the cwd, so this
    works from anywhere inside the container.  Missing or point-less file raises: there is
    no sensible fallback anchor for a caller whose whole job is to go where the
    calibration says.
    """
    path = STARTPOS_JSON if path is None else path
    p = path if os.path.isabs(path) else os.path.join(HERE, path)
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"{path} not found -- run arm_calibrate.py first and press SPACE with the end "
            f"effector placed physically on the centre of the field.")
    with open(p) as fh:
        d = json.load(fh)
    pt = d.get("point") or next(iter(d.get("points") or []), None)
    if not pt or "tcp_pose" not in pt:
        raise ValueError(f"{path}: no calibration point with a tcp_pose")
    pose = np.array([float(v) for v in pt["tcp_pose"]], dtype=float)
    if pose.shape != (6,):
        raise ValueError(f"{path}: tcp_pose is not 6 numbers: {pose.tolist()}")
    return pose


def sim_to_robot(xy_sim, origin_pose, field_cm=None):
    """Normalized sim ``(x, y)`` -> a full UR base-frame pose.

    Scale by the whole field, flip both axes, and add the origin::

        base_xy = origin_pose[:2] + SIM_TO_BASE * xy_sim * field_cm * 0.01

    ``origin_pose`` is p0, so sim ``(0, 0)`` comes back as p0 itself -- the flip is a
    half-turn about p0, and a rotation fixes its own centre.  Away from it the two frames
    run opposite: sim ``+x`` is base ``-x`` and sim ``+y`` is base ``-y``.  Since
    :func:`table_to_sim` already turns table into sim by ``ROTATION_DEG`` (180), the two
    cancel: table ``+x`` is base ``+x``, and a point 35 cm along the table from p0 is
    35 cm along base ``+x`` from p0.

    ``field_cm`` defaults to the same per-axis divisor :func:`table_to_sim` uses, so a
    rectangular field stays consistent between the two; the normalized square runs
    ``-0.5 .. +0.5``, so half the field lands on ``0.5``.

    ``z`` and the rotation vector are copied from p0 verbatim -- this is what keeps the
    tool at the calibrated height and pointing where it was calibrated pointing.  The flip
    is in the plane only and never reaches them.

    Scalars in, a ``(6,)`` array out: ``(x, y, z, rx, ry, rz)``, metres / rotvec radians.
    """
    o = np.asarray(origin_pose, dtype=float)
    xy = np.asarray(xy_sim, dtype=float)[:2]
    f = _field_cm() if field_cm is None else np.broadcast_to(np.asarray(field_cm,
                                                                       dtype=float), (2,))
    return np.concatenate([o[:2] + SIM_TO_BASE * xy * f * 0.01, o[2:6]])


def robot_to_sim(pose, origin_pose, field_cm=None):
    """A UR pose (or bare base ``(x, y)`` metres) -> normalized sim ``(x, y)``.

    Exact inverse of :func:`sim_to_robot` in the plane -- the same flip, which is its own
    inverse.  The height is simply dropped, since every sim point maps to the one
    calibrated z.
    """
    o = np.asarray(origin_pose, dtype=float)
    b = np.asarray(pose, dtype=float)[:2]
    f = _field_cm() if field_cm is None else np.broadcast_to(np.asarray(field_cm,
                                                                       dtype=float), (2,))
    return SIM_TO_BASE * (b - o[:2]) / (f * 0.01)


def table_to_robot(xy_cm, origin_pose, field_cm=None):
    """Table centimetres -> a UR pose, through the sim frame.

    The click path in one call: :func:`table_to_sim` (centre on the field, turn 180 deg,
    divide) then :func:`sim_to_robot` (scale, flip, shift), so a change to either
    calibration lands in exactly one of them.
    """
    xy_sim = table_to_sim(float(xy_cm[0]), float(xy_cm[1]))[:2]
    return sim_to_robot(xy_sim, origin_pose, field_cm)


def robot_to_table(pose, origin_pose, field_cm=None):
    """A UR pose -> table centimetres.  Inverse of :func:`table_to_robot`."""
    xy_sim = robot_to_sim(pose, origin_pose, field_cm)
    return np.asarray(sim_to_table(xy_sim[0], xy_sim[1])[:2], dtype=float)


def clamp_sim(xy_sim, limit=NORM_HALF):
    """Hold a normalized point inside the field; returns ``(xy, was_clamped)``.

    A click can land anywhere on the table plane, including well outside the field, and an
    unclamped one would be a full-speed lunge to somewhere nobody meant to point.
    """
    xy = np.asarray(xy_sim, dtype=float)[:2]
    held = np.clip(xy, -abs(limit), abs(limit))
    return held, bool(np.any(np.abs(held - xy) > 1e-12))


# ======================================================================================
# the bridge to push_t_demo_sim
# ======================================================================================
# Everything above stops at the normalized frame.  ``push_t_demo_realworld`` owns the rest
# of the chain -- centre on the env, rotate, normalize, then scale into pymunk mesh units
# -- so these call it rather than reimplementing it, and a change to
# ``real_world_params.json`` takes effect with no edit here.  The import is lazy and
# failure is a ``None``, which is what keeps this module usable with no sim stack.
def sim_bridge():
    """``push_t_demo_realworld``, or None.  Imported lazily -- it is optional here."""
    try:
        import push_t_demo_realworld as rw
        return rw
    except Exception:                                     # noqa: BLE001
        return None


def to_sim_pose(x_cm, y_cm, theta_rad):
    """Table-frame SE(2) -> the sim **world** SE(2) ``push_t_demo_sim`` uses, or None.

    Not :func:`table_to_sim`, which stops at the normalized square: this carries on into
    pymunk mesh units, which is what the physics and the planner's own world are in.
    Straight through ``push_t_demo_realworld.real_to_sim_pose``; None when that module (or
    its ``meta.json``) is not available, so perception still runs without the planner.
    """
    rw = sim_bridge()
    if rw is None:
        return None
    try:
        return [float(v) for v in rw.real_to_sim_pose(x_cm, y_cm, theta_rad)]
    except Exception:                                     # noqa: BLE001
        return None


# ======================================================================================
def field_corners_cm():
    """The field's four corners in table cm, from ``(-0.5, -0.5)`` round to ``(-0.5, 0.5)``.

    Handy for drawing the field on the table and for checking it lands where you measured.
    """
    unit = np.array([(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)], dtype=float)
    return sim_to_table(unit[:, 0], unit[:, 1])[:, :2]


def main():
    print(SHAPE.describe())
    for problem in SHAPE.check():
        print(f"  !! {problem}")
    print()
    cx, cy = FIELD_CENTER_CM
    print(f"table frame: origin at the ID-{ORIGIN_ID} marker, +x along the length, "
          f"+y along the width, cm")
    print(f"field       {FIELD_X_CM:.1f} x {FIELD_Y_CM:.1f} cm, centred at "
          f"({cx:+.2f}, {cy:+.2f}) cm, rotation {ROTATION_DEG:+.2f} deg")
    print(f"normalized  -{NORM_HALF} .. +{NORM_HALF}, so half the field "
          f"({FIELD_X_CM / 2:.1f} cm) is {NORM_HALF}")
    print("\n  table cm            ->  normalized")
    pts = [(cx, cy, "field centre"), *((x, y, "corner") for x, y in field_corners_cm())]
    for x, y, what in pts:
        sx, sy, _ = table_to_sim(x, y)
        print(f"  ({x:+7.2f}, {y:+7.2f})  ->  ({sx:+6.3f}, {sy:+6.3f})   {what}")
    ox, oy, _ = table_to_sim(0.0, 0.0)
    print(f"  ({0.0:+7.2f}, {0.0:+7.2f})  ->  ({ox:+6.3f}, {oy:+6.3f})   "
          f"the ID-{ORIGIN_ID} marker itself"
          + ("" if max(abs(ox), abs(oy)) <= NORM_HALF else "  -- outside the field"))

    # ---- marker -> shape: one row per marker, and the round trip through each ----
    ready = tee_marker_ids()
    yaws = {tee_marker_pos(i)[2] for i in ready}
    mount = (f"mount {yaws.pop():+.1f} deg shared by all" if len(yaws) == 1
             else "mount angle PER MARKER (see the rows)")
    print(f"\nmarkers on the shape: {list(TEE_MARKER_IDS)}, usable {list(ready)}, {mount}")
    print(f"  {'':4s}{'measured from top-left':<24s}{'-> body frame':<18s}"
          f"-> offset to the centroid")
    for mid in sorted(TEE_MARKER_POS_CM):
        if not tee_marker_configured(mid):
            print(f"  id {mid}   NOT MEASURED -- put (x_cm, y_cm) from the shape's "
                  f"top-left corner in shapes.SHAPES[{SHAPE.name!r}].markers[{mid}]")
            continue
        px, py, yaw = tee_marker_pos(mid)
        bx, by = tee_topleft_to_body_cm(px, py)
        try:
            dx, dy, _ = tee_marker_offset(mid)
        except ValueError as exc:
            print(f"  id {mid}   ({px:+6.2f}, {py:+6.2f}) cm   REJECTED: {exc}")
            continue
        print(f"  id {mid}   ({px:+6.2f}, {py:+6.2f}) cm         "
              f"({bx:+6.2f}, {by:+6.2f})     ({dx:+6.2f}, {dy:+6.2f}) cm, "
              f"mount {yaw:+.1f} deg")
    if not ready:
        print("  no marker has a position measured yet -- nothing to check against")
        return
    if MARKER_FROM_TOP_CM is None:
        print(f"  (no nominal placement for {SHAPE.name}'s markers -- nothing to hold "
              f"id {DEFAULT_TEE_MARKER_ID}'s row against)")
    else:
        _nominal_check()
    _round_trip(ready)
    _sim_to_robot_report(pts)


def _nominal_check():
    # The lowest id has a nominal placement the others do not -- horizontally centred,
    # MARKER_FROM_TOP_CM down -- so its row can be held against the geometry.  A measured row
    # is EXPECTED to differ from that by millimetres; only a gross gap means a slipped ruler.
    gx, gy = marker_to_center_offset_cm()
    tx, ty, _ = tee_marker_offset(DEFAULT_TEE_MARKER_ID)
    gap = math.hypot(tx - float(gx), ty - float(gy))
    nom = tee_body_to_topleft_cm(*(tee_center_local_cm() - np.array([float(gx), float(gy)])))
    print(f"  id {DEFAULT_TEE_MARKER_ID} vs its nominal placement "
          f"({nom[0]:+.2f}, {nom[1]:+.2f}) cm (centred, {MARKER_FROM_TOP_CM:.2f} down): "
          f"{gap:.2f} cm apart"
          + ("   -- as measured" if gap <= MARKER_NOMINAL_TOL_CM else
             f"   -- MORE THAN {MARKER_NOMINAL_TOL_CM:.1f} cm, check that row for a slip"))


def _round_trip(ready):
    # Every usable marker has to land the SAME shape centre from the same shape pose --
    # that identity is exactly what lets a blocked marker be swapped for another one.
    print("\n  shape pose            ->  centre seen through each marker  [round trip]")
    for sx0, sy0, sth_deg in [(20.0, 20.0, 0.0), (20.0, 20.0, 90.0), (40.0, 30.0, -45.0)]:
        sth = math.radians(sth_deg)
        cells = []
        for mid in ready:
            mx, my = center_to_aruco(sx0, sy0, sth, mid)
            cx2, cy2 = aruco_to_center(mx, my, marker_theta_from_tee(sth, mid), mid)
            cells.append(f"id {mid} ({cx2:+6.2f}, {cy2:+6.2f})")
        print(f"  ({sx0:+6.2f}, {sy0:+6.2f}) at {sth_deg:+6.1f} deg  ->  "
              + "   ".join(cells))


def _sim_to_robot_report(pts):
    # ---- sim -> robot: only if p0 has been recorded ----
    try:
        p0 = load_origin_pose()
    except (FileNotFoundError, ValueError) as exc:
        print(f"\nsim -> robot: no p0 yet ({exc})")
        return
    print(f"\nsim -> robot: base_xy = p0_xy + ({SIM_TO_BASE[0]:+.0f}, "
          f"{SIM_TO_BASE[1]:+.0f}) * sim_xy * {FIELD_CM:.1f} cm;  p0 = the field centre")
    print(f"  p0  base ({p0[0]:+.4f}, {p0[1]:+.4f}, {p0[2]:+.4f}) m")
    print("\n    table cm         ->     sim          ->  base x, y [m]")
    for x, y, what in pts:
        sim = table_to_sim(x, y)[:2]
        b = sim_to_robot(sim, p0)
        print(f"  ({x:+7.2f}, {y:+7.2f})  ->  ({sim[0]:+6.3f}, {sim[1]:+6.3f})  ->  "
              f"({b[0]:+7.4f}, {b[1]:+7.4f})   {what}")


if __name__ == "__main__":
    main()
