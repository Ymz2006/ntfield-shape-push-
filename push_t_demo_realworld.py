"""Real-world push-T -- coordinate bridge + the pushing stack from the sim demo.

``push_t_demo_sim.py`` plans and runs everything in the pymunk *world* frame: raw
mesh units, the physics plane is XY, and the environment sits wherever the mesh
puts it (roughly ``meta['env_center']`` +- ``meta['env_scale']/2``).  On hardware
we instead measure things in the **table frame** ``camera_test_id10.py`` defines --
origin at the **ID-10** corner marker, ``+x`` along the length, ``+y`` along the width,
centimetres.  Every hop between frames is imported from ``frame_conversions``, never
re-derived here, so this file and the id-10 test programs agree by construction.

Two halves:

1. The frame transforms (``real_to_sim`` / ``sim_to_real`` and their pose twins).
   For now the only calibrated quantity is a translation; scale and rotation are
   identity hooks -- fill them in if calibration shows they are needed.

2. The pushing stack, imported wholesale from ``push_t_demo_sim``: the boundary
   IK (``PushIK``), the measured primitive library (``PushPrimitives``), the
   transit planner + action selector (``PushController``), and the reference
   re-planner (``Replanner``).  ``build_stack`` recreates everything
   ``push_t_demo_sim.main()`` builds EXCEPT the pymunk world, and
   ``ArmPushSession.step`` runs one action of the loop for a physical arm: given
   the observed T pose and the observed end-effector position it returns the
   ordered end-effector waypoints -- retract off the current contact, ride an arc
   (or slide a face) around the T to the next contact, then push -- in both sim
   units and table centimetres.

Only the interpolated-transit mode is supported here (``--linear-interp``): a real
arm cannot teleport, so every transit is flown.  ``teleport_interp`` is forced on
and there is no plain-teleport path.

The reference is planned between the two ends of the real setup, NOT the checkpoint's
test set: the goal is the hand-set constant ``GOAL_POSE_NORM`` below, and the start is
always the shape's measured pose, handed to ``build_stack(start_pose_real=...)`` by
whoever is watching the T (the camera, in ``push_t_realworld_run.py``).  The goal is set
in the planner's NORMALIZED frame; the start arrives in table centimetres, as everything
the camera reports does.

Frames
------
* **real**  -- table frame, cm, from the camera (see ``camera_test_id10.py``); what
  ``locate_functions.locate_shape`` reports the T's pose in.
* **sim**   -- pymunk world frame, mesh units (see ``pymunk_viser_push.py``).
* **base**  -- UR base frame, metres + rotation vector, anchored on p0 = the FIELD CENTRE.

``frame_conversions`` owns table <-> normalized (``table_to_sim`` / ``sim_to_table``) and
normalized <-> base (``sim_to_robot`` / ``robot_to_sim``); the only hop this module owns is
normalized <-> pymunk mesh units, ``* ENV_SCALE + ENV_CENTER``.

The mapping goes through the planner's normalized frame so it matches
``push_t_demo_sim`` exactly (``world_to_planner`` / ``planner_to_world``)::

    norm = R(CALIB_ROTATION_DEG) @ (real - CALIB_OFFSET) / (ENV_X, ENV_Y)
    sim  = norm * ENV_SCALE + ENV_CENTER
    norm = (sim - ENV_CENTER) / ENV_SCALE
    real = R(CALIB_ROTATION_DEG).T @ (norm * (ENV_X, ENV_Y)) + CALIB_OFFSET

``CALIB_OFFSET`` is the table-frame centimetres of the environment *centre* (the
sim's ``env_center``); ``ENV_X`` / ``ENV_Y`` are the real env's full width / height
in centimetres -- the physical size of the box the planner normalizes to
``[-0.5, 0.5]^2``.  ``ENV_SCALE`` / ``ENV_CENTER`` come from the same ``meta.json``
``load_geometry`` reads.  Heading is only rotated + offset, never divided by the env
size -- the (x, y) -> turns normalization stays inside ``world_to_planner``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, field

import numpy as np

import frame_conversions as FC
import pymunk_viser_push as base
from frame_conversions import (
    clamp_sim,
    load_origin_pose,
    robot_to_sim,
    sim_to_robot,
    sim_to_table,
    table_to_sim,
)
from real_world_params import PARAMS
from push_t_demo_sim import (
    PushController,
    PushIK,
    PushPrimitives,
    Replanner,
    load_planner,
    plan_from,
    planner_to_world,
    push_length_ladder,
    resample_path,
    to_local,
    to_world,
    world_to_planner,
    wrap,
)

# ======================================================================================
# calibration
# ======================================================================================
# The dataset whose meta.json defines the sim env; must match make_args()' dataPath.
_ENV_DATA_PATH = './testing_data/3dshape/Tshape3d_env4'


def _load_env_meta(data_path=_ENV_DATA_PATH):
    """``(env_scale, env_center_xy)`` from ``<data_path>/meta.json``.

    The planner normalizes sim world coordinates as ``(p - env_center) / env_scale``
    (see ``push_t_demo_sim.world_to_planner``); these are the two constants that frame.
    Falls back to ``(1.0, (0, 0))`` -- an identity normalization -- when the file is
    absent, so the module still imports without the dataset.
    """
    try:
        import json

        with open(os.path.join(data_path, 'meta.json')) as f:
            m = json.load(f)
        return float(m['env_scale']), np.asarray(m['env_center'], dtype=float)[:2]
    except (OSError, KeyError, ValueError, TypeError):
        return 1.0, np.zeros(2, dtype=float)


# Sim env normalization, read once from meta.json (same source as ``load_geometry``).
ENV_SCALE, ENV_CENTER = _load_env_meta()

# The table <-> normalized calibration is NOT owned here: ``frame_conversions`` owns every
# frame hop in the rig, and ``camera_test_id10`` / ``locate_functions`` / ``arm_test_id10``
# all measure against that one copy.  These names are kept as ALIASES so the module reads
# the same and nothing importing them breaks, but the numbers -- and the arithmetic that
# uses them below -- now come from ``frame_conversions``, which is what makes this file and
# the id-10 test programs agree by construction rather than by inspection.
#
# ENV_X / ENV_Y  the real env's full width / height in cm: the box the planner normalizes
#               to ``[-0.5, 0.5]^2`` (FC calls it the FIELD).
# CALIB_OFFSET  table-frame cm of the env CENTRE (== the sim's ``env_center``).
# CALIB_ROTATION_DEG  table -> sim rotation, CCW deg, about that centre.  180 measured.
ENV_X = FC.FIELD_X_CM
ENV_Y = FC.FIELD_Y_CM
CALIB_OFFSET_X, CALIB_OFFSET_Y = FC.FIELD_CENTER_CM
CALIB_ROTATION_DEG = FC.ROTATION_DEG


# ======================================================================================
# the goal
# ======================================================================================
# WHERE THE T SHOULD END UP, in the planner's NORMALIZED (sim) frame -- the same numbers
# the checkpoint's test set carries, so a pose can be pasted straight out of
# ``sampled_points.npy``.  Either ``(x, y, rz)`` or the full ``(x, y, z, rx, ry, rz)``:
#
#   x, y   position in the env's ``[-0.5, 0.5]^2`` box: (0, 0) is the env CENTRE, +0.5 its
#          +x / +y edge.  (In table centimetres, if you want to check it:
#          ``Geometry.goal_pose_real()`` / ``PushStack.goal_real()``.)
#   rz     heading in TURNS, not radians or degrees -- 0.25 == 90 deg CCW.
#   z, rx, ry  planar problem, always 0.
#
# This is the ONLY hand-set end of the plan.  The START is never a constant -- it is
# always the shape's MEASURED pose, passed to ``build_stack(start_pose_real=...)`` -- so
# the reference is planned from wherever the T actually is to here.  The checkpoint's
# ``--case`` test-set pair no longer sets either end.
GOAL_POSE_NORM = (-0.35, 0.0 ,0.0)


def goal_pose_norm(goal=None):
    """``GOAL_POSE_NORM`` as the planner's ``(6,)`` pose ``(x, y, z, rx, ry, rz)``.

    ``goal`` overrides the constant and takes the same form -- ``(x, y, rz)`` or all six.
    ``rz`` stays in turns; this is exactly what ``plan_from`` / :class:`Replanner` and
    ``push_t_demo_sim.world_to_planner`` speak.
    """
    return _pose6(GOAL_POSE_NORM if goal is None else goal)


def _pose6(pose_norm):
    """``(x, y, rz)`` or ``(x, y, z, rx, ry, rz)`` -> the ``(6,)`` planner layout."""
    g = np.asarray(pose_norm, dtype=float).ravel()
    if g.size == 3:
        return np.array([g[0], g[1], 0.0, 0.0, 0.0, g[2]])
    if g.size == 6:
        return g.copy()
    raise ValueError(f'normalized pose must be (x, y, rz) or (x, y, z, rx, ry, rz), '
                     f'got {g.size} values')


def _offset():
    """Table-frame cm of the env centre -- ``frame_conversions``' field centre."""
    return np.asarray(FC._center(), dtype=float)


def _env_cm():
    """``[ENV_X, ENV_Y]`` cm -- ``frame_conversions``' field size, the normalizing box."""
    return np.asarray(FC._field_cm(), dtype=float)


def sim_units_per_cm():
    """Mean sim mesh units per real centimetre: ``ENV_SCALE / mean(ENV_X, ENV_Y)``.

    A single-scalar stand-in for the per-axis scale -- exact for the square env -- used
    where only a magnitude is needed (``RobotFrame.speed_mps``, the transit radius).
    """
    return float(ENV_SCALE / np.mean(_env_cm()))


# Back-compat alias: sim mesh units per cm, evaluated once at import.  New code should
# call ``sim_units_per_cm()`` so edits to ENV_X / ENV_Y take effect without a reload.
CALIB_SCALE = sim_units_per_cm()


def _rot(deg):
    """CCW rotation matrix, deg -- ``frame_conversions._rot``, so both files turn alike."""
    return FC._rot(deg)


# ======================================================================================
# transforms
# ======================================================================================
def real_to_sim(x, y):
    """Table-frame coordinates [cm] -> sim world coordinates [mesh units].

    ``frame_conversions.table_to_sim`` does the centre-rotate-divide -- the SAME call
    ``camera_test_id10``, ``locate_functions`` and ``arm_test_id10`` make, so a point can
    never mean one thing to the camera and another to the planner -- and this carries its
    answer the last hop into pymunk mesh units (``* ENV_SCALE + ENV_CENTER``), which is the
    only part of the chain that is this module's own.  ``push_t_demo_sim.world_to_planner``
    then recovers exactly the normalized coordinates the network was trained on.

    ``x`` and ``y`` may be scalars or matching array-likes (any shape).  Returns a
    ``(2,)`` array for scalar input, or ``(..., 2)`` for arrays -- the last axis is
    ``(x, y)``.
    """
    norm = table_to_sim(x, y)[..., :2]
    return norm * ENV_SCALE + ENV_CENTER


def sim_to_real(x, y):
    """Sim world coordinates [mesh units] -> table-frame coordinates [cm].

    Inverse of ``real_to_sim``, through ``frame_conversions.sim_to_table``; same calling
    convention.
    """
    sim = np.stack(np.broadcast_arrays(np.asarray(x, dtype=float),
                                       np.asarray(y, dtype=float)), axis=-1)
    norm = (sim - ENV_CENTER) / ENV_SCALE
    return sim_to_table(norm[..., 0], norm[..., 1])[..., :2]


def real_to_sim_pose(x, y, theta):
    """Table-frame SE(2) pose (x cm, y cm, theta rad) -> sim world SE(2) pose.

    Position and heading both come out of ``frame_conversions.table_to_sim``, so the T's
    pose reaches the planner turned by exactly what the camera's pose was turned by.  The
    heading stays in RADIANS -- the sim world frame carries raw radians
    (``planner_to_world`` outputs ``norm_theta * 2*pi``), so the ``/ 2*pi`` normalization
    the network wants happens later in ``world_to_planner``, NOT here.

    Note ``table_to_sim`` folds the heading into ``(-pi, pi]``; the old local arithmetic
    did not, so a heading here can differ from an older printout by exactly ``2*pi``.
    That is the same angle, and it is the form every other module in the rig reports.
    """
    pose = table_to_sim(x, y, theta)
    xy = np.asarray(pose[..., :2]) * ENV_SCALE + ENV_CENTER
    return np.array([xy[0], xy[1], float(pose[..., 2])])


def sim_to_real_pose(x, y, theta):
    """Sim world SE(2) pose -> table-frame SE(2) pose (x cm, y cm, theta rad)."""
    norm = (np.array([float(x), float(y)]) - ENV_CENTER) / ENV_SCALE
    return np.asarray(sim_to_table(norm[0], norm[1], theta), dtype=float)


def real_pose_to_planner(pose_real, env_scale, env_center, bbox_to_centroid):
    """Table-frame SE(2) pose (x cm, y cm, theta rad) -> the planner's normalized ``(6,)``.

    ``real_to_sim_pose`` then ``push_t_demo_sim.world_to_planner``: the layout the
    Eikonal planner and ``Replanner`` consume, ``(x, y, z, rx, ry, rz)`` with ``z, rx,
    ry`` pinned to 0 and ``rz`` in turns.  This is how ``build_stack`` lifts the shape's
    measured start into the frame ``GOAL_POSE_NORM`` is already written in.
    """
    return world_to_planner(real_to_sim_pose(*pose_real), env_scale, env_center,
                            bbox_to_centroid)


def planner_pose_to_real(pose_norm, env_scale, env_center, bbox_to_centroid):
    """The planner's normalized pose -> a table-frame SE(2) pose (x cm, y cm, theta rad).

    Inverse of :func:`real_pose_to_planner`; takes ``(x, y, rz)`` or the full ``(6,)``.
    Only for reading a normalized pose (``GOAL_POSE_NORM``) back in table units -- the
    plan itself never leaves the normalized frame on this side.
    """
    g = _pose6(pose_norm)
    world = planner_to_world(np.array([[g[0], g[1], g[5]]]), env_scale, env_center,
                             bbox_to_centroid)[0]
    return sim_to_real_pose(*world)


def path_sim_to_real(path):
    """(T,2) or (T,3) sim-frame path -> the same path in table centimetres."""
    path = np.asarray(path, dtype=float)
    norm = (path[:, :2] - ENV_CENTER) / ENV_SCALE
    if path.shape[1] == 2:
        return sim_to_table(norm[:, 0], norm[:, 1])[:, :2]
    return sim_to_table(norm[:, 0], norm[:, 1], path[:, 2])


# ======================================================================================
# parameters
# ======================================================================================
# Extra clearance on the WIDEST transit radius, centimetres.  ``PushController`` builds
# its arc ladder up to ``ik.radius + pusher_radius + standoff`` and that widest circle is
# also ``ctrl.standoff`` -- the one the pusher parks on and the one ``home_pose_real``
# aims at.  On the table that circle is what has to clear the real shape rather than the
# footprint the planner carries, so it is grown by this much here; the sim keeps its own
# 8 units.  Everything else about the ladder is unchanged.
STANDOFF_EXTRA_CM = 8.0

# Margin added to the transit radius the ladder settles on, centimetres.  ``plan_transit``
# walks ``ctrl.radii`` tightest first and stops at the first circle that comes back clear,
# so the disc flies as close to the shape as the cost function allows -- fine against a
# footprint the simulator knows exactly, optimistic against a cardboard T seen through a
# camera.  The chosen rung is re-flown this much wider and the wider one is kept when it
# is clear too, with the verified tight arc as the fallback.
ARC_CLEARANCE_CM = 3.0

# Air gap the transit keeps between the pusher RIM and the shape's footprint, in
# centimetres of ``clearance``.  This is the one keep-out knob that was left at the sim's
# raw unit value: every other length in the bag below is converted with
# ``sim_units_per_cm()``, and ``clearance=1.0`` -- one MESH unit -- is 2 mm on this table.
#
# What it buys is HALF this, because ``PushController`` buffers the footprint by
# ``pusher_radius + 0.5 * clearance``: the disc radius is already in there, so the actual
# air between rim and cardboard is ``0.5 * clearance``.  At the old 1.0 unit that was
# 1 mm, and the arc ladder deliberately settles on the TIGHTEST rung that passes -- so
# every transit was planned to miss the shape by a millimetre, before any camera error.
# 2 cm here = a 1 cm gap.
#
# It also sets the wall keep-out (``pusher_radius + clearance``) and pushes the start of
# each push that much further back along the push line; the push END is unchanged
# (``final = contact + u * L`` -- push_t_demo_sim.py:1113), so only free-air travel grows.
#
# And it sets how far a SAME-FACE transit has to lift off the face before it may slide
# along it: that slide is travel like any other, so it is held to this keep-out, and the
# pusher starts it pressed against the face one radius off.  ``PushController.edge_lift``
# is derived from both (``0.5 * (pusher_radius + clearance)``) for exactly that reason --
# a lift fixed at half a radius, as it was, is inside the keep-out at this clearance and
# sends every same-face move the long way round the arc instead.
TRANSIT_CLEARANCE_CM = 2.0

# ENDGAME PUSH SHRINK.  Once the T is within ``ENDGAME_DIST_CM`` of the goal centroid AND
# within ``ENDGAME_DEG`` of the goal heading, the executed push stroke is cut to
# ``ENDGAME_PUSH_SCALE`` of the length the primitive was selected at -- BOTH tests, the
# same conjunction the run loop stops on, so the shrink only ever applies to the approach
# to the goal POSE.  (It was either test once, which fired on any T whose heading passed
# through the band with the whole translation still ahead of it.)
#
# This exists because the sim's own way of shortening the endgame push is unavailable on
# this rig.  There, ``prims.select`` searches a LADDER of push lengths and the short rungs
# win as the demand shrinks (push_t_demo_sim.py:541).  Here the ladder is one rung --
# ``push_len_min == push_len``, ``push_len_steps == 1`` -- so ``select`` has nothing short
# to pick, and note that the ``push_len`` argument threaded into ``begin_primitive`` is
# DEAD once the primitives are calibrated: ``select`` reads ``self.lengths`` and ignores
# its ``s`` parameter entirely (push_t_demo_sim.py:548).  Changing ``self.push_len`` at
# run time therefore does nothing at all.  So the stroke is cut after the fact, in
# :meth:`ArmPushSession.step`, by pulling the push TARGET back along the push direction.
#
# The consequence is a deliberate, known undershoot: the primitive was chosen against the
# table's prediction for the FULL length, and only a fraction of it gets executed, so the
# T moves roughly that fraction of what was predicted.  That is the point -- near the goal
# an action that overshoots costs more than one that falls short -- and the loop is
# closed, re-observing the T after every action, so the residual is simply picked up by
# the next one.  It does mean more actions to converge.
ENDGAME_DIST_CM = 3.0
ENDGAME_DEG = 5.0
ENDGAME_PUSH_SCALE = 1.0 / 3.0

# Diameter of the real pusher tip, centimetres -- the 3 cm disc on the end effector.
# ``pusher_radius`` below is this in sim mesh units; the sim's own default is 5.0 units
# (= 1 cm here), so the hardware disc is wider than the one the checkpoint was tuned
# against.  ``transit_approach`` / ``transit_reach`` are derived from it in make_args().
PUSHER_DIAMETER_CM = 3.0

# Every knob push_t_demo_sim.build_args() exposes that the pushing stack below actually
# reads, with the same defaults.  Kept as a plain dict so make_args() can splat it.
_DEFAULTS = dict(
    # geometry / planner
    env='datasets/3dshape/2denv4.obj',
    shape='datasets/3dshape/Tshape3d.obj',
    shape_zup='datasets/3dshape/Tshape3d_zup.obj',
    dataPath=_ENV_DATA_PATH,
    modelPath='./Experiments/3dshape',
    ckpt='./latest.pt',
    case=0,                              # unused: the ends are the shape + GOAL_POSE_NORM
    traj=None,                           # sim-only knobs, kept so the bag matches
    save_path=None,                      # build_args key for key -- nothing here reads them
    headless=False,
    max_seconds=0.0,
    mppi_steps=200,
    plan_device='cuda',
    spacing=1.0,                         # --spacing 1   (the tuned sim run)
    smooth=9,
    # physics (only the subset the primitive calibration + IK read)
    pusher_radius=PUSHER_DIAMETER_CM / 2.0 * sim_units_per_cm(),   # sim 5.0, here 3 cm
    pusher_height=60.0,
    pusher_speed=20.0,
    tee_mass=1.0,
    friction=300.0,
    spin_friction=8000.0,
    # base.Sim reads this, and PushPrimitives.calibrate builds one PushSim per primitive,
    # so it has to be here even though nothing on the arm has obstacles that move.
    dynamic_obstacles=False,
    fps=60.0,
    substeps=4,
    port=8080,
    # T geometry / limit surface
    c_length=None,
    n_boundary=360,
    # reference tracking
    lookahead=3,
    advance_window=25,
    goal_dist=1.5,
    goal_deg=2.0,
    # transit planning
    clearance=TRANSIT_CLEARANCE_CM * sim_units_per_cm(),   # sim 1.0 unit = 2 mm here
    standoff=8.0 + STANDOFF_EXTRA_CM * sim_units_per_cm(),   # sim 8.0, plus 8 cm
    arc_radius_min=None,
    arc_radii=10,
    arc_clearance=ARC_CLEARANCE_CM * sim_units_per_cm(),    # sim 0.0, here 10 cm
    # Degrees of sweep per arc sample.  It sets how well the flown path tracks the circle
    # the ladder verified clear: the samples are the polyline ``--no-movec`` steps through
    # and ``--blend`` rides, and they are also where ``moveC``'s via point comes from -- a
    # sweep under 2 * this samples to its two ENDPOINTS alone, leaving no interior via and
    # no curve at all, just the chord.  At 15 cm radius a 90 deg sweep is 8 samples 3.4 cm
    # apart here, and the chord of a whole 50 deg step would cut 4.4 cm inside the circle
    # -- past the 3 cm ``ARC_CLEARANCE_CM`` margin, into the shape.
    arc_step=12.0,
    edge_lift=None,
    no_edge_slide=False,
    # --no_trans_collision: skip the collision check on a same-face slide and just fly it.
    no_trans_collision=False,
    wp_tol=1.0,
    entry_tol=0.1,
    transit_reach=None,
    transit_approach=None,
    transit_speed=80.0,
    # replanning
    replan_steps=200,
    no_replan=False,
    # discrete push primitives
    action_steps=10,
    push_len=10.0,                       # --push-len 10  (= spacing * action_steps)
    # No taper hardware DEFAULT: one rung, and it is the full push length.  The sim
    # measures a ladder (5 -> 10 in 4 steps) so the action can shrink as the T closes on
    # the goal; here every action is the same 10 units, and ENDGAME_PUSH_SCALE above
    # stands in for the taper by cutting the chosen stroke near the goal.
    #
    # To get the real taper instead, raise BOTH with push_t_realworld_run.py's
    # --push-len-min and --push-len-steps: push_length_ladder collapses to [push_len]
    # whenever steps == 1 OR push_len_min >= push_len, so moving one alone does nothing.
    # Every extra rung is measured, not interpolated -- one rollout per primitive per rung
    # (1000 at --n-contacts 100 x --n-dirs 10), cached to .push_primitives_*.npy.  Turn
    # ENDGAME_PUSH_SCALE back to 1.0 if you do, or the two taper on top of each other.
    push_len_min=10.0,                   # == push_len
    push_len_steps=1,                    # 1 disables the taper
    # The endgame shrink, in the bag so it is overridable per run; ENDGAME_PUSH_SCALE
    # at 1.0 turns it off entirely and every action is then the same length.
    endgame_dist_cm=ENDGAME_DIST_CM,
    endgame_deg=ENDGAME_DEG,
    endgame_scale=ENDGAME_PUSH_SCALE,
    len_bias=0.02,
    n_contacts=100,                      # --n-contacts 100
    n_dirs=30,                           # --n-dirs 10
    dir_spread=25.0,
    corner_margin=None,                  # None -> CONTACT_CORNER_MARGIN_CM, in world units
    action_timeout=4.0,
)

# How close to a footprint CORNER a contact may be, centimetres.  The pusher is a
# cylinder: pressed against a corner it touches an edge rather than a face, which is the
# one primitive the measured table cannot trust and the one a real pusher slides off.
# ``make_args`` turns this into the world units ``PushPrimitives`` wants.
#
# On the 12 cm T this is not a small trim: it strikes out the two 2 cm end faces of the
# crossbar and the 2 cm bottom of the stem entirely (both ends of a 2 cm face are inside
# 1.5 cm of a corner), leaving 27 of the 48 cm perimeter -- the long faces, shortened --
# to sample from.  Lower it if the controller needs to push on those small faces.
CONTACT_CORNER_MARGIN_CM = 1.0


def make_args(**overrides):
    """A ready-to-use parameter bag for the pushing stack (``argparse.Namespace``).

    Same names as ``push_t_demo_sim.build_args``; the defaults are the ones the sim run
    was tuned at -- ``--spacing 1 --n-contacts 100 --n-dirs 10 --push-len 10`` -- plus
    ``corner_margin``, which the sim leaves at 0 and this file resolves from
    ``CONTACT_CORNER_MARGIN_CM``.  ``teleport_interp`` --
    the "linear interp" flying mode -- is always on here (the real arm never teleports),
    and ``teleport`` is the baseline it builds on, exactly as ``push_t_demo_sim.main()``
    sets them.  Pass any knob as a keyword to override it.
    """
    d = dict(_DEFAULTS)
    d.update(overrides)
    d['teleport'] = True
    d['teleport_interp'] = True
    d['linear_interp'] = True
    ns = argparse.Namespace(**d)
    if ns.transit_approach is None:
        ns.transit_approach = ns.pusher_radius
    if ns.transit_reach is None:
        ns.transit_reach = 2.0 * ns.pusher_radius
    if ns.corner_margin is None:
        ns.corner_margin = CONTACT_CORNER_MARGIN_CM * sim_units_per_cm()
    return ns


# ======================================================================================
# the pushing stack (push_t_demo_sim.main(), minus the pymunk world)
# ======================================================================================
@dataclass
class PushStack:
    """Everything ``push_t_demo_sim.main()`` builds except the simulator itself."""

    args: argparse.Namespace
    ik: PushIK
    prims: PushPrimitives
    ctrl: PushController
    ref: np.ndarray                       # world-frame (T,3) SE(2) reference path
    obstacles: list
    env_polys: list
    tee_poly: object
    tee_height: float
    bbox_to_centroid: np.ndarray
    env_scale: float = 1.0
    env_center: tuple = (0.0, 0.0, 0.0)
    replanner: Replanner | None = None
    push_len: float = 0.0
    push_lens: list = field(default_factory=list)
    goal_norm: np.ndarray | None = None   # planner-frame (6,) goal; None with a raw ref=

    def reference_real(self):
        """The current reference path, in table centimetres."""
        return path_sim_to_real(self.ctrl.ref)

    def goal_real(self):
        """The goal this stack was planned to, in table cm ``(x, y, theta rad)``.

        ``None`` when the stack was built from a raw ``ref=`` path (no goal was set).
        """
        if self.goal_norm is None:
            return None
        return planner_pose_to_real(self.goal_norm, self.env_scale, self.env_center,
                                    self.bbox_to_centroid)


@dataclass
class Geometry:
    """The cheap half of the stack: meshes, footprints, and the planner-frame constants.

    No torch, no pymunk rollouts -- just what a visual or a coordinate query needs.
    """

    args: argparse.Namespace
    env_mesh: object
    tee_mesh: object
    env_polys: list                       # sim-frame shapely polygons, largest first
    tee_poly: object                      # sim-frame T footprint about its centroid
    tee_height: float
    bbox_to_centroid: np.ndarray
    env_scale: float = 1.0
    env_center: tuple = (0.0, 0.0, 0.0)

    def goal_pose_real(self, goal=None):
        """``GOAL_POSE_NORM`` (or ``goal``) in table cm ``(x, y, theta rad)``."""
        return planner_pose_to_real(goal_pose_norm(goal), self.env_scale, self.env_center,
                                    self.bbox_to_centroid)

    def env_rings_real(self):
        """Every env polygon boundary (exterior + holes) as ``(N,2)`` loops, table cm."""
        rings = []
        for poly in self.env_polys:
            for ring in [poly.exterior, *poly.interiors]:
                rings.append(path_sim_to_real(np.asarray(ring.coords)))
        return rings

    def tee_ring_real(self, tee_pose_real=None, tee_pose_world=None):
        """The T footprint as an ``(N,2)`` loop in table cm, posed at the given SE(2)."""
        if (tee_pose_real is None) == (tee_pose_world is None):
            raise ValueError('pass exactly one of tee_pose_real / tee_pose_world')
        if tee_pose_world is None:
            tee_pose_world = real_to_sim_pose(*tee_pose_real)
        x, y, th = (float(v) for v in tee_pose_world)
        c, s = math.cos(th), math.sin(th)
        body = np.asarray(self.tee_poly.exterior.coords)
        world = np.column_stack([c * body[:, 0] - s * body[:, 1] + x,
                                 s * body[:, 0] + c * body[:, 1] + y])
        return path_sim_to_real(world)


def load_geometry(args=None):
    """Meshes + footprints + planner-frame constants -- the torch-free part of the stack.

    Reads ``meta.json`` for ``env_scale`` / ``env_center`` when it is there (needed for
    the normalized frame); leaves them identity otherwise.
    """
    from shapely.geometry import Polygon

    args = args or make_args()

    env_mesh = base.load_mesh(args.env)
    tee_mesh = base.load_mesh(args.shape)
    env_polys = sorted(base.footprint(env_mesh), key=lambda p: -p.area)
    tee_polys = base.footprint(tee_mesh)
    assert len(tee_polys) == 1, f'expected one footprint for the shape, got {len(tee_polys)}'

    c = tee_polys[0].centroid
    z0 = tee_mesh.bounds[0][2]
    tee_mesh.apply_translation([-c.x, -c.y, -z0])
    tee_poly = Polygon([(x - c.x, y - c.y) for x, y in tee_polys[0].exterior.coords])
    tee_height = float(tee_mesh.extents[2])

    # The planner poses the shape about the z-up mesh's bounding-box centre.
    Vz = np.array([[float(t) for t in ln.split()[1:4]]
                   for ln in open(args.shape_zup) if ln.startswith('v ')])
    bbox_c = 0.5 * (Vz.min(0) + Vz.max(0))
    bbox_to_centroid = np.array([c.x - bbox_c[0], c.y - bbox_c[1]])

    env_scale, env_center = 1.0, (0.0, 0.0, 0.0)
    meta_path = os.path.join(args.dataPath, 'meta.json')
    if os.path.exists(meta_path):
        import json

        with open(meta_path) as f:
            meta = json.load(f)
        env_scale, env_center = meta['env_scale'], tuple(meta['env_center'])

    return Geometry(args=args, env_mesh=env_mesh, tee_mesh=tee_mesh, env_polys=env_polys,
                    tee_poly=tee_poly, tee_height=tee_height,
                    bbox_to_centroid=bbox_to_centroid,
                    env_scale=env_scale, env_center=env_center)


def load_womodel(args):
    """The Eikonal planner network alone -- ``load_planner`` minus its test-set pair.

    ``load_planner`` also hands back case ``i``'s start/goal poses; here both ends come
    from the real setup instead (the shape's measured pose and ``GOAL_POSE_NORM``), so
    case 0 is loaded only to satisfy the signature and its poses are dropped.
    """
    womodel, _start_norm, _goal_norm = load_planner(
        args.dataPath, args.modelPath, args.ckpt, 0, args.plan_device)
    return womodel


def build_stack(args=None, ref=None, start_pose_real=None, goal=None):
    """Recreate the sim demo's geometry + IK + primitives + controller.

    ``ref`` may be a precomputed world-frame ``(T,3)`` SE(2) path; if omitted it is
    planned (needs torch + the checkpoint + ``meta.json``) between:

    * **start** -- ``start_pose_real``, the shape's MEASURED table-frame pose
      ``(x cm, y cm, theta rad)``.  Required when ``ref`` is not given; there is no
      constant for it and no test-set fallback, because on hardware the T starts
      wherever it is sitting.
    * **goal** -- ``GOAL_POSE_NORM``, or ``goal`` when passed: the planner's NORMALIZED
      frame, ``(x, y, rz)`` or the full ``(x, y, z, rx, ry, rz)``.

    The same goal is handed to the :class:`Replanner`, so every re-plan aims at it too.
    The measured primitive tables are calibrated the same way and cached to the same
    files, so a table built by the sim demo is reused here and vice versa.
    """
    args = args or make_args()
    if ref is None and start_pose_real is None:
        raise ValueError("build_stack needs start_pose_real -- the shape's measured "
                         'table-frame pose (x cm, y cm, theta rad) -- or a precomputed ref=')

    geo = load_geometry(args)
    env_mesh, tee_mesh = geo.env_mesh, geo.tee_mesh
    env_polys, tee_poly = geo.env_polys, geo.tee_poly
    tee_height, bbox_to_centroid = geo.tee_height, geo.bbox_to_centroid
    env_scale, env_center = geo.env_scale, geo.env_center

    womodel = goal_norm = None
    if ref is None:
        goal_norm = goal_pose_norm(goal)
        start_norm = real_pose_to_planner(start_pose_real, env_scale, env_center,
                                          bbox_to_centroid)
        womodel = load_womodel(args)
        path_norm, dist = plan_from(womodel, start_norm, goal_norm,
                                    args.plan_device, args.mppi_steps)
        sx, sy, sth = (float(v) for v in start_pose_real)
        gr = planner_pose_to_real(goal_norm, env_scale, env_center, bbox_to_centroid)
        print(f'[plan] shape ({sx:.1f}, {sy:.1f}, {math.degrees(sth):+.0f}deg) cm -> goal '
              f'norm ({goal_norm[0]:+.3f}, {goal_norm[1]:+.3f}, {goal_norm[5]:+.3f} turns)'
              f' == ({gr[0]:.1f}, {gr[1]:.1f}, {math.degrees(gr[2]):+.0f}deg) cm: '
              f'{len(path_norm)} waypoints, final |goal - x| = {dist:.4f}')
        ref = planner_to_world(path_norm, env_scale, env_center, bbox_to_centroid)
    ref = resample_path(np.asarray(ref, dtype=float), args.spacing, args.smooth)

    # c is read off the simulator's friction anisotropy, not tuned separately.
    c_len = args.c_length if args.c_length else args.spin_friction / args.friction
    ik = PushIK(tee_poly, c_len, n_boundary=args.n_boundary)

    obstacles = list(env_polys)          # wall ring + interior blocks, all solid
    ctrl = PushController(ik, ref, args, obstacles)

    push_len = args.push_len if args.push_len else args.spacing * args.action_steps
    push_lens = push_length_ladder(push_len, args.push_len_min, args.push_len_steps)
    prims = PushPrimitives(ik, n_points=args.n_contacts, n_dirs=args.n_dirs,
                           spread_deg=args.dir_spread,
                           corner_margin=getattr(args, 'corner_margin', 0.0) or 0.0)
    print(f'[primitives] {len(prims)} actions = {prims.n_points} contacts x '
          f'{prims.n_dirs} directions; contacts kept off the corners by '
          f'{prims.corner_margin:.1f} units '
          f'= {prims.corner_margin / max(sim_units_per_cm(), 1e-9):.1f} cm '
          f'({prims.n_boundary_kept}/{len(ik.P)} boundary samples eligible)')
    prims.calibrate(args, tee_poly, push_lens,
                    cache_dir=os.path.dirname(args.dataPath) or '.')

    rep = None
    if womodel is not None and not args.no_replan:
        rep = Replanner(womodel, goal_norm, args.plan_device, args.replan_steps,
                        env_scale, env_center, bbox_to_centroid, args.spacing, args.smooth)

    return PushStack(args=args, ik=ik, prims=prims, ctrl=ctrl, ref=ref,
                     obstacles=obstacles, env_polys=env_polys, tee_poly=tee_poly,
                     tee_height=tee_height, bbox_to_centroid=bbox_to_centroid,
                     env_scale=env_scale, env_center=env_center, replanner=rep,
                     push_len=push_len, push_lens=push_lens, goal_norm=goal_norm)


# ======================================================================================
# table centimetres -> UR base frame, and the two transit-kind converters
# ======================================================================================
# The SDK is ``ur_rtde`` (``rtde_control.RTDEControlInterface``); poses are UR base-frame
# ``(x, y, z, rx, ry, rz)``, metres + rotation-vector radians, tool pointing straight
# down -- exactly the convention ``arm_calibrate.py`` uses.
#
# Curves: ur_rtde DOES support them.  ``moveC(pose_via, pose_to, speed, accel, blend,
# mode)`` traces a true circular arc through a via point, and ``moveL`` / ``moveJ`` /
# ``movePath`` accept blended waypoint paths (each corner rounded by a blend radius).
# There is no arbitrary spline.  So the same-face ("edge") transit -- three straight
# legs -- is a plain ``moveL`` sequence, and the around-the-object ("arc") transit is a
# ``moveL`` (retract) + ``moveC`` (the circle) + ``moveL`` (re-enter).


def _down_rotvec(yaw):
    """UR rotation vector for a straight-down tool at heading ``yaw`` (rad).

    A pi rotation about ``(cos(yaw/2), sin(yaw/2), 0)`` -- see ``arm_calibrate.down_rotvec``.
    Only the fallback anchor uses this; a calibrated ``RobotFrame`` takes the real
    recorded orientation out of ``p0``.
    """
    return np.array([math.pi * math.cos(yaw / 2.0),
                     math.pi * math.sin(yaw / 2.0), 0.0])


def load_startpos(path=None):
    """The **p0** calibration point ``arm_calibrate.py`` records.

    Straight through ``frame_conversions.load_origin_pose``, so this file and
    ``arm_test_id10`` read the same anchor out of the same file.  Returns the ``(6,)`` TCP
    pose ``(x, y, z, rx, ry, rz)`` in the UR base frame, metres / rotation vector.

    **p0 is the FIELD CENTRE, not the ID-10 marker.**  It is the TCP pose the UR reports
    with the end effector placed physically on the centre of the field -- sim ``(0, 0)``,
    ``offset_to_origin_cm`` (39.8, 32.5) cm away from the marker the camera measures
    against.  A missing file raises rather than falling back: an anchor that is silently
    wrong by a third of a metre is worse than no anchor at all.
    """
    return load_origin_pose(path)


@dataclass
class RobotFrame:
    """Table frame (cm) -> UR base frame (m, rotation vector), anchored on **p0**.

    This is ``arm_test_id10``'s click path, imported rather than reimplemented::

        table cm --FC.table_to_sim--> normalized --clamp_sim--> --FC.sim_to_robot--> base

    so the arm goes where the camera says, and the two programs cannot drift apart on what
    a table centimetre is worth.

    **p0 is the field centre.**  ``arm_calibrate.py`` records it with the end effector
    placed physically on the centre of the green box -- sim ``(0, 0)``, NOT the ID-10
    marker, which is 39.8 / 32.5 cm away and is where the *camera* frame starts.  (An
    earlier version of this class anchored on the marker and so sent every waypoint 51 cm
    off; that is the one thing to check first if the arm lands somewhere surprising.)

    * ``p0_pose[:2]``  is the field centre in base coordinates;
    * ``p0_pose[2]``   is the tool-tip height held for every waypoint (override with
      ``z_push_m``) -- **the tool never changes height**, there is no lift or approach leg;
    * ``p0_pose[3:6]`` is the tool orientation reproduced at every waypoint -- a
      cylindrical pusher is rotationally symmetric, so the wrist never turns.

    The scale/flip is ``frame_conversions.sim_to_robot``: ``base_xy = p0_xy + SIM_TO_BASE *
    sim_xy * FIELD_CM * 0.01`` with ``SIM_TO_BASE = (-1, -1)``, a half-turn about p0.  Since
    ``table_to_sim`` already turns table -> sim by 180 deg, the two cancel: **table +x is
    base +x**, and table -> base is a pure translation by p0.  They are kept as two
    measurements rather than one so that when the arm lands wrong it is clear which is off.

    Build it with :meth:`from_startpos`.
    """

    p0_pose: tuple = (0.0, 0.0, 0.0, math.pi, 0.0, 0.0)   # base frame, EE at FIELD CENTRE
    p0_joints: tuple | None = None
    z_push_m: float | None = None                          # override p0's height
    limit: float = FC.NORM_HALF          # hold targets inside the field; None = no clamp
    box: dict | None = None              # UR workspace clamp box, base metres
    cm_to_m: float = 0.01

    @property
    def origin_pose(self):
        """p0 as ``sim_to_robot`` wants it -- the ``(6,)`` pose, z optionally overridden."""
        o = np.asarray(self.p0_pose, dtype=float).copy()
        if self.z_push_m is not None:
            o[2] = float(self.z_push_m)
        return o

    @property
    def origin_xyz_m(self):
        """Base-frame ``(x, y, z)`` of the FIELD CENTRE -- p0, with z optionally overridden."""
        return self.origin_pose[:3]

    @property
    def tool_rotvec(self):
        """The rotation vector held at every waypoint -- p0's recorded orientation."""
        return self.origin_pose[3:6]

    def xy_to_base(self, xy_cm):
        """``(x, y)`` table centimetres -> ``(x, y)`` base metres.

        ``table_to_sim`` then ``sim_to_robot``, the same two hops ``arm_test_id10`` makes.
        Unclamped: :meth:`pose` is what applies the field and workspace limits.
        """
        xy_sim = table_to_sim(float(xy_cm[0]), float(xy_cm[1]))[:2]
        return sim_to_robot(xy_sim, self.origin_pose)[:2]

    def base_to_xy(self, pose_or_xy):
        """A UR pose (or bare ``(x, y)`` base metres) -> ``(x, y)`` table centimetres.

        Inverse of :meth:`xy_to_base`; reads the arm's live TCP back into the table frame
        that ``step_real`` wants.
        """
        xy_sim = robot_to_sim(np.asarray(pose_or_xy, dtype=float), self.origin_pose)
        return np.asarray(sim_to_table(xy_sim[0], xy_sim[1])[:2], dtype=float)

    def clamp(self, xy_cm):
        """``(pose, note)`` -- the base pose for a table point, held inside the limits.

        Two clamps, exactly ``arm_test_id10``'s: ``clamp_sim`` holds the point inside the
        field (a plan that leaves the box would otherwise be a full-speed lunge off the
        table), then the workspace box holds the base pose inside reach.  ``note`` is ''
        when neither fired.
        """
        xy_sim = table_to_sim(float(xy_cm[0]), float(xy_cm[1]))[:2]
        held = False
        if self.limit is not None:
            xy_sim, held = clamp_sim(xy_sim, self.limit)
        pose = sim_to_robot(xy_sim, self.origin_pose)
        boxed = False
        if self.box:
            lo = np.array([self.box['x'][0], self.box['y'][0]], dtype=float)
            hi = np.array([self.box['x'][1], self.box['y'][1]], dtype=float)
            keep = np.clip(pose[:2], lo, hi)
            boxed = bool(np.any(np.abs(keep - pose[:2]) > 1e-9))
            pose[:2] = keep
        note = ('held at the field edge' if held else
                'held at the workspace box' if boxed else '')
        return pose, note

    def pose(self, xy_cm, yaw_table=0.0):
        """``(x, y)`` table centimetres -> a full UR pose.

        p0's height and p0's orientation, translated by the clamped in-plane offset.
        ``yaw_table`` is accepted for signature compatibility and ignored -- the pusher is
        symmetric, so the wrist never turns.
        """
        pose, _note = self.clamp(xy_cm)
        return [float(v) for v in pose]

    @classmethod
    def from_startpos(cls, path=None, table_theta_deg=None, z_push_m=None, box=None,
                      limit=FC.NORM_HALF):
        """Build from the p0 point ``arm_calibrate.py`` wrote.

        ``path`` defaults to ``real_world_params.json`` -> ``robot.startpos_json``;
        ``z_push_m`` overrides p0's recorded height (``None`` keeps it); ``box`` defaults to
        ``robot.workspace_box_m``.

        ``table_theta_deg`` is accepted and IGNORED, with a warning when it is not 0: the
        table -> base rotation is no longer a free parameter.  ``table_to_sim`` supplies the
        180 deg table -> sim turn and ``SIM_TO_BASE`` supplies the sim -> base one, and the
        two cancel to leave table +x along base +x.  A third rotation here would double-count
        whichever of them was actually measured.
        """
        if table_theta_deg not in (None, 0.0):
            print(f'note: table_theta_deg={table_theta_deg} ignored -- the table -> base '
                  f'rotation now comes from frame_conversions (table_to_sim 180 deg + '
                  f'SIM_TO_BASE), which is what arm_test_id10 uses.')
        pose = load_startpos(path)
        return cls(p0_pose=tuple(float(v) for v in pose), z_push_m=z_push_m,
                   limit=limit,
                   box=PARAMS.workspace_box_m if box is None else box)

    def speed_mps(self, sim_units_per_s):
        """Sim units/s -> base-frame m/s (``sim_units_per_cm()`` = mesh units per cm)."""
        return float(sim_units_per_s) / max(sim_units_per_cm(), 1e-9) * self.cm_to_m


@dataclass
class RobotCmd:
    """One ``ur_rtde`` motion call in the UR base frame.

    ``op`` is ``'moveL'`` (straight to ``pose``) or ``'moveC'`` (circular arc from the
    arm's current pose, through ``via``, to ``pose``).  ``mode`` is the ur_rtde ``moveC``
    orientation mode: 0 = unconstrained, 1 = fixed relative to the arc.
    """

    op: str
    pose: list                              # (x, y, z, rx, ry, rz), metres / rotvec rad
    via: list | None = None                 # moveC only
    speed: float = 0.1                       # m/s
    accel: float = 0.25                      # m/s^2
    blend: float = 0.0                       # m
    mode: int = 0
    tag: str = ''
    # moveC only: the arc samples this one call stands for, base poses, ending ON ``pose``.
    # Carried so a controller without moveC can still fly the same arc -- see
    # :func:`expand_arcs`.  Never sent to ur_rtde; ``as_call`` ignores it.
    path: list | None = None

    def as_call(self):
        """``(method_name, positional_args)`` for ``RTDEControlInterface``."""
        if self.op == 'moveC':
            return 'moveC', (self.via, self.pose, self.speed, self.accel,
                             self.blend, self.mode)
        return 'moveL', (self.pose, self.speed, self.accel)


@dataclass
class RobotMove:
    """An ordered list of :class:`RobotCmd` -- the base-frame motion for one transit
    (plus, once ``step`` appends it, the push leg)."""

    kind: str                               # 'edge' or 'arc'
    cmds: list = field(default_factory=list)
    radius_cm: float | None = None

    def moveL_path(self):
        """The leading run of straight moves as one blended ``moveL`` path payload:
        ``[[x, y, z, rx, ry, rz, speed, accel, blend], ...]`` (ends at the first
        ``moveC``, if any)."""
        path = []
        for c in self.cmds:
            if c.op != 'moveL':
                break
            path.append(list(c.pose) + [c.speed, c.accel, c.blend])
        return path


def edge_transit_to_robot(transit_cm, robot, speed, accel, yaw_table=0.0, blend=0.0):
    """Same-face ("edge") transit -> a sequence of LINEAR robot moves.

    ``transit_cm`` = ``[out0, out1, hover]`` in table centimetres: the transit legs of
    ``StepPlan.waypoints_real`` (everything but the final push target).  The legs are lift
    off the face, slide along it, drop onto the hand-over point -- each a ``moveL``.  No
    curve is needed or wanted.
    """
    tags = ['lift', 'slide', 'drop'][-len(transit_cm):]
    cmds = [RobotCmd('moveL', robot.pose(p, yaw_table), speed=speed, accel=accel,
                     blend=blend, tag=t)
            for p, t in zip(transit_cm, tags)]
    if cmds:
        cmds[-1].blend = 0.0                 # land exactly on the hand-over point
    return RobotMove(kind='edge', cmds=cmds)


def arc_transit_to_robot(transit_cm, robot, speed, accel, radius_cm=None,
                         yaw_table=0.0, blend=0.0, mode=0):
    """Around-the-object ("arc") transit -> ``moveL`` + ``moveC`` + ``moveL``.

    ``transit_cm`` = ``[out0, <arc samples...>, out1, hover]`` in table centimetres.  The
    retract (to ``out0``) and re-entry (``out1`` -> ``hover``) legs are straight; the arc
    between ``out0`` and ``out1`` collapses into ONE ``moveC`` through the middle arc
    sample, so the UR controller traces the true circle instead of us streaming the
    polyline.  ``radius_cm`` is carried through for logging only.

    The via point has to be an INTERIOR sample.  ``_arc_transit``'s samples run from
    ``out0`` to ``out1`` INCLUSIVE -- its ``_arc`` starts on one and ends on the other --
    so the ends are repeats of the retract point and the arc target, and a ``moveC`` whose
    via IS its target is a degenerate circle the controller refuses outright.  A sweep
    shorter than ``2 * --arc-step`` samples to nothing but those two endpoints; there is
    then no circle to fly, because the chord is the whole of what the transit planner
    drew, so it goes out as the straight ``moveL`` it already is.
    """
    wp = [np.asarray(p, dtype=float) for p in transit_cm]
    out0, out1, hover = wp[0], wp[-2], wp[-1]
    arc_pts = wp[1:-2]
    interior = arc_pts[1:-1] if len(arc_pts) > 1 else arc_pts
    cmds = [RobotCmd('moveL', robot.pose(out0, yaw_table), speed=speed, accel=accel,
                     blend=blend, tag='retract')]
    if len(interior):
        via = interior[len(interior) // 2]
        cmds.append(RobotCmd('moveC', robot.pose(out1, yaw_table),
                             via=robot.pose(via, yaw_table), speed=speed, accel=accel,
                             blend=blend, mode=mode, tag='arc',
                             path=[robot.pose(p, yaw_table) for p in arc_pts]
                                  + [robot.pose(out1, yaw_table)]))
    else:
        cmds.append(RobotCmd('moveL', robot.pose(out1, yaw_table), speed=speed,
                             accel=accel, blend=blend, tag='arc'))
    cmds.append(RobotCmd('moveL', robot.pose(hover, yaw_table), speed=speed, accel=accel,
                         tag='reenter'))
    return RobotMove(kind='arc', cmds=cmds, radius_cm=radius_cm)


def expand_arcs(move: "RobotMove"):
    """A :class:`RobotMove` with every ``moveC`` replaced by the polyline it came from.

    ``moveC`` is not in every ``ur_rtde`` build -- an older ``RTDEControlInterface`` raises
    ``AttributeError: no attribute 'moveC'`` -- and the arc is not decoration: it is how
    the end effector gets around the T without pushing it over on the way.  So rather than
    cutting the corner (a straight line from the retract point to the re-entry point goes
    through the shape), each ``moveC`` becomes the ``moveL`` chain over the arc samples the
    transit planner drew it from, which is the same curve at the resolution of ``--arc-step``.

    The samples are held on the command by :func:`arc_transit_to_robot`, so nothing is
    re-derived here.  ``blend`` is dropped on the intermediate points: a blocking
    ``moveL(pose, speed, accel)`` takes no blend radius, so the arm comes to rest at each
    sample -- slower than one true circle, and the reason ``moveC`` stays the default where
    the controller has it.
    """
    cmds = []
    for c in move.cmds:
        if c.op != 'moveC':
            cmds.append(c)
            continue
        # No samples (a degenerate arc): the via point is still on the circle, so going
        # through it beats going straight at the target.
        pts = c.path if c.path else [p for p in (c.via, c.pose) if p is not None]
        cmds.extend(RobotCmd('moveL', p, speed=c.speed, accel=c.accel, tag='arc')
                    for p in pts)
    return RobotMove(kind=move.kind, cmds=cmds, radius_cm=move.radius_cm)


# ======================================================================================
# one action per step, for a physical arm
# ======================================================================================
@dataclass
class StepPlan:
    """The output of one ``ArmPushSession.step``: end-effector waypoints for one action.

    ``waypoints_world`` is the transit legs followed by the straight push, in sim units;
    ``waypoints_real`` is the same in table centimetres; ``speeds`` is the arm speed for
    each leg (sim units/s -- transit speed for the transit legs, push speed for the last).
    ``robot_move`` is the base-frame :class:`RobotMove` (transit converted by kind, with
    the push leg appended) when ``step`` was given a :class:`RobotFrame`, else ``None``.
    """

    status: str
    contact_world: np.ndarray | None = None
    push_length: float | None = None
    transit_kind: str | None = None                 # 'arc' or 'edge'
    transit_radius: float | None = None
    transit_widened: bool = False                   # arc flew the --arc-clearance margin
    edge_blocked: bool = False                      # an arc a same-face slide was possible for
    transit_world: list = field(default_factory=list)
    push_target_world: np.ndarray | None = None
    waypoints_world: list = field(default_factory=list)
    waypoints_real: list = field(default_factory=list)
    speeds: list = field(default_factory=list)
    robot_move: RobotMove | None = None
    done: bool = False


class ArmPushSession:
    """Drives ``push_t_demo_sim``'s controller for a physical arm, one action per ``step``.

    No pymunk world: the T's pose is whatever the camera reports and the end effector's
    position is whatever the arm reports, both passed in.  The cycle mirrors
    ``PrimitiveRunner`` in the sim demo but returns a plan instead of stepping physics:

        1. (optional) re-plan the T's reference from its measured pose
        2. pick the next (contact, push direction, push length) -- the argmin over the
           MEASURED primitive table against the chunk of path ``--action-steps`` ahead
        3. build the transit off the current contact: retract to the chosen circle, ride
           an arc (or slide one face) around the T, drop onto the hand-over point
        4. append the straight push to the action's body-frame end point

    Execute the waypoints, re-observe the poses, call ``step`` again.
    """

    def __init__(self, stack: PushStack, robot_frame: "RobotFrame | None" = None):
        self.stack = stack
        self.ctrl = stack.ctrl
        self.prims = stack.prims
        self.args = stack.args
        self.rep = stack.replanner
        self.push_len = stack.push_len
        self.robot = robot_frame          # table cm -> UR base frame; None = no conversion
        self.n_actions = 0

    # -- initial move: park the end effector on the stand-off circle ------------------
    def home_pose_world(self, tee_pose_world):
        """Where to send the end effector before the first push, in sim units.

        The stand-off circle (``ctrl.standoff`` = T bounding radius + pusher radius +
        ``--standoff``) BEHIND the reference's opening move, so the first thing the arm
        does is press rather than walk around the T.  Mirrors the ``pusher_start`` that
        ``push_t_demo_sim.main()`` computes.
        """
        ref = self.ctrl.ref
        pos = np.asarray(tee_pose_world[:2], dtype=float)
        j = min(self.args.lookahead, len(ref) - 1)
        heading = ref[j, :2] - ref[0, :2]
        if np.linalg.norm(heading) < 1e-9:
            heading = np.array([1.0, 0.0])
        heading = heading / np.linalg.norm(heading)
        return pos - heading * self.ctrl.standoff

    def home_pose_real(self, tee_pose_real):
        """``home_pose_world`` with a table-frame T pose in and a table-frame point out."""
        w = self.home_pose_world(real_to_sim_pose(*tee_pose_real))
        return np.asarray(sim_to_real(w[0], w[1]), dtype=float)

    def home_move(self, tee_pose_world, pusher_pos_world, *, robot=None, accel=0.25,
                  blend=0.0, movec_mode=0):
        """The FIRST move as a flown transit, not a straight line.  Sim-unit inputs.

        ``home_pose_world`` says WHERE the end effector has to be before the first push;
        going there with one ``moveL`` from wherever the run parked the arm draws a chord
        across the table that is perfectly happy to pass through the T -- which moves the
        shape before the controller has planned a single action, and the plan was made
        from the pose the camera read a second earlier.

        So the home move is flown the way every LATER move is flown: out to the stand-off
        circle, round it, down onto the target.  The one difference is the circle -- see
        ``PrimitiveRunner.plan_widest_transit``: the step loop lets the radius ladder pick
        the tightest arc that fits, which is right for a pusher already standing against
        the shape and wrong for one coming from across the table, so the home move is
        pinned to the widest rung (the object plus the disc plus ``--standoff``).  The
        home point sits on exactly that circle by construction (``ctrl.standoff``), so the
        arc runs all the way to it and the drop leg is a no-op.

        Returns a :class:`StepPlan` with ``status='HOME'``, the waypoints in both frames,
        and -- given a :class:`RobotFrame` -- ``robot_move``, an arc transit at TRANSIT
        speed throughout (there is no push leg to append).
        """
        pos = np.asarray(tee_pose_world[:2], dtype=float)
        ang = float(tee_pose_world[2])
        home_world = np.asarray(self.home_pose_world(tee_pose_world), dtype=float)

        start_local = to_local(np.asarray(pusher_pos_world[:2], dtype=float), pos, ang)
        q_local, _marks = self.ctrl.plan_widest_transit(
            start_local, to_local(home_world, pos, ang), pos, ang)
        waypoints_world = [np.asarray(to_world(w, pos, ang), dtype=float) for w in q_local]
        waypoints_real = [np.asarray(sim_to_real(w[0], w[1]), dtype=float)
                          for w in waypoints_world]

        transit_speed = self.args.transit_speed or self.args.pusher_speed
        radius = float(self.ctrl.transit_radius)

        rf = robot if robot is not None else self.robot
        robot_move = None
        if rf is not None:
            robot_move = arc_transit_to_robot(
                waypoints_real, rf, rf.speed_mps(transit_speed), float(accel),
                radius_cm=radius / max(sim_units_per_cm(), 1e-9),
                blend=blend, mode=movec_mode)

        return StepPlan(
            status='HOME',
            transit_kind='arc',
            transit_radius=radius,
            transit_world=list(waypoints_world),
            waypoints_world=list(waypoints_world),
            waypoints_real=waypoints_real,
            speeds=[transit_speed] * len(waypoints_world),
            robot_move=robot_move,
        )

    def home_move_real(self, tee_pose_real, pusher_xy_real, **kw):
        """``home_move`` with table-frame inputs: T pose ``(x cm, y cm, theta rad)`` and
        the end-effector position ``(x cm, y cm)``.  Extra keywords (``robot``, ``accel``,
        ``blend``, ``movec_mode``) pass straight through."""
        return self.home_move(real_to_sim_pose(*tee_pose_real),
                              real_to_sim(pusher_xy_real[0], pusher_xy_real[1]), **kw)

    # -- the T's pose in the planner's normalized frame ------------------------------
    def get_shape_pos(self, tee_pose_real=None, tee_pose_world=None):
        """The T's pose in the planner's VIRTUAL (normalized) frame -> ``(6,)`` array.

        Layout ``(x, y, z, rx, ry, rz)`` with ``z, rx, ry`` pinned to 0 and ``rz`` in
        turns -- exactly what the Eikonal planner, the checkpoint's test set, and
        ``Replanner`` consume.  This is the inverse of the lift ``push_t_demo_sim`` does
        at load time:

            camera table pose (cm, about whatever the marker frame is)
              -> sim world pose  (mesh units, about the footprint centroid)
              -> normalized pose (about the z-up bbox centre, / env_scale)

        Pass the T pose EITHER in table centimetres (``tee_pose_real``) OR already in sim
        world units (``tee_pose_world``); exactly one.  Needs ``env_scale`` / ``env_center``
        from ``meta.json``, so it only means anything when the stack was built with a
        planned reference (not a bare ``--ref`` array).
        """
        if (tee_pose_real is None) == (tee_pose_world is None):
            raise ValueError('pass exactly one of tee_pose_real / tee_pose_world')
        if tee_pose_real is not None:
            tee_pose_world = real_to_sim_pose(*tee_pose_real)
        return world_to_planner(np.asarray(tee_pose_world, dtype=float),
                                self.stack.env_scale, self.stack.env_center,
                                self.stack.bbox_to_centroid)

    # -- endgame ---------------------------------------------------------------------
    def endgame_scale(self, tee_pos_world, tee_ang):
        """``args.endgame_scale`` when the T is nearly there, else ``1.0``.

        The knobs come off the parameter bag (``ENDGAME_*`` are only their defaults), so a
        run can turn the shrink off with ``--endgame-scale 1.0`` and get one push length
        for every action.

        "Nearly there" is BOTH tests: the centroid within ``ENDGAME_DIST_CM`` of the goal
        centroid AND the heading within ``ENDGAME_DEG`` of the goal heading -- the same
        conjunction the run loop stops on, so the shrink covers exactly the approach to
        the pose the run is trying to reach.  It used to be either test, which fired the
        shrink on any T whose heading happened to pass through the band with the whole
        translation still to do: the stroke dropped to a third for actions that were not
        endgame actions at all, and the run crept the last 20 cm in 0.7 cm bites.

        Measured against ``ctrl.ref[-1]`` rather than the reference the stack was built
        with, so a replan that moves the path still ends at the pose actually being
        tracked.
        """
        scale = float(getattr(self.args, 'endgame_scale', ENDGAME_PUSH_SCALE))
        if scale >= 1.0:                     # switched off: one push length for every case
            return 1.0
        goal = np.asarray(self.ctrl.ref[-1], dtype=float)
        d_cm = float(np.linalg.norm(goal[:2] - np.asarray(tee_pos_world, dtype=float)[:2])
                     / max(sim_units_per_cm(), 1e-9))
        d_deg = abs(math.degrees(wrap(float(goal[2]) - float(tee_ang))))
        near = (d_cm <= float(getattr(self.args, 'endgame_dist_cm', ENDGAME_DIST_CM))
                and d_deg <= float(getattr(self.args, 'endgame_deg', ENDGAME_DEG)))
        return scale if near else 1.0

    def _shrink_endgame_push(self, pos, ang):
        """Cut the push the controller just chose to ``endgame_scale`` of its length.

        Only the push TARGET moves.  ``ctrl.final_local`` is ``contact + u * L``
        (push_t_demo_sim.py:1129), so backing it off by ``u * L * (1 - scale)`` leaves
        ``contact + u * L * scale`` -- the same contact point, the same push direction, the
        same entry point, and therefore the same transit.  ``last_len`` is trued up with
        it so ``StepPlan.push_length`` and the run log report the stroke actually flown.
        """
        scale = self.endgame_scale(pos, ang)
        if scale >= 1.0:
            return 1.0
        cut = float(self.ctrl.last_len) * (1.0 - scale)
        self.ctrl.final_local = (np.asarray(self.ctrl.final_local, dtype=float)
                                 - np.asarray(self.ctrl.push_dir_local, dtype=float) * cut)
        self.ctrl.last_len = float(self.ctrl.last_len) * scale
        return scale

    # -- one action -----------------------------------------------------------------
    def step(self, tee_pose_world, pusher_pos_world, replan=True, *,
             robot=None, accel=0.25, transit_accel=None, blend=0.0, movec_mode=0):
        """Plan ONE push action from the observed poses.  Sim-unit inputs.

        ``tee_pose_world`` is ``(x, y, theta)``; ``pusher_pos_world`` is ``(x, y)`` (the
        current end-effector position).  Returns a :class:`StepPlan` -- ``status`` is
        ``'GOAL'`` / ``'BLOCKED'`` / ``'ACTION ...'``; when it is not ``GOAL`` or
        ``BLOCKED`` the waypoint lists are populated.

        When a :class:`RobotFrame` is available (``robot=`` here, or passed to the
        constructor) the transit is also converted to UR base-frame motion: an edge slide
        goes through ``edge_transit_to_robot`` (all ``moveL``), an arc through
        ``arc_transit_to_robot`` (``moveL`` + ``moveC`` + ``moveL``), and the straight
        push is appended as a final ``moveL`` at push speed.  The result is
        ``StepPlan.robot_move``.

        ``accel`` is the push leg's acceleration; ``transit_accel`` (default: ``accel``)
        is the transit's.  They are separate because the transit legs are SHORT -- a 6 cm
        retract at 0.25 m/s^2 tops out at 12 cm/s no matter what ``--transit-speed`` says,
        so on the fly-around it is the acceleration, not the speed cap, that sets the
        clock.  The push wants the gentle one; the transit does not.
        """
        pos = np.asarray(tee_pose_world[:2], dtype=float)
        ang = float(tee_pose_world[2])

        if replan and self.rep is not None:
            fresh = self.rep.solve((pos[0], pos[1], ang))
            if fresh is not None:
                self.ctrl.adopt(fresh)

        tele, _final, contact_w, status = self.ctrl.begin_primitive(
            pos, ang, self.prims, self.push_len)
        if tele is None:
            return StepPlan(status=status, done=(status == 'GOAL'))

        self.n_actions += 1
        self._shrink_endgame_push(pos, ang)
        pusher_local = to_local(np.asarray(pusher_pos_world[:2], dtype=float), pos, ang)
        q_local, _marks = self.ctrl.plan_transit(
            pusher_local, self.ctrl.entry_local, pos, ang)

        transit_world = [np.asarray(to_world(w, pos, ang), dtype=float) for w in q_local]
        push_target_world = np.asarray(
            to_world(self.ctrl.final_local, pos, ang), dtype=float)
        waypoints_world = transit_world + [push_target_world]

        transit_speed = self.args.transit_speed or self.args.pusher_speed
        speeds = [transit_speed] * len(transit_world) + [self.args.pusher_speed]

        waypoints_real = [np.asarray(sim_to_real(w[0], w[1]), dtype=float)
                          for w in waypoints_world]
        transit_radius = (None if self.ctrl.transit_kind == 'edge'
                          else float(self.ctrl.transit_radius))

        # Convert the transit to UR base-frame motion, by kind.
        rf = robot if robot is not None else self.robot
        robot_move = None
        if rf is not None:
            transit_cm, push_cm = waypoints_real[:-1], waypoints_real[-1]
            v_transit = rf.speed_mps(transit_speed)
            v_push = rf.speed_mps(self.args.pusher_speed)
            a_transit = float(accel if transit_accel is None else transit_accel)
            if self.ctrl.transit_kind == 'edge':
                robot_move = edge_transit_to_robot(transit_cm, rf, v_transit, a_transit,
                                                   blend=blend)
            else:
                robot_move = arc_transit_to_robot(
                    transit_cm, rf, v_transit, a_transit,
                    radius_cm=(None if transit_radius is None
                               else transit_radius / max(sim_units_per_cm(), 1e-9)),
                    blend=blend, mode=movec_mode)
            robot_move.cmds.append(RobotCmd('moveL', rf.pose(push_cm), speed=v_push,
                                            accel=accel, tag='push'))

        return StepPlan(
            status=status,
            contact_world=np.asarray(contact_w, dtype=float),
            push_length=float(self.ctrl.last_len),
            transit_kind=self.ctrl.transit_kind,
            transit_radius=transit_radius,
            transit_widened=bool(getattr(self.ctrl, 'transit_widened', False)),
            edge_blocked=bool(getattr(self.ctrl, 'edge_blocked', False)),
            transit_world=transit_world,
            push_target_world=push_target_world,
            waypoints_world=waypoints_world,
            waypoints_real=waypoints_real,
            speeds=speeds,
            robot_move=robot_move,
            done=False,
        )

    def step_real(self, tee_pose_real, pusher_xy_real, replan=True, **kw):
        """``step`` with table-frame inputs: T pose ``(x cm, y cm, theta rad)`` and the
        end-effector position ``(x cm, y cm)``.  Extra keywords (``robot``, ``accel``,
        ``transit_accel``, ``blend``, ``movec_mode``) pass straight through to
        :meth:`step`."""
        tee_w = real_to_sim_pose(*tee_pose_real)
        pusher_w = real_to_sim(pusher_xy_real[0], pusher_xy_real[1])
        return self.step(tee_w, pusher_w, replan=replan, **kw)


# ======================================================================================
# smoke test / CLI
# ======================================================================================
def build_cli():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--linear-interp', action='store_true', default=True,
                    help='the only transit mode here: the arm FLIES each transit '
                         '(retract -> arc/slide -> re-enter) and is never teleported. '
                         'On by default, kept for parity with push_t_demo_sim.')
    ap.add_argument('--start', type=float, nargs=3, default=(20.0, 32.5, 0.0),
                    metavar=('X_CM', 'Y_CM', 'DEG'),
                    help='stand-in for the shape reading this file has no camera for: '
                         'the T pose the reference is planned FROM, table cm + degrees')
    ap.add_argument('--ref', default=None,
                    help='replay a saved world-frame (T,3) path instead of planning')
    ap.add_argument('--plan-device', default='cuda')
    ap.add_argument('--no-replan', action='store_true')
    return ap.parse_args()


def main():
    cli = build_cli()
    args = make_args(plan_device=cli.plan_device, no_replan=cli.no_replan)
    ref = np.load(cli.ref) if cli.ref else None
    sx, sy, sdeg = cli.start
    stack = build_stack(args, ref=ref,
                        start_pose_real=(sx, sy, math.radians(sdeg)))
    # Base-frame transform anchored on p0 -- the TCP pose arm_calibrate.py recorded with
    # the end effector on the FIELD CENTRE, sim (0, 0) (calibration/arm/startpos.json).
    robot = RobotFrame.from_startpos()
    sess = ArmPushSession(stack, robot_frame=robot)

    r = stack.ctrl.standoff
    g, gr = goal_pose_norm(), stack.goal_real()
    print(f'[realworld] reference: {len(stack.ref)} waypoints; stand-off radius '
          f'{r:.1f} sim units ({r / sim_units_per_cm():.1f} cm)')
    print(f'[realworld] goal norm ({g[0]:+.3f}, {g[1]:+.3f}, {g[5]:+.3f} turns)'
          + ('' if gr is None else f' == ({gr[0]:.1f}, {gr[1]:.1f}) cm'))

    # Geometry smoke test -- no hardware, no physics.  Pretend the T is sitting at the
    # reference start and the arm is anywhere; show the home move and the first action.
    tee = stack.ref[0].copy()
    home = sess.home_pose_world(tee)
    print(f'[realworld] 1. home the end effector to '
          f'sim {np.round(home, 1)} / table {np.round(sim_to_real(home[0], home[1]), 1)} cm')

    plan = sess.step(tee, home, replan=False)
    print(f'[realworld] 2. step() -> {plan.status}')
    if plan.waypoints_world:
        rad = '' if plan.transit_radius is None else f' r={plan.transit_radius:.0f}'
        print(f'            push length {plan.push_length:.0f}, transit '
              f'{plan.transit_kind}{rad}, {len(plan.waypoints_real)} waypoints:')
        for k, (w, v) in enumerate(zip(plan.waypoints_real, plan.speeds)):
            tag = 'push' if k == len(plan.waypoints_real) - 1 else 'transit'
            print(f'              {tag:7s}  ({w[0]:7.2f}, {w[1]:7.2f}) cm   @ {v:.0f}')
    if plan.robot_move is not None:
        print(f'[realworld] 3. robot_move ({plan.robot_move.kind}): '
              f'{len(plan.robot_move.cmds)} ur_rtde calls')
        for c in plan.robot_move.cmds:
            xyz = ', '.join(f'{v:+.3f}' for v in c.pose[:3])
            print(f'              {c.op:5s} {c.tag:8s} ({xyz}) m  @ {c.speed:.3f} m/s'
                  + (f'  via ({", ".join(f"{v:+.3f}" for v in c.via[:3])})'
                     if c.via else ''))
    print('[realworld] wire step()/step_real() into the camera+arm loop: observe the T '
          'pose and end-effector position, run robot_move.cmds on the RTDEControlInterface, '
          'repeat.')


if __name__ == '__main__':
    main()
