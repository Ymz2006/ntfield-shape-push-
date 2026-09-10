"""Real-world push-T -- the full control loop.

Ties the pieces in ``push_t_demo_realworld.py`` together into one closed loop:

    home the end effector onto the stand-off circle (flown round it, never straight
        through the shape)
    repeat:
        observe the T pose            (camera, table frame)
        observe the end-effector pos  (robot RTDE, table frame)
        plan one action               (ArmPushSession.step_real)
        execute it                    (moveL / moveC on the UR)
    until the T is close enough to the goal -- centroid within ``--goal-cm`` AND heading
    within ``--goal-deg`` -- blocked, or the action budget runs out.

Two back ends behind small interfaces so the loop itself is hardware-agnostic:

* **Arm** -- ``RtdeArm`` (``ur_rtde``) or ``DryRunArm`` (prints, tracks a virtual TCP).
* **TeePose** -- ``RealSenseTeePose`` (ArUco marker on the T) or ``ReplayTeePose``
  (walks the T along the planned reference; no camera, no GPU).

Usage (inside the Docker image; from the repo root):

    # no hardware -- exercises homing, the step loop and the live view
    python push_t_realworld_run.py --dry-run --ref plan.npy --viz

    # live: UR at --ip, RealSense tracking marker --tee-marker-id glued to the T
    python push_t_realworld_run.py --ip 192.168.10.2 --z-push 0.02 \\
        --tee-marker-id 2 --viz

``--viz`` opens ``push_t_realworld_viz.TableView`` in this process -- the env, the T where
the camera sees it, the goal ghost, the Eikonal planner's path and the end effector, all
as meshes in table centimetres, fed the same observations the controller acts on.
``--viz-state`` instead writes those observations to a JSON file, for
``push_t_realworld_viz.py --state /tmp/table_state.json`` in a second process.

The reference is planned from the T's MEASURED pose (the camera is opened before the
plan) to ``push_t_demo_realworld.GOAL_POSE_NORM``, which is in the planner's NORMALIZED
frame -- edit that constant to move the goal; there is no test-set case to pick any more.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass

import numpy as np

import frame_conversions as FC
import push_t_demo_realworld as rw
from real_world_params import PARAMS

DEF_ACCEL = 0.25          # m/s^2 for every moveL / moveC unless overridden

# Blend radius in METRES for the around-the-object ARC, and for nothing else -- see
# ``RtdeArm.run``.  0.0 keeps every command a blocking move that ends at rest.
DEF_BLEND = 0.0

# Heading tolerance, degrees: how far the T's measured heading may sit from the goal's and
# still count as arrived.  The stop test is BOTH this and ``--goal-cm`` -- a T parked on
# the goal centroid but turned a quarter turn out of it is not the pose the plan asked
# for, and the centroid distance alone cannot see that.
DEF_GOAL_DEG = 3.0

# Two arc waypoints closer together than this (metres) are the same point as far as a
# blended path is concerned -- see ``RtdeArm._fly_arc_blended``.  A millimetre, because
# the first thing this is measured against is the arm's OWN position after a blocking
# move: it lands within a fraction of a millimetre of the target, not on it, and a leg
# that short is not a leg to be flown, it is where the tool already is.
DUP_TOL_M = 1e-3

# --------------------------------------------------------------------------------------
# the corner park -- where the arm goes first, before it homes
# --------------------------------------------------------------------------------------
# Table-frame centimetres of the ID-20 table corner.  The table frame is pinned to the
# ID-10 marker at (0, 0) with +x along the LENGTH and +y along the WIDTH, and 10 -> 20 is
# the width (``camera_test_id10``'s header: "10 -> 20 is the width and is now +y"), so
# ID 20 is straight up +y at the table's width.  63.0 cm is ``camera_test_id10.WIDTH_CM``,
# the nominal that file draws the table out to; it is repeated rather than imported
# because that module pulls in OpenCV and the RealSense SDK, which this file keeps behind
# a lazy import so ``--dry-run`` works without them.  Re-measure one, re-measure both.
CORNER_ID20_CM = (0.0, 63.0)

# How far IN from that corner the park sits, along the straight line to the field centre.
CORNER_INSET_CM = 10.0

# How high the park move flies: straight up this far, across at that height, straight back
# down.  The descent lands on the pushing height, not on wherever the arm started.
CORNER_LIFT_CM = 10.0


def corner_park_cm(inset_cm=CORNER_INSET_CM, corner_cm=CORNER_ID20_CM):
    """Table cm ``inset_cm`` in from a table corner, along the line to the FIELD CENTRE.

    The field centre is ``FC.FIELD_CENTER_CM`` -- the sim's ``(0, 0)`` and the point p0 is
    calibrated on, so it is the origin the arm's own frame is anchored to, not the ID-10
    marker 39.8 / 32.5 cm away.
    """
    corner = np.asarray(corner_cm, dtype=float)
    d = np.asarray(FC.FIELD_CENTER_CM, dtype=float) - corner
    n = float(np.linalg.norm(d))
    if n < 1e-9:
        return corner
    return corner + d / n * float(inset_cm)


def move_via_lift(arm, robot, xy_cm, speed, accel=DEF_ACCEL, lift_cm=CORNER_LIFT_CM):
    """Up ``lift_cm``, straight across at that height, straight down ``lift_cm``.

    Three ``moveL``s.  The traverse is flown ``lift_cm`` above the PUSHING height at the
    far end, so the final leg is exactly ``lift_cm`` of descent and lands on the height
    every other move in this file uses -- rather than on whatever height the arm happened
    to start at, which is not a plane the table knows about.  The climb is ``lift_cm``
    from wherever the tool is now.

    Returns the table centimetres actually reached: ``RobotFrame.pose`` clamps to the
    field and the workspace box, so the target and the landing are not always the same
    point.
    """
    dz = float(lift_cm) * 0.01
    here = list(arm.tcp_pose())
    there = list(robot.pose(xy_cm))
    arm.move_pose(here[:2] + [here[2] + dz] + here[3:], speed, accel)
    arm.move_pose(there[:2] + [there[2] + dz] + there[3:], speed, accel)
    arm.move_pose(there, speed, accel)
    return robot.base_to_xy(there)


def _wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


# ======================================================================================
# arm back ends
# ======================================================================================
class DryRunArm:
    """No robot: prints each move and tracks a virtual TCP in table centimetres."""

    def __init__(self, robot: rw.RobotFrame, start_xy_cm=(0.0, 0.0), blend=DEF_BLEND):
        self.robot = robot
        self._xy = np.asarray(start_xy_cm, dtype=float)
        self.blend = max(float(blend), 0.0)

    def tcp_xy_cm(self):
        return self._xy.copy()

    def move_to_cm(self, xy_cm, speed, accel=DEF_ACCEL):
        self._xy = np.asarray(xy_cm, dtype=float)
        print(f'    [arm] moveL  -> ({self._xy[0]:7.2f}, {self._xy[1]:7.2f}) cm  @ {speed:.3f} m/s')

    def tcp_pose(self):
        """The virtual TCP as a full base pose, at the working height."""
        return list(self.robot.pose(self._xy))

    def move_pose(self, pose, speed, accel=DEF_ACCEL):
        """moveL to a base pose given outright -- the one move that may leave the plane."""
        pose = [float(v) for v in pose]
        self._xy = self.robot.base_to_xy(pose)
        print(f'    [arm] moveL  -> ({self._xy[0]:7.2f}, {self._xy[1]:7.2f}) cm  '
              f'z={pose[2]:.3f} m  @ {speed:.3f} m/s')

    def run(self, move: rw.RobotMove):
        for c in move.cmds:
            self._xy = self.robot.base_to_xy(c.pose)
            extra = ''
            if c.via is not None:
                v = self.robot.base_to_xy(c.via)
                extra = f'  via ({v[0]:6.1f}, {v[1]:6.1f})'
                if self.blend > 0.0 and c.path:
                    extra += (f'  [blended moveL path, {len(c.path)} pts, '
                              f'r<={self.blend * 100:.1f} cm]')
            print(f'    [arm] {c.op:5s} {c.tag:8s} -> ({self._xy[0]:7.2f}, '
                  f'{self._xy[1]:7.2f}) cm{extra}  @ {c.speed:.3f} m/s')

    def close(self):
        pass


class RtdeArm:
    """UR arm over ``ur_rtde``.  Blocking moves; every target is safety-checked first.

    ``moveC`` is only in some ``ur_rtde`` builds.  When this one does not have it (or
    ``--no-movec`` is passed) every arc is flown as the ``moveL`` chain over the arc
    samples it was planned from -- ``rw.expand_arcs`` -- instead of one circular move.
    The path is the same curve; the arm just stops at each sample along it.

    ``blend`` (``--blend``, metres) buys the curve back on those builds and takes
    precedence over ``moveC`` when it is set: the arc samples go out as ONE blended
    ``moveL`` path, so the tool rounds every sample corner instead of coming to rest on
    it.  It applies to the ARC ONLY -- the retract, the re-entry, the push, the edge
    slides and the corner park stay plain blocking ``moveL``s that land exactly on their
    target, which is what the hand-over point and the push start need.
    """

    def __init__(self, ip, robot: rw.RobotFrame, accel=DEF_ACCEL, use_movec=True,
                 blend=DEF_BLEND):
        from rtde_control import RTDEControlInterface
        from rtde_receive import RTDEReceiveInterface

        self.robot = robot
        self.accel = accel
        print(f'[arm] connecting to {ip} ...')
        self.c = RTDEControlInterface(ip)
        self.r = RTDEReceiveInterface(ip)
        print(f'[arm] connected; TCP offset {np.round(self.c.getTCPOffset(), 4).tolist()}')
        self.has_movec = bool(use_movec) and hasattr(self.c, 'moveC')
        self.blend = max(float(blend), 0.0)
        if self.has_movec:
            how = 'moveC (one true circle)'
            if self.blend > 0.0:
                how += f'; --blend {self.blend * 100:.1f} cm held in reserve, unused'
        elif self.blend > 0.0:
            how = (f'one blended moveL path over the arc samples, '
                   f'r<={self.blend * 100:.1f} cm')
        else:
            how = ('a moveL per arc sample, one stop each -- pass --blend to fly them as '
                   'one path')
        why = '' if use_movec else ' (--no-movec)'
        if not self.has_movec:
            why = (' (this ur_rtde has no moveC; moves available: '
                   f'{", ".join(sorted(m for m in dir(self.c) if m.startswith("move")))})'
                   if use_movec else why)
        print(f'[arm] arcs fly as {how}{why}')

    def tcp_xy_cm(self):
        return self.robot.base_to_xy(self.r.getActualTCPPose())

    def _guard(self, pose):
        if not self.c.isPoseWithinSafetyLimits(list(pose)):
            raise RuntimeError(f'pose outside safety limits: {np.round(pose, 4).tolist()}')

    def move_to_cm(self, xy_cm, speed, accel=None):
        pose = self.robot.pose(xy_cm)
        self._guard(pose)
        if not self.c.moveL(pose, speed, accel or self.accel):
            raise RuntimeError('moveL refused')

    def tcp_pose(self):
        """The arm's live TCP pose, base frame."""
        return list(self.r.getActualTCPPose())

    def move_pose(self, pose, speed, accel=DEF_ACCEL):
        """moveL to a base pose given outright -- the one move that may leave the plane.

        Everything else in this file goes through ``RobotFrame.pose``, which pins z to the
        pushing height; the corner park needs to lift off it, so the pose arrives built.
        Still safety-checked, exactly like every other target.
        """
        pose = [float(v) for v in pose]
        self._guard(pose)
        if not self.c.moveL(pose, speed, accel or self.accel):
            raise RuntimeError('moveL refused')

    def _fly_arc_blended(self, cmd):
        """The arc samples of one ``moveC`` as a single blended ``moveL`` path.

        ur_rtde's path overload -- ``moveL([[x, y, z, rx, ry, rz, speed, accel, blend],
        ...])`` -- runs the whole list as ONE motion: each corner is replaced by an arc
        that leaves the incoming leg ``blend`` before the waypoint and rejoins the
        outgoing one ``blend`` after it, so the tool rounds the corner at speed instead of
        stopping on it.  The last waypoint has to keep blend 0 -- there is no leg after it
        to round into, and that is the one point the path is guaranteed to hit exactly.

        Two things have to be trued up before the payload goes out:

        * DUPLICATES.  ``_arc_transit`` builds its transit as ``[out0] + arc + [out1,
          hover]`` and its ``arc`` already STARTS on ``out0`` and ENDS on ``out1``, so the
          samples ``arc_transit_to_robot`` hands on repeat the point the arm is parked at
          and the point it is aiming for.  A zero-length leg has no room for any blend at
          all, which would clamp the whole path back to a stop-at-every-sample chain.
        * THE CLAMP, PER CORNER.  A blend may not reach past the middle of either leg it
          joins -- two adjacent blends would overlap and the controller rejects the whole
          path -- so each waypoint is held to just under half of the shorter of ITS OWN
          two legs.  One clamp taken over the whole path instead would let a single short
          leg (the arm sitting a hair off the retract point, a sample pair the transit
          planner put close together) shrink the radius on every OTHER corner to nothing,
          and a corner blended by a fraction of a millimetre is a corner the arm stops
          on: with ``accel`` at 0.25 m/s^2 a 0.07 mm blend has to be taken at ~1 cm/s
          against a 10 cm/s transit.  That is the stutter this whole flag exists to
          remove, so it must not come back through the clamp.
        """
        here = self.tcp_pose()
        pts = [[float(v) for v in here]]
        for p in cmd.path:
            p = [float(v) for v in p]
            if float(np.linalg.norm(np.subtract(p[:3], pts[-1][:3]))) > DUP_TOL_M:
                pts.append(p)
        # legs[i] is the leg ENDING at pts[i] -- legs[0] is the one from ``here``, which
        # the path does not list but does fly.
        legs = [float(np.linalg.norm(np.subtract(b[:3], a[:3])))
                for a, b in zip(pts, pts[1:])]
        pts = pts[1:]                       # where the arm already is is not a waypoint
        if not pts:                          # the whole arc collapsed onto this point
            return True
        for p in pts:
            self._guard(p)
        accel = cmd.accel or self.accel
        path = []
        for i, p in enumerate(pts):
            joins = legs[i:i + 2]            # incoming leg, and the outgoing one if any
            r = 0.0 if i == len(pts) - 1 else min(self.blend, 0.49 * min(joins))
            path.append(p + [cmd.speed, accel, r])
        return bool(self.c.moveL(path))

    def _fly_arc_stepped(self, cmd):
        """A ``moveC`` this build cannot fly as one motion: the same samples, one blocking
        ``moveL`` each -- ``rw.expand_arcs``' chain, for this one command."""
        for c in rw.expand_arcs(rw.RobotMove(kind='arc', cmds=[cmd])).cmds:
            self._guard(c.pose)
            if not self.c.moveL(c.pose, c.speed, c.accel or self.accel):
                return False
        return True

    def run(self, move: rw.RobotMove):
        """Fly one planned move.  Only the ``moveC`` (the arc) has a choice of how.

        ``moveC`` first when this build has it: it is the true circle, one continuous
        motion with no corners in it at all, so it is smooth at ANY radius -- including
        the tight rungs of the arc ladder, where a sampled polyline has legs too short to
        blend and the arm has to slow into every one of them.  ``--blend`` is what the
        arc falls back to when there is no ``moveC`` to call.
        """
        for c in move.cmds:
            self._guard(c.pose)
            if c.op != 'moveC':
                ok = self.c.moveL(c.pose, c.speed, c.accel or self.accel)
            elif self.has_movec:
                self._guard(c.via)
                ok = self.c.moveC(c.via, c.pose, c.speed, c.accel or self.accel,
                                  c.blend, c.mode)
            elif self.blend > 0.0 and c.path:
                ok = self._fly_arc_blended(c)
            else:
                ok = self._fly_arc_stepped(c)
            if not ok:
                raise RuntimeError(f'{c.op} ({c.tag}) refused')

    def close(self):
        try:
            self.c.stopScript()
            self.c.disconnect()
            self.r.disconnect()
        except Exception:
            pass


# ======================================================================================
# T-pose back ends
# ======================================================================================
class ReplayTeePose:
    """No camera: report the T walking along the (possibly replanned) reference.

    First read is the reference start; every read after that is the waypoint
    ``--lookahead`` beyond the controller's current carrot index, so as ``step`` advances
    the carrot the reported pose closes on the goal.  A stand-in for perception that still
    drives homing, action selection and the termination test end to end.
    """

    def __init__(self, sess: rw.ArmPushSession, noise_cm=0.0, noise_deg=0.0, seed=0):
        self.sess = sess
        self.noise_cm = float(noise_cm)
        self.noise_deg = float(noise_deg)
        self.rng = np.random.default_rng(seed)
        self._started = False
        self.last_miss = None          # never misses; the attribute keeps the loop simple

    def read(self):
        ctrl = self.sess.ctrl
        ref_cm = rw.path_sim_to_real(ctrl.ref)
        if not self._started:
            self._started = True
            pose = ref_cm[0]
        else:
            j = min(ctrl.k + self.sess.args.lookahead, len(ref_cm) - 1)
            pose = ref_cm[j]
        pose = pose.copy()
        if self.noise_cm:
            pose[:2] += self.rng.normal(0.0, self.noise_cm, size=2)
        if self.noise_deg:
            pose[2] += math.radians(self.rng.normal(0.0, self.noise_deg))
        return np.array([pose[0], pose[1], _wrap(pose[2])])


class RealSenseTeePose:
    """T pose from the camera, via ``locate_functions.locate_shape``.

    Every step of the measurement is IMPORTED, not reimplemented, so this reports exactly
    what ``camera_test_id10.py`` and ``locate_functions.py`` report when pointed at the
    same table:

    * the table frame  -- ``camera_test_id10.pnp_frame`` / ``depth_frame`` on the **ID-10**
      corner marker, spun back by ``--yaw-offset`` (``aruco.origin_marker_yaw_deg``);
    * the shape markers -- ``camera_test_id10.tee_in_table``, which places one marker every
      way the hardware allows in one call.  **x, y come from depth** (the marker centre
      deprojected, which owes nothing to the marker's printed size) and **theta from the
      plane method** (two corners dropped onto the table plane, which needs no marker size
      and does not flip the way ``solvePnP`` does near fronto-parallel);
    * marker -> shape  -- ``frame_conversions.aruco_to_center`` /
      ``tee_theta_from_marker``, which take that marker's mount turn off the measured
      heading and walk the offset from its own ``TEE_MARKER_POS_CM`` position out to the
      shape's centroid -- the same origin ``push_t_demo_sim`` poses the body about.

    **The shape carries four markers, and that is the point here.**  The arm spends the run
    leaning over the shape, so with one marker the controller went blind exactly when it was
    closest to acting.  ``locate_shape`` looks for all four ids in one pass and reads the
    lowest one in view, so a covered marker costs nothing until all four are covered at
    once.  Which marker each frame came off is in ``last_obs['tee_id']``; because they all
    resolve to the same shape centre, a median taken across frames (``_read_tee``) is still
    valid when the id changes part-way through it.

    ``read()`` returns ``(x_cm, y_cm, theta_rad)`` in the table frame, the shape's own
    pose, ready for ``ArmPushSession.step_real``; ``None`` when the table marker is out of
    view, when no shape marker at all is in view, or when every visible one failed to
    measure (the reason, per id, is in ``last_miss``).

    Note there are **two dictionaries**: the ID-10 table marker is
    ``camera_test_id10.ARUCO_DICT`` (DICT_ARUCO_ORIGINAL) and the shape's markers are all
    ``TEE_DICT`` (DICT_4X4_50).  One detector for both finds neither.
    """

    def __init__(self, tee_ids=None, serial=None, width=None, height=None,
                 dict_name=None, tee_dict=None, marker_len_cm=None, yaw_offset_deg=None,
                 tee_len_cm=None, tee_height_cm=None, center_mode='centroid'):
        from camera_test_id10 import (ARUCO_DICT, MARKER_LEN_CM, RealSenseCamera, TEE_DICT,
                                      TEE_LEN_CM, YAW_OFFSET_DEG, make_detector,
                                      realsense_present)
        import locate_functions as LF

        self.LF = LF
        # Default to every id whose position on the shape has actually been measured into
        # frame_conversions.TEE_MARKER_POS_CM.  An id that is still None there would give a
        # confidently wrong pose, so it is left out rather than guessed at.
        self.tee_ids = (FC.tee_marker_ids() if tee_ids is None
                        else tuple(sorted({int(i) for i in tee_ids})))
        if not self.tee_ids:
            raise SystemExit(
                'no shape marker has a position measured -- measure each marker\'s centre '
                'from the T\'s TOP-LEFT corner (+x right, -y DOWN) into '
                'frame_conversions.TEE_MARKER_POS_CM')
        missing = [i for i in self.tee_ids if not FC.tee_marker_configured(i)]
        if missing:
            raise SystemExit(f'shape marker ids {missing} have no position measured -- fill '
                             f'in frame_conversions.TEE_MARKER_POS_CM first')
        # Resolve every position to an offset up front: a point measured off the shape (a
        # dropped minus sign, x and y swapped) raises, and it should raise here at startup
        # rather than turning into a per-id miss on every frame of the run.
        for i in self.tee_ids:
            FC.tee_marker_offset(i, mode=center_mode)
        self.marker_len_cm = MARKER_LEN_CM if marker_len_cm is None else float(marker_len_cm)
        self.yaw_offset_deg = (YAW_OFFSET_DEG if yaw_offset_deg is None
                               else float(yaw_offset_deg))
        self.tee_len_cm = TEE_LEN_CM if tee_len_cm is None else float(tee_len_cm)
        self.tee_height_cm = tee_height_cm       # None -> the shape's own thickness
        self.mode = center_mode

        # The planner was trained on one particular T, so take its proportions from that
        # mesh -- the same call locate_functions.main() makes.
        dims = LF.sim_tee_dims()
        self.bar_cm, self.stem_cm = dims if dims else (None, None)

        if not realsense_present():
            raise SystemExit('no RealSense found -- locate_shape needs depth for x, y.')
        self.cam = RealSenseCamera(serial, width or PARAMS.camera_width,
                                   height or PARAMS.camera_height)
        print(f'[cam] {self.cam.desc}')
        self.detect_table = make_detector(dict_name or ARUCO_DICT)
        self.detect_tee = make_detector(tee_dict or TEE_DICT)
        print(f'[cam] table marker id {LF.ORIGIN_ID} ({dict_name or ARUCO_DICT}, '
              f'{self.marker_len_cm:.1f} cm, yaw offset {self.yaw_offset_deg:+.1f} deg); '
              f'shape markers {list(self.tee_ids)} ({tee_dict or TEE_DICT}), read lowest '
              f'id in view first')
        if len(self.tee_ids) < 2:
            print('[cam] WARNING: only one usable shape marker, so the arm covering it is '
                  'still a lost frame -- measure the others into '
                  'frame_conversions.TEE_MARKER_POS_CM')
        self.last_miss = None
        self.last_obs = None

    def observe(self):
        """The full ``locate_shape`` dict for one frame, or ``None``."""
        obs = self.LF.locate_shape(
            self.cam, self.detect_table, self.detect_tee,
            marker_len_cm=self.marker_len_cm, yaw_offset_deg=self.yaw_offset_deg,
            tee_ids=self.tee_ids, tee_len_cm=self.tee_len_cm,
            tee_height_cm=self.tee_height_cm, bar_cm=self.bar_cm, stem_cm=self.stem_cm,
            mode=self.mode, with_sim_pose=False)
        self.last_miss = self.LF.locate_shape.last_miss
        self.last_obs = obs
        return obs

    def read(self):
        """``(x_cm, y_cm, theta_rad)`` -- the SHAPE's table-frame pose, or ``None``."""
        obs = self.observe()
        if obs is None:
            return None
        return np.array([obs['x_cm'], obs['y_cm'], _wrap(obs['theta_rad'])])

    def close(self):
        try:
            self.cam.close()
        except Exception:
            pass


# ======================================================================================
# the loop
# ======================================================================================
@dataclass
class RunResult:
    status: str                  # 'GOAL' / 'SHORT' / 'BLOCKED' / 'MAX_ACTIONS' / 'NO_TEE'
    actions: int
    dist_cm: float
    ang_deg: float = float('nan')   # |heading - goal heading|, wrapped, degrees


# How many good frames one reading of the T is the median of.  A single ArUco frame is
# noisy -- the depth return at the marker centre jitters a few millimetres and the plane
# heading a degree or two -- and the controller acts on every reading, so one bad frame is
# one bad action.  The median (not the mean) because the failure mode is an outlier: a
# corner mis-detected once puts a frame centimetres away, which drags a mean and does not
# move a median.
READ_SAMPLES = 10


def _median_pose(poses):
    """Element-wise median of ``(x, y, theta)`` readings; theta as an ANGLE.

    x and y are ordinary medians.  Headings cannot be: they live on a circle, so a set
    straddling +-pi has no meaningful sort order -- median([+179, -179]) is 0 degrees, a
    quarter turn from both.  The differences from one sample are wrapped first, the median
    of THOSE is taken, and it is put back on the reference, which is the same answer for a
    tight cluster and the right one across the branch cut.
    """
    P = np.asarray(poses, dtype=float)
    ref = float(P[0, 2])
    return np.array([float(np.median(P[:, 0])), float(np.median(P[:, 1])),
                     _wrap(ref + float(np.median(_wrap(P[:, 2] - ref))))])


def _read_tee(tracker, samples=READ_SAMPLES, attempts=20, pause=0.1):
    """One T pose: the median of up to ``samples`` good frames, or ``None`` if none came.

    Keeps reading until ``samples`` frames have landed or ``attempts`` reads have been
    made, whichever is first -- a miss costs a ``pause`` and another try, so a marker that
    is simply hidden still gives up in bounded time.  Fewer than ``samples`` good frames
    is not a failure: whatever arrived is what the median is taken over.
    """
    good = []
    for _ in range(max(int(attempts), int(samples))):
        pose = tracker.read()
        if pose is not None:
            good.append(np.asarray(pose, dtype=float))
            if len(good) >= samples:
                break
        else:
            time.sleep(pause)
    return _median_pose(good) if good else None


def viz_state_writer(path):
    """Return ``fn(tee, robot_xy, sess)`` that atomically writes the viz state file."""

    def _emit(tee, robot_xy, sess):
        state = {
            'shape': [float(v) for v in tee],
            'robot': [float(v) for v in robot_xy],
            'reference': [[float(x), float(y)]
                          for x, y in sess.stack.reference_real()[:, :2]],
        }
        tmp = f'{path}.tmp'
        with open(tmp, 'w') as f:
            json.dump(state, f)
        os.replace(tmp, path)

    return _emit


def _fanout(emits):
    """One ``emit`` from several -- the state file and the live view, or neither."""
    if not emits:
        return None
    if len(emits) == 1:
        return emits[0]
    return lambda *a, **kw: [e(*a, **kw) for e in emits]


def _transit_note(plan):
    """``  ARC r=12.4cm +margin`` -- what the transit actually flew, for the log.

    Three tags, because the cheap transit failing is invisible otherwise -- the arc looks
    exactly the same whether or not a slide was ever an option:

    * ``TRANS``  the same-face slide: lift, translate along the face, drop.  No circle.
    * ``ARC``    round the object, because the two contacts are not on one face.
    * ``FAILED TRANS`` an ARC that a ``TRANS`` was available for -- both ends stood off
      the SAME face and the slide was still rejected (it would have run inside the
      transit keep-out, past a corner into the T's own arm, or into a wall).  The move is
      correct, just the long way round; a run full of these means the slide is being
      thrown away on geometry, not on the contacts it was handed.

    The radius alone does not say whether ``ARC_CLEARANCE_CM`` did anything: the ladder
    stops at the tightest rung that comes back clear, then tries ONE circle that much
    wider and silently keeps the tight one when the wider is blocked.  ``+margin`` means
    the wider circle was flown, ``tight`` means it was rejected and the arc is riding the
    bare rung.  A ``TRANS`` never consults the margin at all.
    """
    if plan.transit_kind is None:
        return ''
    if plan.transit_kind == 'edge':
        return '  TRANS'
    note = '  FAILED TRANS -> ARC' if plan.edge_blocked else '  ARC'
    if plan.transit_radius is not None:
        note += f' r={plan.transit_radius / rw.sim_units_per_cm():.1f}cm'
        note += ' +margin' if plan.transit_widened else ' tight'
    return note


def _push_note(plan):
    """``  push 0.7cm`` -- the stroke actually flown, which the status string does not say.

    ``plan.status`` carries the ``L=`` the primitive was SELECTED at, built inside
    ``begin_primitive`` before ``ArmPushSession`` shortens the endgame push (see
    ``rw.ENDGAME_PUSH_SCALE``), so near the goal the two disagree on purpose.
    """
    if plan.push_length is None:
        return ''
    return f'  push {plan.push_length / rw.sim_units_per_cm():.1f}cm'


def run_pipeline(sess: rw.ArmPushSession, arm, tracker, *, settle_s=0.6,
                 max_actions=250, goal_cm=2.0, goal_deg=DEF_GOAL_DEG, accel=DEF_ACCEL,
                 transit_accel=None, emit=None, verbose=True, park_corner=True):
    """Park at the corner, home, then step until close enough / blocked / out of budget.

    The home move is an ARC, not a straight line: it retracts to the widest stand-off
    circle -- the one that encloses the T plus the pusher disc plus ``--standoff`` -- and
    rides it round to the home point, the same fly-around every transit in the step loop
    uses.  Driving straight there from the corner park would cut a chord across the table
    that runs through the shape as readily as around it.

    "Close enough" is a POSE test, not a position one: the T's centroid within
    ``goal_cm`` of the goal centroid AND its heading within ``goal_deg`` of the goal
    heading.  Both come from the same table-frame goal ``(x, y, theta)``.  When the
    controller runs out of primitives before that test passes the run ends ``SHORT``
    rather than ``GOAL`` -- there is nothing left to execute, but the T is not where the
    plan wanted it.

    Returns a RunResult.  A reading the camera cannot make ends the run with ``NO_TEE`` --
    the arm is never moved to go looking for the marker.

    ``accel`` is the acceleration of the PUSH leg; ``transit_accel`` (default: ``accel``)
    is the acceleration of everything else -- park, home, retract, arc, re-entry.

    ``park_corner`` runs the ID-20 corner park FIRST, before the T is ever read: it is a
    fixed point in the table frame, needs no observation, and getting the arm out to a
    corner is the one move that can only help the camera see the shape.
    """
    goal = sess.stack.reference_real()[-1]                 # planned goal, table cm
    v_transit = sess.robot.speed_mps(sess.args.transit_speed or sess.args.pusher_speed)
    # The park, the home move and every transit leg run on this; only the push itself
    # keeps ``accel``.  See --transit-accel: the transit legs are far too short to ever
    # reach ``v_transit`` at the push's acceleration, so raising the speed cap alone
    # changes nothing on them.
    a_transit = float(accel if transit_accel is None else transit_accel)

    if park_corner:
        park = corner_park_cm()
        if verbose:
            print(f'[run] park: {CORNER_INSET_CM:.0f} cm in from the ID-20 corner '
                  f'({CORNER_ID20_CM[0]:.1f}, {CORNER_ID20_CM[1]:.1f}) toward the field '
                  f'centre ({FC.FIELD_CENTER_CM[0]:.1f}, {FC.FIELD_CENTER_CM[1]:.1f}) '
                  f'-> ({park[0]:.1f}, {park[1]:.1f}) cm, '
                  f'up {CORNER_LIFT_CM:.0f} cm / across / down {CORNER_LIFT_CM:.0f} cm')
        landed = move_via_lift(arm, sess.robot, park, v_transit, a_transit)
        if verbose and float(np.hypot(*(np.asarray(landed) - park))) > 0.1:
            print(f'[run] park HELD to ({landed[0]:.1f}, {landed[1]:.1f}) cm by the field '
                  f'/ workspace clamp')

    tee = _read_tee(tracker)
    if tee is None:
        return RunResult('NO_TEE', 0, float('nan'))

    # The home move is FLOWN, not driven straight at: out to the widest stand-off circle
    # and round it (``ArmPushSession.home_move_real``).  A plain moveL from the park to
    # the home point is a chord across the table, and the T is on it as often as not --
    # which shoves the shape out from under the plan that was just made from its pose.
    home_plan = sess.home_move_real(tee, arm.tcp_xy_cm(), robot=sess.robot,
                                    accel=a_transit)
    home = np.asarray(home_plan.waypoints_real[-1], dtype=float)
    if verbose:
        print(f'[run] T=({tee[0]:.1f},{tee[1]:.1f},{math.degrees(tee[2]):+.0f}deg)  '
              f'home the end effector -> ({home[0]:.1f},{home[1]:.1f}) cm'
              f'  ARC r={home_plan.transit_radius / rw.sim_units_per_cm():.1f}cm widest')
    if home_plan.robot_move is None:                  # no RobotFrame: nothing to fly
        arm.move_to_cm(home, v_transit, a_transit)
    else:
        arm.run(home_plan.robot_move)
    if emit:
        emit(tee, arm.tcp_xy_cm(), sess)

    dgoal = dang = float('nan')
    for i in range(max_actions):
        tee = _read_tee(tracker)
        if tee is None:
            if verbose:
                miss = getattr(tracker, 'last_miss', None) or 'no reading'
                print(f'[run] #{i:03d}  no T ({miss})')
            return RunResult('NO_TEE', i, dgoal, dang)
        tcp = np.asarray(arm.tcp_xy_cm(), dtype=float)
        dgoal = float(np.hypot(*(tee[:2] - goal[:2])))
        dang = abs(math.degrees(_wrap(float(tee[2]) - float(goal[2]))))

        plan = sess.step_real(tee, tcp, robot=sess.robot, accel=accel,
                              transit_accel=a_transit)
        if verbose:
            print(f'[run] #{i:03d}  T=({tee[0]:6.1f},{tee[1]:6.1f},'
                  f'{math.degrees(tee[2]):+4.0f})  |goal|={dgoal:5.1f}cm/'
                  f'{dang:4.1f}deg  {plan.status}'
                  f'{_push_note(plan)}{_transit_note(plan)}')
        if emit:
            emit(tee, tcp, sess)

        if dgoal <= goal_cm and dang <= goal_deg:
            return RunResult('GOAL', i, dgoal, dang)
        if plan.done:               # the controller is finished, the pose test is not
            return RunResult('SHORT', i, dgoal, dang)
        if plan.robot_move is None:                        # BLOCKED (or GOAL w/o a move)
            return RunResult('BLOCKED', i, dgoal, dang)

        arm.run(plan.robot_move)
        time.sleep(settle_s)

    return RunResult('MAX_ACTIONS', max_actions, dgoal, dang)


# ======================================================================================
# CLI
# ======================================================================================
def build_cli():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # Defaults come from real_world_params.json; any flag below still overrides it.
    ap.add_argument('--dry-run', action='store_true',
                    help='no hardware: DryRunArm + ReplayTeePose')
    ap.add_argument('--ip', default=PARAMS.robot_ip, help='UR robot IP')
    # reference
    ap.add_argument('--dry-run-start', type=float, nargs=3, default=(20.0, 32.5, 0.0),
                    metavar=('X_CM', 'Y_CM', 'DEG'),
                    help='--dry-run only: the T pose to plan FROM, standing in for the '
                         'camera reading (with hardware the shape is read before the '
                         'reference is planned).  Table cm + degrees')
    ap.add_argument('--ref', default=None,
                    help='replay a saved world-frame (T,3) path instead of planning '
                         '(no torch / GPU needed)')
    ap.add_argument('--plan-device', default='cuda')
    ap.add_argument('--no-replan', action='store_true')
    # --- the push length ladder -------------------------------------------------------
    # These three are calibration inputs, not run-time knobs: build_stack turns them into
    # push_length_ladder(push_len, push_len_min, push_len_steps) and prims.calibrate
    # MEASURES every rung -- one rollout per primitive per rung, 1000 rollouts a rung at
    # --n-contacts 100 x --n-dirs 10, cached to .push_primitives_*.npy beside the dataset.
    # A rung the cache has not seen is paid for once, at startup.
    ap.add_argument('--push-len', type=float, default=None, metavar='U',
                    help=f'length of a full push, SIM UNITS ({rw.sim_units_per_cm():.0f} '
                         f'= 1 cm; default {rw._DEFAULTS["push_len"]:.0f} = '
                         f'{rw._DEFAULTS["push_len"] / rw.sim_units_per_cm():.1f} cm)')
    ap.add_argument('--push-len-min', type=float, default=None, metavar='U',
                    help='SHORT end of the ladder, sim units.  ON ITS OWN THIS DOES '
                         'NOTHING: push_length_ladder collapses to [--push-len] whenever '
                         '--push-len-steps is 1 (the default) or this is >= --push-len, '
                         'so it must be raised together with --push-len-steps')
    ap.add_argument('--endgame-scale', type=float, default=None, metavar='F',
                    help=f'near the goal, cut the push to this fraction of the length it '
                         f'was selected at (default {rw.ENDGAME_PUSH_SCALE:.3f}; '
                         f'"near" = within {rw.ENDGAME_DIST_CM:.0f} cm AND '
                         f'{rw.ENDGAME_DEG:.0f} deg of the goal).  Pass 1.0 to switch it '
                         f'off and give every action the same push length -- which is '
                         f'what you want with a one-rung ladder you have already sized')
    ap.add_argument('--push-len-steps', type=int, default=None, metavar='N',
                    help='rungs on the ladder, geometrically spaced from --push-len-min '
                         'to --push-len; 1 (the default) disables the taper entirely.  '
                         'With N > 1 prims.select picks the LENGTH as well as the contact, '
                         'so the push shortens on its own as the T closes on the goal -- '
                         'the mechanism rw.ENDGAME_PUSH_SCALE stands in for when there is '
                         'only one rung.  Costs one calibration pass per new rung')
    # table -> UR base frame: anchored on p0, the TCP pose arm_calibrate.py records with
    # the end effector on the FIELD CENTRE (sim 0,0) -- not the ID-10 marker, which is
    # where the camera frame starts, 39.8 / 32.5 cm away.  frame_conversions owns the hop.
    ap.add_argument('--startpos', default=None,
                    help='arm_calibrate.py output holding the p0 point '
                         '(default: real_world_params.json robot.startpos_json)')
    ap.add_argument('--z-push', type=float, default=None,
                    help='override the tool-tip height; default keeps p0 recorded z')
    ap.add_argument('--no-clamp', action='store_true',
                    help='do not hold waypoints inside the field / workspace box')
    # The camera: every default is camera_test_id10's, so this tracker and
    # `python camera_test_id10.py` / `python locate_functions.py` see the same table.
    ap.add_argument('--tee-marker-ids', type=int, nargs='+', default=None,
                    help='ArUco ids of the markers mounted on the T, read in increasing '
                         'order until one measures cleanly (default: every id with a '
                         'position in frame_conversions.TEE_MARKER_POS_CM)')
    ap.add_argument('--serial', default=None, help='RealSense serial (optional)')
    ap.add_argument('--width', type=int, default=PARAMS.camera_width)
    ap.add_argument('--height', type=int, default=PARAMS.camera_height)
    ap.add_argument('--dict', default=None,
                    help='dictionary of the ID-10 table marker '
                         '(default: camera_test_id10.ARUCO_DICT)')
    ap.add_argument('--tee-dict', default=None,
                    help='dictionary of the T marker (default: camera_test_id10.TEE_DICT)')
    ap.add_argument('--marker-len', type=float, default=None, metavar='CM',
                    help="side of the ID-10 marker -- the table frame's metric scale")
    ap.add_argument('--yaw-offset', type=float, default=None, metavar='DEG',
                    help="heading of the table marker's own +x edge, deg CCW from table +x")
    ap.add_argument('--tee-len', type=float, default=None, metavar='CM',
                    help='side of the T marker (the pnp cross-check only)')
    ap.add_argument('--tee-height', type=float, default=None, metavar='CM',
                    help='how far the T marker rides above the table; default: the '
                         "shape's own thickness")
    ap.add_argument('--center', default='centroid', choices=('centroid', 'bbox'),
                    help="what 'the centre' means; centroid is what the sim poses about")
    # loop
    ap.add_argument('--settle', type=float, default=0.01,
                    help='seconds to wait after each action before re-observing')
    ap.add_argument('--max-actions', type=int, default=250)
    ap.add_argument('--goal-cm', type=float, default=1.0,
                    help='stop once the T centroid is within this of the planned goal '
                         '(with --goal-deg: BOTH have to pass)')
    ap.add_argument('--goal-deg', type=float, default=DEF_GOAL_DEG, metavar='DEG',
                    help='heading half of the same test: the T must also be turned to '
                         'within this many degrees of the goal heading.  A run that '
                         'passes one and not the other keeps going, and ends SHORT if '
                         'the controller gives up first')
    ap.add_argument('--accel', type=float, default=DEF_ACCEL,
                    help='acceleration of the PUSH leg, m/s^2 (--transit-accel covers '
                         'everything else)')
    ap.add_argument('--transit-accel', type=float, default=None, metavar='A',
                    help='acceleration for the park, the home move and every transit leg '
                         '(retract / arc / re-entry), m/s^2.  Defaults to --accel.  THIS '
                         'is the knob that makes relocating faster: a transit leg is only '
                         f'a few cm long, so at the default {DEF_ACCEL} m/s^2 it tops out '
                         'well under --transit-speed and never sees the cap at all.  Try '
                         '1.0')
    ap.add_argument('--transit-speed', type=float, default=None, metavar='U',
                    help='speed cap for the park, the home move and every transit leg, in '
                         f'SIM UNITS/s ({rw.sim_units_per_cm():.0f} = 1 cm/s; default '
                         f'{rw._DEFAULTS["transit_speed"]:.0f} = '
                         f'{rw._DEFAULTS["transit_speed"] / rw.sim_units_per_cm():.0f} '
                         'cm/s).  Only binds on the long legs -- raise --transit-accel '
                         'with it or the short ones will not move any faster')
    ap.add_argument('--no_trans_collision', action='store_true',
                    help='fly the same-face SLIDE (the TRANS transit) without collision-'
                         'checking it: whenever the pusher and the next hand-over point '
                         'stand off one face of the T, lift off that face, translate along '
                         'it and drop back down, full stop.  Without this the slide is '
                         'held to the transit keep-out like any other leg and a blocked '
                         'one falls back to the arc round the whole object -- the '
                         '"FAILED TRANS -> ARC" in the log.  The lift is unchanged; only '
                         'the veto goes away, so a slide that runs past a corner into the '
                         "T's own arm or into a wall is now flown")
    ap.add_argument('--no-park', action='store_true',
                    help=f'skip the corner park: normally the run starts by going '
                         f'{CORNER_INSET_CM:.0f} cm in from the ID-20 table corner toward '
                         f'the field centre -- up {CORNER_LIFT_CM:.0f} cm, straight '
                         f'across, down {CORNER_LIFT_CM:.0f} cm -- before homing')
    ap.add_argument('--no-movec', action='store_true',
                    help='never call moveC, even if this ur_rtde has it: fly every arc as '
                         'a moveL chain over its arc samples (same curve, one stop per '
                         'sample).  Builds without moveC fall back to this on their own')
    ap.add_argument('--blend', type=float, default=DEF_BLEND, metavar='M',
                    help='blend radius in METRES for the around-the-object ARC ONLY, and '
                         'only when there is no moveC to fly it with (--no-movec, or a '
                         'ur_rtde without it): the arc samples go out as one blended '
                         'moveL path -- corners rounded, no stop between samples -- '
                         'instead of a chain of blocking moveLs.  moveC comes first where '
                         'it exists, being the true circle with no corners to round at '
                         'all.  Clamped per corner to just under half the shorter of its '
                         'two legs, as the controller requires, so a tight arc blends '
                         'less than a wide one.  The retract, re-entry, push, edge slides '
                         'and corner park are untouched.  Try 0.01 (1 cm)')
    ap.add_argument('--viz-state', default=None,
                    help='write {shape, robot, reference} here each iteration for '
                         'push_t_realworld_viz.py --state')
    ap.add_argument('--viz', action='store_true',
                    help='open push_t_realworld_viz.TableView in THIS process: the env, '
                         'the T, the goal ghost, the planned path and the end effector, '
                         'updated from the same observations the controller acts on')
    ap.add_argument('--viz-port', type=int, default=8080, help='--viz viser port')
    return ap.parse_args()


def main():
    cli = build_cli()

    over = {k: v for k, v in (('transit_speed', cli.transit_speed),
                              ('push_len', cli.push_len),
                              ('push_len_min', cli.push_len_min),
                              ('push_len_steps', cli.push_len_steps),
                              ('endgame_scale', cli.endgame_scale))
            if v is not None}
    args = rw.make_args(plan_device=cli.plan_device,
                        no_replan=cli.no_replan or bool(cli.ref),
                        no_trans_collision=cli.no_trans_collision, **over)
    ref = np.load(cli.ref) if cli.ref else None

    # The reference runs from WHERE THE T IS to rw.GOAL_POSE_NORM, so with hardware the
    # camera comes up first and its reading is the plan's start.  (--dry-run has no
    # camera -- and its ReplayTeePose needs the session that needs the plan -- so there
    # the start is --dry-run-start.)
    tracker = None
    if cli.dry_run:
        sx, sy, sdeg = cli.dry_run_start
        start = np.array([sx, sy, math.radians(sdeg)])
    else:
        tracker = RealSenseTeePose(tee_ids=cli.tee_marker_ids, serial=cli.serial,
                                   width=cli.width, height=cli.height,
                                   dict_name=cli.dict, tee_dict=cli.tee_dict,
                                   marker_len_cm=cli.marker_len,
                                   yaw_offset_deg=cli.yaw_offset,
                                   tee_len_cm=cli.tee_len, tee_height_cm=cli.tee_height,
                                   center_mode=cli.center)
        start = _read_tee(tracker)
        if start is None:
            miss = tracker.last_miss or 'no reading'
            tracker.close()
            raise SystemExit(f'[run] cannot see the shape ({miss}) -- cannot plan '
                             'without its pose')
    try:                                   # the camera is already streaming by now
        stack = rw.build_stack(args, ref=ref, start_pose_real=start)
        robot = rw.RobotFrame.from_startpos(cli.startpos, z_push_m=cli.z_push,
                                            limit=None if cli.no_clamp else FC.NORM_HALF,
                                            box={} if cli.no_clamp else None)
    except BaseException:
        if tracker is not None:
            tracker.close()
        raise
    sess = rw.ArmPushSession(stack, robot_frame=robot)
    o = robot.origin_xyz_m
    g, gr = rw.goal_pose_norm(), stack.goal_real()
    print(f'[run] reference {len(stack.ref)} waypoints; stand-off radius '
          f'{sess.ctrl.standoff / rw.sim_units_per_cm():.1f} cm; '
          f'replan {"OFF" if sess.rep is None else "per action"}')
    if cli.no_trans_collision:
        print('[run] --no_trans_collision: a same-face slide is flown as soon as it is '
              'available, unchecked -- no FAILED TRANS -> ARC fallback')
    print(f'[run] goal (GOAL_POSE_NORM) norm ({g[0]:+.3f}, {g[1]:+.3f}, '
          f'{g[5]:+.3f} turns)'
          + ('' if gr is None else f' == ({gr[0]:.1f}, {gr[1]:.1f}, '
                                   f'{math.degrees(gr[2]):+.0f}deg) cm'))
    print(f'[run] p0 anchor: the FIELD CENTRE at base ({o[0]:+.3f}, {o[1]:+.3f}, '
          f'{o[2]:+.3f}) m; table ({FC.FIELD_CENTER_CM[0]:.1f}, {FC.FIELD_CENTER_CM[1]:.1f}) '
          f'cm -> that same point, {FC.FIELD_CM:.0f} cm across the field')

    if cli.dry_run:
        tracker = ReplayTeePose(sess)
        arm = DryRunArm(robot, start_xy_cm=sess.stack.reference_real()[0, :2],
                        blend=cli.blend)
    else:
        arm = RtdeArm(cli.ip, robot, accel=cli.accel, use_movec=not cli.no_movec,
                      blend=cli.blend)

    emits = []
    if cli.viz_state:
        emits.append(viz_state_writer(cli.viz_state))
    if cli.viz:
        from push_t_realworld_viz import TableView, view_emitter

        # The goal ghost is the plan's own end: GOAL_POSE_NORM normally, and with a raw
        # --ref (which carries no goal) the last waypoint of that path.
        goal = stack.goal_real()
        view = TableView(rw.load_geometry(args), port=cli.viz_port,
                         goal=stack.reference_real()[-1] if goal is None else goal,
                         reference=stack.reference_real()[:, :2])
        print(f'[run] viser on http://localhost:{cli.viz_port}')
        emits.append(view_emitter(view))
    emit = _fanout(emits)
    try:
        res = run_pipeline(sess, arm, tracker, settle_s=cli.settle,
                           max_actions=cli.max_actions, goal_cm=cli.goal_cm,
                           goal_deg=cli.goal_deg, accel=cli.accel,
                           transit_accel=cli.transit_accel, emit=emit,
                           park_corner=not cli.no_park)
    finally:
        arm.close()
        if hasattr(tracker, 'close'):
            tracker.close()

    print(f'\n[run] {res.status} after {res.actions} actions; '
          f'final |goal| = {res.dist_cm:.1f} cm / {res.ang_deg:.1f} deg '
          f'(tolerance {cli.goal_cm:.1f} cm / {cli.goal_deg:.1f} deg)')
    if cli.viz:          # the run is over in a blink under --dry-run; hold the scene up
        print('[run] viser still serving the final state -- ctrl-c to quit')
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass
    raise SystemExit(0 if res.status == 'GOAL' else 1)


if __name__ == '__main__':
    main()
