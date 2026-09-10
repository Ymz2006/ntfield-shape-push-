"""arm_test_id10 -- click a point on the ID-10 table, the arm goes there.

The one thing this program does: turn a click in the table scene into a single ``moveL``
of the physical end effector.  It is the smallest possible end-to-end check that
``calibration/arm/startpos.json`` is right -- click the field centre, the pusher should
sit on it; click a field corner, it should sit on that corner.

The view is ``camera_test_id10``'s frame, drawn the way ``locate_functions.visualize``
draws it: **table centimetres**, origin at the ID-10 corner marker, ``+x`` along the
length, ``+y`` along the width, and the same three-colour separation, because mistaking
one of them for another is the alignment bug that is hardest to see --

* **grey**  the nominal ``--length`` x ``--width`` table rectangle, carried out from the
  one ID-10 marker.  Drawn, not measured: with a single marker nothing observes the far
  corners.
* **green** the normalization box -- the field ``real_world_params.json`` maps onto
  ``[-0.5, 0.5]^2``.  **This is what you click on**; a click outside it is held at its
  edge rather than flung at the table.
* **dark**  the real env geometry -- the walls and blocks ``push_t_demo_sim`` collides
  against, when the sim stack is importable; the box alone if it is not.

Each frame in play gets an RGB triad and a label (red ``+x``, green ``+y``, blue ``+z``
out of the table): the ID-10 marker, table ``(0, 0)``, where the camera frame starts, and
the field centre, which is sim ``(0, 0)`` and the arm's p0.  Two different zeros a third
of a metre apart, drawn so they cannot be confused for each other.

The click path
--------------
::

    click -> ray/table plane -> table cm -> FC.table_to_sim -> sim_to_robot -> moveL

**The anchor is p0**, and p0 is the **field's** ``(0, 0)``: the TCP pose
``arm_calibrate.py`` records with the end effector placed physically on the centre of the
green box -- ``offset_to_origin_cm``, 39.8 / 32.5 cm from the ID-10 marker.  Not the
marker itself: the marker is where the *camera* frame starts, and the two are a third of a
metre apart, which is why both are drawn.  p0's ``x, y`` are that centre, its ``z`` is the
height every target is held at, and its rotation vector is the tool orientation reproduced
at every target (a cylindrical pusher is rotationally symmetric, so the wrist never turns).

**The conversion is a scale, a flip and a shift** (``frame_conversions.sim_to_robot``,
which this module imports rather than owning -- every frame hop in the rig lives there)::

    base_xy = p0_xy + SIM_TO_BASE * sim_xy * FIELD_CM * 0.01        # cm -> m

``SIM_TO_BASE`` is ``(-1, -1)``: **the base axes run opposite the sim's**, so sim ``+x`` is
base ``-x`` and sim ``+y`` is base ``-y``.  That is a half-turn about p0 rather than a
mirror -- the frame stays right-handed, and sim ``(0, 0)`` is still p0, because a rotation
fixes its own centre.  The multiplier is the **whole** field, 70 cm
(``real_world_params.json`` -> ``environment.env_len_cm``): the normalized square runs
``-0.5 .. +0.5``, so half the field has to land on ``0.5``, and ``35 / 70 = 0.5``.  A
corner is therefore 35 cm out along each base axis from p0.

Table centimetres reach sim through ``frame_conversions`` -- centre on the field, turn
180 deg, divide by 70 -- so this module owns sim <-> base and nothing else, and the two
calibrations stay separable when one of them is wrong.  The two 180s cancel, which is
worth knowing at the pendant: **table** ``+x`` **is base** ``+x``, so table to base is a
pure translation by p0, and dragging right on the screen moves the arm along base ``+x``.

**The tool never changes height.**  ``z`` is copied from p0 into every target and is never
an argument, a slider or a clamp -- there is no lift, no retract and no approach leg in
this program.  Because a ``moveL`` from the arm's current pose to a p0-height target would
itself be a vertical move if the arm is not already at that height, startup *refuses* to
run when the live TCP is more than ``--z-tol`` off p0's z: jog it back down with
``arm_calibrate.py`` first.

Frames
------
table      the ID-10 frame, centimetres -- what the scene is drawn in and clicked in.
sim        the planner's normalized square, ``[-0.5, +0.5]^2``, dimensionless.
base       UR base frame ``(x, y, z, rx, ry, rz)``, metres and rotation-vector radians.

``frame_conversions`` owns table <-> sim; this module owns sim <-> base.

Usage (inside the Docker image, from the repo root -- ``--network=host`` for RTDE):

    python arm_test_id10.py --dry-run        # no robot at all; click, watch it print
    python arm_test_id10.py                  # live, viser at http://localhost:8080
    python arm_test_id10.py --no-env         # skip the sim stack; box and table only
"""

from __future__ import annotations

import argparse
import math
import os
import threading
import time

import numpy as np
import viser

import frame_conversions as FC
from frame_conversions import (
    FIELD_CM,
    NORM_HALF,
    ORIGIN_ID,
    SIM_TO_BASE,
    clamp_sim,
    load_origin_pose,
    robot_to_sim,
    sim_to_robot,
)
from real_world_params import PARAMS

# camera_test_id10's nominal table, cm -- drawn out from the one marker, not measured.
LENGTH_CM = 79.0
WIDTH_CM = 63.0
MARKER_LEN_CM = 5.5

HERE = os.path.dirname(os.path.abspath(__file__))
SIM_ENV = os.path.join(HERE, "datasets", "3dshape", "2denv4.obj")
SIM_SHAPE = os.path.join(HERE, "datasets", "3dshape", "Tshape3d.obj")
SIM_SHAPE_ZUP = os.path.join(HERE, "datasets", "3dshape", "Tshape3d_zup.obj")

# The scene's palette, straight from locate_functions so the two views read as one world.
COL_ENV = (110, 118, 132)            # the obstacle geometry
COL_FLOOR = (232, 232, 228)          # the table top
COL_MARKER = (60, 200, 90)           # the ID-10 marker square
COL_SIM_ENV = (150, 210, 150)        # the normalization box -- the field
COL_ENV_EDGE = (150, 150, 175)
COL_TABLE = (150, 150, 165)          # the nominal table rectangle
COL_TARGET = (232, 138, 62)          # where the last click sent the arm
COL_ROBOT = (90, 200, 230)           # the live end effector

PUSHER_R_CM = 2.0                    # drawn radius of the pusher disc
ROBOT_MODE_RUNNING = 7

# ======================================================================================
# workspace bounds
# ======================================================================================
def clamp_base_xy(pose, box):
    """Hold a target inside the workspace box **in x and y only**; ``(pose, clamped)``.

    ``z`` is not in the box check on purpose: it is p0's height, it is never commanded to
    anything else, and clamping it would be the one way this program could move the tool
    vertically.  A p0 outside the box's z range is reported at startup instead.
    """
    pose = np.asarray(pose, dtype=float).copy()
    lo = np.array([box["x"][0], box["y"][0]])
    hi = np.array([box["x"][1], box["y"][1]])
    held = np.clip(pose[:2], lo, hi)
    was = bool(np.any(np.abs(held - pose[:2]) > 1e-9))
    pose[:2] = held
    return pose, was


# ======================================================================================
# arm back ends
# ======================================================================================
class DryRunArm:
    """No robot: print each move and track a virtual TCP that starts at p0."""

    live = False

    def __init__(self, origin_pose):
        self._pose = np.asarray(origin_pose, dtype=float).copy()

    def tcp_pose(self):
        return self._pose.copy()

    def move_to(self, pose, speed, accel):
        self._pose = np.asarray(pose, dtype=float).copy()
        print(f"  [arm] moveL -> ({pose[0]:+.4f}, {pose[1]:+.4f}, {pose[2]:+.4f}) m "
              f"@ {speed:.3f} m/s   (dry run)")
        return True, ""

    def close(self):
        pass


class RtdeArm:
    """UR arm over ``ur_rtde``: blocking ``moveL``, safety-checked before it is sent."""

    live = True

    def __init__(self, ip):
        from rtde_receive import RTDEReceiveInterface

        print(f"[arm] connecting to {ip} ...")
        self.r = RTDEReceiveInterface(ip)
        mode = self.r.getRobotMode()
        if mode != ROBOT_MODE_RUNNING:
            raise SystemExit(
                f"robot is in mode {mode}, not RUNNING ({ROBOT_MODE_RUNNING}).\n"
                "Power on and release the brakes on the pendant (Remote Control if the "
                "control interface refuses to connect), or re-run with --dry-run.")

        from rtde_control import RTDEControlInterface

        self.c = RTDEControlInterface(ip)
        print(f"[arm] connected; TCP offset {np.round(self.c.getTCPOffset(), 5).tolist()}")

    def tcp_pose(self):
        return np.array(self.r.getActualTCPPose(), dtype=float)

    def move_to(self, pose, speed, accel):
        target = [float(v) for v in pose]
        if not self.c.isPoseWithinSafetyLimits(target):
            return False, "REFUSED: outside the robot's safety limits"
        if not self.c.moveL(target, speed, accel):
            return False, "REFUSED: moveL failed"
        return True, ""

    def close(self):
        for shut in (lambda: self.c.stopScript(), lambda: self.c.disconnect(),
                     lambda: self.r.disconnect()):
            try:
                shut()
            except Exception:
                pass


# ======================================================================================
# the scene's geometry
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


def _cross(centre, size, z=0.0):
    cx, cy = float(centre[0]), float(centre[1])
    return np.array([[[cx - size, cy, z], [cx + size, cy, z]],
                     [[cx, cy - size, z], [cx, cy + size, z]]], dtype=np.float32)


def _disc(radius, n=41):
    a = np.linspace(0.0, 2.0 * math.pi, n)
    return np.column_stack([radius * np.cos(a), radius * np.sin(a)])


def table_rect_cm(length_cm=LENGTH_CM, width_cm=WIDTH_CM):
    """The nominal table rectangle carried out from the ID-10 marker, ``(4, 2)`` cm.

    ``camera_test_id10``'s rectangle: the marker is one corner, so the table occupies
    ``x >= 0, y >= 0``.  Drawn, never measured -- one marker observes no far corner.
    """
    return np.array([(0.0, 0.0), (length_cm, 0.0),
                     (length_cm, width_cm), (0.0, width_cm)], dtype=float)


def load_env(env_obj=SIM_ENV, shape_obj=SIM_SHAPE, shape_zup=SIM_SHAPE_ZUP):
    """The real sim environment in table centimetres -- ``dict`` or ``None``.

    Exactly ``locate_functions.load_env``: the footprint of ``2denv4.obj`` *is* the
    obstacle set ``push_t_demo_sim`` collides against, so drawing it is drawing the
    obstacles.  Imported lazily and never fatal -- an arm test must still run on a box
    where the sim stack (pymunk / trimesh / shapely) is missing; the caller falls back to
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
        return {"rings": rings,
                "vertices": np.column_stack([xy, np.zeros(len(xy))]).astype(np.float32),
                "faces": np.asarray(geo.env_mesh.faces, dtype=np.uint32),
                "n_polys": len(geo.env_polys),
                "n_holes": sum(len(p.interiors) for p in geo.env_polys)}
    except Exception as exc:                              # noqa: BLE001 -- optional
        load_env.last_error = f"{type(exc).__name__}: {exc}"
        return None


load_env.last_error = None


def ray_to_plane(origin, direction, z=0.0):
    """Where a click ray meets the table plane, or ``None`` if it never does.

    viser hands a scene click back as a ray in world coordinates; the table is the flat
    ``z = 0`` plane, so one intersection is the whole of "which point was clicked".
    """
    o = np.asarray(origin, dtype=float)
    d = np.asarray(direction, dtype=float)
    if abs(d[2]) < 1e-9:
        return None
    t = (z - o[2]) / d[2]
    if t <= 0.0:
        return None
    return (o + t * d)[:2]


# ======================================================================================
# the view
# ======================================================================================
class TableClickView:
    """``locate_functions``' table scene, with a click wired to one ``moveL``.

    Scene units are table centimetres, so what is drawn and what is clicked need no
    conversion between them; the click is converted once, on its way to the arm.  Moves
    run on the callback thread under a lock: ``moveL`` blocks until the arm arrives, and a
    click that lands while one is in flight is dropped rather than queued, so an impatient
    double-click can never turn into a backlog of stale motion.
    """

    def __init__(self, arm, origin_pose, args):
        self.arm = arm
        self.origin_pose = np.asarray(origin_pose, dtype=float)
        self.args = args
        # p0 is the field's (0, 0) -- the green box's centre, not the ID-10 marker.
        self.p0_cm = np.asarray(FC.FIELD_CENTER_CM, dtype=float)
        self.box = PARAMS.workspace_box_m
        self.lock = threading.Lock()
        self.note = ""
        self.target_sim = np.zeros(2)

        self.env = None if args.no_env else load_env()
        self.server = viser.ViserServer(port=args.port)
        self.server.scene.set_up_direction("+z")
        self.server.scene.world_axes.visible = False
        self._scene()
        self._gui()
        self._clicks()

    # -- scene --------------------------------------------------------------------
    def _origin_axes(self, name, xy_cm, length_cm, label, z_cm=0.0):
        """An RGB axes triad at a table point, with a floating label above it.

        One helper for every origin so they are drawn identically and cannot drift apart
        in size or convention: red ``+x``, green ``+y``, blue ``+z`` out of the table.
        """
        node = self.server.scene.add_frame(
            name, show_axes=True, axes_length=length_cm, axes_radius=length_cm / 60.0,
            origin_radius=length_cm / 20.0,
            position=(float(xy_cm[0]), float(xy_cm[1]), float(z_cm)))
        try:
            self.server.scene.add_label(f"{name}/label", label,
                                        position=(0.0, 0.0, length_cm * 0.45))
        except Exception:                     # labels are decoration, never fail over one
            pass
        return node

    def _scene(self):
        s = self.server.scene
        field = np.asarray(FC.field_corners_cm(), dtype=float)
        table = table_rect_cm(self.args.length, self.args.width)

        # ---- the environment, in three colours (see the module docstring) ----
        if self.env is not None:
            s.add_mesh_simple("/env/fill", self.env["vertices"], self.env["faces"],
                              color=COL_ENV, flat_shading=True)
            for i, ring in enumerate(self.env["rings"]):
                s.add_line_segments(f"/env/ring{i}", _loop_segments(ring, 0.01),
                                    colors=COL_ENV_EDGE, thickness=3.0)
        s.add_line_segments("/env/norm_box", _loop_segments(field, 0.03),
                            colors=COL_SIM_ENV, thickness=2.0)
        s.add_line_segments("/env/table", _loop_segments(table), colors=COL_TABLE,
                            thickness=4.0)
        s.add_line_segments(f"/env/marker{ORIGIN_ID}",
                            _square((0.0, 0.0), MARKER_LEN_CM), colors=COL_MARKER,
                            thickness=2.5)

        # Framing: the bounds of everything drawn, padded by 1.15, with the floor and the
        # grid filling that square.  The floor's top face is z = 0 -- the plane clicks are
        # intersected with -- so the thing you see is the thing you point at.
        pts = [field, table]
        if self.env is not None and self.env["rings"]:
            pts.append(np.vstack(self.env["rings"]))
        allpts = np.vstack(pts)
        lo, hi = allpts.min(axis=0), allpts.max(axis=0)
        self.cx, self.cy = float((lo[0] + hi[0]) / 2.0), float((lo[1] + hi[1]) / 2.0)
        self.span = float(max(hi - lo) * 1.15)
        s.add_box("/floor", color=COL_FLOOR, dimensions=(self.span, self.span, 2.0),
                  position=(self.cx, self.cy, -1.0))
        s.add_grid("/grid", width=self.span, height=self.span, plane="xy",
                   cell_size=10.0, section_size=50.0, position=(self.cx, self.cy, -0.01))
        s.add_light_directional("/sun", color=(255, 255, 255), intensity=2.0,
                                position=(self.cx + 100, self.cy - 150, 400))

        # ---- the origins ----
        # Every frame in play, drawn where it actually is, because "which way is +x" and
        # "what is this pose measured from" are the two questions numbers never answer.
        #   table    the ID-10 marker, table (0, 0) -- where the camera frame starts
        #   sim env  the field centre: normalized (0, 0) AND the arm's p0, a third of a
        #            metre away from the marker, which is the confusion worth drawing
        self._origin_axes("/origin", (0.0, 0.0), 12.0,
                          f"table origin - id {ORIGIN_ID} (camera frame)")
        self._origin_axes("/env/origin", FC.FIELD_CENTER_CM, 15.0,
                          "field centre - sim (0, 0) = arm p0")

        # ---- target and end effector ----
        self.target = s.add_frame("/target", show_axes=False,
                                  position=(float(self.p0_cm[0]), float(self.p0_cm[1]), 0.0))
        s.add_line_segments("/target/cross", _cross((0.0, 0.0), 3.0, 0.05),
                            colors=COL_TARGET, thickness=3.0)
        s.add_line_segments("/target/ring", _loop_segments(_disc(2.0), 0.05),
                            colors=COL_TARGET, thickness=3.0)
        self.robot = s.add_frame("/robot", show_axes=False,
                                 position=(float(self.p0_cm[0]), float(self.p0_cm[1]), 0.0))
        s.add_line_segments("/robot/disc", _loop_segments(_disc(PUSHER_R_CM), 0.02),
                            colors=COL_ROBOT, thickness=3.0)

        @self.server.on_client_connect
        def _(client: viser.ClientHandle):
            client.camera.position = (self.cx, self.cy - 1e-3, self.span)
            client.camera.look_at = (self.cx, self.cy, 0.0)
            client.camera.up = (0.0, 1.0, 0.0)

    # -- gui ----------------------------------------------------------------------
    def _gui(self):
        g = self.server.gui
        o = self.origin_pose
        g.add_markdown(
            f"**arm_test_id10** -- click anywhere on the table and the end effector moves "
            f"there. Table centimetres, origin at the ID-{ORIGIN_ID} marker.\n\n"
            f"**p0 is the field's `(0,0)`** -- the pusher on the green box's centre, "
            f"table `({self.p0_cm[0]:.1f}, {self.p0_cm[1]:.1f})` cm, base "
            f"`({o[0]:+.3f}, {o[1]:+.3f})` m. Not the ID-{ORIGIN_ID} marker: that is the "
            f"camera's zero, a third of a metre away, and both triads are drawn.\n\n"
            f"A click is `sim x {FIELD_CM:.0f} cm` from p0 **with both axes flipped** "
            f"(`+x` in sim is `-x` in base) -- a half-turn about p0, so p0 itself stays "
            f"sim `(0,0)`.\n\n"
            f"**Green** is the field: `[-0.5, 0.5]²` for the planner, {FIELD_CM:.0f} cm "
            f"across, centred on p0. Clicks outside it are held at its edge. "
            f"Grey: the nominal {self.args.length:.0f} x {self.args.width:.0f} cm table, "
            f"drawn from the marker, not measured. "
            + (f"Dark: the real env geometry ({self.env['n_polys']} polygons, "
               f"{self.env['n_holes']} holes)."
               if self.env is not None else
               f"_env geometry not loaded: {load_env.last_error}_")
            + f"\n\nHeight is fixed at `z = {o[2]:+.3f}` m and is never commanded.")
        self.enabled = g.add_checkbox("motion enabled", True)
        self.speed = g.add_slider("speed [m/s]", min=0.005, max=0.25, step=0.005,
                                  initial_value=float(self.args.speed))
        self.x_in = g.add_number("table x [cm]", initial_value=float(self.p0_cm[0]),
                                 step=0.1)
        self.y_in = g.add_number("table y [cm]", initial_value=float(self.p0_cm[1]),
                                 step=0.1)
        go = g.add_button("move to table x / y")
        home = g.add_button("go to p0 (the field centre, sim 0,0)")
        self.status = g.add_markdown("_no move yet_")

        @go.on_click
        def _(_event):
            self.request((self.x_in.value, self.y_in.value), "gui")

        @home.on_click
        def _(_event):
            self.request(self.p0_cm, "p0")

    def _clicks(self):
        """Register the scene click handler; fall back to the GUI inputs if unavailable."""
        try:
            @self.server.scene.on_pointer_event(event_type="click")
            def _(event) -> None:
                xy = ray_to_plane(event.ray_origin, event.ray_direction, 0.0)
                if xy is None:
                    return
                self.request(xy, "click")
        except (AttributeError, TypeError) as exc:
            print(f"[view] scene clicks unavailable ({exc}); use the table x / y inputs")

    # -- motion --------------------------------------------------------------------
    def request(self, xy_cm, source):
        """One click -> one bounded ``moveL``.  Never blocks a second caller."""
        if not self.enabled.value:
            self.note = "motion disabled -- tick 'motion enabled'"
            self.refresh()
            return
        if not self.lock.acquire(blocking=False):
            self.note = "busy -- move in flight, click ignored"
            self.refresh()
            return
        try:
            self.move(xy_cm, source)
        finally:
            self.lock.release()

    def move(self, xy_cm, source):
        clicked = np.asarray(xy_cm, dtype=float)[:2]
        xy_sim, held = clamp_sim(FC.table_to_sim(clicked[0], clicked[1])[:2],
                                 self.args.limit)
        pose = sim_to_robot(xy_sim, self.origin_pose)
        pose, boxed = clamp_base_xy(pose, self.box)

        # Read the target back out of the pose actually being sent, so what is drawn is
        # where the arm is going even when a clamp moved it.
        self.target_sim = robot_to_sim(pose, self.origin_pose)
        cm = FC.sim_to_table(self.target_sim[0], self.target_sim[1])[:2]
        self.target.position = (float(cm[0]), float(cm[1]), 0.0)
        self.x_in.value, self.y_in.value = float(cm[0]), float(cm[1])

        # A clamp is the click's whole story, so it outlives the move rather than being
        # overwritten by "arrived" -- otherwise a corner click looks like it just worked.
        held_note = (" (held at the field edge)" if held else
                     " (held at the workspace box)" if boxed else "")
        self.note = "moving ..." + held_note
        self.refresh()

        print(f"[{source}] table ({clicked[0]:+7.2f}, {clicked[1]:+7.2f}) cm -> sim "
              f"({xy_sim[0]:+.3f}, {xy_sim[1]:+.3f}) -> base "
              f"({pose[0]:+.4f}, {pose[1]:+.4f}) m  z={pose[2]:+.4f} (p0's, unchanged)")
        ok, warn = self.arm.move_to(pose, float(self.speed.value), self.args.accel)
        self.note = (warn if warn else ("arrived" if ok else "refused")) + held_note
        self.refresh()

    # -- display -------------------------------------------------------------------
    def refresh(self, tcp=None):
        if tcp is None:
            tcp = self.arm.tcp_pose()
        sim = robot_to_sim(tcp, self.origin_pose)
        cm = FC.sim_to_table(sim[0], sim[1])[:2]
        self.robot.position = (float(cm[0]), float(cm[1]), 0.0)
        tgt_cm = FC.sim_to_table(self.target_sim[0], self.target_sim[1])[:2]
        dz_mm = (tcp[2] - self.origin_pose[2]) * 1000.0
        self.status.content = (
            f"**target** sim ({self.target_sim[0]:+.3f}, {self.target_sim[1]:+.3f}) = "
            f"table ({tgt_cm[0]:+.2f}, {tgt_cm[1]:+.2f}) cm\n\n"
            f"**end effector** sim ({sim[0]:+.3f}, {sim[1]:+.3f}) = table "
            f"({cm[0]:+.2f}, {cm[1]:+.2f}) cm\n\n"
            f"**base** ({tcp[0]:+.4f}, {tcp[1]:+.4f}, {tcp[2]:+.4f}) m "
            f"&nbsp; z-p0 = {dz_mm:+.1f} mm\n\n"
            f"{self.note}")

    def run(self, interval=0.1):
        """Poll the live TCP so the view follows the arm even while it is moving."""
        while True:
            self.refresh()
            time.sleep(interval)


# ======================================================================================
def check_height(arm, origin_pose, z_tol):
    """Refuse to start if the tool is not already at p0's height.

    Every target this program builds carries p0's ``z``, so the *first* ``moveL`` from a
    tool parked higher or lower would itself be the vertical move the program promises
    never to make.  Being at the right height already is the precondition that keeps that
    promise, so it is checked rather than assumed.
    """
    if not arm.live:
        return
    z = float(arm.tcp_pose()[2])
    dz = z - float(origin_pose[2])
    print(f"[arm] live TCP z {z:+.4f} m, p0 z {origin_pose[2]:+.4f} m "
          f"({dz * 1000:+.1f} mm off)")
    if abs(dz) > z_tol:
        raise SystemExit(
            f"\nthe tool is {dz * 1000:+.1f} mm off p0's height, more than --z-tol "
            f"({z_tol * 1000:.0f} mm).\nThis program never moves the tool vertically, so "
            f"it will not make that first move for you:\njog the pusher back to the "
            f"calibrated height with arm_calibrate.py (or widen --z-tol if you are "
            f"certain the gap is safe to close with a slanted moveL).")


def report(origin_pose, args):
    """Print the conversion at the points worth eyeballing before anything moves."""
    o, p0_cm = origin_pose, FC.FIELD_CENTER_CM
    print(f"\nsim -> robot: base_xy = p0_xy + ({SIM_TO_BASE[0]:+.0f}, "
          f"{SIM_TO_BASE[1]:+.0f}) * sim_xy * {FIELD_CM:.1f} cm, z and rotvec from p0")
    print(f"  p0  base ({o[0]:+.4f}, {o[1]:+.4f}, {o[2]:+.4f}) m  "
          f"rotvec ({o[3]:+.4f}, {o[4]:+.4f}, {o[5]:+.4f})")
    print(f"      the end effector on the FIELD CENTRE -- sim (0, 0), table "
          f"({p0_cm[0]:+.2f}, {p0_cm[1]:+.2f}) cm, not the id-{ORIGIN_ID} marker")
    L, W = args.length, args.width
    pts = [(p0_cm[0], p0_cm[1], "the field centre = p0"),
           *((x, y, "field corner") for x, y in FC.field_corners_cm()),
           (0.0, 0.0, f"the id-{ORIGIN_ID} marker"),
           (L, 0.0, "table corner"), (L, W, "table corner"), (0.0, W, "table corner")]
    box = PARAMS.workspace_box_m
    print("\n    table cm         ->     sim          ->  base x, y [m]")
    for x, y, what in pts:
        sim = FC.table_to_sim(x, y)[:2]
        b = sim_to_robot(sim, origin_pose)
        flags = []
        if max(abs(sim[0]), abs(sim[1])) > args.limit + 1e-9:
            flags.append("off the field")
        if not (box["x"][0] <= b[0] <= box["x"][1]
                and box["y"][0] <= b[1] <= box["y"][1]):
            flags.append("outside the box")
        print(f"  ({x:+7.2f}, {y:+7.2f})  ->  ({sim[0]:+6.3f}, {sim[1]:+6.3f})  ->  "
              f"({b[0]:+7.4f}, {b[1]:+7.4f})   {what}"
              + (f"  -- {', '.join(flags)}" if flags else ""))
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ip", default=PARAMS.robot_ip, help="robot IP (default %(default)s)")
    ap.add_argument("--dry-run", action="store_true",
                    help="no robot: clicks print the pose they would have commanded")
    ap.add_argument("--startpos", default=PARAMS.startpos_json,
                    help="arm_calibrate.py's p0 JSON (default %(default)s)")
    ap.add_argument("--speed", type=float, default=0.05, help="moveL tool speed [m/s]")
    ap.add_argument("--accel", type=float, default=0.25, help="moveL tool accel [m/s^2]")
    ap.add_argument("--limit", type=float, default=NORM_HALF,
                    help="clamp clicks to this half-width of the normalized square "
                         "(default %(default)s = the field edge)")
    ap.add_argument("--length", type=float, default=LENGTH_CM,
                    help="nominal table +x extent, cm (default %(default)s)")
    ap.add_argument("--width", type=float, default=WIDTH_CM,
                    help="nominal table +y extent, cm (default %(default)s)")
    ap.add_argument("--no-env", action="store_true",
                    help="skip the env geometry; draw the normalization box alone")
    ap.add_argument("--z-tol", type=float, default=0.005,
                    help="refuse to start if the live TCP is further than this off p0's "
                         "height [m]; the tool is never moved vertically")
    ap.add_argument("--port", type=int, default=8080, help="viser port")
    args = ap.parse_args()

    try:
        origin_pose = load_origin_pose(args.startpos)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc))
    report(origin_pose, args)

    box = PARAMS.workspace_box_m
    if not box["z"][0] <= origin_pose[2] <= box["z"][1]:
        print(f"note: p0's z {origin_pose[2]:+.4f} m is outside the workspace box z "
              f"{box['z']} -- it is held anyway; this program never commands z.")

    arm = DryRunArm(origin_pose) if args.dry_run else RtdeArm(args.ip)
    try:
        check_height(arm, origin_pose, args.z_tol)
        view = TableClickView(arm, origin_pose, args)
        print(f"[view] http://localhost:{args.port}   "
              f"click on the table; Ctrl-C to quit"
              + ("   (DRY RUN)" if args.dry_run else ""))
        view.run()
    except KeyboardInterrupt:
        print("\ninterrupted.")
    finally:
        arm.close()


if __name__ == "__main__":
    main()
