"""Mesh view of the real push-T table -- what the controller thinks is happening.

One viser scene, in table centimetres (the ``camera_test_id10`` frame -- origin at the
ID-10 marker, ``+x`` along the length, ``+y`` along the width), holding only the things
the controller actually reasons about, each drawn as the mesh the planner itself uses:

* the **environment** -- ``2denv4.obj``'s walls and blocks, which *are* the obstacle set;
* the **T**, at its measured pose;
* the **goal** -- the same T mesh, ghosted, at ``push_t_demo_realworld.GOAL_POSE_NORM``;
* the **reference path** -- the SE(2) trace the Eikonal planner solved for, redrawn every
  time the ``Replanner`` re-solves it, in the green ``push_t_demo_sim`` draws it in;
* the **end effector** -- the pusher cylinder, at the TCP the arm reports.

Nothing else: no ArUco squares, no origin triads, no per-frame measurement readout.
``locate_functions.visualize`` is the scene for *checking the table frame* -- it draws
those three boxes and every origin on purpose -- and this one is for *watching a run*.

The meshes are the sim's, carried across by ``push_t_demo_realworld``: table cm and the
sim's mesh units differ by a rotation and one uniform scale (the field is square), so an
object's body-frame mesh only has to be divided by ``sim_units_per_cm()`` and its pose
converted -- no per-frame re-upload, each moving part is a viser frame with its mesh as a
child.

Three ways to drive it::

    # poll the JSON state file push_t_realworld_run.py --viz-state writes
    python push_t_realworld_viz.py --state /tmp/table_state.json

    # no hardware: walk the T along a saved plan (or a lissajous, without --ref)
    python push_t_realworld_viz.py --demo --ref plan.npy

    # in-process, straight out of the control loop
    python push_t_realworld_run.py --viz

State-file schema (every key optional; the last seen value is kept)::

    {
      "shape":     [x_cm, y_cm, theta_rad],
      "robot":     [x_cm, y_cm],
      "reference": [[x_cm, y_cm], ...]
    }

Or import :class:`TableView` and call ``.update(shape=..., robot=..., reference=...)``
from your own loop.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import trimesh
import viser

import push_t_demo_realworld as rw
import pymunk_viser_push as base
from push_t_demo_sim import COL_GHOST, COL_REF

COL_ENV = base.COL_ENV            # the obstacle geometry
COL_TEE = base.COL_TEE            # the shape
COL_PUSHER = base.COL_PUSHER      # the end effector
COL_FLOOR = base.COL_FLOOR        # the table top
# COL_REF / COL_GHOST are push_t_demo_sim's, so the planned path and the goal ghost read
# the same here as they do in the sim demo.


# ======================================================================================
# sim meshes -> table centimetres
# ======================================================================================
def _cm_per_sim():
    """Centimetres per sim mesh unit."""
    return 1.0 / max(rw.sim_units_per_cm(), 1e-9)


def _world_mesh_cm(mesh):
    """A sim-frame mesh placed in the table frame: ``(vertices_cm, faces)``.

    ``x, y`` go through ``path_sim_to_real`` -- the same hop every pose takes -- and ``z``
    is only scaled, then dropped so the mesh stands on the table.
    """
    V = np.asarray(mesh.vertices, dtype=float)
    xy = np.asarray(rw.path_sim_to_real(V[:, :2]), dtype=float)
    z = (V[:, 2] - V[:, 2].min()) * _cm_per_sim()
    return (np.column_stack([xy, z]).astype(np.float32),
            np.asarray(mesh.faces, dtype=np.uint32))


def _body_mesh_cm(mesh):
    """A sim-frame mesh kept in its BODY frame, in cm: ``(vertices_cm, faces)``.

    Table cm and sim units differ by a rotation and one uniform scale, and the rotation is
    already carried by the pose (``sim_to_real_pose`` turns the heading by the same
    amount), so the body geometry itself only needs the scale.  That is what lets the T
    ride a viser frame instead of being re-uploaded every observation.
    """
    V = np.asarray(mesh.vertices, dtype=float) * _cm_per_sim()
    V[:, 2] -= V[:, 2].min()
    return V.astype(np.float32), np.asarray(mesh.faces, dtype=np.uint32)


def _path_segments(xy, z=0.0):
    """``(N, 2)`` polyline -> ``(N-1, 2, 3)`` line segments at height ``z``."""
    p = np.asarray(xy, dtype=float)[:, :2]
    p3 = np.column_stack([p, np.full(len(p), z)]).astype(np.float32)
    return np.stack([p3[:-1], p3[1:]], axis=1)


def _quat_z(theta):
    """Quaternion ``(w, x, y, z)`` for a rotation of ``theta`` radians about ``+z``."""
    return (math.cos(theta / 2.0), 0.0, 0.0, math.sin(theta / 2.0))


# ======================================================================================
# the view
# ======================================================================================
class TableView:
    """A viser server showing the table; call ``update`` as new observations arrive."""

    def __init__(self, geo: rw.Geometry, port=8080, goal=None, reference=None):
        self.geo = geo
        self.server = viser.ViserServer(port=port)
        self.server.scene.set_up_direction('+z')
        self.server.scene.world_axes.visible = False
        self._dyn = {}                        # name -> line-segments handle (redrawn)

        env_v, env_f = _world_mesh_cm(geo.env_mesh)
        tee_v, tee_f = _body_mesh_cm(geo.tee_mesh)
        self.tee_h_cm = float(geo.tee_height) * _cm_per_sim()
        self._ref_z = self.tee_h_cm + 0.5     # the path rides just over the shape

        # framing: the env's own bounds, padded, with the floor and grid filling it
        self.lo, self.hi = env_v[:, :2].min(axis=0), env_v[:, :2].max(axis=0)
        c = 0.5 * (self.lo + self.hi)
        self._center = (float(c[0]), float(c[1]))
        self._span = float(max(self.hi - self.lo) * 1.15)

        self.server.scene.add_box('/floor', color=COL_FLOOR,
                                  dimensions=(self._span, self._span, 2.0),
                                  position=(c[0], c[1], -1.0))
        self.server.scene.add_grid('/grid', width=self._span, height=self._span,
                                   plane='xy', cell_size=10.0, section_size=50.0,
                                   position=(c[0], c[1], 0.02))
        self.server.scene.add_mesh_simple('/env', env_v, env_f, color=COL_ENV,
                                          flat_shading=True)

        # the goal: the same T, ghosted, where the plan is trying to put it
        self.goal = self.server.scene.add_frame('/goal', show_axes=False)
        self.server.scene.add_mesh_simple('/goal/mesh', tee_v, tee_f, color=COL_GHOST,
                                          flat_shading=True, opacity=0.35,
                                          material='standard')
        self.set_goal(geo.goal_pose_real() if goal is None else goal)

        # the shape, at whatever the camera last said
        self.tee = self.server.scene.add_frame('/tee', show_axes=False,
                                               position=(c[0], c[1], 0.0))
        self.server.scene.add_mesh_simple('/tee/mesh', tee_v, tee_f, color=COL_TEE,
                                          flat_shading=True)

        # the end effector: the pusher cylinder, standing on the table
        r_cm = float(geo.args.pusher_radius) * _cm_per_sim()
        h_cm = float(geo.args.pusher_height) * _cm_per_sim()
        pusher = trimesh.creation.cylinder(radius=r_cm, height=h_cm, sections=32)
        pusher.apply_translation([0.0, 0.0, h_cm / 2.0])
        self.robot = self.server.scene.add_frame('/pusher', show_axes=False,
                                                 position=(c[0], c[1], 0.0))
        self.server.scene.add_mesh_simple('/pusher/mesh', pusher.vertices, pusher.faces,
                                          color=COL_PUSHER, flat_shading=True)

        self.server.scene.add_light_directional('/sun', color=(255, 255, 255),
                                                intensity=2.0,
                                                position=(c[0] + 100, c[1] - 150, 400))
        self._connect_camera()
        self._gui()
        if reference is not None:
            self.set_reference(reference)

    # -- setup -----------------------------------------------------------------------
    def _connect_camera(self):
        cx, cy = self._center
        d = self._span

        @self.server.on_client_connect
        def _(client: viser.ClientHandle):
            client.camera.position = (cx, cy - 1e-3, d)
            client.camera.look_at = (cx, cy, 0.0)
            client.camera.up = (0.0, 1.0, 0.0)

    def _gui(self):
        self.server.gui.add_markdown(
            '**Real push-T table** -- table centimetres.\n\n'
            'Orange: the T where the camera sees it. Green ghost: the goal. '
            'Green line: the path the Eikonal planner solved for. Blue: the end effector.')
        self._txt = self.server.gui.add_markdown('_waiting for data_')

    # -- live updates ------------------------------------------------------------------
    def _set(self, name, segs, color, thickness):
        """Redraw a line-segments node in place, or create it the first time."""
        h = self._dyn.get(name)
        if h is not None:
            try:
                h.points = segs
                return
            except Exception:
                try:
                    h.remove()
                except Exception:
                    pass
        self._dyn[name] = self.server.scene.add_line_segments(
            name, segs, colors=color, thickness=thickness)

    def set_goal(self, pose_real):
        """``pose_real`` = ``(x_cm, y_cm, theta_rad)``; ``None`` hides the ghost."""
        if pose_real is None:
            self.goal.visible = False
            self._goal_xy = None
            return
        x, y, th = (float(v) for v in np.asarray(pose_real, dtype=float).ravel()[:3])
        self.goal.position = (x, y, 0.0)
        self.goal.wxyz = _quat_z(th)
        self.goal.visible = True
        self._goal_xy = np.array([x, y])

    def set_shape(self, tee_pose_real):
        """``tee_pose_real`` = ``(x_cm, y_cm, theta_rad)``, the shape's own pose."""
        x, y, th = (float(v) for v in np.asarray(tee_pose_real, dtype=float).ravel()[:3])
        self.tee.position = (x, y, 0.0)
        self.tee.wxyz = _quat_z(th)
        line = (f'**T**  x = {x:+.1f} cm   y = {y:+.1f} cm   '
                f'theta = {math.degrees(th):+.0f}°')
        if self._goal_xy is not None:
            line += f'\n\n**goal** {float(np.hypot(*(self._goal_xy - (x, y)))):.1f} cm away'
        self._txt.content = line

    def set_robot(self, xy_real):
        """``xy_real`` = the end effector's ``(x_cm, y_cm)`` on the table."""
        self.robot.position = (float(xy_real[0]), float(xy_real[1]), 0.0)

    def set_reference(self, path_real):
        """The planned T path, ``(N, 2)`` or ``(N, 3)`` in table cm."""
        p = np.asarray(path_real, dtype=float)
        if len(p) > 1:
            self._set('/ref', _path_segments(p, self._ref_z), COL_REF, 3.0)

    def update(self, shape=None, robot=None, reference=None, goal=None, **_ignored):
        """One observation.  Unknown keys are ignored, so a richer state file is fine."""
        if goal is not None:
            self.set_goal(goal)
        if reference is not None:
            self.set_reference(reference)
        if shape is not None:
            self.set_shape(shape)
        if robot is not None:
            self.set_robot(robot)


def view_emitter(view: TableView):
    """An ``emit(tee, robot_xy, sess)`` for ``push_t_realworld_run.run_pipeline``.

    The reference comes off the session every call, so a replan shows up as soon as it
    lands rather than at the end of the run.
    """
    def _emit(tee, robot_xy, sess):
        view.update(shape=tee, robot=robot_xy,
                    reference=sess.stack.reference_real()[:, :2])

    return _emit


# ======================================================================================
# drivers
# ======================================================================================
def _read_state(path, mtime):
    try:
        m = os.path.getmtime(path)
    except OSError:
        return None, mtime
    if m == mtime:
        return None, mtime
    try:
        with open(path) as f:
            return json.load(f), m
    except (json.JSONDecodeError, OSError):
        return None, mtime          # half-written; try again next tick


def run_state_file(view: TableView, path, interval):
    print(f'[viz] polling {path} every {interval:.2f}s')
    mtime = None
    while True:
        data, mtime = _read_state(path, mtime)
        if data:
            view.update(shape=data.get('shape'), robot=data.get('robot'),
                        reference=data.get('reference'))
        time.sleep(interval)


def run_demo(view: TableView, interval, ref_real=None):
    """No hardware: walk the T along ``ref_real`` if there is one, else a lissajous."""
    r = 3.5 * float(view.geo.args.pusher_radius) * _cm_per_sim()
    if ref_real is not None and ref_real.shape[1] == 2:     # a positions-only path
        ref_real = np.column_stack([ref_real, np.zeros(len(ref_real))])
    if ref_real is not None and len(ref_real) > 1:
        print(f'[viz] replaying a {len(ref_real)}-waypoint plan -- no hardware')
        i = 0
        while True:
            x, y, th = ref_real[i % len(ref_real)]
            a = 0.15 * i
            view.update(shape=(x, y, th),
                        robot=(x + r * math.cos(a), y + r * math.sin(a)))
            i += 1
            time.sleep(interval)

    print('[viz] synthetic demo -- no hardware, no plan')
    mid, amp = 0.5 * (view.lo + view.hi), 0.30 * (view.hi - view.lo)
    t = 0.0
    while True:
        x = mid[0] + amp[0] * math.sin(0.7 * t)
        y = mid[1] + amp[1] * math.sin(0.9 * t + 1.0)
        view.update(shape=(x, y, 0.6 * math.sin(0.4 * t)),
                    robot=(x + r * math.cos(1.3 * t), y + r * math.sin(1.3 * t)))
        t += 0.05
        time.sleep(interval)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--env', default=rw._DEFAULTS['env'])
    ap.add_argument('--shape', default=rw._DEFAULTS['shape'])
    ap.add_argument('--dataPath', default=rw._DEFAULTS['dataPath'])
    ap.add_argument('--port', type=int, default=8080)
    ap.add_argument('--state', default=None, help='JSON state file to poll')
    ap.add_argument('--ref', default=None,
                    help='a saved sim-frame (T,3) plan (push_t_demo_sim --save-path) to '
                         'draw, and to walk the T along under --demo')
    ap.add_argument('--interval', type=float, default=0.05, help='poll / redraw period, s')
    ap.add_argument('--demo', action='store_true', help='run without hardware')
    args = ap.parse_args()

    geo = rw.load_geometry(rw.make_args(env=args.env, shape=args.shape,
                                        dataPath=args.dataPath))
    ref_real = rw.path_sim_to_real(np.load(args.ref)) if args.ref else None
    view = TableView(geo, port=args.port, reference=ref_real)
    goal = geo.goal_pose_real()
    print(f'[viz] http://localhost:{args.port}')
    print(f'[viz] env bbox (table cm): x {view.lo[0]:.1f}..{view.hi[0]:.1f}  '
          f'y {view.lo[1]:.1f}..{view.hi[1]:.1f};  goal ({goal[0]:.1f}, {goal[1]:.1f}, '
          f'{math.degrees(goal[2]):+.0f}deg) cm')

    if args.state and not args.demo:
        run_state_file(view, args.state, args.interval)
    elif args.demo:
        run_demo(view, args.interval, ref_real)
    else:
        print('[viz] no --state / --demo: the static layout only. Ctrl-C to quit.')
        while True:
            time.sleep(1.0)


if __name__ == '__main__':
    main()
