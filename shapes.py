"""Which shape the whole rig is pushing -- one registry, one ``--shape-name`` flag.

The pipeline used to be written for the T and only the T: its mesh, its dataset, its
checkpoint, its 12 x 12 cm proportions and its four marker positions were module
constants spread over ``frame_conversions``, ``locate_functions``,
``push_t_demo_realworld`` and ``push_t_demo_sim``.  This module is where all of that
now lives, once per shape, so adding a fourth shape is a dict entry rather than a sweep
through five files.

Three shapes ship: ``T``, ``rect`` and ``V``.

    python push_t_realworld_run.py --shape-name V --viz
    python push_t_demo_sim.py --shape-name rect
    python locate_functions.py --shape-name rect        # where it is, live, in table cm
    python shapes.py                                    # what each shape resolves to

Selecting one
-------------
``--shape-name`` has to take effect BEFORE ``frame_conversions`` and ``locate_functions`` are
imported, because both of them bake the shape's proportions into function *default
arguments*, which are bound at def time.  So the entry points call
:func:`select_from_argv` at the top of the file, ahead of those imports; it peeks
``sys.argv`` for the flag, falls back to ``$NTRLSHAPE_SHAPE``, and otherwise leaves the
default (``T``) in place.  :func:`add_shape_argument` then adds the real argparse flag so
``--help`` documents it and a typo is still caught by argparse.

Anything importing this module mid-run gets the same answer: the selection is a module
global and :func:`select` refuses to change it once :func:`active` has been read, rather
than letting half the process run on one shape and half on another.

What a shape has to say for itself
----------------------------------
Only the things that genuinely differ.  Everything else -- the env, the table frame, the
calibration, the controller's knobs -- is the rig, not the shape, and stays where it is.

``mesh`` / ``mesh_zup``
    The .obj the planner was trained on.  ``mesh`` is the y-up file
    ``pymunk_viser_push.load_mesh`` maps through ``MESH_TO_WORLD``; ``mesh_zup`` is the
    optional z-up twin, read for nothing but its bounding box.  When it is absent the
    same box is derived from ``mesh`` -- ``MESH_TO_WORLD`` *is* the y-up -> z-up rotation,
    so the two agree exactly (checked on the T: both give (-2.5, 0)).

``data_path``
    The test-set directory, which carries ``meta.json``'s ``env_scale`` / ``env_center``
    -- the normalization the checkpoint was trained in -- alongside the fields the
    Eikonal planner reads.  **It has to be built against the env that is on the table**
    (``2denv4``, 350 units across a 70 cm field).  A dataset generated against some other
    env normalizes differently and every pose out of the planner lands somewhere else;
    :func:`ShapeSpec.check` says so at startup instead of letting it show up as an arm
    that misses by 20 cm.

``ckpt``
    That dataset's checkpoint.

``markers``
    Where each ArUco marker sits on the physical shape, measured from the TOP-LEFT corner
    of its bounding box, ``+x`` right and ``-y`` DOWN -- the frame you can put a ruler in.
    See ``frame_conversions.TEE_MARKER_POS_CM`` for the full picture; an id left ``None``
    is skipped rather than guessed at, so a shape with two markers measured in runs on
    two markers.

``ik_trim_cm``
    How much smaller the cardboard is than the mesh, per side, centimetres.  The planner
    keeps the mesh -- that is the shape the network was trained on -- but the contacts the
    IK samples have to sit on the material that is actually there.

The proportions are NOT in the table
------------------------------------
The outline, the bounding box in centimetres, the area centroid and the thickness are all
READ OFF THE MESH rather than transcribed, because a transcribed number is a number that
can drift from the shape the checkpoint was trained on.  The size in centimetres is not
even a free choice: the field is 350 mesh units across 70 cm, so a mesh unit is 2 mm and
the T's 60-unit box *must* be built 12 cm across or the planner's geometry is wrong.
:attr:`ShapeSpec.size_cm` derives it; there is no constant to keep in step.
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass, field

import numpy as np

from real_world_params import PARAMS

HERE = os.path.dirname(os.path.abspath(__file__))
MESH_DIR = os.path.join(HERE, "datasets", "3dshape")
DATA_DIR = os.path.join(HERE, "testing_data", "3dshape")
# One Eikonal checkpoint per shape, named for the shape rather than all called latest.pt.
CKPT_DIR = os.path.join(HERE, "checkpoints")
# The REAL objects, as manufactured, in millimetres -- see shapes/README.md.  Distinct
# from MESH_DIR, which holds the meshes the checkpoints were TRAINED on.
REAL_DIR = os.path.join(HERE, "shapes")

# mesh -> world, straight out of ``pymunk_viser_push``: X = mx, Y = -mz, Z = my.  Kept
# here as the 3x3 rotation so this module needs neither trimesh nor pymunk to read a
# footprint -- ``frame_conversions`` imports it and must stay numpy-only.
MESH_TO_WORLD_R = np.array([[1.0, 0.0, 0.0],
                            [0.0, 0.0, -1.0],
                            [0.0, 1.0, 0.0]])

# The env that is physically on the table.  Every shape is pushed around this one, so it
# is a property of the rig rather than of the shape, and each shape's dataset has to have
# been generated against it -- see ``ShapeSpec.check``.
TABLE_ENV = os.path.join(MESH_DIR, "2denv4.obj")
TABLE_ENV_SCALE = 350.0          # 2denv4 is 350 x 350 mesh units across the field

# How far a dataset's ``env_scale`` / ``env_center`` may sit from the table env's before
# ``check`` calls it the wrong dataset rather than a re-export.  Tight: the two envs that
# turned up in ``testing_data`` differ by 130 units, 37%.
ENV_SCALE_TOL = 1e-3

_env_norm_cache = {}


def env_norm(path=TABLE_ENV):
    """``(env_scale, env_center_xy)`` of an env mesh -- the planner's normalization for it.

    The dataset generator normalizes as ``(p - env_center) / env_scale`` with
    ``env_scale`` the env's LONGEST plan extent and ``env_center`` its bounding-box
    centre: for ``2denv4`` that is 350 and ``(12.747, 45.972)``, which is exactly what
    ``Tshape3d_env4/meta.json`` records.  Reading it off the mesh here means the frame
    that puts the env on the table -- ``sim_units_per_cm``, the field centre, the env
    outline ``locate_functions`` draws -- comes from the env that IS on the table, for
    every shape alike, rather than from whichever env a shape's dataset happened to be
    generated against.  A dataset built against a different env still normalizes with its
    own ``meta.json`` inside the planner, and ``ShapeSpec.check`` reports that mismatch.

    Falls back to ``(TABLE_ENV_SCALE, (0, 0))`` when the file cannot be read.
    """
    if path not in _env_norm_cache:
        try:
            V, _ = obj_arrays(path)
            W = V @ MESH_TO_WORLD_R.T
            lo, hi = W[:, :2].min(axis=0), W[:, :2].max(axis=0)
            _env_norm_cache[path] = (float((hi - lo).max()), 0.5 * (lo + hi))
        except (OSError, ValueError):
            _env_norm_cache[path] = (TABLE_ENV_SCALE, np.zeros(2))
    scale, center = _env_norm_cache[path]
    return scale, center.copy()


# ======================================================================================
# reading a footprint off an .obj, with numpy alone
# ======================================================================================
def obj_arrays(path):
    """``(V, F)`` -- an .obj's vertices and TRIANGULATED faces, in mesh coordinates.

    Polygons are fanned; ``v//vt/vn`` face syntax is handled.  Deliberately not trimesh:
    ``frame_conversions`` imports this module and has to stay importable with numpy alone
    (it runs outside the Docker image that carries the sim stack).
    """
    V, F = [], []
    with open(path) as fh:
        for ln in fh:
            if ln.startswith("v "):
                V.append([float(t) for t in ln.split()[1:4]])
            elif ln.startswith("f "):
                idx = [int(t.split("/")[0]) - 1 for t in ln.split()[1:]]
                for k in range(1, len(idx) - 1):
                    F.append([idx[0], idx[k], idx[k + 1]])
    if not V or not F:
        raise ValueError(f"{path}: no vertices/faces -- not an .obj this can read")
    return np.asarray(V, dtype=np.float64), np.asarray(F, dtype=np.int64)


def plate_frame(verts, up=None):
    """``(P, axis)`` -- ``verts`` reordered so the plate lies in XY and ``axis`` is up.

    The trained meshes are all authored y-up and go through ``MESH_TO_WORLD``; a CAD
    export is whatever the part happened to be modelled in, and the three real shapes in
    ``shapes/`` are one each of x-, y- and z-up.  Rather than assume, this takes the
    THINNEST axis as the extrusion direction, which is what "a plate" means.  ``up`` names
    it explicitly (``"x"``, ``"y"``, ``"z"``) when a part really is thicker than it is
    wide and the guess would be wrong.
    """
    V = np.asarray(verts, dtype=np.float64)
    if up is None:
        axis = int(np.argmin(V.max(axis=0) - V.min(axis=0)))
    else:
        axis = "xyz".index(str(up).lower())
    keep = [i for i in range(3) if i != axis]
    return V[:, keep + [axis]], "xyz"[axis]


def footprint_loop(verts_world, faces, tol=1e-9):
    """The outline of an extruded prism, as one CCW ``(N, 2)`` loop in world units.

    The inputs are all flat-topped prisms, so the UP-facing cap alone is the footprint:
    keep the triangles whose projection onto XY is counter-clockwise (outward normals, so
    that is the top), weld duplicate XY vertices, and the edges used by exactly one of
    those triangles are the boundary.  Chaining them gives the loop, holes and all --
    which is why the sides are dropped by their zero projected area rather than by a
    z test: a side wall shared between cap and floor would otherwise cancel the boundary
    out entirely.

    Raises if the result is not a single closed loop, which is the honest answer for a
    mesh that is not one prism (the caller has no use for half an outline).
    """
    P = np.asarray(verts_world, dtype=np.float64)[:, :2]
    tri = P[faces]
    area2 = ((tri[:, 1, 0] - tri[:, 0, 0]) * (tri[:, 2, 1] - tri[:, 0, 1])
             - (tri[:, 2, 0] - tri[:, 0, 0]) * (tri[:, 1, 1] - tri[:, 0, 1]))
    cap = faces[area2 > tol]
    if not len(cap):
        raise ValueError("no up-facing faces -- mesh has no footprint to read")

    uniq, inv = np.unique(np.round(P, 6), axis=0, return_inverse=True)
    cap = inv[cap]
    seen = {}
    for t in cap:
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            k = (a, b) if a < b else (b, a)
            seen[k] = seen.get(k, 0) + 1
    edges = [k for k, n in seen.items() if n == 1]
    if not edges:
        raise ValueError("footprint has no boundary -- the cap is not a simple surface")

    adj = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    if any(len(v) != 2 for v in adj.values()):
        raise ValueError("footprint boundary branches -- more than one loop or a hole")

    start = edges[0][0]
    loop, prev, cur = [start], None, start
    while True:
        nxt = [n for n in adj[cur] if n != prev]
        if not nxt:
            raise ValueError("footprint boundary is not closed")
        prev, cur = cur, nxt[0]
        if cur == start:
            break
        loop.append(cur)
    if len(loop) != len(edges):
        raise ValueError(f"footprint is {len(edges)} edges but only {len(loop)} chain "
                         f"into a loop -- the mesh is not a single prism")

    pts = uniq[loop]
    return pts if signed_area(pts) > 0.0 else pts[::-1]


def drop_collinear(pts, tol=1e-6):
    """The same polygon with vertices that lie on the edge between their neighbours gone.

    A CAD export splits a face wherever another body once touched it, which leaves
    vertices in the middle of a straight edge: ``shapes/rect.obj`` comes out of the
    footprint reader as a 6-sided rectangle.  They describe the same polygon, but the
    contact sampler counts corners (``corner_margin`` keeps the pusher off them), so a
    phantom corner mid-edge would strike real contacts out of the table.
    """
    p = np.asarray(pts, dtype=np.float64)
    if len(p) < 4:
        return p
    prev, nxt = np.roll(p, 1, axis=0), np.roll(p, -1, axis=0)
    cross = ((p[:, 0] - prev[:, 0]) * (nxt[:, 1] - prev[:, 1])
             - (p[:, 1] - prev[:, 1]) * (nxt[:, 0] - prev[:, 0]))
    scale = max(float(np.ptp(p[:, 0])), float(np.ptp(p[:, 1])), 1.0)
    keep = np.abs(cross) > tol * scale * scale
    return p[keep] if int(keep.sum()) >= 3 else p


def signed_area(pts):
    """Shoelace signed area of a closed polygon given by its vertices (CCW positive)."""
    p = np.asarray(pts, dtype=np.float64)
    q = np.roll(p, -1, axis=0)
    return float((p[:, 0] * q[:, 1] - q[:, 0] * p[:, 1]).sum()) / 2.0


def polygon_centroid(pts):
    """Area centroid of a simple polygon, by the shoelace formula."""
    p = np.asarray(pts, dtype=np.float64)
    q = np.roll(p, -1, axis=0)
    cross = p[:, 0] * q[:, 1] - q[:, 0] * p[:, 1]
    area = float(cross.sum()) / 2.0
    if abs(area) < 1e-12:
        return np.zeros(2)
    return np.array([float((cross * (p[:, 0] + q[:, 0])).sum()),
                     float((cross * (p[:, 1] + q[:, 1])).sum())]) / (6.0 * area)


def point_in_polygon(x, y, pts, tol=0.0):
    """Is ``(x, y)`` inside the polygon, or within ``tol`` of its boundary?

    Crossing number plus a distance-to-edge test, so ``tol`` grows the shape uniformly --
    which is what a marker centre measured with a ruler needs, since one glued up against
    an edge reads a couple of millimetres over it.
    """
    p = np.asarray(pts, dtype=np.float64)
    q = np.roll(p, -1, axis=0)
    x, y = float(x), float(y)

    straddle = (p[:, 1] > y) != (q[:, 1] > y)
    if np.any(straddle):
        a, b = p[straddle], q[straddle]
        xs = a[:, 0] + (y - a[:, 1]) * (b[:, 0] - a[:, 0]) / (b[:, 1] - a[:, 1])
        if int(np.count_nonzero(xs > x)) % 2 == 1:
            return True
    if tol <= 0.0:
        return False

    d = q - p
    n2 = np.einsum("ij,ij->i", d, d)
    n2[n2 == 0.0] = 1.0
    t = np.clip(((x - p[:, 0]) * d[:, 0] + (y - p[:, 1]) * d[:, 1]) / n2, 0.0, 1.0)
    near = p + t[:, None] * d
    return bool(np.min(np.hypot(near[:, 0] - x, near[:, 1] - y)) <= tol)


# ======================================================================================
# one shape
# ======================================================================================
@dataclass(frozen=True)
class ShapeSpec:
    """Everything that is true of one pushable shape and of no other.

    The geometry accessors are all derived from ``mesh`` and cached on first use; nothing
    here is a transcribed proportion.  ``size_cm`` in particular is derived from the
    field calibration, not chosen -- see the module docstring.
    """

    name: str
    mesh: str
    data_path: str
    ckpt: str
    aliases: tuple = ()
    mesh_zup: str = None
    env: str = TABLE_ENV
    # Where the plan ends, in the planner's NORMALIZED frame: (x, y, rz) with x, y in the
    # env's [-0.5, 0.5]^2 box and rz in TURNS, or the full (x, y, z, rx, ry, rz).  Same
    # numbers as the test set's sampled_points.npy, so a pose can be pasted straight out
    # of it.  Per shape because a goal pose only means anything for the shape it was
    # chosen for -- a T's goal heading says nothing about where a V should end up.
    goal_pose_norm: tuple = (0.0, 0.0, 0.0)
    # id -> (x_cm, y_cm) from the shape's TOP-LEFT corner, +x right, -y DOWN.  None = not
    # measured yet, and skipped rather than guessed at.
    markers: dict = field(default_factory=dict)
    # Heading of a marker's own +x edge in the shape's BODY frame, degrees CCW.  One
    # number for every marker on the shape, because they are all glued on the same way
    # round; a marker deliberately turned differently carries its own as an optional
    # third element of its ``markers`` row.
    marker_yaw_deg: float = 180.0
    # The REAL object as manufactured -- the .obj in ``shapes/``, in MILLIMETRES.  The
    # planner keeps ``mesh`` (that is what the checkpoint was trained on and what the goal
    # is expressed against); this is the material the pusher actually touches, so it is
    # what the IK footprint and the measured contact primitives want.  None = no CAD file,
    # fall back to ``mesh`` clipped by ``ik_trim_cm``.
    real_mesh: str = None
    # Which axis of ``real_mesh`` is the extrusion direction, "x" / "y" / "z".  None reads
    # it as the thinnest axis, which is right for every plate; name it when a part is
    # genuinely thicker than it is wide.
    real_up: str = None
    # Turn applied to ``real_mesh``'s footprint, degrees CCW, to bring it into the TRAINED
    # mesh's orientation -- so that a heading of 0 means the same thing for both.  CAD
    # parts come out however they were modelled: ``shapes/rect.obj`` is landscape where
    # the trained rectangle is portrait, so it needs 90.
    real_rot_deg: float = 0.0
    # Which geometry the ``markers`` table below was measured against:
    #   "trained"  the mesh in ``mesh`` (the historical behaviour, and what the T's rows
    #              were measured against -- a nominal 12 x 12 box);
    #   "real"     ``real_mesh``, turned by ``real_rot_deg`` -- the object you actually
    #              put a ruler on.  Prefer this for anything measured from now on: the
    #              real shape is a few millimetres off the trained one, and the marker
    #              offset is only as good as the box it is measured in.
    marker_frame: str = "trained"
    # Where the LOWEST id was nominally meant to go: horizontally centred, this far down
    # from the shape's top edge, centimetres.  Only the T has such a convention -- its ID 1
    # is a designed placement the measured row can be held against -- so it is None for
    # anything whose markers were simply put where they fit, and the self-test then has
    # nothing to compare and skips it.
    marker_nominal_from_top_cm: float = None
    # How much smaller the cardboard is than the mesh, per side, in centimetres, as
    # ``(x_lo, y_lo, x_hi, y_hi)`` in the shape's BODY frame (+y up, +x right -- so for
    # the T, which hangs its stem downward, ``y_lo`` trims the bottom of the stem and
    # ``x_lo``/``x_hi`` the two ends of the crossbar).  The footprint handed to the IK and
    # to the primitive calibration is the mesh footprint clipped to a box this much
    # smaller; nothing is re-centred, so the body origin stays the mesh centroid the
    # camera reports.  All zeros samples the mesh shape itself.
    ik_trim_cm: tuple = (0.0, 0.0, 0.0, 0.0)
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    # ---------------------------------------------------------------- mesh geometry
    def _mesh(self):
        """``(world verts, faces)`` of ``mesh``, mapped through ``MESH_TO_WORLD``."""
        if "mesh" not in self._cache:
            V, F = obj_arrays(self.mesh)
            self._cache["mesh"] = (V @ MESH_TO_WORLD_R.T, F)
        return self._cache["mesh"]

    def outline_units(self):
        """The footprint as a CCW ``(N, 2)`` loop in WORLD mesh units, as authored.

        Not re-centred: this is the polygon in the mesh's own coordinates, which is what
        ``bbox_center_units`` is measured against.
        """
        if "outline" not in self._cache:
            V, F = self._mesh()
            self._cache["outline"] = footprint_loop(V, F)
        return self._cache["outline"]

    def bbox_units(self):
        """``(lo_xy, hi_xy)`` of the footprint, world mesh units."""
        o = self.outline_units()
        return o.min(axis=0), o.max(axis=0)

    def bbox_center_units(self):
        """The footprint bounding-box centre, world mesh units -- what the planner poses
        the shape about (``push_t_demo_sim.planner_to_world``'s ``bbox_to_centroid``
        starts here).

        Read from ``mesh_zup``'s bounding box when that file exists, so the T keeps
        reading the exact file it always did, and derived from ``mesh`` otherwise.  The
        two agree by construction: ``MESH_TO_WORLD`` is the y-up -> z-up rotation.
        """
        if "bbox_c" not in self._cache:
            if self.mesh_zup and os.path.exists(self.mesh_zup):
                Vz, _ = obj_arrays(self.mesh_zup)
                c = 0.5 * (Vz[:, :2].min(axis=0) + Vz[:, :2].max(axis=0))
            else:
                lo, hi = self.bbox_units()
                c = 0.5 * (lo + hi)
            self._cache["bbox_c"] = c
        return self._cache["bbox_c"]

    def thickness_units(self):
        """How thick the prism is, world mesh units (the z extent)."""
        V, _ = self._mesh()
        return float(V[:, 2].max() - V[:, 2].min())

    # ---------------------------------------------------------------- the cm scale
    def meta(self):
        """``data_path``'s ``meta.json`` as a dict, or ``{}`` when it is not there."""
        if "meta" not in self._cache:
            try:
                with open(os.path.join(self.data_path, "meta.json")) as fh:
                    self._cache["meta"] = json.load(fh)
            except (OSError, ValueError):
                self._cache["meta"] = {}
        return self._cache["meta"]

    def env_scale(self):
        """This dataset's OWN recorded normalizing scale, mesh units.

        Only for :meth:`check` to compare against :meth:`table_env_scale`: the rig plans
        and converts positions with the table's normalization alone, the same for every
        shape, never this per-dataset copy.
        """
        try:
            return float(self.meta()["env_scale"])
        except (KeyError, TypeError, ValueError):
            return TABLE_ENV_SCALE

    def env_center(self):
        """This dataset's OWN recorded normalizing centre, ``(x, y, z)`` mesh units.

        Only for :meth:`check`, same as :meth:`env_scale`.
        """
        c = self.meta().get("env_center", (0.0, 0.0, 0.0))
        c = tuple(float(v) for v in c)
        return c if len(c) == 3 else (c[0], c[1], 0.0)

    def table_env_scale(self):
        """The normalizing scale of the env ON THE TABLE (``env``), mesh units.

        What ``units_per_cm`` and the table <-> sim frame use.  Equal to ``env_scale()``
        whenever the dataset was generated against this env; ``check`` reports when not.
        """
        return env_norm(self.env)[0]

    def table_env_center(self):
        """The normalizing centre of the env on the table, ``(x, y)`` mesh units."""
        return env_norm(self.env)[1]

    def units_per_cm(self):
        """Mesh units per real centimetre -- ``table env scale / mean(field cm)``.

        The same number ``push_t_demo_realworld.sim_units_per_cm()`` reports, derived here
        so the shape's size in centimetres has a source rather than a constant.  2denv4 is
        350 units across a 70 cm field, so this is 5: one mesh unit is 2 mm -- for every
        shape, because it is the env on the table that fixes the scale, not the shape's
        dataset.
        """
        f = np.array([float(PARAMS.env_x_cm), float(PARAMS.env_y_cm)], dtype=float)
        if not np.all(f > 0.0):
            raise ValueError(f"field size must be positive centimetres, got {tuple(f)}")
        return float(self.table_env_scale() / f.mean())

    # ---------------------------------------------------------------- cm geometry
    def size_cm(self):
        """``(width_cm, height_cm)`` of the shape's bounding box on the table.

        Derived, not chosen: a mesh unit is ``1 / units_per_cm()`` centimetres because the
        field is, so the physical shape has to be cut to this or the planner's geometry no
        longer matches the object.  The T comes out 12.0 x 12.0.
        """
        lo, hi = self.bbox_units()
        return tuple((hi - lo) / self.units_per_cm())

    def thickness_cm(self):
        """How thick the physical shape should be, centimetres (same scale)."""
        return self.thickness_units() / self.units_per_cm()

    def outline_cm(self, mode="centroid"):
        """The outline as a CCW ``(N, 2)`` loop in centimetres, about ``mode``'s centre.

        ``bbox``     -> about the middle of the bounding box.
        ``centroid`` -> about the area centroid, which is the pymunk body origin and the
                        point the camera reports, so it is the default everywhere.
        """
        key = ("outline_cm", mode)
        if key not in self._cache:
            o = (self.outline_units() - self.bbox_center_units()) / self.units_per_cm()
            self._cache[key] = o - self.center_local_cm(mode)
        return self._cache[key]

    def center_local_cm(self, mode="centroid"):
        """Where "the centre" sits in BBOX-centred centimetres.

        ``bbox`` is ``(0, 0)`` by definition; ``centroid`` is the area centroid, which for
        the T sits 2.27 cm above the box middle because the crossbar carries more area
        than the stem.
        """
        if mode not in CENTER_MODES:
            raise ValueError(f"center mode must be one of {CENTER_MODES}, got {mode!r}")
        if mode == "bbox":
            return np.zeros(2)
        if "centroid_cm" not in self._cache:
            o = (self.outline_units() - self.bbox_center_units()) / self.units_per_cm()
            self._cache["centroid_cm"] = polygon_centroid(o)
        return self._cache["centroid_cm"]

    def topleft_to_body_cm(self, x_cm, y_cm):
        """A point measured from the shape's TOP-LEFT corner -> BBOX-centred body cm.

        The measuring frame has its origin at that corner with ``-y`` running down the
        shape; the body frame is centred on the bounding box with ``+y`` running up it.
        Same axis directions, so this is a pure translation by half the box -- per axis,
        because a rectangle's box is not square.
        """
        w, h = self.marker_box_cm()
        return np.array([float(x_cm) - w / 2.0, float(y_cm) + h / 2.0])

    def body_to_topleft_cm(self, x_cm, y_cm):
        """Inverse of :meth:`topleft_to_body_cm` -- a body point back onto the ruler."""
        w, h = self.marker_box_cm()
        return np.array([float(x_cm) + w / 2.0, float(y_cm) - h / 2.0])

    # ------------------------------------------------- the real object, from CAD (mm)
    def _real(self):
        """``(P, faces, axis)`` for ``real_mesh``, plate lying in XY.  Raises if unset."""
        if not self.real_mesh:
            raise ValueError(f"shape {self.name} has no real_mesh -- add its CAD .obj to "
                             f"shapes/ and name it in shapes.SHAPES[{self.name!r}]")
        if "real" not in self._cache:
            V, F = obj_arrays(self.real_mesh)
            P, axis = plate_frame(V, self.real_up)
            self._cache["real"] = (P, F, axis)
        return self._cache["real"]

    def has_real_mesh(self):
        """Is there a CAD file for the physical object, and can it be read?"""
        return bool(self.real_mesh) and os.path.exists(self.real_mesh)

    def real_outline_cm(self, center="bbox"):
        """The MANUFACTURED shape's outline, ``(N, 2)`` CCW, in centimetres.

        The file is in millimetres, so this is a straight ``/ 10`` -- **not** a conversion
        through ``units_per_cm``, which is for the trained mesh's arbitrary units.  These
        are the dimensions of the object you can put a ruler against.

        Turned by ``real_rot_deg`` first, so the result is in the TRAINED mesh's
        orientation -- a heading of 0 means the same thing for both.  ``center`` is
        ``"bbox"`` (the bounding-box middle, the default) or ``"centroid"``.
        """
        key = ("real_cm", center)
        if key not in self._cache:
            P, F, _ = self._real()
            loop = drop_collinear(footprint_loop(P, F)) / 10.0
            if self.real_rot_deg:
                t = math.radians(float(self.real_rot_deg))
                c, sn = math.cos(t), math.sin(t)
                loop = loop @ np.array([[c, sn], [-sn, c]])
            loop = loop - 0.5 * (loop.min(axis=0) + loop.max(axis=0))
            if center == "centroid":
                loop = loop - polygon_centroid(loop)
            elif center != "bbox":
                raise ValueError(f"center must be one of {CENTER_MODES}, got {center!r}")
            self._cache[key] = loop
        return self._cache[key]

    def real_size_cm(self):
        """``(width_cm, height_cm)`` of the manufactured shape's bounding box."""
        o = self.real_outline_cm()
        return tuple(o.max(axis=0) - o.min(axis=0))

    def real_thickness_cm(self):
        """How tall the manufactured shape stands, centimetres."""
        P, _F, _axis = self._real()
        return float(P[:, 2].max() - P[:, 2].min()) / 10.0

    # ------------------------------------------------- the frame the markers live in
    def marker_real(self):
        """Is the ``markers`` table measured against the real object rather than the mesh?"""
        if self.marker_frame not in ("trained", "real"):
            raise ValueError(f"marker_frame must be 'trained' or 'real', got "
                             f"{self.marker_frame!r}")
        return self.marker_frame == "real"

    def marker_outline_cm(self, mode="bbox"):
        """The outline the ``markers`` rows were measured against, centimetres."""
        return (self.real_outline_cm(mode) if self.marker_real()
                else self.outline_cm(mode))

    def marker_box_cm(self):
        """``(width, height)`` of the bounding box the ``markers`` rows were read off."""
        o = self.marker_outline_cm()
        return tuple(o.max(axis=0) - o.min(axis=0))

    def marker_center_cm(self, mode="centroid"):
        """Where the pose origin sits in that box's BBOX-centred frame, centimetres.

        With ``marker_frame="real"`` the two shapes are lined up by their bounding-box
        CENTRES -- that is how you would lay the real object over a drawing of the mesh --
        so the point the offset walks out to is the TRAINED mesh's centre-of-pose measured
        from the real box's middle.  Keeping the trained centroid here is what matters:
        it is the origin ``load_geometry`` poses ``tee_poly`` about and therefore the one
        the planner is expecting back.
        """
        return self.center_local_cm(mode)

    def prism_cm(self, mode="centroid", z=0.0, thickness_cm=None):
        """``(vertices, faces)`` for the shape as a solid prism standing on ``z``, in cm.

        The shape sits *on* the table -- ``z`` is its underside and it is extruded up by
        ``thickness_cm`` (its own, scaled, unless overridden) -- so it reads as the same
        object ``push_t_demo_sim`` draws rather than as a flat decal.

        The cap is the mesh's OWN up-facing triangles, scaled into centimetres, so a
        concave outline (the T's notches, the V's opening) is triangulated correctly with
        no ear-clipping of our own and no triangles spilling outside the shape.  The walls
        are quads off the boundary loop.
        """
        V, F = self._mesh()
        k = self.units_per_cm()
        shift = self.bbox_center_units() + self.center_local_cm(mode) * k

        tri = V[:, :2][F]
        area2 = ((tri[:, 1, 0] - tri[:, 0, 0]) * (tri[:, 2, 1] - tri[:, 0, 1])
                 - (tri[:, 2, 0] - tri[:, 0, 0]) * (tri[:, 1, 1] - tri[:, 0, 1]))
        cap = F[area2 > 1e-9]

        P = (V[:, :2] - shift) / k
        thick = self.thickness_cm() if thickness_cm is None else float(thickness_cm)
        z0, z1 = float(z), float(z) + thick

        n = len(P)
        verts = np.concatenate([np.column_stack([P, np.full(n, z0)]),
                                np.column_stack([P, np.full(n, z1)])])
        faces = [cap[:, ::-1], cap + n]                       # floor (flipped) + ceiling

        loop = self.outline_units()
        idx = {tuple(np.round(p, 6)): i for i, p in enumerate(np.round(V[:, :2], 6))}
        ring = [idx[tuple(np.round(p, 6))] for p in loop]
        walls = []
        for j, a in enumerate(ring):                          # one quad per boundary edge
            b = ring[(j + 1) % len(ring)]
            walls += [(a, b, b + n), (a, b + n, a + n)]
        faces.append(np.asarray(walls, dtype=np.int64))
        return (verts.astype(np.float32),
                np.concatenate(faces).astype(np.uint32))

    def on_shape(self, x_cm, y_cm, tol_cm=0.0):
        """Is this BBOX-centred body point on the shape's material?

        ``tol_cm`` grows the outline, for a centre measured a hair outside a real edge.
        """
        return point_in_polygon(x_cm, y_cm, self.marker_outline_cm("bbox"), tol_cm)

    # ---------------------------------------------------------------- self-check
    def check(self):
        """Problems with this shape's files and dataset, as a list of strings.

        Empty means everything lines up.  Called at startup by the entry points, because
        every one of these fails as a plausible-looking run that lands in the wrong place
        rather than as an exception.
        """
        out = []
        for label, p in (("mesh", self.mesh), ("env", self.env)):
            if not os.path.exists(p):
                out.append(f"{label} {p} is missing")
        if self.real_mesh and not os.path.exists(self.real_mesh):
            out.append(f"real_mesh {self.real_mesh} is missing -- the IK has no "
                       f"manufactured outline to strike contacts against")
        if self.mesh_zup and not os.path.exists(self.mesh_zup):
            out.append(f"mesh_zup {self.mesh_zup} is missing (bbox falls back to the mesh)")
        if not os.path.isdir(self.data_path):
            out.append(f"dataset {self.data_path} is missing -- the planner has no "
                       f"env/speed field and no env_scale to normalize with")
            return out
        if not os.path.exists(self.ckpt):
            out.append(f"checkpoint {self.ckpt} is missing")
        meta = self.meta()
        if not meta:
            out.append(f"{self.data_path}/meta.json is missing or unreadable")
            return out
        tscale, tcenter = env_norm(self.env)
        if (abs(self.env_scale() - tscale) > ENV_SCALE_TOL
                or np.abs(np.asarray(self.env_center()[:2]) - tcenter).max()
                > ENV_SCALE_TOL):
            out.append(f"dataset {os.path.basename(self.data_path)} normalizes with "
                       f"env_scale {self.env_scale():g} about "
                       f"({self.env_center()[0]:.3f}, {self.env_center()[1]:.3f}), but "
                       f"the env on the table is {os.path.basename(self.env)} at "
                       f"{tscale:g} about ({tcenter[0]:.3f}, {tcenter[1]:.3f}) -- this "
                       f"dataset was generated against a different environment, so every "
                       f"planned pose will be off. Rebuild it against "
                       f"{os.path.basename(self.env)}.")
        shape_obj = os.path.basename(str(meta.get("shape_obj", "")))
        want = {os.path.basename(self.mesh)}
        if self.mesh_zup:
            want.add(os.path.basename(self.mesh_zup))
        if shape_obj and shape_obj not in want:
            out.append(f"dataset was generated for {shape_obj}, but --shape {self.name} "
                       f"pushes {os.path.basename(self.mesh)}")
        return out

    def describe(self):
        """A one-screen summary, for the banner the entry points print at startup."""
        w, h = self.size_cm()
        cx, cy = self.center_local_cm("centroid")
        ids = sorted(i for i, p in self.markers.items() if p is not None)
        todo = sorted(i for i, p in self.markers.items() if p is None)
        lines = [
            f"shape {self.name}: {os.path.basename(self.mesh)}, "
            f"{w:.1f} x {h:.1f} cm x {self.thickness_cm():.1f} cm thick "
            f"({self.units_per_cm():g} mesh units/cm)",
            f"  dataset {self.data_path}  ckpt {self.ckpt}  env "
            f"{os.path.basename(self.env)}",
            f"  centroid {cx:+.2f}, {cy:+.2f} cm off the bbox centre; "
            f"{len(self.outline_units())}-sided outline",
            f"  markers measured {ids or 'NONE'}"
            + (f"; still to measure {todo}" if todo else ""),
        ]
        if self.has_real_mesh():
            rw, rh = self.real_size_cm()
            lines.append(
                f"  real {os.path.basename(self.real_mesh)}: {rw:.2f} x {rh:.2f} cm x "
                f"{self.real_thickness_cm():.2f} cm tall, "
                f"{len(self.real_outline_cm())}-sided "
                f"(vs {w:.1f} x {h:.1f} cm trained)")
        return "\n".join(lines)


CENTER_MODES = ("centroid", "bbox")


def _mesh(name):
    return os.path.join(MESH_DIR, name)


def _data(name):
    return os.path.join(DATA_DIR, name)


def _real(name):
    return os.path.join(REAL_DIR, name)


def _ckpt(name):
    return os.path.join(CKPT_DIR, name)


# ======================================================================================
# the registry
# ======================================================================================
# Each shape needs a dataset generated against 2denv4 (the env on the table) and that
# dataset's checkpoint.  ``ShapeSpec.check`` reports whichever of those is missing or was
# built against something else, at startup, rather than letting it become a bad plan.
SHAPES = {
    "T": ShapeSpec(
        name="T",
        aliases=("t", "tee", "Tshape3d"),
        mesh=_mesh("Tshape3d.obj"),
        mesh_zup=_mesh("Tshape3d_zup.obj"),
        real_mesh=_real("T.obj"),            # 110.0 x 115.0 mm, 20 mm thick, y-up
        data_path=_data("Tshape3d_env4"),
        ckpt=_ckpt("T.pt"),
        goal_pose_norm=(-0.05, 0.31, 0.0),
        # Measured with a ruler from the T's top-left corner; ids 2 and 3 flank ID 1 on
        # the crossbar, ID 4 is down the stem.
        markers={1: (5.5, -0.7), 2: (1.6, -0.7), 3: (9.8, -0.7), 4: (5.5, -10.3)},
        marker_nominal_from_top_cm=1.2,     # ID 1's designed placement, to check against
        marker_yaw_deg=180.0,
        # The T on the table is smaller than the mesh: 0.5 cm off each end of the
        # crossbar and 1 cm off the bottom of the stem.  Nothing off the top edge.
        ik_trim_cm=(0.5, 1.0, 0.5, 0.0),
    ),
    "rect": ShapeSpec(
        name="rect",
        aliases=("r", "rectangle", "box"),
        mesh=_mesh("rectangle.obj"),
        mesh_zup=None,                      # no z-up twin; the bbox comes off the mesh
        real_mesh=_real("rect.obj"),         # 110.0 x 45.0 mm, 35 mm thick, x-up
        # The CAD part is landscape (11.0 wide x 4.5 tall); the trained mesh -- and so the
        # whole pipeline -- holds the rectangle PORTRAIT, 15 x 60 units.  A quarter turn
        # lines them up.  Which way round does not matter here: a rectangle is symmetric
        # under 180, so +90 and -90 give the same footprint.
        real_rot_deg=90.0,
        # The markers were measured on the real part, so the 4.5 x 11.0 cm box is the one
        # the rows below are read off -- NOT the trained mesh's 3.0 x 12.0.  Measuring in
        # the wrong box would put both markers 0.75 cm off the centre line.
        marker_frame="real",
        data_path=_data("rectangle_env4"),
        ckpt=_ckpt("rect.pt"),
        goal_pose_norm=(0, 0.31, 0.25),
        # TWO markers, on the centre line, one tucked into each end: ids 1 (bottom) and
        # 2 (top), each half a WIDTH (4.5 / 2 = 2.25 cm) in from its end.  Held portrait,
        # from the TOP-LEFT corner, +x right and -y DOWN:
        #   x  = 4.5 / 2                      = 2.25   (centred across the width, both)
        #   y1 = -11.0 + 2.25                 = -8.75  (bottom end)
        #   y2 =   0.0 - 2.25                 = -2.25  (top end)
        # 6.5 cm apart, symmetric about the middle: 2.25 + 6.50 + 2.25 = 11.0.
        markers={1: (2.25, -8.75), 2: (2.25, -2.25)},
        # Stuck on square to the rectangle -- a marker's +x edge runs along the shape's
        # +x -- so there is no mount turn to take off the measured heading.
        marker_yaw_deg=0.0,
        ik_trim_cm=(0.0, 0.0, 0.0, 0.0),
    ),
    "V": ShapeSpec(
        name="V",
        aliases=("v", "vee", "Vshape3d"),
        mesh=_mesh("Vshape3d.obj"),
        mesh_zup=None,
        real_mesh=_real("V.obj"),            # 104.0 x 114.7 mm, 45 mm thick, z-up
        # The CAD part is turned 245 deg from the trained mesh.  Not a guess and not a
        # fit: the V is isoceles with a 65.00 deg opening, and 245 is the only turn that
        # carries BOTH of the real part's outward arm directions onto the trained one's
        # (-90 -> +155 and -155 -> +90, to 0.01 deg).
        real_rot_deg=245.0,
        # Measured on the real part -- a 10.40 x 11.47 cm box, not the trained mesh's
        # 17.3 x 19.1 -- so the rows below are read off that.
        marker_frame="real",
        data_path=_data("Vshape3d_env4"),
        ckpt=_ckpt("V.pt"),
        goal_pose_norm=(0, 0.25, 0.25),
        # TWO markers, one on the free end of each arm, placed exactly like the
        # rectangle's: centred across the arm's 23.00 mm width, and half a width
        # (11.50 mm) in from the very end.  Rows are (x_cm, y_cm, yaw_deg) from the
        # TOP-LEFT corner of that box, +x right and -y DOWN.
        #
        # THE YAW IS PER MARKER HERE, which is why each row carries a third element --
        # the T and the rectangle share one mount angle, this shape does not.  Each
        # marker FACES along its own arm's outward NORMAL: perpendicular to the arm,
        # pointing away from the other arm.  Stand the V up the way the letter reads
        # (point at the bottom, arms opening upward) and both facings slope DOWNWARD and
        # away from each other, 115 deg apart.  The marker's +x EDGE is therefore
        # parallel to the arm it sits on, and yaw -- which measures that edge -- is the
        # facing turned -90.
        #
        #   id 1  right arm (the one that maps to the trained V's long straight side;
        #         outward +90 in the body frame),  facing   +0 deg  -> yaw 270
        #   id 2  left arm  (the angled one, outward +155),         facing -115 deg  -> yaw 155
        markers={1: (9.245, -1.150, 270.0),
                 2: (1.528, -6.067, 155.0)},
        # Unused: every row above carries its own yaw.  Kept as the fallback for a row
        # written without one.
        marker_yaw_deg=270.0,
        ik_trim_cm=(0.0, 0.0, 0.0, 0.0),
    ),
}

DEFAULT_SHAPE = "T"
ENV_VAR = "NTRLSHAPE_SHAPE"
FLAGS = ("--shape-name", "--shape_name")

_selected = None          # what select() was last told
_locked = False           # whether active() has been read, and so baked into defaults


def names():
    """The canonical shape names, in registry order."""
    return tuple(SHAPES)


def resolve(name):
    """A name or alias -> its :class:`ShapeSpec`.  Raises on anything else."""
    if isinstance(name, ShapeSpec):
        return name
    key = str(name).strip()
    if key in SHAPES:
        return SHAPES[key]
    low = key.lower()
    for spec in SHAPES.values():
        if low == spec.name.lower() or low in {a.lower() for a in spec.aliases}:
            return spec
    raise ValueError(f"unknown shape {name!r} -- choose one of {', '.join(names())}")


def select(name):
    """Make ``name`` the shape the process is running.  Returns its :class:`ShapeSpec`.

    Has to happen before ``frame_conversions`` / ``locate_functions`` are imported: both
    bake the shape's proportions into function default arguments, which bind at def time.
    Selecting the shape that is already active is always fine; changing it after anything
    has read :func:`active` raises, because half the process would keep the old geometry.
    """
    global _selected
    spec = resolve(name)
    if _locked and _selected is not None and spec.name != _selected:
        raise RuntimeError(
            f"cannot switch to shape {spec.name!r}: {_selected!r} is already baked into "
            f"the imported modules' defaults. Select the shape before importing "
            f"frame_conversions / locate_functions, or run a second process.")
    _selected = spec.name
    return spec


def active():
    """The :class:`ShapeSpec` in play -- :func:`select`, else ``$NTRLSHAPE_SHAPE``, else T.

    Reading this LOCKS the selection, so a later :func:`select` for a different shape
    raises rather than half-applying.
    """
    global _selected, _locked
    if _selected is None:
        _selected = resolve(os.environ.get(ENV_VAR) or DEFAULT_SHAPE).name
    _locked = True
    return SHAPES[_selected]


def active_name():
    """Just the name of the active shape."""
    return active().name


def peek_argv(argv=None):
    """The shape named on the command line, or ``None`` -- without parsing anything else.

    Handles ``--shape-name V``, ``--shape-name=V`` and the underscore spelling.  Unknown
    values are left alone so argparse reports them properly later, with its own message
    and the full choice list.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    for i, tok in enumerate(argv):
        for flag in FLAGS:
            if tok == flag and i + 1 < len(argv):
                val = argv[i + 1]
            elif tok.startswith(flag + "="):
                val = tok[len(flag) + 1:]
            else:
                continue
            try:
                return resolve(val).name
            except ValueError:
                return None
    return None


def select_from_argv(argv=None):
    """Apply ``--shape-name`` from the command line (or the env var) and return the spec.

    Call this at the TOP of an entry point, before ``frame_conversions`` and
    ``locate_functions`` are imported.  :func:`add_shape_argument` then adds the real
    flag, so this peek never has to validate or report anything.
    """
    name = peek_argv(argv)
    return select(name) if name else active()


def add_shape_argument(ap, flag=FLAGS[0]):
    """Add the documented ``--shape-name`` flag to an ``argparse`` parser.

    The value is already in force by the time argparse sees it (:func:`select_from_argv`
    ran at import); this is what puts it in ``--help`` and what rejects a typo.
    """
    ap.add_argument(flag, dest="shape_name", default=active_name(), choices=names(),
                    help='which shape the rig is pushing. Everything that differs per '
                         'shape -- mesh, dataset, checkpoint, physical size, marker '
                         'positions, goal pose -- comes from shapes.SHAPES[this]. Must be '
                         'the same shape the checkpoint was trained on.')
    return ap


def main():
    """``python shapes.py`` -- what every shape resolves to, and what is missing."""
    import argparse

    ap = argparse.ArgumentParser(description="the shape registry, printed")
    ap.add_argument("shape", nargs="?", help="only this one (default: all)")
    cli = ap.parse_args()

    want = [resolve(cli.shape)] if cli.shape else list(SHAPES.values())
    for spec in want:
        mark = " (active)" if spec.name == active_name() else ""
        print(f"\n=== {spec.name}{mark} " + "=" * (60 - len(spec.name)))
        try:
            print(spec.describe())
        except (OSError, ValueError) as exc:
            print(f"  cannot read geometry: {exc}")
        for problem in spec.check():
            print(f"  !! {problem}")
        for i in sorted(spec.markers):
            p = spec.markers[i]
            if p is None:
                print(f"  marker {i}: NOT MEASURED")
                continue
            b = spec.topleft_to_body_cm(p[0], p[1])
            ok = spec.on_shape(b[0], b[1], tol_cm=0.6)
            print(f"  marker {i}: top-left ({p[0]:+.2f}, {p[1]:+.2f}) cm -> body "
                  f"({b[0]:+.2f}, {b[1]:+.2f}) cm  {'on the shape' if ok else 'OFF THE SHAPE'}")
    print()


if __name__ == "__main__":
    main()
