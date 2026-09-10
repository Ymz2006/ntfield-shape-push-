"""Push the T along an Eikonal-planner trajectory with a single cylindrical pusher.

The planner (``models/metric`` + ``Experiments/3dshape/.../latest.pt``) produces an SE(2)
path for the T itself -- it says nothing about a robot.  This demo closes that gap: it
takes the planned T path as a *reference* and solves, at every frame, the inverse problem
"where must the cylinder be so that the T moves the way the plan wants it to".

Three stages:

1. **Plan** (``plan_tee_path``).  Load the checkpoint, take one start/goal pair out of the
   test set, and run the planar MPPI rollout on the travel-time field.  Output is a
   (T, 3) array of SE(2) poses in the planner's normalized frame.

2. **Lift into the sim frame** (``planner_to_world``).  The planner works in
   ``(p - env_center) / env_scale`` on the *z-up* meshes and poses the shape about its
   bounding-box centre.  ``pymunk_viser_push`` works in raw mesh units on the *y-up*
   meshes mapped through MESH_TO_WORLD, and poses the T about its footprint *centroid*
   (the pymunk body origin).  Those two frames coincide up to the centroid offset --
   MESH_TO_WORLD is exactly the y-up -> z-up rotation -- so the lift is
   ``x_world = x_norm * env_scale + env_center`` plus ``R(theta) @ (centroid - bbox_centre)``.

3. **Discrete push primitives** (``PushPrimitives`` + ``PushController``).  An action is
   a (boundary point, push direction, push length) triple: put the cylinder behind that
   point and drive it straight for that far.  Every action in the library is MEASURED
   once by rolling it out in an empty arena, so the forward model is exact by
   construction -- the analytic quasi-static screw model over-predicts rotation about 9x
   and cannot place the object (see ``PushPrimitives.calibrate``).

   Choosing one is then an argmin: which action leaves the smallest residual pose error
   against the chunk of reference path ``--action-steps`` ahead.  Rotation enters that
   cost in length units via ``c``, the limit surface's length scale, which trades rotation
   against translation.  pymunk's top-down friction is a max-force / max-torque pair, so
   the sim's own anisotropy is ``spin_friction / friction``; that is the default, not the
   shape's radius of gyration.

   Searching the push LENGTH alongside the contact is what makes the last few actions
   converge: far from the goal the demand is a full chunk of path and a long push wins, and
   as the T closes in every long push overshoots, so the short end of the ladder wins on
   cost with no distance schedule to tune.

The loop is closed at two levels.  Per action, the next one is chosen from the T's measured
pose, so modelling error is corrected rather than accumulated.  Per few seconds, the
reference itself is re-planned from that pose (``Replanner``).

There are two modes, and they differ only in how the cylinder REACHES each action's entry
point: by default it is teleported there, which is free but unphysical, and under
``--teleport-interp`` it flies.  A flown transit is always three legs -- off the surface,
across, back down -- with no straight-line shortcut, and the middle leg has two forms.
Between two points standing off the SAME face of the footprint it is a slide along that
face, lifted clear of it; otherwise it is an arc about the object, and the arc's radius is
searched, the tightest circle clearing the shape and the environment winning, so the pusher
hugs the T where it can and swings out to the enclosing circle only where it must.

Usage (from the ntrl-demo root):
    python sim/push_t_demo.py                        # http://localhost:8080
    python sim/push_t_demo.py --case 7 --port 8081
    python sim/push_t_demo.py --save-path plan.npy   # dump the planned T path and exit
    python sim/push_t_demo.py --traj plan.npy        # replay a saved path (no torch needed)
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

sys.path.append('.')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pymunk
import trimesh
import viser
from shapely.affinity import rotate, translate
from shapely.geometry import LineString, Polygon, Point
from shapely.geometry.polygon import orient
from shapely.ops import nearest_points

import pymunk_viser_push as base

COL_REF = (120, 200, 120)
COL_ACTUAL = (232, 138, 62)
COL_CONTACT = (220, 60, 60)
COL_GHOST = (150, 210, 150)
COL_STANDOFF = (150, 150, 165)      # the stand-off circle transits ride
COL_EXIT = (240, 200, 60)           # where a transit leaves the shape
COL_ENTRY = (90, 200, 230)          # where it comes back down to the contact
COL_TRANSIT = (240, 200, 60)        # the planned transit path itself
COL_SAMPLE = (210, 120, 245)        # contact points the primitives actually sample

# Candidate escape directions tried when marching a point out to the stand-off circle,
# fanned either side of the surface normal.  37 is one every 5 degrees over a half turn.
EXIT_DIRS = 37


# ======================================================================================
# 1. planning
# ======================================================================================
def planar_mppi(womodel, xp, steps=200, sample_num=50, horizon=5, step=0.015):
    """Single-episode planar MPPI on the travel-time field -> (T, 6) normalized path.

    This mirrors ``evaluate_training_3d_batched.MPPI_batched`` with B = 1 and ``planar``
    always on: same 50-sample cloud, same 10*first + last cost, same softmax(-50*cost)
    weighting, same 0.01 convergence ball.  The displacement is masked onto (x, y, rz) so
    the rollout stays in the sub-space ``preprocess_obj.py --2d`` sampled.
    """
    import torch

    dim = 6
    dev = xp.device
    free = torch.zeros(dim, device=dev)
    free[[0, 1, 5]] = 1.0

    out = [xp[:, 0:dim].clone()]
    for _ in range(steps):
        tmp = xp.clone()[:, None, None, :].repeat(1, sample_num, horizon, 1)
        dP = step * torch.normal(0, 1, size=(1, sample_num, 1, dim), device=dev) \
            + step * torch.normal(0, 1, size=(1, sample_num, horizon, dim), device=dev)
        dP = dP * free
        dP = dP / (torch.clamp(torch.norm(dP, dim=3, keepdim=True), min=step) / step)
        tmp[..., 0:dim] = tmp[..., 0:dim] + torch.cumsum(dP, dim=2)

        ends = tmp[:, :, [0, -1], :]
        cost = womodel.function.TravelTimes(ends.reshape(-1, dim * 2)).reshape(1, sample_num, 2)
        cost = 10 * cost[:, :, 0] + cost[:, :, 1]
        weight = torch.softmax(-50 * cost, dim=1)
        xp[:, 0:dim] = xp[:, 0:dim] + torch.bmm(weight.unsqueeze(1), dP[:, :, 0, :]).squeeze(1)

        out.append(xp[:, 0:dim].clone())
        if torch.norm(xp[:, dim:2 * dim] - xp[:, 0:dim]) < 0.01:
            break
    out.append(xp[:, dim:2 * dim].clone())        # snap the last waypoint onto the goal
    return torch.cat(out, dim=0).detach().cpu().numpy()


def load_planner(data_path, model_path, ckpt, case, device):
    """-> (womodel, start_norm(6), goal_norm(6)).  Kept alive so the demo can replan."""
    import torch
    from models.metric import model_train_metric as md

    womodel = md.Model(model_path, data_path, 6, [0.0] * 6, device=device)
    print(f'[plan] checkpoint {ckpt}')
    womodel.load(ckpt)
    womodel.network.eval()

    arr = np.load(os.path.join(data_path, 'sampled_points.npy'))
    if case >= len(arr):
        raise SystemExit(f'--case {case} out of range (test set has {len(arr)} cases)')
    return womodel, arr[case][:6].copy(), arr[case][6:].copy()


def plan_from(womodel, start_norm, goal_norm, device, steps):
    """MPPI from an arbitrary start -> (T, 3) SE(2) path in NORMALIZED coords.

    Taking the start as an argument rather than reading it out of the test set is what
    makes replanning possible: the demo feeds in wherever the T actually IS.
    """
    import torch

    xp = torch.tensor(np.concatenate([start_norm, goal_norm]), dtype=torch.float32,
                      device=device).reshape(1, 12)
    with torch.no_grad():
        path = planar_mppi(womodel, xp, steps=steps)
    return path[:, [0, 1, 5]], float(np.linalg.norm(path[-2, [0, 1, 5]]
                                                    - np.asarray(goal_norm)[[0, 1, 5]]))


# ======================================================================================
# 2. planner frame -> sim world frame
# ======================================================================================
def planner_to_world(path_norm, env_scale, env_center, bbox_to_centroid):
    """(T,3) normalized planner poses -> (T,3) world SE(2) poses of the pymunk body.

    The planner poses the shape about its bounding-box centre; the pymunk body origin is
    the footprint centroid.  ``bbox_to_centroid`` is that offset in the shape's local
    frame, so it rotates with the body.
    """
    xy = path_norm[:, :2] * env_scale + np.asarray(env_center)[:2]
    th = path_norm[:, 2] * 2.0 * math.pi
    c, s = np.cos(th), np.sin(th)
    d = np.asarray(bbox_to_centroid)
    xy = xy + np.stack([c * d[0] - s * d[1], s * d[0] + c * d[1]], axis=1)
    return np.column_stack([xy, np.unwrap(th)])


def world_to_planner(pose_world, env_scale, env_center, bbox_to_centroid):
    """Inverse of ``planner_to_world`` for one pose -> normalized (x, y, z, rx, ry, rz).

    The replanner needs this: the T's live pose comes out of pymunk in world units about
    the footprint centroid, and the network wants normalized coordinates about the
    bounding-box centre.
    """
    x, y, th = float(pose_world[0]), float(pose_world[1]), float(pose_world[2])
    c, s = math.cos(th), math.sin(th)
    d = np.asarray(bbox_to_centroid)
    xy = np.array([x, y]) - np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])
    xy = (xy - np.asarray(env_center)[:2]) / env_scale
    return np.array([xy[0], xy[1], 0.0, 0.0, 0.0, wrap(th) / (2.0 * math.pi)])


def resample_path(path, spacing, smooth):
    """Uniform arc-length resample + light smoothing of a (T,3) SE(2) path.

    The raw rollout is stochastic and locally jittery; a pushing controller chasing that
    jitter would thrash the contact.  Arc length is measured in position only, with the
    unwrapped heading carried along.
    """
    seg = np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    keep = np.concatenate([[True], seg > 1e-9])
    s, path = s[keep], path[keep]
    if s[-1] < 1e-9:
        return path
    n = max(int(s[-1] / spacing) + 1, 2)
    su = np.linspace(0.0, s[-1], n)
    out = np.stack([np.interp(su, s, path[:, k]) for k in range(3)], axis=1)
    # np.convolve(..., mode='same') returns max(len(signal), len(kernel)), so a kernel
    # wider than the path yields an oversized array.  Replanned paths get very short as
    # the T nears the goal, which is exactly when this bites.
    smooth = min(int(smooth), len(out))
    if smooth > 1 and len(out) > 2:
        # Normalized (edge-shrinking) boxcar: dividing by the same kernel convolved with
        # ones makes the window narrow at the ends instead of being fed invented samples.
        # Constant-padding here would bias a ramp inward -- it dragged the first waypoint
        # a full half-window along the path, and ref[0] / ref[-1] are exactly the poses
        # the planner certified collision-free, so they have to survive untouched.
        k = np.ones(smooth)
        norm = np.convolve(np.ones(len(out)), k, mode='same')
        ends = out[[0, -1]].copy()
        for c in range(3):
            out[:, c] = np.convolve(out[:, c], k, mode='same') / norm
        out[0], out[-1] = ends
    return out


# ======================================================================================
# 3. push inverse kinematics
# ======================================================================================
def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class PushIK:
    """The T's boundary geometry, plus the limit surface's length scale ``c``.

    The boundary is sampled once into ``P`` (points, body frame) with outward normals
    ``N``; ``radius`` is the smallest circle about the body origin that encloses it, which
    is what the stand-off circle is built from.

    ``c`` trades rotation against translation under the ellipsoidal limit surface: a
    wrench ``(fx, fy, tau)`` produces a twist proportional to ``(fx, fy, tau / c^2)``, so
    multiplying an angle by ``c`` puts it in length units and makes position and heading
    error directly comparable.  Every cost in this file that mixes the two uses it.

    This used to also solve the inverse problem -- score every (boundary point, in-cone
    direction) pair against a desired twist and take the best.  That analytic solve only
    ever fed the continuous-contact and direct-wrench modes, both of which are gone; the
    push primitives fan their directions by ``--dir-spread`` and get their forward model
    by measurement instead, so the friction cone is not in the surviving path at all.
    """

    def __init__(self, poly, c_length, n_boundary=360):
        ring = np.asarray(orient(poly, 1.0).exterior.coords)[:-1]     # CCW
        edge = np.roll(ring, -1, axis=0) - ring
        length = np.linalg.norm(edge, axis=1)
        per = np.maximum((n_boundary * length / length.sum()).astype(int), 2)

        pts, nrm = [], []
        for i in range(len(ring)):
            t = np.linspace(0.0, 1.0, per[i], endpoint=False)[:, None]
            pts.append(ring[i] + t * edge[i])
            e = edge[i] / length[i]
            nrm.append(np.tile([e[1], -e[0]], (per[i], 1)))           # CCW -> outward
        self.P = np.concatenate(pts)                                  # (N, 2)
        self.N = np.concatenate(nrm)                                  # (N, 2)
        self.ring = ring                                              # (V, 2) CCW corners

        self.c = float(c_length)
        self.radius = float(np.linalg.norm(self.P, axis=1).max())

    def edge_normal(self, i):
        """Outward unit normal of footprint edge ``i``."""
        e = self.ring[(i + 1) % len(self.ring)] - self.ring[i]
        e = e / np.linalg.norm(e)
        return np.array([e[1], -e[0]])                                # CCW -> outward

    def edge_of(self, q):
        """Index of the footprint EDGE nearest ``q``, or -1 if ``q`` is behind it.

        The footprint's REAL edges, not the ``P`` resampling: ``tee_shape`` is built from
        several hundred boundary samples, so its own segments are a fraction of a unit
        long and no two stand-off points would ever share one.  What matters to a transit
        is the flat face the pusher is standing off, which is a segment of ``ring``.

        A point on the inner side of that face is behind it rather than standing off it,
        and gets -1: whatever it is doing, sliding along the face is not it.
        """
        q = np.asarray(q, dtype=float)
        a = self.ring
        e = np.roll(a, -1, axis=0) - a
        t = np.clip(np.einsum('ij,ij->i', q - a, e)
                    / np.maximum(np.einsum('ij,ij->i', e, e), 1e-12), 0.0, 1.0)
        i = int(np.argmin(np.linalg.norm(q - (a + t[:, None] * e), axis=1)))
        return i if float((q - a[i]) @ self.edge_normal(i)) > 1e-9 else -1


def _circle_point(start, direction, radius):
    """First point at ``radius`` from the origin along the ray ``start + t*direction``.

    Solved rather than marched, because the ladder of arc radii now reaches well inside
    the object as well as outside it: when ``start`` is already further out than the
    circle the ray meets it only if aimed inwards, and at two values of ``t`` rather than
    one.  Returns the nearer of those (or ``None`` when the ray misses the circle), so the
    leg is the shortest one that direction can offer.
    """
    q = np.asarray(start, dtype=float)
    d = np.asarray(direction, dtype=float)
    d = d / (np.linalg.norm(d) + 1e-12)
    b = float(q @ d)
    disc = b * b - (float(q @ q) - radius * radius)
    if disc < 0.0:
        return None
    s = math.sqrt(disc)
    for t in (-b - s, -b + s):
        if t >= -1e-9:
            return q + max(t, 0.0) * d
    return None


def to_world(v, pos, ang, is_vector=False):
    c, s = math.cos(ang), math.sin(ang)
    r = np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])
    return r if is_vector else r + np.asarray(pos)


def to_local(v, pos, ang, is_vector=False):
    r = np.asarray(v, dtype=float) - (0.0 if is_vector else np.asarray(pos))
    c, s = math.cos(-ang), math.sin(-ang)
    return np.array([c * r[0] - s * r[1], s * r[0] + c * r[1]])


def wrap_arr(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def push_length_ladder(push_len, min_len, steps):
    """Geometrically spaced push lengths from ``min_len`` up to ``push_len``.

    Geometric rather than linear because what matters is the *ratio* between consecutive
    rungs: the selector switches rungs when the demanded chunk of path halves, and a
    linear ladder would spend all its resolution at the long end where it is least needed.
    """
    steps = max(int(steps), 1)
    lo = max(min(float(min_len), float(push_len)), 0.5)
    if steps == 1 or lo >= push_len - 1e-9:
        return [float(push_len)]
    return list(np.geomspace(lo, float(push_len), steps))


class PushPrimitives:
    """A discrete library of push actions, plus a closed-form model of what each one does.

    An action is a pair (contact point on the boundary, push direction).  Contacts are
    sampled evenly along the perimeter; directions are the inward normal -- the "centre
    line" at that point -- rotated through a fan of +-``spread``.  Executing one means:
    teleport the pusher behind that contact, then drive it straight for a fixed distance.

    **Forward model.**  Under the quasi-static limit surface a unit force ``u`` at ``p``
    gives the twist ``(u_x, u_y, cross(p,u)/c^2)``.  Because the contact is carried along
    by the object, that twist is constant in the BODY frame, which makes the resulting
    motion an exact screw -- a circular arc in the world, not a straight line.  Integrating
    it properly matters here: over a 40-unit push a primitive with a big moment arm turns
    the object tens of degrees, and a first-order ``dx = s*u`` estimate would put the
    predicted pose in the wrong place entirely.

    Parameterising by translation arc length ``s``:

        kappa  = cross(p, u) / c^2          radians of rotation per unit of translation
        dtheta = s * kappa
        dxy    = SE(2) exponential of that twist, in the body frame

    The body-frame delta depends only on the primitive and ``s``, never on where the object
    currently is, so the whole table is precomputed once and selection is an argmin.
    """

    def __init__(self, ik, n_points=24, n_dirs=5, spread_deg=45.0, corner_margin=0.0):
        # Contacts are spread evenly over the boundary.  ``corner_margin`` (world units,
        # 0 = off) first strikes out every boundary sample within that distance of a
        # footprint CORNER: a disc pressed against a corner touches an edge rather than a
        # face, so the measured primitive there is the least trustworthy one in the table
        # and on hardware it is the one that slips off the shape.  The margin is measured
        # from the contact point itself; the disc centre stands off along the outward
        # normal, so it is at least this far from the corner too.  Whatever survives is
        # then sampled evenly, so ``n_points`` contacts still spread over the whole of the
        # remaining boundary rather than bunching at one end.
        keep = np.arange(len(ik.P))
        if corner_margin > 0.0:
            d = np.linalg.norm(ik.P[:, None, :] - ik.ring[None, :, :], axis=2).min(axis=1)
            keep = np.flatnonzero(d >= float(corner_margin))
        assert len(keep) >= n_points, (
            f'only {len(keep)} of {len(ik.P)} boundary samples are usable as contacts'
            + (f' (the {corner_margin:.2f} corner margin strikes out the rest)'
               if corner_margin > 0.0 else '')
            + '; ask for fewer --n-contacts or raise --n-boundary')
        # Spread the contacts over ALL of the eligible boundary.  A stride of
        # ``len(keep) // n_points`` truncated to ``n_points`` is not the same thing: ask
        # for 100 of 360 and the stride is 3, which runs off the end of the boundary after
        # 300 samples and leaves the last sixth of the perimeter with no contact at all.
        idx = keep[np.round(np.linspace(0.0, len(keep), n_points,
                                        endpoint=False)).astype(int)]
        P, Nrm = ik.P[idx], ik.N[idx]
        self.corner_margin = float(corner_margin)
        self.n_boundary_kept = len(keep)

        ang = np.radians(np.linspace(-spread_deg, spread_deg, n_dirs))
        ca, sa = np.cos(ang), np.sin(ang)
        inward = -Nrm
        U = np.stack([inward[:, None, 0] * ca - inward[:, None, 1] * sa,
                      inward[:, None, 0] * sa + inward[:, None, 1] * ca], axis=2)

        self.P = np.repeat(P[:, None, :], n_dirs, axis=1).reshape(-1, 2)
        self.U = U.reshape(-1, 2)
        self.N = np.repeat(Nrm[:, None, :], n_dirs, axis=1).reshape(-1, 2)
        self.contact_idx = np.repeat(idx[:, None], n_dirs, axis=1).reshape(-1)
        self.c = float(ik.c)
        self.n_points, self.n_dirs = len(idx), n_dirs
        self.spread_deg = float(spread_deg)
        # radians of rotation per unit of translation
        self.kappa = (self.P[:, 0] * self.U[:, 1]
                      - self.P[:, 1] * self.U[:, 0]) / (self.c ** 2)

        # Measured forward model, filled in by calibrate().  One (M, 3) table per push
        # length: the same primitive is a different action depending on how far it drives.
        self.lengths = None          # (nL,) ascending push lengths, world units
        self.tables = None           # (nL, M, 3) body-frame pose delta of each action

    def __len__(self):
        return len(self.P)

    def predict(self, s):
        """(M, 3) body-frame pose delta after a push of translation arc length ``s``."""
        th = s * self.kappa
        tiny = np.abs(self.kappa) < 1e-12
        k = np.where(tiny, 1.0, self.kappa)
        A = np.where(tiny, s, np.sin(th) / k)              # -> s as kappa -> 0
        B = np.where(tiny, 0.0, (1.0 - np.cos(th)) / k)    # -> 0 as kappa -> 0
        return np.column_stack([A * self.U[:, 0] - B * self.U[:, 1],
                                B * self.U[:, 0] + A * self.U[:, 1],
                                th])

    def calibrate(self, args, tee_poly, push_lens, cache_dir='.', gap=2.0):
        """Measure each primitive by executing it, and use that as the forward model.

        The analytic ``predict`` is not good enough to plan with.  Measured against the
        simulator it over-predicts rotation by roughly 9x, and the measured rotation is
        near-constant in the push length (~5 deg at L = 10, 20 and 40 alike) rather than
        growing with it.  The reason is that the screw model assumes the body keeps turning
        under a constant body twist, whereas a rigid single-point push turns the object only
        until the contact normal lines up with the push direction and thereafter simply
        slides it.  Rank correlation stays high (+0.85), so the analytic model knows which
        primitives rotate more, but its magnitudes cannot place the object.

        So each primitive is rolled out once in an empty arena and the resulting body-frame
        delta is recorded.  Exact by construction, and it costs one short rollout per
        primitive, cached to disk against the parameters that affect it.

        A whole ladder of push lengths is measured, not just one.  Selection then picks the
        length as well as the contact, which is what lets the action shrink as the T closes
        on the goal: far away the demanded chunk of path is ~40 units and a long push wins;
        near the goal the demand is a couple of units and every long push overshoots it, so
        the short end of the ladder wins on cost with no distance schedule to tune.
        """
        lens = sorted({round(float(L), 3) for L in np.atleast_1d(push_lens)})
        self.lengths = np.asarray(lens, dtype=float)
        self.tables = np.stack([
            self._calibrate_one(args, tee_poly, L, cache_dir, gap) for L in lens])
        return self.tables

    def _calibrate_one(self, args, tee_poly, push_len, cache_dir, gap):
        """Measure every primitive at one push length (cached to disk)."""
        # The action geometry itself, not the knobs that produced it.  n_points, n_dirs
        # and --dir-spread do not pin it down: the contacts are struck off ``ik.P``, so
        # --n-boundary moves them too, and a key listing only the knobs it happens to know
        # about will silently load a table measured for different contacts.
        geom = hashlib.md5(
            np.round(np.column_stack([self.P, self.U]), 6).tobytes()).hexdigest()[:12]
        key = (os.path.basename(str(args.shape)), round(push_len, 3), geom,
               round(args.pusher_radius, 3), round(args.friction, 3),
               round(args.spin_friction, 3), round(args.tee_mass, 4),
               round(args.pusher_speed, 3), int(args.fps), int(args.substeps),
               # The contact surfaces change what the rollout measures, so they belong in
               # the key -- otherwise editing SURF_PUSHER silently reuses a stale table.
               base.SURF_PUSHER, base.SURF_TEE)
        tag = hashlib.md5(repr(key).encode()).hexdigest()[:12]
        path = os.path.join(cache_dir, f'.push_primitives_{tag}.npy')
        if os.path.exists(path):
            out = np.load(path)
            print(f'[primitives] loaded measured model for {len(self)} actions '
                  f'(L={push_len:.1f}) from {os.path.basename(path)}')
            return out

        print(f'[primitives] measuring {len(self)} actions at L={push_len:.1f} '
              f'(one rollout each; cached afterwards)...')
        arena = [Polygon([(-6000, -6000), (6000, -6000), (6000, 6000), (-6000, 6000)])]
        t0 = time.time()
        out = np.zeros((len(self), 3))
        for i in range(len(self)):
            sim = PushSim(args, arena, tee_poly, (0.0, 0.0), 0.0,
                          (-3000.0, -3000.0))
            p_l, u_l = self.P[i], self.U[i]
            start = p_l - u_l * (args.pusher_radius + gap)
            sim.pusher.position = (float(start[0]), float(start[1]))
            sim.target = pymunk.Vec2d(*(start + u_l * (gap + push_len + args.pusher_radius)))
            sim.max_speed = args.pusher_speed
            dt = 1.0 / args.fps
            for _ in range(int(6.0 * args.fps)):
                for _ in range(args.substeps):
                    sim.step(dt / args.substeps)
                if (pymunk.Vec2d(*sim.target) - sim.pusher.position).length < 0.5:
                    break
            out[i] = (sim.tee.position.x, sim.tee.position.y, wrap(sim.tee.angle))
        np.save(path, out)
        print(f'[primitives] done in {time.time() - t0:.0f}s; '
              f'|dxy| median {np.median(np.linalg.norm(out[:, :2], axis=1)):.1f}, '
              f'|dtheta| median {math.degrees(np.median(np.abs(out[:, 2]))):.1f} deg '
              f'-> {os.path.basename(path)}')
        return out

    def select(self, delta_body, s=None, ang_weight=None, mask=None, len_bias=0.0):
        """Best (primitive, push length) for a desired body-frame pose delta.

        Returns ``(index, length, predicted delta)``.  Rotation is put in length units with
        the same ``c`` the IK uses, so position and heading error trade off consistently
        with everything else in this file.  Uses the MEASURED tables when calibrated, and
        the analytic screw model only as a fallback -- see ``calibrate`` for why the
        analytic one cannot be planned with.

        Searching over the length ladder as well as the contact is what tapers the push:
        the cost is the residual pose error the action would leave behind, so an action
        that overshoots a small demand is penalised exactly as much as one that undershoots
        a large one.  ``len_bias`` adds a small preference for the long end so that far
        from the goal the controller does not dawdle through many tiny actions when a few
        big ones would do.
        """
        if self.tables is not None:
            preds, lens = self.tables, self.lengths
        else:
            lens = np.asarray([s if s is not None else 1.0], dtype=float)
            preds = self.predict(float(lens[0]))[None]
        w = self.c if ang_weight is None else ang_weight
        d = np.asarray(delta_body, dtype=float)
        cost = (np.linalg.norm(preds[:, :, :2] - d[None, None, :2], axis=2)
                + w * np.abs(wrap_arr(preds[:, :, 2] - d[2])))
        if len_bias:
            cost = cost + len_bias * (lens.max() - lens)[:, None]
        if mask is not None:
            cost = np.where(np.asarray(mask)[None, :], cost, np.inf)
        li, i = np.unravel_index(int(np.argmin(cost)), cost.shape)
        if not np.isfinite(cost[li, i]):
            return None, None, None
        return int(i), float(lens[li]), preds[li, i]


# ======================================================================================
# 4. closed-loop tracker
# ======================================================================================
class PushController:
    """Chooses the next discrete push action, and plans how the pusher gets to it.

    One decision per ACTION, not per frame: ``begin_primitive`` picks a (contact, push
    direction, push length) triple from the measured primitive table against the chunk of
    reference path ``--action-steps`` ahead, and nothing is re-decided until that push
    finishes.  Closed-loop-ness comes from the next action being chosen off the T's
    *measured* pose, so per-action modelling error is corrected rather than accumulated.

    How the pusher reaches the action's entry point is the mode switch.  By default it is
    teleported there, which is free but unphysical.  Under ``--teleport-interp`` it flies
    instead, and ``plan_transit`` builds the route -- always three legs, off the surface,
    across, back down.  Across is a slide along one face when both ends stand off the same
    one, and otherwise an arc about the object at the tightest radius that stays clear.

    Transit waypoints are stored in the T's LOCAL frame, so if the T drifts while the
    pusher is travelling the chain follows the object instead of aiming at a stale spot in
    the world.
    """

    def __init__(self, ik, ref, args, obstacles):
        self.ik = ik
        self.ref = ref                                  # (T, 3) world SE(2), live
        self.ref0 = ref                                 # the path we started from
        self.a = args
        self.obstacles = obstacles
        # Keep-out for transit planning: the T grown by the WHOLE pusher radius, i.e. the
        # region the pusher centre may not enter without the disc touching the object.
        # (The old continuous mode used a deliberately undersized buffer here so that a
        # pressing pusher was not inside its own forbidden zone.  That is far too
        # permissive for a transit -- a line clearing it by a hair still buries the disc.)
        self.transit_free = args.pusher_radius + 0.5 * args.clearance
        self.no_sweep = Polygon(ik.P).buffer(self.transit_free)
        # Candidate arc radii, tightest first.  The ladder runs from a circle far smaller
        # than the object -- most of it buried in the T, and rejected on sight -- up to the
        # one enclosing the object plus the pusher disc plus --standoff, which is the only
        # rung guaranteed to clear the shape from every angle and so is always the last
        # resort.  The tight rungs are not wasted: an arc only has to be clear over the
        # sweep it actually flies, so a short hop across a notch takes a circle that cuts
        # well inside the object's extent and the pusher brushes round it instead of
        # swinging out of its way.  Spacing is linear.  The old ladder was geometric, which
        # spent its resolution at the tight end; that was right when every rung was a
        # margin OUTSIDE the object, and is exactly wrong now that the tight end is inside
        # it and the rungs that decide anything are the wide ones.
        hi = ik.radius + args.pusher_radius + float(args.standoff)
        lo = min(args.arc_radius_min if args.arc_radius_min else 0.125 * args.pusher_radius,
                 hi)
        n = max(int(args.arc_radii), 1)
        self.radii = (np.linspace(lo, hi, n) if n > 1 and hi > lo + 1e-9
                      else np.array([hi]))
        self.standoff = float(self.radii[-1])           # widest, for parking and the GUI
        self.transit_radius = self.standoff             # the one the last transit used
        self.transit_kind = 'arc'                       # or 'edge', for the GUI/report
        # Whether the last arc actually flew the --arc-clearance margin, or fell back to
        # the tight rung because the wider circle came back blocked.  Recorded, never read
        # by the planner: without it a radius in the log cannot be told from the rung the
        # ladder found, which is the one thing you want to know when the margin is not
        # doing anything.  Always False for an edge transit -- the margin never applies.
        self.transit_widened = False
        # Whether the last transit COULD have been a same-face slide and was flown as an
        # arc anyway -- both ends off one face, but the slide itself came back blocked.
        # Only ever True on an arc; a slide that was flown is ``transit_kind == 'edge'``.
        # Nothing plans off it, it is there so a run can say WHY it went the long way.
        self.edge_blocked = False
        self._fly_mask = None                           # which primitives can be flown to
        # How far a same-edge transit lifts off the face before sliding along it.  The
        # slide is travel, so it is tested against ``no_sweep`` -- the footprint grown by
        # ``transit_free``, the whole disc plus half the clearance -- and a lift that
        # leaves either end inside THAT is rejected however clear of the face it looks.
        # The pusher finishes a push pressed against the face, one radius off it, so the
        # lift has to make up the difference (``transit_free - pusher_radius``, i.e. half
        # the clearance) and then buy some air: half a disc radius on top, which is what
        # the lift used to be on its own.
        #
        # That distinction was free in the sim, where the clearance is one unit against a
        # five-unit disc, and is not on the table: there the clearance is 2 cm against a
        # 1.5 cm radius, so half a radius of lift left the slide 2.5 mm INSIDE the keep-out
        # and every same-face transit fell back to the arc.  See ``_edge_transit``, which
        # raises the lift again when the pusher is deeper against the face than nominal.
        self.edge_lift = float(args.edge_lift if args.edge_lift
                               else 0.5 * (args.pusher_radius + args.clearance))
        self.tee_shape = Polygon(ik.P)                  # body frame, for clearance queries
        self.reset()

    def reset(self):
        self.ref = self.ref0                            # drop any replanned path
        self.k = 0                                      # reference index (monotonic)
        self.prev_idx = None
        self.contact_world = None
        self.done = False
        self.n_switch = 0           # actions committed, i.e. contact changes
        self.last_len = None        # push length of the most recent primitive
        self.entry_local = None     # body-frame entry point / end of the push
        self.final_local = None
        self.push_dir_local = None  # body-frame push direction of the pending action
        self.transit_marks = None   # (exit, entry) either side of the transit, for the GUI
        self.edge_blocked = False   # last arc that a same-face slide was available for
        self.last_pos = None

    def adopt(self, ref):
        """Replace the reference with a freshly planned one and re-anchor on it.

        The new path starts from where the T was when the replan was requested, which is
        slightly behind where it is now, so the carrot is re-seeded by *proximity* rather
        than reset to 0 -- otherwise the controller spends the first moments after every
        replan driving backwards to a waypoint it has already passed.
        """
        self.ref = ref
        self.k = int(np.argmin(np.linalg.norm(ref[:, :2] - self.last_pos, axis=1))) \
            if self.last_pos is not None else 0

    # -- reference tracking ----------------------------------------------------------
    def _advance(self, pos):
        """Carrot: advance the reference index by monotonic nearest-waypoint projection.

        This used to advance only while the T was within a fixed radius of the current
        waypoint, which silently breaks as soon as the object can move further than that
        radius in one control decision.  A discrete push action displaces the T by ~37
        units against a 9-unit threshold, so the carrot never advanced at all and the whole
        run tracked waypoint 1.  Projecting onto the nearest waypoint instead has no such
        scale assumption, and staying monotonic (searching only forward, within a window)
        keeps it from snapping backwards where the path crosses itself.
        """
        n = len(self.ref)
        hi = min(n, self.k + self.a.advance_window + 1)
        j = int(np.argmin(np.linalg.norm(self.ref[self.k:hi, :2] - pos, axis=1)))
        self.k = min(self.k + j, n - 1)
        # Where the T was when the index last moved, so a landing replan can re-anchor on
        # proximity instead of restarting the path at 0.
        self.last_pos = pos.copy()
        return self.k

    # -- geometry queries ------------------------------------------------------------
    def _free(self, local_pt, pos, ang):
        """Is the pusher centre at this LOCAL point clear of the environment and the T?

        The T check matters for the primitive start points: a direction near the edge of
        the ``--dir-spread`` fan at a concave contact puts the entry point INSIDE the
        object, which is not a pose the pusher can occupy at all.  The disc is allowed to
        touch the T -- it is about to push it -- but its centre may not be within it.
        """
        w = Point(to_world(local_pt, pos, ang))
        if self.tee_shape.contains(Point(np.asarray(local_pt, dtype=float))):
            return False
        # Exact point-to-polygon distance, rather than buffering the point into a 65-gon
        # and intersecting: same answer, less work, and this runs up to 60x per frame.
        return all(o.distance(w) > self.a.pusher_radius for o in self.obstacles)

    # -- transit planning (--teleport-interp) ----------------------------------------
    def _path_cost(self, lead, sweep, pos, ang):
        """How badly this transit collides.  0.0 means clean.

        Whether a transit is clean is all this decides on its own; only the RANKING of
        blocked ones depends on the scale of the two terms.  The exchange rate between
        them is one wall hit per world unit of pusher buried in the object, which keeps a
        hair-thin graze of the T preferable to driving at a wall while making a real
        incursion -- a leg crossing the object rather than brushing it, which the tight
        rungs of the radius ladder are perfectly capable of producing -- worse than any
        number of walls.  Burial is not a scrape: transit waypoints are body-frame, so a
        pusher inside the T shoves it, the target moves with it, and the leg never
        converges at all.

        The obstacle term is measured over the whole SWEPT path rather than its waypoints:
        at ``--arc-step`` degrees on a wide radius consecutive waypoints are further apart
        than the pusher is across, so a per-point test walks straight over thin geometry
        between samples.

        The object is tested twice over, because the two halves of a transit are not
        doing the same thing:

        * the RETRACT leg starts wherever the pusher is, which after a push is pressed
          against the object and inside its keep-out by construction.  Holding it to the
          keep-out would fail every time, so it is held to the weaker test of not running
          THROUGH the object.
        * the SWEEP is everything after that -- the arc or the same-edge slide, plus the
          re-entry leg down to the hand-over point -- and it is pure travel, so the whole
          pusher DISC has to stay off the object: it is tested against ``no_sweep``, the T
          grown by that disc.  A circle inside the T's bounding radius, a slide hugging one
          face, or a re-entry aimed at a pose the disc cannot occupy can all be clear of
          the *polygon* and still drag the object along.  That is what ``_handover`` buys:
          with the flight ending clear of the keep-out, the whole of it can be required to
          stay clear, instead of the last leg being exempt and shoving the T on the way in.
        """
        def line(pts):
            pts = [np.asarray(q, dtype=float) for q in pts]
            pts = [q for i, q in enumerate(pts)                  # drop repeated waypoints;
                   if i == 0 or np.linalg.norm(q - pts[i - 1]) > 1e-9]   # LineString needs 2+
            return LineString(pts) if len(pts) > 1 else None

        whole = line(list(lead) + list(sweep))
        if whole is None:
            return 0.0
        world = LineString([to_world(q, pos, ang) for q in whole.coords])
        hits = sum(1 for o in self.obstacles
                   if world.distance(o) < self.a.pusher_radius + self.a.clearance)
        # The object is fixed in the body frame, so it is tested there directly.
        buried = 0.0
        ln = line(lead)
        if ln is not None:
            buried += ln.intersection(self.tee_shape).length
        ln = line(sweep)
        if ln is not None:
            buried += ln.intersection(self.no_sweep).length
        return float(hits) + buried

    @staticmethod
    def _sweep(a0, a1, direction):
        """Signed angle from ``a0`` to ``a1`` going ``direction`` (+1 CCW, -1 CW), rad.

        The two directions are the two arcs of the same circle: their magnitudes sum to
        ``2*pi``, so ``abs`` of this is what "the shorter arc" means.
        """
        sweep = wrap(a1 - a0)
        if direction > 0 and sweep < 0:
            sweep += 2 * math.pi
        if direction < 0 and sweep > 0:
            sweep -= 2 * math.pi
        return sweep

    def _arc(self, a0, a1, direction, radius):
        """Waypoints along the circle of ``radius`` from angle ``a0`` to ``a1``.

        ``direction`` forces the sense of travel (+1 CCW, -1 CW) so the caller can test
        both ways round and veto the blocked one, rather than always taking the short way
        and driving through a wall.
        """
        sweep = self._sweep(a0, a1, direction)
        n = max(int(abs(sweep) / math.radians(self.a.arc_step)) + 1, 2)
        return [radius * np.array([math.cos(a0 + sweep * t),
                                   math.sin(a0 + sweep * t)])
                for t in np.linspace(0.0, 1.0, n)]

    def _surface_normal(self, q_local):
        """Outward unit normal of the T at the boundary point nearest ``q_local``."""
        q = np.asarray(q_local, dtype=float)
        near = np.asarray(nearest_points(self.tee_shape.exterior, Point(q))[0].coords[0])
        d = q - near
        n = np.linalg.norm(d)
        if n < 1e-6 or self.tee_shape.contains(Point(q)):
            # On the boundary, or inside it (where q - near points the wrong way): take
            # the sampled outward normal at the nearest boundary sample instead.
            return self.ik.N[int(np.argmin(np.linalg.norm(self.ik.P - q, axis=1)))]
        return d / n

    def _reach_circle(self, q_local, radius):
        """A point on the circle of ``radius`` reachable from ``q_local`` in a straight line.

        Which ray to *prefer* depends on which side of the circle ``q`` is on.  Going out
        -- the wide rungs of the ladder -- the preferred ray is the surface normal, "away
        from the shape" taken literally, which on a convex patch is the same as going
        radially and in one of the T's notches is not.  Coming in -- the tight rungs, where
        the circle is inside the pusher's current stand-off -- the normal points away from
        the circle and mostly misses it entirely, so the preferred ray is the radial one:
        straight down to the nearest point of the circle.

        Either way the preference is only a preference, not an answer: the T is concave, so
        the normal out of a notch runs straight into the opposite arm, meets the circle on
        the far side, and leaves an approach leg that comes back down through the object.
        That is not merely a scrape.  Transit waypoints live in the T's body frame, so a
        pusher driving into the T shoves it, the target moves with it, and the leg never
        converges at all -- the pusher chases its own tail until something else interrupts.

        So the ray is TESTED, not assumed.  Directions are tried in order of angular
        distance from the preferred one, and the first whose whole leg clears the object is
        taken -- preferring legs that also keep the pusher disc clear, and settling for
        merely not passing through the object when no direction manages both.  When the
        preferred ray is already fine, which is the common case, nothing changes.
        """
        q = np.asarray(q_local, dtype=float)
        r = np.linalg.norm(q)
        if abs(r - radius) < 1e-9:
            return q                       # already on this circle; nothing to fly
        base = self._surface_normal(q) if r < radius else -q / (r + 1e-9)
        a0 = math.atan2(base[1], base[0])
        offsets = np.linspace(0.0, math.pi, EXIT_DIRS)
        order = [0.0] + [sgn * o for o in offsets[1:] for sgn in (+1.0, -1.0)]

        fallback = clear_of_shape = None
        for off in order:
            a = a0 + off
            out = _circle_point(q, np.array([math.cos(a), math.sin(a)]), radius)
            if out is None:
                continue               # this ray misses the circle altogether
            if fallback is None:
                fallback = out
            seg = LineString([q, out])
            # ``length`` rather than ``intersects``: q sits right against the surface, so a
            # leg that merely touches it is fine -- only one that runs THROUGH the object
            # has a non-degenerate intersection.
            if seg.intersection(self.tee_shape).length > 1e-9:
                continue
            if not seg.intersects(self.no_sweep):
                return out                 # clear of the disc keep-out too: best case
            if clear_of_shape is None:
                clear_of_shape = out
        if clear_of_shape is not None:
            return clear_of_shape
        return fallback if fallback is not None else radius * q / (r + 1e-9)

    def _handover(self, entry, u=None):
        """Where a flown transit ends: back along the PUSH line, clear of the object.

        The entry point itself is not a pose the pusher can fly to.  It is the contact
        backed off by ``pusher_radius + clearance`` along the push direction, and for a
        direction taken obliquely across the face -- most of the ``--dir-spread`` fan --
        that leaves the disc overlapping the T.  Teleporting into such a pose is fine,
        pymunk resolves the overlap in one step.  Flying to it cannot work at all: the
        approach presses the object before arriving, the waypoint is body-frame so the
        target moves away with the object, and the two settle into a fixed gap that never
        closes.  The pusher then shoves the T around for as long as the transit lasts,
        which is the whole point of a transit not happening.

        Backing off along the PUSH direction is the one retreat that costs nothing.
        ``final_local`` is ``entry_local`` advanced along that same direction, so a pusher
        standing further back on that line and driving to ``final`` executes the identical
        straight push -- it just spends more of it in free space first, exactly as the
        measured rollout does (it starts from ``pusher_radius + 2`` where the runtime uses
        ``pusher_radius + clearance``).  The push ends at a body-frame point either way, so
        the T still lands where the model says.

        So the retreat is ``--transit-approach`` at minimum, and as much further as it
        takes for the disc to clear -- the concave-pocket case a fixed retreat gets wrong.
        But only as far as ``--transit-reach``: a contact deep inside one of the T's
        notches needs tens of units, and retreating that far is not free after all.  The
        line back out of a notch runs ALONGSIDE the object, so the pusher scrapes the T for
        the whole approach -- measured on one such action, the disc was in contact for 107
        of its 110 frames -- and a nominally 5-unit push becomes a 77-unit drive.  Those
        contacts are simply not reachable by flying, and ``flyable`` drops them from the
        action set instead, which is honest where a long retreat is not.

        Returns the retreat point, clamped at ``--transit-reach``; ask ``flyable`` whether
        it is a pose the disc can actually occupy.
        """
        entry = np.asarray(entry, dtype=float)
        u = self.push_dir_local if u is None else u
        if u is None:
            return entry
        u = np.asarray(u, dtype=float)
        far = max(float(self.a.transit_reach), float(self.a.transit_approach))
        for t in np.linspace(float(self.a.transit_approach), far, 65):
            q = entry - u * t
            if self.tee_shape.distance(Point(q)) >= self.transit_free:
                return q
        return entry - u * far

    def flyable(self, starts, dirs):
        """Mask of the primitives the pusher can FLY to, i.e. reach without scraping.

        An action is only usable under ``--teleport-interp`` if its push line offers a pose
        the disc can occupy within ``--transit-reach`` of the entry point -- see
        ``_handover`` for why a longer retreat is worse than dropping the action.  Under
        plain ``--teleport`` nothing is dropped: the pusher is placed at the entry pose
        directly, so how it would have got there never arises.

        Both arguments are body-frame and fixed for the run, so this is computed once.
        """
        if self._fly_mask is None or len(self._fly_mask) != len(starts):
            self._fly_mask = np.array([
                self.tee_shape.distance(Point(self._handover(e, u))) >= self.transit_free
                for e, u in zip(starts, dirs)])
        return self._fly_mask

    def _edge_transit(self, start, entry, pos, ang):
        """Lift off the face, slide along it, come back down.  ``None`` if that will not do.

        When the pusher and the next entry point stand off the SAME flat face of the
        footprint, riding a circle around the whole object to get between them is absurd:
        the entire move happens against one face, and the arc's two normal legs are the
        long way round to a slide of a few units.  So that case is split out.  Both points
        are lifted off the face by the same ``--edge-lift`` along its outward normal, and
        the pusher slides between the lifted pair before coming back down to the entry
        point -- three legs still, but along the face instead of around the object.

        The lift is what makes the slide safe.  The pusher finishes a push pressed against
        the face, and an entry point taken obliquely across it stands less than a disc
        radius off it, so sliding at either point's own height would scrape the object
        along the way and shove it.  Lifting both by the same amount keeps the slide
        parallel to the face (up to the difference in their starting heights) and clear of
        it at both ends.

        ``--edge-lift`` is the nominal lift and assumes a push that ended where the model
        said it would, one disc radius off the face.  A real push that the object did not
        fully follow leaves the pusher deeper than that, and the slide is held to the full
        keep-out (``no_sweep``) like any other travel, so the nominal lift would put it
        inside and the whole shortcut would be thrown away for a few millimetres.  So the
        lift is a two-rung ladder: the nominal one, and -- only if that comes back blocked
        -- one raised by however far the lower of the two ends still sits inside the
        keep-out.  Both ends move by the same amount either way, so the slide stays
        parallel to the face, and the raised rung is bounded by the keep-out itself.

        Returns ``(waypoints, marks)``, or ``None`` when the two points are not on one
        face, or when the slide is blocked anyway -- a face whose slide runs past a corner
        into the T's own arm, or into a wall.  Either way the arc ladder takes it from
        there, so this is a shortcut, never a commitment.

        Both ends are the points actually flown to: the pusher's own position, and the
        hand-over point rather than the entry point behind it (see ``_handover``).
        """
        if self.a.no_edge_slide:
            return None
        hover = self._handover(entry)
        i = self.ik.edge_of(start)
        if i < 0 or i != self.ik.edge_of(hover):
            return None
        n = self.ik.edge_normal(i)
        start = np.asarray(start, dtype=float)
        hover = np.asarray(hover, dtype=float)
        # Second rung: how far the deeper end still sits inside the keep-out at the
        # nominal lift, plus that lift again.  Distance to the polygon is not exactly
        # height above the face -- a slide running past a corner is nearer the arm beyond
        # it -- so this only proposes; ``_path_cost`` below still decides.
        deep = min(self.tee_shape.distance(Point(q + n * self.edge_lift))
                   for q in (start, hover))
        lifts = [self.edge_lift]
        if deep < self.transit_free:
            lifts.append(self.edge_lift + self.transit_free - deep)
        if getattr(self.a, 'no_trans_collision', False):
            # --no_trans_collision: the slide is available, so it is taken.  The lift is
            # still the taller rung when the nominal one leaves an end inside the keep-out
            # -- that is sizing the lift, not vetoing the move -- but ``_path_cost`` never
            # gets a say, so a slide running past a corner into the T's own arm or into a
            # wall goes out as flown.  The point of the flag is that the arc round the
            # whole object costs more than that risk is worth on a face the pusher is
            # already standing against.
            lift = lifts[-1]
            out0, out1 = start + n * lift, hover + n * lift
            return [out0, out1, hover], (out0, out1)
        for lift in lifts:
            out0, out1 = start + n * lift, hover + n * lift
            if self._path_cost([start, out0], [out0, out1, hover], pos, ang) <= 0.0:
                return [out0, out1, hover], (out0, out1)
        # Same face, and the slide still did not survive the keep-out at any lift.  The
        # arc will take it, but this is the one failure worth seeing in a log: the
        # shortcut was ON THE TABLE and was not taken.
        self.edge_blocked = True
        return None

    def plan_transit(self, pusher_local, entry_local, pos, ang):
        """Body-frame waypoints from where the pusher is to a primitive's entry point.

        Two cases, and both are three legs -- lift off the surface, travel, come back
        down -- so the motion stays predictable.  What differs is where the middle leg
        goes.  When both points stand off the SAME face of the footprint it is a slide
        along that face (``_edge_transit``), which is the whole move: there is nothing to
        go around.  Otherwise it is an arc about the object, entered and left along the
        surface normals.  No straight-line shortcut and no teleport in either case.

        The arc's width is searched rather than fixed.  ``self.radii`` is a ladder of circles
        from one far tighter than the object out to the one that encloses it completely,
        and they are tried SMALLEST FIRST: the tightest arc that clears both the object and
        the environment wins and the search stops there, so the pusher hugs the T and only
        swings wide when something is in the way.  A rung tighter than the T's bounding
        radius is not automatically hopeless -- an arc only has to be clear over the sweep
        it actually flies, so two contacts either side of a notch can be joined by a circle
        that cuts well inside the object's extent.  Both ways round each circle are tested
        -- they are the two arcs between the same pair of points, so their swept angles sum
        to a full turn -- and of the ones that come back clear the SHORTER SWEEP is taken.
        When neither is clear the least-bad is carried to the next rung, shorter sweep
        first on a tie.

        The winning rung is then re-flown on a circle ``--arc-clearance`` wider and that
        one is kept if it is clear as well, so the radius that actually gets flown is the
        tightest one that fits PLUS that margin.  The tight arc was verified clear first
        and stays the fallback, so the margin can only ever move the path further from the
        object, never into something new.

        Every waypoint is in the T's BODY frame, so if the T shifts mid-transit the path
        deforms with it and stays collision-free instead of going stale.

        Returns ``(waypoints, marks)`` with ``marks`` the pair of points the transit
        leaves the surface at and comes back down from -- on the chosen circle for an arc,
        one lift above each contact for a slide.  ``transit_kind`` says which it was.

        Never fails: the widest rung encloses the object plus the disc plus
        ``--standoff``, so it clears the shape from every angle, and if even that is blocked
        by the environment the least-bad path is flown anyway -- stopping is an absorbing
        state and teleporting is not an option.
        """
        entry = np.asarray(entry_local, dtype=float)
        start = np.asarray(pusher_local, dtype=float)
        hover = self._handover(entry)

        self.edge_blocked = False
        edge = self._edge_transit(start, entry, pos, ang)
        if edge is not None:
            path, marks = edge
            self.transit_marks, self.transit_kind = marks, 'edge'
            self.transit_widened = False       # no circle, so no margin to add
            return path, marks

        self.transit_kind = 'arc'
        self.transit_widened = False
        best = None
        for radius in self.radii:                      # tightest arc first
            cost, path, out0, out1, ok = self._arc_transit(start, hover, radius, pos, ang)
            if ok:
                # The rung that came back clear is only just clear -- the ladder stops at
                # the FIRST circle that fits, so the disc rides as close to whatever it is
                # going round as the cost function will allow.  Fly the same transit on a
                # circle --arc-clearance wider, and keep it if that one is clear too; the
                # verified tight arc is the fallback, so growing it can never turn a clear
                # transit into a blocked one.
                if self.a.arc_clearance > 0.0:
                    r2 = float(radius) + float(self.a.arc_clearance)
                    c2, p2, o0, o1, ok2 = self._arc_transit(start, hover, r2, pos, ang)
                    if ok2:
                        path, out0, out1, radius = p2, o0, o1, r2
                        self.transit_widened = True
                self.transit_marks, self.transit_radius = (out0, out1), float(radius)
                return path, (out0, out1)
            if best is None or cost < best[0]:
                best = (cost, path, out0, out1, float(radius))
        _c, path, out0, out1, radius = best
        self.transit_marks, self.transit_radius = (out0, out1), radius
        return path, (out0, out1)

    def plan_widest_transit(self, pusher_local, target_local, pos, ang):
        """``plan_transit`` pinned to the WIDEST rung: no ladder, no slide, no margin.

        The transit the step loop flies starts from a pusher that has just finished a push
        against the object, so its lead leg is a few units long and the ladder is free to
        pick a circle that cuts inside the T's extent.  The FIRST move of a run is not like
        that: the pusher is parked wherever the run started it -- across the table, at the
        corner park -- and the lead leg into a tight circle is then a long straight line
        that is only ever checked against the polygon itself (``_path_cost`` exempts the
        lead from the disc keep-out), so it is free to shave the shape on the way in and
        shove the T before the first push is ever planned.

        So the approach is taken to ``self.radii[-1]`` and nothing else: the circle that
        encloses the object plus the pusher disc plus ``--standoff``, the one rung that
        clears the shape from EVERY angle.  Coming in from outside it, ``_reach_circle``'s
        preferred ray is the radial one, and a radial leg that stops on that circle cannot
        reach the T at all.  From there it is the ordinary around-the-object arc, on the
        one circle that is always clear.

        Both ways round are still tested and the shorter clear sweep still wins, so the
        environment (walls, obstacles) can veto a direction exactly as it does in
        ``plan_transit``.  Returns ``(waypoints, marks)`` in the body frame, same as
        ``plan_transit``.
        """
        radius = float(self.radii[-1])
        _cost, path, out0, out1, _ok = self._arc_transit(
            np.asarray(pusher_local, dtype=float), np.asarray(target_local, dtype=float),
            radius, pos, ang)
        self.transit_kind = 'arc'
        self.transit_widened = False        # the widest rung IS the margin
        self.edge_blocked = False           # a slide was never on the table
        self.transit_marks, self.transit_radius = (out0, out1), radius
        return path, (out0, out1)

    def _arc_transit(self, start, hover, radius, pos, ang):
        """One circle, both ways round: ``(cost, path, out0, out1, clear)``.

        ``clear`` says whether either way came back collision-free; when both did the one
        with the SHORTER SWEEP is returned, and when neither did the least-bad is, so the
        caller can carry it to the next rung.
        """
        out0 = self._reach_circle(start, radius)
        # The re-entry ray is aimed at the point the pusher actually flies to.  Aiming it
        # at the entry point instead was the subtle half of the same bug: the entry point
        # sits inside the disc keep-out, so ``_reach_circle`` could never find a ray clear
        # of it and always fell back to the weaker "not through the polygon" answer -- a
        # leg that presses the T all the way in.
        out1 = self._reach_circle(hover, radius)
        a0 = math.atan2(out0[1], out0[0])
        a1 = math.atan2(out1[1], out1[0])
        scored = []
        for direction in (+1, -1):
            arc = self._arc(a0, a1, direction, radius)
            # The retract leg is scored from where the pusher actually is, not from the
            # first waypoint: it is flown either way, so leaving it out of the cost hid the
            # one leg that starts in contact with the object.
            cost = self._path_cost([start, out0], arc + [out1, hover], pos, ang)
            # How far round the circle this way actually goes.  The waypoint count is not
            # that number: --arc-step buckets it 12 degrees at a time and floors at two
            # samples, so the long way round a shallow crossing ties with the short way and
            # the tie is broken by list order rather than by distance.
            sweep = abs(self._sweep(a0, a1, direction))
            scored.append((cost, sweep, [out0] + arc + [out1, hover]))
        clear = [t for t in scored if t[0] <= 0.0]
        cost, _sweep, path = (min(clear, key=lambda t: t[1]) if clear
                              else min(scored, key=lambda t: (t[0], t[1])))
        return cost, path, out0, out1, bool(clear)

    # -- discrete push primitives (teleport mode) --------------------------------------
    def begin_primitive(self, tee_pos, tee_ang, prims, push_len):
        """Choose the next push action and return where to teleport the pusher.

        Runs only when the previous action has finished, which is the whole point of the
        scheme: one inference per action instead of one per frame.  The target is the
        reference waypoint ``--action-steps`` ahead, so the action is asked to cover a
        known chunk of path rather than an instantaneous velocity.

        The push *length* is chosen here too, not fixed: ``prims.select`` ranks every
        (contact, direction, length) triple by the pose error it would leave behind.  Far
        from the goal the demand is a full ``--action-steps`` chunk and the long end of the
        ladder wins; as the T closes in, the demand shrinks below one long push and the
        short lengths win instead, which is what makes the last few actions converge rather
        than oscillate across the goal.

        Returns ``(teleport_pos, final_pos, contact_world, status)`` in world coordinates,
        or ``(None, None, None, status)`` when there is nothing sensible to do.
        """
        pos = np.asarray(tee_pos, dtype=float)
        self._advance(pos)
        tgt = min(self.k + self.a.action_steps, len(self.ref) - 1)
        if (self.k >= len(self.ref) - 1
                and np.linalg.norm(self.ref[-1, :2] - pos) < self.a.goal_dist
                and abs(wrap(self.ref[-1, 2] - tee_ang)) < math.radians(self.a.goal_deg)):
            self.done = True
            return None, None, None, 'GOAL'

        # Desired pose change over the next chunk, expressed in the body frame -- the frame
        # the primitive table lives in, so the comparison needs no per-frame geometry.
        d_world = self.ref[tgt, :2] - pos
        d_body = to_local(d_world, None, tee_ang, is_vector=True)
        d_th = wrap(self.ref[tgt, 2] - tee_ang)

        # An action is usable only if the pusher can actually stand at its start point --
        # and, when it has to fly there, only if it can reach that point without dragging
        # itself along the object to get in.
        starts = prims.P - prims.U * (self.a.pusher_radius + self.a.clearance)
        ok = np.array([self._free(sp, pos, tee_ang) for sp in starts])
        if self.a.teleport_interp:
            ok = ok & self.flyable(starts, prims.U)
        i, L, pred = (None, None, None)
        if ok.any():
            i, L, pred = prims.select((d_body[0], d_body[1], d_th), push_len, mask=ok,
                                      len_bias=self.a.len_bias)
        if i is None:
            # Nowhere the pusher can stand.  This is absorbing: the T stops, so the
            # geometry never changes, so it stays blocked.  Replanning cannot help either
            # -- a fresh path from an unchanged pose meets the same obstruction.
            return None, None, None, 'BLOCKED'

        self.prev_idx = int(prims.contact_idx[i])
        self.n_switch += 1
        contact_w = to_world(prims.P[i], pos, tee_ang)
        u_w = to_world(prims.U[i], None, tee_ang, is_vector=True)
        teleport = contact_w - u_w * (self.a.pusher_radius + self.a.clearance)
        final = teleport + u_w * (self.a.clearance + L + self.a.pusher_radius)
        self.contact_world = contact_w
        self.last_len = L
        # Body-frame twins of the two world points, for --teleport-interp: the transit is
        # planned and flown in the T's frame so it survives the T moving underneath it.
        self.entry_local = starts[i]
        self.final_local = prims.P[i] + prims.U[i] * L
        self.push_dir_local = prims.U[i]
        return (teleport, final, contact_w,
                f'ACTION {i} L={L:.0f} ref {self.k}->{tgt}/{len(self.ref) - 1}')

class Replanner:
    """Re-solves the T's reference path from wherever the T actually is.

    Called exactly once per action, between the move that just finished and the choice of
    the next one: move, then plan, then move the plan.  So it is synchronous, and there is
    no interval, no drift trigger and no worker thread -- the physics is stopped anyway
    while the decision is being made, and a path that lands mid-action would describe a
    pose the T has already left.

    That also makes the loop reproducible: every action is chosen from a path planned at
    the T's exact measured pose, rather than from whichever stale path happened to be
    current when a timer fired.
    """

    def __init__(self, womodel, goal_norm, device, steps, env_scale, env_center,
                 bbox_to_centroid, spacing, smooth):
        self.womodel = womodel
        self.goal_norm = np.asarray(goal_norm)
        self.device, self.steps = device, steps
        self.env_scale, self.env_center = env_scale, env_center
        self.b2c, self.spacing, self.smooth = bbox_to_centroid, spacing, smooth
        self.n_done = 0
        self.last_ms = 0.0

    def solve(self, pose_world):
        """Replan from this world pose.  Returns the new reference, or None on failure."""
        t0 = time.time()
        ref = None
        try:
            start = world_to_planner(np.asarray(pose_world, dtype=float),
                                     self.env_scale, self.env_center, self.b2c)
            path, _ = plan_from(self.womodel, start, self.goal_norm, self.device, self.steps)
            ref = planner_to_world(path, self.env_scale, self.env_center, self.b2c)
            ref = resample_path(ref, self.spacing, self.smooth)
            if len(ref) < 2:
                ref = None
        except Exception as exc:                    # a failed replan must not kill the sim
            print(f'[replan] failed: {exc}')
        self.last_ms = 1000.0 * (time.time() - t0)
        if ref is not None:
            self.n_done += 1
        return ref


class PushSim(base.Sim):
    """``base.Sim`` plus the planned start POSE, and the ability to place the pusher.

    The planner hands over an SE(2) start, not just a position, so the T has to be seeded
    at the reference's opening heading rather than at angle 0.
    """

    def __init__(self, args, env_polys, tee_poly, tee_start, tee_angle, pusher_start):
        super().__init__(args, env_polys, tee_poly, tee_start, pusher_start)
        self.tee_angle0 = float(tee_angle)
        self.tee.angle = self.tee_angle0

    def teleport_pusher(self, position):
        """Place the pusher instantly.  Kinematic bodies carry no momentum, so this is
        safe -- but it is also why the default mode cannot sweep the T on the way in.
        ``--teleport-interp`` pays for a real transit instead."""
        self.pusher.position = (float(position[0]), float(position[1]))
        self.pusher.velocity = (0.0, 0.0)
        self.target = pymunk.Vec2d(float(position[0]), float(position[1]))

    def reset(self):
        super().reset()
        self.tee.angle = self.tee_angle0


# ======================================================================================
# 5. app
# ======================================================================================
def build_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # geometry / planner
    ap.add_argument('--env', default='datasets/3dshape/2denv4.obj')
    ap.add_argument('--shape', default='datasets/3dshape/Tshape3d.obj')
    ap.add_argument('--shape-zup', default='datasets/3dshape/Tshape3d_zup.obj',
                    help='z-up twin of --shape; only its bounding box is read, to recover '
                         'the origin the planner poses the shape about.')
    ap.add_argument('--dataPath', default='./testing_data/3dshape/Tshape3d_env4')
    ap.add_argument('--modelPath', default='./Experiments/3dshape')
    ap.add_argument('--ckpt', default='./latest.pt')
    ap.add_argument('--case', type=int, default=0, help='test-set index to plan for')
    ap.add_argument('--mppi-steps', type=int, default=200)
    ap.add_argument('--plan-device', default='cuda')
    ap.add_argument('--traj', default=None,
                    help='replay a saved (T,3) SE(2) path instead of planning (world '
                         'units; skips the torch import entirely)')
    ap.add_argument('--save-path', default=None,
                    help='write the planned world-frame (T,3) path here and exit')
    ap.add_argument('--spacing', type=float, default=4.0,
                    help='arc-length resampling of the reference path, world units')
    ap.add_argument('--smooth', type=int, default=9,
                    help='boxcar width used to smooth the reference path (0 = off)')
    # physics (passed through to base.Sim)
    ap.add_argument('--pusher-radius', type=float, default=5.0)
    ap.add_argument('--pusher-height', type=float, default=60.0)
    ap.add_argument('--pusher-speed', type=float, default=20.0)
    ap.add_argument('--tee-mass', type=float, default=1.0)
    ap.add_argument('--friction', type=float, default=300.0)
    ap.add_argument('--spin-friction', type=float, default=8000.0)
    # base.Sim reads this; the controller's clearance model treats every obstacle as a
    # fixed polygon (_free, _path_cost, the transit arc), and run_viser does not draw
    # movable blocks, so this demo is static-obstacles only.
    ap.set_defaults(dynamic_obstacles=False)
    ap.add_argument('--fps', type=float, default=60.0)
    ap.add_argument('--substeps', type=int, default=4)
    ap.add_argument('--port', type=int, default=8080)
    # T geometry and the limit surface
    ap.add_argument('--c-length', type=float, default=None,
                    help='limit-surface length scale. Default spin_friction/friction, '
                         'which is the anisotropy pymunk actually simulates.')
    ap.add_argument('--n-boundary', type=int, default=360)
    # reference tracking
    ap.add_argument('--lookahead', type=int, default=3)
    ap.add_argument('--advance-window', type=int, default=25,
                    help='how many waypoints ahead the carrot may project in one step. '
                         'Bounds how far the reference index can jump, and keeps the '
                         'projection from snapping across a self-intersection.')
    # Success threshold.  Quartered from the original 9.0 units / 12 deg.  Kept separate
    # from --reach, which used to serve as both: tightening a shared knob would have
    # slowed carrot advance along the whole path rather than only tightening success.
    ap.add_argument('--goal-dist', type=float, default=2.25,
                    help='position tolerance for declaring the goal reached, in world '
                         'units (the T is 60 units across its crossbar)')
    ap.add_argument('--goal-deg', type=float, default=3.0,
                    help='heading tolerance for declaring the goal reached, in degrees')
    ap.add_argument('--clearance', type=float, default=1.0)
    ap.add_argument('--standoff', type=float, default=8.0,
                    help='margin beyond the object plus the pusher disc for the WIDEST '
                         'arc, world units. That circle encloses everything, so it always '
                         'clears the shape: it is the last rung of the radius ladder, and '
                         'a transit rides it only when every tighter one is blocked.')
    ap.add_argument('--arc-radius-min', type=float, default=None,
                    help='TIGHTEST arc radius, world units -- an absolute radius about the '
                         'T body origin, not a margin. Default pusher_radius/8, which is '
                         'well inside the object: rungs that tight survive only over a '
                         'short sweep through a notch, which is exactly when they should.')
    ap.add_argument('--arc-radii', type=int, default=10,
                    help='how many arc radii to try, linearly spaced from --arc-radius-min '
                         'up to and including the widest. Tried smallest first; the first '
                         'that clears the shape and the obstacles is flown.')
    ap.add_argument('--arc-step', type=float, default=12.0, help='degrees per arc waypoint')
    ap.add_argument('--edge-lift', type=float, default=None,
                    help='how far a SAME-EDGE transit lifts off the face before sliding '
                         'along it, world units. Default (pusher_radius + --clearance)/2 '
                         '-- a slide is travel, so it is held to the same keep-out as any '
                         'other transit leg (the disc plus half the clearance), and a push '
                         'finishes pressed against the face one radius off it: the lift '
                         'covers that difference plus half a radius of air. Too small a '
                         'lift does not scrape, it just sends every same-face move round '
                         'the arc instead.')
    ap.add_argument('--no-edge-slide', action='store_true',
                    help='disable the same-edge shortcut, so every transit rides an arc '
                         'around the object even when both ends are on one face.')
    ap.add_argument('--no_trans_collision', action='store_true',
                    help='take the same-face slide WITHOUT collision-checking it: when '
                         'both ends stand off one face, lift and translate, full stop.  '
                         'Normally the slide is still held to the transit keep-out and a '
                         'blocked one falls back to the arc (logged FAILED TRANS -> ARC); '
                         'with this on that check is skipped, so a slide that runs past a '
                         'corner into the T or into a wall is flown anyway.  The lift '
                         'itself is unchanged (--edge-lift, raised when an end sits inside '
                         'the keep-out).')
    ap.add_argument('--wp-tol', type=float, default=1.0,
                    help='how close the pusher must come to an INTERMEDIATE transit '
                         'waypoint before moving on, world units. These only shape the '
                         'path, so cutting their corners keeps the flight smooth -- but '
                         'the corner it cuts is off the polyline that was tested for '
                         'collisions, and the transit keep-out is only half of --clearance '
                         'wider than the disc. Measured on the T: at the old 3.0 the disc '
                         'reached 3.0 units INTO the object on the sharpest corners; at '
                         '1.0 it stays within 0.1. Does NOT apply to the two points that '
                         'matter -- where a transit hands over and where a push stops -- '
                         'which use --entry-tol.')
    ap.add_argument('--entry-tol', type=float, default=0.1,
                    help='how close it must come to the two points an ACTION is defined '
                         'by: the pose a transit hands over at, and the point a push stops '
                         'at. Both are held far tighter than a path waypoint. Loosening '
                         'this towards --wp-tol puts the push back off its own line and '
                         'stops it short of its measured length.')
    ap.add_argument('--transit-reach', type=float, default=None,
                    help='the most a flown transit may retreat along the push line to find '
                         'a pose the disc fits in, world units. Default 2 * pusher_radius. '
                         'Actions needing more than this are dropped from the action set '
                         'under --teleport-interp rather than reached by a long scrape '
                         'along the object; on the T that is about 7%% of them.')
    ap.add_argument('--transit-approach', type=float, default=None,
                    help='how far back along the PUSH LINE a flown transit ends, world '
                         'units, and a MINIMUM: it retreats further if the disc is still '
                         'inside the object there. Default pusher_radius. The entry point '
                         'itself is not a pose the pusher can fly to -- taken obliquely '
                         'across a face it leaves the disc inside the T -- and standing '
                         'back along the push direction is free, since the push drives '
                         'that same line to a body-frame end point either way.')
    # closed-loop replanning (MPC): one replan per action, from the T's live pose
    ap.add_argument('--replan-steps', type=int, default=200,
                    help='MPPI steps per replan (the rollout exits early on convergence, '
                         'so replans get cheaper as the T nears the goal)')
    ap.add_argument('--no-replan', action='store_true',
                    help='track the initial path open-loop at the PLANNING level (the '
                         'pusher control loop stays closed either way)')
    # discrete push primitives -- the only actuation mode
    ap.add_argument('--teleport', action='store_true',
                    help='the default and only base mode, accepted explicitly for '
                         'clarity: sample (surface point, push direction, push length) '
                         'actions, pick the one whose MEASURED effect best matches the '
                         'next chunk of path, teleport the pusher behind it and push. One '
                         'inference per action instead of per frame.')
    ap.add_argument('--action-steps', type=int, default=10,
                    help='how many reference waypoints ahead one action aims to cover. '
                         'With --spacing this also sets the push length.')
    ap.add_argument('--push-len', type=float, default=None,
                    help='pusher travel per action, world units. Default '
                         'spacing*action_steps, i.e. exactly the chunk of path the action '
                         'is aiming at -- measured T travel is ~0.93x pusher travel, so '
                         'the two are close to 1:1.')
    ap.add_argument('--teleport-interp', action='store_true',
                    help='same as --teleport, except the pusher FLIES to each entry point '
                         'instead of being teleported there. Every transit is three legs '
                         'with no straight-line shortcut: off the surface, across, back '
                         'down. Across is a slide along one face when both ends stand off '
                         'the same face of the footprint (see --edge-lift), otherwise an '
                         'arc round the T at the tightest of --arc-radii that clears the '
                         'shape and the obstacles. The pusher is never teleported.')
    ap.add_argument('--transit-speed', type=float, default=0.0,
                    help='pusher speed during a --teleport-interp transit, units/s '
                         '(0 = same as --pusher-speed)')
    ap.add_argument('--push-len-min', type=float, default=5.0,
                    help='shortest push in the length ladder, world units. The action '
                         'length is selected per action, so pushes shrink towards this as '
                         'the T closes on the goal.')
    ap.add_argument('--push-len-steps', type=int, default=4,
                    help='how many push lengths to measure, geometrically spaced between '
                         '--push-len-min and --push-len. 1 disables the taper.')
    ap.add_argument('--len-bias', type=float, default=0.02,
                    help='preference for long pushes, in world units of cost per unit of '
                         'unused length. Keeps the controller from creeping in tiny '
                         'actions far from the goal; too large and it will not taper.')
    ap.add_argument('--n-contacts', type=int, default=24,
                    help='surface points sampled evenly around the perimeter')
    # Keep contacts this far from a footprint corner, world units (0 = corners included).
    # A disc pressed against a corner touches an edge, not a face, so those primitives are
    # the least trustworthy in the table -- and the ones a real pusher slides off.  No
    # flag: the sim keeps every contact, and push_t_demo_realworld sets 1.5 cm in its own
    # parameter bag.  Same treatment as dynamic_obstacles above.
    ap.set_defaults(corner_margin=0.0)
    # Grow the transit circle the ladder settles on by this much before flying it, world
    # units (0 = fly the rung as found).  The ladder stops at the FIRST radius that comes
    # back clear, which is the tightest one that fits; on hardware that is closer to the
    # shape than the measured footprint deserves.  No flag: the sim flies the rung it
    # found, and push_t_demo_realworld sets ARC_CLEARANCE_CM in its own parameter bag.
    ap.set_defaults(arc_clearance=0.0)
    ap.add_argument('--n-dirs', type=int, default=5,
                    help='push directions per contact, fanned about the inward normal')
    ap.add_argument('--dir-spread', type=float, default=45.0,
                    help='half-angle of that fan, degrees')
    ap.add_argument('--action-timeout', type=float, default=4.0,
                    help='seconds before an action is abandoned and re-planned')
    ap.add_argument('--headless', action='store_true',
                    help='run the physics without viser and print a tracking report')
    ap.add_argument('--max-seconds', type=float, default=0.0,
                    help='stop after this much simulated time (0 = run forever)')
    return ap.parse_args()


def main():
    args = build_args()
    # Discrete primitives are the only actuation mode left, so --teleport is the baseline
    # whether or not it was passed; --teleport-interp swaps the teleport for a flown
    # transit.  The flag is still accepted so an explicit invocation reads unambiguously.
    args.teleport = True

    # ---- geometry ------------------------------------------------------------------
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

    # ---- reference path --------------------------------------------------------------
    womodel = goal_norm = meta = None
    if args.traj:
        ref = np.load(args.traj)
        assert ref.ndim == 2 and ref.shape[1] == 3, f'--traj must be (T,3), got {ref.shape}'
        print(f'[plan] replaying {args.traj}: {len(ref)} waypoints')
        if not args.no_replan:
            print('[plan] --traj has no network behind it, so the reference cannot be '
                  'replanned; the pusher control loop is still closed.')
    else:
        with open(os.path.join(args.dataPath, 'meta.json')) as f:
            meta = json.load(f)
        womodel, start_norm, goal_norm = load_planner(
            args.dataPath, args.modelPath, args.ckpt, args.case, args.plan_device)
        path_norm, dist = plan_from(womodel, start_norm, goal_norm,
                                    args.plan_device, args.mppi_steps)
        print(f'[plan] case {args.case}: {len(path_norm)} waypoints, '
              f'final |goal - x| = {dist:.4f} '
              f'({"converged" if dist < 0.01 else "DID NOT converge"})')
        ref = planner_to_world(path_norm, meta['env_scale'], meta['env_center'],
                               bbox_to_centroid)

    ref = resample_path(ref, args.spacing, args.smooth)
    print(f'[plan] reference: {len(ref)} waypoints, '
          f'{np.linalg.norm(np.diff(ref[:, :2], axis=0), axis=1).sum():.0f} units of travel, '
          f'net rotation {math.degrees(ref[-1, 2] - ref[0, 2]):+.0f} deg')
    if args.save_path:
        np.save(args.save_path, ref)
        print(f'[plan] wrote {args.save_path}')
        return

    # ---- push IK -------------------------------------------------------------------
    # c is read off the simulator rather than tuned separately: it is the anisotropy of
    # the ground-friction joint pair, so changing the physics changes the model with it.
    c_len = args.c_length if args.c_length else args.spin_friction / args.friction
    if args.transit_approach is None:
        args.transit_approach = args.pusher_radius
    if args.transit_reach is None:
        args.transit_reach = 2.0 * args.pusher_radius
    ik = PushIK(tee_poly, c_len, n_boundary=args.n_boundary)
    gyration = math.sqrt(sum(
        pymunk.moment_for_poly(Polygon(p).area / tee_poly.area, p, (0, 0))
        for p in base.convex_parts(tee_poly)))
    print(f'[ik] c = {c_len:.1f} (spin/linear friction ratio); '
          f'radius of gyration = {gyration:.1f}; T bounding radius = {ik.radius:.1f}; '
          f'{len(ik.P)} boundary samples'
          + (', all of them usable as contacts' if not args.corner_margin else
             f', those over {args.corner_margin:.1f} from a corner '
             f'usable as contacts'))

    # ---- sim ---------------------------------------------------------------------
    tee_start = (float(ref[0, 0]), float(ref[0, 1]))
    # Park the pusher on the standoff circle BEHIND the initial motion, so the very first
    # thing it does is press rather than walk around the T.
    heading = ref[min(args.lookahead, len(ref) - 1), :2] - ref[0, :2]
    if np.linalg.norm(heading) < 1e-9:
        heading = np.array([1.0, 0.0])
    heading = heading / np.linalg.norm(heading)
    pusher_start = np.asarray(tee_start) - heading * (
        ik.radius + args.pusher_radius + args.standoff)
    sim = PushSim(args, env_polys, tee_poly, tee_start, float(ref[0, 2]),
                  tuple(pusher_start))
    obstacles = list(env_polys)          # wall ring + interior blocks, all solid
    ctrl = PushController(ik, ref, args, obstacles)
    if args.teleport_interp:
        print(f'[transit] same-edge moves slide along the face, lifted '
              f'{ctrl.edge_lift:.1f} off it'
              + ('  [disabled]' if args.no_edge_slide else '')
              + ('  [flown unchecked: --no_trans_collision]'
                 if getattr(args, 'no_trans_collision', False) else '')
              + '; every other move is retract -> arc -> re-enter, no straight-line '
                'shortcut, arc radii tried smallest first: '
              + ', '.join(f'{r:.1f}' for r in ctrl.radii)
              + f'  (the widest, {ctrl.radii[-1]:.1f}, is the T ({ik.radius:.1f}) plus the '
                f'disc plus {args.standoff:.1f} and always clears the shape)')

    push_len = args.push_len if args.push_len else args.spacing * args.action_steps
    push_lens = push_length_ladder(push_len, args.push_len_min, args.push_len_steps)
    prims = PushPrimitives(ik, n_points=args.n_contacts, n_dirs=args.n_dirs,
                           spread_deg=args.dir_spread, corner_margin=args.corner_margin)
    print(f'[primitives] {len(prims)} actions = {prims.n_points} contacts x '
          f'{prims.n_dirs} directions (+-{args.dir_spread:.0f} deg), push length '
          f'{push_len:.1f} units = {args.action_steps} x {args.spacing:.1f}')
    if args.corner_margin > 0.0:
        print(f'[primitives] corner margin {args.corner_margin:.1f} units: '
              f'{prims.n_boundary_kept}/{len(ik.P)} boundary samples are far enough from '
              f'a corner to be sampled')
    print('[primitives] length ladder: '
          + ', '.join(f'{L:.1f}' for L in push_lens)
          + ' units (chosen per action; short pushes win near the goal)')
    if args.teleport_interp:
        _starts = prims.P - prims.U * (args.pusher_radius + args.clearance)
        _fly = ctrl.flyable(_starts, prims.U)
        print(f'[primitives] {int(_fly.sum())}/{len(_fly)} can be FLOWN to within '
              f'--transit-reach {args.transit_reach:.1f}; the other {int((~_fly).sum())} '
              f'sit where the push line offers the disc no room, so they are dropped '
              f'rather than reached by scraping along the object')
    prims.calibrate(args, tee_poly, push_lens,
                    cache_dir=os.path.dirname(args.dataPath) or '.')
    rep = None
    if womodel is not None and not args.no_replan:
        rep = Replanner(womodel, goal_norm, args.plan_device, args.replan_steps,
                        meta['env_scale'], meta['env_center'], bbox_to_centroid,
                        args.spacing, args.smooth)
        print('[replan] closed-loop planning ON: one replan per action, from the T\'s '
              'measured pose -- move, then plan, then move the plan')
    runner = PrimitiveRunner(ctrl, prims, push_len, args, rep)

    if args.headless:
        run_headless(args, sim, ctrl, ref, rep, env_polys[0].bounds, runner)
        return
    run_viser(args, sim, ctrl, ref, env_mesh, env_polys, tee_mesh, tee_poly, tee_height,
              rep, runner)


def escaped(sim, bounds, margin=100.0):
    """Has the T left the world?  pymunk can eject a pinched body through a thin wall.

    Segment walls plus an infinite-mass kinematic pusher can produce an unresolvable
    overlap, and the solver answers by flinging the T thousands of units away.  Detecting
    it keeps a blow-up from being reported as a merely-bad tracking number.
    """
    x, y = sim.tee.position.x, sim.tee.position.y
    return not (bounds[0] - margin <= x <= bounds[2] + margin
                and bounds[1] - margin <= y <= bounds[3] + margin)


class PrimitiveRunner:
    """Runs the plan/move cycle: plan, pick an action, execute it, repeat.

    The whole loop is per-ACTION, not per frame.  Once an action starts nothing is
    re-decided until it finishes -- that is what keeps the motion legible -- and once it
    does, the reference is re-planned from the T's measured pose *before* the next action
    is chosen.  Move, then plan, then move the plan.

    So there is exactly one MPPI solve and one action selection per action, both looking at
    the same pose, and no path is ever acted on that describes somewhere the T has already
    left.
    """

    def __init__(self, ctrl, prims, push_len, args, rep=None):
        self.ctrl, self.prims, self.L, self.a = ctrl, prims, push_len, args
        self.rep = rep
        self.replan_on = rep is not None
        self.reset()

    def reset(self):
        self.final = None
        self.frames = 0
        self.n_actions = 0
        self.push_need = 0.0        # world units this push has to cover, for the budget
        self.queue = []             # --teleport-interp: body-frame transit waypoints
        self.final_local = None
        self.n_transit = 0
        self.n_edge = 0             # of those, same-edge slides rather than arcs
        self.n_stalled = 0          # transits abandoned on their frame budget
        self.fly_left = 0           # frames left in the current transit's budget
        self.radii_used = []
        self.n_replans = 0
        self.need_plan = True       # plan before the first action, and after every one
        self.status = 'idle'

    def _tee(self, sim):
        return np.array([sim.tee.position.x, sim.tee.position.y]), sim.tee.angle

    def _fly(self, sim):
        """Advance along the transit queue.  True while still flying, False when arrived.

        Waypoints are stored in the T's body frame and mapped to world every frame, so a
        T that gets nudged during the transit carries the remaining path with it.

        The last waypoint is not like the others.  It is the action's precondition -- the
        pose the push is measured from -- so it is held to ``--entry-tol`` rather than to
        ``--wp-tol``.  Sharing one tolerance meant it was popped the instant the pusher
        came within ``--wp-tol``, which at 20 units/s and the old 3.0 default was 2.7-3.0
        units EVERY time: the push then started that far off its own line, aimed at a fixed
        world point, so the disc met the object at a different place and angle than the
        primitive that was measured.  Converging properly costs one or two frames, because
        a kinematic body inside ``max_speed * dt`` of its target lands exactly on it in a
        single step.

The others are held to ``--wp-tol`` and cutting their corners is fine:
        it takes the pusher off the polyline ``_path_cost`` tested, but the leg that runs
        closest to the object is the re-entry, and both of its ends are now pinned -- the
        top by the arc it leaves, the bottom by ``--entry-tol``.

        The flight is bounded even so.  Nothing here can make the pusher reach a waypoint
        that runs away from it -- if the disc is pressing the object, the body-frame target
        moves with the object -- and orbiting until something else interrupts is worse than
        starting the push late.  So a transit gets a budget of three times its own length
        at the transit speed, plus a second; past that it is abandoned where it stands and
        counted, rather than flown forever.
        """
        pos, ang = self._tee(sim)
        self.fly_left -= 1
        if self.fly_left < 0 and self.queue:
            self.queue = []
            self.n_stalled += 1
            return False
        while self.queue:
            w = to_world(self.queue[0], pos, ang)
            tol = self.a.entry_tol if len(self.queue) == 1 else self.a.wp_tol
            if (pymunk.Vec2d(*w) - sim.pusher.position).length < tol:
                self.queue.pop(0)
                continue
            sim.target = pymunk.Vec2d(float(w[0]), float(w[1]))
            sim.max_speed = self.a.transit_speed or self.a.pusher_speed
            return True
        self.queue = []
        return False

    def _plan(self, sim):
        """Re-solve the reference from where the T actually ended up."""
        self.need_plan = False
        if self.rep is None or not self.replan_on:
            return
        pos, ang = self._tee(sim)
        fresh = self.rep.solve((pos[0], pos[1], ang))
        if fresh is not None:
            self.ctrl.adopt(fresh)
            self.n_replans += 1

    def _start_push(self, sim):
        """Begin the push leg, re-deriving the end point from the T's CURRENT pose."""
        pos, ang = self._tee(sim)
        self.final = to_world(self.final_local, pos, ang)
        self.push_need = float(
            (pymunk.Vec2d(*self.final) - sim.pusher.position).length)
        self.frames = 0
        sim.target = pymunk.Vec2d(float(self.final[0]), float(self.final[1]))
        sim.max_speed = self.a.pusher_speed

    def update(self, sim):
        """Call once per frame.  Returns a status string."""
        if self.ctrl.done:
            self.final = None
            self.queue = []
            return 'GOAL'

        if self.queue:
            if self._fly(sim):
                return self.status
            self._start_push(sim)
            return self.status

        if self.final is not None:
            self.frames += 1
            # ``entry_tol``, not ``wp_tol``: this is where the action STOPS, and the
            # measured tables were built from rollouts that stop within 0.5 of the same
            # point.  At the 3.0 of a path waypoint every push quit ~2.8 units early --
            # 6% of a full-length one -- so every action under-delivered against the model
            # that chose it.
            reached = (pymunk.Vec2d(*self.final)
                       - sim.pusher.position).length < self.a.entry_tol
            # Timeout scaled by how far the move actually has to travel, so a short
            # action that is going nowhere does not hold the controller for as long as a
            # full-length one -- and so an action handed over from far back along its own
            # push line, which is what a contact inside a notch costs, is not cut off
            # before it arrives.
            budget = self.a.action_timeout * self.a.fps * max(0.25, self.push_need / (
                self.L + self.a.pusher_radius + self.a.clearance))
            if not reached and self.frames < budget:
                sim.target = pymunk.Vec2d(float(self.final[0]), float(self.final[1]))
                sim.max_speed = self.a.pusher_speed
                return self.status
            self.final = None                      # action finished (or timed out)
            self.need_plan = True                  # the move is done -- now plan

        if self.need_plan:
            self._plan(sim)

        tele, final, _c, st = self.ctrl.begin_primitive(
            (sim.tee.position.x, sim.tee.position.y), sim.tee.angle, self.prims, self.L)
        self.status = st
        if tele is None:
            sim.max_speed = 0.0
            return st
        self.n_actions += 1
        self.final_local = self.ctrl.final_local

        if self.a.teleport_interp:
            # Fly there rather than teleport.  The push itself is unchanged; only how the
            # pusher reaches the entry point differs, and plan_transit always returns one.
            pos, ang = self._tee(sim)
            q, marks = self.ctrl.plan_transit(
                to_local((sim.pusher.position.x, sim.pusher.position.y), pos, ang),
                self.ctrl.entry_local, pos, ang)
            self.queue = q
            here = to_local((sim.pusher.position.x, sim.pusher.position.y), pos, ang)
            legs = [np.asarray(here, dtype=float)] + [np.asarray(w) for w in q]
            flown = sum(float(np.linalg.norm(legs[k + 1] - legs[k]))
                        for k in range(len(legs) - 1))
            speed = self.a.transit_speed or self.a.pusher_speed
            self.fly_left = int(3.0 * flown / speed * self.a.fps) + int(self.a.fps)
            self.n_transit += 1
            if self.ctrl.transit_kind == 'edge':
                self.n_edge += 1
                self.status = st + f' +edge slide({len(q)} wp)'
            else:
                self.radii_used.append(self.ctrl.transit_radius)
                self.status = (st + f' +arc r={self.ctrl.transit_radius:.0f}'
                               f'({len(q)} wp)')
            self._fly(sim)
            return self.status

        sim.teleport_pusher(tele)
        self._start_push(sim)
        return st


def pose_error(sim, ref):
    p = np.array([sim.tee.position.x, sim.tee.position.y])
    return np.linalg.norm(ref[-1, :2] - p), abs(wrap(ref[-1, 2] - sim.tee.angle))


def run_headless(args, sim, ctrl, ref, rep=None, bounds=None, runner=None):
    """Physics-only run; prints how well the T tracked the plan.  Used to sanity-check."""
    dt = 1.0 / args.fps
    limit = args.max_seconds or 120.0
    trace, t, states = [], 0.0, {}
    # Distance to the GOAL is the only progress measure that stays comparable when the
    # reference is being replanned -- reference index and cross-track error are both
    # relative to a path that changes underneath them.
    goal_xy = ref[-1, :2].copy()
    goal_hist = []
    blew_up = False
    while t < limit and not ctrl.done:
        if bounds is not None and escaped(sim, bounds):
            blew_up = True
            break
        status = runner.update(sim)
        key = status.split()[0]
        states[key] = states.get(key, 0) + 1
        for _ in range(args.substeps):
            sim.step(dt / args.substeps)
        trace.append([sim.tee.position.x, sim.tee.position.y, sim.tee.angle])
        goal_hist.append(np.linalg.norm(goal_xy - np.array([sim.tee.position.x,
                                                            sim.tee.position.y])))
        t += dt
    trace = np.array(trace)
    gh = np.array(goal_hist)
    dp, dth = pose_error(sim, ref)
    verdict = ('GOAL REACHED' if ctrl.done else
               'BLEW UP -- the T was ejected from the world by the physics solver'
               if blew_up else 'not reached')
    print(f'\n[headless] {t:.1f}s simulated, {verdict}')
    print(f'[headless] distance to goal: start {gh[0]:.1f} -> final {gh[-1]:.1f}  '
          f'(closest approach {gh.min():.1f} at t={dt * int(np.argmin(gh)):.1f}s)')
    print(f'[headless] final pose error: {dp:.1f} units, {math.degrees(dth):.1f} deg')
    print(f'[headless] reference index reached: {ctrl.k}/{len(ctrl.ref) - 1}'
          + ('  [reference was replanned, so this is progress along the LATEST path]'
             if rep is not None and rep.n_done else ''))
    if runner is not None:
        print(f'[headless] primitive actions executed: {runner.n_actions}'
              f'  ({t / max(runner.n_actions, 1):.1f}s per action)')
        if args.teleport_interp:
            n_arc = len(runner.radii_used)
            r = np.array(runner.radii_used) if n_arc else np.zeros(1)
            widest = float(ctrl.radii[-1])
            print(f'[headless] transits flown: {runner.n_transit} '
                  f'({runner.n_edge} same-edge slides, {n_arc} arcs); 0 teleported'
                  + (f'; {runner.n_stalled} ABANDONED on the frame budget'
                     if runner.n_stalled else ''))
            if n_arc:
                print(f'[headless] arc radius min {r.min():.0f} / '
                      f'median {np.median(r):.0f} / max {r.max():.0f}  '
                      f'({100.0 * np.mean(r < widest - 1e-6):.0f}% tighter than the '
                      f'circle enclosing the object and the disc)')
    if rep is not None:
        print(f'[headless] replans completed: {rep.n_done}, one per action'
              + (f'  (last took {rep.last_ms:.0f} ms)' if rep.n_done else ''))
    tot = sum(states.values()) or 1
    print('[headless] time in state: '
          + '  '.join(f'{k}={100.0 * v / tot:.0f}%' for k, v in sorted(states.items())))


def _ref_segments(ref, z):
    """(N,2,3) line segments for a reference path drawn at height ``z``."""
    a = np.column_stack([ref[:-1, 0], ref[:-1, 1], np.full(len(ref) - 1, z)])
    b = np.column_stack([ref[1:, 0], ref[1:, 1], np.full(len(ref) - 1, z)])
    return np.stack([a, b], axis=1).astype(np.float32)


def run_viser(args, sim, ctrl, ref, env_mesh, env_polys, tee_mesh, tee_poly, tee_height,
              rep=None, runner=None):
    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction('+z')
    server.scene.world_axes.visible = False

    lo, hi = env_polys[0].bounds[:2], env_polys[0].bounds[2:]
    fw = max(hi[0] - lo[0], hi[1] - lo[1]) * 1.15
    fc = ((lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0)
    floor = server.scene.add_box('/floor', color=base.COL_FLOOR, dimensions=(fw, fw, 2.0),
                                 position=(fc[0], fc[1], -1.0))
    grid = server.scene.add_grid('/grid', width=fw, height=fw, plane='xy', cell_size=10.0,
                                 section_size=50.0, position=(fc[0], fc[1], 0.02))
    env_node = server.scene.add_mesh_simple('/env', env_mesh.vertices, env_mesh.faces,
                                            color=base.COL_ENV, flat_shading=True)

    # Reference path: the planned trace of the T's origin, plus a ghost T at the goal.
    ref_z = tee_height + 1.0
    ref_node = {'n': server.scene.add_line_segments(
        '/ref', _ref_segments(ref, ref_z), colors=COL_REF, thickness=3.0)}
    goal_frame = server.scene.add_frame('/goal', show_axes=False,
                                        position=(ref[-1, 0], ref[-1, 1], 0.0),
                                        wxyz=base.quat_z(ref[-1, 2]))
    server.scene.add_mesh_simple('/goal/mesh', tee_mesh.vertices, tee_mesh.faces,
                                 color=COL_GHOST, flat_shading=True, opacity=0.35,
                                 material='standard')

    tee_frame = server.scene.add_frame('/tee', show_axes=False, position=(*ref[0, :2], 0.0),
                                       wxyz=base.quat_z(ref[0, 2]))
    server.scene.add_mesh_simple('/tee/mesh', tee_mesh.vertices, tee_mesh.faces,
                                 color=base.COL_TEE, flat_shading=True)

    pusher_mesh = trimesh.creation.cylinder(radius=args.pusher_radius,
                                            height=args.pusher_height, sections=32)
    pusher_mesh.apply_translation([0.0, 0.0, args.pusher_height / 2.0])
    pusher_frame = server.scene.add_frame('/pusher', show_axes=False,
                                          position=(sim.pusher.position.x,
                                                    sim.pusher.position.y, 0.0))
    server.scene.add_mesh_simple('/pusher/mesh', pusher_mesh.vertices, pusher_mesh.faces,
                                 color=base.COL_PUSHER)

    contact_frame = server.scene.add_frame('/contact', show_axes=False, visible=False)
    server.scene.add_icosphere('/contact/dot', radius=2.5, color=COL_CONTACT)

    # The circle the current transit rides, drawn as a child of the T so it follows the
    # object, plus the exit / entry points on it.  Its radius is chosen per transit, so the
    # ring is rebuilt whenever the planner picks a different rung of the ladder.
    circ_z = tee_height + 0.5

    def _ring(radius):
        pts = np.array([[radius * math.cos(a), radius * math.sin(a), circ_z]
                        for a in np.linspace(0.0, 2 * math.pi, 121)], dtype=np.float32)
        return np.stack([pts[:-1], pts[1:]], axis=1)

    standoff = {'n': server.scene.add_line_segments(
        '/tee/standoff', _ring(ctrl.transit_radius), colors=COL_STANDOFF,
        thickness=2.0, visible=args.teleport_interp), 'r': ctrl.transit_radius}
    exit_frame = server.scene.add_frame('/tee/exit', show_axes=False, visible=False)
    server.scene.add_icosphere('/tee/exit/dot', radius=3.0, color=COL_EXIT)
    entry_frame = server.scene.add_frame('/tee/entry', show_axes=False, visible=False)
    server.scene.add_icosphere('/tee/entry/dot', radius=3.0, color=COL_ENTRY)
    transit_node = {'n': None}

    # The sampled contact set, drawn once per run as a child of the T.  This is the action
    # space made visible: one dot per contact the primitives can use, with a stub along
    # each one's outward normal.
    prims = runner.prims
    samp_z = tee_height + 1.0
    contacts = prims.P[::prims.n_dirs]                     # one row per contact, not per dir
    normals = prims.N[::prims.n_dirs]
    stub = 0.8 * args.pusher_radius
    sample_nodes = [
        server.scene.add_point_cloud(
            '/tee/samples/points',
            np.column_stack([contacts, np.full(len(contacts), samp_z)]).astype(np.float32),
            colors=np.tile(np.array(COL_SAMPLE, dtype=np.uint8), (len(contacts), 1)),
            point_size=2.2),
        server.scene.add_line_segments(
            '/tee/samples/normals',
            np.stack([
                np.column_stack([contacts, np.full(len(contacts), samp_z)]),
                np.column_stack([contacts + normals * stub,
                                 np.full(len(contacts), samp_z)]),
            ], axis=1).astype(np.float32),
            colors=COL_SAMPLE, thickness=2.5),
    ]
    server.scene.add_light_directional('/sun', color=(255, 255, 255), intensity=2.0,
                                       position=(fc[0] + 100, fc[1] - 150, 400))

    # ---- GUI -----------------------------------------------------------------------
    with server.gui.add_folder('Actuation'):
        server.gui.add_markdown(
            'The pusher executes discrete **push primitives**: one of a sampled set of '
            '(surface point, push direction, push length) actions is picked using a '
            'MEASURED model of what each one does, and the cylinder is placed behind it '
            'and driven straight. One decision per action, not per frame.')
        interp = server.gui.add_checkbox('interpolate transits', args.teleport_interp)
        server.gui.add_markdown(
            'With **interpolate transits** on, the cylinder is not teleported: it flies '
            'to each entry point, straight in when the line is clear and otherwise out '
            'along the surface normal, round the stand-off circle, and back in along the '
            'entry normal. Same pushes, visible travel between them.')
    with server.gui.add_folder('Run'):
        running = server.gui.add_checkbox('run', True)
        time_scale = server.gui.add_slider('time scale', 0.1, 3.0, 0.05, 1.0)
        reset_btn = server.gui.add_button('reset')
    with server.gui.add_folder('Controller'):
        g_speed = server.gui.add_slider('pusher speed', 5.0, 300.0, 5.0, args.pusher_speed)
    with server.gui.add_folder('Replan (closed-loop planning)'):
        if rep is None:
            server.gui.add_markdown(
                'Replanning needs the network; this run has none (`--traj` / '
                '`--no-replan`). The action loop is closed regardless.')
            do_replan = None
        else:
            do_replan = server.gui.add_checkbox('replan between actions', True)
            server.gui.add_markdown(
                'One replan per action, from the T\'s measured pose, between the move '
                'that just finished and the choice of the next one. The physics pauses '
                'for it -- that is the point: no action is ever chosen from a path that '
                'describes somewhere the T has already left.')
    with server.gui.add_folder('Physics'):
        fric = server.gui.add_slider('ground friction', 0.0, 2000.0, 10.0, args.friction)
        spin = server.gui.add_slider('spin friction', 0.0, 60000.0, 250.0, args.spin_friction)
    with server.gui.add_folder('View'):
        show_env = server.gui.add_checkbox('environment', True)
        show_ref = server.gui.add_checkbox('reference path', True)
        show_goal = server.gui.add_checkbox('goal ghost', True)
        show_trace = server.gui.add_checkbox('actual trace', True)
        show_floor = server.gui.add_checkbox('floor', True)
        show_grid = server.gui.add_checkbox('grid', True)
        show_circle = server.gui.add_checkbox('stand-off circle', args.teleport_interp)
        show_samples = server.gui.add_checkbox('contact samples', True)
        server.gui.add_markdown(
            'Every transit is three legs -- off the surface, across, back down -- and '
            'the middle one has two forms. Between two points standing off the SAME face '
            'of the footprint it is a slide along that face, lifted `--edge-lift` clear '
            'of it, and no circle is drawn. Otherwise it is an arc: `--arc-radii` radii '
            'are tried smallest first and the tightest that clears everything wins, so '
            'the grey ring MOVES -- it shows the circle this transit actually picked. '
            'Yellow dot = where it leaves the surface, blue dot = where it comes back '
            'down, yellow line = the path flown. The flight ends back along the push '
            'line, not at the contact, so the disc never presses the T on its way in.')
        server.gui.add_markdown(
            f'**contact samples**: the {len(contacts)} purple dots are the contacts the '
            f'primitives sample, spread evenly over the whole boundary, each with a stub '
            f'along its outward normal; every action is one of these crossed with one of '
            f'{prims.n_dirs} push directions fanned '
            f'+-{args.dir_spread:.0f} deg about that normal.')
    status = server.gui.add_markdown('')

    trace_pts = [np.array([*ref[0, :2], ref_z])]
    trace_node = {'n': None}

    @g_speed.on_update
    def _(_):
        # The runner sets sim.max_speed itself at the start of every leg, so the slider
        # writes the arg it reads rather than the sim field, which would be overwritten.
        args.pusher_speed = g_speed.value

    @interp.on_update
    def _(_):
        args.teleport_interp = interp.value
        if runner is not None:
            runner.queue = []

    @reset_btn.on_click
    def _(_):
        sim.reset()
        ctrl.reset()
        trace_pts.clear()
        trace_pts.append(np.array([*ref[0, :2], ref_z]))
        if trace_node['n'] is not None:
            trace_node['n'].remove()
            trace_node['n'] = None

    @fric.on_update
    def _(_):
        sim.set_friction(fric.value, spin.value)

    @spin.on_update
    def _(_):
        sim.set_friction(fric.value, spin.value)

    @show_env.on_update
    def _(_):
        env_node.visible = show_env.value

    @show_ref.on_update
    def _(_):
        ref_node['n'].visible = show_ref.value

    @show_goal.on_update
    def _(_):
        goal_frame.visible = show_goal.value

    @show_floor.on_update
    def _(_):
        floor.visible = show_floor.value

    @show_grid.on_update
    def _(_):
        grid.visible = show_grid.value

    @show_trace.on_update
    def _(_):
        if trace_node['n'] is not None:
            trace_node['n'].visible = show_trace.value

    @show_samples.on_update
    def _(_):
        for n in sample_nodes:
            n.visible = show_samples.value

    @show_circle.on_update
    def _(_):
        standoff['n'].visible = show_circle.value
        if not show_circle.value:
            exit_frame.visible = entry_frame.visible = False
            if transit_node['n'] is not None:
                transit_node['n'].remove()
                transit_node['n'] = None

    if do_replan is not None:
        @do_replan.on_update
        def _(_):
            runner.replan_on = do_replan.value

    print(f'[push-demo] serving on http://localhost:{args.port}')
    print(f'[push-demo] reference {len(ref)} waypoints; green line = planned T path, '
          f'green ghost = goal pose, red dot = current contact point')

    frame_dt = 1.0 / args.fps
    next_frame = time.time()
    last_status = 0.0
    sim_t = 0.0
    world = env_polys[0].bounds
    while True:
        if running.value and escaped(sim, world):
            running.value = False
            st = 'BLEW UP -- T ejected from the world; press reset'
            print('[push-demo] ' + st)
        if running.value:
            before = ctrl.ref
            st = runner.update(sim)
            contact = ctrl.contact_world
            if ctrl.ref is not before:              # a replan landed -- redraw the path
                ref_node['n'].remove()
                ref_node['n'] = server.scene.add_line_segments(
                    '/ref', _ref_segments(ctrl.ref, ref_z), colors=COL_REF,
                    thickness=3.0, visible=show_ref.value)
            dt = frame_dt * time_scale.value / args.substeps
            for _ in range(args.substeps):
                sim.step(dt)
            sim_t += frame_dt * time_scale.value

            if contact is None:
                contact_frame.visible = False
            else:
                contact_frame.position = (float(contact[0]), float(contact[1]), tee_height + 1.0)
                contact_frame.visible = True

            p = np.array([sim.tee.position.x, sim.tee.position.y, ref_z])
            if np.linalg.norm(p - trace_pts[-1]) > 2.0:
                trace_pts.append(p)
                if len(trace_pts) > 1:
                    if trace_node['n'] is not None:
                        trace_node['n'].remove()
                    segs = np.stack([np.array(trace_pts[:-1]), np.array(trace_pts[1:])],
                                    axis=1).astype(np.float32)
                    trace_node['n'] = server.scene.add_line_segments(
                        '/trace', segs, colors=COL_ACTUAL, thickness=3.0,
                        visible=show_trace.value)
        else:
            st = 'PAUSED'

        tee_frame.position = (sim.tee.position.x, sim.tee.position.y, 0.0)
        tee_frame.wxyz = base.quat_z(sim.tee.angle)
        pusher_frame.position = (sim.pusher.position.x, sim.pusher.position.y, 0.0)

        # Transit overlay.  The circle is a child of /tee so it tracks the object for
        # free; the marks and the flown path are body-frame points lifted to world.
        if show_circle.value and runner is not None:
            if abs(standoff['r'] - ctrl.transit_radius) > 1e-6:
                standoff['n'].remove()
                standoff['r'] = ctrl.transit_radius
                standoff['n'] = server.scene.add_line_segments(
                    '/tee/standoff', _ring(ctrl.transit_radius), colors=COL_STANDOFF,
                    thickness=2.0, visible=True)
            # A same-edge transit is not riding any circle, so do not draw one under it.
            standoff['n'].visible = not (runner.queue and ctrl.transit_kind == 'edge')
            marks = ctrl.transit_marks if runner.queue else None
            for frame, m in ((exit_frame, 0), (entry_frame, 1)):
                if marks is None:
                    frame.visible = False
                else:
                    frame.position = (float(marks[m][0]), float(marks[m][1]), circ_z)
                    frame.visible = True
            if transit_node['n'] is not None:
                transit_node['n'].remove()
                transit_node['n'] = None
            if runner.queue:
                tp, ta = np.array([sim.tee.position.x, sim.tee.position.y]), sim.tee.angle
                pts = [np.array([sim.pusher.position.x, sim.pusher.position.y])] + \
                      [to_world(w, tp, ta) for w in runner.queue]
                pts = np.array([[q[0], q[1], circ_z] for q in pts], dtype=np.float32)
                transit_node['n'] = server.scene.add_line_segments(
                    '/transit', np.stack([pts[:-1], pts[1:]], axis=1),
                    colors=COL_TRANSIT, thickness=2.5)

        now = time.time()
        if now - last_status > 0.1:
            last_status = now
            dp, dth = pose_error(sim, ctrl.ref)
            status.content = (
                f'**{st}**\n\n'
                f'**T**  x={sim.tee.position.x:7.1f}  y={sim.tee.position.y:7.1f}  '
                f'θ={math.degrees(sim.tee.angle) % 360:6.1f}°\n\n'
                f'**to goal**  {dp:6.1f} units, {math.degrees(dth):5.1f}°\n\n'
                f'**pusher**  |v|={sim.pusher.velocity.length:6.1f}\n\n'
                f'**plan**  {runner.n_replans} replans'
                + (f'   {rep.last_ms:.0f} ms' if rep is not None and rep.n_done else '')
                + (f'\n\n**actions**  {runner.n_actions}'
                   + (f'   **transits**  {runner.n_transit}'
                      f' ({runner.n_edge} edge / {runner.n_transit - runner.n_edge} arc)'
                      if args.teleport_interp else '')
                   if runner is not None else ''))

        next_frame += frame_dt
        time.sleep(max(0.0, next_frame - time.time()))
        if next_frame < time.time() - 0.5:
            next_frame = time.time()


if __name__ == '__main__':
    main()
