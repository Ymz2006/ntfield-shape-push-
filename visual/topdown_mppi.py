"""Top-down rectify the four 'Image from iOS*.jpg' shots, locate the T, plan with MPPI,
and overlay the planned T poses on each image.

Pipeline, per image:

1. **rectify** -- the four field ArUco markers (ids 0/10/20/30, DICT_ARUCO_ORIGINAL)
   register every crop to a reference by pure translation (same camera, same scale), the
   gray field border is line-fitted in the reference and a homography maps it onto an
   axis-aligned square.  The field is ``FIELD_CM`` (70 cm) across and its centre is the
   planner's origin, so ``px <-> cm`` is a scale + flip about the square's middle.
2. **T pose** -- the T's own markers (ids 1-4, DICT_4X4_50) are detected in the original
   image, their corners pushed through the homography, and each gives the shape centre via
   ``frame_conversions.aruco_to_center`` (table cm, origin = ID-10 marker, +x right, +y
   up).  Visible markers are averaged.
3. **goal** -- the GREEN T printed on the paper is segmented (HSV) in the top-down image
   and its pose fitted by maximising IoU with the T footprint; the physical goal is the
   same in every shot, so the best-scoring detection is used for all four.
4. **virtual** -- ``push_t_demo_realworld.real_pose_to_planner`` lifts both poses into
   the planner's normalized frame ([-0.5, 0.5]^2, rz in turns).
5. **MPPI** -- ``push_t_demo_sim.plan_from`` on the Eikonal travel-time field (CPU is
   enough: ~1.5 s for 200 steps).
6. **overlay** -- the path comes back through ``planner_to_world`` / ``sim_to_real_pose``
   to table cm, and the T footprint is drawn translucent every ``GHOST_SPACING_CM`` of
   path length (blue at the start, green at the goal, no separate start/goal markup), so
   a longer path gets more T's,
   on top of the translucent SWEPT VOLUME -- the union of the footprint over the whole
   path, interpolated between waypoints.

Outputs ``topdown{n}.jpg`` (rectified) and ``mppi{n}.jpg`` (rectified + overlay).
Run from the repo root's ``visual/`` directory with the scratch venv (cv2 + torch).
"""
import math
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)                     # make_args() paths are relative to the repo root

import shapes

# ``--shape-name`` before frame_conversions binds the active shape's proportions.
if __name__ == "__main__":
    shapes.select_from_argv()

import frame_conversions as FC
import push_t_demo_realworld as RW
from push_t_demo_sim import planner_to_world

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = ["Image from iOS.jpg", "Image from iOS (1).jpg", "Image from iOS (2).jpg",
         "Image from iOS (3).jpg"]
REF = "Image from iOS (1).jpg"     # the only shot with all four field markers
MARGIN_PX = 80                     # border around the field in the rectified output
GHOST_SPACING_CM = 4.0             # one translucent T footprint per this much path length
SWEEP_SUBSTEPS = 4                 # footprint samples between consecutive waypoints
SWEEP_COLOR = (200, 140, 90)       # BGR, the swept volume fill
SWEEP_ALPHA = 0.30                 # opacity of the swept-volume fill
GHOST_ALPHA = 0.28                 # opacity of each footprint fill
EDGE_ALPHA = 0.45                  # opacity of every outline (sweep boundary + footprints)
GREEN_HSV = ((60, 80, 60), (100, 255, 255))   # the printed goal T (teal-green)
MPPI_STEPS = 200
SEED = 0

# --------------------------------------------------------------------------------------
# detectors
# --------------------------------------------------------------------------------------
def _detector(dict_name):
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    p.minMarkerPerimeterRate = 0.01
    return cv2.aruco.ArucoDetector(d, p)


FIELD_DET = _detector("DICT_ARUCO_ORIGINAL")
TEE_DET = _detector("DICT_4X4_50")


def markers(det, im):
    """{id: (4, 2) corners} in image px."""
    corners, ids, _ = det.detectMarkers(im)
    return {int(i): c[0] for c, i in zip(corners, ids.ravel())} if ids is not None else {}


# --------------------------------------------------------------------------------------
# 1. rectification
# --------------------------------------------------------------------------------------
def fit_square(im_ref, ref_mk):
    """Corners (TL, TR, BR, BL) of the gray field border in the reference image, px."""
    g = cv2.cvtColor(im_ref, cv2.COLOR_BGR2GRAY).astype(float)

    def darkest_near(v, x0, win=12):
        lo, hi = max(0, x0 - win), min(len(v), x0 + win)
        return lo + int(np.argmin(v[lo:hi]))

    def fit(pts):
        return cv2.fitLine(np.array(pts, np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()

    def inter(a, b):
        A = np.array([[a[0], -b[0]], [a[1], -b[1]]])
        t = np.linalg.solve(A, [b[2] - a[2], b[3] - a[3]])
        return np.array([a[2] + a[0] * t[0], a[3] + a[1] * t[0]])

    c = {i: ref_mk[i].mean(0) for i in (0, 10, 20, 30)}
    # the border runs just inside the marker centres
    lx = (c[10][0] + c[20][0]) / 2 + 50
    rx = (c[0][0] + c[30][0]) / 2 - 50
    ty = (c[20][1] + c[30][1]) / 2 - 20
    by = (c[10][1] + c[0][1]) / 2 + 20
    ys = np.arange(int(ty) + 60, int(by) - 60, 10)
    xs = np.arange(int(lx) + 40, int(rx) - 40, 10)
    left = fit([(darkest_near(g[y], int(lx)), y) for y in ys])
    right = fit([(darkest_near(g[y], int(rx)), y) for y in ys])
    top = fit([(x, darkest_near(g[:, x], int(ty))) for x in xs])
    bot = fit([(x, darkest_near(g[:, x], int(by))) for x in xs])
    return inter(top, left), inter(top, right), inter(bot, right), inter(bot, left)


class Rectifier:
    """image px (any of the crops) -> top-down px, and top-down px <-> table cm."""

    def __init__(self, imgs):
        self.imgs = imgs
        self.field_mk = {f: markers(FIELD_DET, im) for f, im in imgs.items()}
        ref = self.field_mk[REF]
        TL, TR, BR, BL = fit_square(imgs[REF], ref)
        self.side = int(round(np.mean([np.linalg.norm(TR - TL), np.linalg.norm(BR - TR),
                                       np.linalg.norm(BL - BR), np.linalg.norm(TL - BL)])))
        src = np.array([TL, TR, BR, BL], np.float32)
        dst = np.array([[0, 0], [self.side, 0], [self.side, self.side], [0, self.side]],
                       np.float32) + MARGIN_PX
        self.H_ref = cv2.getPerspectiveTransform(src, dst)      # ref px -> output px
        self.size = self.side + 2 * MARGIN_PX
        self.center_px = np.array([self.size / 2.0, self.size / 2.0])
        self.px_per_cm = self.side / FC.FIELD_CM
        # translation of each crop into the reference frame, from the shared field markers
        self.offset = {}
        for f, mk in self.field_mk.items():
            common = [i for i in mk if i in ref]
            if not common:
                raise RuntimeError(f"{f}: no field marker in common with {REF}")
            self.offset[f] = np.mean([ref[i].mean(0) - mk[i].mean(0) for i in common], 0)

    def H(self, f):
        T = np.eye(3)
        T[:2, 2] = self.offset[f]
        return self.H_ref @ T

    def warp(self, f, fill):
        return cv2.warpPerspective(self.imgs[f], self.H(f), (self.size, self.size),
                                   flags=cv2.INTER_LINEAR, borderValue=fill)

    def to_topdown(self, f, pts):
        return cv2.perspectiveTransform(np.asarray(pts, np.float32).reshape(-1, 1, 2),
                                        self.H(f)).reshape(-1, 2)

    # table frame: origin ID-10 marker, +x right, +y up -- the field centre is at
    # FIELD_CENTER_CM, and in the top-down image it is the middle of the square.
    def px_to_table_cm(self, px):
        d = (np.asarray(px, float) - self.center_px) / self.px_per_cm
        return np.array([d[..., 0], -d[..., 1]]).T + np.asarray(FC.FIELD_CENTER_CM)

    def table_cm_to_px(self, cm):
        d = np.asarray(cm, float) - np.asarray(FC.FIELD_CENTER_CM)
        return np.stack([d[..., 0], -d[..., 1]], -1) * self.px_per_cm + self.center_px


# --------------------------------------------------------------------------------------
# 2. the T's pose from its own markers
# --------------------------------------------------------------------------------------
def tee_pose_from_markers(mk, to_table_cm):
    """(x cm, y cm, theta rad) of the T from its markers, averaged over the visible ones.

    ``mk`` is ``{id: (4, 2) image px}``; ``to_table_cm`` maps ``(N, 2)`` image px of points
    ON THE MARKERS to table cm (whatever rectification / parallax handling that takes).
    Returns ``(pose, n_markers, xy_spread_cm)``.
    """
    est = []
    for mid in FC.TEE_MARKER_IDS:
        if mid not in mk or not FC.tee_marker_configured(mid):
            continue
        c = to_table_cm(mk[mid])                             # (4, 2) cm
        ctr = c.mean(0)
        e = c[1] - c[0]                                      # marker +x edge
        m_theta = math.atan2(e[1], e[0])
        x, y = FC.aruco_to_center(ctr[0], ctr[1], m_theta, marker_id=mid)
        th = FC.tee_theta_from_marker(m_theta, marker_id=mid)
        est.append((mid, x, y, th))
        print(f"    id {mid}: marker ({ctr[0]:6.2f}, {ctr[1]:6.2f}) cm @ "
              f"{math.degrees(m_theta):+6.1f}deg -> T ({x:6.2f}, {y:6.2f}) cm @ "
              f"{math.degrees(th):+6.1f}deg")
    if not est:
        raise RuntimeError("no T marker visible")
    xs = np.array([e[1] for e in est]); ys = np.array([e[2] for e in est])
    ths = np.array([e[3] for e in est])
    th = math.atan2(np.sin(ths).mean(), np.cos(ths).mean())
    spread = math.hypot(xs.std(), ys.std())
    return (float(xs.mean()), float(ys.mean()), th), len(est), spread


def tee_pose_table(rect, f):
    """(x cm, y cm, theta rad) of the T in the table frame, for one of the crops."""
    return tee_pose_from_markers(markers(TEE_DET, rect.imgs[f]),
                                 lambda px: rect.px_to_table_cm(rect.to_topdown(f, px)))


# --------------------------------------------------------------------------------------
# 3. the green goal T
# --------------------------------------------------------------------------------------
def _tee_mask(rect, shape, x_cm, y_cm, theta):
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [tee_polygon_px(rect, x_cm, y_cm, theta)], 255)
    return m


def green_tee_pose(rect, td):
    """(x cm, y cm, theta rad) of the green T in a top-down image, and its IoU score.

    Segment the green, keep the largest blob, take its area centroid (the footprint is
    posed about its centroid, so that IS the pose position), then fit the heading by IoU
    against the rendered footprint and polish (x, y, theta) with a small local search.
    """
    hsv = cv2.cvtColor(td, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, *GREEN_HSV)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return None, 0.0
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    blob = (lab == k).astype(np.uint8) * 255
    x, y = rect.px_to_table_cm(cen[k])

    def iou(px, py, th):
        r = _tee_mask(rect, blob.shape, px, py, th)
        return cv2.countNonZero(cv2.bitwise_and(r, blob)) / max(1, cv2.countNonZero(cv2.bitwise_or(r, blob)))

    th = max(np.radians(np.arange(0, 360, 2.0)), key=lambda t: iou(x, y, t))
    best = iou(x, y, th)
    for step_cm, step_deg in ((0.5, 2.0), (0.2, 0.5), (0.1, 0.25)):
        improved = True
        while improved:
            improved = False
            for dx, dy, dt in ((step_cm, 0, 0), (-step_cm, 0, 0), (0, step_cm, 0), (0, -step_cm, 0),
                               (0, 0, math.radians(step_deg)), (0, 0, -math.radians(step_deg))):
                v = iou(x + dx, y + dy, th + dt)
                if v > best:
                    best, x, y, th, improved = v, x + dx, y + dy, th + dt, True
    return (float(x), float(y), float(FC._wrap(th))), float(best)


# --------------------------------------------------------------------------------------
# 6. overlay
# --------------------------------------------------------------------------------------
def tee_polygon_px(rect, x_cm, y_cm, theta):
    """Footprint (about the centroid, what the sim poses about) as top-down px."""
    o = FC.tee_body_outline_cm(mode="centroid")
    c, s = math.cos(theta), math.sin(theta)
    pts = o @ np.array([[c, s], [-s, c]]) + np.array([x_cm, y_cm])
    return rect.table_cm_to_px(pts).astype(np.int32)


def swept_mask(rect, shape, path_cm):
    """Union of the T footprint along the path, with the pose interpolated between
    waypoints (heading unwrapped so it turns the short way)."""
    th = np.unwrap(path_cm[:, 2])
    m = np.zeros(shape, np.uint8)
    for a, b in zip(range(len(path_cm) - 1), range(1, len(path_cm))):
        for t in np.linspace(0.0, 1.0, SWEEP_SUBSTEPS, endpoint=False):
            x = (1 - t) * path_cm[a, 0] + t * path_cm[b, 0]
            y = (1 - t) * path_cm[a, 1] + t * path_cm[b, 1]
            cv2.fillPoly(m, [tee_polygon_px(rect, x, y, (1 - t) * th[a] + t * th[b])], 255)
    cv2.fillPoly(m, [tee_polygon_px(rect, *path_cm[-1])], 255)
    return m


def _blend(out, draw, alpha):
    """Apply ``draw(layer)`` on a copy of ``out`` and blend it back at ``alpha``."""
    layer = out.copy()
    draw(layer)
    return cv2.addWeighted(layer, alpha, out, 1.0 - alpha, 0)


def draw_overlay(im, rect, path_cm):
    out = im.copy()
    # swept volume first, so the footprints sit on top of it
    sw = swept_mask(rect, im.shape[:2], path_cm)
    cnts, _ = cv2.findContours(sw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    def sweep_fill(layer):
        layer[sw > 0] = SWEEP_COLOR
    out = _blend(out, sweep_fill, SWEEP_ALPHA)
    out = _blend(out, lambda l: cv2.drawContours(l, cnts, -1, SWEEP_COLOR, 1, cv2.LINE_AA),
                 EDGE_ALPHA)

    # one footprint every GHOST_SPACING_CM of travelled path (always start + end)
    seg = np.linalg.norm(np.diff(path_cm[:, :2], axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    marks = np.arange(0.0, arc[-1], GHOST_SPACING_CM)
    idx = np.unique(np.concatenate([np.searchsorted(arc, marks), [len(path_cm) - 1]]))
    # translucent footprints, colour ramping start (blue) -> goal (green)
    for k, i in enumerate(idx):
        t = k / max(1, len(idx) - 1)
        col = (int(255 * (1 - t)), int(80 + 100 * t), int(30 * (1 - t)))     # BGR
        poly = [tee_polygon_px(rect, *path_cm[i])]
        out = _blend(out, lambda l: cv2.fillPoly(l, poly, col), GHOST_ALPHA)
        out = _blend(out, lambda l: cv2.polylines(l, poly, True, col, 1, cv2.LINE_AA),
                     EDGE_ALPHA)
    return out


# --------------------------------------------------------------------------------------
def main():
    import torch

    imgs = {f: cv2.imread(os.path.join(HERE, f)) for f in FILES}
    rect = Rectifier(imgs)
    print(f"[rectify] field {rect.side}px = {FC.FIELD_CM:g} cm -> {rect.px_per_cm:.2f} px/cm, "
          f"output {rect.size}x{rect.size}")
    # fill for the corners no crop covers: the paper's colour
    ref_td = rect.warp(REF, (0, 0, 0))
    m = MARGIN_PX + 20
    fill = tuple(int(v) for v in np.median(ref_td[m:m + 30, m:m + 30].reshape(-1, 3), 0))

    args = RW.make_args(plan_device="cpu", mppi_steps=MPPI_STEPS)
    geo = RW.load_geometry(args)
    womodel = RW.load_womodel(args)
    # the goal: the green T on the paper.  Same physical target in every shot, so detect
    # it in each and keep the cleanest (least occluded) fit for all of them.
    topdown = {f: rect.warp(f, fill) for f in FILES}
    fits = {}
    for n, f in enumerate(FILES, 1):
        pose, score = green_tee_pose(rect, topdown[f])
        if pose is not None:
            fits[f] = (pose, score)
            print(f"[green T] img {n}: ({pose[0]:.2f}, {pose[1]:.2f}) cm @ "
                  f"{math.degrees(pose[2]):+.1f}deg  IoU {score:.3f}")
    if not fits:
        raise RuntimeError("green T not found in any image")
    best_f = max(fits, key=lambda f: fits[f][1])
    goal_cm = np.asarray(fits[best_f][0])
    goal_norm = RW.real_pose_to_planner(goal_cm, geo.env_scale, geo.env_center,
                                        geo.bbox_to_centroid)
    print(f"[goal] from img {FILES.index(best_f) + 1}: table ({goal_cm[0]:.2f}, {goal_cm[1]:.2f})"
          f" cm @ {math.degrees(goal_cm[2]):+.1f}deg = virtual ({goal_norm[0]:+.4f}, "
          f"{goal_norm[1]:+.4f}, {goal_norm[5]:+.4f} turns)   "
          f"[GOAL_POSE_NORM was {RW.GOAL_POSE_NORM}]")

    for n, f in enumerate(FILES, 1):
        print(f"\n[{n}] {f}")
        td = topdown[f]
        cv2.imwrite(os.path.join(HERE, f"topdown{n}.jpg"), td, [cv2.IMWRITE_JPEG_QUALITY, 95])

        start_cm, k, spread = tee_pose_table(rect, f)
        print(f"  T pose (table): ({start_cm[0]:.2f}, {start_cm[1]:.2f}) cm @ "
              f"{math.degrees(start_cm[2]):+.1f}deg  [{k} markers, spread {spread:.2f} cm]")
        start_norm = RW.real_pose_to_planner(start_cm, geo.env_scale, geo.env_center,
                                             geo.bbox_to_centroid)
        print(f"  T pose (virtual): x={start_norm[0]:+.4f} y={start_norm[1]:+.4f} "
              f"rz={start_norm[5]:+.4f} turns")

        torch.manual_seed(SEED)
        path_norm, dist = RW.plan_from(womodel, start_norm, goal_norm, "cpu", MPPI_STEPS)
        world = planner_to_world(path_norm, geo.env_scale, geo.env_center, geo.bbox_to_centroid)
        path_cm = np.array([RW.sim_to_real_pose(*w) for w in world])
        seg = np.linalg.norm(np.diff(path_cm[:, :2], axis=0), axis=1).sum()
        print(f"  MPPI: {len(path_norm)} waypoints, path {seg:.1f} cm, final |goal - x| = {dist:.4f}")

        out = draw_overlay(td, rect, path_cm)
        cv2.imwrite(os.path.join(HERE, f"mppi{n}.jpg"), out, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"  -> topdown{n}.jpg, mppi{n}.jpg")


if __name__ == "__main__":
    main()
