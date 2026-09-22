"""Oblique ('perturbed' camera) shots: straighten the view, locate the T, plan with MPPI,
and overlay the plan both top-down and projected back onto the table plane.

The four screenshots share one oblique camera pose.  Per image:

1. **plane homography** -- the field ArUco markers (ids 0/10/20/30) are detected and
   their corners, whose table-cm positions were read off the top-down set in
   ``../topdown_mppi.py``, give ``H_plane`` : table cm -> image px (reprojection ~0.5 px).
2. **straighten** -- a virtual camera rotation.  The field's 70 cm square is re-imaged
   as an isosceles trapezoid centred in the frame with the same top / bottom widths and
   height as the measured one, i.e. the camera keeps its downward tilt but loses its
   yaw and roll relative to the field.  For everything on the table plane this is
   exact; the arm (off the plane) is only approximately rotated, but the correction is
   a few degrees so it is not visible.
3. **T pose** -- its markers (ids 1-4) are ray-cast through ``H_plane``; because they sit
   ``TEE_HEIGHT_CM`` above the table, the on-plane hit is pulled back toward the camera
   foot by the parallax (camera centre recovered from the homography decomposition with
   a focal length fitted from the square itself).  ``frame_conversions`` then turns the
   marker pose into the shape pose exactly as in the top-down pipeline.
4. **goal** -- the green T, segmented in a top-down rendering of the image; the cleanest
   detection across the four shots is used for all of them.
5. **MPPI** -- ``push_t_demo_sim.plan_from`` in the planner's normalized frame.
6. **overlays** -- swept volume + footprints, on the top-down rendering and, projected
   through the straightened plane homography, on the straightened oblique image.

Outputs: ``rotated{n}.jpg``, ``topdown{n}.jpg``, ``mppi_topdown{n}.jpg``, ``mppi_plane{n}.jpg``.

``python perturb_mppi.py --rotate-only plane7.png plane8.png`` only straightens the given
frames (same camera, same target trapezoid as the four screenshots) -> ``rotated_<name>.jpg``,
no localisation, no plan.
"""
import math
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # ../topdown_mppi.py
import topdown_mppi as TM                            # noqa: E402  (also chdirs to repo root)
import frame_conversions as FC                       # noqa: E402
import push_t_demo_realworld as RW                   # noqa: E402
from push_t_demo_sim import planner_to_world         # noqa: E402

FILES = sorted(f for f in os.listdir(HERE)
              if f.endswith(".png") and (f.startswith("Screenshot") or f.startswith("plane")))
TEE_HEIGHT_CM = 2.0                # the T's thickness: its markers sit this far above the table
MPPI_STEPS, SEED = 200, 0


# --------------------------------------------------------------------------------------
# frames
# --------------------------------------------------------------------------------------
class HomographyFrame:
    """table cm <-> image px through a 3x3 homography (cm -> px)."""

    def __init__(self, H_cm_to_px):
        self.H = np.asarray(H_cm_to_px, float)
        self.Hinv = np.linalg.inv(self.H)

    @staticmethod
    def _apply(H, pts):
        p = np.asarray(pts, np.float64)
        out = cv2.perspectiveTransform(p.reshape(-1, 1, 2), H).reshape(-1, 2)
        return out[0] if p.ndim == 1 else out.reshape(p.shape)

    def table_cm_to_px(self, cm):
        return self._apply(self.H, cm)

    def px_to_table_cm(self, px):
        return self._apply(self.Hinv, px)


def topdown_matrix(side_px, margin_px):
    """cm -> top-down px (field of ``side_px`` inside ``margin_px``, +y up -> image up)."""
    k = side_px / FC.FIELD_CM
    c = side_px / 2.0 + margin_px
    fx, fy = FC.FIELD_CENTER_CM
    return np.array([[k, 0.0, c - k * fx], [0.0, -k, c + k * fy], [0.0, 0.0, 1.0]])


def field_marker_corners_cm():
    """{id: (4, 2) table cm} of the field markers, read off the top-down reference set."""
    imgs = {f: cv2.imread(os.path.join(TM.HERE, f)) for f in TM.FILES}
    rect = TM.Rectifier(imgs)
    mk = rect.field_mk[TM.REF]
    return {i: rect.px_to_table_cm(rect.to_topdown(TM.REF, mk[i])) for i in mk}


def plane_homography(im, mk_cm):
    mk = TM.markers(TM.FIELD_DET, im)
    ids = [i for i in mk if i in mk_cm]
    src = np.concatenate([mk_cm[i] for i in ids]).astype(np.float32)
    dst = np.concatenate([mk[i] for i in ids]).astype(np.float32)
    H, _ = cv2.findHomography(src, dst, 0)
    err = np.linalg.norm(cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2) - dst, axis=1)
    return H, ids, float(err.max())


def field_square_cm():
    c = np.asarray(FC.FIELD_CENTER_CM, float)
    h = FC.FIELD_CM / 2.0
    return np.array([c + (-h, h), c + (h, h), c + (h, -h), c + (-h, -h)])   # TL TR BR BL


# --------------------------------------------------------------------------------------
# 2. the straightened view
# --------------------------------------------------------------------------------------
def target_trapezoid(square_px_list, w, h):
    """Isosceles trapezoid, centred in a ``w x h`` frame, matching the measured squares'
    mean top width, bottom width and height."""
    sq = np.mean(square_px_list, axis=0)                 # TL TR BR BL
    top = np.linalg.norm(sq[1] - sq[0]); bot = np.linalg.norm(sq[2] - sq[3])
    hgt = (sq[2][1] + sq[3][1]) / 2.0 - (sq[0][1] + sq[1][1]) / 2.0
    cx, cy = w / 2.0, h / 2.0
    return np.array([[cx - top / 2, cy - hgt / 2], [cx + top / 2, cy - hgt / 2],
                     [cx + bot / 2, cy + hgt / 2], [cx - bot / 2, cy + hgt / 2]], np.float32)


# --------------------------------------------------------------------------------------
# 3. camera centre (for the marker-height parallax)
# --------------------------------------------------------------------------------------
def camera_center_cm(H, w, h):
    """Camera centre in table cm (x, y, z) from the plane homography.

    Principal point at the image centre; focal length from the square's own constraint
    ``|K^-1 h1| == |K^-1 h2|``.  Only the parallax correction uses this, and a 10 %
    error in it moves a 2 cm-high marker by a millimetre, so this is precise enough.
    """
    T = np.array([[1, 0, -w / 2.0], [0, 1, -h / 2.0], [0, 0, 1]])
    Hc = T @ H
    h1, h2 = Hc[:, 0], Hc[:, 1]
    f2 = (h1[0] ** 2 + h1[1] ** 2 - h2[0] ** 2 - h2[1] ** 2) / (h2[2] ** 2 - h1[2] ** 2)
    f = math.sqrt(f2)
    K = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1]])
    M = np.linalg.inv(K) @ H
    M /= np.linalg.norm(M[:, 0])
    r1, r2 = M[:, 0], M[:, 1] / np.linalg.norm(M[:, 1])
    r3 = np.cross(r1, r2)
    R = np.column_stack([r1, r2, r3])
    U, _, Vt = np.linalg.svd(R); R = U @ Vt                # nearest rotation
    t = M[:, 2]
    C = -R.T @ t
    if C[2] < 0:                                          # the other sign of the decomposition
        C = -C
    tilt = math.degrees(math.acos(abs(R[2, 2])))          # optical axis vs plane normal
    return C, f, tilt


def lift_to_height(pts_cm_on_plane, C, z_cm):
    """A pixel ray hits the table at ``p``; the same ray at height ``z`` is on the segment
    camera -> p, ``z / C_z`` of the way back from ``p``."""
    p = np.asarray(pts_cm_on_plane, float)
    lam = (z_cm - C[2]) / (0.0 - C[2])
    return C[:2] + lam * (p - C[:2])


# --------------------------------------------------------------------------------------
def straighten(files, mk_cm):
    """-> (imgs, planes, rot_frame, rotated): one straightened view shared by all
    ``files`` (same physical camera), the target trapezoid averaged over them."""
    imgs = {f: cv2.imread(os.path.join(HERE, f)) for f in files}
    h, w = next(iter(imgs.values())).shape[:2]
    sq_cm = field_square_cm()

    planes, squares = {}, []
    for f in files:
        H, ids, err = plane_homography(imgs[f], mk_cm)
        planes[f] = HomographyFrame(H)
        squares.append(planes[f].table_cm_to_px(sq_cm))
        print(f"[plane] {f}: markers {ids}, max reprojection {err:.2f} px")

    trap = target_trapezoid(squares, w, h)
    H_rot_cm = cv2.getPerspectiveTransform(sq_cm.astype(np.float32), trap)   # cm -> rotated px
    rot_frame = HomographyFrame(H_rot_cm)
    print(f"[rotate] field -> trapezoid top {trap[1][0] - trap[0][0]:.0f}px / bottom "
          f"{trap[2][0] - trap[3][0]:.0f}px / height {trap[2][1] - trap[0][1]:.0f}px, centred")
    rotated = {f: cv2.warpPerspective(imgs[f], H_rot_cm @ planes[f].Hinv, (w, h),
                                      flags=cv2.INTER_LINEAR, borderValue=(40, 40, 40))
               for f in files}
    return imgs, planes, rot_frame, rotated


def rotate_only(files):
    mk_cm = field_marker_corners_cm()
    _, _, _, rotated = straighten(files, mk_cm)
    for f in files:
        out = f"rotated_{os.path.splitext(f)[0]}.jpg"
        cv2.imwrite(os.path.join(HERE, out), rotated[f], [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"  {f} -> {out}")


def main():
    import torch

    mk_cm = field_marker_corners_cm()
    imgs, planes, rot_frame, rotated = straighten(FILES, mk_cm)
    h, w = next(iter(imgs.values())).shape[:2]

    # top-down rendering at the same scale as ../topdown{n}.jpg
    side, margin = 576, TM.MARGIN_PX
    T_td = topdown_matrix(side, margin)
    td_frame = HomographyFrame(T_td)
    td_size = side + 2 * margin

    topdown = {f: cv2.warpPerspective(imgs[f], T_td @ planes[f].Hinv, (td_size, td_size),
                                      flags=cv2.INTER_LINEAR, borderValue=(235, 235, 235))
               for f in FILES}

    args = RW.make_args(plan_device="cpu", mppi_steps=MPPI_STEPS)
    geo = RW.load_geometry(args)
    womodel = RW.load_womodel(args)

    # the goal: green T, best detection across the shots
    fits = {}
    for n, f in enumerate(FILES, 1):
        pose, score = TM.green_tee_pose(td_frame, topdown[f])
        if pose is not None:
            fits[f] = (pose, score)
            print(f"[green T] img {n}: ({pose[0]:.2f}, {pose[1]:.2f}) cm @ "
                  f"{math.degrees(pose[2]):+.1f}deg  IoU {score:.3f}")
    best_f = max(fits, key=lambda f: fits[f][1])
    goal_cm = np.asarray(fits[best_f][0])
    goal_norm = RW.real_pose_to_planner(goal_cm, geo.env_scale, geo.env_center, geo.bbox_to_centroid)
    print(f"[goal] from img {FILES.index(best_f) + 1}: table ({goal_cm[0]:.2f}, {goal_cm[1]:.2f}) cm"
          f" @ {math.degrees(goal_cm[2]):+.1f}deg = virtual ({goal_norm[0]:+.4f}, {goal_norm[1]:+.4f},"
          f" {goal_norm[5]:+.4f} turns)")

    for n, f in enumerate(FILES, 1):
        print(f"\n[{n}] {f}")
        cv2.imwrite(os.path.join(HERE, f"rotated{n}.jpg"), rotated[f], [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(os.path.join(HERE, f"topdown{n}.jpg"), topdown[f], [cv2.IMWRITE_JPEG_QUALITY, 95])

        C, fpx, tilt = camera_center_cm(planes[f].H, w, h)
        print(f"  camera: f~{fpx:.0f}px, centre ({C[0]:.0f}, {C[1]:.0f}, {C[2]:.0f}) cm, tilt {tilt:.1f}deg from vertical")
        to_cm = lambda px: lift_to_height(planes[f].px_to_table_cm(px), C, TEE_HEIGHT_CM)
        start_cm, k, spread = TM.tee_pose_from_markers(TM.markers(TM.TEE_DET, imgs[f]), to_cm)
        print(f"  T pose (table): ({start_cm[0]:.2f}, {start_cm[1]:.2f}) cm @ "
              f"{math.degrees(start_cm[2]):+.1f}deg  [{k} markers, spread {spread:.2f} cm]")
        start_norm = RW.real_pose_to_planner(start_cm, geo.env_scale, geo.env_center, geo.bbox_to_centroid)
        print(f"  T pose (virtual): x={start_norm[0]:+.4f} y={start_norm[1]:+.4f} rz={start_norm[5]:+.4f} turns")

        torch.manual_seed(SEED)
        path_norm, dist = RW.plan_from(womodel, start_norm, goal_norm, "cpu", MPPI_STEPS)
        world = planner_to_world(path_norm, geo.env_scale, geo.env_center, geo.bbox_to_centroid)
        path_cm = np.array([RW.sim_to_real_pose(*p) for p in world])
        length = np.linalg.norm(np.diff(path_cm[:, :2], axis=0), axis=1).sum()
        print(f"  MPPI: {len(path_norm)} waypoints, path {length:.1f} cm, final |goal - x| = {dist:.4f}")

        cv2.imwrite(os.path.join(HERE, f"mppi_topdown{n}.jpg"),
                    TM.draw_overlay(topdown[f], td_frame, path_cm), [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(os.path.join(HERE, f"mppi_plane{n}.jpg"),
                    TM.draw_overlay(rotated[f], rot_frame, path_cm), [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"  -> rotated{n}.jpg, topdown{n}.jpg, mppi_topdown{n}.jpg, mppi_plane{n}.jpg")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--rotate-only":
        rotate_only(sys.argv[2:])
    else:
        main()
