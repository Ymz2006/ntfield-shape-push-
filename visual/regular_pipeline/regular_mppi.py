"""Top-down view + MPPI overlay for the wide 4K ``pusht-{k}.jpg`` frames.

Same pipeline as ``../topdown_mppi.py`` (rectify -> locate the T -> green goal -> MPPI ->
swept volume + footprints), but the rectification comes from a per-frame plane
homography fitted on the field markers' corners (table-cm positions read off the first
set), because these frames are a single wide overhead shot rather than crops of one view.
The camera is nearly at nadir here, so the T markers' 2 cm height shifts them by well
under a centimetre and no parallax correction is applied.

Outputs ``topdown-{k}.jpg`` and ``mppi-{k}.jpg`` next to each ``pusht-{k}.jpg``.
"""
import glob
import math
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VISUAL = os.path.dirname(HERE)
sys.path.insert(0, VISUAL)
sys.path.insert(0, os.path.join(VISUAL, "perturb"))
import topdown_mppi as TM                                        # noqa: E402
import perturb_mppi as PM                                        # noqa: E402
import frame_conversions as FC                                   # noqa: E402
import push_t_demo_realworld as RW                               # noqa: E402
from push_t_demo_sim import planner_to_world                     # noqa: E402

FILES = sorted(glob.glob(os.path.join(HERE, "pusht-*.jpg")))
FIELD_IDS = (0, 10, 20, 30)
MIN_FIELD_SIDE_PX = 50            # the real field markers are ~80 px; background prints ~18 px
MPPI_STEPS, SEED = 200, 0

# default parameters: the perimeter-rate override in topdown_mppi drops the real markers
# in these 4K frames, and the size filter below is what rejects the background prints
_p = cv2.aruco.DetectorParameters()
_p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
FIELD_DET = cv2.aruco.ArucoDetector(
    cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL), _p)


def field_markers(im):
    mk = TM.markers(FIELD_DET, im)
    return {i: c for i, c in mk.items()
            if i in FIELD_IDS and np.linalg.norm(c[0] - c[1]) >= MIN_FIELD_SIDE_PX}


def plane_homography(im, mk_cm):
    mk = field_markers(im)
    ids = sorted(i for i in mk if i in mk_cm)
    src = np.concatenate([mk_cm[i] for i in ids]).astype(np.float32)
    dst = np.concatenate([mk[i] for i in ids]).astype(np.float32)
    H, _ = cv2.findHomography(src, dst, 0)
    err = np.linalg.norm(
        cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2) - dst, axis=1)
    return H, ids, float(err.max())


def main():
    import torch

    mk_cm = PM.field_marker_corners_cm()
    side, margin = 576, TM.MARGIN_PX
    T_td = PM.topdown_matrix(side, margin)
    td_frame = PM.HomographyFrame(T_td)
    td_size = side + 2 * margin

    imgs, planes, topdown = {}, {}, {}
    for f in FILES:
        im = cv2.imread(f)
        H, ids, err = plane_homography(im, mk_cm)
        imgs[f], planes[f] = im, PM.HomographyFrame(H)
        topdown[f] = cv2.warpPerspective(im, T_td @ np.linalg.inv(H), (td_size, td_size),
                                         flags=cv2.INTER_AREA, borderValue=(235, 235, 235))
        print(f"[plane] {os.path.basename(f)}: markers {ids}, max reprojection {err:.2f} px")

    args = RW.make_args(plan_device="cpu", mppi_steps=MPPI_STEPS)
    geo = RW.load_geometry(args)
    womodel = RW.load_womodel(args)

    fits = {}
    for f in FILES:
        pose, score = TM.green_tee_pose(td_frame, topdown[f])
        if pose is not None:
            fits[f] = (pose, score)
            print(f"[green T] {os.path.basename(f)}: ({pose[0]:.2f}, {pose[1]:.2f}) cm @ "
                  f"{math.degrees(pose[2]):+.1f}deg  IoU {score:.3f}")
    best_f = max(fits, key=lambda f: fits[f][1])
    goal_cm = np.asarray(fits[best_f][0])
    goal_norm = RW.real_pose_to_planner(goal_cm, geo.env_scale, geo.env_center, geo.bbox_to_centroid)
    print(f"[goal] from {os.path.basename(best_f)}: table ({goal_cm[0]:.2f}, {goal_cm[1]:.2f}) cm"
          f" @ {math.degrees(goal_cm[2]):+.1f}deg = virtual ({goal_norm[0]:+.4f}, {goal_norm[1]:+.4f},"
          f" {goal_norm[5]:+.4f} turns)")

    for f in FILES:
        k = os.path.basename(f)[len("pusht-"):-len(".jpg")]
        print(f"\n[{k}] {os.path.basename(f)}")
        cv2.imwrite(os.path.join(HERE, f"topdown-{k}.jpg"), topdown[f], [cv2.IMWRITE_JPEG_QUALITY, 95])

        start_cm, n, spread = TM.tee_pose_from_markers(TM.markers(TM.TEE_DET, imgs[f]),
                                                       planes[f].px_to_table_cm)
        print(f"  T pose (table): ({start_cm[0]:.2f}, {start_cm[1]:.2f}) cm @ "
              f"{math.degrees(start_cm[2]):+.1f}deg  [{n} markers, spread {spread:.2f} cm]")
        start_norm = RW.real_pose_to_planner(start_cm, geo.env_scale, geo.env_center, geo.bbox_to_centroid)
        print(f"  T pose (virtual): x={start_norm[0]:+.4f} y={start_norm[1]:+.4f} rz={start_norm[5]:+.4f} turns")

        torch.manual_seed(SEED)
        path_norm, dist = RW.plan_from(womodel, start_norm, goal_norm, "cpu", MPPI_STEPS)
        world = planner_to_world(path_norm, geo.env_scale, geo.env_center, geo.bbox_to_centroid)
        path_cm = np.array([RW.sim_to_real_pose(*p) for p in world])
        length = np.linalg.norm(np.diff(path_cm[:, :2], axis=0), axis=1).sum()
        print(f"  MPPI: {len(path_norm)} waypoints, path {length:.1f} cm, final |goal - x| = {dist:.4f}")

        cv2.imwrite(os.path.join(HERE, f"mppi-{k}.jpg"),
                    TM.draw_overlay(topdown[f], td_frame, path_cm), [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"  -> topdown-{k}.jpg, mppi-{k}.jpg")


if __name__ == "__main__":
    main()
