"""calibrate_camera -- OpenCV checkerboard calibration of the RealSense colour camera.

Feed it the checkerboard shots taken with ``take_pics.py`` (default
``calibration/camera/rgb/*.png``); it finds the inner corners, runs
``cv2.calibrateCamera`` and writes the intrinsics to
``calibration/camera/intrinsics.json``.

Everything that opens the camera (``camera_calibrate.py``, ``take_pics.py``) loads that
file through ``camera_intrinsics.py`` and undistorts / deprojects with it.

    python calibrate_camera.py                       # 9x6 inner corners, auto-detected
    python calibrate_camera.py --cols 9 --rows 6 --square-mm 24.0
    python calibrate_camera.py --debug-dir calibration/camera/_corners   # dump overlays

The board is 9x6 *inner corners* (a 10x7-square OpenCV checkerboard).  ``--square-mm`` only
scales the extrinsics/board pose; the camera matrix and distortion do not depend on it.
"""

import argparse
import glob
import json
import os
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IMAGES = os.path.join(HERE, "calibration", "camera", "rgb", "*.png")
DEFAULT_OUT = os.path.join(HERE, "calibration", "camera", "intrinsics.json")

# (cols, rows) inner-corner counts tried when --cols/--rows are not given.
CANDIDATES = [(9, 6), (8, 6), (7, 6), (10, 7), (9, 7), (8, 5), (7, 5),
              (6, 5), (6, 4), (5, 4), (11, 8), (8, 11), (6, 9)]

SUBPIX_CRIT = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
FIND_FLAGS = (cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)


def find_corners(gray, size, allow_sb=True):
    ok, corners = cv2.findChessboardCorners(gray, size, FIND_FLAGS)
    if ok:
        return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), SUBPIX_CRIT)
    # SB is more robust to blur/glare (e.g. a checkerboard on a screen), but it can return
    # a partial grid, so only fall back to it once the size is known -- never in autodetect.
    if allow_sb and hasattr(cv2, "findChessboardCornersSB"):
        ok, corners = cv2.findChessboardCornersSB(gray, size, cv2.CALIB_CB_EXHAUSTIVE)
        if ok and len(corners) == size[0] * size[1]:
            return corners
    return None


def autodetect_size(grays):
    """Pick the (cols, rows) that the classic detector finds in the most images."""
    best, best_hits = None, 0
    for size in CANDIDATES:
        hits = sum(find_corners(g, size, allow_sb=False) is not None for g in grays)
        if hits > best_hits:
            best, best_hits = size, hits
    return best, best_hits


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", default=DEFAULT_IMAGES,
                    help="glob for the checkerboard images (default %(default)s)")
    ap.add_argument("--cols", type=int, default=None, help="inner corners across")
    ap.add_argument("--rows", type=int, default=None, help="inner corners down")
    ap.add_argument("--square-mm", type=float, default=24.0,
                    help="checkerboard square size in mm (extrinsics only; default %(default)s)")
    ap.add_argument("--out", default=DEFAULT_OUT, help="intrinsics JSON path (default %(default)s)")
    ap.add_argument("--debug-dir", default=None,
                    help="if set, write per-image corner overlays here")
    ap.add_argument("--max-rms", type=float, default=1.0,
                    help="reject the fit if the RMS reprojection error (px) exceeds this")
    ap.add_argument("--force", action="store_true",
                    help="write the JSON even if the fit fails the sanity checks")
    # A checkerboard that stays small, central and fronto-parallel (e.g. shown on a
    # tablet) barely constrains fy vs fx or the higher-order distortion, so the fit is
    # locked down by default.  Loosen these once you have big, tilted, edge-filling views.
    ap.add_argument("--free-aspect", action="store_true",
                    help="let fx and fy differ (default: fx == fy)")
    ap.add_argument("--free-tangent", action="store_true",
                    help="fit tangential distortion p1,p2 (default: zero)")
    ap.add_argument("--fit-k3", action="store_true",
                    help="fit the k3 radial term (default: fixed at 0)")
    ap.add_argument("--fix-principal-point", action="store_true",
                    help="pin cx,cy to the image centre")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.images))
    if not paths:
        raise SystemExit(f"no images matched {args.images}")
    print(f"{len(paths)} images from {args.images}")

    images = [(p, cv2.imread(p)) for p in paths]
    bad = [p for p, im in images if im is None]
    if bad:
        raise SystemExit("could not read: " + ", ".join(bad))
    grays = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for _, im in images]

    h, w = grays[0].shape
    if any(g.shape != (h, w) for g in grays):
        raise SystemExit("images are not all the same size")

    if args.cols and args.rows:
        size = (args.cols, args.rows)
    else:
        size, hits = autodetect_size(grays)
        if size is None:
            raise SystemExit("no checkerboard found with any candidate size; pass --cols/--rows")
        print(f"auto-detected inner-corner grid: {size[0]}x{size[1]} "
              f"(found in {hits}/{len(grays)} images)")

    # board points in board frame, z = 0, spaced by the square size (metres)
    objp = np.zeros((size[0] * size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:size[0], 0:size[1]].T.reshape(-1, 2)
    objp *= args.square_mm / 1000.0

    if args.debug_dir:
        os.makedirs(args.debug_dir, exist_ok=True)

    objpoints, imgpoints, used = [], [], []
    for (path, im), gray in zip(images, grays):
        corners = find_corners(gray, size)
        name = os.path.basename(path)
        if corners is None:
            print(f"  {name}: no corners  (skipped)")
            continue
        objpoints.append(objp)
        imgpoints.append(corners)
        used.append(name)
        print(f"  {name}: ok")
        if args.debug_dir:
            vis = im.copy()
            cv2.drawChessboardCorners(vis, size, corners, True)
            cv2.imwrite(os.path.join(args.debug_dir, name), vis)

    if len(objpoints) < 3:
        raise SystemExit(f"only {len(objpoints)} usable images -- need at least 3 (ideally 10+)")

    flags = 0
    if not args.free_aspect:
        flags |= cv2.CALIB_FIX_ASPECT_RATIO           # fx == fy
    if not args.free_tangent:
        flags |= cv2.CALIB_ZERO_TANGENT_DIST          # p1 = p2 = 0
    if not args.fit_k3:
        flags |= cv2.CALIB_FIX_K3                     # k3 = 0
    if args.fix_principal_point:
        flags |= cv2.CALIB_FIX_PRINCIPAL_POINT

    K0 = np.array([[float(w), 0, w / 2.0],
                   [0, float(w), h / 2.0],
                   [0, 0, 1.0]])                      # fx==fy seed needs a sane start
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, (w, h), K0, np.zeros(5), flags=flags)

    # RMS reprojection error per view (matches cv2's overall `rms`)
    per_view = {}
    for name, op, ip, rv, tv in zip(used, objpoints, imgpoints, rvecs, tvecs):
        proj, _ = cv2.projectPoints(op, rv, tv, K, dist)
        per_view[name] = float(cv2.norm(ip, proj, cv2.NORM_L2) / np.sqrt(len(proj)))

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    fov_x = np.degrees(2 * np.arctan(w / (2 * fx)))
    fov_y = np.degrees(2 * np.arctan(h / (2 * fy)))

    # how much of the frame the boards actually cover -- weak coverage => weak fit
    allpts = np.concatenate([c.reshape(-1, 2) for c in imgpoints])
    cover = ((allpts[:, 0].max() - allpts[:, 0].min()) / w *
             (allpts[:, 1].max() - allpts[:, 1].min()) / h)

    # sanity checks -- refuse to hand an obviously broken fit to the rest of the codebase
    problems = []
    if rms > args.max_rms:
        problems.append(f"RMS {rms:.2f}px > {args.max_rms}px")
    if not (0.30 * w < cx < 0.70 * w and 0.30 * h < cy < 0.70 * h):
        problems.append(f"principal point ({cx:.0f},{cy:.0f}) far from centre "
                        f"({w // 2},{h // 2})")
    if not (35.0 < fov_x < 100.0):
        problems.append(f"horizontal FOV {fov_x:.0f} deg implausible for this camera")
    if args.free_aspect and not (0.9 < fx / fy < 1.11):
        problems.append(f"fx/fy = {fx / fy:.2f} (expected ~1.0)")

    out = {
        "model": "opencv_pinhole_brown_conrady",
        "image_width": w,
        "image_height": h,
        "camera_matrix": K.tolist(),
        "dist_coeffs": dist.reshape(-1).tolist(),      # k1 k2 p1 p2 k3 [...]
        "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        "fov_x_deg": fov_x, "fov_y_deg": fov_y,
        "rms_reproj_error_px": float(rms),
        "per_view_error_px": per_view,
        "board_area_coverage": float(cover),
        "sanity_problems": problems,
        "meta": {
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "checkerboard_inner_corners": list(size),
            "square_mm": args.square_mm,
            "fit": {
                "fixed_aspect_ratio": not args.free_aspect,
                "zero_tangent_dist": not args.free_tangent,
                "fixed_k3": not args.fit_k3,
                "fixed_principal_point": args.fix_principal_point,
            },
            "images_glob": args.images,
            "images_used": used,
            "images_total": len(paths),
            "opencv": cv2.__version__,
        },
    }
    accepted = not problems or args.force
    rejected_path = os.path.join(os.path.dirname(os.path.abspath(args.out)),
                                 "intrinsics.rejected.json")
    target = os.path.abspath(args.out) if accepted else rejected_path
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "w") as fh:
        json.dump(out, fh, indent=2)

    print(f"\nRMS reprojection error: {rms:.4f} px   ({len(used)}/{len(paths)} images)")
    print(f"fx={fx:.2f} fy={fy:.2f}  cx={cx:.2f} cy={cy:.2f}  "
          f"FOV {fov_x:.1f}x{fov_y:.1f} deg")
    print("dist:", np.array2string(dist.reshape(-1), precision=5, suppress_small=True))
    worst = max(per_view.items(), key=lambda kv: kv[1])
    print(f"worst view: {worst[0]}  {worst[1]:.4f} px   |   board area coverage ~{cover*100:.0f}%")

    if problems:
        print("\n" + "=" * 72)
        print("this fit did NOT pass the sanity checks:")
        for p in problems:
            print(f"  - {p}")
        print("\nThe provided shots have the checkerboard small, central and nearly\n"
              "fronto-parallel in every frame, which leaves focal length, fx-vs-fy and\n"
              "lens distortion under-determined.  Re-shoot ~20 views with the board:\n"
              "  * LARGE  -- filling a third to a half of the frame\n"
              "  * TILTED -- +/-30 deg in pitch and yaw, not flat to the camera\n"
              "  * SPREAD -- into all four corners and edges, not just the middle\n"
              "then  python calibrate_camera.py  again.")
        if args.force:
            print(f"\n--force given: wrote {target} anyway.")
        else:
            print(f"\nwrote {target}")
            print(f"({os.path.basename(args.out)} left untouched; re-run with --force to "
                  "write it anyway)")
            raise SystemExit(1)
    else:
        if cover < 0.4:
            print("NOTE: board coverage is low; edge distortion is weakly constrained.")
        print(f"\nwrote {target}")


if __name__ == "__main__":
    main()
