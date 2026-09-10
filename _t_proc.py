"""Synthetic check of camera_test's Procrustes method: no camera, no RealSense."""
import sys, types
import numpy as np

# stub out pyrealsense2-free import path; camera_test imports cv2 + params only at module level
import camera_test as ct


def rot(axis, deg):
    a = np.asarray(axis, float); a /= np.linalg.norm(a)
    th = np.radians(deg); K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


RECT = (94.0, 66.0)
model = ct.rect_model_cm(RECT)          # cm, table frame
print("model:\n", model)

# ---- build a synthetic scene: table plane tilted 12 deg, rotated 37 deg, 1.4 m away ----
R = rot([1.0, 0.3, 0.0], 12.0) @ rot([0.0, 0.0, 1.0], 37.0)
# table (x right, y up, z out of table) -> camera (X right, Y down, Z forward): flip y and z
FLIP = np.diag([1.0, -1.0, -1.0])
t = np.array([0.05, -0.02, 1.40])

def to_cam(p2_cm, scale=1.0):
    p = np.array([p2_cm[0], p2_cm[1], 0.0]) / 100.0 * scale
    return R @ (FLIP @ p) + t

for true_scale, label in ((1.0, "perfect"), (1.0 / 1.03, "pnp 3% too small")):
    pts = np.array([to_cam(m, true_scale) for m in model])
    uv, n = ct.plane_coords(pts)
    s, Rf, _, rms = ct.procrustes_2d(uv * 100.0, model)
    a, b, c = (Rf @ (uv * 100.0).T).T
    tilt = np.degrees(np.arccos(np.clip(-n[2], -1, 1)))
    print(f"\n[{label}]  scale={s:.6f} (expect {1/true_scale:.6f})  rms={rms:.6f} cm")
    print(f"  diag  x={c[0]-a[0]:+.4f} y={c[1]-a[1]:+.4f}  (expect "
          f"{94*true_scale:+.4f} {-66*true_scale:+.4f})")
    print(f"  down  x={b[0]-a[0]:+.4f} y={b[1]-a[1]:+.4f}  (expect 0 {-66*true_scale:+.4f})")
    print(f"  tilt={tilt:.4f} deg  (expect 12.0)")

# ---- a genuinely wrong shape must show up as rms, not be absorbed ----
bad = model.copy(); bad[2, 0] += 5.0     # diagonal marker 5 cm off
pts = np.array([to_cam(m) for m in bad])
uv, n = ct.plane_coords(pts)
s, Rf, _, rms = ct.procrustes_2d(uv * 100.0, model)
print(f"\n[diag marker 5 cm off]  scale={s:.4f}  rms={rms:.4f} cm  (must be nonzero)")

# ---- mirrored input must not fit ----
mir = np.array([[to_cam(m)[0] * 1, to_cam(m)[1], to_cam(m)[2]] for m in model])
flipped = model.copy(); flipped[:, 0] *= -1
pts = np.array([to_cam(m) for m in flipped])
uv, n = ct.plane_coords(pts)
s, Rf, _, rms = ct.procrustes_2d(uv * 100.0, model)
print(f"[mirrored layout]       scale={s:.4f}  rms={rms:.4f} cm  (must be large)")

# ---- field_triple identification, over many random in-plane rotations ----
import cv2
half = ct.ARUCO0_LEN_CM / 2.0
ok = 0
rng = np.random.default_rng(0)
for trial in range(200):
    Rt = rot([rng.normal() * 0.3, rng.normal() * 0.3, 1.0], rng.uniform(0, 360))
    tt = np.array([rng.uniform(-.2, .2), rng.uniform(-.2, .2), rng.uniform(1.0, 2.0)])
    K = np.array([[896.0, 0, 640.0], [0, 896.0, 360.0], [0, 0, 1.0]])
    quads = []
    for m in model:                       # 4 corners of each marker, in the table plane
        cs = [[m[0] - half, m[1] + half], [m[0] + half, m[1] + half],
              [m[0] + half, m[1] - half], [m[0] - half, m[1] - half]]
        P = np.array([Rt @ (FLIP @ (np.array([c[0], c[1], 0.0]) / 100.0)) + tt for c in cs])
        px = (K @ P.T).T
        quads.append((px[:, :2] / px[:, 2:3]).astype(np.float32).reshape(1, 4, 2))
    order = rng.permutation(3)             # detector order is arbitrary
    corners = [quads[i] for i in order]
    ids = np.array([[0], [0], [0]])
    tri = ct.field_triple(corners, ids)
    centres = [q.mean(axis=0) for q in tri]
    want = [quads[i].reshape(4, 2).mean(axis=0) for i in (0, 1, 2)]
    ok += all(np.allclose(centres[k], want[k], atol=1e-3) for k in range(3))
print(f"\nfield_triple identified origin/ydown/diag correctly in {ok}/200 random views")
