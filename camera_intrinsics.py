"""camera_intrinsics -- load the checkerboard calibration and apply it.

``calibrate_camera.py`` writes ``calibration/camera/intrinsics.json``.  Everything that
opens the RealSense colour camera loads it through here so the images are undistorted and
pixels deproject with the calibrated pinhole model rather than the factory one.

    from camera_intrinsics import CameraIntrinsics
    calib = CameraIntrinsics.load()          # None if not calibrated yet
    frame = calib.undistort(frame)           # -> ideal pinhole image (uses new_camera_matrix)
    X, Y, Z = calib.deproject(rs, u, v, z)   # metres in the camera frame
"""

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PATH = os.path.join(HERE, "calibration", "camera", "intrinsics.json")


class CameraIntrinsics:
    def __init__(self, camera_matrix, dist_coeffs, width, height, meta=None, path=None,
                 rms=None):
        self.K = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
        self.dist = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)
        self.width = int(width)
        self.height = int(height)
        self.meta = dict(meta or {})
        self.rms = rms
        self.path = path
        self._new_K = None
        self._maps = None

    # -- construction -------------------------------------------------------------
    @classmethod
    def load(cls, path=DEFAULT_PATH, required=False):
        """Return a CameraIntrinsics, or None when the file is missing (unless required)."""
        if not os.path.exists(path):
            if required:
                raise FileNotFoundError(
                    f"{path} not found -- run  python calibrate_camera.py  first")
            return None
        with open(path) as fh:
            d = json.load(fh)
        return cls(d["camera_matrix"], d["dist_coeffs"],
                   d["image_width"], d["image_height"], d.get("meta"), path,
                   d.get("rms_reproj_error_px"))

    # -- basic accessors ---------------------------------------------------------
    @property
    def fx(self): return float(self.K[0, 0])
    @property
    def fy(self): return float(self.K[1, 1])
    @property
    def cx(self): return float(self.K[0, 2])
    @property
    def cy(self): return float(self.K[1, 2])

    def matches(self, width, height):
        return (width, height) == (self.width, self.height)

    # -- undistortion ----------------------------------------------------------------
    def new_camera_matrix(self):
        """Pinhole K of the undistorted image (alpha=0: no invalid border pixels)."""
        if self._new_K is None:
            import cv2
            self._new_K, _ = cv2.getOptimalNewCameraMatrix(
                self.K, self.dist, (self.width, self.height), 0.0,
                (self.width, self.height))
        return self._new_K

    def _undistort_maps(self):
        if self._maps is None:
            import cv2
            self._maps = cv2.initUndistortRectifyMap(
                self.K, self.dist, None, self.new_camera_matrix(),
                (self.width, self.height), cv2.CV_32FC1)
        return self._maps

    def undistort(self, image):
        """Distorted BGR/gray frame -> ideal pinhole frame (same size, new_camera_matrix)."""
        import cv2
        m1, m2 = self._undistort_maps()
        return cv2.remap(image, m1, m2, cv2.INTER_LINEAR)

    def raw_pixel(self, u, v):
        """The distorted-image pixel that an undistorted pixel (u, v) came from.

        Use this to sample the (still distorted) aligned depth frame at a corner found in
        the undistorted colour image.
        """
        m1, m2 = self._undistort_maps()
        ui = min(max(int(round(u)), 0), self.width - 1)
        vi = min(max(int(round(v)), 0), self.height - 1)
        return float(m1[vi, ui]), float(m2[vi, ui])

    # -- RealSense interop ------------------------------------------------------------
    def rs_intrinsics(self, rs):
        """A ``pyrealsense2.intrinsics`` for the *undistorted* image (zero distortion)."""
        K = self.new_camera_matrix()
        it = rs.intrinsics()
        it.width, it.height = self.width, self.height
        it.fx, it.fy = float(K[0, 0]), float(K[1, 1])
        it.ppx, it.ppy = float(K[0, 2]), float(K[1, 2])
        it.model = rs.distortion.none
        it.coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]
        return it

    def deproject(self, rs, u, v, z):
        """Camera-frame point [X, Y, Z] (m) for undistorted pixel (u, v) at depth z (m)."""
        return rs.rs2_deproject_pixel_to_point(
            self.rs_intrinsics(rs), [float(u), float(v)], float(z))

    def __repr__(self):
        rms = f"{self.rms:.3f}px" if self.rms is not None else "?"
        return (f"CameraIntrinsics({self.width}x{self.height}, "
                f"fx={self.fx:.1f} fy={self.fy:.1f} cx={self.cx:.1f} cy={self.cy:.1f}, "
                f"rms={rms})")
