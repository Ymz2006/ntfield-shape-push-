"""real_world_params -- measured constants of the hardware push-T rig.

Everything that needs a measured number -- the environment's size on the table, the
offset from the origin marker, the table -> UR-base transform, the ArUco ids / sizes --
reads it from ``real_world_params.json`` through here, so there is one file to edit after
measuring the table or running ``arm_calibrate.py`` / ``camera_calibrate.py``.

    from real_world_params import PARAMS
    ENV_X = PARAMS.env_x_cm
    ox, oy = PARAMS.offset_to_origin_cm

``PARAMS`` is loaded once at import.  The scripts pass its values as argparse *defaults*,
so a CLI flag still wins.  If the JSON is missing the built-in defaults below are used and
a warning is printed (the module still imports).
"""

import json
import os
from dataclasses import dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PATH = os.path.join(HERE, "real_world_params.json")

_DEFAULTS = {
    "environment": {"env_len_cm": 70.0, "env_x_cm": None, "env_y_cm": None,
                    "offset_to_origin_cm": [0.0, 0.0], "rotation_deg": 0.0},
    "aruco": {"dict": "DICT_4X4_50", "aruco0_len_cm": 7.0, "field_marker_id": 0,
              "field_rect_cm": [94.0, 66.0], "origin_marker_yaw_deg": 0.0,
              "track_marker_id": 1, "tee_marker_id": 2, "tee_marker_offset": [0.0, 0.0, 0.0]},
    "robot": {"ip": "192.168.10.2", "startpos_json": "calibration/arm/startpos.json",
              "table_theta_deg": 0.0, "z_push_m": None,
              "workspace_box_m": {"x": [-0.90, 0.90], "y": [-0.90, 0.90], "z": [0.02, 0.70]}},
    "camera": {"width": 1280, "height": 720},
}


def _merge(base, over):
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = _merge(base[k], v) if isinstance(base.get(k), dict) and isinstance(v, dict) else v
    return out


@dataclass(frozen=True)
class RealWorldParams:
    path: str | None
    raw: dict = field(repr=False)

    # -- environment -------------------------------------------------------------------
    @property
    def env_len_cm(self) -> float:
        return float(self.raw["environment"]["env_len_cm"])

    @property
    def env_x_cm(self) -> float:
        """Table-frame X extent, cm -- the per-axis override, else ``env_len_cm``."""
        v = self.raw["environment"].get("env_x_cm")
        return float(v) if v is not None else self.env_len_cm

    @property
    def env_y_cm(self) -> float:
        v = self.raw["environment"].get("env_y_cm")
        return float(v) if v is not None else self.env_len_cm

    @property
    def offset_to_origin_cm(self) -> tuple:
        """Table-frame (x, y) cm of the environment centre (the sim's ``env_center``)."""
        x, y = self.raw["environment"]["offset_to_origin_cm"]
        return (float(x), float(y))

    @property
    def rotation_deg(self) -> float:
        return float(self.raw["environment"]["rotation_deg"])

    # -- aruco ------------------------------------------------------------------------
    @property
    def aruco_dict(self) -> str:
        return str(self.raw["aruco"]["dict"])

    @property
    def aruco0_len_cm(self) -> float:
        return float(self.raw["aruco"]["aruco0_len_cm"])

    @property
    def field_marker_id(self) -> int:
        return int(self.raw["aruco"]["field_marker_id"])

    @property
    def field_rect_cm(self) -> tuple:
        """``(x_cm, y_cm)`` of the rectangle the three ID-0 field markers sit on.

        Centre to centre, in the table frame: the origin marker is at ``(0, 0)``, the
        second straight down at ``(0, -y_cm)``, the third diagonally across at
        ``(+x_cm, -y_cm)``.
        """
        x, y = self.raw["aruco"]["field_rect_cm"]
        return (float(x), float(y))

    @property
    def origin_marker_yaw_deg(self) -> float:
        """Heading of the single origin marker's own ``+x`` edge, deg CCW from table +x.

        ``camera_test_id10`` builds the table frame from one marker, so the frame's
        direction is that marker's printed orientation; this is the measured correction
        for it being glued down turned.  90 means its ``+x`` edge runs along the width.
        """
        return float(self.raw["aruco"]["origin_marker_yaw_deg"])

    @property
    def track_marker_id(self) -> int:
        return int(self.raw["aruco"]["track_marker_id"])

    @property
    def tee_marker_id(self) -> int:
        """UNUSED -- the shape carries four markers; see frame_conversions.TEE_MARKER_IDS."""
        return int(self.raw["aruco"]["tee_marker_id"])

    @property
    def tee_marker_offset(self) -> tuple:
        """(dx_cm, dy_cm, dtheta_rad): T marker frame -> T body frame.  UNUSED -- per-marker
        placement lives in frame_conversions.TEE_MARKER_POS_CM."""
        dx, dy, dth = self.raw["aruco"]["tee_marker_offset"]
        return (float(dx), float(dy), float(dth))

    # -- robot ----------------------------------------------------------------------------
    @property
    def robot_ip(self) -> str:
        return str(self.raw["robot"]["ip"])

    @property
    def startpos_json(self) -> str:
        """Path to arm_calibrate.py's p0 point -- the TCP pose with the end effector on the
        FIELD CENTRE (sim (0, 0)), NOT the ID-10 marker, which is offset_to_origin_cm away.
        See frame_conversions.load_origin_pose / sim_to_robot."""
        return str(self.raw["robot"]["startpos_json"])

    @property
    def table_theta_deg(self) -> float:
        """Heading of table +x in the UR base frame, CCW degrees.

        UNUSED by the table -> base chain now: frame_conversions supplies that rotation as
        table_to_sim's 180 deg plus SIM_TO_BASE's flip, which cancel to leave table +x
        along base +x.  Kept only so an older config still loads."""
        return float(self.raw["robot"]["table_theta_deg"])

    @property
    def z_push_m(self):
        """Optional override of the tool-tip height; None -> keep p0's recorded z."""
        v = self.raw["robot"].get("z_push_m")
        return float(v) if v is not None else None

    @property
    def workspace_box_m(self) -> dict:
        b = self.raw["robot"]["workspace_box_m"]
        return {ax: [float(b[ax][0]), float(b[ax][1])] for ax in ("x", "y", "z")}

    # -- camera --------------------------------------------------------------------------
    @property
    def camera_width(self) -> int:
        return int(self.raw["camera"]["width"])

    @property
    def camera_height(self) -> int:
        return int(self.raw["camera"]["height"])

    # -- loading -----------------------------------------------------------------------
    @classmethod
    def load(cls, path: str = DEFAULT_PATH) -> "RealWorldParams":
        if os.path.exists(path):
            with open(path) as fh:
                return cls(path=path, raw=_merge(_DEFAULTS, json.load(fh)))
        print(f"note: {path} not found -- using built-in real-world defaults")
        return cls(path=None, raw=_merge(_DEFAULTS, {}))


PARAMS = RealWorldParams.load()
