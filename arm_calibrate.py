"""Real-world push-T on the UR5 -- stage 1: interactive workspace calibration.

Only the calibration stage is implemented.  It is the piece that has to exist before any
planned push can be replayed on hardware: the planner works in its own normalized SE(2)
frame, and mapping that frame onto the physical table needs a handful of TCP poses that a
human has eyeballed into place.  This script is how you collect them -- you jog the pusher
around with the arrow keys and press SPACE at each point you care about.

Two invariants hold for the whole session:

* **The tool always points straight down.**  Every target is rebuilt from scratch as
  ``down_pose(position, yaw)`` rather than by accumulating deltas onto the measured pose,
  so orientation drift cannot creep in over a long jogging session.  On startup the
  current orientation is *snapped* to exactly down while keeping its existing yaw (the arm
  is typically already within a few degrees of down, so this is a small wrist correction).
* **Motion is relative.**  Each keypress offsets the current commanded position; there are
  no absolute waypoints to get wrong.

Everything is bounded before it is sent: the target is clamped to a workspace box, then
checked against the controller's own ``isPoseWithinSafetyLimits`` (reachability + safety
planes), and only then does it become a short ``moveL``.  Keys pressed while a move is in
flight are coalesced into one motion instead of queueing up a backlog of stale jogs.

Frames
------
Poses are UR ``(x, y, z, rx, ry, rz)`` in the robot BASE frame, metres and rotation-vector
radians.  With ``--view-yaw 0`` the right arrow is base +X and the up arrow is base +Y;
set ``--view-yaw`` to the angle you are standing at so the arrows match what you see.

Each keypress is one discrete 1 cm jog (5 deg for yaw).  A jog is a blocking ``moveL``:
anything typed while the arm is still moving is flushed and ignored, so one keypress can
never turn into a backlog of stale motion.

Usage (from the ntrl-demo root, inside the pytorchserver container -- needs a TTY):
    docker exec -it ntrl_ur python push_t_realworld.py --dry-run   # no motion, try the keys
    docker exec -it ntrl_ur python push_t_realworld.py             # live
    docker exec -it ntrl_ur python push_t_realworld.py --view-yaw 90
"""

import argparse
import json
import math
import os
import select
import sys
import termios
import time
import tty
from datetime import datetime

import numpy as np

from real_world_params import PARAMS

# The repo directory -- the same place ``real_world_params.json`` lives.  Inside the
# Docker image this is ``/workspace`` (the mounted host repo), NOT ``/home/jeffrey/...``,
# so ``--out`` is resolved against this rather than the cwd or a hard-coded host path.
REPO_DIR = os.path.dirname(os.path.abspath(__file__))

ROBOT_IP = PARAMS.robot_ip

# Robot modes reported by RTDEReceiveInterface.getRobotMode().
ROBOT_MODE_RUNNING = 7
ROBOT_MODE_NAMES = {
    -1: "NO_CONTROLLER", 0: "DISCONNECTED", 1: "CONFIRM_SAFETY", 2: "BOOTING",
    3: "POWER_OFF", 4: "POWER_ON", 5: "IDLE", 6: "BACKDRIVE", 7: "RUNNING",
    8: "UPDATING_FIRMWARE",
}
SAFETY_MODE_NAMES = {
    1: "NORMAL", 2: "REDUCED", 3: "PROTECTIVE_STOP", 4: "RECOVERY", 5: "SAFEGUARD_STOP",
    6: "SYSTEM_EMERGENCY_STOP", 7: "ROBOT_EMERGENCY_STOP", 8: "VIOLATION", 9: "FAULT",
}

JOG_STEP = 0.01                            # fixed translation per keypress [m]
YAW_STEP = math.radians(5.0)               # per keypress rotation about the down axis


# ======================================================================================
# orientation: the "pointing straight down" family
# ======================================================================================
def rotvec_to_matrix(rv):
    """Rodrigues: rotation vector -> 3x3 rotation matrix."""
    rv = np.asarray(rv, dtype=float)
    theta = float(np.linalg.norm(rv))
    if theta < 1e-12:
        return np.eye(3)
    k = rv / theta
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


def down_matrix(yaw):
    """Rotation whose tool Z is base -Z (straight down) and whose tool X is at `yaw`.

        R = [[ cos y,  sin y,  0],
             [ sin y, -cos y,  0],
             [     0,      0, -1]]

    Columns are the tool axes in base coordinates; X x Y = Z = (0,0,-1), so this is a
    proper right-handed rotation, not a reflection.
    """
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, s, 0.0], [s, -c, 0.0], [0.0, 0.0, -1.0]])


def down_rotvec(yaw):
    """Rotation vector for `down_matrix(yaw)`.

    Every such matrix is a half-turn (its trace is -1), so the general matrix logarithm
    degenerates.  Using R = 2*k*k^T - I for a pi rotation gives (R + I)/2 = k*k^T, whose
    diagonal reads off k = (cos(yaw/2), sin(yaw/2), 0) -- exact, and with no branch.
    """
    return np.array([math.pi * math.cos(yaw / 2.0), math.pi * math.sin(yaw / 2.0), 0.0])


def yaw_of(pose):
    """Yaw of a TCP pose about the vertical, i.e. the heading of the tool X axis."""
    R = rotvec_to_matrix(pose[3:6])
    return math.atan2(R[1, 0], R[0, 0])


def tilt_from_down_deg(pose):
    """Angle between the tool Z axis and straight down, in degrees."""
    R = rotvec_to_matrix(pose[3:6])
    return math.degrees(math.acos(float(np.clip(-R[2, 2], -1.0, 1.0))))


def down_pose(position, yaw):
    """Assemble a UR pose (x,y,z,rx,ry,rz) that points straight down at `yaw`."""
    return np.concatenate([np.asarray(position, dtype=float), down_rotvec(yaw)])


# ======================================================================================
# keyboard
# ======================================================================================
KEY_UP, KEY_DOWN, KEY_RIGHT, KEY_LEFT = "UP", "DOWN", "RIGHT", "LEFT"
KEY_PGUP, KEY_PGDN, KEY_ESC = "PGUP", "PGDN", "ESC"

_ESCAPES = {
    "[A": KEY_UP, "[B": KEY_DOWN, "[C": KEY_RIGHT, "[D": KEY_LEFT,
    "OA": KEY_UP, "OB": KEY_DOWN, "OC": KEY_RIGHT, "OD": KEY_LEFT,   # application mode
    "[5~": KEY_PGUP, "[6~": KEY_PGDN,
}


class RawKeyboard:
    """Put stdin in cbreak mode and hand back decoded keys without blocking."""

    # keys that must never be discarded by flush() -- they are intent, not stale motion
    KEEP_ON_FLUSH = (" ", "\r", "\n", "q", "u")

    def __init__(self):
        self.fd = sys.stdin.fileno()
        self._saved = None
        self._pending = []          # keys rescued from a flush(), replayed on next poll()

    def __enter__(self):
        self._saved = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, *exc):
        if self._saved is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self._saved)

    def poll(self, timeout):
        """Return every key pressed within `timeout` seconds, in order.

        Reads the whole pending buffer at once so that held-down arrow keys arrive as a
        batch the caller can coalesce, rather than one stale jog per iteration.
        """
        pending, self._pending = self._pending, []
        if not select.select([self.fd], [], [], timeout)[0]:
            return pending
        data = os.read(self.fd, 1024).decode("utf-8", errors="ignore")
        return pending + self._tokenize(data)

    def flush(self):
        """Drop stale motion typed while the arm was moving, but keep record/save/quit."""
        buf = ""
        while select.select([self.fd], [], [], 0)[0]:
            buf += os.read(self.fd, 1024).decode("utf-8", errors="ignore")
        self._pending += [k for k in self._tokenize(buf) if k in self.KEEP_ON_FLUSH]

    @staticmethod
    def _tokenize(data):
        keys, i = [], 0
        while i < len(data):
            ch = data[i]
            if ch != "\x1b":
                keys.append(ch)
                i += 1
                continue
            # An escape sequence, or a bare ESC if nothing recognisable follows.
            matched = None
            for length in (3, 2):
                seq = data[i + 1:i + 1 + length]
                if seq in _ESCAPES:
                    matched = (_ESCAPES[seq], 1 + length)
                    break
            if matched is None:
                keys.append(KEY_ESC)
                i += 1
            else:
                keys.append(matched[0])
                i += matched[1]
        return keys


# ======================================================================================
# calibration session
# ======================================================================================
class Calibration:
    """Jog the tool around, record poses, write them out as JSON."""

    def __init__(self, args):
        self.args = args
        self.point = None      # the single recorded calibration pose
        self.points = []       # [self.point] when recorded -- kept for back-compat readers
        self.saved_path = None
        self.message = ""      # feedback from a discrete action, cleared each batch
        self.warn = ""         # clamp / refusal from the last move
        self.rtde_c = None
        self.rtde_r = None
        self.live = False      # True once a control interface exists and may move the arm

        # Arrow-key basis in the base XY plane, rotated to the operator's viewpoint.
        v = math.radians(args.view_yaw)
        self.right = np.array([math.cos(v), math.sin(v), 0.0])
        self.forward = np.array([-math.sin(v), math.cos(v), 0.0])

    # -- connection ---------------------------------------------------------------
    def connect(self):
        from rtde_receive import RTDEReceiveInterface

        print(f"connecting to {self.args.ip} ...")
        self.rtde_r = RTDEReceiveInterface(self.args.ip)
        mode, safety = self.rtde_r.getRobotMode(), self.rtde_r.getSafetyMode()
        print(f"  robot mode  : {mode} ({ROBOT_MODE_NAMES.get(mode, '?')})")
        print(f"  safety mode : {safety} ({SAFETY_MODE_NAMES.get(safety, '?')})")

        if self.args.dry_run:
            print("  --dry-run: no control interface, nothing will move.")
            return
        if mode != ROBOT_MODE_RUNNING:
            raise SystemExit(
                f"\nrobot is in mode {mode} ({ROBOT_MODE_NAMES.get(mode, '?')}), not RUNNING.\n"
                "Power on and release the brakes on the teach pendant (and put it in Remote\n"
                "Control if the control interface refuses to connect), or re-run with --dry-run."
            )

        from rtde_control import RTDEControlInterface

        self.rtde_c = RTDEControlInterface(self.args.ip)
        self.live = True
        print(f"  TCP offset  : {np.round(self.rtde_c.getTCPOffset(), 5).tolist()}")

    def close(self):
        if self.rtde_c is not None:
            self.rtde_c.stopScript()
            self.rtde_c.disconnect()
        if self.rtde_r is not None:
            self.rtde_r.disconnect()

    # -- motion -------------------------------------------------------------------
    def clamp(self, position):
        """Clamp a target into the workspace box; returns (position, was_clamped)."""
        lo = np.array([self.args.xlim[0], self.args.ylim[0], self.args.zlim[0]])
        hi = np.array([self.args.xlim[1], self.args.ylim[1], self.args.zlim[1]])
        clamped = np.clip(position, lo, hi)
        return clamped, bool(np.any(np.abs(clamped - position) > 1e-9))

    def move_to(self, position, yaw):
        """Send one bounded moveL to a straight-down pose. Returns True if accepted."""
        position, was_clamped = self.clamp(position)
        target = down_pose(position, yaw)

        self.warn = "at workspace limit" if was_clamped else ""

        if not self.live:                            # dry run: accept, move nothing
            self.position, self.yaw = position, yaw
            return True

        if not self.rtde_c.isPoseWithinSafetyLimits(target.tolist()):
            self.warn = "REFUSED: outside the robot's safety limits"
            return False

        if not self.rtde_c.moveL(target.tolist(), self.args.speed, self.args.accel):
            self.warn = "REFUSED: moveL failed"
            return False
        self.position, self.yaw = position, yaw
        return True

    # -- point -------------------------------------------------------------------
    def record(self):
        """Capture the current TCP pose as THE calibration point and save straight away.

        Only one point is kept -- pressing SPACE again overwrites it.
        """
        pose = (list(self.rtde_r.getActualTCPPose()) if self.live
                else down_pose(self.position, self.yaw).tolist())
        self.point = {
            "name": "p0",
            "tcp_pose": [float(v) for v in pose],
            "joints": ([float(v) for v in self.rtde_r.getActualQ()] if self.live else None),
            "commanded": [float(v) for v in down_pose(self.position, self.yaw)],
            "yaw_rad": float(self.yaw),
            "time": datetime.now().isoformat(timespec="seconds"),
        }
        self.points = [self.point]              # keep list-shaped for anything that reads it
        self.save()

    def save(self):
        if not self.points:
            self.message = "nothing to save -- press SPACE first"
            return
        payload = {
            "robot_ip": self.args.ip,
            "created": datetime.now().isoformat(timespec="seconds"),
            "dry_run": bool(self.args.dry_run),
            "view_yaw_deg": float(self.args.view_yaw),
            "start_pose": [float(v) for v in self.start_pose],
            "pose_convention": "UR base frame (x,y,z,rx,ry,rz); metres, rotation vector",
            "point": self.point,
            "points": self.points,                 # back-compat: same single point in a list
        }
        try:
            out = self._write_json(payload)
        except OSError as exc:
            self.saved_path = None
            self.message = f"SAVE FAILED: {exc}"
            sys.stdout.write("\n" + self.message + "\n")
            sys.stdout.flush()
            return
        self.saved_path = out
        self.message = f"saved -> {out}"
        # a real newline so the confirmation survives the next keypress / status redraw
        sys.stdout.write(f"\n[saved] {out}\n")
        sys.stdout.flush()

    def _resolve_out(self):
        """``--out`` as an absolute path that actually lands in the repo.

        * relative path            -> under ``REPO_DIR`` (not the cwd)
        * absolute, parent exists   -> used as given
        * absolute, parent missing  -> re-rooted at ``REPO_DIR`` from the first path
          component that is a real repo directory (covers passing a host path like
          ``/home/jeffrey/ntrlshape_arm/calibration/arm/startpos.json`` while running
          inside the container, where the repo is mounted at ``/workspace``)
        """
        out = os.path.expanduser(self.args.out)
        if not os.path.isabs(out):
            return os.path.join(REPO_DIR, out)
        if os.path.isdir(os.path.dirname(out)):
            return out
        # Absolute path whose directory does not exist here: re-root the longest tail whose
        # first component is a real directory in the repo (e.g. .../calibration/arm/x.json
        # -> REPO_DIR/calibration/arm/x.json).
        comps = [p for p in out.split("/") if p]
        for i in range(len(comps) - 1):
            if os.path.isdir(os.path.join(REPO_DIR, comps[i])):
                return os.path.join(REPO_DIR, *comps[i:])
        return os.path.join(REPO_DIR, comps[-1])

    def _write_json(self, payload):
        """Atomically write ``payload`` to the resolved ``--out`` (or ``~`` if that dir is
        not writable); returns the path actually written."""
        target = self._resolve_out()
        for path in (target, os.path.join(os.path.expanduser("~"),
                                          os.path.basename(target))):
            try:
                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                tmp = f"{path}.tmp"
                with open(tmp, "w") as fh:
                    json.dump(payload, fh, indent=2)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp, path)
                if path != target:                 # fell back to the home dir
                    self.warn = f"(--out dir not writable; wrote {path})"
                return path
            except OSError:
                continue
        raise OSError(f"could not write {target} or a fallback")

    # -- display ------------------------------------------------------------------
    def banner(self):
        print(f"""
{'=' * 78}
  push-T real-world calibration -- tool locked pointing straight down
{'=' * 78}
  arrows  jog 1 cm in the table plane   (up/down = forward/back, left/right)
  w / s   jog 1 cm +Z / -Z  (up / down in height)   [PgUp / PgDn also work]
  , / .   yaw -5 / +5 deg about the vertical
  SPACE   record THE point and save it now     u  drop it      l  show it
  o       re-level (snap exactly down, keep yaw)  h  go back to start pose
  ENTER   re-save          q / ESC   quit (saves if a point is recorded)

  one point only -- SPACE overwrites it and writes
    {self._resolve_out()}
  one keypress = one 1 cm move; keys pressed while the arm moves are ignored
{'-' * 78}
  arrow frame: right = base {np.round(self.right, 3).tolist()},
               up    = base {np.round(self.forward, 3).tolist()}
  workspace  : x {self.args.xlim}  y {self.args.ylim}  z {self.args.zlim}  [m]
{'=' * 78}
""")

    def status(self):
        actual = (np.array(self.rtde_r.getActualTCPPose()[:3]) if self.live
                  else self.position)
        line = (f"\r x={actual[0]:+.4f} y={actual[1]:+.4f} z={actual[2]:+.4f} m | "
                f"yaw={math.degrees(self.yaw):+7.2f} deg | pts={len(self.points)}")
        if self.args.dry_run:
            line += " | DRY-RUN"
        for extra in (self.message, self.warn):
            if extra:
                line += f" | {extra}"
        sys.stdout.write(line.ljust(140))
        sys.stdout.flush()

    # -- main loop ----------------------------------------------------------------
    def run(self):
        self.connect()

        self.start_pose = np.array(self.rtde_r.getActualTCPPose(), dtype=float)
        self.position = self.start_pose[:3].copy()
        self.yaw = yaw_of(self.start_pose)
        tilt = tilt_from_down_deg(self.start_pose)
        print(f"  start TCP   : {np.round(self.start_pose, 5).tolist()}")
        print(f"  yaw         : {math.degrees(self.yaw):.2f} deg")
        print(f"  tool tilt   : {tilt:.2f} deg off straight down")

        if tilt > self.args.max_initial_tilt:
            raise SystemExit(
                f"\nthe tool is {tilt:.1f} deg off vertical, more than --max-initial-tilt "
                f"({self.args.max_initial_tilt} deg).\nJog it roughly down on the pendant "
                "first -- levelling it from here would be a large uncommanded wrist motion."
            )

        self.banner()
        if not self.args.dry_run:
            input("press ENTER to level the tool and begin (Ctrl-C to abort) ...")
            print("levelling ...")
            self.move_to(self.position, self.yaw)

        with RawKeyboard() as kb:
            self.status()
            while True:
                keys = kb.poll(0.05)
                if not keys:
                    continue

                self.message = ""         # stale feedback should not outlive its keypress

                # Process every key in the batch.  Motion keys are coalesced -- only the
                # first jog in a batch runs, the rest are ignored -- but SPACE / ENTER / q
                # are ALWAYS handled, never dropped just because a jog came first.
                moved = False
                for key in keys:
                    is_motion = key in (KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT,
                                        KEY_PGUP, KEY_PGDN, "w", "s", ",", ".", "o", "h")

                    if key in ("q", KEY_ESC):
                        if self.points and self.saved_path is None:
                            self.save()
                        print()
                        return

                    if is_motion:
                        if moved:
                            continue          # already jogged once this batch
                        moved = True
                        if key == KEY_UP:
                            self.move_to(self.position + JOG_STEP * self.forward, self.yaw)
                        elif key == KEY_DOWN:
                            self.move_to(self.position - JOG_STEP * self.forward, self.yaw)
                        elif key == KEY_RIGHT:
                            self.move_to(self.position + JOG_STEP * self.right, self.yaw)
                        elif key == KEY_LEFT:
                            self.move_to(self.position - JOG_STEP * self.right, self.yaw)
                        elif key in ("w", KEY_PGUP):
                            self.move_to(self.position + JOG_STEP * np.array([0.0, 0.0, 1.0]),
                                         self.yaw)
                        elif key in ("s", KEY_PGDN):
                            self.move_to(self.position - JOG_STEP * np.array([0.0, 0.0, 1.0]),
                                         self.yaw)
                        elif key == ".":
                            self.move_to(self.position, self.yaw + YAW_STEP)
                        elif key == ",":
                            self.move_to(self.position, self.yaw - YAW_STEP)
                        elif key == "o":
                            if self.live:
                                actual = np.array(self.rtde_r.getActualTCPPose(), dtype=float)
                                self.position, self.yaw = actual[:3].copy(), yaw_of(actual)
                                self.message = (f"re-levelled from "
                                                f"{tilt_from_down_deg(actual):.2f} deg")
                            self.move_to(self.position, self.yaw)
                        elif key == "h":
                            self.message = "returning to start pose"
                            self.status()
                            self.move_to(self.start_pose[:3], yaw_of(self.start_pose))
                        continue

                    if key == " ":
                        self.record()                 # capture the point AND save it
                    elif key in ("\r", "\n"):
                        self.save()                   # re-save on demand
                    elif key == "u":
                        if self.points:
                            self.point, self.points, self.saved_path = None, [], None
                            self.message = "dropped the recorded point"
                        else:
                            self.message = "no point to drop"
                    elif key == "l":
                        sys.stdout.write("\n")
                        if self.point:
                            print(f"  p0: {np.round(self.point['tcp_pose'], 5).tolist()}"
                                  f"   saved={self.saved_path}")
                        else:
                            print("  (no point recorded yet)")

                self.status()
                if moved:
                    kb.flush()


# ======================================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ip", default=ROBOT_IP, help="robot IP (default %(default)s)")
    ap.add_argument("--dry-run", action="store_true",
                    help="read the robot but never command motion; the keys still work")
    ap.add_argument("--speed", type=float, default=0.05, help="moveL tool speed [m/s]")
    ap.add_argument("--accel", type=float, default=0.3, help="moveL tool accel [m/s^2]")
    ap.add_argument("--view-yaw", type=float, default=0.0,
                    help="rotate the arrow-key directions in the base XY plane [deg]; "
                         "0 means right arrow = base +X, up arrow = base +Y")
    _box = PARAMS.workspace_box_m   # real_world_params.json -> robot.workspace_box_m
    ap.add_argument("--xlim", type=float, nargs=2, default=_box["x"], metavar=("LO", "HI"))
    ap.add_argument("--ylim", type=float, nargs=2, default=_box["y"], metavar=("LO", "HI"))
    ap.add_argument("--zlim", type=float, nargs=2, default=_box["z"], metavar=("LO", "HI"),
                    help="height bounds [m]; the lower one is what keeps the pusher off the table")
    ap.add_argument("--max-initial-tilt", type=float, default=25.0,
                    help="refuse to start if the tool is further than this off vertical [deg]")
    ap.add_argument("--out", default="calibration/arm/startpos.json",
                    help="where the recorded point is written; a relative path is resolved "
                         "against the repo dir, not the cwd (default: %(default)s)")
    args = ap.parse_args()

    cal = Calibration(args)
    try:
        cal.run()
    except KeyboardInterrupt:
        print("\ninterrupted.")
        if cal.points and cal.saved_path is None:
            cal.save()
    finally:
        cal.close()
    if cal.saved_path:
        print(f"calibration point saved -> {cal.saved_path}")
    elif cal.points:
        print("a point was recorded but NOT saved; re-run and press SPACE. Target: "
              f"{cal._resolve_out()}")
    else:
        print("no point recorded.")


if __name__ == "__main__":
    main()
