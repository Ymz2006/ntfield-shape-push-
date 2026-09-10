"""take_pics -- grab still frames from the Intel RealSense and save them as PNGs.

Live colour preview in a window.  Press **space** to save the current frame (no HUD
overlay) as a PNG under ``--out-dir`` (default ``calibration/camera/rgb``).  Files are
numbered ``0000.png``, ``0001.png``, ...; on restart the counter continues past whatever
is already in the directory, so it is safe to run again to add more shots.

Colour-only, so no depth stream is opened.

    sudo docker run \\
      --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \\
      --volume="/home/jeffrey/ntrlshape_arm:/workspace" \\
      --privileged --volume="/dev:/dev" --network=host \\
      --runtime=nvidia -ti --rm ntrlshapelocal
    python take_pics.py
    python take_pics.py --out-dir calibration/camera/rgb --width 1920 --height 1080

Keys
----
    space     save the current frame -> <out-dir>/NNNN.png
    q / ESC   quit
"""

import argparse
import os
import re
import time

import cv2
import numpy as np

from camera_intrinsics import CameraIntrinsics

WINDOW = "take_pics -- RealSense colour"


def hud(frame, lines):
    for row, text in enumerate(lines):
        y = 24 + 22 * row
        for colour, thick in (((0, 0, 0), 4), ((255, 255, 255), 1)):
            cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        colour, thick, cv2.LINE_AA)


class RealSenseColour:
    """Colour stream through ``pyrealsense2``."""

    def __init__(self, serial, width, height, fps):
        import pyrealsense2 as rs

        self.rs = rs
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        if serial:
            cfg.enable_device(str(serial))

        # The D415 on a USB-2 link can't do 1280x720@30 ("Couldn't resolve requests"),
        # so clamp to an fps the camera actually offers for this resolution.
        fps = self._pick_fps(serial, width, height, fps)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        try:
            profile = self.pipeline.start(cfg)
        except RuntimeError as exc:
            raise SystemExit(
                f"RealSense start failed: {exc}\n"
                "Plugged in and visible to the container?  Run with --privileged -v /dev:/dev.\n"
                "If the resolution is unsupported, try --width 640 --height 480."
            )
        dev = profile.get_device()
        try:
            usb = dev.get_info(rs.camera_info.usb_type_descriptor)
        except RuntimeError:
            usb = "?"
        self.desc = (f"realsense {dev.get_info(rs.camera_info.name)} "
                     f"({dev.get_info(rs.camera_info.serial_number)}): "
                     f"{width}x{height}@{fps}  USB {usb}")

    def _pick_fps(self, serial, width, height, want):
        """Highest colour fps <= want for this resolution (or the lowest on offer)."""
        rs = self.rs
        ctx = rs.context()
        devs = list(ctx.query_devices())
        if serial:
            devs = [d for d in devs if d.get_info(rs.camera_info.serial_number) == str(serial)]
        if not devs:
            raise SystemExit("no RealSense found -- plugged in and passed into the container?")
        rates = []
        for s in devs[0].query_sensors():
            for p in s.get_stream_profiles():
                try:
                    v = p.as_video_stream_profile()
                except Exception:
                    continue
                if (p.stream_type() == rs.stream.color and v.format() == rs.format.bgr8
                        and v.width() == width and v.height() == height):
                    rates.append(p.fps())
        if not rates:
            return want                       # let pipeline.start raise a clear error
        ok = [r for r in rates if r <= want]
        return max(ok) if ok else min(rates)

    def read(self):
        frames = self.pipeline.wait_for_frames()
        color = frames.get_color_frame()
        if not color:
            return False, None
        return True, np.asanyarray(color.get_data())

    def close(self):
        self.pipeline.stop()


def next_index(out_dir):
    """One past the highest ``NNNN.png`` already in ``out_dir`` (0 if empty)."""
    highest = -1
    if os.path.isdir(out_dir):
        for name in os.listdir(out_dir):
            m = re.fullmatch(r"(\d+)\.png", name)
            if m:
                highest = max(highest, int(m.group(1)))
    return highest + 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--serial", default=None, help="RealSense serial (default: first found)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30,
                    help="preferred colour fps; clamped to what the link supports")
    ap.add_argument("--out-dir", default="calibration/camera/rgb",
                    help="where PNGs are written (default %(default)s)")
    ap.add_argument("--undistort", action="store_true",
                    help="apply the checkerboard calibration to the preview and saved PNGs "
                         "(default: save raw frames -- that is what calibrate_camera.py needs)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    count = start = next_index(args.out_dir)

    cam = RealSenseColour(args.serial, args.width, args.height, args.fps)
    print(cam.desc)

    calib = CameraIntrinsics.load() if args.undistort else None
    if args.undistort:
        if calib is None:
            raise SystemExit("--undistort: no calibration/camera/intrinsics.json "
                             "-- run  python calibrate_camera.py  first")
        if not calib.matches(args.width, args.height):
            raise SystemExit(f"--undistort: intrinsics.json is {calib.width}x{calib.height}, "
                             f"capturing at {args.width}x{args.height}")
        print(f"undistorting with {calib}")
    elif CameraIntrinsics.load() is not None:
        print("note: saving RAW frames (pass --undistort to apply the calibration)")
    print(f"saving to {os.path.abspath(args.out_dir)}  (starting at {count:04d}.png)")

    message = ""
    msg_until = 0.0

    def set_msg(text):
        nonlocal message, msg_until
        message = text
        msg_until = time.time() + 3.0
        print(text)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    fps, last = 0.0, time.time()
    try:
        while True:
            ok, frame = cam.read()
            if ok and calib is not None:
                frame = calib.undistort(frame)
            if not ok:
                blank = np.zeros((240, 640, 3), np.uint8)
                hud(blank, ["no frame from camera -- retrying"])
                cv2.imshow(WINDOW, blank)
                if (cv2.waitKey(200) & 0xFF) in (27, ord("q")):
                    break
                continue

            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
            last = now

            view = frame.copy()
            lines = [f"saved: {count}    {fps:4.1f} fps"
                     + ("    [undistorted]" if calib is not None else ""),
                     "space = save    q/ESC = quit"]
            if message and time.time() < msg_until:
                lines.append(message)
            hud(view, lines)
            cv2.imshow(WINDOW, view)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord(" "):
                path = os.path.join(args.out_dir, f"{count:04d}.png")
                if cv2.imwrite(path, frame):
                    set_msg(f"saved {path}")
                    count += 1
                else:
                    set_msg(f"FAILED to write {path}")

            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cam.close()
        cv2.destroyAllWindows()
        print(f"done -- {count - start} new frame(s), {count} total in {args.out_dir}")


if __name__ == "__main__":
    main()
