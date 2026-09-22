# ntrlshape_arm

Eikonal-planner pushing, in sim and on a physical UR5 + RealSense table.  Three shapes:
the **T**, a **rectangle** and a **V**.

## Which shape -- `--shape-name`

Every entry point takes `--shape-name {T,rect,V}` (`T` by default, or `$NTRLSHAPE_SHAPE`):

```
python push_t_demo_sim.py      --shape-name V
python push_t_realworld_run.py --shape-name rect --viz
python locate_functions.py     --shape-name V
python shapes.py                          # what all three resolve to, and what is missing
```

`shapes.py` is the registry and the only place a shape is described.  One entry carries
its mesh, its z-up twin (optional), its test-set directory, its checkpoint, its goal pose,
its ArUco marker positions and how much smaller the cardboard is than the mesh.
Everything else about a shape -- the outline, the bounding box in centimetres, the area
centroid, the thickness -- is **read off the mesh**, so there are no proportions to keep
in step with the shape the checkpoint was trained on.  The size in centimetres is not even
a free choice: the field is 350 mesh units across 70 cm, so a mesh unit is 2 mm and the
T's 60-unit box *must* be built 12 cm across.

Four directories hold the per-shape files:

| where | what | whose frame |
|---|---|---|
| `datasets/3dshape/*.obj` | the mesh the checkpoint was **trained** on | defines the planner's pose frame; do not change |
| `testing_data/3dshape/<shape>_env4/` | that checkpoint's test set — `meta.json`'s `env_scale` / `env_center`, and the planner's env/speed fields | must be built against `2denv4` |
| `checkpoints/<shape>.pt` | the Eikonal checkpoint | pairs with the test set above |
| `shapes/<shape>.obj` | the **real, manufactured** object, in millimetres | what the pusher touches — the IK footprint |

Adding a shape is one dict entry in `shapes.SHAPES`, plus the things only you can supply:

1. **the trained mesh** in `datasets/3dshape/`, and the **real** CAD `.obj` in `shapes/`;
2. **a test set built against `2denv4`** — the env that is physically on the table, 350
   units across the 70 cm field — in `testing_data/3dshape/`, and its **checkpoint** in
   `checkpoints/`.  A dataset generated against any other env normalizes differently and
   every planned pose lands somewhere else; `shapes.ShapeSpec.check()` says so at startup,
   and every entry point prints it before touching the camera or the arm;
3. **the goal pose** — `goal_pose_norm`, in the planner's normalized frame.  It is per
   shape because a goal only means something for the shape it was chosen for;
4. **the marker positions**, measured with a ruler (see *Measuring a marker in* below)
   into that shape's `markers` dict.  An id left `None` is skipped, not guessed at, so
   the stack runs on however many are filled in.

`--shape-name` has to take effect before `frame_conversions` is imported -- that module
bakes the shape's geometry into function default arguments, which bind at def time -- so
each entry point peeks `sys.argv` for it at the top of the file (`shapes.select_from_argv`)
and argparse then validates it.  Switching shapes mid-process raises rather than
half-applying.

`--ckpt`, `--data-path` and `--goal-norm` on `push_t_realworld_run.py` override that
shape's registry entry for one run.

## Docker

```
sudo docker run \
  --env="DISPLAY" \
  --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  --volume="/home/jeffrey/ntrlshape_arm:/workspace" \
  --volume="/usr/lib/x86_64-linux-gnu/:/glu" \
  --env="QT_X11_NO_MITSHM=1" \
  --runtime=nvidia \
  --privileged \
  --volume="/dev:/dev" \
  --network=host \
  -ti --rm \
  ntrlshapelocal
```

- `--privileged --volume=/dev:/dev` : USB access for the RealSense (pyrealsense2 / `cameratest.py`).
  check inside the box: `python -c "import pyrealsense2 as rs; print(rs.context().query_devices())"`
  a plain UVC webcam instead only needs `--device=/dev/video0`
- `--network=host` : lets the real-world scripts reach the UR5 over RTDE
- before the first GUI app : run `xhost +local:root` on the host so X accepts the container

All Python deps (torch, pymunk, viser, trimesh, shapely, ur_rtde, pyrealsense2, …) live
in the image — see `Dockerfile`. The scripts do not run on the bare host.

## Simulation

`push_t_demo_sim.py` — plan an SE(2) path for the shape with the Eikonal planner, then
push it there with a single cylindrical pusher using measured push primitives, closed-loop
at the action level and (optionally) re-planned per action.  `--shape-name` sets `--env`,
`--shape`, `--shape-zup`, `--dataPath` and `--ckpt` in one go; each is still overridable
on its own.

```
python push_t_demo_sim.py                        # viser at http://localhost:8080
python push_t_demo_sim.py --shape-name V         # the V instead of the T
python push_t_demo_sim.py --case 7 --port 8081
python push_t_demo_sim.py --save-path plan.npy    # dump the planned T path and exit
python push_t_demo_sim.py --traj plan.npy         # replay a saved path (no torch needed)
python push_t_demo_sim.py --headless              # physics only, prints a tracking report
```

Two transit modes for how the pusher reaches each entry point: teleport (default) or
`--teleport-interp` (the pusher flies — retract, arc/slide around the T, re-enter).

## Real world

### `push_t_demo_realworld.py` — sim → table bridge + the pushing stack

Ports the sim controller to the physical arm. **Only the interpolated ("linear interp")
transit mode is supported** — a real arm never teleports — so `make_args()` forces
`teleport_interp=True` and there is no plain-teleport path.

Frames:
- **real** — table frame from `camera_test_id10.py`: origin at the **ID-10** corner marker,
  `+x` along the length, `+y` along the width, centimetres.
- **sim** — pymunk world frame, raw mesh units.
- **base** — UR base frame, metres + rotation vector.

**Every frame hop is imported from `frame_conversions.py`, not re-derived here**, so this
module, `camera_test_id10.py`, `locate_functions.py` and `arm_test_id10.py` cannot drift
apart. `real_to_sim` / `sim_to_real` (+ the `_pose` twins) are `table_to_sim` /
`sim_to_table` plus the one hop this file owns — normalized → pymunk mesh units,
`* ENV_SCALE + ENV_CENTER`. Those two are read off the env mesh that is **on the table**
(`shapes.env_norm(TABLE_ENV)`: longest plan extent, bounding-box centre — 350 about
(12.747, 45.972) for `2denv4`), not off the shape's `meta.json`, so the table lands in the
same place in sim units for every shape; a dataset built against that env records the
same numbers, and `ShapeSpec.check()` flags one that does not. `CALIB_OFFSET_X/Y` and
`CALIB_ROTATION_DEG` are now aliases of `frame_conversions.FIELD_CENTER_CM` /
`ROTATION_DEG`; all the numbers live in `real_world_params.json`.

The plan's two ends: **goal** is `GOAL_POSE_NORM` — an alias for the active shape's
`shapes.SHAPES[...].goal_pose_norm`, since a goal pose only means anything for the shape
it was chosen for — written in the planner's **normalized** frame — `(x, y, rz)` or the full `(x, y, z, rx, ry, rz)`, with
`x, y` in the env's `[-0.5, 0.5]²` box (`(0,0)` = env centre) and `rz` in **turns**
(`0.25` = 90°), i.e. the same numbers as the test set's `sampled_points.npy`. Edit that
shape's entry to move the target, or pass `--goal-norm X Y TURNS` for one run; `Geometry.goal_pose_real()` / `PushStack.goal_real()` read it back in
table cm. **Start** is always the shape's measured pose, in table cm, passed in as
`build_stack(start_pose_real=(x_cm, y_cm, theta_rad))` (`push_t_realworld_run.py` opens
the camera and reads the T before planning). The checkpoint's `--case` test-set pair sets
neither end any more.

Building blocks:

| symbol | what it is |
|---|---|
| `make_args(**overrides)` | parameter bag, same names/defaults as `push_t_demo_sim.build_args` |
| `load_geometry(args)` → `Geometry` | torch-free half: meshes, footprints, `env_scale`/`env_center`; has `env_rings_real()` / `tee_ring_real(...)` |
| `build_stack(args, ref=None, start_pose_real=..., goal=None)` → `PushStack` | full stack: IK + calibrated primitives + `PushController` + `Replanner` (plans from the shape's measured pose to `GOAL_POSE_NORM` unless `ref` is given) |
| `ArmPushSession(stack)` | drives one push action per `step` |

`ArmPushSession`:
- `home_pose_world(tee_pose)` / `home_pose_real(tee_pose)` — the **first move**: park the
  end effector on the stand-off circle behind the reference's opening move.
- `step(tee_pose_world, pusher_pos_world)` / `step_real(tee_pose_real, pusher_xy_real)` —
  **one action**: (optionally) replan the reference from the measured T pose → pick the
  next (contact, direction, push length) from the measured primitive table → build the
  transit (retract off the current contact, arc/slide around the T, drop onto the
  hand-over point) → append the straight push. Returns a `StepPlan` with the end-effector
  waypoints in **both** sim units and table centimetres, plus per-leg speeds.
- `get_shape_pos(tee_pose_real=… | tee_pose_world=…)` — the T's pose in the planner's
  **normalized** 6-vector frame `(x, y, z, rx, ry, rz)`, the layout the Eikonal planner /
  `Replanner` consume.

Wire `step_real` into a camera + arm loop: observe the T pose and end-effector position,
execute the returned waypoints (transit legs at `transit_speed`, last leg at
`pusher_speed`), re-observe, repeat.

**Robot-frame motion.** A `RobotFrame` maps table cm → UR base frame by exactly
`arm_test_id10.py`'s click path, imported from `frame_conversions`:

```
table cm --table_to_sim--> normalized --clamp_sim--> --sim_to_robot--> base
```

anchored on **`p0`** — the TCP pose `arm_calibrate.py` records
(`calibration/arm/startpos.json`) with the end effector placed physically on the **centre
of the field** (sim `(0, 0)`). **Not the ID-10 marker**, which is where the *camera* frame
starts, `offset_to_origin_cm` (39.8, 32.5) cm away; anchoring on the marker sends every
waypoint 51 cm off, which is the first thing to check if the arm lands somewhere
surprising. `p0`'s z is the push height and its rotation vector the tool orientation held
at every waypoint — **the tool never changes height**, there is no lift or approach leg.
The rotation is not a free parameter: `table_to_sim`'s 180° and `SIM_TO_BASE`'s flip cancel
to leave table `+x` along base `+x`, so `robot.table_theta_deg` is unused. Targets are held
inside the field (`clamp_sim`) and the workspace box (`robot.workspace_box_m`) — note the
field's two low-`y` corners sit outside that box. Build it with
`RobotFrame.from_startpos()`.

Pass it to `ArmPushSession(stack, robot_frame=…)` or `step(..., robot=…)` and each transit
is also converted to `ur_rtde` calls by kind — `StepPlan.robot_move` (a `RobotMove` of
typed `RobotCmd`s):

- **edge slide** → `edge_transit_to_robot` — three `moveL`s (lift / slide / drop).
- **arc** → `arc_transit_to_robot` — `moveL` (retract) + **`moveC`** (the true circle,
  through the middle arc sample) + `moveL` (re-enter).
- the straight push is appended as a final `moveL` at push speed.

`ur_rtde` **does support curves**: `moveC` traces a real circular arc through a via point,
and `moveL`/`movePath` accept blended waypoint paths (corners rounded by a blend radius).
There is no arbitrary spline. Run `robot_move.cmds` on the `RTDEControlInterface`
(`c.as_call()` gives `(method, args)`).

**Not every `ur_rtde` build has `moveC`** — an older `RTDEControlInterface` raises
`AttributeError: no attribute 'moveC'` the first time an arc transit is executed. The arc
samples the transit was drawn from ride along on the command (`RobotCmd.path`), and
`expand_arcs(move)` turns each `moveC` into the `moveL` chain over them: the same curve at
`--arc-step` resolution, with the arm resting at each sample. **Never** just cut from the
retract point to the re-entry point — that straight line goes through the T. `RtdeArm`
checks for `moveC` at connect and falls back on its own; `--no-movec` forces the fallback
where the method does exist.

`Replanner` (imported from `push_t_demo_sim`) re-solves the T's whole reference path from
its measured pose, once per action — "move, then plan, then move the plan." Disable with
`--no-replan`.

### `push_t_realworld_run.py` — the full control loop

Ties it together:

```
home the end effector onto the stand-off circle
repeat:
    observe the T pose            (camera, table frame)
    observe the end-effector pos  (robot RTDE, table frame)
    plan one action               (ArmPushSession.step_real)
    execute it                    (moveL / moveC on the UR)
until the T is within --goal-cm of the goal, BLOCKED, or --max-actions is hit.
```

Two back ends behind small interfaces (`.read()` / `.tcp_xy_cm()` + `.move_to_cm()` +
`.run(RobotMove)`), so `run_pipeline` is hardware-agnostic:

| | real | dry run |
|---|---|---|
| arm | `RtdeArm` (`ur_rtde`, blocking moves, `isPoseWithinSafetyLimits` on every target) | `DryRunArm` (prints, tracks a virtual TCP) |
| T pose | `RealSenseTeePose` — a thin wrapper over `locate_functions.locate_shape`, so it measures *exactly* what `camera_test_id10.py` / `locate_functions.py` report | `ReplayTeePose` (walks the T along the reference; no camera, no GPU) |

The tracker imports every step rather than reimplementing it: the table frame from the
**ID-10** marker (`pnp_frame`/`depth_frame`, spun back by `origin_marker_yaw_deg`), a shape
marker through `camera_test_id10.tee_in_table` (**x, y from depth**, **θ from the plane
method**), then `frame_conversions.aruco_to_center` / `tee_theta_from_marker` for
marker→shape. Defaults — marker ids, both dictionaries, marker sizes — come from
`camera_test_id10` and `frame_conversions`, not `real_world_params.json`, so the two
programs always agree. Note there are **two** dictionaries: the table marker is
`DICT_ARUCO_ORIGINAL`, the shape's are `DICT_4X4_50`.

**Four markers on the shape, not one.** The arm spends the run leaning over the T, so a
single marker went blind exactly when the controller was closest to acting. The shape now
carries four ids on four different parts of it and any one of them fixes the whole pose:
`locate_shape` finds all four in one detection pass and reads the **lowest id in view**,
falling through to the next when one is covered. Only "not one of them visible" is a real
failure. Lowest-first rather than an average — the markers sit at different distances from
the centre, so their errors differ, and one sliding part-covered out of frame would drag a
mean without ever failing outright; reading the lowest also stops the answer dithering
between markers while several are visible. `obs['tee_id']` says which marker each frame came
off, `obs['tee_ids_seen']` / `['tee_ids_blocked']` what the frame had to offer.

**Measuring a marker in** — two numbers with a ruler. Hold the T the way it reads in text
(crossbar on top, stem hanging down) and measure each marker's centre **from the top-left
corner** of its 12 × 12 cm bounding box: `x` right along the top edge, `y` **downward and
therefore negative**. That pair is the whole entry in that shape's `markers` dict in `shapes.py`
(`frame_conversions.TEE_MARKER_POS_CM` is the active shape's copy of it):

```python
# shapes.py
"T": ShapeSpec(
    ...
    # id: (x_cm, y_cm) -- marker centre, from the TOP-LEFT corner, +x right, -y down
    markers={1: (5.5, -0.7), 2: (1.6, -0.7), 3: (9.8, -0.7), 4: (5.5, -10.3)},
),
"rect": ShapeSpec(
    ...
    real_rot_deg=90.0,          # the CAD part is landscape; the pipeline holds it portrait
    marker_frame="real",        # rows below are read off the 4.5 x 11.0 cm real part
    markers={1: (2.25, -8.75), 2: (2.25, -2.25)},   # centre line, half a width in from
    marker_yaw_deg=0.0,                             # each end; stuck on square
),
"V": ShapeSpec(
    ...
    real_rot_deg=245.0,         # the only turn carrying BOTH arm directions onto the mesh
    marker_frame="real",
    # (x_cm, y_cm, yaw_deg) -- a THIRD element gives that marker its own mount angle,
    # which the V needs: each faces along its own arm's outward normal, so they differ.
    markers={1: (9.245, -1.150, 270.0),     # right arm (outward +90), facing   +0
             2: (1.528, -6.067, 155.0)},    # left arm  (outward +155), facing -115
),
```

**Which box are you measuring in?**  `marker_frame` on the shape's entry says so:
`"real"` means the rows are read off `shapes/<shape>.obj`, the manufactured part (turned
by `real_rot_deg` into the orientation the pipeline holds the shape in) — use this for
anything measured from now on.  `"trained"` is the old behaviour, the nominal box of the
mesh in `datasets/3dshape/`; the T's four rows were taken that way, against a nominal
12 × 12 box, and are left on it.  The difference is not cosmetic: measuring the rectangle
in the trained mesh's 3.0 × 12.0 box instead of the real 4.5 × 11.0 would put both its
markers 0.75 cm off the centre line on every frame.

`marker_nominal_from_top_cm` is the *designed* spot for the lowest id — centred, that far
down — which the self-test holds the measured row against.  Only the T has one; leave it
`None` for a shape whose markers went wherever they fit, and the test is skipped rather
than reporting a false slip.

`tee_marker_offset` does the rest — the top-left→body translation, the centroid arithmetic,
and the mount turn, which is **not** in the table because all four markers are glued on the
same way round and share the one `MARKER_YAW_ON_TEE_DEG`. Switching `--center` between the
centroid and the bbox middle needs no re-measuring. A position that does not land on the shape's
material — tested against the **mesh outline**, so the V's opening and the T's notches
are both handled — is **rejected with the reason** rather than believed, so a dropped minus sign, an
`x` past an edge, or `x`/`y` swapped shows up at startup instead of biasing every frame
(tolerance `MARKER_ON_TEE_TOL_CM`, 6 mm, so a marker glued right up to an edge still passes).
An id left as `None` is **skipped, not guessed at**, so the stack runs on however many rows
are filled in; `python shapes.py`, `python locate_functions.py` and
`python frame_conversions.py` all print which those are, alongside the body-frame and
offset numbers they convert to.

```
# no hardware — exercises homing, the step loop, the live view
python push_t_realworld_run.py --dry-run --ref plan.npy --viz

# live  (needs calibration/arm/startpos.json from arm_calibrate.py)
python push_t_realworld_run.py --ip 192.168.10.2 --viz
```

The table→base transform comes from `--startpos` (default: `robot.startpos_json`);
`--z-push` overrides `p0`'s recorded height and `--no-clamp` lifts the two clamps.

Two ways to watch it. `--viz` opens `push_t_realworld_viz.TableView` **in this process**
(`--viz-port`, default 8080) and feeds it the same observations the controller acts on;
`--viz-state` instead writes `{shape, robot, reference}` to a JSON file each iteration for
`push_t_realworld_viz.py --state <same path>` in a second process. Both may be on at once.

### `push_t_realworld_viz.py` — the live view, in meshes

What the controller thinks is happening, as one viser scene in table centimetres — the
same meshes the planner reasons about, and nothing else:

- the **environment** — `2denv4.obj`'s walls and blocks (the obstacle set), solid;
- the **T** at its measured pose, and the **goal**, the same mesh ghosted, at
  `GOAL_POSE_NORM`;
- the **path the Eikonal planner solved for**, drawn in `push_t_demo_sim`'s green and
  redrawn whenever the `Replanner` re-solves it;
- the **end effector**, the pusher cylinder, at the TCP the arm reports.

No ArUco squares, no origin triads, no per-frame measurement dump: `locate_functions.py`
is the scene for *checking the table frame* (it draws the three boxes and every origin on
purpose), this one is for *watching a run*.

Table cm and sim mesh units differ by a rotation and one uniform scale (the field is
square), so each moving part is a viser frame with its body mesh — divided by
`sim_units_per_cm()` — as a child; an update is one position and one quaternion, never a
re-uploaded mesh.

```
python push_t_realworld_run.py --viz                      # in-process, live
python push_t_realworld_viz.py --state /tmp/table.json    # poll a state file
python push_t_realworld_viz.py --demo --ref plan.npy      # no hardware: walk a saved plan
python push_t_realworld_viz.py --ref plan.npy             # static layout + that plan
```

State-file schema (every key optional; last value kept):

```json
{
  "shape":     [x_cm, y_cm, theta_rad],
  "robot":     [x_cm, y_cm],
  "reference": [[x_cm, y_cm], ...]
}
```

Or import it and drive it from your loop:

```python
from push_t_realworld_viz import TableView
import push_t_demo_realworld as rw

view = TableView(rw.load_geometry())             # goal = GOAL_POSE_NORM unless passed
view.update(shape=(x, y, th), robot=(rx, ry), reference=stack.reference_real())
```

## Calibration

- `camera_test_id10.py` — live RealSense + ArUco window; defines the table frame off the
  single **ID-10** corner marker and reports what one marker can and cannot tell you.
- `shapes.py` — the shape registry: `python shapes.py` resolves all three and reports a
  missing mesh, a missing checkpoint, a dataset built against the wrong env, or a marker
  measured off the shape.
- `locate_functions.py` — where the shape is, live, in table cm (viser); `locate_shape` is the
  one measurement the real-world loop consumes.
- `arm_test_id10.py` — click the table, the arm goes there; the reference for table→base.
- `arm_calibrate.py` — jog the arm and record **p0**, the TCP pose with the end effector on
  the centre of the field.
- `calibrate_camera.py` / `take_pics.py` — RealSense intrinsics.





python push_t_realworld_run.py --viz --blend 0.01 --transit-accel 1.0 --transit-speed 200 --push-len 10 --no_trans_collision


IMPORTANT

edge_lift
