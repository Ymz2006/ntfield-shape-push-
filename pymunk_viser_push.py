"""Top-down Pymunk sandbox with a viser GUI: push the T-shape around with a cylinder.

Physics is 2-D (pymunk) in a top-down plane; viser renders the real 3-D meshes on top of
it.  The mapping between the two is a single rigid transform:

    mesh (mx, my, mz)  ->  world (mx, -mz, my)

so the meshes' extrusion axis (mesh +Y) becomes world +Z ("up"), and the pymunk plane is
world XY.  A 2-D pose (x, y, angle) therefore becomes a world pose:
translation (x, y, z_rest) and a rotation of `angle` about world +Z.

Bodies:
  * 2denv4.obj   -> STATIC.  Outer wall ring + interior blocks, as pymunk segments
                    (or dynamic blocks with --dynamic-obstacles).
  * Tshape3d.obj -> DYNAMIC.  Convex-decomposed footprint, one rigid body.
  * cylinder     -> KINEMATIC.  Small radius, tall; driven from the GUI.  It is moved by
                    setting its velocity (never by teleporting) so it pushes properly.
  * floor        -> the ground plane.  In a top-down 2-D world the floor is not a collider;
                    it shows up physically as ground friction, applied to every dynamic
                    body with the standard pivot+gear "max_force, zero bias" joint pair.

Usage (from the ntrl-demo root):
    python sim/pymunk_viser_push.py                       # http://localhost:8080
    python sim/pymunk_viser_push.py --port 8081 --dynamic-obstacles
"""

import argparse
import math
import time

import numpy as np
import pymunk
import pymunk.autogeometry
import trimesh
import viser
from shapely.geometry import Polygon
from shapely.ops import unary_union

# mesh -> world:  X = mx, Y = -mz, Z = my
MESH_TO_WORLD = np.array(
    [[1.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, -1.0, 0.0],
     [0.0, 1.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 1.0]]
)

# Surface properties, as (Coulomb friction mu, restitution).  These live on the SHAPE,
# not on the contact pair: when two shapes touch, Chipmunk combines them MULTIPLICATIVELY
# (mu = a.friction * b.friction -- not the sqrt() rule Box2D uses), so the coefficient the
# solver actually applies between the pusher and the T is SURF_PUSHER[0] * SURF_TEE[0].
# push_t_demo.py reads these to build its friction cone, so the planner's contact model
# follows whatever is set here instead of being a second, independently-tuned number.
SURF_WALL = (0.6, 0.1)          # static wall ring
SURF_TEE = (0.5, 0.05)          # the pushed object
SURF_PUSHER = (0.2, 0.0)        # the cylinder; no restitution, it must not bounce off
SURF_BLOCK = (0.6, 0.05)        # interior obstacles under --dynamic-obstacles

COL_ENV = (110, 118, 132)
COL_TEE = (232, 138, 62)
COL_PUSHER = (70, 160, 230)
COL_FLOOR = (232, 232, 228)


# --------------------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------------------
def load_mesh(path):
    """Load an .obj and rotate it into the world frame (Z up, physics plane = XY)."""
    mesh = trimesh.load(path, force="mesh")
    mesh.apply_transform(MESH_TO_WORLD)
    return mesh


def footprint(mesh, simplify=0.05):
    """Vertical projection of a world-frame mesh onto the XY plane.

    The inputs are extruded prisms, so unioning every projected triangle recovers the exact
    2-D footprint (holes included).  Returns a list of shapely Polygons.
    """
    tris = mesh.vertices[mesh.faces][:, :, :2]  # (F, 3, 2)
    polys = [Polygon(t).buffer(0) for t in tris if abs(Polygon(t).area) > 1e-9]
    merged = unary_union(polys).buffer(0.01).buffer(-0.01)  # weld shared edges
    parts = list(merged.geoms) if merged.geom_type == "MultiPolygon" else [merged]
    return [p.simplify(simplify) for p in parts if p.area > 1e-6]


def convex_parts(poly, tolerance=0.1):
    """Convex decomposition of a hole-free shapely polygon -> list of vertex lists."""
    ring = list(poly.exterior.coords)[:-1]
    for verts in (ring, ring[::-1]):
        try:
            parts = pymunk.autogeometry.convex_decomposition(verts + [verts[0]], tolerance)
        except AssertionError:
            continue  # wrong winding for chipmunk; try the other one
        return [[(float(v.x), float(v.y)) for v in part] for part in parts]
    raise RuntimeError("convex decomposition failed")


def ring_segments(poly):
    """Every edge of a polygon (exterior + holes) as (a, b) pairs, for static colliders."""
    out = []
    for ring in [poly.exterior, *poly.interiors]:
        pts = list(ring.coords)
        out += [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
    return out


def outline_segments(polys, z):
    """(N, 2, 3) line-segment array for drawing polygon outlines in viser."""
    segs = []
    for poly in polys:
        for a, b in ring_segments(poly):
            segs.append([[a[0], a[1], z], [b[0], b[1], z]])
    return np.array(segs, dtype=np.float32)


def quat_z(angle):
    """wxyz quaternion for a rotation about world +Z."""
    return np.array([math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0)])


# --------------------------------------------------------------------------------------
# physics
# --------------------------------------------------------------------------------------
def pair_friction(a, b):
    """Contact friction coefficient Chipmunk uses between two surfaces.

    Chipmunk multiplies the two shape coefficients (confirmed against pymunk 7.3 by reading
    ``arbiter.friction`` in a pre_solve callback: 0.9 and 0.5 give 0.45, not sqrt(0.45)).
    """
    return a[0] * b[0]


def add_ground_friction(space, body, force, torque):
    """Top-down ground friction: velocity-only joints against the static body."""
    pivot = pymunk.PivotJoint(space.static_body, body, (0, 0), (0, 0))
    pivot.max_bias = 0.0          # no positional correction -- pure damping
    pivot.max_force = force
    gear = pymunk.GearJoint(space.static_body, body, 0.0, 1.0)
    gear.max_bias = 0.0
    gear.max_force = torque
    space.add(pivot, gear)
    return pivot, gear


class Sim:
    def __init__(self, args, env_polys, tee_poly, tee_start, pusher_start):
        self.args = args
        self.space = pymunk.Space()
        self.space.gravity = (0.0, 0.0)          # top-down: gravity is out of the plane
        self.space.damping = 0.9                 # air drag; ground friction is explicit
        self.space.iterations = 20

        self.tee_start = tee_start
        self.pusher_start = pusher_start
        self.frictions = []

        # --- environment -------------------------------------------------------------
        wall_poly, block_polys = env_polys[0], env_polys[1:]
        static = self.space.static_body
        walls = []
        for a, b in ring_segments(wall_poly):
            seg = pymunk.Segment(static, a, b, 1.0)
            seg.friction, seg.elasticity = SURF_WALL
            walls.append(seg)
        if args.dynamic_obstacles:
            self.blocks = [self._add_block(p) for p in block_polys]
        else:
            self.blocks = []
            for poly in block_polys:
                for a, b in ring_segments(poly):
                    seg = pymunk.Segment(static, a, b, 1.0)
                    seg.friction, seg.elasticity = SURF_WALL
                    walls.append(seg)
        self.space.add(*walls)

        # --- the T (dynamic) ---------------------------------------------------------
        parts = convex_parts(tee_poly)
        mass = args.tee_mass
        moment = sum(
            pymunk.moment_for_poly(mass * Polygon(p).area / tee_poly.area, p, (0, 0))
            for p in parts
        )
        self.tee = pymunk.Body(mass, moment)
        self.tee.position = tee_start
        shapes = []
        for p in parts:
            s = pymunk.Poly(self.tee, p)
            s.mass = mass * Polygon(p).area / tee_poly.area
            s.friction, s.elasticity = SURF_TEE
            shapes.append(s)
        self.space.add(self.tee, *shapes)
        self.frictions.append(
            (self.tee, *add_ground_friction(self.space, self.tee, args.friction, args.spin_friction))
        )

        # --- the pusher (kinematic) --------------------------------------------------
        self.pusher = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self.pusher.position = pusher_start
        circle = pymunk.Circle(self.pusher, args.pusher_radius)
        circle.friction, circle.elasticity = SURF_PUSHER
        self.space.add(self.pusher, circle)

        self.target = pymunk.Vec2d(*pusher_start)
        self.max_speed = args.pusher_speed

    def _add_block(self, poly):
        """An interior obstacle as a heavy dynamic body (--dynamic-obstacles)."""
        c = poly.centroid
        local = [(x - c.x, y - c.y) for x, y in list(poly.exterior.coords)[:-1]]
        mass = self.args.tee_mass * poly.area / 1000.0
        body = pymunk.Body(mass, pymunk.moment_for_poly(mass, local, (0, 0)))
        body.position = (c.x, c.y)
        shape = pymunk.Poly(body, local)
        shape.friction, shape.elasticity = SURF_BLOCK
        self.space.add(body, shape)
        self.frictions.append(
            (body, *add_ground_friction(
                self.space, body, self.args.friction * mass, self.args.spin_friction * mass))
        )
        return (body, (c.x, c.y), poly)

    def set_friction(self, force, torque):
        for body, pivot, gear in self.frictions:
            scale = body.mass / self.args.tee_mass
            pivot.max_force = force * scale
            gear.max_force = torque * scale

    def step(self, dt):
        # Kinematic bodies are driven by velocity, never by teleporting, so that contacts
        # resolve and the pusher actually pushes instead of tunnelling through.
        delta = self.target - self.pusher.position
        vel = delta / dt
        if vel.length > self.max_speed:
            vel = vel.normalized() * self.max_speed
        self.pusher.velocity = vel
        self.pusher.angular_velocity = 0.0
        self.space.step(dt)

    def reset(self):
        self.tee.position = self.tee_start
        self.tee.angle = 0.0
        self.tee.velocity = (0, 0)
        self.tee.angular_velocity = 0.0
        self.pusher.position = self.pusher_start
        self.pusher.velocity = (0, 0)
        self.target = pymunk.Vec2d(*self.pusher_start)
        for body, home, _ in self.blocks:
            body.position = home
            body.angle = 0.0
            body.velocity = (0, 0)
            body.angular_velocity = 0.0


# --------------------------------------------------------------------------------------
# app
# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", default="datasets/3dshape/2denv4.obj")
    ap.add_argument("--shape", default="datasets/3dshape/Tshape3d.obj")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--pusher-radius", type=float, default=5.0)
    ap.add_argument("--pusher-height", type=float, default=60.0)
    ap.add_argument("--pusher-speed", type=float, default=60.0, help="units / second")
    ap.add_argument("--tee-mass", type=float, default=1.0)
    ap.add_argument("--friction", type=float, default=300.0,
                    help="ground friction force on the T")
    ap.add_argument("--spin-friction", type=float, default=8000.0,
                    help="ground friction torque on the T")
    ap.add_argument("--dynamic-obstacles", action="store_true",
                    help="make the interior blocks pushable too (walls stay static)")
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--substeps", type=int, default=4)
    args = ap.parse_args()

    # ---- geometry ----------------------------------------------------------------
    env_mesh = load_mesh(args.env)
    tee_mesh = load_mesh(args.shape)

    env_polys = sorted(footprint(env_mesh), key=lambda p: -p.area)  # [0] = wall ring
    tee_polys = footprint(tee_mesh)
    assert len(tee_polys) == 1, f"expected one footprint for the shape, got {len(tee_polys)}"

    # Move the T mesh so its footprint centroid is the body origin and it rests on z=0.
    c = tee_polys[0].centroid
    z0 = tee_mesh.bounds[0][2]
    tee_mesh.apply_translation([-c.x, -c.y, -z0])
    tee_poly = Polygon([(x - c.x, y - c.y) for x, y in tee_polys[0].exterior.coords])
    tee_height = float(tee_mesh.extents[2])

    env_lo, env_hi = env_polys[0].bounds[:2], env_polys[0].bounds[2:]
    tee_start = (float(c.x), float(c.y))            # keep the T where the mesh authored it
    pusher_start = (tee_start[0], tee_start[1] - 45.0)

    sim = Sim(args, env_polys, tee_poly, tee_start, pusher_start)

    # ---- viser scene ---------------------------------------------------------------
    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")
    server.scene.world_axes.visible = False

    fw = max(env_hi[0] - env_lo[0], env_hi[1] - env_lo[1]) * 1.15
    fc = ((env_lo[0] + env_hi[0]) / 2.0, (env_lo[1] + env_hi[1]) / 2.0)
    floor = server.scene.add_box(
        "/floor", color=COL_FLOOR, dimensions=(fw, fw, 2.0), position=(fc[0], fc[1], -1.0)
    )
    grid = server.scene.add_grid(
        "/grid", width=fw, height=fw, plane="xy", cell_size=10.0, section_size=50.0,
        position=(fc[0], fc[1], 0.02),
    )

    env_node = server.scene.add_mesh_simple(
        "/env", env_mesh.vertices, env_mesh.faces, color=COL_ENV, flat_shading=True
    )
    env_outline = server.scene.add_line_segments(
        "/env_outline", outline_segments(env_polys, 0.05), colors=(40, 44, 52),
        thickness=1.0, visible=False,
    )

    # /tee is the body frame; the mesh + collision outline ride along as children.
    tee_frame = server.scene.add_frame("/tee", show_axes=False, position=(*tee_start, 0.0))
    server.scene.add_mesh_simple(
        "/tee/mesh", tee_mesh.vertices, tee_mesh.faces, color=COL_TEE, flat_shading=True
    )
    tee_outline = server.scene.add_line_segments(
        "/tee/outline",
        outline_segments([Polygon(p) for p in convex_parts(tee_poly)], tee_height + 0.5),
        colors=(120, 60, 10), thickness=1.0, visible=False,
    )

    pusher_mesh = trimesh.creation.cylinder(
        radius=args.pusher_radius, height=args.pusher_height, sections=32
    )
    pusher_mesh.apply_translation([0.0, 0.0, args.pusher_height / 2.0])
    pusher_frame = server.scene.add_frame("/pusher", show_axes=False,
                                          position=(*pusher_start, 0.0))
    server.scene.add_mesh_simple(
        "/pusher/mesh", pusher_mesh.vertices, pusher_mesh.faces, color=COL_PUSHER
    )

    block_nodes = []
    if args.dynamic_obstacles:
        wall_h = float(env_mesh.bounds[1][2])
        for i, (body, home, poly) in enumerate(sim.blocks):
            block = trimesh.creation.extrude_polygon(
                Polygon([(x - home[0], y - home[1]) for x, y in poly.exterior.coords]),
                height=wall_h,
            )
            node = server.scene.add_frame(f"/block_{i}", show_axes=False, position=(*home, 0.0))
            server.scene.add_mesh_simple(
                f"/block_{i}/mesh", block.vertices, block.faces, color=(150, 120, 100),
                flat_shading=True,
            )
            block_nodes.append((body, node))
        env_node.visible = False  # blocks are drawn separately; keep only the wall ring
        wall = trimesh.creation.extrude_polygon(env_polys[0], height=wall_h)
        server.scene.add_mesh_simple("/walls", wall.vertices, wall.faces, color=COL_ENV,
                                     flat_shading=True)

    server.scene.add_light_directional("/sun", color=(255, 255, 255), intensity=2.0,
                                       position=(fc[0] + 100, fc[1] - 150, 400))

    # ---- GUI -----------------------------------------------------------------------
    lim_x = (float(env_lo[0]) + args.pusher_radius, float(env_hi[0]) - args.pusher_radius)
    lim_y = (float(env_lo[1]) + args.pusher_radius, float(env_hi[1]) - args.pusher_radius)

    with server.gui.add_folder("Pusher (kinematic)"):
        gizmo = server.scene.add_transform_controls(
            "/pusher_target", scale=40.0, disable_rotations=True, active_axes=(True, True, False),
            translation_limits=(lim_x, lim_y, (0.0, 0.0)), opacity=0.8,
            position=(*pusher_start, args.pusher_height * 0.6),
        )
        sx = server.gui.add_slider("x", lim_x[0], lim_x[1], 0.5, pusher_start[0])
        sy = server.gui.add_slider("y", lim_y[0], lim_y[1], 0.5, pusher_start[1])
        speed = server.gui.add_slider("max speed", 5.0, 300.0, 5.0, args.pusher_speed)
        server.gui.add_markdown("Drag the blue gizmo in the scene, or use the sliders.")

    with server.gui.add_folder("Physics"):
        running = server.gui.add_checkbox("run", True)
        time_scale = server.gui.add_slider("time scale", 0.1, 2.0, 0.05, 1.0)
        fric = server.gui.add_slider("ground friction", 0.0, 2000.0, 10.0, args.friction)
        spin = server.gui.add_slider("spin friction", 0.0, 60000.0, 250.0, args.spin_friction)
        damp = server.gui.add_slider("linear damping", 0.0, 1.0, 0.01, sim.space.damping)
        reset_btn = server.gui.add_button("reset")

    with server.gui.add_folder("View"):
        show_env = server.gui.add_checkbox("environment", True)
        show_floor = server.gui.add_checkbox("floor", True)
        show_grid = server.gui.add_checkbox("grid", True)
        show_coll = server.gui.add_checkbox("collision geometry", False)

    status = server.gui.add_markdown("")

    _syncing = {"v": False}

    def push_target(x, y, from_gizmo):
        sim.target = pymunk.Vec2d(float(x), float(y))
        if _syncing["v"]:
            return
        _syncing["v"] = True
        try:
            if from_gizmo:
                sx.value, sy.value = float(x), float(y)
            else:
                gizmo.position = (float(x), float(y), args.pusher_height * 0.6)
        finally:
            _syncing["v"] = False

    @gizmo.on_update
    def _(_):
        push_target(gizmo.position[0], gizmo.position[1], True)

    @sx.on_update
    def _(_):
        push_target(sx.value, sy.value, False)

    @sy.on_update
    def _(_):
        push_target(sx.value, sy.value, False)

    @speed.on_update
    def _(_):
        sim.max_speed = speed.value

    @fric.on_update
    def _(_):
        sim.set_friction(fric.value, spin.value)

    @spin.on_update
    def _(_):
        sim.set_friction(fric.value, spin.value)

    @damp.on_update
    def _(_):
        sim.space.damping = damp.value

    @reset_btn.on_click
    def _(_):
        sim.reset()
        _syncing["v"] = True
        sx.value, sy.value = pusher_start
        gizmo.position = (*pusher_start, args.pusher_height * 0.6)
        _syncing["v"] = False

    @show_env.on_update
    def _(_):
        env_node.visible = show_env.value

    @show_floor.on_update
    def _(_):
        floor.visible = show_floor.value

    @show_grid.on_update
    def _(_):
        grid.visible = show_grid.value

    @show_coll.on_update
    def _(_):
        env_outline.visible = tee_outline.visible = show_coll.value

    print(f"[pymunk-viser] serving on http://localhost:{args.port}")
    print(f"[pymunk-viser] env parts={len(env_polys)}  T convex parts="
          f"{len(convex_parts(tee_poly))}  dynamic obstacles={args.dynamic_obstacles}")

    # ---- loop ----------------------------------------------------------------------
    frame_dt = 1.0 / args.fps
    next_frame = time.time()
    last_status = 0.0
    while True:
        if running.value:
            dt = frame_dt * time_scale.value / args.substeps
            for _ in range(args.substeps):
                sim.step(dt)

        tee_frame.position = (sim.tee.position.x, sim.tee.position.y, 0.0)
        tee_frame.wxyz = quat_z(sim.tee.angle)
        pusher_frame.position = (sim.pusher.position.x, sim.pusher.position.y, 0.0)
        for body, node in block_nodes:
            node.position = (body.position.x, body.position.y, 0.0)
            node.wxyz = quat_z(body.angle)

        now = time.time()
        if now - last_status > 0.1:
            last_status = now
            status.content = (
                f"**T**  x={sim.tee.position.x:7.1f}  y={sim.tee.position.y:7.1f}  "
                f"θ={math.degrees(sim.tee.angle) % 360:6.1f}°\n\n"
                f"**pusher**  x={sim.pusher.position.x:7.1f}  "
                f"y={sim.pusher.position.y:7.1f}  |v|={sim.pusher.velocity.length:6.1f}"
            )

        next_frame += frame_dt
        time.sleep(max(0.0, next_frame - time.time()))
        if next_frame < time.time() - 0.5:
            next_frame = time.time()


if __name__ == "__main__":
    main()
