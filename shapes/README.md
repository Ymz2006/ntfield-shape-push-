# `shapes/` — the REAL objects, as measured

The `.obj` files the physical shapes on the table were manufactured from, straight out of
CAD, **in millimetres**.  Distinct from `datasets/3dshape/*.obj`, which are the meshes the
Eikonal checkpoints were *trained* on: those define the planner's pose frame and must not
change, while these are the material the pusher actually touches, and so are what the IK
footprint and the contact primitives should be struck against.

| file | footprint | thickness | trained twin (`datasets/3dshape/`) |
|---|---|---|---|
| `T.obj`    | 110.0 × 115.0 mm | 20 mm | `Tshape3d.obj`  — 60 × 60 units = 120 × 120 mm |
| `rect.obj` | 110.0 × 45.0 mm  | 35 mm | `rectangle.obj` — 15 × 60 units = 30 × 120 mm  |
| `V.obj`    | 104.0 × 114.7 mm | 45 mm | `Vshape3d.obj`  — 54.4 × 60 units = 109 × 120 mm |

Registered as `real_mesh` on each `shapes.SHAPES` entry; `ShapeSpec.real_outline_cm()`
reads the outline back in centimetres.  `python shapes.py` prints all of it.

## Two things to know before using these

**The up-axis is not the same in all three.**  `T.obj` is y-up (like the trained meshes,
which `pymunk_viser_push.MESH_TO_WORLD` assumes); `rect.obj` is x-up and `V.obj` is z-up.
The loader picks the **thinnest axis** as up rather than assuming, and each entry can name
its axis explicitly with `real_up`.  Reading them through `MESH_TO_WORLD` like a trained
mesh gives the wrong footprint for two of the three.

**They are NOT scaled 10x.**  Measured longest sides are 110.0 / 115.0 / 114.7 mm, i.e.
already the intended ~11 cm.  What they do need is the mm → mesh-unit conversion: the
field is 350 mesh units across 70 cm, so **1 cm = 5 mesh units and 1 mm = 0.5**, and a
110 mm shape is 55 units.  Dropped in raw they would read as 110 units = 22 cm, twice
too big.  `ShapeSpec.real_outline_cm()` does the conversion.

## Do not add `__init__.py`

`shapes.py` (the registry module) sits beside this directory.  A plain directory loses to
a module of the same name on Python's import path, so `import shapes` resolves to
`shapes.py` as intended — but adding `__init__.py` here would make this a package and
shadow it, breaking every entry point.
