# Brain mesh display scaffold

Turn a multi-region brain mesh (FreeSurfer or otherwise) into a kit of 3D-printable parts: each anatomical region as its own colored piece, plus anti-rotation scaffold bars and a three-legged base stand.

The generator is `brain_scaffold.py`.

## What it does

1. Loads a scene or folder where **each brain region is a separate mesh**.
2. Finds the closest surface points between regions and builds a tree that prefers those short gaps, so bar length equals the original empty space.
3. Forces **at least three** vertical bars from a base stand to the most stable inferior regions.
4. Cuts matching **hex or D-shaped** sockets into the region meshes. Those profiles cannot rotate.
5. Checks a slide-on assembly order so a region is only added if it can travel along its bar axis without hitting already-installed regions.
6. Exports individual STLs plus `assembly_sequence.json` and a colored `assembly_preview.glb`.

## Install

```bash
python -m pip install -r requirements.txt
```

`manifold3d` is required for cutting sockets. If a boolean fails, the script keeps the uncut region and writes the cutter solids into `cutters/` so you can subtract them in Blender.

Optional, more accurate collision checks:

```bash
python -m pip install python-fcl
python brain_scaffold.py brain.glb --precise-collision
```

## Run

```bash
# Multi-object GLB / GLTF / OBJ / 3MF
python brain_scaffold.py brain_regions.glb --bar-width 6 --clearance 0.3 --out ./print

# Folder of per-region STL/PLY files
python brain_scaffold.py ./meshes --socket-shape d --out ./print

# Built-in clustered spheres, useful to test the pipeline
python brain_scaffold.py --demo --out ./demo_print
```

### Parameters that matter for printing

| Flag | Default | Meaning |
| --- | --- | --- |
| `--bar-width` | 6 | Minimum bar width in mm. Hex uses flat-to-flat; D uses the round diameter. |
| `--clearance` | 0.3 | Extra width **per side** on the socket. Sliding FDM fits usually want 0.25–0.40. |
| `--socket-shape` | hex | `hex` or `d`. |
| `--socket-depth` | max(8, 1.25×width) | Blind hole depth in each region. |
| `--base-legs` | 3 | Stand connections (minimum 3). |
| `--base-gap` | 18 | Empty space from the lowest vertex down to the top of the stand. |
| `--up-axis` | z | World up. FreeSurfer RAS is usually `z`. |
| `--scale` | 1 | Scale meshes before scaffolding. |
| `--through-holes` | off | Cut sockets all the way through thin pieces. |
| `--preview-only` | off | Skip booleans; still writes bars and a preview. |

Hole width is `bar_width + 2 × clearance`. A 6 mm hex bar with 0.3 mm clearance gets a 6.6 mm flat-to-flat socket.

## Output

```
print/
  regions/<name>.stl          # one color per region
  scaffold/base_stand.stl
  scaffold/bar_<parent>_<child>.stl          # in-place pose
  scaffold/print_bar_<parent>_<child>.stl    # laid flat for slicing
  cutters/                    # socket tools if you need a manual boolean
  preview/assembly_preview.glb
  assembly_sequence.json
  README.md
```

Assemble in the order listed in `assembly_sequence.json`: stand first, then each bar into the already-placed parent, then slide the child region on along the bar. Do not install a later region first — that is the case the collision planner is preventing.

## Mesh input notes

- Units are millimeters after `--scale`.
- Regions should be separate closed meshes, not one fused brain.
- A GLB with one primitive per FreeSurfer label works. So does a directory of `Left-Thalamus.stl`, `Right-Putamen.stl`, …
- Keep a little gap between labels (as in the MRI segmentation). The bar length is that gap.
- If sockets sit on paper-thin walls, lower `--socket-depth`, raise `--min-wall`, or use `--through-holes`.

## Fit troubleshooting

- **Too tight:** raise `--clearance` by 0.05 mm and recut, or sand the bars.
- **Too loose / rotation:** lower `--clearance` or switch `--socket-shape d`.
- **Boolean failed:** install `manifold3d`, or boolean the exported cutters yourself.
- **Assembly warning:** that child’s inbound path hits a neighbor. Rotate the source MRI so up is inferior→superior, or rerun with a larger `--samples` so a cleaner attachment point is found.
