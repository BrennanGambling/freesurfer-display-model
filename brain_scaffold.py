#!/usr/bin/env python3
"""Build a 3D-printable brain display scaffold from a multi-region mesh.

Each anatomical region is exported as its own colored-print STL with hexagonal
or D-shaped sockets cut in. Separate scaffold bars span the original surface
gaps, and at least three bars meet a base stand. Assembly order is planned so
every region can slide onto its connector without hitting already-installed
parts.

Example
-------
python brain_scaffold.py brain_regions.glb --bar-width 6 --clearance 0.3
python brain_scaffold.py ./stl_dir --socket-shape d --out ./print
python brain_scaffold.py --demo --out ./demo_print
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import trimesh
from shapely.geometry import Polygon
from trimesh.transformations import rotation_matrix

BASE_NAME = "__base__"
MESH_SUFFIXES = {".stl", ".obj", ".ply", ".glb", ".gltf", ".3mf", ".off", ".dae"}


@dataclass
class ScaffoldConfig:
    bar_width: float = 6.0
    clearance: float = 0.3
    socket_shape: str = "hex"
    socket_depth: float = 8.0
    axial_clearance: float = 0.25
    cutter_overshoot: float = 1.5
    d_flat_offset: float = 0.45
    chamfer: float = 0.4
    samples: int = 4000
    candidate_pool: int = 16
    min_wall: float = 1.5
    max_normal_angle: float = 55.0
    base_legs: int = 3
    base_gap: float = 18.0
    base_thickness: float = 6.0
    base_margin: float = 16.0
    up_axis: str = "z"
    scale: float = 1.0
    through_holes: bool = False
    preview_only: bool = False
    precise_collision: bool = False
    collision_steps: int = 18
    boolean_engine: str = "manifold"
    seed: int = 0


@dataclass
class Attachment:
    parent: str
    child: str
    parent_point: np.ndarray
    child_point: np.ndarray
    parent_depth: float
    child_depth: float
    frame: np.ndarray
    kind: str


@dataclass
class Region:
    name: str
    mesh: trimesh.Trimesh
    original: trimesh.Trimesh


@dataclass
class PairContact:
    a: str
    b: str
    pa: np.ndarray
    pb: np.ndarray
    distance: float
    normal_a: np.ndarray
    normal_b: np.ndarray
    face_a: int
    face_b: int


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("Zero-length vector")
    return v / n


def _stable_seed(*names: str, base: int = 0) -> int:
    digest = hashlib.md5("::".join(names).encode("utf-8")).hexdigest()
    return (base + int(digest[:8], 16)) % (2**31 - 1)


def _up_vector(axis: str) -> np.ndarray:
    mapping = {
        "x": np.array([1.0, 0.0, 0.0]),
        "y": np.array([0.0, 1.0, 0.0]),
        "z": np.array([0.0, 0.0, 1.0]),
        "-x": np.array([-1.0, 0.0, 0.0]),
        "-y": np.array([0.0, -1.0, 0.0]),
        "-z": np.array([0.0, 0.0, -1.0]),
    }
    key = axis.lower()
    if key not in mapping:
        raise ValueError(f"Unknown up axis {axis}")
    return mapping[key]


def make_frame(origin: np.ndarray, z_axis: np.ndarray, up: np.ndarray) -> np.ndarray:
    z = _unit(z_axis)
    helper = up.astype(float)
    if abs(np.dot(z, helper)) > 0.95:
        helper = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(helper, z)
    if np.linalg.norm(x) < 1e-9:
        x = np.cross(np.array([0.0, 1.0, 0.0]), z)
    x = _unit(x)
    y = _unit(np.cross(z, x))
    x = _unit(np.cross(y, z))
    frame = np.eye(4)
    frame[:3, 0] = x
    frame[:3, 1] = y
    frame[:3, 2] = z
    frame[:3, 3] = origin
    return frame


def transform_from_z_segment(a: np.ndarray, b: np.ndarray, up: np.ndarray) -> Tuple[np.ndarray, float]:
    height = float(np.linalg.norm(b - a))
    if height < 1e-8:
        raise ValueError("Zero-length segment")
    return make_frame(a, b - a, up), height


def hex_polygon(flat_to_flat: float) -> Polygon:
    radius = flat_to_flat / math.sqrt(3.0)
    angles = np.arange(6) * (math.pi / 3.0)
    coords = [(radius * math.cos(a), radius * math.sin(a)) for a in angles]
    return Polygon(coords)


def d_polygon(width: float, flat_offset: float, sections: int = 48) -> Polygon:
    radius = width / 2.0
    chord_x = -radius * float(np.clip(flat_offset, 0.05, 0.95))
    span = math.acos(float(np.clip(chord_x / radius, -1.0, 1.0)))
    angles = np.linspace(-span, span, sections)
    pts = [(radius * math.cos(a), radius * math.sin(a)) for a in angles]
    pts.append((chord_x, -radius * math.sin(span)))
    poly = Polygon(pts)
    if not poly.is_valid or poly.area <= 0:
        poly = Polygon(pts).buffer(0)
    return poly


def bar_polygon(cfg: ScaffoldConfig) -> Polygon:
    shape = cfg.socket_shape.lower()
    if shape in {"hex", "hexagon", "hexagonal"}:
        return hex_polygon(cfg.bar_width)
    if shape in {"d", "dshape", "d-shaped", "d_shaped"}:
        return d_polygon(cfg.bar_width, cfg.d_flat_offset)
    raise ValueError("socket_shape must be 'hex' or 'd'")


def socket_polygon(cfg: ScaffoldConfig) -> Polygon:
    return bar_polygon(cfg).buffer(cfg.clearance, join_style=2)


def extrude_profile(poly: Polygon, height: float, transform: np.ndarray) -> trimesh.Trimesh:
    if height <= 1e-8:
        raise ValueError("Extrusion height must be positive")
    mesh = trimesh.creation.extrude_polygon(poly, height)
    mesh.apply_transform(transform)
    return mesh


def oriented_prism(poly: Polygon, start: np.ndarray, end: np.ndarray, up: np.ndarray) -> trimesh.Trimesh:
    transform, height = transform_from_z_segment(start, end, up)
    return extrude_profile(poly, height, transform)


def make_bar_mesh(att: Attachment, cfg: ScaffoldConfig, up: np.ndarray) -> trimesh.Trimesh:
    z = att.frame[:3, 2]
    insert_p = max(att.parent_depth - cfg.axial_clearance, cfg.bar_width * 0.5)
    insert_c = max(att.child_depth - cfg.axial_clearance, cfg.bar_width * 0.5)
    start = att.parent_point - z * insert_p
    end = att.child_point + z * insert_c
    bar = oriented_prism(bar_polygon(cfg), start, end, up)
    bar.metadata["name"] = f"bar_{att.parent}_{att.child}"
    return bar


def make_cutter(
    point: np.ndarray,
    inward: np.ndarray,
    depth: float,
    cfg: ScaffoldConfig,
    up: np.ndarray,
    through: bool,
    mesh: trimesh.Trimesh,
) -> trimesh.Trimesh:
    poly = socket_polygon(cfg)
    if through:
        far = float(np.linalg.norm(mesh.extents) + cfg.bar_width * 4)
        start = point - inward * cfg.cutter_overshoot
        end = point + inward * far
    else:
        start = point - inward * cfg.cutter_overshoot
        end = point + inward * (depth + cfg.axial_clearance)
    return oriented_prism(poly, start, end, up)


def sanitize_name(name: str, used: set) -> str:
    raw = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name.strip())
    raw = raw.strip("_") or "region"
    candidate = raw
    i = 2
    while candidate in used:
        candidate = f"{raw}_{i}"
        i += 1
    used.add(candidate)
    return candidate


def repair_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh = mesh.copy()
    mesh.remove_infinite_values()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices()
    try:
        trimesh.repair.fix_normals(mesh)
    except Exception:
        pass
    try:
        mesh.fill_holes()
    except Exception:
        pass
    mesh.process(validate=False)
    return mesh


def _as_trimesh(geom) -> Optional[trimesh.Trimesh]:
    return geom if isinstance(geom, trimesh.Trimesh) else None


def _scene_to_named_meshes(scene: trimesh.Scene) -> List[Tuple[str, trimesh.Trimesh]]:
    out = []
    for node_name in scene.graph.nodes_geometry:
        transform, geom_name = scene.graph[node_name]
        mesh = _as_trimesh(scene.geometry.get(geom_name))
        if mesh is None:
            continue
        mesh = mesh.copy()
        mesh.apply_transform(transform)
        out.append((str(node_name), mesh))
    if not out:
        dumped = scene.dump(concatenate=False)
        if isinstance(dumped, list):
            for i, geom in enumerate(dumped):
                mesh = _as_trimesh(geom)
                if mesh is not None:
                    out.append((f"region_{i}", mesh))
    return out


def _require_regions(regions: List[Region], path: Path) -> List[Region]:
    if len(regions) < 2:
        raise ValueError(f"Need at least 2 region meshes from {path}, got {len(regions)}")
    print(f"Loaded {len(regions)} regions from {path}")
    return regions


def load_regions(path: Path) -> List[Region]:
    used: set = set()
    regions: List[Region] = []

    def add(name: str, geom) -> None:
        mesh = _as_trimesh(geom)
        if mesh is None or len(mesh.faces) == 0:
            return
        mesh = repair_mesh(mesh)
        label = sanitize_name(name, used)
        regions.append(Region(name=label, mesh=mesh, original=mesh.copy()))

    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in MESH_SUFFIXES)
        if not files:
            raise FileNotFoundError(f"No mesh files in {path}")
        for f in files:
            loaded = trimesh.load(f, force="scene")
            if isinstance(loaded, trimesh.Scene):
                dumped = _scene_to_named_meshes(loaded)
                if len(dumped) == 1:
                    add(f.stem, dumped[0][1])
                else:
                    for n, m in dumped:
                        add(f"{f.stem}_{n}", m)
            else:
                add(f.stem, loaded)
        return _require_regions(regions, path)

    loaded = trimesh.load(path, force="scene")
    if isinstance(loaded, trimesh.Scene):
        for n, m in _scene_to_named_meshes(loaded):
            add(n, m)
    else:
        add(path.stem, loaded)
    return _require_regions(regions, path)


def demo_regions(cfg: ScaffoldConfig) -> List[Region]:
    rng = np.random.default_rng(cfg.seed)
    specs = [
        ("brainstem", np.array([0.0, 0.0, 28.0]), 11.0),
        ("left_thalamus", np.array([18.0, 4.0, 42.0]), 10.0),
        ("right_thalamus", np.array([-17.0, 3.0, 41.0]), 10.0),
        ("left_cortex", np.array([26.0, -8.0, 58.0]), 14.0),
        ("right_cortex", np.array([-25.0, -6.0, 57.0]), 14.0),
        ("cerebellum", np.array([0.0, -22.0, 32.0]), 12.0),
    ]
    used: set = set()
    regions = []
    for name, center, radius in specs:
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=radius)
        mesh.apply_translation(center + rng.normal(0, 0.4, size=3))
        mesh = repair_mesh(mesh)
        label = sanitize_name(name, used)
        regions.append(Region(name=label, mesh=mesh, original=mesh.copy()))
    return regions


def _face_normal(mesh: trimesh.Trimesh, face_id: int) -> np.ndarray:
    return _unit(mesh.face_normals[int(face_id)])


def closest_contacts(a: Region, b: Region, cfg: ScaffoldConfig) -> List[PairContact]:
    rng = np.random.default_rng(_stable_seed(a.name, b.name, base=cfg.seed))
    sa = a.mesh.sample(cfg.samples, seed=int(rng.integers(1e6)))
    sb = b.mesh.sample(cfg.samples, seed=int(rng.integers(1e6)))
    cb, db, fb = a.mesh.nearest.on_surface(sb)
    ca, da, fa = b.mesh.nearest.on_surface(sa)
    contacts: List[PairContact] = []

    def consider(pa, pb, fa_id, fb_id, dist):
        if not np.isfinite(dist):
            return
        try:
            na = _face_normal(a.mesh, fa_id)
            nb = _face_normal(b.mesh, fb_id)
        except Exception:
            delta = pb - pa
            na = _unit(delta) if np.linalg.norm(delta) > 1e-9 else np.array([0.0, 0.0, 1.0])
            nb = -na
        contacts.append(
            PairContact(
                a=a.name,
                b=b.name,
                pa=np.asarray(pa, float),
                pb=np.asarray(pb, float),
                distance=float(dist),
                normal_a=na,
                normal_b=nb,
                face_a=int(fa_id),
                face_b=int(fb_id),
            )
        )

    for i in np.argsort(db)[: cfg.candidate_pool]:
        _, _, fb2 = b.mesh.nearest.on_surface([sb[i]])
        consider(cb[i], sb[i], fb[i], fb2[0], db[i])
    for i in np.argsort(da)[: cfg.candidate_pool]:
        _, _, fa2 = a.mesh.nearest.on_surface([sa[i]])
        consider(sa[i], ca[i], fa2[0], fa[i], da[i])

    contacts.sort(key=lambda c: c.distance)
    dedup: List[PairContact] = []
    for c in contacts:
        if any(
            np.linalg.norm(c.pa - d.pa) < cfg.bar_width * 0.25
            and np.linalg.norm(c.pb - d.pb) < cfg.bar_width * 0.25
            for d in dedup
        ):
            continue
        dedup.append(c)
        if len(dedup) >= cfg.candidate_pool:
            break
    return dedup


def all_pair_contacts(regions: List[Region], cfg: ScaffoldConfig) -> Dict[Tuple[str, str], List[PairContact]]:
    by_name = {r.name: r for r in regions}
    names = [r.name for r in regions]
    table: Dict[Tuple[str, str], List[PairContact]] = {}
    total = len(names) * (len(names) - 1) // 2
    done = 0
    for i, na in enumerate(names):
        for nb in names[i + 1 :]:
            table[(na, nb)] = closest_contacts(by_name[na], by_name[nb], cfg)
            done += 1
            if done % 5 == 0 or done == total:
                print(f"  closest-point pairs {done}/{total}")
    return table


def pair_key(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a < b else (b, a)


def local_thickness(mesh: trimesh.Trimesh, point: np.ndarray, inward: np.ndarray) -> float:
    origin = point + inward * 1e-3
    try:
        locations, _, _ = mesh.ray.intersects_location(ray_origins=[origin], ray_directions=[inward])
    except Exception:
        return 0.0
    if len(locations) == 0:
        return 0.0
    return float(np.linalg.norm(locations - origin, axis=1).min())


def segment_hits_other(p0: np.ndarray, p1: np.ndarray, others: Iterable[trimesh.Trimesh], ignore_ends: float) -> bool:
    direction = p1 - p0
    length = np.linalg.norm(direction)
    if length < 1e-8:
        return False
    direction = direction / length
    origin = p0 + direction * ignore_ends
    dist = max(length - 2 * ignore_ends, 0.0)
    if dist <= 1e-6:
        return False
    for mesh in others:
        try:
            hits, _, _ = mesh.ray.intersects_location(ray_origins=[origin], ray_directions=[direction])
        except Exception:
            continue
        if len(hits) == 0:
            continue
        t = np.linalg.norm(hits - origin, axis=1)
        if np.any((t > 1e-4) & (t < dist)):
            return True
    return False


def score_contact(
    contact: PairContact,
    parent: Region,
    child: Region,
    cfg: ScaffoldConfig,
    others: List[trimesh.Trimesh],
) -> Optional[Tuple[float, PairContact, float, float]]:
    axis = contact.pb - contact.pa
    gap = float(np.linalg.norm(axis))
    z = _unit(axis) if gap > 1e-6 else _unit(contact.normal_a)
    inward_p = -z
    inward_c = z
    align_p = float(np.dot(contact.normal_a, z))
    align_c = float(np.dot(contact.normal_b, -z))
    thick_p = local_thickness(parent.mesh, contact.pa, inward_p)
    thick_c = local_thickness(child.mesh, contact.pb, inward_c)
    depth_p = min(cfg.socket_depth, max(thick_p - cfg.min_wall, cfg.bar_width * 0.6))
    depth_c = min(cfg.socket_depth, max(thick_c - cfg.min_wall, cfg.bar_width * 0.6))
    if thick_p < cfg.min_wall + cfg.bar_width * 0.4 or thick_c < cfg.min_wall + cfg.bar_width * 0.4:
        return None
    if segment_hits_other(contact.pa, contact.pb, others, ignore_ends=0.4):
        return None
    score = gap
    score += (1.0 - max(align_p, 0.0)) * cfg.bar_width * 2
    score += (1.0 - max(align_c, 0.0)) * cfg.bar_width * 2
    return score, contact, depth_p, depth_c


class CollisionWorld:
    def __init__(self, precise: bool):
        self.precise = precise
        self.meshes: Dict[str, trimesh.Trimesh] = {}
        self._manager = None
        if precise:
            try:
                self._manager = trimesh.collision.CollisionManager()
            except Exception:
                self._manager = None

    def add(self, name: str, mesh: trimesh.Trimesh) -> None:
        self.meshes[name] = mesh
        if self._manager is not None:
            try:
                self._manager.add_object(name, mesh)
            except Exception:
                pass

    def collides(self, mesh: trimesh.Trimesh, clearance: float = 0.0) -> bool:
        probed = mesh
        if clearance > 0:
            try:
                probed = mesh.offset(clearance)
            except Exception:
                probed = mesh
        if self._manager is not None:
            try:
                return bool(self._manager.in_collision_single(probed))
            except Exception:
                pass
        pts = probed.sample(min(1500, max(200, len(probed.faces) // 2)))
        for other in self.meshes.values():
            if not _aabb_overlap(probed.bounds, other.bounds, pad=clearance + 0.2):
                continue
            try:
                sd = trimesh.proximity.signed_distance(other, pts)
                if np.any(sd > 0.15):
                    return True
            except Exception:
                _, dist, _ = other.nearest.on_surface(pts)
                if other.is_watertight:
                    try:
                        if np.any(other.contains(pts)):
                            return True
                    except Exception:
                        return True
                elif np.any(dist < 0.05):
                    return True
        return False


def _aabb_overlap(a: np.ndarray, b: np.ndarray, pad: float) -> bool:
    return bool(np.all(a[0] <= b[1] + pad) and np.all(b[0] <= a[1] + pad))


def insertion_blocked(child_mesh: trimesh.Trimesh, away: np.ndarray, world: CollisionWorld, travel: float, steps: int) -> bool:
    away = _unit(away)
    for t in np.linspace(travel, 0.6, steps):
        moved = child_mesh.copy()
        moved.apply_translation(away * t)
        if world.collides(moved):
            return True
    return False


def required_travel(child: trimesh.Trimesh, assembled: List[trimesh.Trimesh], away: np.ndarray) -> float:
    if not assembled:
        return float(np.linalg.norm(child.extents) * 1.2)
    amin = np.min([m.bounds[0] for m in assembled], axis=0)
    amax = np.max([m.bounds[1] for m in assembled], axis=0)
    diag = float(np.linalg.norm(child.bounds[1] - child.bounds[0]) + np.linalg.norm(amax - amin))
    return max(diag * 0.65, float(np.linalg.norm(child.extents)) * 1.5)


def region_lowest_point(region: Region, up: np.ndarray) -> np.ndarray:
    dots = region.mesh.vertices @ up
    return region.mesh.vertices[int(np.argmin(dots))].copy()


def choose_support_regions(regions: List[Region], cfg: ScaffoldConfig, up: np.ndarray) -> List[Region]:
    k = min(max(cfg.base_legs, 3), len(regions))
    ranked = sorted(regions, key=lambda r: float(np.min(r.mesh.vertices @ up)))
    pool = ranked[: max(k + 3, min(len(ranked), k * 3))]
    best = None
    best_score = -1.0
    origin = np.mean([r.mesh.centroid for r in regions], axis=0)
    helper = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    basis_x = _unit(np.cross(up, helper))
    basis_y = _unit(np.cross(up, basis_x))
    for combo in combinations(pool, k):
        pts = np.array([region_lowest_point(r, up) for r in combo])
        xy = np.stack([pts @ basis_x, pts @ basis_y], axis=1)
        if k == 3:
            area = 0.5 * abs(np.cross(xy[1] - xy[0], xy[2] - xy[0]))
        else:
            area = float(np.sum(np.linalg.norm(xy - xy.mean(axis=0), axis=1)))
        spread = float(np.min(np.linalg.norm(xy[:, None] - xy[None, :], axis=2) + np.eye(k) * 1e6))
        origin_xy = np.array([origin @ basis_x, origin @ basis_y])
        score = area + 0.25 * spread * spread - 0.15 * float(np.linalg.norm(xy.mean(axis=0) - origin_xy))
        if score > best_score:
            best_score = score
            best = combo
    chosen = list(best) if best is not None else ranked[:k]
    print("Base supports: " + ", ".join(r.name for r in chosen))
    return chosen


def make_base_mesh(regions: List[Region], supports: List[Attachment], cfg: ScaffoldConfig, up: np.ndarray) -> trimesh.Trimesh:
    all_pts = np.vstack([r.mesh.vertices for r in regions])
    dots = all_pts @ up
    min_dot = float(dots.min())
    top_dot = min_dot - cfg.base_gap
    bot_dot = top_dot - cfg.base_thickness
    centroid = all_pts.mean(axis=0)
    helper = np.array([0.0, 1.0, 0.0]) if abs(up[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    basis_x = _unit(np.cross(up, helper))
    basis_y = _unit(np.cross(up, basis_x))
    xy = np.stack([(all_pts - centroid) @ basis_x, (all_pts - centroid) @ basis_y], axis=1)
    minxy = xy.min(axis=0) - cfg.base_margin
    maxxy = xy.max(axis=0) + cfg.base_margin
    box = trimesh.creation.box(extents=[float(maxxy[0] - minxy[0]), float(maxxy[1] - minxy[1]), cfg.base_thickness])
    frame = np.eye(4)
    frame[:3, 0] = basis_x
    frame[:3, 1] = basis_y
    frame[:3, 2] = up
    center_xy = 0.5 * (minxy + maxxy)
    frame[:3, 3] = centroid + basis_x * center_xy[0] + basis_y * center_xy[1] + up * (0.5 * (top_dot + bot_dot))
    box.apply_transform(frame)
    cutters = []
    for att in supports:
        cutters.append(make_cutter(att.parent_point, inward=-up, depth=cfg.base_thickness - 1.0, cfg=cfg, up=up, through=False, mesh=box))
    if cutters and not cfg.preview_only:
        try:
            box = box.difference(cutters, engine=cfg.boolean_engine, check_volume=False)
        except Exception as exc:
            print(f"Warning: base boolean failed ({exc}); exporting uncut base")
    return box


def base_attachments(supports: List[Region], cfg: ScaffoldConfig, up: np.ndarray) -> List[Attachment]:
    min_dot = min(float(np.min(r.mesh.vertices @ up)) for r in supports)
    top_dot = min_dot - cfg.base_gap
    atts = []
    for region in supports:
        child_pt = region_lowest_point(region, up)
        parent_pt = child_pt - up * float((child_pt @ up) - top_dot)
        thick = local_thickness(region.mesh, child_pt, up)
        depth_c = min(cfg.socket_depth, max(thick - cfg.min_wall, cfg.bar_width * 0.7))
        depth_p = min(cfg.socket_depth, cfg.base_thickness - 1.0)
        frame = make_frame(parent_pt, child_pt - parent_pt, up)
        atts.append(
            Attachment(
                parent=BASE_NAME,
                child=region.name,
                parent_point=parent_pt,
                child_point=child_pt,
                parent_depth=max(depth_p, cfg.bar_width * 0.6),
                child_depth=max(depth_c, cfg.bar_width * 0.6),
                frame=frame,
                kind="base",
            )
        )
    return atts


def pick_best_contact(
    parent: Region,
    child: Region,
    contacts: Sequence[PairContact],
    cfg: ScaffoldConfig,
    other_meshes: List[trimesh.Trimesh],
) -> Optional[Tuple[PairContact, float, float]]:
    oriented = []
    for raw in contacts:
        if raw.a == parent.name and raw.b == child.name:
            c = raw
        elif raw.a == child.name and raw.b == parent.name:
            c = PairContact(parent.name, child.name, raw.pb, raw.pa, raw.distance, raw.normal_b, raw.normal_a, raw.face_b, raw.face_a)
        else:
            continue
        scored = score_contact(c, parent, child, cfg, other_meshes)
        if scored is not None:
            oriented.append(scored)
    if not oriented:
        return None
    oriented.sort(key=lambda t: t[0])
    _, contact, dp, dc = oriented[0]
    return contact, dp, dc


def _axis_or_normal(contact: PairContact) -> np.ndarray:
    delta = contact.pb - contact.pa
    if np.linalg.norm(delta) > 1e-8:
        return delta
    return contact.normal_a


def build_scaffold(regions: List[Region], cfg: ScaffoldConfig) -> Tuple[List[Attachment], List[str]]:
    up = _up_vector(cfg.up_axis)
    by_name = {r.name: r for r in regions}
    print("Computing closest surface pairs...")
    contacts = all_pair_contacts(regions, cfg)
    supports = choose_support_regions(regions, cfg, up)
    attachments: List[Attachment] = base_attachments(supports, cfg, up)
    attached = {BASE_NAME} | {s.name for s in supports}
    remaining = [r.name for r in regions if r.name not in attached]
    print("Growing assembly tree...")
    safety = 0
    while remaining:
        safety += 1
        if safety > len(regions) * 8:
            raise RuntimeError("Could not connect all regions into an assemblable tree")
        candidates = []
        for child_name in remaining:
            for parent_name in list(attached):
                if parent_name == BASE_NAME:
                    continue
                picked = pick_best_contact(
                    by_name[parent_name],
                    by_name[child_name],
                    contacts.get(pair_key(parent_name, child_name), []),
                    cfg,
                    [by_name[n].mesh for n in by_name if n not in {parent_name, child_name}],
                )
                if picked is None:
                    continue
                contact, dp, dc = picked
                candidates.append((contact.distance, parent_name, child_name, contact, dp, dc))
        candidates.sort(key=lambda t: t[0])
        world = CollisionWorld(cfg.precise_collision)
        for name in attached:
            if name != BASE_NAME:
                world.add(name, by_name[name].mesh)
        placed = False
        for dist, parent_name, child_name, contact, dp, dc in candidates:
            child_mesh = by_name[child_name].mesh
            away = _axis_or_normal(contact)
            travel = required_travel(child_mesh, [by_name[n].mesh for n in attached if n != BASE_NAME], away)
            if insertion_blocked(child_mesh, away, world, travel, cfg.collision_steps):
                continue
            attachments.append(
                Attachment(
                    parent=parent_name,
                    child=child_name,
                    parent_point=contact.pa,
                    child_point=contact.pb,
                    parent_depth=dp,
                    child_depth=dc,
                    frame=make_frame(contact.pa, away, up),
                    kind="region",
                )
            )
            attached.add(child_name)
            remaining.remove(child_name)
            placed = True
            print(f"  link {parent_name} -> {child_name}  gap={dist:.2f}")
            break
        if not placed:
            if not candidates:
                raise RuntimeError(f"No valid socket contacts remain for {remaining}")
            dist, parent_name, child_name, contact, dp, dc = candidates[0]
            print(f"  WARNING: forcing {parent_name} -> {child_name}; check assembly")
            attachments.append(
                Attachment(
                    parent=parent_name,
                    child=child_name,
                    parent_point=contact.pa,
                    child_point=contact.pb,
                    parent_depth=dp,
                    child_depth=dc,
                    frame=make_frame(contact.pa, _axis_or_normal(contact), up),
                    kind="region",
                )
            )
            attached.add(child_name)
            remaining.remove(child_name)
    order = assembly_order(attachments, [r.name for r in regions])
    validate_full_sequence(regions, attachments, order, cfg)
    return attachments, order


def assembly_order(attachments: Sequence[Attachment], region_names: Sequence[str]) -> List[str]:
    children: Dict[str, List[str]] = {BASE_NAME: []}
    for n in region_names:
        children.setdefault(n, [])
    parent = {}
    for att in attachments:
        children.setdefault(att.parent, []).append(att.child)
        parent[att.child] = att.parent
    order = [BASE_NAME]
    ready = list(children.get(BASE_NAME, []))
    seen = {BASE_NAME}
    while ready:
        ready.sort()
        node = ready.pop(0)
        if node in seen:
            continue
        if parent.get(node) not in seen:
            ready.append(node)
            continue
        order.append(node)
        seen.add(node)
        for ch in children.get(node, []):
            if ch not in seen:
                ready.append(ch)
    missing = [n for n in region_names if n not in seen]
    if missing:
        raise RuntimeError(f"Assembly order incomplete, missing {missing}")
    return order


def validate_full_sequence(regions: List[Region], attachments: Sequence[Attachment], order: Sequence[str], cfg: ScaffoldConfig) -> None:
    by_name = {r.name: r for r in regions}
    att_by_child = {a.child: a for a in attachments}
    world = CollisionWorld(cfg.precise_collision)
    print("Validating slide-on assembly order...")
    for name in order:
        if name == BASE_NAME:
            continue
        att = att_by_child[name]
        mesh = by_name[name].mesh
        away = att.child_point - att.parent_point
        if np.linalg.norm(away) < 1e-8:
            away = att.frame[:3, 2]
        assembled = [by_name[n].mesh for n in world.meshes]
        travel = required_travel(mesh, assembled, away)
        if world.meshes and insertion_blocked(mesh, away, world, travel, cfg.collision_steps):
            print(f"  WARNING: {name} may collide while sliding onto {att.parent}")
        else:
            print(f"  OK  {name} slides onto {att.parent}")
        world.add(name, mesh)


def cut_sockets(regions: List[Region], attachments: Sequence[Attachment], cfg: ScaffoldConfig) -> Dict[str, List[trimesh.Trimesh]]:
    up = _up_vector(cfg.up_axis)
    by_name = {r.name: r for r in regions}
    cutters: Dict[str, List[trimesh.Trimesh]] = {r.name: [] for r in regions}
    for att in attachments:
        z = att.frame[:3, 2]
        if att.child in by_name:
            cutters[att.child].append(
                make_cutter(att.child_point, inward=z, depth=att.child_depth, cfg=cfg, up=up, through=cfg.through_holes, mesh=by_name[att.child].mesh)
            )
        if att.kind == "region" and att.parent in by_name:
            cutters[att.parent].append(
                make_cutter(att.parent_point, inward=-z, depth=att.parent_depth, cfg=cfg, up=up, through=cfg.through_holes, mesh=by_name[att.parent].mesh)
            )
    if cfg.preview_only:
        return cutters
    print("Boolean-subtracting sockets from regions...")
    for region in regions:
        tools = cutters.get(region.name, [])
        if not tools:
            continue
        try:
            region.mesh = region.mesh.difference(tools, engine=cfg.boolean_engine, check_volume=False)
            if isinstance(region.mesh, list):
                region.mesh = trimesh.util.concatenate(region.mesh)
            region.mesh = repair_mesh(region.mesh)
            print(f"  cut {region.name} ({len(tools)} socket(s))")
        except Exception as exc:
            print(f"  WARNING: boolean failed for {region.name}: {exc}")
            print("           original mesh kept; cutters exported for manual boolean")
    return cutters


def lay_flat_for_print(mesh: trimesh.Trimesh, att: Attachment) -> trimesh.Trimesh:
    out = mesh.copy()
    out.apply_transform(np.linalg.inv(att.frame))
    out.apply_transform(rotation_matrix(math.pi / 2, [1, 0, 0]))
    out.apply_translation(-out.bounds[0])
    return out


def colorize(mesh: trimesh.Trimesh, rgba: Sequence[int]) -> trimesh.Trimesh:
    mesh = mesh.copy()
    mesh.visual.face_colors = np.array(rgba, dtype=np.uint8)
    return mesh


def palette(n: int) -> np.ndarray:
    base = np.array(
        [
            [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48],
            [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 190],
            [0, 128, 128], [230, 190, 255], [170, 110, 40], [255, 250, 200], [128, 0, 0],
            [170, 255, 195], [128, 128, 0], [255, 215, 180], [0, 0, 128], [128, 128, 128],
        ],
        dtype=np.uint8,
    )
    return np.tile(base, (int(math.ceil(n / len(base))), 1))[:n]


def export_all(
    regions: List[Region],
    attachments: List[Attachment],
    order: List[str],
    cutters: Dict[str, List[trimesh.Trimesh]],
    cfg: ScaffoldConfig,
    out_dir: Path,
) -> None:
    up = _up_vector(cfg.up_axis)
    out_dir.mkdir(parents=True, exist_ok=True)
    region_dir = out_dir / "regions"
    bar_dir = out_dir / "scaffold"
    cutter_dir = out_dir / "cutters"
    preview_dir = out_dir / "preview"
    for d in (region_dir, bar_dir, cutter_dir, preview_dir):
        d.mkdir(exist_ok=True)

    colors = palette(len(regions))
    scene = trimesh.Scene()
    support_atts = [a for a in attachments if a.kind == "base"]
    base = make_base_mesh(regions, support_atts, cfg, up)
    base.export(bar_dir / "base_stand.stl")
    scene.add_geometry(colorize(base, [90, 90, 90, 255]), node_name="base_stand")

    for i, region in enumerate(regions):
        region.mesh.export(region_dir / f"{region.name}.stl")
        scene.add_geometry(colorize(region.mesh, list(colors[i]) + [255]), node_name=region.name)
        for j, cutter in enumerate(cutters.get(region.name, [])):
            cutter.export(cutter_dir / f"{region.name}_socket_{j}.stl")

    bars_meta = []
    for att in attachments:
        bar = make_bar_mesh(att, cfg, up)
        printable = lay_flat_for_print(bar, att)
        fname = f"bar_{att.parent}_{att.child}.stl"
        bar.export(bar_dir / fname)
        printable.export(bar_dir / f"print_{fname}")
        scene.add_geometry(colorize(bar, [40, 40, 40, 255]), node_name=fname)
        bars_meta.append(
            {
                "file": fname,
                "parent": att.parent,
                "child": att.child,
                "kind": att.kind,
                "gap": float(np.linalg.norm(att.child_point - att.parent_point)),
                "parent_point": att.parent_point.tolist(),
                "child_point": att.child_point.tolist(),
                "parent_socket_depth": att.parent_depth,
                "child_socket_depth": att.child_depth,
                "slide_direction_child_to_seat": (-att.frame[:3, 2]).tolist(),
            }
        )

    scene.export(preview_dir / "assembly_preview.glb")
    steps = [{"part": "base_stand", "action": "Place the stand on the table."}]
    att_by_child = {a.child: a for a in attachments}
    for name in order:
        if name == BASE_NAME:
            continue
        att = att_by_child[name]
        bar_file = f"bar_{att.parent}_{att.child}.stl"
        if att.kind == "base":
            steps.append({"part": bar_file, "action": "Insert this bar into the matching hex/D socket in the base stand."})
            steps.append(
                {
                    "part": f"{name}.stl",
                    "action": f"Slide {name} down onto the bar. Do not rotate; the socket only seats one way.",
                    "direction": (-att.frame[:3, 2]).tolist(),
                }
            )
        else:
            steps.append({"part": bar_file, "action": f"Fully seat this bar in the socket already cut in {att.parent}."})
            steps.append(
                {
                    "part": f"{name}.stl",
                    "action": f"Slide {name} onto the exposed bar until it seats. Install {name} before any later region that would block this path.",
                    "direction": (-att.frame[:3, 2]).tolist(),
                }
            )
    report = {
        "config": asdict(cfg),
        "regions": [r.name for r in regions],
        "assembly_order": [n for n in order],
        "steps": steps,
        "bars": bars_meta,
        "notes": [
            "bar_width is the minimum cross-section width of the printed bar (hex flat-to-flat, or D diameter).",
            "clearance is applied per side around that profile to form the socket.",
            "print_*.stl bars are rotated onto the XY plane for slicing; assembly_preview.glb shows in-place poses.",
            "Print each region in a different color. Print bars and the base in a neutral color.",
        ],
    }
    (out_dir / "assembly_sequence.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_readme(out_dir, order, cfg)
    print(f"Wrote print files to {out_dir}")


def write_readme(out_dir: Path, order: List[str], cfg: ScaffoldConfig) -> None:
    lines = [
        "# Brain display scaffold",
        "",
        "Print every file in `regions/` in a different color. Print `scaffold/base_stand.stl` and each `scaffold/bar_*.stl` separately.",
        "Use the `print_bar_*.stl` copies if you want bars already laid flat for the slicer.",
        "",
        "## Suggested print settings",
        "",
        f"- Bar minimum width: {cfg.bar_width} mm",
        f"- Socket clearance (per side): {cfg.clearance} mm",
        f"- Socket shape: {cfg.socket_shape}",
        "- Material: PLA or PETG",
        "- Layer height: 0.16–0.20 mm on bars and sockets",
        "- Perimeters: 3+",
        "",
        "## Assembly order",
        "",
    ]
    for i, name in enumerate(order, 1):
        lines.append(f"{i}. {name if name != BASE_NAME else 'base stand'}")
    lines += [
        "",
        "Each region slides onto its bar along the bar axis. Hex and D sockets block rotation.",
        "If a fit is tight, sand the bar slightly or rerun with a larger `--clearance`.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def build_config(args: argparse.Namespace) -> ScaffoldConfig:
    depth = args.socket_depth if args.socket_depth is not None else max(8.0, args.bar_width * 1.25)
    return ScaffoldConfig(
        bar_width=args.bar_width,
        clearance=args.clearance,
        socket_shape=args.socket_shape,
        socket_depth=depth,
        axial_clearance=args.axial_clearance,
        d_flat_offset=args.d_flat_offset,
        chamfer=args.chamfer,
        samples=args.samples,
        min_wall=args.min_wall,
        max_normal_angle=args.max_normal_angle,
        base_legs=max(3, args.base_legs),
        base_gap=args.base_gap,
        base_thickness=args.base_thickness,
        base_margin=args.base_margin,
        up_axis=args.up_axis,
        scale=args.scale,
        through_holes=args.through_holes,
        preview_only=args.preview_only,
        precise_collision=args.precise_collision,
        boolean_engine=args.engine,
        seed=args.seed,
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", nargs="?", help="Multi-mesh file (GLB/GLTF/OBJ/3MF) or a folder of STL/PLY meshes")
    p.add_argument("--out", type=Path, default=Path("scaffold_output"), help="Output directory")
    p.add_argument("--bar-width", type=float, default=6.0, help="Minimum bar width in mm (hex flat-to-flat or D diameter)")
    p.add_argument("--clearance", type=float, default=0.3, help="Per-side socket clearance in mm")
    p.add_argument("--socket-shape", choices=["hex", "d"], default="hex")
    p.add_argument("--socket-depth", type=float, default=None, help="Blind socket depth in mm (default: max(8, 1.25*bar-width))")
    p.add_argument("--axial-clearance", type=float, default=0.25)
    p.add_argument("--d-flat-offset", type=float, default=0.45, help="D-profile flat position as a fraction of radius")
    p.add_argument("--chamfer", type=float, default=0.4)
    p.add_argument("--samples", type=int, default=4000, help="Surface samples per region for closest-point search")
    p.add_argument("--min-wall", type=float, default=1.5, help="Keep at least this much wall behind a blind socket")
    p.add_argument("--max-normal-angle", type=float, default=55.0, help="Preferred max angle between bar and surface normal")
    p.add_argument("--base-legs", type=int, default=3)
    p.add_argument("--base-gap", type=float, default=18.0, help="Empty space from lowest vertex to top of stand")
    p.add_argument("--base-thickness", type=float, default=6.0)
    p.add_argument("--base-margin", type=float, default=16.0)
    p.add_argument("--up-axis", default="z", help="World up used for the stand (z, y, -z, ...)")
    p.add_argument("--scale", type=float, default=1.0, help="Scale applied to input meshes before scaffolding")
    p.add_argument("--through-holes", action="store_true", help="Cut sockets all the way through each region")
    p.add_argument("--preview-only", action="store_true", help="Skip boolean cuts; still export bars and preview")
    p.add_argument("--precise-collision", action="store_true", help="Use FCL if python-fcl is installed")
    p.add_argument("--engine", default="manifold", help="trimesh boolean engine (manifold or blender)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--demo", action="store_true", help="Run on built-in sphere regions")
    return p.parse_args(argv)


def apply_scale(regions: List[Region], scale: float) -> None:
    if abs(scale - 1.0) < 1e-12:
        return
    for r in regions:
        r.mesh.apply_scale(scale)
        r.original.apply_scale(scale)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    cfg = build_config(args)
    if args.demo:
        regions = demo_regions(cfg)
    else:
        if not args.input:
            print("Provide an input mesh/folder or pass --demo", file=sys.stderr)
            return 2
        regions = load_regions(Path(args.input))
    apply_scale(regions, cfg.scale)
    attachments, order = build_scaffold(regions, cfg)
    cutters = cut_sockets(regions, attachments, cfg)
    export_all(regions, attachments, order, cutters, cfg, args.out)
    print("Assembly order:", " -> ".join("base" if n == BASE_NAME else n for n in order))
    return 0


if __name__ == "__main__":
    sys.exit(main())
