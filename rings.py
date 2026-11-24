#!/usr/bin/env python3
"""rings.py — convert an image into a concentric-ring halftone STL
for 3D printing (e.g. printing an image in PLA onto fabric or paper).

How it works
------------
For each tone level L, take the mask of pixels darker than L and draw
rings at a fixed spatial pitch inside it (via the distance transform).
Darker areas belong to more masks, so more ring sets overlap there and
the line density rises — while flat white belongs to no mask and stays
truly white. The result is a line-only halftone: no solid fills, every
line has the same width, which keeps the print flexible and makes the
slicer treat each line as a line.

The raster is then meshed faithfully: each connected component becomes
a polygon (with a +0.5 px correction for the half-pixel shrink of
pixel-centre contours); components that still miss their pixel area
fall back to exact rectangle decomposition.

Outputs (next to the input image, or in --out-dir):
    <name>-rings.png          full-resolution black-on-white raster
    <name>-rings-preview.png  downscaled preview
    <name>-rings.stl          3D-printable mesh

Requires: opencv-python, numpy, shapely, trimesh
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh
from shapely.geometry import LineString, MultiPolygon, Polygon, box

# ---------------------------------------------------------------- rings

def make_rings(gray, levels, pitch, line_w):
    """Return the ring pattern as a 0/1 ink array (1 = line)."""
    sm = cv2.bilateralFilter(gray, 9, 55, 55)
    lo, hi = np.percentile(sm, [1, 99])
    sm = np.clip((sm.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
    ink = np.zeros(gray.shape, dtype=np.uint8)
    for L in levels:
        mask = (sm < L).astype(np.uint8)
        if mask.sum() == 0:
            continue
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        rings = ((dist % pitch) < line_w) & (dist >= line_w)
        ink[rings] = 1
    return ink


# ---------------------------------------------------------------- mesh

MIN_AREA = 6      # connected components smaller than this (px) are dropped
EPS = 0.35        # contour simplification tolerance (px)


def rect_fallback(comp):
    """Exact pixel-square polygons via run-length + vertical merging."""
    rows = {}
    for y in np.unique(np.nonzero(comp)[0]):
        row = comp[y]
        runs = []
        x = 0
        while x < len(row):
            if row[x]:
                x0 = x
                while x < len(row) and row[x]:
                    x += 1
                runs.append((x0, x - 1))
            else:
                x += 1
        rows[y] = runs
    rects = []
    active = {}
    prev_y = None
    for y in sorted(rows):
        cur = set(rows[y])
        if prev_y is None or y != prev_y + 1:
            for (x0, x1), y0 in active.items():
                rects.append((x0, y0, x1, prev_y))
            active = {}
        for key in list(active):
            if key not in cur:
                rects.append((key[0], active.pop(key), key[1], y - 1))
        for key in cur:
            if key not in active:
                active[key] = y
        prev_y = y
    for (x0, x1), y0 in active.items():
        rects.append((x0, y0, x1, prev_y))
    return [box(x0 - 0.5, y0 - 0.5, x1 + 0.5, y1 + 0.5) for x0, y0, x1, y1 in rects]


def component_polys(comp, px_area):
    """Polygons (pixel coordinates) covering one connected component."""
    cnts, hier = cv2.findContours(comp * 255, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hier is None:
        return rect_fallback(comp)
    hier = hier[0]
    polys = []
    for i, c in enumerate(cnts):
        if hier[i][3] != -1:
            continue
        a = cv2.approxPolyDP(c, EPS, True).reshape(-1, 2).astype(np.float64)
        if len(a) < 3:
            continue
        holes = []
        j = hier[i][2]
        while j != -1:
            hc = cnts[j]
            hp = cv2.approxPolyDP(hc, EPS, True).reshape(-1, 2).astype(np.float64)
            if len(hp) >= 3 and cv2.contourArea(hc) >= 3:
                holes.append(hp)
            j = hier[j][0]
        try:
            p = Polygon(a, holes)
            if not p.is_valid:
                p = p.buffer(0)
        except Exception:
            continue
        if not p.is_empty:
            polys.append(p)
    if not polys:
        return rect_fallback(comp)
    if sum(p.area for p in polys) < 0.4 * px_area:
        # degenerate hairline component: rescue as a buffered centreline
        out = []
        for i, c in enumerate(cnts):
            if hier[i][3] != -1:
                continue
            a = cv2.approxPolyDP(c, EPS, True).reshape(-1, 2).astype(np.float64)
            if len(a) >= 2:
                out.append(LineString(a).buffer(1.25, cap_style=2, join_style=2))
        return [p for p in out if not p.is_empty] or rect_fallback(comp)
    fixed = []
    for p in polys:
        q = p.buffer(0.5, join_style=1)   # undo the half-pixel shrink
        fixed.extend(list(q.geoms) if isinstance(q, MultiPolygon) else [q])
    if abs(sum(p.area for p in fixed) - px_area) / px_area > 0.04:
        return rect_fallback(comp)
    return fixed


def mesh_ink(ink, width_mm, thick_mm):
    """Extrude a 0/1 ink array into a trimesh mesh. Returns (mesh, stats)."""
    h, w = ink.shape
    scale = width_mm / w
    n, lab, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    polys_px = []
    for cid in range(1, n):
        px_area = stats[cid, cv2.CC_STAT_AREA]
        if px_area < MIN_AREA:
            continue
        polys_px.extend(component_polys((lab == cid).astype(np.uint8), px_area))
    meshes = []
    for p in polys_px:
        geoms = list(p.geoms) if isinstance(p, MultiPolygon) else [p]
        for gg in geoms:
            if gg.is_empty or gg.area <= 0:
                continue
            ext = np.column_stack([np.array(gg.exterior.coords)[:, 0] * scale,
                                   (h - np.array(gg.exterior.coords)[:, 1]) * scale])
            holes = [np.column_stack([np.array(r.coords)[:, 0] * scale,
                                      (h - np.array(r.coords)[:, 1]) * scale])
                     for r in gg.interiors]
            try:
                m = trimesh.creation.extrude_polygon(Polygon(ext, holes), height=thick_mm)
                if len(m.faces) > 0:
                    meshes.append(m)
            except Exception:
                pass
    mesh = trimesh.util.concatenate(meshes)
    top = mesh.face_normals[:, 2] > 0.99
    fidelity = mesh.area_faces[top].sum() / (ink.sum() * scale * scale) * 100
    return mesh, {"components": n - 1, "parts": len(meshes),
                  "faces": len(mesh.faces), "fidelity": fidelity}


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("input", help="input image (any format OpenCV reads)")
    ap.add_argument("--out-dir", default=None, help="output directory (default: next to input)")
    ap.add_argument("--width-mm", type=float, default=200.0, help="print width in mm (default 200)")
    ap.add_argument("--thickness", type=float, default=0.8, help="mesh thickness in mm (default 0.8)")
    ap.add_argument("--levels", default="70,105,140,170,200,228",
                    help="comma-separated tone thresholds, dark side (default: 70,105,140,170,200,228)")
    ap.add_argument("--pitch", type=float, default=11.0, help="ring pitch in px (default 11)")
    ap.add_argument("--line-width", type=float, default=3.0, help="line width in px (default 3)")
    ap.add_argument("--preview-width", type=int, default=820, help="preview width in px (default 820)")
    args = ap.parse_args()

    src = Path(args.input)
    if not src.exists():
        sys.exit(f"input not found: {src}")
    out_dir = Path(args.out_dir) if args.out_dir else src.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = src.stem + "-rings"

    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None:
        sys.exit(f"could not read image: {src}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    levels = [int(x) for x in args.levels.split(",") if x.strip()]

    ink = make_rings(gray, levels, args.pitch, args.line_width)
    print(f"ink coverage: {(ink > 0).mean() * 100:.1f}%")

    bw = 255 - ink * 255
    cv2.imwrite(str(out_dir / f"{stem}.png"), bw)
    pv_w = args.preview_width
    pv = cv2.resize(bw, (pv_w, int(bw.shape[0] * pv_w / bw.shape[1])),
                    interpolation=cv2.INTER_AREA)
    _, pv = cv2.threshold(pv, 170, 255, cv2.THRESH_BINARY)
    cv2.imwrite(str(out_dir / f"{stem}-preview.png"), pv)

    mesh, st = mesh_ink(ink, args.width_mm, args.thickness)
    stl_path = out_dir / f"{stem}.stl"
    mesh.export(str(stl_path))
    print(f"components: {st['components']}, parts: {st['parts']}, faces: {st['faces']}")
    print(f"mesh fidelity vs raster ink: {st['fidelity']:.1f}%")
    print(f"wrote {stl_path} ({stl_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
