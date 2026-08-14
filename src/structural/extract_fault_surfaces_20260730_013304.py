#!/usr/bin/env python3
"""
Candidate Fault-Surface Extraction
----------------------------------

Extracts multiple separated structural surfaces from the
structure-tensor fault-likelihood volume.

Outputs
-------
candidate_fault_surfaces.vtp
candidate_fault_preview.png
candidate_fault_report.txt
"""

from pathlib import Path

import numpy as np
import pyvista as pv
from scipy.ndimage import label

# ============================================================
# Configuration
# ============================================================

THRESHOLD_PERCENTILE = 95.0

MIN_COMPONENT_VOXELS = 25
MIN_COMPONENT_CELLS = 50

MAX_COMPONENTS = 30

# Boundary artifacts were already removed by structural_mask.npy.
REMOVE_BOUNDARY_COMPONENTS = False

VERTICAL_EXAGGERATION = 3.0

# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = ROOT / "final_outputs"

NPZ_FILE = OUTPUT_DIR / "colorado_joint_final.npz"
FAULT_FILE = OUTPUT_DIR / "fault_probability.npy"
MASK_FILE = OUTPUT_DIR / "structural_mask.npy"

OUTPUT_SURFACE = (
    OUTPUT_DIR / "candidate_fault_surfaces.vtp"
)

OUTPUT_PREVIEW = (
    OUTPUT_DIR / "candidate_fault_preview.png"
)

OUTPUT_REPORT = (
    OUTPUT_DIR / "candidate_fault_report.txt"
)

# ============================================================
# Validate files
# ============================================================

print("=" * 70)
print("CANDIDATE FAULT-SURFACE EXTRACTION")
print("=" * 70)

if not NPZ_FILE.exists():
    raise FileNotFoundError(
        f"Missing inversion file:\n{NPZ_FILE}"
    )

if not FAULT_FILE.exists():
    raise FileNotFoundError(
        f"Missing fault-likelihood file:\n{FAULT_FILE}"
    )

# ============================================================
# Load inversion geometry
# ============================================================

with np.load(NPZ_FILE) as data:

    required_keys = {
        "rho_cube",
        "resolution",
        "x0",
    }

    missing_keys = required_keys.difference(
        data.files
    )

    if missing_keys:
        raise KeyError(
            "Missing required NPZ arrays: "
            + ", ".join(sorted(missing_keys))
        )

    model_shape = np.asarray(
        data["rho_cube"]
    ).shape

    resolution = float(
        np.asarray(
            data["resolution"]
        ).squeeze()
    )

    origin = np.asarray(
        data["x0"],
        dtype=np.float64,
    ).reshape(-1)

if origin.size < 3:
    raise ValueError(
        "x0 must contain X, Y and Z coordinates."
    )

if not np.isfinite(resolution) or resolution <= 0:
    raise ValueError(
        f"Invalid model resolution: {resolution}"
    )

origin_x = float(origin[0])
origin_y = float(origin[1])
origin_z = float(origin[2])

# ============================================================
# Load structural volumes
# ============================================================

fault_probability = np.load(
    FAULT_FILE
).astype(np.float32)

if MASK_FILE.exists():

    structural_mask = np.load(
        MASK_FILE
    ).astype(np.float32)

else:

    structural_mask = np.ones_like(
        fault_probability,
        dtype=np.float32,
    )

if fault_probability.shape != model_shape:
    raise ValueError(
        "Fault-likelihood shape does not match inversion model: "
        f"{fault_probability.shape} versus {model_shape}"
    )

if structural_mask.shape != model_shape:
    raise ValueError(
        "Structural-mask shape does not match inversion model."
    )

if not np.all(np.isfinite(fault_probability)):
    raise ValueError(
        "Fault-likelihood volume contains NaN or infinite values."
    )

nz, ny, nx = fault_probability.shape

print(f"\nVolume shape : {fault_probability.shape}")
print(f"Resolution   : {resolution:.3f} m")
print(
    "Origin       : "
    f"({origin_x:.3f}, {origin_y:.3f}, {origin_z:.3f})"
)

# ============================================================
# Determine extraction threshold
# ============================================================

valid_values = fault_probability[
    (fault_probability > 0)
    & (structural_mask > 0)
]

if valid_values.size == 0:
    raise RuntimeError(
        "No positive structural-likelihood values were found."
    )

threshold = float(
    np.percentile(
        valid_values,
        THRESHOLD_PERCENTILE,
    )
)

binary_volume = (
    (fault_probability >= threshold)
    & (structural_mask > 0)
)

selected_voxels = int(
    np.count_nonzero(binary_volume)
)

print("\nThresholding")
print("-" * 50)
print(f"Percentile      : {THRESHOLD_PERCENTILE:.2f}")
print(f"Threshold       : {threshold:.6f}")
print(f"Selected voxels : {selected_voxels:,}")

if selected_voxels == 0:
    raise RuntimeError(
        "Thresholding selected no voxels."
    )

# ============================================================
# Connected-component labeling
# ============================================================

connectivity_kernel = np.ones(
    (3, 3, 3),
    dtype=np.uint8,
)

labeled_volume, raw_component_count = label(
    binary_volume,
    structure=connectivity_kernel,
)

print(f"Raw components  : {raw_component_count}")

component_records = []

for component_id in range(
    1,
    raw_component_count + 1,
):

    voxel_indices = np.argwhere(
        labeled_volume == component_id
    )

    voxel_count = int(
        voxel_indices.shape[0]
    )

    if voxel_count < MIN_COMPONENT_VOXELS:
        continue

    zmin, ymin, xmin = voxel_indices.min(
        axis=0
    )

    zmax, ymax, xmax = voxel_indices.max(
        axis=0
    )

    extent_z = int(
        zmax - zmin + 1
    )

    extent_y = int(
        ymax - ymin + 1
    )

    extent_x = int(
        xmax - xmin + 1
    )

    touches_boundary = any(
        [
            zmin <= 0,
            zmax >= nz - 1,
            ymin <= 0,
            ymax >= ny - 1,
            xmin <= 0,
            xmax >= nx - 1,
        ]
    )

    if (
        REMOVE_BOUNDARY_COMPONENTS
        and touches_boundary
    ):
        continue

    horizontal_long = max(
        extent_x,
        extent_y,
    )

    horizontal_short = max(
        1,
        min(
            extent_x,
            extent_y,
        ),
    )

    elongation = (
        horizontal_long
        / horizontal_short
    )

    spatial_extent = (
        extent_x
        + extent_y
        + extent_z
    )

    score = (
        voxel_count
        * max(
            1.0,
            elongation,
        )
        * max(
            1.0,
            spatial_extent / 5.0,
        )
    )

    component_records.append(
        {
            "component_id": component_id,
            "voxel_count": voxel_count,
            "extent_x": extent_x,
            "extent_y": extent_y,
            "extent_z": extent_z,
            "elongation": elongation,
            "touches_boundary": touches_boundary,
            "score": score,
        }
    )

component_records.sort(
    key=lambda item: item["score"],
    reverse=True,
)

component_records = component_records[
    :MAX_COMPONENTS
]

if not component_records:
    raise RuntimeError(
        "No structural components survived filtering."
    )

print(
    f"Retained components: "
    f"{len(component_records)}"
)

# ============================================================
# Create shared PyVista volume
# ============================================================

grid = pv.ImageData()

grid.dimensions = (
    nx + 1,
    ny + 1,
    nz + 1,
)

grid.spacing = (
    resolution,
    resolution,
    resolution,
)

grid.origin = (
    origin_x,
    origin_y,
    origin_z,
)

grid.cell_data[
    "FaultLikelihood"
] = fault_probability.ravel(
    order="C"
)

# ============================================================
# Extract each connected surface
# ============================================================

accepted_surfaces = []

report_lines = [
    "Candidate Fault-Surface Extraction Report",
    "=" * 60,
    "",
    f"Model shape: {fault_probability.shape}",
    f"Resolution: {resolution:.6f} m",
    (
        "Origin: "
        f"{origin_x:.6f}, "
        f"{origin_y:.6f}, "
        f"{origin_z:.6f}"
    ),
    f"Threshold percentile: {THRESHOLD_PERCENTILE}",
    f"Threshold value: {threshold:.8f}",
    f"Selected voxels: {selected_voxels}",
    f"Raw components: {raw_component_count}",
    "",
    "Accepted components:",
]

fault_number = 0

for record in component_records:

    component_id = record[
        "component_id"
    ]

    component_mask = (
        labeled_volume == component_id
    ).astype(np.float32)

    component_grid = grid.copy()

    component_grid.cell_data[
        "ComponentMask"
    ] = component_mask.ravel(
        order="C"
    )

    point_grid = (
        component_grid
        .cell_data_to_point_data(
            pass_cell_data=False
        )
    )

    surface = point_grid.contour(
        isosurfaces=[0.5],
        scalars="ComponentMask",
        method="flying_edges",
    )

    if surface.n_cells == 0:
        continue

    surface = (
        surface
        .extract_surface()
        .triangulate()
        .clean()
    )

    if surface.n_cells < MIN_COMPONENT_CELLS:
        continue

    surface = surface.sample(
        point_grid
    )

    fault_number += 1

    surface.cell_data[
        "FaultId"
    ] = np.full(
        surface.n_cells,
        fault_number,
        dtype=np.int32,
    )

    surface.point_data[
        "FaultId"
    ] = np.full(
        surface.n_points,
        fault_number,
        dtype=np.int32,
    )

    accepted_surfaces.append(
        surface
    )

    report_lines.append(
        (
            f"Fault {fault_number}: "
            f"voxels={record['voxel_count']}, "
            f"extent_x={record['extent_x']}, "
            f"extent_y={record['extent_y']}, "
            f"extent_z={record['extent_z']}, "
            f"elongation={record['elongation']:.3f}, "
            f"surface_cells={surface.n_cells}, "
            f"surface_points={surface.n_points}"
        )
    )

if not accepted_surfaces:
    raise RuntimeError(
        "No candidate surfaces were generated."
    )

# ============================================================
# Merge surfaces
# ============================================================

combined = accepted_surfaces[0]

for surface in accepted_surfaces[1:]:

    combined = combined.merge(
        surface,
        merge_points=False,
    )

combined = (
    combined
    .extract_surface()
    .triangulate()
    .clean()
)

combined.save(
    OUTPUT_SURFACE
)

# ============================================================
# Report
# ============================================================

report_lines.extend(
    [
        "",
        f"Final surface count: {len(accepted_surfaces)}",
        f"Combined cells: {combined.n_cells}",
        f"Combined points: {combined.n_points}",
        f"Combined area: {combined.area:.6f}",
        "",
        (
            "These surfaces are candidate fault-like structural "
            "features derived from the joint inversion structure tensor."
        ),
    ]
)

OUTPUT_REPORT.write_text(
    "\n".join(report_lines),
    encoding="utf-8",
)

# ============================================================
# Preview with vertical exaggeration
# ============================================================

preview_surface = combined.copy()

preview_surface.scale(
    (
        1.0,
        1.0,
        VERTICAL_EXAGGERATION,
    ),
    inplace=True,
)

xmin = origin_x
xmax = origin_x + nx * resolution

ymin = origin_y
ymax = origin_y + ny * resolution

original_zmin = origin_z
original_zmax = origin_z + nz * resolution

display_zmin = (
    original_zmin
    * VERTICAL_EXAGGERATION
)

display_zmax = (
    original_zmax
    * VERTICAL_EXAGGERATION
)

plotter = pv.Plotter(
    off_screen=True,
    window_size=(
        2400,
        1700,
    ),
)

plotter.set_background(
    "white"
)

plotter.add_mesh(
    preview_surface,
    scalars="FaultId",
    preference="cell",
    cmap="tab20",
    smooth_shading=True,
    show_scalar_bar=False,
)

plotter.add_title(
    (
        "Candidate Fault-Like Structural Surfaces\n"
        "Structure-Tensor Extraction"
    ),
    font_size=17,
)

plotter.show_bounds(
    bounds=(
        xmin,
        xmax,
        ymin,
        ymax,
        display_zmin,
        display_zmax,
    ),
    grid="back",
    location="outer",
    all_edges=True,
    xtitle="Easting X (m)",
    ytitle="Northing Y (m)",
    ztitle=(
        "Z coordinate — "
        f"{VERTICAL_EXAGGERATION:.1f}x vertical exaggeration"
    ),
    fmt="%.0f",
    font_size=10,
)

plotter.add_axes(
    xlabel="X",
    ylabel="Y",
    zlabel="Z",
)

plotter.view_isometric()
plotter.reset_camera()

plotter.screenshot(
    OUTPUT_PREVIEW
)

plotter.close()

# ============================================================
# Completion
# ============================================================

print("\n" + "=" * 70)
print("EXTRACTION COMPLETE")
print("=" * 70)

print(
    f"Candidate surfaces : "
    f"{len(accepted_surfaces)}"
)

print(
    f"Combined cells     : "
    f"{combined.n_cells:,}"
)

print(
    f"Combined points    : "
    f"{combined.n_points:,}"
)

print("\nSaved:")
print(OUTPUT_SURFACE)
print(OUTPUT_PREVIEW)
print(OUTPUT_REPORT)

print("\nFinished.")