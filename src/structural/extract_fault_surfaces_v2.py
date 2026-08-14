
import numpy as np
import pyvista as pv
from scipy import ndimage
from pathlib import Path

OUTPUT_DIR = Path("final_outputs")

print("="*80)
print("GENERATING SURFACE")
print("="*80)

ridge = np.load(OUTPUT_DIR/"ridge_volume.npy")

print("Volume shape:", ridge.shape)
print("Active voxels:", int(ridge.sum()))

print("\n" + "="*80)
print("COMPONENT ANALYSIS")
print("="*80)

labels, n = ndimage.label(ridge)

sizes = np.bincount(labels.ravel())

for label in range(1, n + 1):

    coords = np.argwhere(labels == label)

    if len(coords) == 0:
        continue

    zmin, ymin, xmin = coords.min(axis=0)
    zmax, ymax, xmax = coords.max(axis=0)

    print(
        f"Label {label:2d} | "
        f"voxels={len(coords):4d} | "
        f"depth={zmin}-{zmax} | "
        f"bbox=({ymax-ymin+1} x {xmax-xmin+1})"
    )


# Convert boolean volume to float
volume = ridge.astype(np.float32)

# Create ImageData
grid = pv.ImageData()

# PyVista dimensions are number of points, so add 1
grid.dimensions = np.array(volume.shape) + 1

grid.spacing = (1.0, 1.0, 1.0)
grid.origin = (0.0, 0.0, 0.0)

# Point data
point_volume = np.pad(
    volume,
    ((0,1),(0,1),(0,1)),
    mode="edge"
)

grid.point_data["ridge"] = point_volume.ravel(order="F")

# Extract isosurface
surface = grid.contour(
    isosurfaces=[0.5],
    scalars="ridge"
)

print("Surface points :", surface.n_points)
print("Surface cells  :", surface.n_cells)



# ---------------------------------------------------------
# Show each connected mesh component separately
# ---------------------------------------------------------

parts = surface.split_bodies()

print("\nMesh components:", len(parts))

plotter = pv.Plotter(off_screen=True)

colors = [
    "red","blue","green","yellow",
    "cyan","magenta","orange",
    "white","pink","purple"
]

for i, part in enumerate(parts):

    bounds = part.bounds

    dx = bounds[1] - bounds[0]
    dy = bounds[3] - bounds[2]
    dz = bounds[5] - bounds[4]

    print(
        f"Mesh {i+1}: "
        f"points={part.n_points:4d} "
        f"cells={part.n_cells:4d} "
        f"area={part.area:8.2f} "
        f"dx={dx:6.2f} "
        f"dy={dy:6.2f} "
        f"dz={dz:6.2f}"
    )

    plotter.add_mesh(
        part,
        color=colors[i % len(colors)],
        show_edges=False
    )

plotter.show(
    screenshot=str(
        OUTPUT_DIR/"mesh_components.png"
    )
)

surface.save(OUTPUT_DIR/"ridge_surface.vtp")

print("Saved:")
print(" ", OUTPUT_DIR/"ridge_surface.vtp")

# Quick preview
plotter = pv.Plotter(off_screen=True)
plotter.add_mesh(surface, color="red")
plotter.show(screenshot=str(OUTPUT_DIR/"ridge_surface_preview.png"))

print("Preview saved:")
print(" ", OUTPUT_DIR/"ridge_surface_preview.png")

print("\nCleaning mesh...")

# Split into connected components
parts = surface.split_bodies()

clean = pv.PolyData()

kept = 0

for part in parts:

    if part.n_cells < 100:
        continue

    # Convert UnstructuredGrid -> PolyData
    part = part.extract_surface().triangulate()

    part = part.smooth(
        n_iter=30,
        relaxation_factor=0.08,
        feature_smoothing=False,
        boundary_smoothing=True,
    )

    clean = clean.merge(part)
    kept += 1

print("Components kept:", kept)
print("Clean cells:", clean.n_cells)

clean.save(OUTPUT_DIR/"ridge_surface_clean.vtp")

plotter = pv.Plotter(off_screen=True)
plotter.add_mesh(clean, color="red")
plotter.show(
    screenshot=str(
        OUTPUT_DIR/"ridge_surface_clean.png"
    )
)

print("Saved clean mesh.")

