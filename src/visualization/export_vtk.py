from pathlib import Path
import numpy as np
import pyvista as pv

# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

NPZ_FILE = ROOT / "final_outputs" / "colorado_joint_final.npz"
FAULT_FILE = ROOT / "final_outputs" / "fault_probability.npy"
OUTPUT_FILE = ROOT / "final_outputs" / "structural_model.vti"

# ============================================================
# Load inversion results
# ============================================================

npz = np.load(NPZ_FILE)

rho = npz["rho_cube"].astype(np.float32)
chi = npz["chi_cube"].astype(np.float32)
fault = np.load(FAULT_FILE).astype(np.float32)

assert rho.shape == chi.shape, "rho and chi shapes do not match."
assert rho.shape == fault.shape, "fault volume shape does not match inversion."

nz, ny, nx = rho.shape

# ============================================================
# Geometry
# ============================================================

resolution = float(npz["resolution"])
origin = npz["x0"]

grid = pv.ImageData()

grid.dimensions = (nx + 1, ny + 1, nz + 1)

grid.spacing = (
    resolution,
    resolution,
    resolution,
)

grid.origin = (
    float(origin[0]),
    float(origin[1]),
    float(origin[2]),
)

# ============================================================
# Cell Data
# ============================================================

grid.cell_data["Density"] = rho.ravel(order="C")

grid.cell_data["Susceptibility"] = chi.ravel(order="C")

grid.cell_data["FaultProbability"] = fault.ravel(order="C")

# ============================================================
# Save
# ============================================================

grid.save(OUTPUT_FILE)

print("\n==============================")
print("VTK Export Complete")
print("==============================")

print(f"Grid size        : {nx} x {ny} x {nz}")
print(f"Resolution       : {resolution:.3f} m")
print(f"Origin           : {grid.origin}")

print("\nDensity")
print(f"  Min : {rho.min():.6f}")
print(f"  Max : {rho.max():.6f}")

print("\nSusceptibility")
print(f"  Min : {chi.min():.6f}")
print(f"  Max : {chi.max():.6f}")

print("\nFault Probability")
print(f"  Min : {fault.min():.6f}")
print(f"  Max : {fault.max():.6f}")

print("\nSaved:")
print(OUTPUT_FILE)