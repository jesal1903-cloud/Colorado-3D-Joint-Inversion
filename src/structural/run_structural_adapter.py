#!/usr/bin/env python3

from pathlib import Path

import numpy as np
from scipy.ndimage import (
    gaussian_filter,
    maximum_filter,
)

# ============================================================
# Configuration
# ============================================================

# Smooth the physical-property models before gradients.
MODEL_SMOOTH_SIGMA = 1.0

# Smooth the structure-tensor components.
TENSOR_SMOOTH_SIGMA = 1.5

# Final light smoothing of structural likelihood.
FINAL_SMOOTH_SIGMA = 0.55

# Robust scaling percentiles.
MODEL_LOW_PERCENTILE = 1.0
MODEL_HIGH_PERCENTILE = 99.0
GRADIENT_SCALE_PERCENTILE = 98.0

# Structural likelihood controls.
COHERENCE_POWER = 2.0
GRADIENT_POWER = 0.75

# Remove weak structural responses.
LIKELIHOOD_THRESHOLD = 0.22

# Keep locally significant responses without reducing the
# volume to isolated single voxels.
LOCAL_MAXIMUM_SIZE = 5
LOCAL_MAXIMUM_RATIO = 0.72

# Boundary suppression.
XY_MARGIN_FRACTION = 0.05
Z_MARGIN_FRACTION = 0.10

EPSILON = 1.0e-12

# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = ROOT / "final_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NPZ_FILE = OUTPUT_DIR / "colorado_joint_final.npz"

# ============================================================
# Utility functions
# ============================================================


def robust_normalize_volume(
    volume: np.ndarray,
) -> np.ndarray:
    """
    Robustly normalize a physical-property volume to [0, 1].
    """

    finite_values = volume[np.isfinite(volume)]

    if finite_values.size == 0:
        raise ValueError(
            "Volume contains no finite values."
        )

    low = float(
        np.percentile(
            finite_values,
            MODEL_LOW_PERCENTILE,
        )
    )

    high = float(
        np.percentile(
            finite_values,
            MODEL_HIGH_PERCENTILE,
        )
    )

    scale = high - low

    if not np.isfinite(scale) or scale <= EPSILON:
        scale = float(
            finite_values.max()
            - finite_values.min()
        )

        low = float(
            finite_values.min()
        )

    if scale <= EPSILON:
        raise ValueError(
            "Volume has insufficient variation."
        )

    normalized = (
        volume.astype(np.float32) - low
    ) / scale

    return np.clip(
        normalized,
        0.0,
        1.0,
    ).astype(np.float32)


def calculate_margin(
    dimension: int,
    fraction: float,
) -> int:
    """
    Calculate a safe boundary margin.
    """

    if dimension <= 2:
        return 0

    margin = max(
        1,
        int(round(dimension * fraction)),
    )

    maximum_margin = max(
        1,
        (dimension - 1) // 3,
    )

    return min(
        margin,
        maximum_margin,
    )


def create_structural_mask(
    shape: tuple[int, int, int],
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """
    Create an interior mask that removes artificial gradients
    along all external model boundaries.
    """

    nz, ny, nx = shape

    z_margin = calculate_margin(
        nz,
        Z_MARGIN_FRACTION,
    )

    y_margin = calculate_margin(
        ny,
        XY_MARGIN_FRACTION,
    )

    x_margin = calculate_margin(
        nx,
        XY_MARGIN_FRACTION,
    )

    mask = np.ones(
        shape,
        dtype=np.float32,
    )

    if z_margin > 0:
        mask[:z_margin, :, :] = 0.0
        mask[-z_margin:, :, :] = 0.0

    if y_margin > 0:
        mask[:, :y_margin, :] = 0.0
        mask[:, -y_margin:, :] = 0.0

    if x_margin > 0:
        mask[:, :, :x_margin] = 0.0
        mask[:, :, -x_margin:] = 0.0

    if not np.any(mask > 0):
        raise RuntimeError(
            "Boundary mask removed the entire model."
        )

    return mask, (
        z_margin,
        y_margin,
        x_margin,
    )


def normalize_gradient(
    gradient: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Robustly normalize gradient strength using interior,
    positive gradient values only.
    """

    valid = (
        (mask > 0)
        & np.isfinite(gradient)
        & (gradient > 0)
    )

    values = gradient[valid]

    if values.size == 0:
        raise RuntimeError(
            "No positive interior gradients were found."
        )

    scale = float(
        np.percentile(
            values,
            GRADIENT_SCALE_PERCENTILE,
        )
    )

    if not np.isfinite(scale) or scale <= EPSILON:
        scale = float(
            values.max()
        )

    if scale <= EPSILON:
        raise RuntimeError(
            "Unable to normalize gradient strength."
        )

    normalized = np.clip(
        gradient / scale,
        0.0,
        1.0,
    ).astype(np.float32)

    normalized *= mask

    return normalized, scale


def compute_structure_tensor_likelihood(
    volume: np.ndarray,
    resolution: float,
    mask: np.ndarray,
    label: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Compute a planar structural likelihood from a scalar volume.

    For a sheet-like boundary, the local structure tensor has
    one dominant eigenvalue corresponding to the surface normal.
    The planar coherence is therefore estimated from the
    separation between the largest and second-largest eigenvalues.
    """

    print(
        f"\nProcessing {label} structure tensor..."
    )

    smoothed = gaussian_filter(
        volume,
        sigma=MODEL_SMOOTH_SIGMA,
        mode="nearest",
    ).astype(np.float32)

    gz, gy, gx = np.gradient(
        smoothed,
        resolution,
        resolution,
        resolution,
        edge_order=1,
    )

    gradient_magnitude = np.sqrt(
        gx * gx
        + gy * gy
        + gz * gz
    ).astype(np.float32)

    gradient_magnitude *= mask

    gradient_norm, gradient_scale = (
        normalize_gradient(
            gradient_magnitude,
            mask,
        )
    )

    print(
        f"{label} gradient scale: "
        f"{gradient_scale:.8e}"
    )

    # Structure tensor components.
    jxx = gaussian_filter(
        gx * gx,
        sigma=TENSOR_SMOOTH_SIGMA,
        mode="nearest",
    )

    jyy = gaussian_filter(
        gy * gy,
        sigma=TENSOR_SMOOTH_SIGMA,
        mode="nearest",
    )

    jzz = gaussian_filter(
        gz * gz,
        sigma=TENSOR_SMOOTH_SIGMA,
        mode="nearest",
    )

    jxy = gaussian_filter(
        gx * gy,
        sigma=TENSOR_SMOOTH_SIGMA,
        mode="nearest",
    )

    jxz = gaussian_filter(
        gx * gz,
        sigma=TENSOR_SMOOTH_SIGMA,
        mode="nearest",
    )

    jyz = gaussian_filter(
        gy * gz,
        sigma=TENSOR_SMOOTH_SIGMA,
        mode="nearest",
    )

    shape = volume.shape

    tensor = np.empty(
        shape + (3, 3),
        dtype=np.float32,
    )

    tensor[..., 0, 0] = jxx
    tensor[..., 0, 1] = jxy
    tensor[..., 0, 2] = jxz

    tensor[..., 1, 0] = jxy
    tensor[..., 1, 1] = jyy
    tensor[..., 1, 2] = jyz

    tensor[..., 2, 0] = jxz
    tensor[..., 2, 1] = jyz
    tensor[..., 2, 2] = jzz

    print(
        f"Calculating {label} tensor eigenvalues..."
    )

    eigenvalues = np.linalg.eigvalsh(
        tensor
    ).astype(np.float32)

    # eigvalsh returns ascending eigenvalues:
    # lambda_1 <= lambda_2 <= lambda_3
    lambda_1 = eigenvalues[..., 0]
    lambda_2 = eigenvalues[..., 1]
    lambda_3 = eigenvalues[..., 2]

    # Planar coherence:
    # one dominant eigenvalue and two smaller eigenvalues.
    coherence = (
        lambda_3 - lambda_2
    ) / (
        lambda_3 + EPSILON
    )

    coherence = np.clip(
        coherence,
        0.0,
        1.0,
    ).astype(np.float32)

    coherence *= mask

    # Penalize isotropic/noisy responses.
    anisotropy = (
        lambda_3 - lambda_1
    ) / (
        lambda_3 + EPSILON
    )

    anisotropy = np.clip(
        anisotropy,
        0.0,
        1.0,
    ).astype(np.float32)

    anisotropy *= mask

    likelihood = (
        np.power(
            coherence,
            COHERENCE_POWER,
        )
        * np.power(
            gradient_norm,
            GRADIENT_POWER,
        )
        * anisotropy
    ).astype(np.float32)

    likelihood *= mask

    del tensor
    del eigenvalues
    del jxx
    del jyy
    del jzz
    del jxy
    del jxz
    del jyz

    return (
        gradient_magnitude,
        coherence,
        likelihood,
    )


def print_statistics(
    name: str,
    array: np.ndarray,
) -> None:
    """
    Print compact array statistics.
    """

    finite = array[np.isfinite(array)]

    print(f"\n{name}")
    print("-" * 55)

    if finite.size == 0:
        print("No finite values")
        return

    print(f"Min      : {finite.min():.8f}")
    print(f"Max      : {finite.max():.8f}")
    print(f"Mean     : {finite.mean():.8f}")
    print(f"Std      : {finite.std():.8f}")
    print(
        f"Nonzero  : "
        f"{np.count_nonzero(finite):,}"
    )

    positive = finite[finite > 0]

    if positive.size > 0:
        for percentile in [
            50,
            75,
            90,
            92,
            94,
            95,
            96,
            97,
            98,
            99,
        ]:
            value = np.percentile(
                positive,
                percentile,
            )

            print(
                f"{percentile:>2}%      : "
                f"{value:.8f}"
            )


# ============================================================
# Validate input
# ============================================================

print("=" * 70)
print("STRUCTURE-TENSOR STRUCTURAL ADAPTER")
print("=" * 70)

if not NPZ_FILE.exists():
    raise FileNotFoundError(
        f"Inversion file not found:\n{NPZ_FILE}"
    )

# ============================================================
# Load inversion results
# ============================================================

print("\nLoading inversion results...")

with np.load(NPZ_FILE) as data:

    required_keys = {
        "rho_cube",
        "chi_cube",
    }

    missing = required_keys.difference(
        data.files
    )

    if missing:
        raise KeyError(
            "Missing required NPZ arrays: "
            + ", ".join(sorted(missing))
        )

    rho = np.asarray(
        data["rho_cube"],
        dtype=np.float32,
    )

    chi = np.asarray(
        data["chi_cube"],
        dtype=np.float32,
    )

    if "resolution" in data.files:
        resolution = float(
            np.asarray(
                data["resolution"]
            ).squeeze()
        )
    else:
        resolution = 1.0

        print(
            "Warning: resolution was not found. "
            "Using unit spacing."
        )

if rho.ndim != 3:
    raise ValueError(
        f"rho_cube must be 3D. Received {rho.shape}"
    )

if chi.ndim != 3:
    raise ValueError(
        f"chi_cube must be 3D. Received {chi.shape}"
    )

if rho.shape != chi.shape:
    raise ValueError(
        "Density and susceptibility cube shapes differ: "
        f"{rho.shape} versus {chi.shape}"
    )

if not np.all(np.isfinite(rho)):
    raise ValueError(
        "Density cube contains NaN or infinite values."
    )

if not np.all(np.isfinite(chi)):
    raise ValueError(
        "Susceptibility cube contains NaN or infinite values."
    )

if not np.isfinite(resolution) or resolution <= 0:
    raise ValueError(
        f"Invalid resolution: {resolution}"
    )

print(f"Density cube        : {rho.shape}")
print(f"Susceptibility cube : {chi.shape}")
print(f"Resolution          : {resolution:.3f} m")

# ============================================================
# Normalize models
# ============================================================

print("\nNormalizing inversion models...")

rho_normalized = robust_normalize_volume(
    rho
)

chi_normalized = robust_normalize_volume(
    chi
)

# ============================================================
# Structural mask
# ============================================================

structural_mask, margins = (
    create_structural_mask(
        rho.shape
    )
)

z_margin, y_margin, x_margin = margins

print("\nBoundary suppression")
print("-" * 55)
print(f"Z margin : {z_margin} voxel(s)")
print(f"Y margin : {y_margin} voxel(s)")
print(f"X margin : {x_margin} voxel(s)")
print(
    f"Interior : "
    f"{np.count_nonzero(structural_mask):,} voxel(s)"
)

# ============================================================
# Density structure tensor
# ============================================================

(
    rho_gradient,
    rho_coherence,
    rho_likelihood,
) = compute_structure_tensor_likelihood(
    volume=rho_normalized,
    resolution=resolution,
    mask=structural_mask,
    label="density",
)

# ============================================================
# Susceptibility structure tensor
# ============================================================

(
    chi_gradient,
    chi_coherence,
    chi_likelihood,
) = compute_structure_tensor_likelihood(
    volume=chi_normalized,
    resolution=resolution,
    mask=structural_mask,
    label="susceptibility",
)

# ============================================================
# Joint structural likelihood
# ============================================================

print(
    "\nCombining density and susceptibility "
    "structural likelihoods..."
)

# A feature may be clear in either physical-property model.
joint_maximum = np.maximum(
    rho_likelihood,
    chi_likelihood,
)

# Reward locations supported by both models.
joint_agreement = np.sqrt(
    rho_likelihood
    * chi_likelihood
)

fault_probability = (
    0.75 * joint_maximum
    + 0.25 * joint_agreement
).astype(np.float32)

# Light final smoothing to create extractable surfaces.
fault_probability = gaussian_filter(
    fault_probability,
    sigma=0.35,
    mode="constant",
    cval=0.0,
).astype(np.float32)

fault_probability *= structural_mask

# ============================================================
# Local structural thinning
# ============================================================

print("Applying ridge extraction...")

candidate = fault_probability > LIKELIHOOD_THRESHOLD

ridge = np.zeros_like(candidate, dtype=bool)

# X-direction non-maximum suppression
ridge |= (
    candidate &
    (fault_probability >= np.roll(fault_probability, 1, axis=2)) &
    (fault_probability >= np.roll(fault_probability, -1, axis=2))
)

# Y-direction non-maximum suppression
ridge |= (
    candidate &
    (fault_probability >= np.roll(fault_probability, 1, axis=1)) &
    (fault_probability >= np.roll(fault_probability, -1, axis=1))
)

# Z-direction non-maximum suppression
ridge |= (
    candidate &
    (fault_probability >= np.roll(fault_probability, 1, axis=0)) &
    (fault_probability >= np.roll(fault_probability, -1, axis=0))
)

ridge &= (structural_mask > 0)

fault_probability = np.where(
    ridge,
    fault_probability,
    0.0
).astype(np.float32)

fault_probability /= (
    fault_probability.max() + 1e-8
)

fault_probability = np.clip(
    fault_probability,
    0.0,
    1.0,
).astype(np.float32)


# ============================================================
# Statistics
# ============================================================

print("\n" + "=" * 70)
print("STRUCTURAL ATTRIBUTE STATISTICS")
print("=" * 70)

print_statistics(
    "Density gradient",
    rho_gradient,
)

print_statistics(
    "Density planar coherence",
    rho_coherence,
)

print_statistics(
    "Susceptibility gradient",
    chi_gradient,
)

print_statistics(
    "Susceptibility planar coherence",
    chi_coherence,
)

print_statistics(
    "Joint structural likelihood",
    fault_probability,
)

if np.count_nonzero(
    fault_probability
) == 0:
    raise RuntimeError(
        "The current parameters removed all structural responses. "
        "Reduce LIKELIHOOD_THRESHOLD."
    )

# ============================================================
# Save
# ============================================================

print("\nSaving structural products...")

files_to_save = {
    "density_gradient.npy": rho_gradient,
    "susceptibility_gradient.npy": chi_gradient,
    "density_coherence.npy": rho_coherence,
    "susceptibility_coherence.npy": chi_coherence,
    "density_structural_likelihood.npy": rho_likelihood,
    "susceptibility_structural_likelihood.npy": chi_likelihood,
    "fault_probability.npy": fault_probability,
    "structural_mask.npy": structural_mask,
}

for filename, array in files_to_save.items():

    output_path = OUTPUT_DIR / filename

    np.save(
        output_path,
        array.astype(np.float32),
    )

    print(output_path)

print("\n" + "=" * 70)
print("STRUCTURE-TENSOR PROCESSING COMPLETE")
print("=" * 70)

print(
    "\nThe fault_probability.npy volume now represents "
    "planar structural likelihood rather than raw gradient magnitude."
)

print("\nFinished.")