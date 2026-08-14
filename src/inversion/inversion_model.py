# ─────────────────────────────────────────────────────────────────────────────
#  Standard library
# ─────────────────────────────────────────────────────────────────────────────
import argparse
import logging
import os
import sys

# ─────────────────────────────────────────────────────────────────────────────
#  Third-party
# ─────────────────────────────────────────────────────────────────────────────
import h5py
import json
import re

import numpy as np



def compute_structure_tensor(volume):
    """
    Compute the 3D structure tensor components.
    """

    gx, gy, gz = np.gradient(volume)

    Jxx = gx * gx
    Jyy = gy * gy
    Jzz = gz * gz

    Jxy = gx * gy
    Jxz = gx * gz
    Jyz = gy * gz

    sigma = 1.0

    Jxx = gaussian_filter(Jxx, sigma)
    Jyy = gaussian_filter(Jyy, sigma)
    Jzz = gaussian_filter(Jzz, sigma)

    Jxy = gaussian_filter(Jxy, sigma)
    Jxz = gaussian_filter(Jxz, sigma)
    Jyz = gaussian_filter(Jyz, sigma)

    return Jxx, Jyy, Jzz, Jxy, Jxz, Jyz



def tensor_eigen_analysis(Jxx, Jyy, Jzz, Jxy, Jxz, Jyz):
    """
    Compute eigenvalues of the 3D structure tensor.

    Returns:
        l1, l2, l3 : sorted eigenvalues
    """

    shape = Jxx.shape

    tensor = np.zeros(
        shape + (3, 3),
        dtype=np.float32
    )

    tensor[...,0,0] = Jxx
    tensor[...,1,1] = Jyy
    tensor[...,2,2] = Jzz

    tensor[...,0,1] = Jxy
    tensor[...,1,0] = Jxy

    tensor[...,0,2] = Jxz
    tensor[...,2,0] = Jxz

    tensor[...,1,2] = Jyz
    tensor[...,2,1] = Jyz

    eigenvalues = np.linalg.eigvalsh(tensor)

    l1 = eigenvalues[...,0]
    l2 = eigenvalues[...,1]
    l3 = eigenvalues[...,2]

    return l1, l2, l3



def compute_planarity(l1, l2, l3):
    """
    Compute planar structure likelihood from tensor eigenvalues.

    High values indicate planar geological features.
    """

    eps = 1e-8

    planarity = (
        (l3 - l2)
        /
        (l3 + eps)
    )

    planarity = np.clip(
        planarity,
        0.0,
        1.0,
    )

    return planarity.astype(np.float32)



def compute_fault_likelihood(volume):
    """
    Compute geological fault likelihood using:
    gradient magnitude × planar coherence.
    """

    gx, gy, gz = np.gradient(volume)

    gradient_strength = np.sqrt(
        gx**2 +
        gy**2 +
        gz**2
    )

    Jxx, Jyy, Jzz, Jxy, Jxz, Jyz = (
        compute_structure_tensor(volume)
    )

    l1, l2, l3 = tensor_eigen_analysis(
        Jxx,
        Jyy,
        Jzz,
        Jxy,
        Jxz,
        Jyz,
    )

    planarity = compute_planarity(
        l1,
        l2,
        l3,
    )

    fault_likelihood = (
        gradient_strength *
        planarity
    )

    # Normalize
    fault_likelihood -= fault_likelihood.min()
    fault_likelihood /= (
        fault_likelihood.max()
        - fault_likelihood.min()
        + 1e-8
    )

    return fault_likelihood.astype(np.float32)


def extract_fault_surfaces(fault_confidence):
    """
    Extract ridge-like fault surfaces using local maxima.
    """

    output = np.zeros_like(fault_confidence)

    Z, Y, X = fault_confidence.shape

    for z in range(1, Z - 1):
        for y in range(1, Y - 1):
            for x in range(1, X - 1):

                value = fault_confidence[z, y, x]

                if value == 0:
                    continue

                neighborhood = fault_confidence[
                    z-1:z+2,
                    y-1:y+2,
                    x-1:x+2,
                ]

                # Keep only directional local maxima
                center = neighborhood[1, 1, 1]

                x_max = center >= neighborhood[1, 1, :].max()
                y_max = center >= neighborhood[1, :, 1].max()
                z_max = center >= neighborhood[:, 1, 1].max()

                if x_max or y_max or z_max:
                    output[z, y, x] = value

    # Bridge nearby ridge voxels to improve continuity
    bridged = output.copy()

    Z, Y, X = output.shape

    for z in range(1, Z - 1):
        for y in range(1, Y - 1):
            for x in range(1, X - 1):

                if output[z, y, x] > 0:
                    continue

                neighborhood = output[
                    z-1:z+2,
                    y-1:y+2,
                    x-1:x+2,
                ]

                if np.count_nonzero(neighborhood) >= 4:
                    bridged[z, y, x] = neighborhood.max()

    # Remove very small fault fragments
    filtered = bridged.copy()

    Z, Y, X = bridged.shape

    for z in range(1, Z - 1):
        for y in range(1, Y - 1):
            for x in range(1, X - 1):

                if bridged[z, y, x] == 0:
                    continue

                neighborhood = bridged[
                    z-1:z+2,
                    y-1:y+2,
                    x-1:x+2,
                ]

                # Keep only voxels that belong to a larger connected feature
                if np.count_nonzero(neighborhood) < 5:
                    filtered[z, y, x] = 0.0

    return filtered


def load_structural_model(path):
    """Load structural geology volume."""

    import numpy as np
    from pathlib import Path

    if path is None:
        return None

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    model = np.load(path).astype(np.float32)

    # Compute structure-tensor fault likelihood
    fault_confidence = compute_fault_likelihood(
        model
    )

    print("=" * 60)
    print("STRUCTURE TENSOR FAULT DIAGNOSTICS")
    print("=" * 60)
    print("Shape:", fault_confidence.shape)
    print("Min:", float(fault_confidence.min()))
    print("Max:", float(fault_confidence.max()))
    print("Mean:", float(fault_confidence.mean()))
    print(
        "Non-zero:",
        int(np.count_nonzero(fault_confidence))
    )
    print("=" * 60)

    # Suppress weak gradients (noise)
    threshold = 0.20
    fault_confidence[fault_confidence < threshold] = 0.0

    # Keep only the strongest fault responses
    percentile = 90.0
    cutoff = np.percentile(
        fault_confidence[fault_confidence > 0],
        percentile
    ) if np.any(fault_confidence > 0) else 0.0

    fault_confidence = np.where(
        fault_confidence >= cutoff,
        fault_confidence,
        0.0,
    )

    # Remove isolated responses (3×3×3 neighborhood)
    padded = np.pad(fault_confidence > 0, 1, mode="constant")
    cleaned = np.zeros_like(fault_confidence, dtype=bool)

    for z in range(fault_confidence.shape[0]):
        for y in range(fault_confidence.shape[1]):
            for x in range(fault_confidence.shape[2]):
                neighborhood = padded[
                    z:z+3,
                    y:y+3,
                    x:x+3,
                ]
                if neighborhood.sum() >= 3:
                    cleaned[z, y, x] = True

    fault_confidence *= cleaned.astype(np.float32)

    fault_confidence = extract_fault_surfaces(
        fault_confidence
    )

    return fault_confidence.astype(np.float32)


import xarray as xr
import rioxarray  # activates .rio accessor on xarray objects

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")  # non-interactive backend – safe for headless runs
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from typing import Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
#  Logger
# ─────────────────────────────────────────────────────────────────────────────

def _build_logger(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("inversion")
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)-8s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


log = _build_logger()

# ─────────────────────────────────────────────────────────────────────────────
#  Petrophysical constants
# ─────────────────────────────────────────────────────────────────────────────
RHO_REFERENCE = 2.67   # g/cc  Bouguer reference density
RHO_MATRIX    = 2.65   # g/cc  quartz/sandstone matrix
RHO_FLUID     = 1.05   # g/cc  brine
RW            = 0.05   # ohm·m formation water resistivity

GR_CLEAN  = 20.0       # GAPI clean sand end-member
GR_SHALE  = 150.0      # GAPI shale end-member
CHI_CLEAN = 5e-5       # SI   susceptibility for clean quartz sand
CHI_SHALE = 5e-3       # SI   susceptibility for shale

FT_TO_M = 0.3048       # feet → metres

GARDNER_A = 0.31       # rho [g/cc] = A * vp[m/s]^B
GARDNER_B = 0.25


# =============================================================================
#  SECTION 1 — Physics Forward Models
# =============================================================================

class GravityForwardPhysics(nn.Module):
    """FFT-accelerated point-mass forward model: 3-D density → 2-D Bouguer anomaly.

    Pre-computes Green's-function kernel FFTs at construction time for fast,
    autograd-compatible forward passes during training.

    DEM-aware: a first-order Taylor correction around mean(DEM) ensures each
    observation point uses its own elevation.
    """

    def __init__(
        self,
        D: int,
        H: int,
        W: int,
        resolution: float,
        dem: xr.DataArray,
        device="cpu",
    ):
        super().__init__()
        self.D, self.H, self.W = D, H, W
        self.resolution = resolution

        dem_np = np.asarray(dem.values, dtype=np.float64)
        z_top = float(dem_np.mean())
        self.register_buffer(
            "delta_h",
            torch.tensor(dem_np - z_top, dtype=torch.float32, device=device),
        )

        C = 6.67430e-11 * 1e8   # G × 10⁸ → g/cc · m → mGal
        V = resolution ** 3

        fH = 1 << (2 * H - 2).bit_length()
        fW = 1 << (2 * W - 2).bit_length()
        self.fH, self.fW = fH, fW

        with torch.no_grad():
            iy = torch.arange(-(H - 1), H, device=device, dtype=torch.float64)
            ix = torch.arange(-(W - 1), W, device=device, dtype=torch.float64)
            IY, IX = torch.meshgrid(iy, ix, indexing="ij")
            r_horiz_sq = (IX * resolution) ** 2 + (IY * resolution) ** 2

            K_hat_list  = []
            dK_hat_list = []

            for k in range(D):
                dz   = (k + 0.5) * resolution
                r_sq = r_horiz_sq + dz ** 2 + 1e-25
                r    = torch.sqrt(r_sq)

                kern = C * V * dz / (r_sq * r)
                K = torch.zeros(fH, fW, device=device, dtype=torch.float64)
                K[:2 * H - 1, :2 * W - 1] = kern
                K = torch.roll(K, shifts=(-(H - 1), -(W - 1)), dims=(0, 1))
                K_hat_list.append(torch.fft.rfft2(K))

                dkern = C * V * (r_horiz_sq - 2.0 * dz ** 2) / (r_sq * r_sq * r)
                dK = torch.zeros(fH, fW, device=device, dtype=torch.float64)
                dK[:2 * H - 1, :2 * W - 1] = dkern
                dK = torch.roll(dK, shifts=(-(H - 1), -(W - 1)), dims=(0, 1))
                dK_hat_list.append(torch.fft.rfft2(dK))

            self.register_buffer("K_hat",  torch.stack(K_hat_list))
            self.register_buffer("dK_hat", torch.stack(dK_hat_list))

    def forward(self, density: torch.Tensor) -> torch.Tensor:
        """[D, H, W] density contrast (g/cc) → [H, W] Δg (mGal)."""
        rho_pad = F.pad(density.double(),
                        (0, self.fW - self.W, 0, self.fH - self.H))
        rho_hat = torch.fft.rfft2(rho_pad)

        GZ  = (self.K_hat  * rho_hat).sum(0)
        dGZ = (self.dK_hat * rho_hat).sum(0)

        g  = torch.fft.irfft2(GZ,  s=(self.fH, self.fW))[:self.H, :self.W]
        dg = torch.fft.irfft2(dGZ, s=(self.fH, self.fW))[:self.H, :self.W]

        return (g + self.delta_h.double() * dg).float()


class MagneticForwardPhysics(nn.Module):
    """FFT-accelerated dipole forward model: 3-D susceptibility → 2-D TMI anomaly.

    DEM-aware: a first-order Taylor correction around mean(DEM) accounts for
    topography.
    """

    def __init__(
        self,
        D: int,
        H: int,
        W: int,
        resolution: float,
        inc: float,
        dec: float,
        B0: float = 50_000.0,
        dem: Optional[xr.DataArray] = None,
        device="cpu",
    ):
        super().__init__()
        self.D, self.H, self.W = D, H, W
        self.resolution = resolution
        self.B0 = B0

        dem_np = np.asarray(dem.values, dtype=np.float64)
        z_top  = float(dem_np.mean())
        self.register_buffer(
            "delta_h",
            torch.tensor(dem_np - z_top, dtype=torch.float32, device=device),
        )

        CV = resolution ** 3 / (4.0 * np.pi)
        fH = 1 << (2 * H - 2).bit_length()
        fW = 1 << (2 * W - 2).bit_length()
        self.fH, self.fW = fH, fW

        inc_r = np.radians(float(inc))
        dec_r = np.radians(float(dec))
        Fx = np.cos(inc_r) * np.sin(dec_r)
        Fy = np.cos(inc_r) * np.cos(dec_r)
        Fz = -np.sin(inc_r)

        with torch.no_grad():
            iy = torch.arange(-(H - 1), H, device=device, dtype=torch.float64)
            ix = torch.arange(-(W - 1), W, device=device, dtype=torch.float64)
            IY, IX = torch.meshgrid(iy, ix, indexing="ij")
            dx_grid    = IX * resolution
            dy_grid    = IY * resolution
            r_horiz_sq = dx_grid ** 2 + dy_grid ** 2
            Fr_horiz   = Fx * dx_grid + Fy * dy_grid

            kern_rows = torch.arange(-(H - 1), H, device=device) % fH
            kern_cols = torch.arange(-(W - 1), W, device=device) % fW

            K_hat_list  = []
            dK_hat_list = []

            for k in range(D):
                dz    = (k + 0.5) * resolution
                r_sq  = r_horiz_sq + dz ** 2 + 1e-25
                r     = torch.sqrt(r_sq)
                inv_r3 = 1.0 / (r_sq * r)
                inv_r5 = inv_r3 / r_sq
                Fr    = Fr_horiz + Fz * dz

                kern = CV * (3.0 * Fr * Fr * inv_r5 - inv_r3)
                K = torch.zeros(fH, fW, device=device, dtype=torch.float64)
                K[kern_rows[:, None], kern_cols[None, :]] = kern
                K_hat_list.append(torch.fft.rfft2(K))

                inv_r7 = inv_r5 / r_sq
                dkern  = CV * (
                    6.0 * Fr * Fz * inv_r5
                    - 15.0 * Fr * Fr * dz * inv_r7
                    + 3.0 * dz * inv_r5
                )
                dK = torch.zeros(fH, fW, device=device, dtype=torch.float64)
                dK[kern_rows[:, None], kern_cols[None, :]] = dkern
                dK_hat_list.append(torch.fft.rfft2(dK))

            self.register_buffer("K_hat",  torch.stack(K_hat_list))
            self.register_buffer("dK_hat", torch.stack(dK_hat_list))

    def forward(self, susceptibility: torch.Tensor) -> torch.Tensor:
        """[D, H, W] susceptibility (SI) → [H, W] ΔT (nT)."""
        chi_pad = F.pad(susceptibility.double(),
                        (0, self.fW - self.W, 0, self.fH - self.H))
        chi_hat = torch.fft.rfft2(chi_pad)

        DT  = (self.K_hat  * chi_hat).sum(0)
        dDT = (self.dK_hat * chi_hat).sum(0)

        m  = torch.fft.irfft2(DT,  s=(self.fH, self.fW))[:self.H, :self.W]
        dm = torch.fft.irfft2(dDT, s=(self.fH, self.fW))[:self.H, :self.W]

        return ((m + self.delta_h.double() * dm) * self.B0).float()


# =============================================================================
#  SECTION 2 — U-Net Architecture
# =============================================================================

class DoubleConv2D(nn.Module):
    """Conv2d → BN → ReLU → Conv2d → BN → ReLU."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Down(nn.Module):
    """MaxPool2d → DoubleConv2D."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2, stride=2, padding=0)
        self.conv = DoubleConv2D(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    """ConvTranspose2d up-sample then DoubleConv2D with skip connection."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv2D(in_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        x = F.pad(x, [
            diff_x // 2, diff_x - diff_x // 2,
            diff_y // 2, diff_y - diff_y // 2,
        ])
        return self.conv(torch.cat([skip, x], dim=1))


class UNet2DEncoder(nn.Module):
    """2-channel-input U-Net that outputs 2*D feature maps (D rho layers + D chi layers)."""

    def __init__(self, in_channels: int = 2, base_channels: int = 32, out_channels: int = 64):
        super().__init__()
        self.inc   = DoubleConv2D(in_channels,       base_channels)
        self.down1 = Down(base_channels,             base_channels * 2)
        self.down2 = Down(base_channels * 2,         base_channels * 4)
        self.down3 = Down(base_channels * 4,         base_channels * 8)

        self.up1 = Up(base_channels * 8, base_channels * 4)
        self.up2 = Up(base_channels * 4, base_channels * 2)
        self.up3 = Up(base_channels * 2, base_channels)

        self.outc = DoubleConv2D(base_channels, out_channels)
        self.head = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        nn.init.xavier_uniform_(self.head.weight, gain=0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x  = self.up1(x4, x3)
        x  = self.up2(x,  x2)
        x  = self.up3(x,  x1)
        x  = self.outc(x)
        return self.head(x)


# =============================================================================
#  SECTION 3 — Loss Functions
# =============================================================================

def data_misfit(
    d_pred: torch.Tensor,
    d_obs:  torch.Tensor,
    sigma:  Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Weighted L2 data misfit: (1/N) Σ [(d_pred − d_obs) / σ]²."""
    r = d_pred - d_obs
    if sigma is not None:
        r = r / (sigma + 1e-12)
    return (r ** 2).mean()


def _gradient_3d(vol: torch.Tensor):
    """Finite-difference gradients of a [B, 1, D, H, W] volume."""
    gD = vol[:, :, 1:, :, :]  - vol[:, :, :-1, :, :]
    gH = vol[:, :, :, 1:, :]  - vol[:, :, :, :-1, :]
    gW = vol[:, :, :, :, 1:]  - vol[:, :, :, :, :-1]
    return gD, gH, gW


def smoothness_loss(vol: torch.Tensor, wt_z: float = 1.0) -> torch.Tensor:
    """Tikhonov first-order smoothness regularisation."""
    gD, gH, gW = _gradient_3d(vol)
    return wt_z * (gD ** 2).mean() + (gH ** 2).mean() + (gW ** 2).mean()


def cross_gradient_loss(
    m1: torch.Tensor,
    m2: torch.Tensor,
    structural_model: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Cross-gradient structural constraint ‖∇m1 × ∇m2‖² (Gallardo & Meju 2011)."""
    g1D, g1H, g1W = _gradient_3d(m1)
    g2D, g2H, g2W = _gradient_3d(m2)

    d = min(g1D.shape[2], g1H.shape[2], g1W.shape[2])
    h = min(g1D.shape[3], g1H.shape[3], g1W.shape[3])
    w = min(g1D.shape[4], g1H.shape[4], g1W.shape[4])
    s = (slice(None), slice(None), slice(d), slice(h), slice(w))

    cx = g1H[s] * g2W[s] - g1W[s] * g2H[s]
    cy = g1W[s] * g2D[s] - g1D[s] * g2W[s]
    cz = g1D[s] * g2H[s] - g1H[s] * g2D[s]

    cross_grad = cx ** 2 + cy ** 2 + cz ** 2

    if structural_model is not None:

        weight = structural_model.float()

        weight = (weight - weight.min()) / (
            weight.max() - weight.min() + 1e-8
        )

        weight = weight[s]

        cross_grad = cross_grad * (1.0 + weight)

    return cross_grad.mean()


def well_log_loss(
    model_vol:   torch.Tensor,
    constraints: List[Dict],
    param_key:   str = "rho_contrast",
) -> torch.Tensor:
    """Weighted soft point constraint: model_vol[k, i, j] ≈ target[k]."""
    if not constraints:
        return torch.tensor(0.0, device=model_vol.device)

    residuals = []
    for c in constraints:
        ri, rj = c["i"], c["j"]
        target = c[param_key]
        w = c.get("weight", 1.0)
        for k in range(model_vol.shape[0]):
            if not np.isnan(target[k]):
                t = torch.tensor(float(target[k]),
                                 dtype=model_vol.dtype, device=model_vol.device)
                residuals.append(w * (model_vol[k, ri, rj] - t) ** 2)

    if not residuals:
        return torch.tensor(0.0, device=model_vol.device)
    return torch.stack(residuals).mean()


def structural_geology_loss(
    density,
    susceptibility,
    structural_mask=None,
    fault_mask=None,
    interface_weight=0.15,
    fault_weight=0.10,
):
    """
    Structural geology regularization.

    Encourages sharp changes along geological contacts while
    preserving smooth regions elsewhere.
    """

    import torch

    loss = torch.tensor(
        0.0,
        device=density.device,
        dtype=density.dtype,
    )

    if structural_mask is not None:

        gx = torch.abs(density[:, :, :, 1:] - density[:, :, :, :-1]).mean()

        gy = torch.abs(density[:, :, 1:, :] - density[:, :, :-1, :]).mean()

        gz = torch.abs(density[:, 1:, :, :] - density[:, :-1, :, :]).mean()

        # Weight the loss by the structural geology model
        weight = structural_mask.float()

        # Normalize to [0,1]
        weight = (weight - weight.min()) / (
            weight.max() - weight.min() + 1e-8
        )

        # Geological interfaces
        wx = torch.abs(weight[:, :, :, 1:] - weight[:, :, :, :-1]).mean()
        wy = torch.abs(weight[:, :, 1:, :] - weight[:, :, :-1, :]).mean()
        wz = torch.abs(weight[:, 1:, :, :] - weight[:, :-1, :, :]).mean()

        confidence = wx + wy + wz

        density_grad = gx + gy + gz

        # Density is trusted more in well-defined geological regions
        loss += interface_weight * density_grad * confidence

        # Magnetic susceptibility should also follow geological structure
        sgx = torch.abs(
            susceptibility[:, :, :, 1:] - susceptibility[:, :, :, :-1]
        ).mean()

        sgy = torch.abs(
            susceptibility[:, :, 1:, :] - susceptibility[:, :, :-1, :]
        ).mean()

        sgz = torch.abs(
            susceptibility[:, 1:, :, :] - susceptibility[:, :-1, :, :]
        ).mean()

        susceptibility_grad = sgx + sgy + sgz

        loss += 0.5 * interface_weight * susceptibility_grad * confidence


    if fault_mask is not None:

        fx = torch.abs(
            susceptibility[:, :, :, 1:]
            - susceptibility[:, :, :, :-1]
        ).mean()

        fy = torch.abs(
            susceptibility[:, :, 1:, :]
            - susceptibility[:, :, :-1, :]
        ).mean()

        fz = torch.abs(
            susceptibility[:, 1:, :, :]
            - susceptibility[:, :-1, :, :]
        ).mean()

        loss += fault_weight * (fx + fy + fz)

    return loss





def joint_inversion_loss(
    g_pred:       torch.Tensor,
    m_pred:       torch.Tensor,
    g_obs:        torch.Tensor,
    m_obs:        torch.Tensor,
    rho:          torch.Tensor,
    chi:          torch.Tensor,
    g_sigma:      Optional[torch.Tensor] = None,
    m_sigma:      Optional[torch.Tensor] = None,
    structural_model: Optional[torch.Tensor] = None,
    lambda_grav:  float = 1.0,
    lambda_mag:   float = 1.0,
    beta_smooth:  float = 0.01,
    lambda_xgrad: float = 0.1,
    smooth_wt_z:  float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    L_g = data_misfit(g_pred, g_obs, g_sigma)
    L_m = data_misfit(m_pred, m_obs, m_sigma)
    L_s_rho = smoothness_loss(rho, wt_z=smooth_wt_z)
    L_s_chi = smoothness_loss(chi, wt_z=smooth_wt_z)
    L_xg = cross_gradient_loss(
        rho,
        chi,
        structural_model,
    )

    L = (
        lambda_grav  * L_g
        + lambda_mag   * L_m
        + beta_smooth  * (L_s_rho + L_s_chi)
        + lambda_xgrad * L_xg
    )
    return L, {
        "L_grav":   L_g.item(),
        "L_mag":    L_m.item(),
        "L_smooth": (L_s_rho + L_s_chi).item(),
        "L_xgrad":  L_xg.item(),
        "L_total":  L.item(),
    }


def apply_bounds(
    rho_raw: torch.Tensor,
    chi_raw: torch.Tensor,
    rho_lo: float = -1.0,
    rho_hi: float =  1.0,
    chi_lo: float =  0.0,
    chi_hi: float =  0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Map unconstrained U-Net output → physical range via tanh."""
    rho_mid   = (rho_lo + rho_hi) / 2.0
    rho_range = (rho_hi - rho_lo) / 2.0
    chi_mid   = (chi_lo + chi_hi) / 2.0
    chi_range = (chi_hi - chi_lo) / 2.0
    rho = rho_mid + rho_range * torch.tanh(rho_raw)
    chi = chi_mid + chi_range * torch.tanh(chi_raw)
    return rho, chi


# =============================================================================
#  SECTION 4 — Well-Log Constraint Loading
# =============================================================================

def _bin_to_layers(
    depths:     np.ndarray,
    values:     np.ndarray,
    valid_mask: np.ndarray,
    D:          int,
    resolution: float,
    min_weight: float = 0.1,
) -> np.ndarray:
    """Gaussian-kernel weighted average of well samples into D model layers.

    Remaining NaN gaps are filled by linear interpolation / edge clamping.
    """
    layer_centres = (np.arange(D) + 0.5) * resolution
    sigma  = resolution / 2.0
    cutoff = 2.0 * resolution
    layers = np.full(D, np.nan)

    valid_depths = depths[valid_mask]
    valid_vals   = values[valid_mask]
    if len(valid_depths) == 0:
        return layers

    for k, z_c in enumerate(layer_centres):
        dz = valid_depths - z_c
        w  = np.exp(-0.5 * (dz / sigma) ** 2)
        w[np.abs(dz) > cutoff] = 0.0
        wsum = w.sum()
        if wsum >= min_weight:
            layers[k] = float(np.dot(w, valid_vals) / wsum)

    non_nan_idx = np.where(np.isfinite(layers))[0]
    if len(non_nan_idx) > 0:
        all_idx = np.arange(D, dtype=float)
        layers  = np.interp(all_idx, non_nan_idx.astype(float), layers[non_nan_idx])
    return layers


def _las_depths_to_metres(las, df: "pd.DataFrame") -> np.ndarray:
    """Return LAS index depths in metres using the declared index-curve unit."""
    raw_depths = df.index.values.astype(float)

    unit = ""
    try:
        unit = str(las.curves[0].unit or "").strip().upper()
    except Exception:
        pass

    if unit in {"M", "METRE", "METRES", "METER", "METERS"}:
        return raw_depths

    if unit in {"FT", "F", "FEET", "FOOT"}:
        return raw_depths * FT_TO_M

    raise ValueError(
        f"Unsupported or missing LAS depth unit {unit!r} "
        f"for well {getattr(las.well.get('WELL'), 'value', 'unknown')}"
    )


def _nearest_grid_cell(wx, wy, grid_x, grid_y, resolution):
    """Return (i, j) of nearest grid cell, or (None, None) if outside tolerance."""
    j = int(np.argmin(np.abs(grid_x - wx)))
    i = int(np.argmin(np.abs(grid_y - wy)))
    if np.abs(grid_x[j] - wx) > resolution or np.abs(grid_y[i] - wy) > resolution:
        return None, None
    return i, j


def load_all_well_constraints(
    las_dir:   str,
    json_path: str,
    grid_x:    np.ndarray,
    grid_y:    np.ndarray,
    D:         int,
    resolution: float,
    grid_crs:  str,
) -> Tuple[List[Dict], List[Dict]]:
    """Load LAS logs and derive density / susceptibility constraints.

    Returns
    -------
    density_constraints : list of dicts with 'i','j','rho_contrast',[D],'weight','source'
    chi_constraints     : list of dicts with 'i','j','chi'[D],'weight','source'
    """
    import lasio
    from pyproj import Transformer

    with open(json_path) as fh:
        meta = json.load(fh)
    well_locs = {w["file"]: (w["latitude"], w["longitude"]) for w in meta["wells"]}

    transformer = Transformer.from_crs("EPSG:4326", grid_crs, always_xy=True)

    density_constraints: List[Dict] = []
    chi_constraints:     List[Dict] = []
    skipped = 0

    audit = {
        "las_files_found": 0,
        "missing_location": 0,
        "outside_grid": 0,
        "density_rhob": 0,
        "density_rhob_avg": 0,
        "density_porosity": 0,
        "density_nphi": 0,
        "density_resd": 0,
        "density_unusable": 0,
        "chi_magsus": 0,
        "chi_vsh": 0,
        "chi_gr": 0,
        "chi_unusable": 0,
    }

    las_files = sorted(f for f in os.listdir(las_dir) if f.endswith(".las"))
    audit["las_files_found"] = len(las_files)
    log.info("Processing %d LAS files from %s", len(las_files), las_dir)

    for idx, fname in enumerate(las_files):
        log.debug("  [%d/%d] %s", idx + 1, len(las_files), fname)

        if fname not in well_locs:
            log.warning("  No location info for %s — skipping.", fname)
            audit["missing_location"] += 1
            skipped += 1
            continue

        lat, lon = well_locs[fname]
        wx, wy   = transformer.transform(lon, lat)
        ci, cj   = _nearest_grid_cell(wx, wy, grid_x, grid_y, resolution)
        if ci is None:
            audit["outside_grid"] += 1
            skipped += 1
            continue

        las    = lasio.read(os.path.join(las_dir, fname))
        df     = las.df()
        depths = _las_depths_to_metres(las, df)

        # ── Density (priority: RHOB > RHOB_AVG > PHID/PHIT/PHIE > NPHI > RESD) ──
        added_density = False

        if "RHOB" in df.columns:
            raw   = df["RHOB"].values
            valid = np.isfinite(raw) & (raw > 1.5) & (raw < 3.5)
            layers = _bin_to_layers(depths, raw - RHO_REFERENCE, valid, D, resolution)
            if not np.all(np.isnan(layers)):
                density_constraints.append({"i": ci, "j": cj, "rho_contrast": layers,
                                            "name": fname, "weight": 0.9, "source": "RHOB"})
                audit["density_rhob"] += 1
                added_density = True

        if not added_density and "RHOB_AVG" in df.columns:
            raw   = df["RHOB_AVG"].values
            valid = np.isfinite(raw) & (raw > 1.5) & (raw < 3.5)
            layers = _bin_to_layers(depths, raw - RHO_REFERENCE, valid, D, resolution)
            if not np.all(np.isnan(layers)):
                density_constraints.append({"i": ci, "j": cj, "rho_contrast": layers,
                                            "name": fname, "weight": 0.7, "source": "RHOB_AVG"})
                audit["density_rhob_avg"] += 1
                added_density = True

        if not added_density:
            for phi_col in ("PHID", "PHIT", "PHIE"):
                if phi_col in df.columns:
                    raw   = df[phi_col].values
                    valid = np.isfinite(raw) & (raw > -0.15) & (raw < 0.60)
                    phi   = np.clip(raw, 0.01, 0.50)
                    rhob  = RHO_MATRIX * (1 - phi) + RHO_FLUID * phi
                    layers = _bin_to_layers(depths, rhob - RHO_REFERENCE, valid, D, resolution)
                    if not np.all(np.isnan(layers)):
                        density_constraints.append({"i": ci, "j": cj, "rho_contrast": layers,
                                                    "name": fname, "weight": 0.5, "source": phi_col})
                        audit["density_porosity"] += 1
                        added_density = True
                        break

        if not added_density and "NPHI" in df.columns:
            raw   = df["NPHI"].values
            valid = np.isfinite(raw) & (raw > -0.15) & (raw < 0.60)
            phi   = np.clip(raw, 0.01, 0.50)
            rhob  = RHO_MATRIX * (1 - phi) + RHO_FLUID * phi
            layers = _bin_to_layers(depths, rhob - RHO_REFERENCE, valid, D, resolution)
            if not np.all(np.isnan(layers)):
                density_constraints.append({"i": ci, "j": cj, "rho_contrast": layers,
                                            "name": fname, "weight": 0.4, "source": "NPHI"})
                audit["density_nphi"] += 1
                added_density = True

        if not added_density and "RESD" in df.columns:
            raw   = df["RESD"].values
            valid = np.isfinite(raw) & (raw > 0)
            phi   = np.clip(np.sqrt(RW / np.maximum(raw, 1e-4)), 0.01, 0.50)
            rhob  = RHO_MATRIX * (1 - phi) + RHO_FLUID * phi
            layers = _bin_to_layers(depths, rhob - RHO_REFERENCE, valid, D, resolution)
            if not np.all(np.isnan(layers)):
                density_constraints.append({"i": ci, "j": cj, "rho_contrast": layers,
                                            "name": fname, "weight": 0.2, "source": "RESD"})
                audit["density_resd"] += 1

        if not added_density:
            audit["density_unusable"] += 1

        # ── Susceptibility (priority: MAGSUS > VSH > GR) ──────────────────
        added_chi = False

        if "MAGSUS" in df.columns:
            raw   = df["MAGSUS"].values
            valid = np.isfinite(raw) & (raw >= 0)
            layers = _bin_to_layers(depths, raw, valid, D, resolution)
            if not np.all(np.isnan(layers)):
                chi_constraints.append({"i": ci, "j": cj, "chi": layers,
                                        "name": fname, "weight": 1.0, "source": "MAGSUS"})
                audit["chi_magsus"] += 1
                added_chi = True

        if not added_chi and "VSH" in df.columns:
            raw   = df["VSH"].values
            valid = np.isfinite(raw) & (raw >= 0) & (raw <= 1.0)
            vsh   = np.clip(raw, 0.0, 1.0)
            chi   = CHI_CLEAN * (1 - vsh) + CHI_SHALE * vsh
            layers = _bin_to_layers(depths, chi, valid, D, resolution)
            if not np.all(np.isnan(layers)):
                chi_constraints.append({"i": ci, "j": cj, "chi": layers,
                                        "name": fname, "weight": 0.15, "source": "VSH"})
                audit["chi_vsh"] += 1
                added_chi = True

        if not added_chi and "GR" in df.columns:
            raw   = df["GR"].values
            valid = np.isfinite(raw) & (raw >= 0)
            vsh   = np.clip((raw - GR_CLEAN) / (GR_SHALE - GR_CLEAN), 0.0, 1.0)
            chi   = CHI_CLEAN * (1 - vsh) + CHI_SHALE * vsh
            layers = _bin_to_layers(depths, chi, valid, D, resolution)
            if not np.all(np.isnan(layers)):
                chi_constraints.append({"i": ci, "j": cj, "chi": layers,
                                        "name": fname, "weight": 0.1, "source": "GR"})
                audit["chi_gr"] += 1
                added_chi = True

        if not added_chi:
            audit["chi_unusable"] += 1

    log.info("Well-log audit summary:")
    for key, value in audit.items():
        log.info("  %-22s : %s", key, value)

    log.info("Well-log loading complete: %d density, %d chi constraints (%d skipped)",
             len(density_constraints), len(chi_constraints), skipped)

    if len(density_constraints) == 0:
        log.warning(
            "No usable density constraints were loaded. "
            "L_well_rho will be zero."
        )

    if len(chi_constraints) == 0:
        log.warning(
            "No usable susceptibility constraints were loaded. "
            "L_well_chi will be zero."
        )

    return density_constraints, chi_constraints


# =============================================================================
#  SECTION 5 — Seismic Constraint Loading
# =============================================================================

def load_2d_fwi_prior(
    h5_path:    str,
    grid_x:     np.ndarray,
    grid_y:     np.ndarray,
    D:          int,
    resolution: float,
    grid_crs:   str,
) -> np.ndarray:
    """Build [D, H, W] density contrast prior from all 2-D FWI lines via Gardner."""
    from pyproj import Transformer

    transformer = Transformer.from_crs("EPSG:4326", grid_crs, always_xy=True)
    H, W = len(grid_y), len(grid_x)
    accum = np.zeros((D, H, W), dtype=np.float64)
    count = np.zeros((D, H, W), dtype=np.float64)

    log.info("Loading 2D FWI prior from %s", h5_path)
    with h5py.File(h5_path, "r") as f:
        for line_name in f["survey_lines"].keys():
            grp = f[f"survey_lines/{line_name}"]
            lat = grp["coordinates/latitude"][()]
            lon = grp["coordinates/longitude"][()]
            z_m = grp["coordinates/z_m"][()]
            vp  = grp["velocity_model/vp_true"][()].astype(np.float64)

            rho = GARDNER_A * (vp ** GARDNER_B) - RHO_REFERENCE

            for px in range(len(lat)):
                wx, wy = transformer.transform(lon[px], lat[px])
                cj = int(np.argmin(np.abs(grid_x - wx)))
                ci = int(np.argmin(np.abs(grid_y - wy)))
                if (np.abs(grid_x[cj] - wx) > resolution or
                        np.abs(grid_y[ci] - wy) > resolution):
                    continue
                lyr = _bin_to_layers(z_m, rho[px], np.isfinite(rho[px]), D, resolution)
                valid_k = ~np.isnan(lyr)
                accum[valid_k, ci, cj] += lyr[valid_k]
                count[valid_k, ci, cj] += 1.0

    prior   = np.where(count > 0, accum / count, np.nan).astype(np.float32)
    covered = int((count > 0).any(axis=0).sum())
    log.info("2D FWI prior: %d/%d surface cells covered  rho_contrast [%.3f, %.3f] g/cc",
             covered, H * W, np.nanmin(prior), np.nanmax(prior))
    return prior


def load_3d_reflection_prior(
    h5_path:    str,
    grid_x:     np.ndarray,
    grid_y:     np.ndarray,
    D:          int,
    resolution: float,
    grid_crs:   str,
) -> np.ndarray:
    """Resample 3-D reflection density volume to [D, H, W]."""
    from scipy.interpolate import RegularGridInterpolator
    from pyproj import Transformer

    transformer_inv = Transformer.from_crs(grid_crs, "EPSG:4326", always_xy=True)
    H, W = len(grid_y), len(grid_x)

    log.info("Loading 3D reflection prior from %s", h5_path)
    with h5py.File(h5_path, "r") as f:
        density_3d = f["density_model/density_3d"][()] / 1000.0  # kg/m³ → g/cc
        lon_axis   = f["coordinates/longitude"][()]
        lat_axis   = f["coordinates/latitude"][()]
        depth_axis = f["coordinates/depth_m"][()]

    if lon_axis[0] > lon_axis[-1]:
        lon_axis   = lon_axis[::-1].copy()
        density_3d = density_3d[::-1]
    if lat_axis[0] > lat_axis[-1]:
        lat_axis   = lat_axis[::-1].copy()
        density_3d = density_3d[:, ::-1]

    interp = RegularGridInterpolator(
        (lon_axis, lat_axis, depth_axis),
        density_3d,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )

    XX, YY  = np.meshgrid(grid_x, grid_y)
    lon_q, lat_q = transformer_inv.transform(XX.ravel(), YY.ravel())
    lon_q = lon_q.reshape(H, W)
    lat_q = lat_q.reshape(H, W)

    layer_z = (np.arange(D) + 0.5) * resolution

    prior = np.zeros((D, H, W), dtype=np.float32)
    for k in range(D):
        pts = np.column_stack([
            lon_q.ravel(),
            lat_q.ravel(),
            np.full(H * W, layer_z[k]),
        ])
        rho_bulk = interp(pts).reshape(H, W).astype(np.float32)
        prior[k] = rho_bulk - RHO_REFERENCE

    nan_frac = float(np.isnan(prior).mean()) * 100
    log.info("3D reflection prior: shape=%s  rho_contrast [%.3f, %.3f] g/cc  NaN: %.1f%%",
             prior.shape, np.nanmin(prior), np.nanmax(prior), nan_frac)
    return prior


# =============================================================================
#  SECTION 6 — Output Helpers (plots + NPZ)
# =============================================================================

def save_loss_curves(history: List[Dict], out_path: str) -> None:
    """Save training loss curves to a PNG file."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    axes[0].semilogy([h["L_grav"]  for h in history], label="L_grav")
    axes[0].semilogy([h["L_mag"]   for h in history], label="L_mag")
    axes[0].semilogy([h["L_total"] for h in history], label="L_total", ls="--", c="k")
    axes[0].set(xlabel="Iteration", ylabel="Loss", title="Data misfit")
    axes[0].legend()

    axes[1].semilogy([h["L_smooth"] for h in history], label="L_smooth")
    axes[1].semilogy([h["L_xgrad"]  for h in history], label="L_xgrad")
    axes[1].set(xlabel="Iteration", ylabel="Loss", title="Regularisation")
    axes[1].legend()

    l_seis2d = [h.get("L_seis2d", 0.0) for h in history]
    l_seis3d = [h.get("L_seis3d", 0.0) for h in history]
    axes[2].semilogy(l_seis2d, label="L_seis2d")
    axes[2].semilogy(l_seis3d, label="L_seis3d")
    axes[2].set(xlabel="Iteration", ylabel="Loss", title="Seismic constraints")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Loss curves saved → %s", out_path)


def save_model_slices(
    rho_np: np.ndarray,
    chi_np: np.ndarray,
    res:    float,
    out_path: str,
    n_slices: int = 5,
) -> None:
    """Save recovered model depth-slice plots to a PNG file."""
    D = rho_np.shape[0]
    show_layers = np.linspace(0, D - 1, min(D, n_slices), dtype=int)

    fig, axes = plt.subplots(2, len(show_layers), figsize=(4 * len(show_layers), 7))
    if len(show_layers) == 1:
        axes = axes[:, np.newaxis]

    for col_idx, k in enumerate(show_layers):
        depth_m = (k + 0.5) * res

        ax = axes[0, col_idx]
        im = ax.imshow(rho_np[k], cmap="RdBu_r", aspect="auto")
        ax.set_title(f"rho @ {depth_m:.0f}m")
        plt.colorbar(im, ax=ax, shrink=0.7)

        ax = axes[1, col_idx]
        im = ax.imshow(chi_np[k], cmap="magma", aspect="auto")
        ax.set_title(f"chi @ {depth_m:.0f}m")
        plt.colorbar(im, ax=ax, shrink=0.7)

    plt.suptitle("Recovered model — depth slices (U-Net)", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Model slices saved → %s", out_path)


def save_npz(
    out_path:    str,
    rho_np:      np.ndarray,
    chi_np:      np.ndarray,
    x_coords:    np.ndarray,
    y_coords:    np.ndarray,
    z_layer_centres: np.ndarray,
    g_final:     np.ndarray,
    m_final:     np.ndarray,
    g_obs_vals:  np.ndarray,
    m_obs_vals:  np.ndarray,
    g_resid:     np.ndarray,
    m_resid:     np.ndarray,
    g_resid_centered: np.ndarray,
    m_resid_centered: np.ndarray,
    res:         float,
    num_iterations: int,
    rho_lo: float,
    rho_hi: float,
    chi_lo: float,
    chi_hi: float,
    dx: float,
    dy: float,
    z_min: float,
) -> None:
    """Save the joint inversion results in a structured NPZ file."""
    D, H, W = rho_np.shape

    ZZ, YY, XX = np.meshgrid(z_layer_centres, y_coords, x_coords, indexing="ij")
    cell_centers_export = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=-1)

    x0_export     = np.array([x_coords[0] - dx / 2, y_coords[0] - dy / 2, z_min])
    h0_export     = np.full(W, res)
    h1_export     = np.full(H, res)
    h2_export     = np.full(D, res)
    shape_cells_export = np.array([W, H, D])

    np.savez(
        out_path,
        # Recovered models
        rho_model=rho_np.ravel(),
        rho_cube=rho_np,
        chi_model=chi_np.ravel(),
        chi_cube=chi_np,
        # Geometry
        cell_centers=cell_centers_export,
        shape_cells=shape_cells_export,
        x0=x0_export,
        h0=h0_export,
        h1=h1_export,
        h2=h2_export,
        # Coordinate arrays
        x_coords=x_coords,
        y_coords=y_coords,
        z_layer_centres=z_layer_centres,
        # Forward model & residuals
        g_pred=g_final,
        m_pred=m_final,

        g_obs=g_obs_vals,
        m_obs=m_obs_vals,

# Physical residuals
        g_resid=g_resid,
        m_resid=m_resid,

# Optimization residuals
        g_resid_centered=g_resid_centered,
        m_resid_centered=m_resid_centered,
        # Inversion parameters
        resolution=np.float64(res),
        num_iterations=np.int64(num_iterations),
        rho_bounds=np.array([rho_lo, rho_hi]),
        chi_bounds=np.array([chi_lo, chi_hi]),
    )
    log.info("NPZ saved → %s", out_path)
    log.info("  rho_cube     : %s  [%.4f, %.4f] g/cc", rho_np.shape, rho_np.min(), rho_np.max())
    log.info("  chi_cube     : %s  [%.6f, %.6f] SI",  chi_np.shape, chi_np.min(), chi_np.max())
    log.info("  cell_centers : %s", cell_centers_export.shape)
    log.info("  shape_cells  : %s  (nx=%d, ny=%d, nz=%d)", shape_cells_export, W, H, D)


# =============================================================================
#  SECTION 7 — Argument Parser
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Joint gravity-magnetic physics-informed inversion (U-Net).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Required data inputs ──────────────────────────────────────────────
    data = p.add_argument_group("Data inputs (required)")
    data.add_argument("--grav",    required=True, help="Path to Bouguer anomaly NetCDF (.nc)")
    data.add_argument("--rtp_mag", required=True, help="Path to RTP magnetic NetCDF (.nc)")
    data.add_argument("--mag",     required=True, help="Path to raw magnetic NetCDF (.nc)")
    data.add_argument("--dem",     required=True, help="Path to DEM NetCDF (.nc)")
    data.add_argument("--structural_model", default=None,
                      help="Path to structural geology model (.npy)")

    # ── Model geometry ────────────────────────────────────────────────────
    geom = p.add_argument_group("3D model geometry")
    geom.add_argument("--coarsen_factor", type=int,   default=1,    help="Coarsen factor applied to all grids")
    geom.add_argument("--depth",          type=float, default=3000, help="Total model depth (metres)")

    # ── Training ──────────────────────────────────────────────────────────
    train = p.add_argument_group("Training")
    train.add_argument("--num_iterations", type=int,   default=5000, help="Number of training iterations")
    train.add_argument("--lr",             type=float, default=1e-3, help="Peak learning rate")

    # ── Optional well-log constraints ─────────────────────────────────────
    well = p.add_argument_group("Well-log constraints (optional)")
    well.add_argument("--las_dir",   default=None, help="Directory containing .las files")
    well.add_argument("--json_path", default=None, help="JSON file with well locations")

    # ── Optional seismic constraints ──────────────────────────────────────
    seis = p.add_argument_group("Seismic constraints (optional)")
    seis.add_argument("--seismic_2d", default=None,
                      help="Path to 2D FWI HDF5 file (survey_lines/…/vp_true)")
    seis.add_argument("--seismic_3d", default=None,
                      help="Path to 3D reflection HDF5 file (density_model/density_3d)")

    # ── Output ────────────────────────────────────────────────────────────
    out = p.add_argument_group("Output")
    out.add_argument("--output_dir",  default="Output/Inversion/Joint",
                     help="Directory to write outputs")
    out.add_argument("--output_name", default="joint_inversion_results",
                     help="Base filename (without extension) for outputs")

    return p


# =============================================================================
#  SECTION 8 — Main Pipeline
# =============================================================================

def main(args: argparse.Namespace) -> None:

    # ── Device ───────────────────────────────────────────────────────────
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ── Step 1: Load raw data ─────────────────────────────────────────────
    log.info("=" * 60)
    log.info("STEP 1: Loading input data")
    log.info("=" * 60)

    grav    = xr.open_dataarray(args.grav,    decode_coords="all").squeeze()
    rtp_mag = xr.open_dataarray(args.rtp_mag, mask_and_scale=True, decode_coords="all").squeeze()
    mag     = xr.open_dataarray(args.mag,     mask_and_scale=True, decode_coords="all").squeeze()
    dem     = xr.open_dataarray(args.dem,     decode_coords="all").squeeze()

    # Optional structural geology model
    structural_model = load_structural_model(args.structural_model)


    log.info("Gravity    : %s", grav.shape)
    log.info("RTP Mag    : %s", rtp_mag.shape)
    log.info("Magnetic   : %s", mag.shape)
    log.info("DEM        : %s", dem.shape)

    if structural_model is not None:
        structural_model = torch.tensor(
            structural_model,
            dtype=torch.float32,
            device=device
        )
        log.info("Structural model: %s", tuple(structural_model.shape))


    # ── Step 2: Coarsen grids ─────────────────────────────────────────────
    log.info("=" * 60)
    log.info("STEP 2: Coarsening grids (factor=%d)", args.coarsen_factor)
    log.info("=" * 60)

    cf = args.coarsen_factor
    grav_c = grav.coarsen(y=cf, x=cf, boundary="trim").mean()
    mag_c  = mag.coarsen( y=cf, x=cf, boundary="trim").mean()
    dem_c  = dem.coarsen( y=cf, x=cf, boundary="trim").mean()

    if grav_c.shape != mag_c.shape or grav_c.shape != dem_c.shape:
        raise ValueError(
            f"Coarsened grids must have the same shape. "
            f"Got grav {grav_c.shape}, mag {mag_c.shape}, dem {dem_c.shape}."
        )

    log.info("Coarsened grid: %s  |  res ~%.0f m",
             grav_c.shape, grav_c.x.diff(dim="x").mean().item())

    # ── Step 3: Define 3D model geometry ──────────────────────────────────
    log.info("=" * 60)
    log.info("STEP 3: Defining 3D model geometry")
    log.info("=" * 60)

    H, W = grav_c.shape
    res  = float(np.abs(np.diff(grav_c.x.values)).mean())
    D    = int(args.depth / res)

    log.info("Resolution : %.1f m", res)
    log.info("Depth      : %.1f km  (D=%d layers)", D * res / 1000, D)
    log.info("Grid cells : D=%d  H=%d  W=%d  total=%d", D, H, W, D * H * W)

    # Coordinate arrays (used for NPZ output)
    x_coords = grav_c.x.values.astype(float)
    y_coords = grav_c.y.values.astype(float)
    z_top_val = float(dem_c.values.mean())
    z_min     = z_top_val - D * res
    z_layer_centres = z_min + res / 2 + np.arange(D) * res

    dx = float(np.diff(x_coords).mean())
    dy = float(np.diff(y_coords).mean())

    # ── Step 4: Normalise observations ────────────────────────────────────
    log.info("=" * 60)
    log.info("STEP 4: Normalising observations")
    log.info("=" * 60)

    g_obs = torch.tensor(grav_c.values, dtype=torch.float32, device=device)
    m_obs = torch.tensor(mag_c.values,  dtype=torch.float32, device=device)

    g_obs_mean, g_sigma = g_obs.mean(), g_obs.std()
    m_obs_mean, m_sigma = m_obs.mean(), m_obs.std()

    g_norm = (g_obs - g_obs_mean) / (g_sigma + 1e-8)
    m_norm = (m_obs - m_obs_mean) / (m_sigma + 1e-8)
    x_input = torch.stack([g_norm, m_norm], dim=0)[None]   # [1, 2, H, W]

    log.info("Gravity  : mean=%.3f  std=%.3f  mGal", g_obs_mean.item(), g_sigma.item())
    log.info("Magnetic : mean=%.3f  std=%.3f  nT",   m_obs_mean.item(), m_sigma.item())

    # ── Step 5: Build forward physics kernels ─────────────────────────────
    log.info("=" * 60)
    log.info("STEP 5: Pre-computing physics kernels (DEM-aware)")
    log.info("=" * 60)

    inc = rtp_mag.attrs["inclination"]
    dec = rtp_mag.attrs["declination"]
    B0  = rtp_mag.attrs["amplitude_nT"]

    log.info("Geomagnetic: inc=%.2f°  dec=%.2f°  B0=%.0f nT", inc, dec, B0)

    grav_fwd = GravityForwardPhysics(D, H, W, res, dem=dem_c, device=device)
    mag_fwd  = MagneticForwardPhysics(D, H, W, res,
                                      inc=inc, dec=dec, B0=B0,
                                      dem=dem_c, device=device)

    log.info("DEM z_top=%.0f m  |  kernel FFTs: %d × (%d, %d)",
             z_top_val, D, grav_fwd.fH, grav_fwd.fW // 2 + 1)

    # ── Step 6: Determine grid CRS ────────────────────────────────────────
    try:
        _epsg   = int(grav_c.rio.crs.to_epsg())
        grid_crs = f"EPSG:{_epsg}"
    except Exception:
        _sr = grav_c.coords.get("spatial_ref", None)
        if _sr is not None:
            grid_crs = _sr.attrs.get("crs_wkt",
                       _sr.attrs.get("grid_mapping_name", str(_sr.values)))
        else:
            raise RuntimeError(
                "Could not determine grid CRS from grav_c. "
                "Ensure rioxarray is imported and the NetCDF file stores a spatial_ref."
            )
    log.info("Grid CRS: %s", grid_crs)

    # ── Step 7: Load well-log constraints (optional) ──────────────────────
    density_constraints: List[Dict] = []
    chi_constraints:     List[Dict] = []

    use_well_logs = args.las_dir is not None and args.json_path is not None
    if use_well_logs:
        log.info("=" * 60)
        log.info("STEP 7: Loading well-log constraints")
        log.info("=" * 60)
        density_constraints, chi_constraints = load_all_well_constraints(
            las_dir   = args.las_dir,
            json_path = args.json_path,
            grid_x    = grav_c.x.values,
            grid_y    = grav_c.y.values,
            D         = D,
            resolution= res,
            grid_crs  = grid_crs,
        )
    else:
        log.info("STEP 7: Well-log constraints skipped (--las_dir / --json_path not provided)")

    # ── Step 8: Load seismic constraints (optional) ───────────────────────
    seismic_2d_prior = None
    seismic_2d_mask  = None
    seismic_3d_prior = None
    seismic_3d_mask  = None

    use_seismic_2d = args.seismic_2d is not None
    use_seismic_3d = args.seismic_3d is not None

    if use_seismic_2d or use_seismic_3d:
        log.info("=" * 60)
        log.info("STEP 8: Loading seismic constraints")
        log.info("=" * 60)

    if use_seismic_2d:
        seismic_2d_prior_np = load_2d_fwi_prior(
            args.seismic_2d,
            grid_x=grav_c.x.values, grid_y=grav_c.y.values,
            D=D, resolution=res, grid_crs=grid_crs,
        )
        seismic_2d_prior = torch.tensor(seismic_2d_prior_np, dtype=torch.float32, device=device)
        seismic_2d_mask  = ~torch.isnan(seismic_2d_prior)
        log.info("2D FWI mask: %d constrained voxels", seismic_2d_mask.sum().item())
    else:
        log.info("2D FWI seismic skipped (--seismic_2d not provided)")

    if use_seismic_3d:
        seismic_3d_prior_np = load_3d_reflection_prior(
            args.seismic_3d,
            grid_x=grav_c.x.values, grid_y=grav_c.y.values,
            D=D, resolution=res, grid_crs=grid_crs,
        )
        seismic_3d_prior = torch.tensor(
            np.nan_to_num(seismic_3d_prior_np, nan=0.0), dtype=torch.float32, device=device
        )
        seismic_3d_mask = torch.tensor(~np.isnan(seismic_3d_prior_np),
                                       dtype=torch.bool, device=device)
        log.info("3D reflection mask: %d/%d constrained voxels",
                 seismic_3d_mask.sum().item(), seismic_3d_mask.numel())
    else:
        log.info("3D reflection seismic skipped (--seismic_3d not provided)")

    # ── Step 9: Build U-Net model + optimizer ─────────────────────────────
    log.info("=" * 60)
    log.info("STEP 9: Building U-Net model")
    log.info("=" * 60)

    # Physical bounds
    rho_lo, rho_hi = -1.0, 1.0   # density contrast (g/cc)
    chi_lo, chi_hi =  0.0, 0.1   # susceptibility (SI)

    model = UNet2DEncoder(in_channels=2, base_channels=32, out_channels=2 * D).to(device)
    model.train()

    total_params = sum(p.numel() for p in model.parameters())
    log.info("U-Net parameters: %s", f"{total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.num_iterations,
        pct_start=0.05, anneal_strategy="cos",
    )

    # Constraint weights
    lambda_well_rho   = 1.0
    lambda_well_chi   = 200.0
    lambda_seismic_2d = 0.3
    lambda_seismic_3d = 0.5

    # ── Step 10: Training loop ────────────────────────────────────────────
    log.info("=" * 60)
    log.info("STEP 10: Training  (%d iterations)", args.num_iterations)
    log.info("=" * 60)

    history: List[Dict] = []

    for it in range(1, args.num_iterations + 1):
        optimizer.zero_grad()

        out     = model(x_input)                       # [1, 2D, H, W]
        rho_raw = out[0, :D]
        chi_raw = out[0, D:]
        rho_vol, chi_vol = apply_bounds(rho_raw, chi_raw, rho_lo, rho_hi, chi_lo, chi_hi)

        g_pred = grav_fwd(rho_vol)                     # [H, W]
        m_pred = mag_fwd(chi_vol)                      # [H, W]

        # Mean-subtracted misfit
        g_pred_c = g_pred - g_pred.mean()
        m_pred_c = m_pred - m_pred.mean()
        g_obs_c  = g_obs  - g_obs_mean
        m_obs_c  = m_obs  - m_obs_mean

        loss, parts = joint_inversion_loss(
            g_pred=g_pred_c[None, None], m_pred=m_pred_c[None, None],
            g_obs=g_obs_c[None, None],   m_obs=m_obs_c[None, None],
            rho=rho_vol[None, None],     chi=chi_vol[None, None],
            g_sigma=g_sigma,
            m_sigma=m_sigma,
            structural_model=structural_model,
            lambda_grav=1.0,
            lambda_mag=1.0,
            beta_smooth=8e-3, lambda_xgrad=0.1,
        )



        # ===== Structural Geology Constraint =====
        structural_loss = structural_geology_loss(
            rho_vol[None, None],
            chi_vol[None, None],
            structural_mask=structural_model,
            fault_mask=None,
        )

        loss = loss + structural_loss
        parts["structural"] = structural_loss.item()

        # Seismic constraints
        if use_seismic_2d and seismic_2d_mask.any():
            L_seismic_2d = ((rho_vol - seismic_2d_prior)[seismic_2d_mask] ** 2).mean()
        else:
            L_seismic_2d = rho_vol.new_tensor(0.0)

        if use_seismic_3d and seismic_3d_mask.any():
            L_seismic_3d = ((rho_vol - seismic_3d_prior)[seismic_3d_mask] ** 2).mean()
        else:
            L_seismic_3d = rho_vol.new_tensor(0.0)

        loss = (loss
                + lambda_seismic_2d * L_seismic_2d
                + lambda_seismic_3d * L_seismic_3d)
        parts["L_seis2d"] = L_seismic_2d.item()
        parts["L_seis3d"] = L_seismic_3d.item()
        parts["L_total"] += (lambda_seismic_2d * L_seismic_2d.item()
                             + lambda_seismic_3d * L_seismic_3d.item())

        # Well-log constraints
        L_well_rho = well_log_loss(rho_vol, density_constraints, "rho_contrast")
        L_well_chi = well_log_loss(chi_vol, chi_constraints,     "chi")
        loss = loss + lambda_well_rho * L_well_rho + lambda_well_chi * L_well_chi
        parts["L_well_rho"] = L_well_rho.item()
        parts["L_well_chi"] = L_well_chi.item()
        parts["L_total"]   += (lambda_well_rho * L_well_rho.item()
                               + lambda_well_chi * L_well_chi.item())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=50.0)
        optimizer.step()
        scheduler.step()

        history.append(parts)

        if it % 100 == 0 or it == 1:
            rho_np_diag = rho_vol.detach().cpu().numpy()
            chi_np_diag = chi_vol.detach().cpu().numpy()
            log.info(
                "[%5d/%d] L=%.3e Lg=%.3e Lm=%.3e Ls=%.3e Lx=%.3e Lstruct=%.3e "
                "Ls2d=%.3e Ls3d=%.3e Lwr=%.3e Lwc=%.3e | "
                "rho=[%.3f,%.3f] chi=[%.5f,%.5f] lr=%.1e",
                it, args.num_iterations,
                parts["L_total"],
                parts["L_grav"], parts["L_mag"],
                parts["L_smooth"], parts["L_xgrad"],
                parts["structural"],
                parts["L_seis2d"], parts["L_seis3d"],
                parts["L_well_rho"], parts["L_well_chi"],
                rho_np_diag.min(), rho_np_diag.max(),
                chi_np_diag.min(), chi_np_diag.max(),
                scheduler.get_last_lr()[0],
            )

    # ── Step 11: Evaluate recovered model ─────────────────────────────────
    log.info("=" * 60)
    log.info("STEP 11: Evaluating recovered model")
    log.info("=" * 60)

    model.eval()
    with torch.no_grad():
        out = model(x_input)
        rho_vol, chi_vol = apply_bounds(out[0, :D], out[0, D:], rho_lo, rho_hi, chi_lo, chi_hi)
        g_final = grav_fwd(rho_vol).cpu().numpy()
        m_final = mag_fwd(chi_vol).cpu().numpy()

    rho_np = rho_vol.cpu().numpy()
    chi_np = chi_vol.cpu().numpy()

    # Residuals used by inversion (keep existing behavior)
    g_resid_centered = g_final - g_final.mean() - (grav_c.values - grav_c.values.mean())
    m_resid_centered = m_final - m_final.mean() - (mag_c.values - mag_c.values.mean())

    # Residuals used during optimization
    g_resid_centered = g_final - g_final.mean() - (grav_c.values - grav_c.values.mean())
    m_resid_centered = m_final - m_final.mean() - (mag_c.values - mag_c.values.mean())

# Physical residuals
    g_resid = g_final - grav_c.values
    m_resid = m_final - mag_c.values

    log.info("Density contrast range   : [%.4f, %.4f] g/cc", rho_np.min(), rho_np.max())
    log.info("Susceptibility range     : [%.6f, %.6f] SI",   chi_np.min(), chi_np.max())
    log.info("Gravity residual RMS     : %.4f mGal",
             float(np.sqrt((g_resid ** 2).mean())))
    log.info("Magnetic residual RMS    : %.4f nT",
             float(np.sqrt((m_resid ** 2).mean())))

    # ── Step 12: Save outputs ─────────────────────────────────────────────
    log.info("=" * 60)
    log.info("STEP 12: Saving outputs to %s", args.output_dir)
    log.info("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)
    base = os.path.join(args.output_dir, args.output_name)

    save_loss_curves(history, out_path=base + "_loss_curves.png")
    save_model_slices(rho_np, chi_np, res, out_path=base + "_model_slices.png")
    save_npz(
        out_path=base + ".npz",
        rho_np=rho_np, chi_np=chi_np,
        x_coords=x_coords, y_coords=y_coords, z_layer_centres=z_layer_centres,
        g_final=g_final, m_final=m_final,
        g_obs_vals=grav_c.values, m_obs_vals=mag_c.values,
        g_resid=g_resid, m_resid=m_resid,
        g_resid_centered=g_resid_centered,
        m_resid_centered=m_resid_centered,
        res=res, num_iterations=args.num_iterations,
        rho_lo=rho_lo, rho_hi=rho_hi, chi_lo=chi_lo, chi_hi=chi_hi,
        dx=dx, dy=dy, z_min=z_min,
    )

    log.info("All outputs written. Pipeline complete.")


# =============================================================================
#  Entry point
# =============================================================================

if __name__ == "__main__":
    parser = build_parser()
    args   = parser.parse_args()
    main(args)