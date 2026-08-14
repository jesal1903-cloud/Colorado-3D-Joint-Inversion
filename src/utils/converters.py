#!/usr/bin/env python3
"""
converters.py
=============
Format conversion pipeline for multiphysics inversion inputs.

All outputs are plug-and-play with ``inversion_model_seismic.ipynb``.
No notebook cell needs modification — just point the notebook paths at
the files produced here.

Gravity / Magnetic → NetCDF4
─────────────────────────────
  csv_xyz_to_netcdf4(csv_path, out_path, ...)
      CSV/XYZ scattered survey → regular-grid NetCDF4 DataArray (y, x)
      Applies QC, CRS reprojection, unit scaling, and sigma-clip.

  geotiff_to_netcdf4(tiff_path, out_path, ...)
      GeoTIFF / COG raster → NetCDF4 DataArray (y, x)
      Handles nodata, scale/offset, and optional reprojection.

  netcdf_cf_to_netcdf4(nc_path, out_path, ...)
      Existing NetCDF / CF file → harmonised pipeline-ready NetCDF4.
      Renames spatial dims to x/y, writes spatial_ref, reprojects if needed.

Seismic → HDF5
──────────────
  segy_to_hdf5(segy_path, out_path, ...)
      SEG-Y → HDF5.  Three modes controlled by the *mode* parameter:
        "2d_fwi"        survey_lines/{line}/coordinates/ + velocity_model/vp_true
        "3d_reflection" density_model/density_3d + coordinates/
        "validation_3d" velocity_model/vp_3d + coordinates/ + horizons/

  segd_to_hdf5(segd_path, out_path, ...)
      SEG-D Rev 1/2 (format codes 8036/8058 32-bit float, 8015 24-bit int)
      → HDF5 in the 2D FWI layout used by the notebook.

Well logs → normalised LAS + JSON
──────────────────────────────────
  las_to_json(las_files, out_dir, json_path, ...)
      Raw / non-standard LAS → normalised LAS with standard curve mnemonics
      (RHOB, GR, NPHI, …) + ``well_locations_summary.json``.

  dlis_to_json(dlis_files, out_dir, json_path, ...)
      DLIS → same normalised LAS + JSON pair.

Required packages
─────────────────
  numpy  scipy  h5py  xarray  rioxarray  lasio  pyproj  netCDF4  segyio  dlisio

  Install all at once:
    pip install numpy scipy h5py xarray rioxarray lasio pyproj netCDF4 segyio dlisio
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

import h5py
import lasio
import numpy as np
import pandas as pd
import pyproj
import rioxarray  # noqa: F401 — registers .rio accessor on xarray objects
import scipy.interpolate
import xarray as xr
from rasterio.enums import Resampling as _Resampling

import dlisio
import segyio

# ─────────────────────────────────────────────────────────────────────
# Dependency check
# ─────────────────────────────────────────────────────────────────────

_REQUIRED = {
    "h5py":      h5py,
    "lasio":     lasio,
    "numpy":     np,
    "pandas":    pd,
    "pyproj":    pyproj,
    "rioxarray": rioxarray,
    "scipy":     scipy,
    "xarray":    xr,
    "segyio":    segyio,
    "dlisio":    dlisio,
}

_missing = [name for name, mod in _REQUIRED.items() if mod is None]
if _missing:
    raise ImportError(
        "The following required packages are not installed.  "
        f"Run: pip install {' '.join(_missing)}\n"
        f"Missing: {_missing}"
    )


# ─────────────────────────────────────────────────────────────────────
# Constants & lookup tables
# ─────────────────────────────────────────────────────────────────────

FT_TO_M: float = 0.3048
M_TO_FT: float = 1.0 / FT_TO_M

# Multiplicative factors to convert input units → pipeline target units
#   gravity  → mGal   |   magnetic → nT   |   dem → m
_UNIT_SCALE: Dict[str, Dict[str, float]] = {
    "gravity": {
        "mgal": 1.0, "mGal": 1.0,
        "ugal": 1e-3, "uGal": 1e-3,
        "gal":  1e3,  "Gal":  1e3,
        "m/s2": 1e5,  "m/s^2": 1e5,
    },
    "magnetic": {
        "nt": 1.0,  "nT": 1.0,
        "t":  1e9,  "T":  1e9,
        "mt": 1e6,  "mT": 1e6,
        "ut": 1e3,  "uT": 1e3,
    },
    "dem": {
        "m": 1.0, "meter": 1.0, "metre": 1.0,
        "ft": FT_TO_M, "feet": FT_TO_M, "foot": FT_TO_M,
        "km": 1e3,
    },
}

_DATA_UNITS: Dict[str, str] = {"gravity": "mGal", "magnetic": "nT", "dem": "m"}

# Standard output mnemonic → accepted input aliases (case-insensitive)
# Used by both las_to_json and dlis_to_json.
_DEFAULT_MNEMONIC_MAP: Dict[str, List[str]] = {
    "RHOB":     ["RHOB", "RHOZ", "DEN", "DENS", "DENB", "ZDEN", "PEFZ"],
    "RHOB_AVG": ["RHOB_AVG", "RHOZ_AVG", "RHOB_SMOOTH", "RHOB_SM"],
    "PHID":     ["PHID", "DPHI", "DPHZ", "DPOR", "DPHI_D"],
    "PHIT":     ["PHIT", "TPHI", "TPOR", "PHI_T"],
    "PHIE":     ["PHIE", "EPHI", "EPOR", "PHI_E"],
    "NPHI":     ["NPHI", "TNPH", "NPHI_SS", "NPHI_LS", "NPORS", "CNPHI"],
    "RESD":     ["RESD", "RD", "ILD", "LLD", "HLLD", "AT90", "RXO", "RT90"],
    "MAGSUS":   ["MAGSUS", "MSUS", "MAG_SUS", "KAPPA", "CHI", "SUSC"],
    "VSH":      ["VSH", "VCL", "VCLAY", "GR_VSH", "VSHALE"],
    "GR":       ["GR", "GRS", "SGR", "CGR", "GRC", "GR_RAW", "NGAM"],
}


# ─────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────

def _reproject(
    x: np.ndarray, y: np.ndarray, src_crs: str, dst_crs: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Reproject (x, y) coordinate arrays from *src_crs* to *dst_crs*."""
    t = pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return t.transform(x, y)


def _sigma_clip_mask(values: np.ndarray, sigma: float) -> np.ndarray:
    """Boolean mask: True where *values* is within *sigma* MADs of the median."""
    if sigma <= 0:
        return np.ones(len(values), dtype=bool)
    finite = np.isfinite(values)
    med = np.nanmedian(values[finite])
    mad = np.nanmedian(np.abs(values[finite] - med)) * 1.4826 + 1e-30
    return finite & (np.abs(values - med) < sigma * mad)


def _scatter_to_regular_grid(
    x: np.ndarray,
    y: np.ndarray,
    v: np.ndarray,
    res: float,
    method: str = "linear",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate scattered (x, y, v) to a regular grid of spacing *res*.

    Returns
    -------
    xi   : (W,) x-coordinates (easting, metres)
    yi   : (H,) y-coordinates (northing, metres)
    grid : (H, W) float32 gridded values (NaN outside convex hull)
    """
    xi = np.arange(x.min(), x.max() + res * 0.5, res)
    yi = np.arange(y.min(), y.max() + res * 0.5, res)
    XI, YI = np.meshgrid(xi, yi)
    grid = scipy.interpolate.griddata((x, y), v, (XI, YI), method=method).astype(np.float32)
    return xi, yi, grid


def _build_dataarray(
    data: np.ndarray,
    xi: np.ndarray,
    yi: np.ndarray,
    crs: str,
    attrs: dict,
    name: Optional[str] = None,
) -> xr.DataArray:
    """Construct an xr.DataArray with (y, x) dims and an embedded CRS."""
    da = xr.DataArray(
        data.astype(np.float32),
        dims=("y", "x"),
        coords={"y": yi.astype(np.float64), "x": xi.astype(np.float64)},
        attrs=attrs,
        name=name,
    )
    da = da.rio.set_spatial_dims(x_dim="x", y_dim="y")
    da = da.rio.write_crs(crs)
    return da


def _write_dataarray(da: xr.DataArray, out_path: str) -> None:
    """Write DataArray to a compressed NETCDF4 file with spatial_ref preserved."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    vname = da.name or "__xarray_dataarray_variable__"
    enc = {vname: {"zlib": True, "complevel": 4, "dtype": "float32"}}
    da.to_netcdf(out_path, format="NETCDF4", encoding=enc)
    print(f"  → {out_path}  shape={da.shape}")


def _unit_factor(unit: str, data_type: str) -> float:
    """Return the multiplicative factor to convert *unit* to the pipeline target."""
    table = _UNIT_SCALE.get(data_type, {})
    factor = table.get(unit) or table.get(unit.lower())
    if factor is None:
        warnings.warn(
            f"Unknown unit '{unit}' for data_type='{data_type}'. "
            "Assuming factor=1.0 (no unit conversion)."
        )
        return 1.0
    return factor


def _apply_segy_coord_scalar(
    coords: np.ndarray, scalars: np.ndarray
) -> np.ndarray:
    """Apply SEG-Y coordinate scalars element-wise.

    SEG-Y convention:
    * scalar < 0  →  divide by abs(scalar)
    * scalar > 0  →  multiply by scalar
    * scalar == 0 →  treat as 1
    """
    out = coords.astype(np.float64)
    pos = scalars > 0
    neg = scalars < 0
    out[pos] *= scalars[pos]
    out[neg] /= np.abs(scalars[neg])
    return out


# ─────────────────────────────────────────────────────────────────────
# Output QC helpers
# ─────────────────────────────────────────────────────────────────────

def _qc_netcdf4(
    path: str,
    *,
    source_values: Optional[np.ndarray] = None,
) -> None:
    """Re-open a written NetCDF4 file and verify structure and value integrity.

    Checks performed:
    * Readable by xarray with ``decode_coords="all"``.
    * Dimensions are exactly ``('y', 'x')``.
    * ``spatial_ref`` coordinate is present (CRS embedded).
    * Grid has more than one cell in each axis.
    * Not all values are NaN; warns if > 90% are NaN.
    * All finite values are not identical (not a constant fill).
    * If *source_values* is supplied, output range has not exploded (> 100×
      source span) or collapsed (< 1% of source span), and mean shift is
      less than 10× source span.
    """
    tag = f"[QC {Path(path).name}]"
    try:
        da = xr.open_dataarray(path, decode_coords="all").squeeze()
    except Exception as exc:
        warnings.warn(f"{tag} could not re-open file: {exc}")
        return

    issues: List[str] = []

    # ── Structural ────────────────────────────────────────────────────
    if set(da.dims) != {"y", "x"}:
        issues.append(f"dims are {tuple(da.dims)!r}, expected ('y', 'x')")
    if "spatial_ref" not in da.coords:
        issues.append("'spatial_ref' coordinate missing — CRS not embedded")
    if da.shape[0] < 2 or da.shape[1] < 2:
        issues.append(f"grid is too small: shape={da.shape}")

    # ── Value integrity ───────────────────────────────────────────────
    vals = da.values.ravel().astype(np.float64)
    n_nan = int(np.isnan(vals).sum())
    nan_frac = n_nan / vals.size if vals.size else 1.0
    finite = vals[np.isfinite(vals)]

    if nan_frac == 1.0:
        issues.append("ALL values are NaN — data not written or entirely masked")
    elif nan_frac > 0.9:
        issues.append(
            f"{nan_frac*100:.1f}% NaN — check nodata handling, CRS, or extent"
        )

    vmin = vmax = vmean = float("nan")
    if len(finite):
        vmin, vmax = float(finite.min()), float(finite.max())
        vmean = float(finite.mean())
        if float(finite.std()) == 0.0:
            issues.append(
                f"all finite values are identical ({vmin:.6g})"
                " — possible fill-value or unit-scale error"
            )
        if source_values is not None:
            src = source_values[np.isfinite(source_values)].astype(np.float64)
            if len(src):
                src_span = float(src.max() - src.min())
                out_span = float(vmax - vmin)
                src_mean = float(src.mean())
                if src_span > 0:
                    ratio = out_span / src_span
                    if ratio > 100:
                        issues.append(
                            f"value range exploded: input span={src_span:.4g}, "
                            f"output span={out_span:.4g} (\u00d7{ratio:.0f})"
                        )
                    elif ratio < 0.01:
                        issues.append(
                            f"value range collapsed: input span={src_span:.4g}, "
                            f"output span={out_span:.4g} (\u00d7{ratio:.3g})"
                        )
                    shift = abs(vmean - src_mean)
                    if shift > 10 * src_span:
                        issues.append(
                            f"large mean offset: input mean={src_mean:.4g}, "
                            f"output mean={vmean:.4g} (shift={shift:.4g})"
                        )

    crs_str = str(da.rio.crs) if da.rio.crs is not None else "None"
    shape = da.shape
    da.close()

    if issues:
        for iss in issues:
            warnings.warn(f"{tag} {iss}")
    else:
        print(
            f"{tag} OK  shape={shape}  CRS={crs_str}  "
            f"range=[{vmin:.4g}, {vmax:.4g}]  NaN={nan_frac*100:.1f}%"
        )


def _qc_hdf5(
    path: str,
    mode: str,
    line_name: Optional[str] = None,
) -> None:
    """Re-open a written HDF5 file and verify structure and value integrity.

    Checks are specific to each *mode* (see ``segy_to_hdf5`` docstring for the
    expected dataset paths and shapes).

    ``"2d_fwi"``
        Required groups and datasets exist.  lat/lon/z_m/vp_true shapes are
        mutually consistent.  lat in [-90, 90], lon in [-180, 180].  z_m is
        non-negative and monotonically non-decreasing.  vp_true has finite
        values.

    ``"3d_reflection"``
        ``density_3d`` is 3-D with axes matching coordinate arrays.  Not
        all-NaN.

    ``"validation_3d"``
        ``vp_3d`` is 3-D with depth axis matching ``depth_m``.  Not all-NaN.
    """
    tag = f"[QC {Path(path).name}]"
    try:
        hf = h5py.File(path, "r")
    except Exception as exc:
        warnings.warn(f"{tag} could not re-open file: {exc}")
        return

    issues: List[str] = []
    summary = ""

    with hf:
        if mode == "2d_fwi":
            if "survey_lines" not in hf:
                issues.append("missing group 'survey_lines'")
            else:
                lines = list(hf["survey_lines"].keys())
                if not lines:
                    issues.append("'survey_lines' group is empty")
                else:
                    lname = line_name or lines[0]
                    base  = f"survey_lines/{lname}"
                    for ds in [
                        f"{base}/coordinates/latitude",
                        f"{base}/coordinates/longitude",
                        f"{base}/coordinates/z_m",
                        f"{base}/velocity_model/vp_true",
                    ]:
                        if ds not in hf:
                            issues.append(f"missing dataset '{ds}'")
                    if not issues:
                        lat = hf[f"{base}/coordinates/latitude"][:]
                        lon = hf[f"{base}/coordinates/longitude"][:]
                        z_m = hf[f"{base}/coordinates/z_m"][:]
                        vp  = hf[f"{base}/velocity_model/vp_true"][:]
                        if lat.shape != lon.shape:
                            issues.append(
                                f"lat/lon shape mismatch: {lat.shape} vs {lon.shape}"
                            )
                        if vp.ndim != 2:
                            issues.append(f"vp_true is {vp.ndim}-D, expected 2-D")
                        elif vp.shape[0] != len(lat):
                            issues.append(
                                f"trace count mismatch: lat={len(lat)}, "
                                f"vp_true dim0={vp.shape[0]}"
                            )
                        elif vp.shape[1] != len(z_m):
                            issues.append(
                                f"sample count mismatch: z_m={len(z_m)}, "
                                f"vp_true dim1={vp.shape[1]}"
                            )
                        if len(lat) and (lat.min() < -90 or lat.max() > 90):
                            issues.append(
                                f"latitude out of [-90, 90]: "
                                f"[{lat.min():.4g}, {lat.max():.4g}]"
                            )
                        if len(lon) and (lon.min() < -180 or lon.max() > 180):
                            issues.append(
                                f"longitude out of [-180, 180]: "
                                f"[{lon.min():.4g}, {lon.max():.4g}]"
                            )
                        if len(z_m):
                            if float(z_m.min()) < 0:
                                issues.append(
                                    f"z_m has negative values: min={z_m.min():.4g}"
                                )
                            if not np.all(np.diff(z_m) >= 0):
                                issues.append(
                                    "z_m is not monotonically non-decreasing"
                                )
                        finite_vp = vp[np.isfinite(vp)]
                        nan_frac = 1.0 - len(finite_vp) / vp.size if vp.size else 1.0
                        if nan_frac == 1.0:
                            issues.append("vp_true contains no finite values")
                        elif nan_frac > 0.5:
                            issues.append(f"vp_true is {nan_frac*100:.1f}% NaN/inf")
                        if not issues:
                            summary = (
                                f"traces={vp.shape[0]}  samples={vp.shape[1]}  "
                                f"vp=[{finite_vp.min():.4g}, {finite_vp.max():.4g}] m/s  "
                                f"z=[{z_m.min():.1f}, {z_m.max():.1f}] m"
                            )

        elif mode == "3d_reflection":
            for ds in [
                "density_model/density_3d",
                "coordinates/longitude",
                "coordinates/latitude",
                "coordinates/depth_m",
            ]:
                if ds not in hf:
                    issues.append(f"missing dataset '{ds}'")
            if not issues:
                dens = hf["density_model/density_3d"][:]
                lon  = hf["coordinates/longitude"][:]
                lat  = hf["coordinates/latitude"][:]
                z_m  = hf["coordinates/depth_m"][:]
                if dens.ndim != 3:
                    issues.append(f"density_3d is {dens.ndim}-D, expected 3-D")
                else:
                    if dens.shape[0] != len(lon):
                        issues.append(
                            f"lon axis {len(lon)} \u2260 density_3d dim0 {dens.shape[0]}"
                        )
                    if dens.shape[1] != len(lat):
                        issues.append(
                            f"lat axis {len(lat)} \u2260 density_3d dim1 {dens.shape[1]}"
                        )
                    if dens.shape[2] != len(z_m):
                        issues.append(
                            f"depth axis {len(z_m)} \u2260 density_3d dim2 {dens.shape[2]}"
                        )
                finite_d = dens[np.isfinite(dens)]
                if len(finite_d) == 0:
                    issues.append("density_3d contains no finite values")
                elif not issues:
                    summary = (
                        f"shape={dens.shape}  "
                        f"density=[{finite_d.min():.4g}, {finite_d.max():.4g}] kg/m\u00b3"
                    )

        elif mode == "validation_3d":
            for ds in ["velocity_model/vp_3d", "coordinates/depth_m"]:
                if ds not in hf:
                    issues.append(f"missing dataset '{ds}'")
            if not issues:
                vp  = hf["velocity_model/vp_3d"][:]
                z_m = hf["coordinates/depth_m"][:]
                if vp.ndim != 3:
                    issues.append(f"vp_3d is {vp.ndim}-D, expected 3-D")
                elif vp.shape[2] != len(z_m):
                    issues.append(
                        f"depth axis {len(z_m)} \u2260 vp_3d dim2 {vp.shape[2]}"
                    )
                finite_vp = vp[np.isfinite(vp)]
                if len(finite_vp) == 0:
                    issues.append("vp_3d contains no finite values")
                elif not issues:
                    summary = (
                        f"shape={vp.shape}  "
                        f"vp=[{finite_vp.min():.4g}, {finite_vp.max():.4g}] m/s"
                    )

    if issues:
        for iss in issues:
            warnings.warn(f"{tag} {iss}")
    else:
        print(f"{tag} OK  {summary}")


def _qc_las_json(json_path: str, out_dir: str) -> None:
    """Verify well_locations_summary.json structure and a sample normalised LAS.

    Checks performed:
    * JSON is readable and has a non-empty ``"wells"`` list.
    * Each entry has ``file``, ``latitude``, ``longitude`` in valid ranges.
    * At least one ``.las`` file exists in *out_dir*.
    * Sample LAS is readable; depth index is monotone; no curve is entirely
      NaN / null-filled.
    """
    tag = f"[QC {Path(json_path).name}]"
    issues: List[str] = []

    # ── JSON structure ────────────────────────────────────────────────
    try:
        with open(json_path) as fh:
            data = json.load(fh)
    except Exception as exc:
        warnings.warn(f"{tag} could not read JSON: {exc}")
        return

    if "wells" not in data:
        issues.append("missing top-level 'wells' key")
    else:
        wells = data["wells"]
        if not wells:
            issues.append("'wells' list is empty — no wells were written")
        for w in wells:
            missing_keys = [k for k in ("file", "latitude", "longitude") if k not in w]
            if missing_keys:
                issues.append(f"entry {w} missing keys: {missing_keys}")
                continue
            lat, lon = w["latitude"], w["longitude"]
            if not (-90 <= lat <= 90):
                issues.append(
                    f"{w['file']}: latitude {lat:.4g} out of range [-90, 90]"
                )
            if not (-180 <= lon <= 180):
                issues.append(
                    f"{w['file']}: longitude {lon:.4g} out of range [-180, 180]"
                )

    # ── Sample LAS integrity ──────────────────────────────────────────
    las_files = sorted(
        p for p in Path(out_dir).iterdir() if p.suffix.lower() == ".las"
    )
    if not las_files:
        issues.append(f"no .las files found in '{out_dir}'")
    else:
        sample = las_files[0]
        try:
            las = lasio.read(str(sample))
            depths = las.curves[0].data
            if len(depths) > 1 and not np.all(np.diff(depths) >= 0):
                issues.append(
                    f"{sample.name}: depth index is not monotonically non-decreasing"
                )
            try:
                null_val = las.well.NULL.value
            except AttributeError:
                null_val = -9999.25
            for curve in las.curves[1:]:
                arr = np.array(curve.data, dtype=np.float64)
                arr[arr == null_val] = np.nan
                if len(arr) > 0 and not np.any(np.isfinite(arr)):
                    issues.append(
                        f"{sample.name}: curve '{curve.mnemonic}' "
                        "is entirely NaN / null-filled"
                    )
        except Exception as exc:
            issues.append(f"could not read sample LAS '{sample.name}': {exc}")

    if issues:
        for iss in issues:
            warnings.warn(f"{tag} {iss}")
    else:
        n_wells = len(data.get("wells", []))
        n_las   = len(las_files)
        print(
            f"{tag} OK  {n_wells} well(s) in JSON  "
            f"{n_las} LAS file(s) in '{out_dir}'"
        )


# ─────────────────────────────────────────────────────────────────────
# 1.  CSV / XYZ  →  NetCDF4
# ─────────────────────────────────────────────────────────────────────

def csv_xyz_to_netcdf4(
    csv_path: str,
    out_path: str,
    *,
    x_col: str = "x",
    y_col: str = "y",
    value_col: str = "value",
    input_crs: str = "EPSG:4326",
    target_crs: Optional[str] = "EPSG:3857",
    grid_res: Optional[float] = None,
    data_type: str = "gravity",
    unit_in: str = "mGal",
    inclination: Optional[float] = None,
    declination: Optional[float] = None,
    amplitude_nT: Optional[float] = None,
    outlier_sigma: float = 5.0,
    interp_method: str = "linear",
    sep: str = ",",
    comment: str = "#",
    name: Optional[str] = None,
) -> xr.DataArray:
    """Convert a scattered CSV / XYZ gravity or magnetic survey to NetCDF4.

    The output DataArray has dims ``(y, x)`` in projected metres and a
    ``spatial_ref`` coordinate so it loads directly with::

        xr.open_dataarray(out_path, decode_coords="all").squeeze()

    Parameters
    ----------
    csv_path      : Input delimited file (CSV, space- or tab-separated XYZ, …).
    out_path      : Destination NetCDF4 file path.
    x_col, y_col  : Column names for horizontal coordinates.
    value_col     : Column name for the measured field value.
    input_crs     : CRS of the input x/y columns ("EPSG:4326" for lon/lat).
    target_crs    : Output projected CRS in metres.  Defaults to EPSG:3857.
    grid_res      : Regular grid cell size in metres.  Auto (~span/50) when None.
    data_type     : "gravity" | "magnetic" | "dem" — drives unit conversion & attrs.
    unit_in       : Input unit string ("mGal", "uGal", "nT", "m", …).
    inclination   : Geomagnetic inclination (°) — stored in attrs for magnetic data.
    declination   : Geomagnetic declination (°).
    amplitude_nT  : Ambient field strength (nT).
    outlier_sigma : Sigma-clip threshold (0 = disabled).
    interp_method : scipy.interpolate.griddata method ("linear", "cubic", "nearest").
    sep           : Column separator (default: comma).
    comment       : Comment character to skip header lines.
    name          : Variable name written to the NetCDF4 file.

    Returns
    -------
    xr.DataArray  (also written to *out_path*)
    """
    print(f"[csv_xyz_to_netcdf4] {csv_path}")
    df = pd.read_csv(csv_path, sep=sep, comment=comment, engine="python")
    for col in (x_col, y_col, value_col):
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found.  Available: {list(df.columns)}")

    x = df[x_col].to_numpy(dtype=np.float64)
    y = df[y_col].to_numpy(dtype=np.float64)
    v = df[value_col].to_numpy(dtype=np.float64)

    # ── QC: NaN removal + sigma-clip ────────────────────────────────
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(v)
    mask &= _sigma_clip_mask(v, outlier_sigma)
    n_removed = int((~mask).sum())
    if n_removed:
        print(f"  QC: removed {n_removed} point(s) (NaN / outlier)")
    x, y, v = x[mask], y[mask], v[mask]

    # ── CRS: project to metres ───────────────────────────────────────
    if input_crs != target_crs:
        x, y = _reproject(x, y, input_crs, target_crs)

    # ── Unit conversion ──────────────────────────────────────────────
    scale = _unit_factor(unit_in, data_type)
    if scale != 1.0:
        v = v * scale
        print(f"  Unit: {unit_in} × {scale} → {_DATA_UNITS.get(data_type, '?')}")

    # ── Auto grid resolution ─────────────────────────────────────────
    if grid_res is None:
        span = max(float(x.max() - x.min()), float(y.max() - y.min()))
        grid_res = span / 50.0
        print(f"  Auto grid_res: {grid_res:.1f} m")

    xi, yi, grid = _scatter_to_regular_grid(x, y, v, grid_res, interp_method)
    print(f"  Grid: {grid.shape}")

    # ── Attributes ───────────────────────────────────────────────────
    attrs: dict = {"units": _DATA_UNITS.get(data_type, ""), "data_type": data_type}
    if data_type == "magnetic":
        if inclination  is not None: attrs["inclination"]  = float(inclination)
        if declination  is not None: attrs["declination"]  = float(declination)
        if amplitude_nT is not None: attrs["amplitude_nT"] = float(amplitude_nT)

    da = _build_dataarray(grid, xi, yi, target_crs, attrs, name=name or data_type)
    _write_dataarray(da, out_path)
    _qc_netcdf4(out_path, source_values=v)
    return da


# ─────────────────────────────────────────────────────────────────────
# 2.  GeoTIFF / COG  →  NetCDF4
# ─────────────────────────────────────────────────────────────────────

def geotiff_to_netcdf4(
    tiff_path: str,
    out_path: str,
    *,
    band: int = 1,
    data_type: str = "gravity",
    unit_in: str = "mGal",
    target_crs: Optional[str] = "EPSG:3857",
    inclination: Optional[float] = None,
    declination: Optional[float] = None,
    amplitude_nT: Optional[float] = None,
    nodata_override: Optional[float] = None,
    resampling: str = "bilinear",
    name: Optional[str] = None,
) -> xr.DataArray:
    """Ingest a GeoTIFF or Cloud-Optimised GeoTIFF and export to NetCDF4.

    Handles nodata replacement with NaN, GeoTIFF scale/offset metadata, and
    optional reprojection.  The output loads directly with::

        xr.open_dataarray(out_path, decode_coords="all").squeeze()

    Parameters
    ----------
    tiff_path       : GeoTIFF / COG file path.
    out_path        : Destination NetCDF4 file path.
    band            : Raster band to extract (1-based).
    data_type       : "gravity" | "magnetic" | "dem".
    unit_in         : Input band unit for scaling.
    target_crs      : Reproject to this CRS (None = keep native CRS).  Defaults to EPSG:3857.
    inclination / declination / amplitude_nT : Stored in attrs for magnetic data.
    nodata_override : Use this nodata value instead of the file's metadata.
    resampling      : Rasterio resampling algorithm for reprojection.
    name            : Variable name in the output NetCDF4.
    """
    print(f"[geotiff_to_netcdf4] {tiff_path}")

    da = xr.open_dataarray(tiff_path, engine="rasterio")

    # Select requested band and drop degenerate dims
    if "band" in da.dims:
        da = da.sel(band=band, drop=True)
    da = da.squeeze(drop=True)

    # ── Nodata → NaN ─────────────────────────────────────────────────
    nd = nodata_override if nodata_override is not None else da.rio.nodata
    if nd is not None and np.isfinite(float(nd)):
        da = da.where(da != float(nd))
    if da.rio.encoded_nodata is not None:
        da = da.where(da != da.rio.encoded_nodata)

    # ── GeoTIFF scale / offset ───────────────────────────────────────
    sf = da.attrs.pop("scale_factor", None) or da.attrs.pop("Scale", None)
    ao = da.attrs.pop("add_offset",   None) or da.attrs.pop("Offset", None)
    if sf is not None:
        da = da * float(sf)
    if ao is not None:
        da = da + float(ao)

    # ── Unit conversion ──────────────────────────────────────────────
    scale = _unit_factor(unit_in, data_type)
    if scale != 1.0:
        da = da * scale

    # ── Georeferencing validation ────────────────────────────────────
    if da.rio.crs is None:
        raise ValueError(
            "GeoTIFF has no CRS metadata.  "
            "Set the CRS with a tool such as QGIS or gdal_edit.py before converting."
        )
    print(f"  Native CRS: {da.rio.crs}")

    # ── Reproject if requested ───────────────────────────────────────
    if target_crs is not None and str(da.rio.crs) != target_crs:
        _rmap = {"nearest": _Resampling.nearest, "bilinear": _Resampling.bilinear, "cubic": _Resampling.cubic}
        da = da.rio.reproject(target_crs, resampling=_rmap.get(resampling, _Resampling.bilinear))
        print(f"  Reprojected → {target_crs}")

    # ── Ensure dims are named x / y ──────────────────────────────────
    rename_map = {}
    for d in list(da.dims):
        dl = d.lower()
        if   dl in ("longitude", "lon", "long"): rename_map[d] = "x"
        elif dl in ("latitude",  "lat"):          rename_map[d] = "y"
    if rename_map:
        da = da.rename(rename_map)

    da = da.rio.write_crs(da.rio.crs)

    # ── Attributes ───────────────────────────────────────────────────
    da.attrs.update({"units": _DATA_UNITS.get(data_type, ""), "data_type": data_type})
    if data_type == "magnetic":
        if inclination  is not None: da.attrs["inclination"]  = float(inclination)
        if declination  is not None: da.attrs["declination"]  = float(declination)
        if amplitude_nT is not None: da.attrs["amplitude_nT"] = float(amplitude_nT)

    if name:
        da.name = name
    print(f"  Shape: {da.shape}")
    _write_dataarray(da, out_path)
    _qc_netcdf4(out_path)
    return da


# ─────────────────────────────────────────────────────────────────────
# 3.  NetCDF / CF  →  NetCDF4  (metadata harmonisation)
# ─────────────────────────────────────────────────────────────────────

def netcdf_cf_to_netcdf4(
    nc_path: str,
    out_path: str,
    *,
    variable: Optional[str] = None,
    data_type: str = "gravity",
    target_crs: Optional[str] = "EPSG:3857",
    inclination: Optional[float] = None,
    declination: Optional[float] = None,
    amplitude_nT: Optional[float] = None,
    name: Optional[str] = None,
) -> xr.DataArray:
    """Harmonise an existing NetCDF / CF-compliant file to the pipeline format.

    Handles:
    * Auto-selection of the first 2-D data variable.
    * Spatial dimension renaming (lon/lat/longitude/latitude/easting/northing → x/y).
    * CRS detection from ``grid_mapping``, ``spatial_ref``, or coordinate range.
    * Reprojection from geographic coordinates to projected metres when needed.
    * Attribute standardisation (units, data_type, magnetic metadata).

    The output loads directly with::

        xr.open_dataarray(out_path, decode_coords="all").squeeze()
    """
    print(f"[netcdf_cf_to_netcdf4] {nc_path}")

    ds = xr.open_dataset(nc_path, decode_coords="all", mask_and_scale=True)

    # ── Select variable ──────────────────────────────────────────────
    if variable is None:
        candidates = [v for v in ds.data_vars if ds[v].squeeze().ndim == 2]
        if not candidates:
            raise ValueError(
                f"No 2-D variable found in {nc_path}.  "
                f"Available: {list(ds.data_vars)}"
            )
        variable = candidates[0]
        print(f"  Auto-selected variable: '{variable}'")

    da = ds[variable].squeeze(drop=True)

    # ── Rename spatial dims → x / y ──────────────────────────────────
    rename_map: dict = {}
    for d in list(da.dims):
        dl = d.lower()
        if   dl in ("longitude", "lon", "long", "easting",  "east", "col"): rename_map[d] = "x"
        elif dl in ("latitude",  "lat",          "northing", "north", "row"): rename_map[d] = "y"
    if rename_map:
        da = da.rename(rename_map)
        print(f"  Renamed dims: {rename_map}")

    # ── Detect CRS ───────────────────────────────────────────────────
    src_crs: Optional[str] = None
    if da.rio.crs is not None:
        epsg = da.rio.crs.to_epsg()
        src_crs = f"EPSG:{epsg}" if epsg else str(da.rio.crs)
    # Fallback: grid_mapping attribute → inspect coordinate variable
    if src_crs is None and "grid_mapping" in da.attrs:
        gm_name = da.attrs["grid_mapping"]
        if gm_name in ds.coords:
            gm_attrs = ds.coords[gm_name].attrs
            wkt = gm_attrs.get("crs_wkt") or gm_attrs.get("spatial_ref")
            if wkt:
                src_crs = wkt
    # Fallback: infer from coordinate range
    if src_crs is None and "x" in da.dims:
        src_crs = "EPSG:4326" if float(da.x.max() - da.x.min()) < 360 else "EPSG:32613"
        warnings.warn(
            f"CRS not detected; assuming {src_crs}.  "
            "Pass target_crs explicitly if this is incorrect."
        )
    src_crs = src_crs or "EPSG:4326"
    print(f"  Detected CRS: {src_crs}")

    # ── Reproject geographic → projected metres ──────────────────────
    is_geographic = "4326" in str(src_crs) or "WGS84" in str(src_crs).upper()
    if is_geographic:
        print(f"  Reprojecting {src_crs} → {target_crs}")
        da = da.rio.write_crs(src_crs).rio.reproject(target_crs)
        src_crs = target_crs
    elif target_crs and target_crs != src_crs:
        da = da.rio.write_crs(src_crs).rio.reproject(target_crs)
        src_crs = target_crs

    da = da.rio.write_crs(src_crs)

    # ── Attributes ───────────────────────────────────────────────────
    da.attrs.setdefault("units", _DATA_UNITS.get(data_type, ""))
    da.attrs["data_type"] = data_type
    if data_type == "magnetic":
        if inclination  is not None: da.attrs["inclination"]  = float(inclination)
        if declination  is not None: da.attrs["declination"]  = float(declination)
        if amplitude_nT is not None: da.attrs["amplitude_nT"] = float(amplitude_nT)

    if name:
        da.name = name
    print(f"  Shape: {da.shape}")
    _write_dataarray(da, out_path)
    _qc_netcdf4(out_path)
    return da


# ─────────────────────────────────────────────────────────────────────
# 4.  SEG-Y  →  HDF5
# ─────────────────────────────────────────────────────────────────────

def segy_to_hdf5(
    segy_path: str,
    out_path: str,
    *,
    mode: str = "2d_fwi",
    line_name: Optional[str] = None,
    input_crs: str = "EPSG:32613",
    is_depth: bool = True,
    vp_avg: float = 2000.0,
    amp_scale: float = 1.0,
    amp_offset: float = 0.0,
    horizons: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    """Convert a SEG-Y file to HDF5 in the structure expected by the notebook.

    Three modes, selected by *mode*:

    ``"2d_fwi"``
        One 2-D seismic line.  Writes::

            survey_lines/{line_name}/
                coordinates/latitude   (N_traces,) float64
                coordinates/longitude  (N_traces,) float64
                coordinates/z_m        (N_samples,) float64  depth in metres
                velocity_model/vp_true (N_traces, N_samples) float64  m/s

        Trace values are taken as velocity [m/s].  When the SEG-Y contains
        amplitudes, set *amp_scale* / *amp_offset* to map them to m/s.

    ``"3d_reflection"``
        Post-stack 3-D volume.  Writes::

            density_model/density_3d  (N_lons, N_lats, N_depths) float32  kg/m³
            coordinates/longitude     (N_lons,) float64
            coordinates/latitude      (N_lats,) float64
            coordinates/depth_m       (N_depths,) float64

        Amplitudes are converted via ``density_kg_m3 = amplitude * amp_scale + amp_offset``.
        The notebook divides by 1 000 internally (kg/m³ → g/cc), so store in kg/m³.

    ``"validation_3d"``
        Same 3-D volume but written as a velocity model.  Writes::

            velocity_model/vp_3d  (N_x, N_y, N_depths) float32  m/s
            coordinates/depth_m   (N_depths,) float64
            horizons/{name}       (N_x, N_y) float32  depth in metres (optional)

    Parameters
    ----------
    segy_path      : Input SEG-Y file.
    out_path       : Output HDF5 file.
    mode           : Conversion mode (see above).
    line_name      : HDF5 group name for 2d_fwi (default: stem of *segy_path*).
    input_crs      : CRS of X/Y coordinates in trace headers.
    is_depth       : True → samples are already in depth domain.
                     False → time domain; converted using *vp_avg* via TWT formula.
    vp_avg         : Average velocity [m/s] for time→depth conversion.
    amp_scale / amp_offset : Linear transform applied to trace amplitudes before writing.
    horizons       : Optional dict ``{name: (N_x, N_y) ndarray}`` for validation_3d.
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    if mode not in ("2d_fwi", "3d_reflection", "validation_3d"):
        raise ValueError(f"Unknown mode '{mode}'.  Choose: 2d_fwi, 3d_reflection, validation_3d")

    print(f"[segy_to_hdf5] {segy_path}  mode={mode}")

    with segyio.open(segy_path, ignore_geometry=True, strict=False) as f:
        n_traces = f.tracecount
        n_samples = f.samples.size

        # segyio: f.samples is in milliseconds (= sample_index * dt_μs / 1000)
        # Depth / time axis → metres
        samples_ms = f.samples.astype(np.float64)
        if is_depth:
            # Depth-domain file: treat f.samples directly as depth in metres
            z_m = samples_ms
        else:
            # Time-domain file: TWT (ms) → depth (m) using half the average velocity
            z_m = samples_ms * 1e-3 * vp_avg / 2.0

        # Read trace header coordinates
        TF = segyio.TraceField
        scalars = np.array(
            [f.header[i][TF.SourceGroupScalar] for i in range(n_traces)], dtype=np.float64
        )
        raw_x = np.array([f.header[i][TF.SourceX] for i in range(n_traces)], dtype=np.float64)
        raw_y = np.array([f.header[i][TF.SourceY] for i in range(n_traces)], dtype=np.float64)
        scalars[scalars == 0] = 1
        coord_x = _apply_segy_coord_scalar(raw_x, scalars)
        coord_y = _apply_segy_coord_scalar(raw_y, scalars)

        # Reproject header coordinates to EPSG:4326 (lat/lon) for HDF5
        try:
            lon, lat = _reproject(coord_x, coord_y, input_crs, "EPSG:4326")
        except Exception:
            # If reprojection fails (e.g. CRS already geographic), use as-is
            lon, lat = coord_x, coord_y

        # Read all trace data
        traces = f.trace.raw[:].astype(np.float64)     # (N_traces, N_samples)
        traces = traces * amp_scale + amp_offset

        # ── Write HDF5 ────────────────────────────────────────────────
        with h5py.File(out_path, "w") as hf:

            if mode == "2d_fwi":
                lname = line_name or Path(segy_path).stem
                grp = hf.require_group(f"survey_lines/{lname}")
                coord_grp = grp.require_group("coordinates")
                coord_grp.create_dataset("latitude",  data=lat.astype(np.float64))
                coord_grp.create_dataset("longitude", data=lon.astype(np.float64))
                coord_grp.create_dataset("z_m",       data=z_m)
                vm_grp = grp.require_group("velocity_model")
                vm_grp.create_dataset(
                    "vp_true", data=traces.astype(np.float64),
                    compression="gzip", compression_opts=4,
                )

            elif mode in ("3d_reflection", "validation_3d"):
                # Organise traces into a 3-D volume via inline / crossline
                inlines = np.array(
                    [f.header[i][TF.INLINE_3D] for i in range(n_traces)]
                )
                xlines = np.array(
                    [f.header[i][TF.CROSSLINE_3D] for i in range(n_traces)]
                )
                unique_il = np.unique(inlines)
                unique_xl = np.unique(xlines)
                n_il, n_xl = len(unique_il), len(unique_xl)

                il_idx = {v: i for i, v in enumerate(unique_il)}
                xl_idx = {v: i for i, v in enumerate(unique_xl)}

                vol = np.full((n_il, n_xl, n_samples), np.nan, dtype=np.float32)
                for t in range(n_traces):
                    ii = il_idx[inlines[t]]
                    xi = xl_idx[xlines[t]]
                    vol[ii, xi, :] = traces[t].astype(np.float32)

                # Mean lon/lat per inline and crossline
                lon_axis = np.array([
                    float(np.nanmean(lon[inlines == il])) for il in unique_il
                ])
                lat_axis = np.array([
                    float(np.nanmean(lat[xlines == xl])) for xl in unique_xl
                ])

                hf.create_dataset("coordinates/depth_m",  data=z_m)
                hf.create_dataset("coordinates/longitude", data=lon_axis)
                hf.create_dataset("coordinates/latitude",  data=lat_axis)

                if mode == "3d_reflection":
                    # Store as density proxy in kg/m³
                    hf.create_dataset(
                        "density_model/density_3d",
                        data=vol,
                        compression="gzip", compression_opts=4,
                    )
                else:  # validation_3d
                    hf.create_dataset(
                        "velocity_model/vp_3d",
                        data=vol,
                        compression="gzip", compression_opts=4,
                    )
                    if horizons:
                        hz_grp = hf.require_group("horizons")
                        for hz_name, hz_data in horizons.items():
                            hz_grp.create_dataset(
                                hz_name,
                                data=np.asarray(hz_data, dtype=np.float32),
                            )

    print(f"  → {out_path}  mode={mode}  traces={n_traces}  samples={n_samples}")
    _qc_hdf5(out_path, mode=mode, line_name=line_name)


# ─────────────────────────────────────────────────────────────────────
# 5.  SEG-D  →  HDF5
# ─────────────────────────────────────────────────────────────────────

# SEG-D format codes and their data types
_SEGD_FORMAT_DTYPE: Dict[int, str] = {
    8036: "float32",   # 32-bit IEEE float, demultiplexed
    8038: "float32",   # 32-bit IEEE float, multiplexed
    8042: "float32",
    8044: "int32",
    8048: "int32",
    8058: "float32",   # 32-bit IEEE float, demultiplexed (most common modern)
    8015: "int24",     # 24-bit integer
    8022: "int24",
}


def _decode_bcd_nibbles(data: bytes, byte_offset: int, n_nibbles: int) -> int:
    """Decode *n_nibbles* BCD nibbles starting at *byte_offset*."""
    result = 0
    for i in range(n_nibbles):
        byte = data[byte_offset + i // 2]
        nibble = (byte >> 4) & 0x0F if i % 2 == 0 else byte & 0x0F
        result = result * 10 + nibble
    return result


def _read_int24_be(data: bytes, offset: int) -> int:
    """Read a 24-bit big-endian signed integer at *offset*."""
    b0, b1, b2 = data[offset], data[offset + 1], data[offset + 2]
    val = (b0 << 16) | (b1 << 8) | b2
    if val >= 0x800000:
        val -= 0x1000000
    return val


class _SegDReader:
    """Minimal SEG-D Rev 1/2 reader.

    Supports format codes 8036, 8038, 8058 (32-bit IEEE float) and 8015,
    8022 (24-bit integer).  Only the demultiplexed layout (format 8058) is
    parsed by default; multiplexed formats are demultiplexed automatically.

    General Header Block 1 offsets used (all 1-based nibble indices per spec):
    * Bytes  3-5  (nibbles  7-12): format code (6-digit BCD)
    * Byte   6    (nibbles 13-14): year (2-digit BCD)
    * Byte  18    (nibble  35):    channel-sets per scan (high nibble)
    * Byte  23    (nibble  46 low): extended header block count
    * Bytes 20-21 (nibbles 39-42): record length (BCD units of 512 ms)
    """

    GH1_SIZE = 32   # General Header Block 1 / Extended header block size (bytes)
    CS_SIZE  = 32   # Channel Set Descriptor size (bytes)
    TH_SIZE  = 20   # Trace Header size (bytes)

    def __init__(self, path: str) -> None:
        self.path = path
        with open(path, "rb") as fh:
            self.raw = fh.read()
        self._parse_general_header()

    # ── Header parsing ────────────────────────────────────────────────
    def _parse_general_header(self) -> None:
        d = self.raw
        fmt_str = (
            f"{_decode_bcd_nibbles(d, 3, 2):02d}"
            f"{_decode_bcd_nibbles(d, 4, 2):02d}"
            f"{_decode_bcd_nibbles(d, 5, 2):02d}"
        )
        self.format_code = int(fmt_str)
        if self.format_code not in _SEGD_FORMAT_DTYPE:
            raise ValueError(
                f"Unsupported SEG-D format code {self.format_code}.  "
                f"Supported: {sorted(_SEGD_FORMAT_DTYPE)}"
            )
        self.dtype_name = _SEGD_FORMAT_DTYPE[self.format_code]

        # Number of channel sets (high nibble of byte 18)
        self.n_chan_sets = (d[18] >> 4) & 0x0F
        if self.n_chan_sets == 0:
            self.n_chan_sets = 1

        # Extended header blocks (low nibble of byte 23)
        n_ext = d[23] & 0x0F

        # Base header size: GH1 + extended + channel set descriptors
        self.base_header_size = (
            self.GH1_SIZE * (1 + n_ext)
            + self.n_chan_sets * self.CS_SIZE
        )

        # Parse channel set descriptors → samples per channel set
        self.chan_sets: List[dict] = []
        for i in range(self.n_chan_sets):
            cs_off = self.GH1_SIZE * (1 + n_ext) + i * self.CS_SIZE
            n_samp = (d[cs_off + 7] << 8) | d[cs_off + 8]
            n_chan = (d[cs_off + 9] << 8) | d[cs_off + 10]
            self.chan_sets.append({"n_samples": n_samp, "n_channels": n_chan})

        # Total channels and samples for the primary channel set
        cs0 = self.chan_sets[0]
        self.n_samples = cs0["n_samples"] or 1000
        self.n_channels = sum(cs["n_channels"] for cs in self.chan_sets)
        if self.n_channels == 0:
            self.n_channels = 1

        # Bytes per trace sample
        self._bytes_per_sample = 4 if self.dtype_name in ("float32", "int32") else 3

        # Record length in bytes (approximate for demux layout)
        self._record_size = (
            self.base_header_size
            + self.n_channels * (self.TH_SIZE + self.n_samples * self._bytes_per_sample)
        )

    # ── Record iteration ──────────────────────────────────────────────
    def _decode_traces_from_record(
        self, rec_start: int, n_channels: int, n_samples: int
    ) -> np.ndarray:
        """Decode demultiplexed traces from one SEG-D record."""
        traces = np.zeros((n_channels, n_samples), dtype=np.float32)
        offset = rec_start + self.base_header_size
        for ch in range(n_channels):
            offset += self.TH_SIZE   # skip trace header
            if self.dtype_name == "float32":
                chunk = self.raw[offset: offset + n_samples * 4]
                arr = np.frombuffer(chunk, dtype=">f4").astype(np.float32)
            elif self.dtype_name == "int32":
                chunk = self.raw[offset: offset + n_samples * 4]
                arr = np.frombuffer(chunk, dtype=">i4").astype(np.float32)
            else:  # int24
                arr = np.array(
                    [_read_int24_be(self.raw, offset + j * 3) for j in range(n_samples)],
                    dtype=np.float32,
                )
            traces[ch, : len(arr)] = arr
            offset += n_samples * self._bytes_per_sample
        return traces

    def iter_records(self) -> Iterator[Tuple[int, np.ndarray]]:
        """Yield ``(record_index, traces)`` where ``traces`` is (N_ch, N_samp)."""
        pos = 0
        rec_idx = 0
        file_size = len(self.raw)
        while pos + self.base_header_size < file_size:
            n_ch   = self.n_channels
            n_samp = self.n_samples
            try:
                traces = self._decode_traces_from_record(pos, n_ch, n_samp)
            except Exception:
                break
            yield rec_idx, traces
            rec_idx += 1
            pos += self.base_header_size + n_ch * (self.TH_SIZE + n_samp * self._bytes_per_sample)


def segd_to_hdf5(
    segd_path: str,
    out_path: str,
    *,
    line_name: Optional[str] = None,
    sample_rate_hz: Optional[float] = None,
    is_depth: bool = False,
    vp_avg: float = 2000.0,
    depth_per_sample: float = 1.0,
    trace_coords: Optional[Union[str, Dict[int, Tuple[float, float]]]] = None,
) -> None:
    """Convert a SEG-D Rev 1/2 field recording to HDF5 in the 2D FWI layout.

    Output structure (matches ``load_2d_fwi_prior`` in the notebook)::

        survey_lines/{line_name}/
            coordinates/latitude   (N_traces,) float64
            coordinates/longitude  (N_traces,) float64
            coordinates/z_m        (N_samples,) float64  depth in metres
            velocity_model/vp_true (N_traces, N_samples) float64

    .. note::
       SEG-D field recordings rarely embed absolute trace lat/lon in the
       standard headers.  Pass coordinates via *trace_coords* — either a CSV
       file path (columns: ``record``, ``trace``, ``latitude``, ``longitude``)
       or a dict ``{trace_global_index: (lat, lon)}``.  Without it, sequential
       dummy coordinates are generated so the file structure is still valid.

    Parameters
    ----------
    segd_path        : Input SEG-D file.
    out_path         : Output HDF5 file.
    line_name        : HDF5 group name (default: file stem).
    sample_rate_hz   : Recording sample rate [Hz].  Computed from SEG-D header
                       if None; defaults to 500 Hz if header is unreadable.
    is_depth         : True → samples already in depth [m].  False → time [s].
    vp_avg           : Average velocity [m/s] for time→depth (``d = vp * t / 2``).
    depth_per_sample : When *is_depth* is True, metres per sample index.
    trace_coords     : Explicit trace coordinates — CSV path or dict.
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    print(f"[segd_to_hdf5] {segd_path}")

    reader = _SegDReader(segd_path)
    print(f"  Format: {reader.format_code}  channels: {reader.n_channels}  samples: {reader.n_samples}")

    # ── Sample-rate / depth axis ──────────────────────────────────────
    if is_depth:
        z_m = np.arange(reader.n_samples, dtype=np.float64) * depth_per_sample
    else:
        sr = sample_rate_hz or 500.0
        t_s = np.arange(reader.n_samples, dtype=np.float64) / sr
        z_m = t_s * vp_avg / 2.0

    # ── Coordinate loading ───────────────────────────────────────────
    coord_dict: Dict[int, Tuple[float, float]] = {}
    if isinstance(trace_coords, str):
        tc_df = pd.read_csv(trace_coords)
        idx = 0
        for _, row in tc_df.iterrows():
            lat_val = float(row.get("latitude", row.get("lat", 0.0)))
            lon_val = float(row.get("longitude", row.get("lon", 0.0)))
            coord_dict[idx] = (lat_val, lon_val)
            idx += 1
    elif isinstance(trace_coords, dict):
        coord_dict = trace_coords

    # ── Collect all records ───────────────────────────────────────────
    all_traces: List[np.ndarray] = []
    all_lats:   List[float] = []
    all_lons:   List[float] = []

    global_trace = 0
    for _, traces in reader.iter_records():
        n_ch = traces.shape[0]
        for ch in range(n_ch):
            all_traces.append(traces[ch].astype(np.float64))
            if global_trace in coord_dict:
                la, lo = coord_dict[global_trace]
            else:
                # Dummy sequential coordinate
                la = float(global_trace) * 0.0001
                lo = float(global_trace) * 0.0001
            all_lats.append(la)
            all_lons.append(lo)
            global_trace += 1

    if not all_traces:
        raise RuntimeError(f"No traces could be decoded from {segd_path}")

    lname = line_name or Path(segd_path).stem
    traces_arr = np.stack(all_traces, axis=0)    # (N_total_traces, N_samples)
    lat_arr    = np.array(all_lats,  dtype=np.float64)
    lon_arr    = np.array(all_lons,  dtype=np.float64)

    with h5py.File(out_path, "w") as hf:
        grp = hf.require_group(f"survey_lines/{lname}")
        cg  = grp.require_group("coordinates")
        cg.create_dataset("latitude",  data=lat_arr)
        cg.create_dataset("longitude", data=lon_arr)
        cg.create_dataset("z_m",       data=z_m)
        vm  = grp.require_group("velocity_model")
        vm.create_dataset(
            "vp_true", data=traces_arr,
            compression="gzip", compression_opts=4,
        )

    print(f"  → {out_path}  traces={len(all_traces)}  samples={reader.n_samples}")
    _qc_hdf5(out_path, mode="2d_fwi", line_name=lname)


# ─────────────────────────────────────────────────────────────────────
# Shared well-log utilities
# ─────────────────────────────────────────────────────────────────────

def _build_mnemonic_lookup(
    custom: Optional[Dict[str, List[str]]] = None
) -> Dict[str, str]:
    """Build a case-insensitive {input_alias_upper → standard_mnemonic} dict."""
    base = dict(_DEFAULT_MNEMONIC_MAP)
    if custom:
        for std, aliases in custom.items():
            base.setdefault(std, [])
            base[std] = list(dict.fromkeys(base[std] + aliases))
    # Invert: alias_upper → standard
    lookup: Dict[str, str] = {}
    for std, aliases in base.items():
        for alias in aliases:
            lookup[alias.upper()] = std
    return lookup


def _las_depth_to_ft(las: lasio.LASFile) -> lasio.LASFile:
    """Return *las* with its depth index converted to feet if currently in metres."""
    try:
        depth_unit = (las.well.STRT.unit or "").upper()
    except AttributeError:
        depth_unit = ""
    is_metres = depth_unit in ("M", "METER", "METRE", "METERS", "METRES", "")

    if not is_metres:
        return las   # already in feet or unrecognised — leave unchanged

    idx_curve = las.curves[0]
    idx_curve.data = idx_curve.data * M_TO_FT
    idx_curve.unit = "FT"
    las.well.STRT.value = float(idx_curve.data.min())
    las.well.STOP.value = float(idx_curve.data.max())
    return las


def _normalise_las_mnemonics(
    las: lasio.LASFile,
    lookup: Dict[str, str],
) -> lasio.LASFile:
    """Rename curves in *las* using *lookup* (alias_upper → standard).

    Duplicate standard mnemonics are deduplicated by keeping the first
    occurrence (highest-priority alias listed first in _DEFAULT_MNEMONIC_MAP).
    Uses in-place slice replacement to satisfy lasio's SectionItems list type.
    """
    seen_standards: set = set()
    keep: List = [las.curves[0]]   # always keep depth index
    seen_standards.add(las.curves[0].mnemonic.upper())

    for curve in las.curves[1:]:
        std = lookup.get(curve.mnemonic.upper())
        if std and std not in seen_standards:
            curve.mnemonic = std
            keep.append(curve)
            seen_standards.add(std)
        elif std is None:
            keep.append(curve)   # pass-through non-aliased curves unchanged

    # lasio's CurveList is a SectionItems (list subclass); use slice assignment
    las.curves[:] = keep
    return las


def _extract_well_location_from_las(las: lasio.LASFile) -> Tuple[Optional[float], Optional[float]]:
    """Return (latitude, longitude) from LAS WELL section, or (None, None)."""
    lat = lon = None
    well_dict = {k.upper(): v for k, v in {item.mnemonic: item.value
                                             for item in las.well}.items()}
    for key in ("SLAT", "LAT", "LATITUDE", "Y_LOC", "WELLLATITUDE"):
        if key in well_dict:
            try:
                lat = float(well_dict[key])
                break
            except (ValueError, TypeError):
                pass
    for key in ("SLON", "LON", "LONG", "LONGITUDE", "X_LOC", "WELLLONGITUDE"):
        if key in well_dict:
            try:
                lon = float(well_dict[key])
                break
            except (ValueError, TypeError):
                pass
    return lat, lon


def _update_well_locations_json(
    json_path: str,
    new_entries: List[Dict],
) -> None:
    """Upsert *new_entries* into *well_locations_summary.json*.

    Each entry must have keys ``file``, ``latitude``, ``longitude``.
    Existing entries with the same ``file`` key are replaced.
    """
    existing: List[Dict] = []
    if os.path.exists(json_path):
        with open(json_path) as fh:
            existing = json.load(fh).get("wells", [])

    by_file = {e["file"]: e for e in existing}
    for entry in new_entries:
        by_file[entry["file"]] = entry

    with open(json_path, "w") as fh:
        json.dump({"wells": list(by_file.values())}, fh, indent=2)


# ─────────────────────────────────────────────────────────────────────
# 6.  LAS  →  normalised LAS + well_locations_summary.json
# ─────────────────────────────────────────────────────────────────────

def las_to_json(
    las_files: Union[str, List[str]],
    out_dir: str,
    json_path: str,
    *,
    mnemonic_map: Optional[Dict[str, List[str]]] = None,
    locations: Optional[Dict[str, Tuple[float, float]]] = None,
    location_csv: Optional[str] = None,
    force_depth_unit: str = "FT",
) -> Dict:
    """Normalise raw LAS files and generate ``well_locations_summary.json``.

    For each input LAS file:
    1. Reads the file with ``lasio``.
    2. Aliases non-standard curve mnemonics to the standard names expected by
       ``load_all_well_constraints`` (RHOB, NPHI, GR, MAGSUS, …).
    3. Converts the depth index to feet if it is currently in metres.
    4. Writes a normalised ``.las`` file to *out_dir* (same filename as input).
    5. Extracts or accepts explicit lat/lon and records them in *json_path*.

    The *out_dir* directory and the *json_path* file are directly usable as
    the ``las_dir`` and ``json_path`` arguments of ``load_all_well_constraints``
    in the notebook.

    Parameters
    ----------
    las_files       : A single LAS file path, a list of paths, or a directory.
    out_dir         : Directory to write normalised LAS files.
    json_path       : Path to write / update ``well_locations_summary.json``.
    mnemonic_map    : Extra aliases ``{standard_mnemonic: [aliases]}`` to merge
                      with the built-in map.
    locations       : Explicit ``{filename: (latitude, longitude)}`` overrides.
    location_csv    : CSV file with columns ``file``, ``latitude``, ``longitude``
                      as an alternative to *locations*.
    force_depth_unit: Output depth unit — "FT" (default) or "M".

    Returns
    -------
    dict : The complete contents of the written ``well_locations_summary.json``.
    """
    os.makedirs(out_dir, exist_ok=True)
    lookup = _build_mnemonic_lookup(mnemonic_map)

    # ── Collect LAS paths ────────────────────────────────────────────
    if isinstance(las_files, str):
        p = Path(las_files)
        paths = sorted(p.glob("*.las")) if p.is_dir() else [p]
    else:
        paths = [Path(f) for f in las_files]

    # ── External location table ──────────────────────────────────────
    ext_locs: Dict[str, Tuple[float, float]] = {}
    if location_csv:
        ldf = pd.read_csv(location_csv)
        for _, row in ldf.iterrows():
            ext_locs[str(row["file"])] = (float(row["latitude"]), float(row["longitude"]))
    if locations:
        ext_locs.update(locations)

    new_entries: List[Dict] = []
    skipped = 0

    for las_path in paths:
        fname = las_path.name
        print(f"[las_to_json] {fname}")
        try:
            las = lasio.read(str(las_path))
        except Exception as exc:
            warnings.warn(f"  Could not read {fname}: {exc}")
            skipped += 1
            continue

        # ── Mnemonic aliasing ─────────────────────────────────────────
        las = _normalise_las_mnemonics(las, lookup)

        # ── Depth unit → FT ──────────────────────────────────────────
        if force_depth_unit.upper() == "FT":
            las = _las_depth_to_ft(las)

        # ── Write normalised LAS ──────────────────────────────────────
        out_las = os.path.join(out_dir, fname)
        las.write(out_las, version=2.0)
        print(f"  → {out_las}")

        # ── Well location ─────────────────────────────────────────────
        if fname in ext_locs:
            lat, lon = ext_locs[fname]
        else:
            lat, lon = _extract_well_location_from_las(las)

        if lat is None or lon is None:
            warnings.warn(
                f"  No location found for {fname}.  "
                "Provide coordinates via 'locations' or 'location_csv'."
            )
            skipped += 1
            continue

        new_entries.append({"file": fname, "latitude": float(lat), "longitude": float(lon)})

    _update_well_locations_json(json_path, new_entries)
    print(f"  JSON → {json_path}  ({len(new_entries)} wells, {skipped} skipped)")
    _qc_las_json(json_path, out_dir)

    with open(json_path) as fh:
        return json.load(fh)


# ─────────────────────────────────────────────────────────────────────
# 7.  DLIS  →  normalised LAS + well_locations_summary.json
# ─────────────────────────────────────────────────────────────────────

def dlis_to_json(
    dlis_files: Union[str, List[str]],
    out_dir: str,
    json_path: str,
    *,
    mnemonic_map: Optional[Dict[str, List[str]]] = None,
    locations: Optional[Dict[str, Tuple[float, float]]] = None,
    location_csv: Optional[str] = None,
    frame_name: Optional[str] = None,
    force_depth_unit: str = "FT",
) -> Dict:
    """Convert DLIS files to normalised LAS + ``well_locations_summary.json``.

    For each DLIS file:
    1. Opens the file with ``dlisio``.
    2. Selects the depth frame (by *frame_name* or auto-detects the frame
       with a DEPT / DEPTH / MD channel).
    3. Aliases channel mnemonics to the standard names expected by the notebook.
    4. Converts depth to feet if needed.
    5. Writes a ``{dlis_stem}.las`` file to *out_dir*.
    6. Extracts lat/lon from DLIS Origin/Parameter objects or *locations*.

    The output directory and JSON are compatible with ``load_all_well_constraints``.

    Parameters
    ----------
    dlis_files      : DLIS file path, list of paths, or directory.
    out_dir         : Output directory for normalised LAS files.
    json_path       : Path to write / update ``well_locations_summary.json``.
    mnemonic_map    : Extra mnemonic aliases.
    locations       : Explicit ``{dlis_filename: (lat, lon)}`` overrides.
    location_csv    : CSV with columns ``file``, ``latitude``, ``longitude``.
    frame_name      : Select a specific DLIS frame by name; auto if None.
    force_depth_unit: Output depth unit ("FT" or "M").

    Returns
    -------
    dict : Contents of the written ``well_locations_summary.json``.
    """
    os.makedirs(out_dir, exist_ok=True)
    lookup = _build_mnemonic_lookup(mnemonic_map)

    # ── Collect DLIS paths ───────────────────────────────────────────
    if isinstance(dlis_files, str):
        p = Path(dlis_files)
        exts = ("*.dlis", "*.DLIS")
        paths = sorted(f for ext in exts for f in p.glob(ext)) if p.is_dir() else [p]
    else:
        paths = [Path(f) for f in dlis_files]

    # ── External location table ──────────────────────────────────────
    ext_locs: Dict[str, Tuple[float, float]] = {}
    if location_csv:
        ldf = pd.read_csv(location_csv)
        for _, row in ldf.iterrows():
            ext_locs[str(row["file"])] = (float(row["latitude"]), float(row["longitude"]))
    if locations:
        ext_locs.update(locations)

    new_entries: List[Dict] = []
    skipped = 0

    for dlis_path in paths:
        fname = dlis_path.name
        las_fname = dlis_path.stem + ".las"
        print(f"[dlis_to_json] {fname}")

        try:
            with dlisio.load(str(dlis_path)) as logical_files:
                # dlisio.load returns (main_lf, *extras)
                if isinstance(logical_files, tuple):
                    lf_list = list(logical_files)
                else:
                    lf_list = [logical_files]

                for lf in lf_list:
                    # ── Frame selection ──────────────────────────────
                    target_frame = None
                    if frame_name:
                        for fr in lf.frames:
                            if fr.name == frame_name:
                                target_frame = fr
                                break
                    if target_frame is None:
                        # Auto: pick frame containing a depth-like channel
                        for fr in lf.frames:
                            ch_names = {ch.name.upper() for ch in fr.channels}
                            if ch_names & {"DEPT", "DEPTH", "MD", "TVD"}:
                                target_frame = fr
                                break
                    if target_frame is None and lf.frames:
                        target_frame = lf.frames[0]
                    if target_frame is None:
                        warnings.warn(f"  No frame found in {fname}")
                        skipped += 1
                        continue

                    channels = {ch.name.upper(): ch for ch in target_frame.channels}

                    # ── Depth channel ────────────────────────────────
                    depth_ch = None
                    for dn in ("DEPT", "DEPTH", "MD", "TVD", "DEPTM"):
                        if dn in channels:
                            depth_ch = channels[dn]
                            break
                    if depth_ch is None:
                        # Fallback: use first channel as depth
                        depth_ch = target_frame.channels[0]

                    depths = depth_ch.curves().ravel().astype(np.float64)
                    depth_unit = (depth_ch.units or "").strip().upper()

                    # ── Curve extraction ─────────────────────────────
                    las_curves: Dict[str, Tuple[np.ndarray, str]] = {}
                    for ch_name_upper, ch in channels.items():
                        if ch is depth_ch:
                            continue
                        try:
                            data = ch.curves().ravel().astype(np.float64)
                            unit = (ch.units or "").strip()
                            std_name = lookup.get(ch_name_upper, ch_name_upper)
                            las_curves[std_name] = (data, unit)
                        except Exception:
                            pass

                    # ── Build lasio LASFile ───────────────────────────
                    las = lasio.LASFile()

                    # Depth step (handle irregular sampling gracefully)
                    step = float(np.nanmedian(np.diff(depths))) if len(depths) > 1 else 1.0
                    depth_unit_las = "FT" if force_depth_unit.upper() == "FT" else "M"
                    if depth_unit in ("M", "METER", "METRE") and force_depth_unit.upper() == "FT":
                        depths = depths * M_TO_FT
                        step   = step   * M_TO_FT
                        depth_unit_las = "FT"
                    elif depth_unit == "FT" and force_depth_unit.upper() == "M":
                        depths = depths * FT_TO_M
                        step   = step   * FT_TO_M
                        depth_unit_las = "M"

                    las.well.STRT = lasio.HeaderItem("STRT", unit=depth_unit_las, value=float(depths.min()))
                    las.well.STOP = lasio.HeaderItem("STOP", unit=depth_unit_las, value=float(depths.max()))
                    las.well.STEP = lasio.HeaderItem("STEP", unit=depth_unit_las, value=step)
                    las.well.NULL = lasio.HeaderItem("NULL", value=-9999.25)

                    # Depth curve
                    las.append_curve("DEPT", depths, unit=depth_unit_las, descr="Depth")

                    # Data curves
                    n_depth = len(depths)
                    for mnem, (arr, unit) in las_curves.items():
                        data = arr[:n_depth] if len(arr) >= n_depth else np.pad(
                            arr, (0, n_depth - len(arr)), constant_values=np.nan
                        )
                        las.append_curve(mnem, data, unit=unit)

                    # ── Write LAS ─────────────────────────────────────
                    out_las = os.path.join(out_dir, las_fname)
                    las.write(out_las, version=2.0)
                    print(f"  → {out_las}")

                    # ── Well location ─────────────────────────────────
                    lat = lon = None
                    if fname in ext_locs:
                        lat, lon = ext_locs[fname]
                    else:
                        # Try DLIS parameters
                        for param in lf.parameters:
                            pn = param.name.upper()
                            if pn in ("LAT", "LATI", "LATITUDE", "WGS84LAT") and lat is None:
                                try:
                                    lat = float(param.values[0])
                                except Exception:
                                    pass
                            if pn in ("LON", "LONG", "LONGITUDE", "WGS84LON") and lon is None:
                                try:
                                    lon = float(param.values[0])
                                except Exception:
                                    pass

                    if lat is None or lon is None:
                        warnings.warn(
                            f"  No location found for {fname}.  "
                            "Provide coordinates via 'locations' or 'location_csv'."
                        )
                        skipped += 1
                        continue

                    new_entries.append({
                        "file": las_fname,
                        "latitude": float(lat),
                        "longitude": float(lon),
                    })

        except Exception as exc:
            warnings.warn(f"  Failed to process {fname}: {exc}")
            skipped += 1
            continue

    _update_well_locations_json(json_path, new_entries)
    print(f"  JSON → {json_path}  ({len(new_entries)} wells, {skipped} skipped)")
    _qc_las_json(json_path, out_dir)

    with open(json_path) as fh:
        return json.load(fh)


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def _build_parser():
    import argparse

    p = argparse.ArgumentParser(
        prog="converters",
        description="Multiphysics inversion input format converter.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # ── csv-to-nc4 ───────────────────────────────────────────────────
    c = sub.add_parser("csv-to-nc4", help="CSV / XYZ → NetCDF4")
    c.add_argument("csv_path")
    c.add_argument("out_path")
    c.add_argument("--x-col",     default="x")
    c.add_argument("--y-col",     default="y")
    c.add_argument("--value-col", default="value")
    c.add_argument("--input-crs", default="EPSG:4326")
    c.add_argument("--target-crs")
    c.add_argument("--grid-res",  type=float)
    c.add_argument("--data-type", default="gravity",
                   choices=["gravity", "magnetic", "dem"])
    c.add_argument("--unit-in",   default="mGal")
    c.add_argument("--inclination",  type=float)
    c.add_argument("--declination",  type=float)
    c.add_argument("--amplitude-nT", type=float)
    c.add_argument("--sep",       default=",")
    c.add_argument("--name")

    # ── geotiff-to-nc4 ───────────────────────────────────────────────
    g = sub.add_parser("geotiff-to-nc4", help="GeoTIFF / COG → NetCDF4")
    g.add_argument("tiff_path")
    g.add_argument("out_path")
    g.add_argument("--band",       type=int, default=1)
    g.add_argument("--data-type",  default="gravity",
                   choices=["gravity", "magnetic", "dem"])
    g.add_argument("--unit-in",    default="mGal")
    g.add_argument("--target-crs")
    g.add_argument("--inclination",  type=float)
    g.add_argument("--declination",  type=float)
    g.add_argument("--amplitude-nT", type=float)
    g.add_argument("--name")

    # ── nc-to-nc4 ────────────────────────────────────────────────────
    n = sub.add_parser("nc-to-nc4", help="NetCDF/CF → harmonised NetCDF4")
    n.add_argument("nc_path")
    n.add_argument("out_path")
    n.add_argument("--variable")
    n.add_argument("--data-type",  default="gravity",
                   choices=["gravity", "magnetic", "dem"])
    n.add_argument("--target-crs")
    n.add_argument("--inclination",  type=float)
    n.add_argument("--declination",  type=float)
    n.add_argument("--amplitude-nT", type=float)
    n.add_argument("--name")

    # ── segy-to-hdf5 ─────────────────────────────────────────────────
    s = sub.add_parser("segy-to-hdf5", help="SEG-Y → HDF5")
    s.add_argument("segy_path")
    s.add_argument("out_path")
    s.add_argument("--mode",       default="2d_fwi",
                   choices=["2d_fwi", "3d_reflection", "validation_3d"])
    s.add_argument("--line-name")
    s.add_argument("--input-crs",  default="EPSG:32613")
    s.add_argument("--is-time",    action="store_true",
                   help="Samples are in time domain (default: depth)")
    s.add_argument("--vp-avg",     type=float, default=2000.0)
    s.add_argument("--amp-scale",  type=float, default=1.0)
    s.add_argument("--amp-offset", type=float, default=0.0)

    # ── segd-to-hdf5 ─────────────────────────────────────────────────
    d = sub.add_parser("segd-to-hdf5", help="SEG-D → HDF5")
    d.add_argument("segd_path")
    d.add_argument("out_path")
    d.add_argument("--line-name")
    d.add_argument("--sample-rate-hz", type=float)
    d.add_argument("--is-depth",       action="store_true")
    d.add_argument("--vp-avg",         type=float, default=2000.0)
    d.add_argument("--depth-per-sample", type=float, default=1.0)
    d.add_argument("--trace-coords")

    # ── las-to-json ──────────────────────────────────────────────────
    l = sub.add_parser("las-to-json", help="LAS → normalised LAS + JSON")
    l.add_argument("las_files", help="LAS file, list (comma-sep), or directory")
    l.add_argument("out_dir")
    l.add_argument("json_path")
    l.add_argument("--location-csv")

    # ── dlis-to-json ─────────────────────────────────────────────────
    dl = sub.add_parser("dlis-to-json", help="DLIS → normalised LAS + JSON")
    dl.add_argument("dlis_files", help="DLIS file, list (comma-sep), or directory")
    dl.add_argument("out_dir")
    dl.add_argument("json_path")
    dl.add_argument("--location-csv")
    dl.add_argument("--frame-name")

    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.cmd == "csv-to-nc4":
        csv_xyz_to_netcdf4(
            args.csv_path, args.out_path,
            x_col=args.x_col, y_col=args.y_col, value_col=args.value_col,
            input_crs=args.input_crs, target_crs=args.target_crs,
            grid_res=args.grid_res, data_type=args.data_type,
            unit_in=args.unit_in, inclination=args.inclination,
            declination=args.declination, amplitude_nT=args.amplitude_nT,
            sep=args.sep, name=args.name,
        )

    elif args.cmd == "geotiff-to-nc4":
        geotiff_to_netcdf4(
            args.tiff_path, args.out_path,
            band=args.band, data_type=args.data_type,
            unit_in=args.unit_in, target_crs=args.target_crs,
            inclination=args.inclination, declination=args.declination,
            amplitude_nT=args.amplitude_nT, name=args.name,
        )

    elif args.cmd == "nc-to-nc4":
        netcdf_cf_to_netcdf4(
            args.nc_path, args.out_path,
            variable=args.variable, data_type=args.data_type,
            target_crs=args.target_crs, inclination=args.inclination,
            declination=args.declination, amplitude_nT=args.amplitude_nT,
            name=args.name,
        )

    elif args.cmd == "segy-to-hdf5":
        segy_to_hdf5(
            args.segy_path, args.out_path,
            mode=args.mode, line_name=args.line_name,
            input_crs=args.input_crs, is_depth=not args.is_time,
            vp_avg=args.vp_avg, amp_scale=args.amp_scale,
            amp_offset=args.amp_offset,
        )

    elif args.cmd == "segd-to-hdf5":
        segd_to_hdf5(
            args.segd_path, args.out_path,
            line_name=args.line_name,
            sample_rate_hz=args.sample_rate_hz,
            is_depth=args.is_depth,
            vp_avg=args.vp_avg,
            depth_per_sample=args.depth_per_sample,
            trace_coords=args.trace_coords,
        )

    elif args.cmd == "las-to-json":
        sources = args.las_files
        if "," in sources:
            sources = [s.strip() for s in sources.split(",")]
        las_to_json(sources, args.out_dir, args.json_path, location_csv=args.location_csv)

    elif args.cmd == "dlis-to-json":
        sources = args.dlis_files
        if "," in sources:
            sources = [s.strip() for s in sources.split(",")]
        dlis_to_json(
            sources, args.out_dir, args.json_path,
            location_csv=args.location_csv,
            frame_name=getattr(args, "frame_name", None),
        )


if __name__ == "__main__":
    main()
