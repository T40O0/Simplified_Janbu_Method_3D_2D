import math
import os
import time
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
from shapely.geometry import Polygon, Point
from shapely import contains_xy  # shapely 2.0+
import matplotlib.pyplot as plt
from scipy import ndimage
from shapely.geometry import LineString

# Optional Numba acceleration.  When numba is present, the hot inner sweep in
# SimpJanbu3D is JIT-compiled (~10x faster); otherwise a pure-Python fallback
# is used so the module still imports.
try:
    from numba import njit, prange
    _HAS_NUMBA = True
except ImportError:  # pragma: no cover
    _HAS_NUMBA = False
    prange = range  # fallback when numba isn't installed

    def njit(*args, **kwargs):  # type: ignore[no-redef]
        def _wrap(fn):
            return fn
        # Allow @njit (no args) or @njit(cache=True) usage.
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return _wrap


@njit(cache=True, fastmath=False)
def _inner_sweep_jit(
    st, c0, phi0, phi_inc, c_inc, maxC, maxPhi, FSy, At,
    gz_m, sin_dy_m, cos_dy_m, tan_dy_m,
    sin_dx_m, cos_dx_m, tan_dx_m, Atb_m,
    W_m, u_m, ky_m, kx_m, Ey_m, Ex_m,
):
    """JIT-compiled SimpJanbu3D inner sweep.

    Mirrors the linear 1-deg phi (or c) sweep + bracket-end interpolation.
    All array arguments must be float64 1-D arrays of the same length.
    Scalar params (kx, ky, Ex, Ey, u) should be pre-broadcast to arrays
    by the caller for type stability.

    Returns
    -------
    success : bool
    phi3d_final : float
    c3d_final : float
    FDx : float
    """
    M = W_m.shape[0]
    iter_count = 0
    FSY_prev = 0.0
    x_prev = 0.0
    phi3d = phi0
    c3d = c0

    while True:
        if st == 1:
            phi3d += phi_inc
            if phi3d >= maxPhi:
                return False, phi0, c0, 0.0
            x_curr_test = phi3d
        else:
            c3d += c_inc
            if c3d >= maxC:
                return False, phi0, c0, 0.0
            x_curr_test = c3d
        iter_count += 1

        tan_phi = math.tan(math.radians(phi3d))

        # Longitudinal sums
        FRy = 0.0
        FDy = 0.0
        for i in range(M):
            gz = gz_m[i]
            md_i = gz * (1.0 + (sin_dy_m[i] * tan_phi) / (FSy * gz))
            t1 = (c3d * At - u_m[i] * At * tan_phi + W_m[i] * tan_phi) \
                / cos_dy_m[i] / md_i
            t3 = W_m[i] * tan_dy_m[i] + ky_m[i] * W_m[i] + Ey_m[i]
            FRy += t1
            FDy += t3
        FSY_curr = FRy / (FDy + 1e-10)

        if FRy >= FDy and FRy >= 0.0 and FDy >= 0.0:
            # Compute FDx at the converged phi / c (two passes: FSx2 then FDx).
            FSx1 = 100.0
            num_t1x = 0.0
            den_t2x = 0.0
            den_t3x = 0.0
            for i in range(M):
                gz = gz_m[i]
                mdx_i = gz * (1.0 + (sin_dx_m[i] * tan_phi) / (FSx1 * gz))
                Nx_i = (W_m[i]
                        - c3d * Atb_m[i] * sin_dx_m[i] / FSx1
                        + u_m[i] * Atb_m[i] * tan_phi * sin_dx_m[i] / FSx1
                        ) / mdx_i
                num_t1x += c3d * Atb_m[i] * gz \
                    + (Nx_i - u_m[i] * Atb_m[i]) * tan_phi * cos_dx_m[i]
                den_t2x += Nx_i * gz * tan_dx_m[i]
                den_t3x += kx_m[i] * W_m[i] + Ex_m[i]
            FSx2 = num_t1x / (den_t2x + den_t3x + 1e-10)

            FDx = 0.0
            for i in range(M):
                gz = gz_m[i]
                mdx_i = gz * (1.0 + (sin_dx_m[i] * tan_phi) / (FSx2 * gz))
                Nx_i = (W_m[i]
                        - c3d * Atb_m[i] * sin_dx_m[i] / FSx2
                        + u_m[i] * Atb_m[i] * tan_phi * sin_dx_m[i] / FSx2
                        ) / mdx_i
                FDx += Nx_i * gz * tan_dx_m[i] + kx_m[i] * W_m[i] + Ex_m[i]

            # Interpolation matching the original linear-sweep behaviour.
            if iter_count > 1:
                if FSY_curr != FSY_prev:
                    interp_x = x_prev + (x_curr_test - x_prev) \
                        * (1.0 - FSY_prev) / (FSY_curr - FSY_prev)
                else:
                    interp_x = x_curr_test
                if st == 1:
                    return True, interp_x, c3d, FDx
                else:
                    return True, phi3d, interp_x, FDx
            else:
                if st == 1:
                    return True, phi3d - phi_inc, c3d, FDx
                else:
                    return True, phi3d, c3d - c_inc, FDx

        FSY_prev = FSY_curr
        x_prev = x_curr_test

# ====================================================
# Utility functions
# ====================================================
def worldGrid(transform, shape):
    """
    Given a rasterio transform and image size (shape=(rows, cols)),
    returns 2D ndarray of X, Y coordinates for each pixel center.
    """
    rows, cols = shape
    a = transform.a
    c_val = transform.c
    e = transform.e  # usually negative
    f_val = transform.f
    x = c_val + (np.arange(cols) + 0.5) * a
    y = f_val + (np.arange(rows) + 0.5) * e
    X, Y = np.meshgrid(x, y)
    return X, Y

def create_shp_mask(geom, transform, shape):
    """
    Create a 2D Boolean mask from a shapely geometry.
    geom:     a shapely geometry object
    transform: a rasterio transform

    Performance note: the polygon is point-in-polygon-tested only against
    the sub-grid covering its bounding box (with a 1-cell margin), then
    the result is stamped into the full-shape mask.  For small landslides
    on a large raster this is orders of magnitude faster than testing
    every raster pixel.
    """
    rows, cols = shape
    full = np.zeros((rows, cols), dtype=bool)

    # Empty / degenerate geometry guard.
    bounds = getattr(geom, "bounds", None)
    if bounds is None or not bounds:
        return full
    minx, miny, maxx, maxy = bounds

    a = transform.a
    c_val = transform.c
    e = transform.e  # usually negative
    f_val = transform.f

    # Pixel-index bounding box (inclusive on the low side, exclusive on the
    # high side).  Pixel-centre formulas: col = (x - c)/a - 0.5,
    # row = (y - f)/e - 0.5.  Add a 1-cell margin to make sure the mask
    # captures pixels whose centres are just inside the polygon edge.
    col_lo = int(np.floor((minx - c_val) / a - 0.5)) - 1
    col_hi = int(np.ceil((maxx - c_val) / a - 0.5)) + 2
    if e < 0:
        row_lo = int(np.floor((maxy - f_val) / e - 0.5)) - 1
        row_hi = int(np.ceil((miny - f_val) / e - 0.5)) + 2
    else:
        row_lo = int(np.floor((miny - f_val) / e - 0.5)) - 1
        row_hi = int(np.ceil((maxy - f_val) / e - 0.5)) + 2
    col_lo = max(0, col_lo)
    col_hi = min(cols, col_hi)
    row_lo = max(0, row_lo)
    row_hi = min(rows, row_hi)
    if col_lo >= col_hi or row_lo >= row_hi:
        return full

    x_sub = c_val + (np.arange(col_lo, col_hi) + 0.5) * a
    y_sub = f_val + (np.arange(row_lo, row_hi) + 0.5) * e
    Xs, Ys = np.meshgrid(x_sub, y_sub)
    sub_mask = contains_xy(geom, Xs.ravel(), Ys.ravel()).reshape(Xs.shape)
    full[row_lo:row_hi, col_lo:col_hi] = sub_mask
    return full

def safe_float(val):
    """
    Convert an input safely to float for factor of safety (FS) calculations.
    Returns:
        A finite float value if the conversion succeeds and the input is not
        a placeholder. Otherwise returns 0.0.
    """
    try:
        sval = str(val)
        if sval.startswith("(") and sval.endswith(")"):
            return 0.0
        if '[card]' in sval or 'card' in sval:
            return 0.0
        f = float(val)
        if not np.isfinite(f):
            return 0.0
        return f
    except Exception:
        return 0.0
        
# ====================================================
# Implementation of gradient_king 
# (replicating the MATLAB process in Python)
# ====================================================
def gradient_king(Elevation, csize):
    """
    Elevation : 2D ndarray (elevation data)
    csize     : cell size [m]

    Aspect : array of aspect (direction) angles for each cell[deg]
             starts north and rotates clockwise
    Slope  : array of slope angles for each cell[deg]

    Vectorised numpy implementation of the original ArcGIS-style 3x3
    Sobel gradient.  A Numba-JIT version was tried but produced ULP-level
    drift in Slope/Aspect that propagates to ~1e-6 differences in phi3d
    and rot3d after the rot search, so the numpy version is retained for
    numerical stability.  The one-time cost is small (~0.3s on a typical
    2801x2251 DEM).
    """
    m, n = Elevation.shape
    Slope = np.zeros((m, n))
    Aspect = np.zeros((m, n))
    if m < 3 or n < 3:
        return Slope, Aspect
    E = Elevation
    dz_dx = ((E[:-2, 2:] + 2.0 * E[1:-1, 2:] + E[2:, 2:])
             - (E[:-2, :-2] + 2.0 * E[1:-1, :-2] + E[2:, :-2])) / (8.0 * csize)
    dz_dy = ((E[2:, :-2] + 2.0 * E[2:, 1:-1] + E[2:, 2:])
             - (E[:-2, :-2] + 2.0 * E[:-2, 1:-1] + E[:-2, 2:])) / (8.0 * csize)
    rise_run = np.sqrt(dz_dx**2 + dz_dy**2)
    Slope[1:-1, 1:-1] = np.degrees(np.arctan(rise_run))
    aspect_math = np.degrees(np.arctan2(dz_dy, -dz_dx))
    cell_val = np.where(
        aspect_math < 0, 90.0 - aspect_math,
        np.where(aspect_math > 90, 360.0 - aspect_math + 90.0, 90.0 - aspect_math)
    )
    Aspect[1:-1, 1:-1] = cell_val
    return Slope, Aspect

# ====================================================
# SimpJanbu3D (back analysis)
# ====================================================
def SimpJanbu3D(mask_red, csize, Slope, Aspect, asp, c0, phi0, W0, u, gs, strength, kx, ky, Ex, Ey):
    """
    3D back analysis

    Parameters:
      mask_red  : boolean mask (2D ndarray) defining the analysis region
      csize     : cell size [m]
      Aspect    : array of aspect (direction) angles for each cell[deg]
                  starts north and rotates clockwise
      Slope     : array of slope angles for each cell[deg]
      c0, phi0  : initial cohesion[kN/m2] and friction angle[deg]
      W0        : volume per cell [m3]
      u         : pore pressure coefficient
      gw        : Unit weight of water [kN/m3]
      gd        : Unit weight of dry soil [kN/m3]
      gs        : Unit weight of saturated soil [kN/m3]
      ky        : Pseudo-static coeff for EQ. in longitudinal
      kx        : Pseudo-static coeff for EQ. in transverse
      Ey        : Applied horizontal load in longitudinal
      Ex        : Applied horizontal load in transverse
      strength  : 'phi' to iterate friction angle, otherwise 'c' to iterate cohesion

    Returns:
      rot3d   : corrected angle [deg] of sliding direction
      phi3d   : internal friction angle [deg] when FS = 1
      c3d     : cohesion when FS = 1
    """
    st = 1 if strength == 'phi' else 2
    asp_inc = 1
    phi_inc = 1
    c_inc = 10

    W = W0 * gs
    asp_shift = 0 #Rotation of failure surface from mean aspect
    #              0 means that failure direction equals mean aspect
    #              Clockwise is positive
    switchA = 0 # 0 means sum of transverse forces has not been positive
    switchB = 0 # 0 means sum of transverse forces has not been negative
    # Only once both positive and negative sums have been achieved does
    # switchA*switchB = 0, and the looping stops.

    iter2 = 0       # number of *successful* outer iterations (matches MATLAB)
    fail_attempts = 0 # number of consecutive inner-loop failures before any success
    ias = 1         # +1: shift positive, -1: shift negative
    asp0 = asp # Record initial aspect so that asp may be manipulated

    ROT_3d = []
    FDx_list_3d = []
    MAX_OUTER_ITER = 359

    # Polygon-constant per-cell arrays.  These depend only on mask_red,
    # so we compute them ONCE here instead of every asp_shift iteration
    # (the asp_shift loop only varies the trig terms via dy_vals/dx_vals).
    mask_flat = mask_red.ravel()
    M_const = int(np.count_nonzero(mask_flat))
    def _broadcast(v, length):
        if isinstance(v, np.ndarray) and v.ndim > 0:
            return np.ascontiguousarray(v.ravel()[mask_flat].astype(np.float64, copy=False))
        return np.full(length, float(v), dtype=np.float64)
    W_a_poly = _broadcast(W, M_const)
    u_a_poly = _broadcast(u, M_const)
    ky_a_poly = _broadcast(ky, M_const)
    kx_a_poly = _broadcast(kx, M_const)
    Ey_a_poly = _broadcast(Ey, M_const)
    Ex_a_poly = _broadcast(Ex, M_const)

    while switchA * switchB == 0:
        # Check the upper limit of the outer loop.
        if abs(asp_shift) >= MAX_OUTER_ITER:
            print(f" [Warning] The maximum search angle ({MAX_OUTER_ITER}°)has been reached. Stop the external iteration.")
            return asp, phi0, c0  # Return the initial value if convergence fails.
            break
        # Apply aspect shift to aspect (0 for attempt 1)
        asp_current = asp0 + asp_shift
        # Project slope vectors into longitudinal (failure
        # direction) and transverse directions.
        # Apparent dip formula: tan(alpha_app) = tan(Slope) * cos(d_aspect)
        # (Bug fix: previously used the linear approximation Slope*cos(...),
        # which under-estimates apparent dip on steep slopes.)
        dlon = Aspect - asp_current # Difference between each pixel's aspect
                                    # and the longitudinal direction
        dtra = (Aspect + 90) - asp_current # Difference between each pixel's
                                           # aspect and the transverse direction
        tan_slope = np.tan(np.deg2rad(Slope))
        dy_vals = np.rad2deg(np.arctan(tan_slope * np.cos(np.deg2rad(dlon)))) # longitudinal pixel slopes
        dx_vals = np.rad2deg(np.arctan(tan_slope * np.cos(np.deg2rad(dtra)))) # transverse pixel slopes

        # Speedup A + B: cache every per-cell quantity that is constant
        # within the inner phi/c loop.  Only the trig terms depend on
        # asp_shift; the polygon-level masked arrays (W_a_poly, u_a_poly,
        # ky/kx/Ey/Ex) were pre-broadcast once above and are reused.
        dy_rad_m = np.deg2rad(dy_vals.ravel()[mask_flat])
        dx_rad_m = np.deg2rad(dx_vals.ravel()[mask_flat])
        sin_dy_m = np.sin(dy_rad_m)
        cos_dy_m = np.cos(dy_rad_m)
        tan_dy_m = np.tan(dy_rad_m)
        sin_dx_m = np.sin(dx_rad_m)
        cos_dx_m = np.cos(dx_rad_m)
        tan_dx_m = np.tan(dx_rad_m)
        Atb_m = (csize * csize) * np.sqrt(
            1 - (sin_dx_m**2 * sin_dy_m**2)
        ) / (cos_dx_m * cos_dy_m)
        gz_m = np.sqrt(1 / (tan_dy_m**2 + tan_dx_m**2 + 1))
        At = csize * csize

        # Inner sweep: linear 1-deg phi (or c) walk + bracket-end interp,
        # delegated to the Numba-JIT'd helper.
        FSy = 1.0
        maxC = c0 + 50.0
        maxPhi = 90.0
        # Trig terms are the only per-asp_shift inputs; the polygon-constant
        # arrays were pre-broadcast above (W_a_poly etc.) so we just need to
        # ensure the trig arrays are contiguous float64 for the JIT call.
        success, phi_final, c_final, FDx = _inner_sweep_jit(
            int(st), float(c0), float(phi0),
            float(phi_inc), float(c_inc),
            float(maxC), float(maxPhi), float(FSy), float(At),
            np.ascontiguousarray(gz_m), np.ascontiguousarray(sin_dy_m),
            np.ascontiguousarray(cos_dy_m), np.ascontiguousarray(tan_dy_m),
            np.ascontiguousarray(sin_dx_m), np.ascontiguousarray(cos_dx_m),
            np.ascontiguousarray(tan_dx_m), np.ascontiguousarray(Atb_m),
            W_a_poly, u_a_poly, ky_a_poly, kx_a_poly, Ey_a_poly, Ex_a_poly,
        )
        inner_loop_success = bool(success)
        phi3d_final = float(phi_final)
        c3d_final = float(c_final)
        FDx = float(FDx)

        if not inner_loop_success:
            # Recovery: try alternate rotation directions before giving up.
            if iter2 == 0:
                if fail_attempts == 0:
                    asp_shift += asp_inc
                elif fail_attempts == 1:
                    asp_shift = -asp_inc
                    ias = -1
                else:
                    asp_shift += ias * asp_inc
                fail_attempts += 1
            else:
                asp_shift += ias * asp_inc
            continue

        FDx_list_3d.append(FDx)
        ROT_3d.append(asp_shift)
        iter2 += 1

        # Update transverse-force switches.  Each successful iteration is
        # responsible for setting *one* of A or B based on the sign of FDx;
        # once both have been set (i.e. we have bracketed FDx = 0) the outer
        # loop terminates.
        if FDx > 0:
            switchA = 1
        else:
            switchB = 1

        # MATLAB heuristic (#11): after the second successful iteration,
        # reverse the shift direction if |FDx| is *growing* (i.e. we are
        # walking away from the zero-crossing).  Mirrors
        # `if iter2 == 2 and abs(FDX(1)) < abs(FDX(2)): ias = -1` in
        # SimpJanbu3D.m.
        if iter2 == 2 and abs(FDx_list_3d[0]) < abs(FDx_list_3d[1]):
            ias = -1

        asp_shift += ias * asp_inc

        if switchA * switchB != 0:
            break

    # If neither transverse‐force condition is satisfied (switchA * switchB == 0)
    if switchA * switchB == 0:
        print(" [Warning] could not find a transverse‐force balance in either direction.")

    if len(FDx_list_3d) >= 2:
        # Take the last two entries
        xp = [FDx_list_3d[-2], FDx_list_3d[-1]]
        fp = [ROT_3d[-2], ROT_3d[-1]]
        # Sort the (xp, fp) pairs: xp is in ascending order
        xp_sorted, fp_sorted = zip(*sorted(zip(xp, fp)))
        print(xp_sorted, fp_sorted)
        # Interpolate to find rot3d at xp = 0
        rot3d = np.interp(0, xp_sorted, fp_sorted)
    else:
        rot3d = ROT_3d[-1]
        
    return rot3d , phi3d_final, c3d_final

# ====================================================
# SimpJanbu2D (back‐and‐forward analyses)
# ====================================================
def SimpleJanbu2D_slice(longest_mask, csize, Slope, Aspect, rot3d,
                          c0, phi0, W0, u, gs, strength, kx, ky, Ex, Ey,  mode="inverse"):
    """
    Simplified Janbu method (slice analysis) applied to 2D cross‐sections.

    Parameters:
      longest_mask: Boolean mask of valid cells
      csize:        Cell size[m]
      Aspect:       array of aspect (direction) angles for each cell[deg]
                    starts north and rotates clockwise
      Slope:        array of slope angles for each cell[deg]
      rot3d:        Slip azimuth angle [deg]
      c0:           Initial cohesion [kN/m2]
      phi0:         Initial internal friction angle [deg]
      W0:           Per-cell volume [m^3] (i.e. csize*csize*(G-S)).
                    Internally multiplied by ``gs`` to obtain weight.
      u:            Pore pressure [kN/m^2]
      gs:           Unit weight of soil [kN/m^3] used to convert W0 to weight
      kx, ky:       Pseudo-static seismic coefficients (transverse / longitudinal).
                    Only ``ky`` participates in the 2D longitudinal balance.
      Ex, Ey:       Applied horizontal loads (transverse / longitudinal).
                    Only ``Ey`` participates in the 2D longitudinal balance.
      strength:     'phi' to adjust friction angle, 'c' to adjust cohesion
      mode:         "inverse" (default) or "fs" for forward FS calculation
    """

    # --------------------------------------------------------------
    # 1. Calculate the effective tilt angle of each cell
    # --------------------------------------------------------------
    # Difference between slope aspect and slip section azimuth [rad]
    dtheta = np.deg2rad(Aspect - rot3d)
    # Slope: Slope angle at each cell [deg]
    # dtheta: azimuth difference between the slope aspect and the slip section [rad]
    Slope_rad = np.deg2rad(Slope)
    dy_vals = np.rad2deg(np.arctan(np.tan(Slope_rad) * np.cos(dtheta)))

    # Bug fix (#1): the function previously treated the per-cell volume W0 as
    # if it were already a weight, leaving the gs argument unused.  That made
    # the cohesion / pore-pressure terms numerically incommensurate with the
    # gravity-driven terms (off by ~gs ≈ 20×) and biased phi2d / c2d / FS2D.
    # Convert volume to weight here so the rest of the function works in kN.
    W = W0 * gs

    # --------------------------------------------------------------
    # 2. Slice width
    # --------------------------------------------------------------
    # Bug fix: the previous formula `csize / cos(rot3d)` blew up at
    # rot3d = ±90° / ±270° and produced **negative** AtbH whenever
    # cos(rot3d) < 0 (i.e. rot3d in (90°, 270°) mod 360°), which is a
    # large fraction of real cases.  The correct horizontal slice width
    # for grid-aligned cells stepped along the slip direction is
    #     b_i = csize / max(|cos(rot3d)|, |sin(rot3d)|)
    # because the cells in the strip step by csize in whichever grid
    # axis is closer to the slip direction.
    ang_rad = np.deg2rad(rot3d)
    _denom = max(abs(np.cos(ang_rad)), abs(np.sin(ang_rad)))
    AtbH = csize / max(_denom, 1e-6)

    # --------------------------------------------------------------
    # 3. Select cells for 2D analysis
    # --------------------------------------------------------------
    # Bug fix (#7): the previous filter `(W0 > 0) & (dy_vals > 0)` discarded
    # every horizontal- or counter-sloped cell, biasing the analysis at the
    # toe / crown of the slip.  We now keep all cells with positive volume
    # and finite tilt; the slice may legitimately include a few cells whose
    # local apparent dip is ≤ 0 (those simply contribute negative drive).
    valid_cells = (longest_mask
                   & ~np.isnan(W) & ~np.isnan(dy_vals)
                   & (W > 0))
    if np.sum(valid_cells) == 0:
        print(" [Warning] No valid cells available")
        return phi0, c0

    # --------------------------------------------------------------
    # 4. Calculate the factor of safety FS (calc_fs)
    # --------------------------------------------------------------
    # Speedup A + B: precompute the per-cell trigonometric terms and the
    # mask-filtered 1D vectors so calc_fs reduces to a handful of vector
    # ops + sums.  Results are identical to the previous form.
    dy_rad_m = np.deg2rad(dy_vals[valid_cells])
    cosA_m = np.cos(dy_rad_m)
    tanA_m = np.tan(dy_rad_m)
    cos2_m = cosA_m * cosA_m
    def _mask(v):
        return v[valid_cells] if isinstance(v, np.ndarray) and v.ndim > 0 else v
    W_m = _mask(W)
    u_m = _mask(u)
    ky_m = _mask(ky)
    Ey_m = _mask(Ey)
    drive_const = W_m * tanA_m + ky_m * W_m + Ey_m  # phi-independent
    sum_drive = float(np.sum(drive_const))

    def calc_fs(phi_val, c_val):
        # Calculate the Factor of Safety FS using the internal
        # friction angle phi_val [deg] and c_val[kN/m2]
        if sum_drive < 1e-10:
            return float('inf')
        fs_est = 1.0
        tan_phi = np.tan(np.deg2rad(phi_val))
        resist_num = c_val * AtbH + (W_m - u_m * AtbH) * tan_phi  # phi/c-only numerator
        for _ in range(30):
            # denom = cos^2 * (1 + tan_phi*tanA / FS)
            denom_m = cos2_m * (1.0 + (tan_phi * tanA_m) / fs_est)
            denom_m = np.where(np.abs(denom_m) < 1e-6, 1e-6, denom_m)
            fs_new = np.sum(resist_num / denom_m) / sum_drive
            if abs(fs_new - fs_est) < 0.001:
                return fs_new
            fs_est = 0.7 * fs_est + 0.3 * fs_new
        return fs_est

    # forward calculation
    if mode == "fs":
        fs = calc_fs(phi0, c0)
        return fs
        
    # --------------------------------------------------------------
    # 5. Evaluate the initial Factor of Safety FS
    # --------------------------------------------------------------
    target_fs = 1.0
    fs_initial = calc_fs(phi0, c0)

    # --------------------------------------------------------------
    # 6. Correct parameters according to "strength" (bisection search)
    # --------------------------------------------------------------
    if strength == 'phi':
        """
        If phi3d (here phi0) does not converge at 1.0 dspite “not insufficient strength" 
        (i.e. fs_initial >= 1.0), set the search range to 1° to 90° 
        """
        if fs_initial >= target_fs and phi0 == 1.0:
            phi_lower = 1.0
            phi_upper = 90.0
        else:
            # if Fs is less than 1.0 (insufficient strength) or otherwise,
            # set conventional search range 
            if fs_initial > target_fs:
                phi_lower = max(1.0, phi0 - 30.0)
                phi_upper = phi0
            else:
                phi_lower = phi0
                phi_upper = 90.0
            
        fs_lower = calc_fs(phi_lower, c0)
        fs_upper = calc_fs(phi_upper, c0)
        if (fs_lower - target_fs) * (fs_upper - target_fs) > 0:
            print(f" [Warning] There is no condition satisfying FS=1 within the search interval  [{phi_lower}, {phi_upper}] of φ")
            return phi0, c0
        
        max_iter = 20
        tol = 0.01
        for iter_i in range(max_iter):
            phi_mid = (phi_lower + phi_upper) / 2.0
            fs_mid = calc_fs(phi_mid, c0)
            #print(f" Iteration {iter_i+1}: φ = {phi_mid:.2f}°, FS = {fs_mid:.4f}")
            if abs(fs_mid - target_fs) < tol:
                return phi_mid, c0
            if (fs_mid - target_fs) * (fs_lower - target_fs) < 0:
                phi_upper = phi_mid
                fs_upper = fs_mid
            else:
                phi_lower = phi_mid
                fs_lower = fs_mid
        
        phi_result = phi_lower + (target_fs - fs_lower) * (phi_upper - phi_lower) / (fs_upper - fs_lower)
        return phi_result, c0

    else:  # If strength == ‘c’
        if fs_initial > target_fs:
            c_lower = max(0.0, c0 - 30.0)
            c_upper = c0
        else:
            c_lower = c0
            c_upper = c0 + 50
        
        fs_lower = calc_fs(phi0, c_lower)
        fs_upper = calc_fs(phi0, c_upper)
        if (fs_lower - target_fs) * (fs_upper - target_fs) > 0:
            print(f" [Warning] There is no condition satisfying FS=1 within the search interval  [{c_lower}, {c_upper}] of c")
            return phi0, c0
        
        max_iter = 20
        tol = 0.01
        for iter_i in range(max_iter):
            c_mid = (c_lower + c_upper) / 2.0
            fs_mid = calc_fs(phi0, c_mid)
            #print(f" Iteration {iter_i+1}: c = {c_mid:.2f}, FS = {fs_mid:.4f}")
            if abs(fs_mid - target_fs) < tol:
                return phi0, c_mid
            if (fs_mid - target_fs) * (fs_lower - target_fs) < 0:
                c_upper = c_mid
                fs_upper = fs_mid
            else:
                c_lower = c_mid
                fs_lower = fs_mid
        
        c_result = c_lower + (target_fs - fs_lower) * (c_upper - c_lower) / (fs_upper - fs_lower)
        return phi0, c_result

# ====================================================
# Extraction of 2D section
# ====================================================
def extract_longest_contiguous_slice(mask, X, Y, rot3d, csize):
    if np.sum(mask) == 0:
        return np.zeros_like(mask, dtype=bool)
    theta = np.deg2rad(rot3d)
    Y_rot = X * np.sin(theta) + Y * np.cos(theta)   # rot3d direction
    X_rot = X * np.cos(theta) - Y * np.sin(theta)   # rot3d orthogonal
    coords = np.argwhere(mask)
    xr = X_rot[mask]
    yr = Y_rot[mask]

    bin_width = csize * 1.0 # <---- input
    bin_step =  csize * 0.5 # <----
    x_min, x_max = xr.min(), xr.max()
    bins = np.arange(x_min, x_max + bin_step, bin_step)
    best_seq = []
    for b in bins:
        in_bin = (xr >= b) & (xr < b + bin_width)
        if np.count_nonzero(in_bin) == 0:
            continue
        idxs = coords[in_bin]
        yv = yr[in_bin]
        # Y sort
        order = np.argsort(yv)
        yv = yv[order]
        idxs = idxs[order]
        # Y Difference
        diffs = np.diff(yv)
        seqs = []
        cur = [0]
        for i, d in enumerate(diffs):
            if d <= csize * 1.5:
                cur.append(i+1)
            else:
                seqs.append(cur)
                cur = [i+1]
        seqs.append(cur)
        # Longest of any contiguous sequence 
        max_seqs = max(seqs, key=len)
        if len(max_seqs) > len(best_seq):
            best_seq = idxs[max_seqs]
    # mask generation
    out = np.zeros_like(mask, dtype=bool)
    for ij in best_seq:
        out[tuple(ij)] = True

    # Bug fix (#4): the previous version called `plt.show()` here, which
    # blocks the loop in `main` once per slide and is hostile to batch /
    # CI runs.  Visualisation is kept as a disabled debug snippet — flip
    # the constant below to True for one-off debugging.
    DEBUG_PLOT = False
    if DEBUG_PLOT:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(mask, cmap='gray')
        ax[0].set_title('Original Mask')
        ax[1].imshow(out, cmap='gray')
        ax[1].set_title('Extracted Longest Slice')
        plt.show()
    return out

def extract_deepest_contiguous_slice(mask, X, Y, rot3d, csize, G, S, Aspect=None):
    """
    mask    : 2d boolean mask
    X, Y    : 2d coordinate array 
    rot3d   : azimuth (north 0, clockwise positive) [deg]
    csize   : cell size [m]
    depth : landslide layer thickness G-S (same shape as mask)
    """

    if np.sum(mask) == 0:
        return np.zeros_like(mask, dtype=bool)

    theta = np.deg2rad(rot3d)
    Y_rot = X * np.sin(theta) + Y * np.cos(theta)   # rot3d direction
    X_rot = X * np.cos(theta) - Y * np.sin(theta)   # rot3d orthogonal
    coords = np.argwhere(mask)
    xr = X_rot[mask]
    yr = Y_rot[mask]
    
    depth = G-S
    bin_width = csize * 1.0 # <---- input
    bin_step =  csize * 0.5 # <----
    x_min, x_max = xr.min(), xr.max()
    bins = np.arange(x_min, x_max + bin_step, bin_step)

    best_mean_depth = -np.inf
    best_seq = []

    masked_depth = depth[mask]
    for b in bins:
        in_bin = (xr >= b) & (xr < b + bin_width)
        if np.count_nonzero(in_bin) == 0:
            continue
        idxs = coords[in_bin]
        yv = yr[in_bin]
        dval = masked_depth[in_bin]
        order = np.argsort(yv)
        yv = yv[order]
        idxs = idxs[order]
        dval = dval[order]
        diffs = np.diff(yv)
        seqs = []
        cur = [0]
        for i, d in enumerate(diffs):
            if d <= csize * 1.5:
                cur.append(i+1)
            else:
                seqs.append(cur)
                cur = [i+1]
        seqs.append(cur)
        # The average depth is calculated for each bin and each successive column, and the maximum is recorded.
        # Bug fix (#13): skip sequences whose depth values are all NaN so that
        # `np.nanmean` does not emit a "Mean of empty slice" RuntimeWarning.
        for seq in seqs:
            seq_vals = dval[seq]
            if seq_vals.size == 0 or np.all(np.isnan(seq_vals)):
                continue
            mean_d = np.nanmean(seq_vals)
            if mean_d > best_mean_depth:
                best_seq = idxs[seq]
                best_mean_depth = mean_d

    out = np.zeros_like(mask, dtype=bool)
    for ij in best_seq:
        out[tuple(ij)] = True

    """
    # Visualization

    if Aspect is not None:
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        # Create 3 subplots
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        
        # 1. Original mask
        ax[0].imshow(mask, cmap='gray')
        ax[0].set_title('Original Mask')
        ax[0].axis('equal')
        
        # 2. Extracted 2D slice
        ax[1].imshow(out, cmap='gray')
        ax[1].set_title('Extracted Deepest Slice')
        ax[1].axis('equal')
        
        # 3. Visualization of Aspect
        masked_aspect = np.ma.masked_array(Aspect, ~mask)
        
        # Use circular color map since azimuth is 0-360 degrees
        cmap_aspect = plt.cm.hsv
        norm = Normalize(vmin=0, vmax=360)
        im = ax[2].imshow(masked_aspect, cmap=cmap_aspect, norm=norm)
        ax[2].set_title('Aspect (Direction)')
        ax[2].axis('equal')
        
        # Add color bar
        cbar = fig.colorbar(im, ax=ax[2])
        cbar.set_label('Direction (degrees)')
        
        # Arrows to indicate scanning direction
        h, w = mask.shape
        center_y, center_x = h//2, w//2
        arrow_length = min(h, w) * 0.2
        dx = arrow_length * np.sin(np.deg2rad(rot3d))
        dy = arrow_length * np.cos(np.deg2rad(rot3d))
        
        ax[2].arrow(center_x, center_y, dx, -dy, 
                   head_width=arrow_length*0.15, 
                   head_length=arrow_length*0.15, 
                   fc='white', ec='black', linewidth=2)
        ax[2].text(center_x + dx*1.1, center_y - dy*1.1, 
                  f'rot3d={rot3d}°', 
                  color='white', fontsize=10,
                  bbox=dict(facecolor='black', alpha=0.5))
        
        plt.tight_layout()
        plt.show()
    """
    return out

# ====================================================
# main process
# ====================================================

def main(
    poly_shp=None,
    slip_tif=None,
    dem_tif=None,
    out_path=None,
    phi_thresh=1.0,
    c_thresh=1.0,
    gw=9.8,
    gd=16.0,
    gs=20.0,
    ru=0.25,
    water_mode='Ru',
    water_depth_GL=0.0,
    kx=0.0,
    ky=0.0,
    Ex=0.0,
    Ey=0.0,
    pga_raster=None,
    pga_scaling=1.0,
    strength='phi',
    fail_type='Progressive',
    skip_2d=False,
    n_jobs=1,
    parallel_backend='threading',
    limit=None,
):
    """Run the full Janbu back-analysis pipeline.

    All arguments have defaults equivalent to the original hard-coded values,
    so calling ``main()`` with no arguments reproduces the legacy behaviour
    (reads ``input/landslide_poly.shp``, ``input/slide.tif``,
    ``input/DEM10.tif`` and writes everything under ``output/``).

    Parameters
    ----------
    poly_shp, slip_tif, dem_tif : str | Path, optional
        Input file paths.  Exactly one DEM is consumed; ``fail_type`` is a
        metadata label describing what that DEM represents.
    out_path : str | Path, optional
        Output directory; auto-created if missing.
    phi_thresh, c_thresh : float
        Initial guesses for friction angle [deg] and cohesion [kN/m^2] fed
        into the back-analysis inner loop.
    gw, gd, gs : float
        Unit weights of water, dry soil, and saturated soil [kN/m^3].
    ru : float
        Pore-pressure ratio.  Used only when ``water_mode='Ru'``.
        ``u = gw * (G - S) * Ru`` so this is effectively the height of
        the water column above the slip surface expressed as a fraction
        of the slip-mass thickness (NOT Skempton's Ru).
    water_mode : {'Ru', 'GL'}, default ``'Ru'``
        How pore pressure is parameterised.  ``'Ru'`` keeps the legacy
        Ru-based formula.  ``'GL'`` switches to a uniform groundwater
        depth below ground surface (in metres): the water column above
        the slip surface at cell (i,j) is ``max(0, (G - S) - water_depth_GL)``
        and ``u = gw * h_water``.
    water_depth_GL : float, default 0.0
        Groundwater depth below ground surface [m].  Only used when
        ``water_mode='GL'``.  0.0 means the water table is at the
        ground surface (= fully saturated above the slip surface).
    kx, ky : float
        Pseudo-static seismic coefficients (transverse / longitudinal).
        ``ky`` is overridden per cell when ``pga_raster`` is supplied.
    Ex, Ey : float
        Applied horizontal loads (transverse / longitudinal).
    pga_raster : str | Path, optional
        Path to a PGA raster (.tif) co-registered with the slip
        surface.  When supplied, each cell's ``ky`` is replaced by the
        sampled PGA value times ``pga_scaling``.  Scalar ``ky`` is
        ignored in that case.
    pga_scaling : float, default 1.0
        Multiplier applied to PGA raster values before being used as
        per-cell ``ky``.
    strength : {'phi', 'c'}
        Which parameter to back-solve when targeting FS = 1.
    fail_type : {'Progressive', 'Catastrophic'}
        Metadata label describing what the supplied DEM represents:
        ``Progressive`` = current (post-failure) ground surface,
        ``Catastrophic`` = pre-failure top surface.  The label is recorded
        on every output feature for documentation; it does NOT change the
        math (analysis always uses ``G - S`` where ``G`` is the supplied
        DEM).
    skip_2d : bool, default ``False``
        When ``True``, skip the 2D cross-section back-analysis and forward
        FS calculation entirely (no slice extraction, no phi2d / c2d /
        FS2Dby3D, no polyline shapefile, no phi2d histogram).  Saves
        roughly half the per-slide cost when only the 3D results are
        needed.
    """
    start = time.time()

    # Resolve defaults
    inPath = 'input'
    poly_shp = poly_shp if poly_shp is not None else os.path.join(inPath, 'landslide_poly.shp')
    slip_tif = slip_tif if slip_tif is not None else os.path.join(inPath, 'slide.tif')
    dem_tif = dem_tif if dem_tif is not None else os.path.join(inPath, 'DEM10.tif')
    outPath = str(out_path) if out_path is not None else 'output'

    gi = (gd + gs) / 2

    # Bug fix (#3): make sure the output directory exists before any
    # `to_file` / `to_csv` call.  geopandas raises an opaque error if the
    # parent directory is missing, which used to surface only at the very
    # end of a long batch run.
    os.makedirs(outPath, exist_ok=True)

    # Read landslide extents
    dep_shp = str(poly_shp)
    F_gdf = gpd.read_file(dep_shp)
    F = F_gdf.to_dict('records')
    print(f" [Info] Shapefile '{dep_shp}' has been read. (1/4)")

    # Name output extents
    ba_shp = os.path.join(outPath, 'back_analysis.shp')

    # Read slip surface raster
    slip_surf = str(slip_tif)
    with rasterio.open(slip_surf) as src:
        Slip = src.read(1)
        transform = src.transform
        csize = src.res[0]  # Assume square cells
    print(f" [Info] TIF file '{slip_surf}' has been read. (2/4)")

    # Read the ground surface raster.  The original MATLAB / earlier Python
    # port read two DEMs (DEM + TOP) and switched between them per fail_type,
    # but only one was ever consumed.  The interface is now "one DEM raster;
    # fail_type just labels what it represents".
    dem_surf = str(dem_tif)
    with rasterio.open(dem_surf) as src:
        DEM = src.read(1)
    print(f" [Info] TIF file '{dem_surf}' has been read "
          f"(treated as {fail_type}). (3/4)")

    # Calculate slip surface slope, aspect (gradient_king function)
    # Bug fix (#15): gradient_king no longer returns the unused dz/dx, dz/dy
    # arrays.  Both the original MATLAB and this Python port computed them
    # but never consumed them downstream.
    SLOPE, ASPECT = gradient_king(Slip, csize)
    shape_img = Slip.shape  # (rows, cols)
    X, Y = worldGrid(transform, shape_img)
    print(f" [Info] Slip surface slope calculation and grid creation completed. (4/4)")

    # Optional PGA raster (per-cell ky source).  Must share the slip raster's
    # grid (same transform / shape).  We reproject if needed via rasterio.
    PGA_local = None
    if pga_raster is not None:
        with rasterio.open(str(pga_raster)) as src_pga:
            if src_pga.transform == transform and src_pga.shape == shape_img:
                PGA_local = src_pga.read(1).astype(float)
            else:
                # Reproject onto the slip raster's grid using nearest neighbour.
                from rasterio.warp import reproject, Resampling
                PGA_local = np.zeros(shape_img, dtype=float)
                reproject(
                    source=src_pga.read(1),
                    destination=PGA_local,
                    src_transform=src_pga.transform,
                    src_crs=src_pga.crs,
                    dst_transform=transform,
                    dst_crs=src_pga.crs,
                    resampling=Resampling.nearest,
                )
        print(f" [Info] PGA raster '{pga_raster}' loaded (scaling = {pga_scaling}).")
    
    
    # List to store continuous cross sections (converted to polylines) used in 2D analysis.
    polyline_features = []

    # Optional: clip the polygon list to the first ``limit`` features for
    # quick smoke / verification runs.  ``limit=None`` processes everything.
    if limit is not None:
        F = F[:int(limit)]
    total_features = len(F)

    def _process_slide(idx, feature):
        """Per-polygon body.  Returns (feature, polyline_records: list).

        Closes over the shared rasters / params in the surrounding ``main``
        scope.  Each early "continue" path in the original loop is now a
        ``return`` here so the function can be dispatched in parallel.
        """
        poly_local = []
        progress = (idx + 1) / total_features * 100
        print(f"Processing slide {idx + 1}/{total_features} ({progress:.1f}% complete)")
        feature['skip_reason'] = ""
        try:
            geom = feature['geometry']

            # create mask
            mask = create_shp_mask(geom, transform, shape_img)
            rows, cols = shape_img
            if mask.ndim != 2:
                mask = mask.reshape(rows, cols)
            
            # Extract coordinates within mask
            x_inside = X[mask]
            y_inside = Y[mask]
            if x_inside.size == 0 or y_inside.size == 0:
                print(f"Slide {idx+1}: No data within the mask (skipped)")
                feature['skip_reason'] = "No data within the mask"
                feature['c3d']    = 0.0
                feature['phi3d']  = 0.0
                feature['rot3d']  = 0.0
                feature['c2d']    = 0.0
                feature['phi2d']  = 0.0
                feature['FS2D']   = 0.0
                return feature, poly_local
            
            # Range with 50 m buffer.  Compute the corresponding pixel-index
            # bbox directly from the affine transform - this used to build a
            # full-grid Boolean mask (6.3M cells) and call argwhere per
            # polygon, which dominated the loop for many slides.
            xmin, xmax = float(np.min(x_inside)) - 50.0, float(np.max(x_inside)) + 50.0
            ymin, ymax = float(np.min(y_inside)) - 50.0, float(np.max(y_inside)) + 50.0
            a = transform.a
            c_val = transform.c
            e = transform.e
            f_val = transform.f
            col_lo = int(np.floor((xmin - c_val) / a - 0.5))
            col_hi = int(np.ceil((xmax - c_val) / a - 0.5))
            if e < 0:
                row_lo = int(np.floor((ymax - f_val) / e - 0.5))
                row_hi = int(np.ceil((ymin - f_val) / e - 0.5))
            else:
                row_lo = int(np.floor((ymin - f_val) / e - 0.5))
                row_hi = int(np.ceil((ymax - f_val) / e - 0.5))
            XYminr = max(0, row_lo)
            XYmaxr = min(rows - 1, row_hi)
            XYminc = max(0, col_lo)
            XYmaxc = min(cols - 1, col_hi)
            if XYminr > XYmaxr or XYminc > XYmaxc:
                print(f"Slide {idx+1}: Target area not found (skipped)")
                feature['skip_reason'] = "Target area not found"
                feature['c3d']    = 0.0
                feature['phi3d']  = 0.0
                feature['rot3d']  = 0.0
                feature['c2d']    = 0.0
                feature['phi2d']  = 0.0
                feature['FS2D']   = 0.0
                return feature, poly_local
            
            # Sub mask to be analyzed
            mask_red = mask[XYminr:XYmaxr+1, XYminc:XYmaxc+1]
            print(f"Slide {idx+1}: mask_red shape: {mask_red.shape}")
            if np.sum(mask_red) == 0:
                print(f"Slide {idx+1}: mask_red is empty (skipped)")
                feature['skip_reason'] = "mask_red is empty"
                feature['c3d']    = 0.0
                feature['phi3d']  = 0.0
                feature['rot3d']  = 0.0
                feature['c2d']    = 0.0
                feature['phi2d']  = 0.0
                feature['FS2D']   = 0.0
                return feature, poly_local
            
            # Extract sub-regions.  ``G`` is the supplied DEM regardless of
            # ``fail_type`` (the label is metadata only - see the docstring).
            S = Slip[XYminr:XYmaxr+1, XYminc:XYmaxc+1]
            G = DEM[XYminr:XYmaxr+1, XYminc:XYmaxc+1]
            Slope_local = SLOPE[XYminr:XYmaxr+1, XYminc:XYmaxc+1]
            Aspect_local = ASPECT[XYminr:XYmaxr+1, XYminc:XYmaxc+1]
            
            
            """
            # XYZ of sub-areas to CSV file (for checking)
            # (1) Cut out the X, Y coordinates of the sub-region
            X_sub = X[XYminr:XYmaxr+1, XYminc:XYmaxc+1]
            Y_sub = Y[XYminr:XYmaxr+1, XYminc:XYmaxc+1]

            # (2) Flatten and pack into a DataFrame
            df = pd.DataFrame({
                'X':X_sub.ravel(),
                'Y':Y_sub.ravel(),
                'G':G.ravel(),
                'S':S.ravel(),
                'Slope_local':Slope_local.ravel(),
                'Aspect_local':Aspect_local.ravel()
            })

            # (3) Save to CSV
            csv_path = './output/subregion_xyz.csv'
            df.to_csv(csv_path, index=False)
            print(f"Saved XYZ data of sub-areas to CSV file: {csv_path}")
            
            # (4) Extra: Export polygon vertices to CS
            # First, determine if it is Shapely geometry or GeoJSON-dict
            if isinstance(geom, dict):
                # For GeoJSON-dict, keep the original code
                geom_type = geom['type']
                coords_list = geom['coordinates']
            else:
                # For Shapely object
                geom_type = geom.geom_type            # Polygon or MultiPolygon
                # If GeoJSON-style nested list is needed, mapping () can be used
                from shapely.geometry import mapping
                coords_list = mapping(geom)['coordinates']

            if geom_type == 'Polygon':
                # first element is perimeter, second and later are holes
                outer = coords_list[0]                # [(x1,y1),(x2,y2),…]
                df_outer = pd.DataFrame(outer, columns=['X','Y'])
                csv_outer = f'./output/polygon_{idx+1:03d}_vertices.csv'
                df_outer.to_csv(csv_outer, index=False)
                print(f"[Slide {idx+1}] perimeter {len(outer)} points → {csv_outer}")

                # output holes if needed
                for h, hole in enumerate(coords_list[1:], start=1):
                    df_hole = pd.DataFrame(hole, columns=['X','Y'])
                    csv_hole = f'./output/polygon_{idx+1:03d}_hole{h:02d}.csv'
                    df_hole.to_csv(csv_hole, index=False)
                    print(f"  hole {h}: {len(hole)} points → {csv_hole}")

            elif geom_type == 'MultiPolygon':
                # loop for each part 
                for p, part in enumerate(coords_list, start=1):
                    outer = part[0]
                    df_outer = pd.DataFrame(outer, columns=['X','Y'])
                    csv_outer = f'./output/multipoly_{idx+1:03d}_part{p:02d}_outer.csv'
                    df_outer.to_csv(csv_outer, index=False)
                    print(f"[Slide {idx+1}] MultiPolygon part {p} perimeter {len(outer)} points → {csv_outer}")
        	"""
        
        
            
            Rgh = np.nanstd(Slope_local)
            if np.isnan(Rgh):
                Rgh = 0

            # Model Parameter
            feature['g_d'] = float(gd)
            feature['g_s'] = float(gs)
            feature['g_w'] = float(gw)
            feature['g_i'] = float(gi)
            feature['Ru'] = float(ru)
            feature['Rgh'] = float(Rgh)
            # Stamp the analysis-mode label on every feature so downstream
            # consumers can tell whether a run was treated as Progressive
            # or Catastrophic (fail_type does not affect the math).
            feature['fail_type'] = str(fail_type)
            
            # Volume
            W0 = (csize * csize) * (G - S)
            if np.nansum(W0[mask_red]) == 0 or np.isnan(np.nansum(W0[mask_red])):
                print(f"Slide {idx+1}: Volume is zero or NaN (skipped)")
                feature['skip_reason'] = "Volume is zero or NaN"
                feature['c3d']    = 0.0
                feature['phi3d']  = 0.0
                feature['rot3d']  = 0.0
                feature['c2d']    = 0.0
                feature['phi2d']  = 0.0
                feature['FS2D']   = 0.0
                return feature, poly_local

            # Hydraulic head
            if water_mode == 'GL':
                # Water table at GL - water_depth_GL m below ground.
                # Pore pressure at slip = gw * max(0, (G - S) - water_depth_GL).
                h_water = np.clip((G - S) - water_depth_GL, 0.0, None)
                u_i_val = gw * h_water
            else:
                # Default 'Ru' branch (legacy)
                u_i_val = gw * (G - S) * ru

            # Per-cell ky derived from PGA raster, if supplied.  PGA is
            # sampled to the same sub-grid as the slip surface.
            if PGA_local is not None:
                ky_in = (PGA_local[XYminr:XYmaxr+1, XYminc:XYmaxc+1]
                         * float(pga_scaling))
            else:
                ky_in = ky

            # 3D back analysis
            asp = 0
            try:
                rot3d, phi3d, c3d = SimpJanbu3D(mask_red, csize, Slope_local, Aspect_local,
                                                 asp, c_thresh, phi_thresh, W0, u_i_val, gi,
                                                 strength, kx, ky_in, Ex, Ey)
            except Exception as e:
                print(f"Slide {idx+1}: SimpJanbu3D error (skipped): {e}")
                feature['skip_reason'] = f"SimpJanbu3D error: {e}"
                feature['c3d']    = 0.0
                feature['phi3d']  = 0.0
                feature['rot3d']  = 0.0
                feature['c2d']    = 0.0
                feature['phi2d']  = 0.0
                feature['FS2D']   = 0.0
                return feature, poly_local

            feature['c3d']    = c3d
            feature['phi3d']  = phi3d
            feature['rot3d']  = rot3d

            # ------------------------------------------------------------
            # 2D back / forward analysis (optional - skip via --no-2d).
            # When disabled, leave phi2d / c2d / FS2Dby3D / cell_count at
            # their schema-default 0 / empty values and skip slice
            # extraction + polyline generation for this slide.
            # ------------------------------------------------------------
            if skip_2d:
                feature['skip_reason'] = ""
                print(f"Slide {idx+1} processed successfully (3D only).")
                return feature, poly_local

            # local grid
            sub_X = X[XYminr:XYmaxr+1, XYminc:XYmaxc+1]
            sub_Y = Y[XYminr:XYmaxr+1, XYminc:XYmaxc+1]
            try:
                #longest_mask = extract_longest_contiguous_slice(mask_red, sub_X, sub_Y, rot3d, csize)
                longest_mask = extract_deepest_contiguous_slice(mask_red, sub_X, sub_Y, rot3d, csize,G,S, Aspect_local)
            except Exception as e:
                print(f"Slide {idx+1}: extract_longest_contiguous_slice error(skipped): {e}")
                feature['skip_reason'] = f"extract_longest_contiguous_slice error: {e}"
                feature['c2d']    = 0.0
                feature['phi2d']  = 0.0
                feature['FS2D']   = 0.0
                return feature, poly_local

            if np.sum(longest_mask) == 0:
                print(f"Slide {idx+1}: longest_mask is empty (skipped)")
                feature['skip_reason'] = "longest_mask is empty"
                feature['c2d']    = 0.0
                feature['phi2d']  = 0.0
                feature['FS2D']   = 0.0
                return feature, poly_local

            # 2D back analysis
            try:
                phi2d, c2d = SimpleJanbu2D_slice(longest_mask, csize, Slope_local, Aspect_local,
                                                  rot3d, c_thresh, phi_thresh, W0, u_i_val, gi,
                                                  strength, kx, ky_in, Ex, Ey, "inverse")
            except Exception as e:
                print(f"Slide {idx+1}: SimpleJanbu2D_slice_inverse error: {e}")
                feature['skip_reason'] = f"SimpleJanbu2D_slice_inverse error: {e}"
                feature['c2d']    = 0.0
                feature['phi2d']  = 0.0
                feature['FS2Dby3D']   = 0.0
                return feature, poly_local

            feature['c2d']    = c2d
            feature['phi2d']  = phi2d

            # 2D forward analysis
            try:
                FS2Dby3D = SimpleJanbu2D_slice(longest_mask, csize, Slope_local, Aspect_local,
                                           rot3d, c3d, phi3d, W0, u_i_val, gi,
                                           strength, kx, ky_in, Ex, Ey, "fs")
            except Exception as e:
                print(f"Slide {idx+1}: SimpleJanbu2D_slice_fs error(skipped): {e}")
                feature['skip_reason'] = f"SimpleJanbu2D_slice_fs error: {e}"
                feature['FS2Dby3D']   = 0.0
                return feature, poly_local

            feature['FS2Dby3D'] = safe_float(FS2Dby3D)
            feature['skip_reason'] = ""

            # Polyline
            # Connect cells with polyline
            indices = np.argwhere(longest_mask)
            pts = [(sub_X[i,j], sub_Y[i,j]) for (i,j) in indices]
            # θ = azimuth angle from north, clockwise positive.
            # Coordinates are (X = East, Y = North), so the unit vector
            # along the slip direction θ is (sinθ, cosθ):
            #   θ=0   (N) → (0, 1)
            #   θ=90  (E) → (1, 0)
            #   θ=180 (S) → (0, -1)
            #   θ=270 (W) → (-1, 0)
            # Bug fix: previously the code converted θ to a math-angle
            # (90 - θ) and then took (sin, cos) again, which effectively
            # swapped ux/uy and rotated the projection axis by 90°.
            theta_rad = np.deg2rad(rot3d)
            ux = np.sin(theta_rad)   # East component
            uy = np.cos(theta_rad)   # North component
            
            feature['cell_count'] = len(pts)
            
            if len(pts) >= 2:
                # Calculate the projected value in the sliding direction
                projs = np.array([x*ux + y*uy for x,y in pts])
                order = np.argsort(projs)
                sorted_pts = [pts[k] for k in order]
                line = LineString(sorted_pts)

                poly_local.append({
                    'slide':      idx+1,
                    'FS2Dby3D':   safe_float(FS2Dby3D),
                    'phi3d':      phi3d,
                    'cell_count': len(sorted_pts),
                    'geometry':   line
                })
            print(f"Slide {idx+1} processed successfully.")
            return feature, poly_local

        except Exception as e:
            print(f"Slide {idx+1}: Error occurred. Skipping... {e}")
            feature['skip_reason'] = f"Exception: {e}"
            feature['c3d']    = 0.0
            feature['phi3d']  = 0.0
            feature['rot3d']  = 0.0
            feature['FS2Dby3D']   = 0.0
            feature['c2d']    = 0.0
            feature['phi2d']  = 0.0
            return feature, poly_local

    # Dispatch the per-polygon work.  Threading is the default because
    # Numba (and numpy on large arrays) release the GIL during the hot
    # inner sweep, so multiple polygons can overlap their JIT execution.
    if n_jobs == 1 or total_features < 2:
        for idx, feature in enumerate(F):
            _, poly_recs = _process_slide(idx, feature)
            polyline_features.extend(poly_recs)
    else:
        from joblib import Parallel, delayed
        _results = Parallel(n_jobs=n_jobs, backend=parallel_backend)(
            delayed(_process_slide)(i, F[i]) for i in range(total_features)
        )
        # With process backends (loky/multiprocessing), the worker mutates a
        # copy of the feature dict.  Copy the updates back into the master F
        # so the downstream schema-fill / shapefile-write sees them.
        for idx, (feat_out, poly_recs) in enumerate(_results):
            F[idx].update(feat_out)
            polyline_features.extend(poly_recs)

    # Bug fix (#14): every skip path used to set its own subset of keys —
    # e.g. one path forgot `FS2Dby3D`, another forgot `c2d`/`phi2d`.  The
    # resulting dict-list produced a GeoDataFrame whose columns were riddled
    # with NaNs in unrelated rows, and shapefile writers complained.
    # Normalise the schema once here so every feature carries every key.
    _expected_defaults = {
        'c3d': 0.0, 'phi3d': 0.0, 'rot3d': 0.0,
        'c2d': 0.0, 'phi2d': 0.0,
        'FS2Dby3D': 0.0,
        'cell_count': 0,
        'g_d': 0.0, 'g_s': 0.0, 'g_w': 0.0, 'g_i': 0.0,
        'Ru': 0.0, 'Rgh': 0.0,
        'fail_type': str(fail_type),
        'skip_reason': "",
    }
    for feature in F:
        for key, default in _expected_defaults.items():
            feature.setdefault(key, default)

    # Save shapefile (back-analysis results)
    F_gdf_out = gpd.GeoDataFrame(F, crs=F_gdf.crs)
    F_gdf_out.to_file(ba_shp)

    # Save polyline shapefile (only when 2D analysis produced lines).
    if not skip_2d and polyline_features:
        polyline_gdf = gpd.GeoDataFrame(polyline_features, geometry='geometry', crs=F_gdf.crs)
        polyline_shp = os.path.join(outPath, '2D_stability_polyline.shp')
        polyline_gdf.to_file(polyline_shp)
        print(f" [Info] Save polyline shapefile '{polyline_shp}'")
    elif skip_2d:
        print(" [Info] 2D analysis skipped (--no-2d): polyline shapefile not written.")
    
    # Create phi3d histogram and output to CSV
    phi3d_arr = np.array([feature.get('phi3d', 0) for feature in F])
    idx_valid = phi3d_arr > phi_thresh
    phi3d_filt = phi3d_arr[idx_valid]

    plt.figure()
    plt.hist(phi3d_filt, bins=10, density=True)
    plt.xlabel('phi3d (deg)')
    plt.ylabel('Probability Density')
    plt.title('Distribution of phi3d')
    histogram_path = os.path.join(outPath, 'phi3d_hist.png')
    plt.savefig(histogram_path)
    plt.close()

    counts, bin_edges = np.histogram(phi3d_filt, bins=10, density=True)
    center_phi = (bin_edges[:-1] + bin_edges[1:]) / 2
    lower_bounds = bin_edges[:-1]
    upper_bounds = bin_edges[1:]
    includes_lower = [True] * len(lower_bounds)
    includes_upper = [False] * len(lower_bounds)
    includes_upper[-1] = True

    histogram_csv_path = os.path.join(outPath, 'phi3d_hist.csv')
    hist_df = pd.DataFrame({
        'bin_center': center_phi,
        'density': counts,
        'lower_bound': lower_bounds,
        'upper_bound': upper_bounds,
        'includes_lower': includes_lower,
        'includes_upper': includes_upper
    })
    hist_df.to_csv(histogram_csv_path, index=False)
    
    """
    # for RegionGrow3D
    # DOI: 10.5066/P1BSMGGD
    from scipy.io import savemat
    center_phi = np.array(center_phi)      # shape (N,)
    prob = np.array(counts)                # shape (N,)
    prob_coh = np.repeat(c_thresh, center_phi.size)
    # prob_coh = np.repeat(c_thresh, center_phi.size)[None, :]
    mdic = {
        'prob':     prob,
        'prob_phi': center_phi,
        'prob_coh': prob_coh
    }
    matfile = os.path.join(out_path, 'phi3d_hist.mat')
    savemat(matfile, mdic)
    print(f"Saved MATLAB .mat file to {matfile}")
    """
    
    if not skip_2d:
        # Create phi2d histogram and output to CSV
        phi2d_arr = np.array([feature.get('phi2d', 0.0) for feature in F])
        idx_valid_phi2d = phi2d_arr > phi_thresh
        phi2d_filt = phi2d_arr[idx_valid_phi2d]

        plt.figure()
        plt.hist(phi2d_filt, bins=10, density=True)
        plt.xlabel('phi2d (deg)')
        plt.ylabel('Probability Density')
        plt.title('Distribution of phi2d')
        histogram_phi2d_path = os.path.join(outPath, 'phi2d_hist.png')
        plt.savefig(histogram_phi2d_path)
        plt.close()

        counts2d, bin_edges2d = np.histogram(phi2d_filt, bins=10, density=True)
        center_phi2d = (bin_edges2d[:-1] + bin_edges2d[1:]) / 2
        lower_bounds2d = bin_edges2d[:-1]
        upper_bounds2d = bin_edges2d[1:]
        includes_lower2d = [True] * len(lower_bounds2d)
        includes_upper2d = [False] * len(lower_bounds2d)
        includes_upper2d[-1] = True

        histogram_phi2d_csv_path = os.path.join(outPath, 'phi2d_hist.csv')
        phi2d_hist_df = pd.DataFrame({
            'bin_center': center_phi2d,
            'density': counts2d,
            'lower_bound': lower_bounds2d,
            'upper_bound': upper_bounds2d,
            'includes_lower': includes_lower2d,
            'includes_upper': includes_upper2d
        })
        phi2d_hist_df.to_csv(histogram_phi2d_csv_path, index=False)
    
    # Save FS2D results to CSV
    safety_data = []
    for i, feature in enumerate(F):
        safety_data.append({
            'slide': i + 1,
            'phi3d': feature.get('phi3d', 0.0),
            'c3d': feature.get('c3d', 0.0),
            'rot3d': feature.get('rot3d', 0.0),
            'FS2D_by_phi3d_c3d': safe_float(feature.get('FS2Dby3D', 0.0)),
            'phi2d': feature.get('phi2d', 0.0),
            'c2d': feature.get('c2d', 0.0),
            'cell_count': feature.get('cell_count', 0),
            'skip_reason': feature.get('skip_reason', "")
        })
    safety_csv_path = os.path.join(outPath, 'results.csv')
    safety_df = pd.DataFrame(safety_data)
    safety_df.to_csv(safety_csv_path, index=False)
    
    end = time.time()
    print("Elapsed time:", end - start)

def _build_cli():
    """argparse front-end for ``main`` (also used by the Streamlit GUI)."""
    import argparse
    p = argparse.ArgumentParser(
        description="Simplified Janbu method - 3D/2D back & forward analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--poly', dest='poly_shp', default=None,
                   help="Landslide polygon shapefile.")
    p.add_argument('--slip', dest='slip_tif', default=None,
                   help="Slip-surface raster (.tif).")
    p.add_argument('--dem', dest='dem_tif', default=None,
                   help="Ground-surface DEM raster (.tif). Exactly one DEM is "
                        "needed; use --fail-type to label what it represents.")
    p.add_argument('--out', dest='out_path', default=None,
                   help="Output directory.")
    p.add_argument('--phi-init', dest='phi_thresh', type=float, default=1.0,
                   help="Initial guess for phi [deg].")
    p.add_argument('--c-init', dest='c_thresh', type=float, default=1.0,
                   help="Initial guess for cohesion [kN/m^2].")
    p.add_argument('--gw', type=float, default=9.8,
                   help="Unit weight of water [kN/m^3].")
    p.add_argument('--gd', type=float, default=16.0,
                   help="Unit weight of dry soil [kN/m^3].")
    p.add_argument('--gs', type=float, default=20.0,
                   help="Unit weight of saturated soil [kN/m^3].")
    p.add_argument('--ru', type=float, default=0.25,
                   help="Pore-pressure ratio (used only when --water-mode Ru).")
    p.add_argument('--water-mode', dest='water_mode',
                   choices=('Ru', 'GL'), default='Ru',
                   help="Pore-pressure parameterisation. 'Ru' (legacy) or "
                        "'GL' (uniform groundwater depth below ground).")
    p.add_argument('--water-depth', dest='water_depth_GL', type=float, default=0.0,
                   help="Groundwater depth below ground [m] (only when --water-mode GL).")
    p.add_argument('--kx', type=float, default=0.0)
    p.add_argument('--ky', type=float, default=0.0)
    p.add_argument('--Ex', type=float, default=0.0)
    p.add_argument('--Ey', type=float, default=0.0)
    p.add_argument('--pga-raster', dest='pga_raster', default=None,
                   help="Path to a PGA raster (.tif). When supplied, per-cell "
                        "ky = PGA * --pga-scaling overrides scalar --ky.")
    p.add_argument('--pga-scaling', dest='pga_scaling', type=float, default=1.0,
                   help="Multiplier applied to PGA raster values.")
    p.add_argument('--strength', choices=('phi', 'c'), default='phi',
                   help="Which strength parameter to back-solve.")
    p.add_argument('--fail-type', dest='fail_type',
                   choices=('Progressive', 'Catastrophic'), default='Progressive',
                   help="Metadata label only: Progressive = DEM is the current "
                        "ground surface, Catastrophic = DEM is the pre-failure "
                        "top surface. Does not affect the math.")
    p.add_argument('--no-2d', dest='skip_2d', action='store_true', default=False,
                   help="Skip the 2D cross-section back/forward analysis (only "
                        "3D rot/phi/c are produced). Halves per-slide cost.")
    p.add_argument('--jobs', dest='n_jobs', type=int, default=1,
                   help="Number of polygon-level workers. 1 = serial, "
                        "-1 = all available cores.")
    p.add_argument('--backend', dest='parallel_backend',
                   choices=('threading', 'loky'), default='threading',
                   help="joblib backend when --jobs != 1. 'threading' is "
                        "lightweight but limited by the GIL (mostly helpful "
                        "for large per-polygon JIT work). 'loky' uses true "
                        "processes (better speedup for many polygons, but "
                        "with ~1-2s process startup overhead on Windows).")
    p.add_argument('--limit', dest='limit', type=int, default=None,
                   help="Process only the first N polygons (smoke / verification).")
    return p


if __name__ == "__main__":
    args = _build_cli().parse_args()
    main(**vars(args))
