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
    """
    rows, cols = shape
    X, Y = worldGrid(transform, shape)
    mask = contains_xy(geom, X.ravel(), Y.ravel())
    return mask.reshape(rows, cols)  # 明示的に行数、列数を指定

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
    
    dy negative: south  positive: north  values are rise/run
    dx negative: west   positive: east   values are rise/run
    Aspect : array of aspect (direction) angles for each cell[deg]
             starts north and rotates clockwise
    Slope  : array of slope angles for each cell[deg]
    """
    m, n = Elevation.shape
    Slope = np.zeros((m, n))
    Aspect = np.zeros((m, n))
    dx = np.zeros((m, n))
    dy = np.zeros((m, n))
    
    for j in range(1, m-1):
        for k in range(1, n-1):
            #Perform gradient calculations like in ArcGIS
            dz_dx = (((Elevation[j-1, k+1]) + 2 * (Elevation[j, k+1]) + (Elevation[j+1, k+1]) -
                      (Elevation[j-1, k-1]) - 2 * (Elevation[j, k-1]) - (Elevation[j+1, k-1]))
                     / (8 * csize))
            dz_dy = (((Elevation[j+1, k-1]) + 2 * (Elevation[j+1, k]) + (Elevation[j+1, k+1]) -
                      (Elevation[j-1, k-1]) - 2 * (Elevation[j-1, k]) - (Elevation[j-1, k+1]))
                     / (8 * csize))
            # Compute Slope
            rise_run = np.sqrt(dz_dx**2 + dz_dy**2)
            Slope[j, k] = np.degrees(np.arctan(rise_run))
            
            # Aspect calculation with corrections matching the MATLAB version
            aspect = np.degrees(np.arctan2(dz_dy, -dz_dx))
            if aspect < 0:
                cell_val = 90 - aspect
            elif aspect > 90:
                cell_val = 360 - aspect + 90
            else:
                cell_val = 90 - aspect
            Aspect[j, k] = cell_val
            
            # Package dx and dy
            dy[j, k] = dz_dy
            dx[j, k] = -dz_dx

    return Slope, Aspect, dx, dy

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

    iter2 = 0 # FILL IN
    ias = 1 # FILL IN
    asp0 = asp # Record initial aspect so that asp may be manipulated

    ROT_3d = []
    FDx_list_3d = []
    MAX_OUTER_ITER = 359
    
    while switchA * switchB == 0:
        # Check the upper limit of the outer loop.
        if abs(asp_shift) >= MAX_OUTER_ITER:
            print(f" [Warning] The maximum search angle ({MAX_OUTER_ITER}°)has been reached. Stop the external iteration.")
            return asp, phi0, c0  # Return the initial value if convergence fails.
            break
        # Apply aspect shift to aspect (0 for attempt 1)
        asp_current = asp0 + asp_shift
        # Project slope vectors into longitudinal (failure
        # direction) and transverse directions - simple dot product
        dlon = Aspect - asp_current # Difference between each pixel's aspect
                                    # and the longitudinal direction
        dtra = (Aspect + 90) - asp_current # Difference between each pixel's
                                           # aspect and the transverse direction
        dy_vals = Slope * np.cos(np.deg2rad(dlon)) # longitudinal pixel slopes
        dx_vals = Slope * np.cos(np.deg2rad(dtra)) # transverse pixel slopes
        # Compute basic geometric properties of each column
        # Compute area of column's true base
        Atb = (csize * csize) * np.sqrt(
            1 - (np.sin(np.deg2rad(dx_vals))**2 * np.sin(np.deg2rad(dy_vals))**2)
        ) / (np.cos(np.deg2rad(dx_vals)) * np.cos(np.deg2rad(dy_vals)))
        # Compute local dip of sliding surface
        gz = np.sqrt(1 / (np.tan(np.deg2rad(dy_vals))**2 +
                          np.tan(np.deg2rad(dx_vals))**2 + 1))
        
        # Inner loop: adjust phi3d or c3d until FS ≈ 1
        phi3d = phi0
        c3d = c0
        FSy = 1.0
        FRy = 0
        FDy = 1.0
        iter_count = 0
        FSY_hist = []
        if st == 1:
            phi3d_hist = []
        else:
            c3d_hist = []
        maxC = c0 + 50 # max cohesion [kN/m2]
        maxPhi = 90    # max friction angle [deg]
        inner_loop_success = False  # inner loop flag
        
        while (FRy < FDy) or (FRy < 0 ) or (FDy <0):
            if st == 1:
                phi3d += phi_inc
                if phi3d >= maxPhi:
                    # print(f"[Warning] φ has reached its upper limit ({maxPhi}°). Stop inner iteration.")
                    break  # Stop iteration
            else:
                c3d += c_inc
                if c3d >= maxC:
                    # print(f"[Warning] c has reached its upper limit ({maxC} kN/m^2). Stop inner iteration.")
                    break
            iter_count += 1

            ## Compute LONGITUDINAL stability
            # Compute normal force at base of column (Assume FS = 1)

            md = gz * (1 + (np.sin(np.deg2rad(dy_vals)) * np.tan(np.deg2rad(phi3d))) / (FSy*gz))
            N = (W - c3d * Atb * np.sin(np.deg2rad(dy_vals)) / FSy +
                 u * Atb * np.tan(np.deg2rad(phi3d)) * np.sin(np.deg2rad(dy_vals)) / FSy) / md

            # Compute FS
            """
            # Bunn(2020)######################################################
            # https://doi.org/10.1029/2019JF005461
            term1 = c3d * Atb * gz + (N - u * Atb) * np.tan(np.deg2rad(phi3d)) * np.cos(np.deg2rad(dy_vals))
            term2 = N * gz * np.tan(np.deg2rad(dy_vals))
            term3 = ky * W + Ey
            
            FRy = np.sum(term1[mask_red])
            FDy = np.sum(term2[mask_red]) + np.sum(term3[mask_red])
            FSY = np.sum(term1[mask_red]) / (np.sum(term2[mask_red]) + np.sum(term3[mask_red]) + 1e-10)
            ####################################################################
            """
            # UGAI(1988)######################################################
            # https://doi.org/10.2208/jscej.1988.394_21
            At = (csize * csize)
            term1 = c3d * At - u * At * np.tan(np.deg2rad(phi3d)) + W * np.tan(np.deg2rad(phi3d))
            term1 = term1 / np.cos(np.deg2rad(dy_vals))/md
            term3 = W * np.tan(np.deg2rad(dy_vals))
            
            FRy = np.sum(term1[mask_red])
            FDy = np.sum(term3[mask_red])
            FSY = np.sum(term1[mask_red]) / (np.sum(term3[mask_red]) + 1e-10)
            ####################################################################
            
            FSY_hist.append(FSY)
            if st == 1:
                phi3d_hist.append(phi3d)
            else:
                c3d_hist.append(c3d)

            # Identify correct factor of safety
            # Transverse (x) safety factor evaluation
            FSx1 = 100
            mdx = gz * (1 + (np.sin(np.deg2rad(dx_vals)) * np.tan(np.deg2rad(phi3d))) / (FSx1 * gz))
            Nx = (W - c3d * Atb * np.sin(np.deg2rad(dx_vals)) / FSx1 +
                  u * Atb * np.tan(np.deg2rad(phi3d)) * np.sin(np.deg2rad(dx_vals)) / FSx1) / mdx
            term1x = c3d * Atb * gz + (Nx - u * Atb) * np.tan(np.deg2rad(phi3d)) * np.cos(np.deg2rad(dx_vals))
            term2x = N * gz * np.tan(np.deg2rad(dx_vals))
            term3x = kx * W + Ex
            FSx2 = np.sum(term1x[mask_red]) / (np.sum(term2x[mask_red]) + np.sum(term3x[mask_red]) + 1e-10)
            
            # Run equation again to obtain driving forces at correct FS
            mdx = gz * (1 + (np.sin(np.deg2rad(dx_vals)) * np.tan(np.deg2rad(phi3d))) / (FSx2 * gz))
            Nx = (W - c3d * Atb * np.sin(np.deg2rad(dx_vals)) / FSx1 +
                  u * Atb * np.tan(np.deg2rad(phi3d)) * np.sin(np.deg2rad(dx_vals)) / FSx1) / mdx
            term1x = c3d * Atb * gz + (Nx - u * Atb) * np.tan(np.deg2rad(phi3d)) * np.cos(np.deg2rad(dx_vals))
            term2x = N * gz * np.tan(np.deg2rad(dx_vals))
            term3x = kx * W + Ex
            #FSx2 = np.sum(term1x[mask_red]) / (np.sum(term2x[mask_red]) + np.sum(term3x[mask_red]) + 1e-10
            #FRx = np.sum(term1x[mask_red])
            FDx = np.sum(term2x[mask_red]) + np.sum(term3x[mask_red])
            
            # If the stability conditions are satisfied, mark the inner loop as successful
            if (FRy >= FDy) and (FRy >= 0) and (FDy >= 0) and (FSY < 1.1) and (FSY >1.00) :
                inner_loop_success = True
                break

        # If the inner loop fails to converge, skip this direction and try the next one
        if not inner_loop_success:
            if iter2 == 0:
                # first try: + direction
                asp_shift += asp_inc
                iter2 += 1
                continue
            elif iter2 == 1:
                # second try: – direction
                asp_shift = -asp_inc
                iter2 += 1
                ias = -1  # reverse the shift direction
                continue
            else:
                #print("iter2: False")
                #print(iter2,"FRy: ",FRy, "FDy: ", FDy,"FSY: ", FSY, " FDx: ", FDx, " A: ", switchA, " B: ", switchB, "asp_current: ", asp_current) 
                asp_shift += ias * asp_inc
                iter2 += 1
                continue

        if iter_count > 1:
            if st == 1:
                phi3d_final = np.interp(1, [FSY_hist[-2], FSY_hist[-1]], [phi3d_hist[-2], phi3d_hist[-1]])
                c3d_final = c3d
            else:
                c3d_final = np.interp(1, [FSY_hist[-2], FSY_hist[-1]], [c3d_hist[-2], c3d_hist[-1]])
                phi3d_final = phi3d
        else:
            if st == 1:
                phi3d_final = phi3d - phi_inc
                c3d_final = c3d
            else:
                c3d_final = c3d - c_inc
                phi3d_final = phi3d
                
        FDx_list_3d.append(FDx)
        ROT_3d.append(asp_shift)
        iter2 += 1
        
        if (len(ROT_3d) >= 2) and (np.abs(ROT_3d[-2]-ROT_3d[-1])!=1):
            if FDx > 0:
                switchA = 1
                switchB = 0
            else:
                switchA = 0
                switchB = 1
        else:
            if FDx > 0:
                switchA = 1
            else:
                switchB = 1
            
        #print(iter2,"FRy: ",FRy, "FDy: ", FDy,"FSY: ", FSY, " FDx: ", FDx, " A: ", switchA, " B: ", switchB, "asp_current: ", asp_current) 
        asp_shift += ias * asp_inc
        
        if switchA * switchB != 0:
            break

    # If neither transverse‐force condition is satisfied (switchA * switchB == 0)
    if switchA * switchB == 0:
        print("[Warning] could not find a transverse‐force balance in either direction.")

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
      W0:           volume weight per cell
      u:            Effective pore pressure per cell (or equivalent)
      gs, kx, ky,
      Ex, Ey:       Other parameters (not used here; no seismic support)
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
    
    # --------------------------------------------------------------
    # 2. Slice width
    # --------------------------------------------------------------
    AtbH = csize /np.cos(np.deg2rad(rot3d))
    
    # --------------------------------------------------------------
    # 3. Select cells for 2D analysis
    # --------------------------------------------------------------
    # Warnings are generated under the following conditions
    # ・When longest_mask is all False (no cross section candidates can be extracted)
    # ・When there are many missing values (NaN) in W0 and no valid cells remain.
    # ・When there are too many missing values (NaN) in dy_vals and no valid cells remain.
    # ・When all W0 values are less than or equal to 0
    # ・when all dy_vals values are less than or equal to 0

    valid_cells = longest_mask & ~np.isnan(W0) & ~np.isnan(dy_vals) & (W0 > 0) & (dy_vals > 0)
    if np.sum(valid_cells) == 0:
        print("[Warning] No valid cells available") 
        return phi0, c0

    # --------------------------------------------------------------
    # 4. Calculate the factor of safety FS (calc_fs)
    # --------------------------------------------------------------
    def calc_fs(phi_val, c_val):
        # Calculate the Factor of Safety FS using the internal
        # friction angle phi_val [deg] and c_val[kN/m2] 
        fs_est = 1.0  
        for _ in range(30):
            sinA = np.sin(np.deg2rad(dy_vals))
            cosA = np.cos(np.deg2rad(dy_vals))
            tanA = np.tan(np.deg2rad(dy_vals))
            tan_phi = np.tan(np.deg2rad(phi_val))
            
            denom = cosA * cosA * (1.0 + (tan_phi * tanA)/ fs_est)
            denom = np.where(np.abs(denom) < 1e-6, 1e-6, denom)
            resist_term = c_val * AtbH + (W0 - u * AtbH) * tan_phi
            resist_term = resist_term / denom
            drive_term = W0 * tanA
            sum_drive = np.sum(drive_term[valid_cells])
            if sum_drive < 1e-10:
                return float('inf')
            fs_new = np.sum(resist_term[valid_cells]) / sum_drive
            
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
        If phi3d (here phi0) does not converge at 1.0 dspite “not insufficient strength” 
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
            print(f"[Warning] There is no condition satisfying FS=1 within the search interval  [{phi_lower}, {phi_upper}] of φ")
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
            print(f"[Warning] There is no condition satisfying FS=1 within the search interval  [{c_lower}, {c_upper}] of c")
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
        
        
    import matplotlib.pyplot as plt   
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
        for seq in seqs:
            mean_d = np.nanmean(dval[seq])
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

def main():
    start = time.time()
    
    # Specify model inputs
    phi_thresh = 1
    c_thresh = 1
    gw = 9.8      # [kN/m^3]
    gd = 16.0     # [kN/m^3]
    gs = 20.0     # [kN/m^3]
    gi = (gd + gs) / 2
    ru = 0.25

    ky = 0.00
    kx = 0
    Ey = 0
    Ex = 0

    # Read input data
    inPath = 'input'
    outPath="output"
    # Read landslide extents (use attributes.shp for aspect)
    dep_shp = os.path.join(inPath, 'landslide_poly.shp') # <------ input
    F_gdf = gpd.read_file(dep_shp)
    F = F_gdf.to_dict('records')
    print(f"[INFO] Shapefile '{dep_shp}' has been read. (1/5)")
    
    # Name output extents
    ba_shp = os.path.join(outPath, 'back_analysis.shp') # <------ input
    
    # Read slip surface DEM
    slip_surf = os.path.join(inPath, 'slide.tif') # <------ input
    with rasterio.open(slip_surf) as src:
        Slip = src.read(1)
        transform = src.transform
        csize = src.res[0]  # Assume square cells
    print(f"[INFO] TIF file '{slip_surf}' has been read. (2/5)")

    # Read ground surfaces（Progressive）
    dem_surf = os.path.join(inPath, 'DEM10.tif') # <------ input
    with rasterio.open(dem_surf) as src:
        DEM = src.read(1)
    print(f"[INFO] TIF file '{dem_surf}' has been read. (3/5)")
    
    # TOP also uses the same file (modified as needed)
    top_surf = os.path.join(inPath, 'DEM10.tif') # <------ input
    with rasterio.open(top_surf) as src:
        TOP = src.read(1)
    print(f"[INFO] TIF file '{top_surf}' has been read. (4/5)")
    
    # Calculate slip surface slope, aspect（gradient_king function）
    SLOPE, ASPECT, DX_arr, DY_arr = gradient_king(Slip, csize)
    shape_img = Slip.shape  # (rows, cols)
    X, Y = worldGrid(transform, shape_img)
    print(f"[INFO] Slip surface slope calculation and grid creation completed. (5/5)")
    
    
    # List to store continuous cross sections (converted to polylines) used in 2D analysis.
    polyline_features = []
    
    total_features = len(F)#[47:48]) 

    for idx, feature in enumerate(F):#[47:48]):
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
                print(f"Slide {idx+1}: No data within the mask (skipped)”)
                feature['skip_reason'] = "No data within the mask"
                feature['c3d']    = 0.0
                feature['phi3d']  = 0.0
                feature['rot3d']  = 0.0
                feature['c2d']    = 0.0
                feature['phi2d']  = 0.0
                feature['FS2D']   = 0.0
                continue
            
            # Range with 50 units of buffer added
            xmin, xmax = np.min(x_inside) - 50, np.max(x_inside) + 50
            ymin, ymax = np.min(y_inside) - 50, np.max(y_inside) + 50
            
            # Create Boolean mask of target area from entire grid 
            xmask = (X >= xmin) & (X <= xmax)
            ymask = (Y >= ymin) & (Y <= ymax)
            XY_mask = (xmask & ymask).reshape(rows, cols)
            indices = np.argwhere(XY_mask)
            if indices.size == 0:
                print(f"Slide {idx+1}: Target area not found (skipped)")
                feature['skip_reason'] = "Target area not found"
                feature['c3d']    = 0.0
                feature['phi3d']  = 0.0
                feature['rot3d']  = 0.0
                feature['c2d']    = 0.0
                feature['phi2d']  = 0.0
                feature['FS2D']   = 0.0
                continue
                    
            XYminr = np.min(indices[:, 0])
            XYmaxr = np.max(indices[:, 0])
            XYminc = np.min(indices[:, 1])
            XYmaxc = np.max(indices[:, 1])
            
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
                continue
            
            # Extract sub-regions
            S = Slip[XYminr:XYmaxr+1, XYminc:XYmaxc+1]  # Slip surface
            fail_type = 'Progressive'
            if fail_type == 'Progressive':
                G = DEM[XYminr:XYmaxr+1, XYminc:XYmaxc+1] # Ground surface DEM
            else:
                G = TOP[XYminr:XYmaxr+1, XYminc:XYmaxc+1]
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
                continue

            # Hydraulic head
            u_i_val = gw * (G - S) * ru
            
            # 3D back analysis
            asp = 0
            try:
                rot3d, phi3d, c3d = SimpJanbu3D(mask_red, csize, Slope_local, Aspect_local,
                                                 asp, c_thresh, phi_thresh, W0, u_i_val, gi,
                                                 'phi', kx, ky, Ex, Ey)
            except Exception as e:
                print(f"Slide {idx+1}: SimpJanbu3D error (skipped): {e}")
                feature['skip_reason'] = f"SimpJanbu3D error: {e}"
                feature['c3d']    = 0.0
                feature['phi3d']  = 0.0
                feature['rot3d']  = 0.0
                feature['c2d']    = 0.0
                feature['phi2d']  = 0.0
                feature['FS2D']   = 0.0
                continue

            feature['c3d']    = c3d
            feature['phi3d']  = phi3d
            feature['rot3d']  = rot3d
                    
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
                continue

            if np.sum(longest_mask) == 0:
                print(f"Slide {idx+1}: longest_mask is empty (skipped)")
                feature['skip_reason'] = "longest_mask is empty"
                feature['c2d']    = 0.0
                feature['phi2d']  = 0.0
                feature['FS2D']   = 0.0
                continue

            # 2D back analysis
            try:
                phi2d, c2d = SimpleJanbu2D_slice(longest_mask, csize, Slope_local, Aspect_local,
                                                  rot3d, c_thresh, phi_thresh, W0, u_i_val, gi,
                                                  'phi', kx, ky, Ex, Ey, "inverse")
            except Exception as e:
                print(f"Slide {idx+1}: SimpleJanbu2D_slice_inverse error: {e}")
                feature['skip_reason'] = f"SimpleJanbu2D_slice_inverse error: {e}"
                feature['c2d']    = 0.0
                feature['phi2d']  = 0.0
                feature['FS2Dby3D']   = 0.0
                continue

            feature['c2d']    = c2d
            feature['phi2d']  = phi2d

            # 2D forward analysis
            try:
                FS2Dby3D = SimpleJanbu2D_slice(longest_mask, csize, Slope_local, Aspect_local,
                                           rot3d, c3d, phi3d, W0, u_i_val, gi,
                                           'phi', kx, ky, Ex, Ey, "fs")
            except Exception as e:
                print(f"Slide {idx+1}: SimpleJanbu2D_slice_fs error(skipped): {e}")
                feature['skip_reason'] = f"SimpleJanbu2D_slice_fs error: {e}"
                feature['FS2Dby3D']   = 0.0
                continue

            feature['FS2Dby3D'] = safe_float(FS2Dby3D)
            feature['skip_reason'] = ""

            # Polyline
            # Connect cells with polyline
            indices = np.argwhere(longest_mask)
            pts = [(sub_X[i,j], sub_Y[i,j]) for (i,j) in indices]
            # θ = azimuth abgle of north reference CW
            # (sinθ,cosθ) is the unit vector in that direction
            theta_rad = np.deg2rad(-rot3d+90)
            ux = np.sin(theta_rad)
            uy = np.cos(theta_rad)
            
            feature['cell_count'] = len(pts)
            
            if len(pts) >= 2:
                # Calculate the projected value in the sliding direction
                projs = np.array([x*ux + y*uy for x,y in pts])
                order = np.argsort(projs)
                sorted_pts = [pts[k] for k in order]
                line = LineString(sorted_pts)

                polyline_features.append({
                    'slide':      idx+1,
                    'FS2Dby3D':   safe_float(FS2Dby3D),
                    'phi3d':      phi3d,
                    'cell_count': len(sorted_pts),
                    'geometry':   line
                })
            print(f"Slide {idx+1} processed successfully.")

        except Exception as e:
            print(f"Slide {idx+1}: Error occurred. Skipping... {e}")
            feature['skip_reason'] = f"Exception: {e}"
            feature['c3d']    = 0.0
            feature['phi3d']  = 0.0
            feature['rot3d']  = 0.0
            feature['FS2Dby3D']   = 0.0
            feature['c2d']    = 0.0
            feature['phi2d']  = 0.0
            continue

    # Save shapefile (back-analysis results)
    F_gdf_out = gpd.GeoDataFrame(F, crs=F_gdf.crs)
    F_gdf_out.to_file(ba_shp)
    
    # Save polyline shapefile '{polyline_shp}')
    polyline_gdf = gpd.GeoDataFrame(polyline_features, geometry='geometry', crs=F_gdf.crs)
    polyline_shp = os.path.join(outPath, '2D_stability_polyline.shp')
    polyline_gdf.to_file(polyline_shp)
    print(f"[INFO] Save polyline shapefile '{polyline_shp}'")
    
    # Create φ3d histogram and output to CSV
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
    # Bin boundaries
    lower_bounds = bin_edges[:-1]
    upper_bounds = bin_edges[1:]
    includes_lower = [True] * len(lower_bounds)
    includes_upper = [False] * len(lower_bounds)
    includes_upper[-1] = True  # Only the last bin contains the upper boundary

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
    
    # Create φ2d histogram and output to CSV
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
    includes_upper2d[-1] = True  # Only the last bin contains the upper boundary

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

if __name__ == "__main__":
    main()
