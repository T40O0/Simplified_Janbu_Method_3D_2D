# Slope Stability: Simplified Janbu Method (3D & 2D)  

 - This is a Python port of the following MATLAB code.  
([Landslide-Forensics](https://github.com/benalesh/Landslide-Forensics))  
Bunn, M., Leshchinsky, B., & Olsen, M. J. (2020). Geologic trends in shear strength properties inferred through three-dimensional back analysis of landslide inventories. Journal of Geophysical Research: Earth Surface, 125, https://doi.org/10.1029/2019JF005461
 - 2D back-and-forward analysis added.

## Requirements
This project requires the following Python packages (with versions tested):
 - geopandas  1.0.1
 - matplotlib 3.10.1
 - numpy      2.2.4
 - pandas     2.2.3
 - rasterio   1.4.3
 - scipy      1.15.2
 - shapely    2.1.0

## Features
 - The sliding direction and c or φ can be calculated by 3D back analysis using the simplified Janbu method.
 - Pseudo-static seismic coefficients (kx, ky) and applied horizontal loads (Ex, Ey) can be applied to the 3D back analysis.
 - 2D back/forward analysis applies the longitudinal pseudo-static coefficient (ky) and applied horizontal load (Ey) along the slip direction. The transverse components (kx, Ex) are intentionally not used in 2D since a single-section longitudinal Janbu balance has no transverse force component; project them onto the slip direction (i.e. pass the projected component as ky / Ey) if your loads are given in world coordinates.
 - 2D back calculation is possible for the sliding direction calculated in 3D.
 - Forward calculation in 2D is possible for the sliding direction, c and φ calculated in 3D.
 - Saves c, φ, FS and sliding direction as histogram, csv and shapefile.

Note:  
 - This code does not include the correction factor f0. Please add it if necessary.
 - Histogram output for c is not currently supported.
 - 2D back/forward analysis is valid for any slip azimuth (rot3d), including east–west and south-facing orientations.
 - The output directory (`output/`) is created automatically on the first run.

## Recent fixes
A second audit against the original MATLAB (Bunn et al. 2020) sources surfaced
the following corrections beyond the initial port:

 - 2D analysis now multiplies the per-cell volume by the unit weight `gs`, so the cohesion / pore-pressure and gravity terms are dimensionally consistent (φ2d / c2d / FS2D were systematically biased before).
 - 2D drive term now includes `ky·W + Ey` so the seismic / external-load arguments are no longer silently ignored.
 - 3D transverse force balance uses the transverse normal `Nx` in `term2x` (the MATLAB reference inherited the same bug, biasing the rot3d search).
 - The outer rot3d search now reproduces the MATLAB direction-reversal heuristic, and the inner FS loop drops a fragile (1.00, 1.10) acceptance window in favour of a clean `FRy ≥ FDy` break.
 - `gradient_king` is fully vectorised; the unused `dz/dx`, `dz/dy` outputs were removed.
 - Various robustness fixes: auto-create output directory, suppress blocking `plt.show()` in batch loops, silence `np.nanmean` warnings, and normalise the GeoDataFrame schema so skip paths do not leave NaN-riddled columns.

## References
Ugai, K., and Hosobori, K. (1988). Extension of simplified Bishop method, simplified Janbu method and Spencer method to three-dimensions. Japanese Soc. of Civ. Engrs., 394, 21–26 (in Japanese).  
Hungr, O., Salgado, F. M., & Byrne, P. M. (1989). Evaluation of a three-dimensional method of slope stability analysis. Canadian Geotechnical Journal, 26(4), 679–686.  
O. Hungr (1987). An extension of Bishop's simplified method of slope stability analysis to three dimensions. Géotechnique, 37(1), 113-117.

## Acknowledgments
Special thanks to Michael Bunn, Ben Leshchinsky, and Michael J. Olsen, whose work (Bunn et al., 2020) laid the foundation for this code. Their original research and MATLAB code were  invaluable in the development of this Python port.

## Licence
This project is licensed under the MIT licence - see the [LICENSE file](LICENSE) for details.

