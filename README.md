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
 - streamlit  ≥ 1.30 (only required for the GUI)
 - numba      ≥ 0.59 (optional, used to JIT the inner FS sweep; the code
   falls back to pure numpy if numba is not installed)
 - joblib     ≥ 1.4  (optional, only needed when `--jobs N` with N != 1 is
   used to dispatch polygon-level workers)

## Input data layout
The CLI / GUI scan `input/` recursively for shapefiles and rasters. Sample
data is **not** committed to the repository; place your own copies under
`input/` (the folder is git-ignored). Example layout:

```
input/
├── DEM10.tif                 # single DEM (post-failure or pre-failure - see fail_type)
├── slide.tif                 # slip-surface raster
└── SHP/
    ├── landslide_poly.shp    # landslide polygons (with .dbf / .shx / .prj / .cpg)
    ├── landslide_poly.shx
    ├── landslide_poly.dbf
    ├── landslide_poly.prj
    └── landslide_poly.cpg
```

File names are free-form: the GUI lets you pick any `.shp` / `.tif` under
`input/`, and the CLI accepts arbitrary paths via `--poly`, `--slip`,
`--dem`. The shapefile DBF columns are preserved in the
`output/back_analysis.shp` so you can carry your own attributes through.

The rasters must share a CRS and grid with the polygons; cell size is read
from the slip raster's `transform`.

### About `fail_type`
The `--fail-type` flag (`Progressive` | `Catastrophic`) is a **metadata
label** that records what the supplied DEM represents:

| `fail_type` | DEM meaning | When to choose |
|---|---|---|
| `Progressive` (default) | Post-failure / current ground surface | Back-analysing the geometry of the current deposit, or a slope that has been slowly creeping. |
| `Catastrophic` | Pre-failure top surface | Back-analysing the original mass that failed at a single moment (e.g. a co-seismic slide). Provide a pre-failure DEM. |

The math is identical either way (`G - S` column thickness with `G` = the
supplied DEM). The label is stamped on every output feature
(`back_analysis.shp` / `results.csv`) for record-keeping.

## Usage

### CLI (single run, reproducible)

```bash
python BackAnalysis_3D.py \
    --poly input/SHP/landslide_poly.shp \
    --slip input/slide.tif \
    --dem  input/DEM10.tif \
    --out  output \
    --strength phi --fail-type Progressive
```

Run `python BackAnalysis_3D.py --help` for the full parameter list (initial
phi / c, unit weights, Ru, seismic coefficients kx / ky, external loads
Ex / Ey, etc.). Defaults match the legacy behaviour, so the original
`python BackAnalysis_3D.py` invocation still works if your files are named
`input/landslide_poly.shp`, `input/slide.tif`, `input/DEM10.tif`.

#### Pore-pressure model: Ru or uniform GL depth
Two parameterisations are available via `--water-mode`:

| mode | formula | when to use |
|---|---|---|
| `Ru` (default) | `u = gw * (G - S) * Ru` | Want a single ratio of water-column-height to slip-mass-thickness. |
| `GL`           | `u = gw * max(0, (G - S) - depth)` | Want a uniform groundwater table depth (in metres below ground), e.g. `--water-mode GL --water-depth 2.0`. |

#### Seismic input: scalar or PGA raster
By default `--ky` / `--kx` are scalars applied uniformly. Supplying a
co-registered PGA raster instead overrides the per-cell longitudinal
coefficient: `--pga-raster path/to/PGA.tif --pga-scaling 1.0`. Each cell's
effective `ky` becomes `PGA_cell * pga_scaling` (i.e. `K_h = PGA × scaling`);
`kx` remains scalar. If the PGA raster is on a different grid, it is
nearest-neighbour resampled onto the slip-surface grid internally.

#### Other CLI flags

| Flag | Default | Effect |
|---|---|---|
| `--no-2d` | off | Skip the 2D cross-section back/forward analysis. Useful when only 3D `phi3d` / `c3d` / `rot3d` are needed. `phi2d`, `c2d`, `FS2D_by_phi3d_c3d` are left at `0` in the output; the 2D polyline shapefile and phi2d histogram are not written. |
| `--limit N` | unlimited | Process only the first `N` polygons. Intended for verification / smoke runs. |
| `--jobs N` | `1` | Polygon-level parallel workers via joblib. `1` = serial (recommended for typical inputs), `-1` = all cores. On the sample dataset (small per-polygon cost) any value > 1 is slower than serial; the knob is left in for large workloads with heavy polygons. |
| `--backend {threading, loky}` | `threading` | joblib backend when `--jobs != 1`. `threading` is lightweight but limited by the GIL. `loky` uses real processes; helpful only if the per-polygon work is large enough to amortise the Windows process-startup cost (~1-2 s per worker). |

### Output files

A complete run writes the following into the `--out` directory:

| File | Always | Description |
|---|---|---|
| `results.csv` | yes | One row per slide with `phi3d`, `c3d`, `rot3d`, `phi2d`, `c2d`, `FS2D_by_phi3d_c3d`, `cell_count`, `skip_reason` |
| `back_analysis.shp` | yes | Input polygons plus the back-analysis fields (`phi3d`, `c3d`, `rot3d`, `fail_type`, etc.) |
| `phi3d_hist.png` / `phi3d_hist.csv` | yes | 3D back-analysed-φ distribution (PDF + bin metadata) |
| `shear_strength.mat` | yes | 3D back-analysed strength distribution as a PMF in the schema expected by **RegionGrow3D** (Mathews et al., 2024 — see References below) and similar regional susceptibility tools. Three 1-D float64 arrays of length N: `prob` (Σ ≈ 1.0), `prob_phi` [deg], `prob_coh` [kPa]. When `--strength c` is used the roles of φ / c swap (varying coh, constant phi). Loadable with `scipy.io.loadmat`. |
| `phi2d_hist.png` / `phi2d_hist.csv` | unless `--no-2d` | 2D back-analysed-φ distribution |
| `2D_stability_polyline.shp` | unless `--no-2d` | Per-slide polyline of the deepest cross-section used for the 2D analysis |

### Streamlit GUI

```bash
streamlit run gui.py
```

The browser UI walks `input/` for file pickers, exposes every solver
parameter, and shells out to `BackAnalysis_3D.py`'s CLI as a detached
subprocess. The run keeps going if you close the browser; reopening
`gui.py` reattaches via the manifest at `output/.gui_manifest.json`.
Results, histograms, and downloadable CSVs appear in tabs when the run
finishes.

## Features
 - The sliding direction and c or φ can be calculated by 3D back analysis using the simplified Janbu method.
 - Pseudo-static seismic coefficients (kx, ky) and applied horizontal loads (Ex, Ey) can be applied to the 3D back analysis.
 - Pseudo-static seismic coefficients can also be supplied as a **PGA raster** (per-cell `K_h = PGA × pseudo_scaling`).
 - Pore-pressure can be parameterised either as the legacy `Ru` ratio or as a **uniform groundwater depth below ground** (`--water-mode GL --water-depth m`).
 - 2D back/forward analysis applies the longitudinal pseudo-static coefficient (ky) and applied horizontal load (Ey) along the slip direction. The transverse components (kx, Ex) are intentionally not used in 2D since a single-section longitudinal Janbu balance has no transverse force component; project them onto the slip direction (i.e. pass the projected component as ky / Ey) if your loads are given in world coordinates.
 - 2D back calculation is possible for the sliding direction calculated in 3D.
 - Forward calculation in 2D is possible for the sliding direction, c and φ calculated in 3D.
 - Saves c, φ, FS and sliding direction as histogram, csv and shapefile. The 3D back-analysis distribution is also exported as a **`shear_strength.mat` PMF** in the schema consumed by USGS **RegionGrow3D** (Mathews et al., 2024) — see References below.

Note:  
 - This code does not include the correction factor f0. Please add it if necessary.
 - Histogram output for c is not currently supported.
 - 2D back/forward analysis is valid for any slip azimuth (rot3d), including east–west and south-facing orientations.
 - The output directory (`output/`) is created automatically on the first run.

## Performance

Measured wall-clock time on the bundled-style sample (sample polygon set,
2801 × 2251 raster), Numba JIT cache hot:

| Polygons | Time | Notes |
|---|---|---|
| 5    | ~2.7 s   | Mostly fixed overhead (raster I/O + matplotlib init + Numba JIT compile cache check) |
| 30   | ~7.3 s   | |
| 100  | ~9.5 s   | |
| 200  | ~10.1 s  | ≈ 5 s of per-polygon work amortised across the batch |

The bottleneck used to be `shapely.contains_xy` over the **full DEM**
(6.3 M cells) per polygon; the current implementation point-in-polygon
tests only the polygon's pixel-index bounding box and walks the inner
sweep through a Numba-JIT helper. As a result the per-polygon cost is
small (~16–40 ms) and most of the wall time on small batches is fixed
startup. Polygon-level parallel workers (`--jobs > 1`) typically do **not**
help at this scale and may be slower than serial.

## Recent fixes
A second audit against the original MATLAB (Bunn et al. 2020) sources surfaced
the following corrections beyond the initial port:

 - 2D analysis now multiplies the per-cell volume by the unit weight `gs`, so the cohesion / pore-pressure and gravity terms are dimensionally consistent (φ2d / c2d / FS2D were systematically biased before).
 - 2D drive term now includes `ky·W + Ey` so the seismic / external-load arguments are no longer silently ignored.
 - 3D transverse force balance uses the transverse normal `Nx` in `term2x` (the MATLAB reference inherited the same bug, biasing the rot3d search).
 - The outer rot3d search now reproduces the MATLAB direction-reversal heuristic, and the inner FS loop drops a fragile (1.00, 1.10) acceptance window in favour of a clean `FRy ≥ FDy` break.
 - `gradient_king` is fully vectorised; the unused `dz/dx`, `dz/dy` outputs were removed.
 - Various robustness fixes: auto-create output directory, suppress blocking `plt.show()` in batch loops, silence `np.nanmean` warnings, and normalise the GeoDataFrame schema so skip paths do not leave NaN-riddled columns.

## Recent optimisations

A profile-driven optimisation pass on top of the fixes above gives an
~11–22× speed-up on the sample dataset depending on polygon count, while
keeping numerical results within ~2.3 × 10⁻⁸ of the legacy baseline:

 - **bbox-clipped point-in-polygon**: `create_shp_mask` now tests only the
   polygon's pixel-index bounding box instead of the full 6.3 M-cell raster.
 - **bbox-only sub-grid extraction**: the secondary `xmask & ymask`
   full-grid Boolean pass in `main()` is replaced by an affine-transform
   bbox calculation.
 - **A + B caching**: per-cell trig / `Atb` / `gz` and polygon-constant
   masked arrays are computed once per polygon (not per `asp_shift`).
 - **Numba JIT inner sweep**: the 1-deg phi (or c) walk is compiled to
   native code via `@njit(cache=True)`; the polygon-constant arrays are
   pre-broadcast so the JIT call has stable signatures.
 - Optional **joblib polygon-level parallelism** (`--jobs N --backend
   {threading, loky}`) — left in as an opt-in, but on this dataset class
   the per-polygon cost is small enough that any value > 1 is slower than
   serial. CLI-only; the GUI does not expose it.

## References
Ugai, K., and Hosobori, K. (1988). Extension of simplified Bishop method, simplified Janbu method and Spencer method to three-dimensions. Japanese Soc. of Civ. Engrs., 394, 21–26 (in Japanese).  
Hungr, O., Salgado, F. M., & Byrne, P. M. (1989). Evaluation of a three-dimensional method of slope stability analysis. Canadian Geotechnical Journal, 26(4), 679–686.  
O. Hungr (1987). An extension of Bishop's simplified method of slope stability analysis to three dimensions. Géotechnique, 37(1), 113-117.  
Mathews, N., Leshchinsky, B., Mirus, B., Olsen, M., & Booth, A. (2024). RegionGrow3D: A Deterministic Analysis for Characterizing Discrete Three‐Dimensional Landslide Source Areas on a Regional Scale. *Journal of Geophysical Research: Earth Surface*, 129, e2024JF007815. https://doi.org/10.1029/2024JF007815  
  – USGS software page: https://www.usgs.gov/software/regiongrow3d  
  – Source: https://code.usgs.gov/ghsc/lhp/regiongrow3d

## Acknowledgments
Special thanks to Michael Bunn, Ben Leshchinsky, and Michael J. Olsen, whose work (Bunn et al., 2020) laid the foundation for this code. Their original research and MATLAB code were  invaluable in the development of this Python port.

## Licence
This project is licensed under the MIT licence - see the [LICENSE file](LICENSE) for details.

