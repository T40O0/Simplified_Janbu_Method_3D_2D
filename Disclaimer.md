# Disclaimer

This software is distributed under the MIT licence (see [LICENSE](LICENSE))
**without any warranty**. The notes below are technical caveats — not
legal terms — that users should understand before relying on the outputs.

## Method limitations

- Force equilibrium only; moment equilibrium is not enforced.
- The Janbu correction factor `f0` is **not** applied.
- Back-analysis sweeps only one strength parameter at a time
  (`--strength phi` or `--strength c`); joint (phi, c) inference is
  not performed.
- Pore pressure is steady-state (Ru ratio or uniform GL depth).
  Time-dependent flow and unsaturated effects are not modelled.
- Seismic loading is pseudo-static (`K_h = PGA × pseudo_scaling`).

## Inputs and outputs

- Garbage in, garbage out: result accuracy depends on the polygon,
  slip-surface raster, and DEM. All three must share a CRS / grid.
- `phi3d` / `c3d` are **back-calculated** values that bring FS to 1
  under the user-chosen assumptions — not direct soil strength
  measurements. Changing the assumptions changes the inferred values.
- `shear_strength.mat` describes the population of back-calculated
  values, not soil-mechanics ground truth.

## Differences from the reference implementation

This is a Python re-implementation of the MATLAB code published with
Bunn, Leshchinsky & Olsen (2020). Several bugs in the original have
been fixed and the feature set has been extended (see `README.md`).
Outputs may therefore differ from a strict MATLAB run on the same
inputs; publications comparing to Bunn et al. (2020) should cite which
implementation was used.

By using this software you accept full responsibility for any decision
made using its outputs.
