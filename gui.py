"""Streamlit UI for the Simplified Janbu Method 3D/2D back analysis.

Run from the repo root:
    streamlit run gui.py
or with an explicit env:
    & 'C:\\Users\\040869\\AppData\\Local\\miniconda3\\envs\\gis_conda\\Scripts\\streamlit.exe' run gui.py

The UI shells out to ``BackAnalysis_3D.py`` (its CLI front-end) so a run keeps
going even if the browser is closed.  Job state is persisted to
``output/.gui_manifest.json`` for crash / refresh resilience.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import streamlit as st

REPO = Path(__file__).resolve().parent
DRIVER = REPO / "BackAnalysis_3D.py"
PYTHON_EXE = sys.executable

INPUT_DIR = REPO / "input"
OUTPUT_DIR = REPO / "output"

# Manifest under output/ so the path is predictable and gets included in
# whatever the user is already inspecting.
MANIFEST_ROOT = OUTPUT_DIR
MANIFEST_PATH = MANIFEST_ROOT / ".gui_manifest.json"
LAST_PATH = MANIFEST_ROOT / ".gui_last.json"


# ---------------------------------------------------------------------------
# Manifest / PID utilities (lightweight reimplementation, no external deps)
# ---------------------------------------------------------------------------
def _read_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _clear(path: Path):
    if path.exists():
        try:
            path.unlink()
        except OSError:
            pass


def pid_alive(pid) -> bool:
    if not pid:
        return False
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if os.name == "nt":
        try:
            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                capture_output=True, text=True, timeout=5)
            return str(pid) in (out.stdout or "")
        except Exception:
            return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def kill_pid(pid):
    if not pid:
        return
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                       capture_output=True)
    else:
        import signal
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except Exception:
            try:
                os.kill(pid, signal.SIGTERM)
            except Exception:
                pass


def tail_log(path: Path, max_lines: int = 80) -> list[str]:
    if not path.exists():
        return []
    try:
        # Read the tail without slurping the whole file when it gets large.
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            chunk = min(size, max_lines * 200)
            fh.seek(size - chunk)
            data = fh.read().decode("utf-8", errors="replace")
        return data.splitlines()[-max_lines:]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Simplified Janbu (3D/2D)",
                   layout="wide", initial_sidebar_state="expanded")

SIDEBAR_PX = 480
st.markdown(
    f"""
    <style>
      [data-testid="stSidebar"] {{
        min-width: {SIDEBAR_PX}px !important;
        max-width: {SIDEBAR_PX}px !important;
        width: {SIDEBAR_PX}px !important;
      }}
      [data-testid="stSidebar"] > div:first-child {{
        width: {SIDEBAR_PX}px !important;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Resolve current run state (source of truth = manifest on disk).
_manifest = _read_json(MANIFEST_PATH)
IS_RUNNING = _manifest is not None and pid_alive(_manifest.get("pid"))
DIS = IS_RUNNING

st.sidebar.title("Simplified Janbu 3D/2D")
st.sidebar.caption("Back-analyse phi or c, then forward-check FS in 2D.")
if IS_RUNNING:
    st.sidebar.caption("Locked while running")


# ---- Input file pickers ----------------------------------------------------
st.sidebar.subheader("Input files")

with st.sidebar.expander("Expected input layout", expanded=False):
    st.markdown(
        f"""
| File type | Where the picker looks |
|---|---|
| Polygon shapefile (.shp) | `{INPUT_DIR.relative_to(REPO).as_posix()}/**/*.shp` |
| Slip-surface raster (.tif) | `{INPUT_DIR.relative_to(REPO).as_posix()}/**/*.tif` |
| DEM raster (.tif) | same (a single DEM - label its role below) |

The pickers walk the `input/` tree, so subfolders such as `input/SHP/` are
fine.  You can also paste an absolute path into the text fallback.
"""
    )


def _scan(pattern: str) -> list[Path]:
    if not INPUT_DIR.exists():
        return []
    # Match in lower- AND upper-case extensions because Windows is case-insensitive.
    return sorted({p for p in INPUT_DIR.rglob(pattern)} |
                  {p for p in INPUT_DIR.rglob(pattern.upper())})


def _picker(label: str, files: list[Path], key: str, *,
            help: str | None = None):
    rel = [str(p.relative_to(INPUT_DIR)) for p in files]
    options = ["(choose...)"] + rel + ["[Enter custom path]"]
    selected = st.sidebar.selectbox(label, options, index=1 if rel else 0,
                                    key=f"sel_{key}", disabled=DIS, help=help)
    if selected == "[Enter custom path]":
        custom = st.sidebar.text_input(
            f"  {label} (absolute path)", value="",
            key=f"custom_{key}", disabled=DIS)
        return Path(custom).expanduser() if custom else None
    if selected == "(choose...)":
        return None
    return INPUT_DIR / selected


shp_files = _scan("*.shp")
tif_files = _scan("*.tif")

poly_path = _picker("Landslide polygon (.shp)", shp_files, "poly",
                    help="Polygon defining each landslide. The DBF attribute "
                         "table is preserved in the output.")
slip_path = _picker("Slip surface raster (.tif)", tif_files, "slip",
                    help="Elevation of the failure surface.")
dem_path = _picker("DEM raster (.tif)", tif_files, "dem",
                   help="A single ground / top DEM. Use the radio below to "
                        "label what it represents.")

# fail_type is a metadata label only - the math always uses (DEM - Slip).
fail_type_label = st.sidebar.radio(
    "DEM represents",
    ["Post-failure / current ground surface (Progressive)",
     "Pre-failure top surface (Catastrophic)"],
    index=0, disabled=DIS, key="fail_type_radio",
    help="Recorded on every output feature as `fail_type` for documentation; "
         "does not change the calculation.")
fail_type = "Progressive" if fail_type_label.startswith("Post") else "Catastrophic"


# ---- Solver parameters -----------------------------------------------------
st.sidebar.subheader("Solver")
strength = st.sidebar.radio(
    "Back-solve",
    ["phi (find friction angle at FS=1)",
     "c (find cohesion at FS=1)"], index=0, disabled=DIS, key="strength_radio")
strength = "phi" if strength.startswith("phi") else "c"

c1, c2 = st.sidebar.columns(2)
phi_init = c1.number_input("Initial phi [deg]", 0.0, 89.0, 1.0, 1.0,
                            key="phi_init", disabled=DIS)
c_init = c2.number_input("Initial c [kN/m^2]", 0.0, 200.0, 1.0, 1.0,
                          key="c_init", disabled=DIS)
skip_2d = st.sidebar.checkbox(
    "Skip 2D back/forward analysis (only run 3D)", value=False,
    key="skip_2d", disabled=DIS,
    help="When ticked, no 2D cross-section is extracted; phi2d / c2d / "
         "FS2D_by_phi3d_c3d are left at 0 and the 2D polyline shapefile / "
         "phi2d histogram are not produced. Useful when only the 3D "
         "back-analysis is needed.")

with st.sidebar.expander("Unit weights / pore pressure", expanded=False):
    cw1, cw2, cw3 = st.columns(3)
    gw = cw1.number_input("gw [kN/m^3]", 9.0, 10.5, 9.8, 0.1,
                           key="gw", disabled=DIS)
    gd = cw2.number_input("gd [kN/m^3]", 12.0, 22.0, 16.0, 0.5,
                           key="gd", disabled=DIS)
    gs = cw3.number_input("gs [kN/m^3]", 15.0, 24.0, 20.0, 0.5,
                           key="gs", disabled=DIS)

    water_mode_label = st.radio(
        "Pore-pressure model",
        ["Ru (fraction of slip-mass thickness)",
         "Uniform groundwater depth (GL - m)"],
        index=0, disabled=DIS, key="water_mode_radio",
        help="Ru: u = gw * (G - S) * Ru. GL: u = gw * max(0, (G - S) - depth).")
    if water_mode_label.startswith("Ru"):
        water_mode = "Ru"
        ru = st.slider("Ru", 0.0, 1.0, 0.25, 0.05,
                        key="ru", disabled=DIS,
                        help="Water column height above slip / slip-mass thickness.")
        water_depth_GL = 0.0
    else:
        water_mode = "GL"
        water_depth_GL = st.number_input(
            "Water table depth below ground (GL - m)", 0.0, 100.0, 2.0, 0.5,
            key="water_depth_GL", disabled=DIS,
            help="0 = saturated up to ground surface. Cells whose slip mass "
                 "is shallower than this depth contribute u = 0.")
        ru = 0.25  # ignored in GL mode

with st.sidebar.expander("Seismic / external loads", expanded=False):
    pga_mode = st.radio(
        "Seismic source",
        ["off", "uniform (scalar kx, ky)", "raster (PGA tif)"],
        index=0, disabled=DIS, key="pga_mode_radio")
    pga_raster_path = ""
    pga_scaling = 1.0
    if pga_mode == "uniform (scalar kx, ky)":
        sk1, sk2 = st.columns(2)
        kx = sk1.number_input("kx (transverse)", -0.5, 0.5, 0.0, 0.05,
                               key="kx", disabled=DIS)
        ky = sk2.number_input("ky (longitudinal)", -0.5, 0.5, 0.0, 0.05,
                               key="ky", disabled=DIS,
                               help="Pseudo-static seismic coefficient along "
                                    "slip direction. Also used by the 2D "
                                    "analysis.")
    elif pga_mode == "raster (PGA tif)":
        pga_files = _scan("*.tif") if "_scan" in dir() else []  # may be undefined
        # _scan is defined later in the file; build pickers similarly here.
        pga_options = ["(choose...)"] + [
            str(p.relative_to(INPUT_DIR)) for p in INPUT_DIR.rglob("*.tif")
        ] + ["[Enter custom path]"]
        sel = st.selectbox("PGA raster (.tif)", pga_options, index=0,
                            disabled=DIS, key="pga_sel",
                            help="Co-registered with the slip raster ideally. "
                                 "Mismatched grids are nearest-neighbour "
                                 "resampled internally.")
        if sel == "[Enter custom path]":
            pga_raster_path = st.text_input("PGA raster absolute path", "",
                                              key="pga_custom", disabled=DIS)
        elif sel != "(choose...)":
            pga_raster_path = str(INPUT_DIR / sel)
        pga_scaling = st.slider("PGA scaling factor", 0.0, 2.0, 1.0, 0.05,
                                  key="pga_scaling", disabled=DIS)
        kx = st.number_input("kx (transverse, scalar)", -0.5, 0.5, 0.0, 0.05,
                              key="kx_with_pga", disabled=DIS,
                              help="Per-cell ky comes from the PGA raster; "
                                   "kx remains scalar.")
        ky = 0.0  # overridden per-cell by the raster
    else:
        kx = 0.0
        ky = 0.0

    sE1, sE2 = st.columns(2)
    Ex = sE1.number_input("Ex (transverse) [kN]", -1e6, 1e6, 0.0, 1.0,
                           key="Ex", disabled=DIS, format="%.1f")
    Ey = sE2.number_input("Ey (longitudinal) [kN]", -1e6, 1e6, 0.0, 1.0,
                           key="Ey", disabled=DIS, format="%.1f",
                           help="Applied horizontal load along slip direction. "
                                "Also used by the 2D analysis.")


# ---- Output destination ----------------------------------------------------
st.sidebar.subheader("Output")
out_root = st.sidebar.text_input(
    "Parent directory", value=str(OUTPUT_DIR), key="out_root", disabled=DIS,
    help="Parent folder where results are written. Each run goes into a "
         "sub-folder under this directory.")
test_no = st.sidebar.number_input(
    "test_no (run ID, integer)", 1, 99999, 1, 1, key="test_no", disabled=DIS,
    help="Used as the sub-folder name when 'Run ID / sub-folder name' is "
         "empty (zero-padded to 5 digits, e.g. 00001).")
custom_run_name = st.sidebar.text_input(
    "Run ID / sub-folder name (empty = test_no zero-padded)",
    value="", key="custom_run_name", disabled=DIS,
    help="Free-form sub-folder name. Leave blank to use test_no.")

# Effective sub-folder name (susname) - matches the template's convention.
susname = custom_run_name.strip() if custom_run_name.strip() \
    else f"{int(test_no):05d}"
out_path_preview = Path(out_root) / susname
st.sidebar.caption(f"Output: `{Path(out_root).name}/{susname}/`")

# Overwrite warning - applies only to the susname sub-folder.
folder_dirty = (out_path_preview.exists() and out_path_preview.is_dir()
                and any(p for p in out_path_preview.iterdir()
                        if p.name not in (".gui_manifest.json",
                                          ".gui_last.json")))
if folder_dirty:
    st.sidebar.warning("Output folder is not empty - existing files may be "
                       "overwritten.")
    overwrite_ok = st.sidebar.checkbox(
        "Allow overwrite", value=False, key="overwrite_ok", disabled=DIS)
else:
    overwrite_ok = True


# ---- Run button ------------------------------------------------------------
st.sidebar.subheader("Run")

_missing = [n for n, p in [("polygon", poly_path), ("slip", slip_path),
                            ("DEM", dem_path)] if p is None]
_block = (DIS or (folder_dirty and not overwrite_ok) or bool(_missing))

if _missing:
    _btn_label = f"Pick missing inputs: {', '.join(_missing)}"
elif IS_RUNNING:
    _btn_label = "Running..."
elif folder_dirty and not overwrite_ok:
    _btn_label = "Tick 'Allow overwrite' first"
else:
    _btn_label = "Start analysis"

start = st.sidebar.button(_btn_label, type="primary",
                          use_container_width=True, disabled=_block,
                          key="start_btn")


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------
st.title("Simplified Janbu 3D/2D - Back Analysis")

if "last_status" not in st.session_state:
    st.session_state.last_status = None
if "output_dir" not in st.session_state:
    st.session_state.output_dir = None


def _build_cmd():
    cmd = [PYTHON_EXE, "-u", str(DRIVER),
           "--poly", str(poly_path),
           "--slip", str(slip_path),
           "--dem", str(dem_path),
           "--out", str(out_path_preview),
           "--phi-init", str(phi_init),
           "--c-init", str(c_init),
           "--gw", str(gw), "--gd", str(gd), "--gs", str(gs),
           "--water-mode", water_mode,
           "--ru", str(ru),
           "--water-depth", str(water_depth_GL),
           "--kx", str(kx), "--ky", str(ky),
           "--Ex", str(Ex), "--Ey", str(Ey),
           "--strength", strength,
           "--fail-type", fail_type]
    if pga_raster_path:
        cmd += ["--pga-raster", pga_raster_path,
                "--pga-scaling", str(pga_scaling)]
    if skip_2d:
        cmd += ["--no-2d"]
    # --jobs is intentionally not exposed in the GUI (this dataset class
    # is slower with workers > 1 due to per-polygon overhead).  CLI users
    # can still pass --jobs N --backend threading|loky directly.
    return cmd


# ---- Start: spawn detached subprocess -------------------------------------
if start and not IS_RUNNING:
    out_path_preview.mkdir(parents=True, exist_ok=True)
    log_path = out_path_preview / "_run.log"
    log_path.write_text("", encoding="utf-8")
    log_file = open(log_path, "a", encoding="utf-8", buffering=1)

    cmd = _build_cmd()
    popen_kwargs = dict(stdout=log_file, stderr=subprocess.STDOUT,
                        text=True, cwd=str(REPO), close_fds=True)
    if os.name == "nt":
        popen_kwargs["creationflags"] = (subprocess.CREATE_NEW_PROCESS_GROUP
                                          | subprocess.DETACHED_PROCESS)
    else:
        popen_kwargs["start_new_session"] = True
    proc = subprocess.Popen(cmd, **popen_kwargs)

    _write_json(MANIFEST_PATH, {
        "pid": proc.pid,
        "out_dir": str(out_path_preview),
        "log_path": str(log_path),
        "start_time": time.time(),
        "cmd": cmd,
    })
    st.session_state.last_status = None
    st.session_state.output_dir = None
    st.rerun()


# ---- Running display -------------------------------------------------------
if _manifest is not None:
    log_path = Path(_manifest["log_path"])
    out_dir_active = Path(_manifest.get("out_dir", ""))
    start_time = float(_manifest.get("start_time", time.time()))
    elapsed = max(0.0, time.time() - start_time)
    recent = tail_log(log_path, max_lines=80)
    alive = pid_alive(_manifest.get("pid"))

    if alive:
        st.error("Computing - sidebar is locked. The subprocess keeps "
                  "running even if you close the browser.", icon="⚠️")

        cols = st.columns([4, 1])
        with cols[0]:
            st.info(f"Running ({elapsed:.0f} s elapsed) - PID "
                    f"{_manifest['pid']} -> `{out_dir_active.name}/`")
        with cols[1]:
            if st.button("Stop", type="secondary", use_container_width=True,
                          key="stop_btn"):
                kill_pid(_manifest.get("pid"))
                _clear(MANIFEST_PATH)
                st.session_state.last_status = ("error", "Stopped by user")
                st.rerun()

        # Lightweight progress: parse "Processing slide X/Y" from log.
        n_done, n_total = 0, 0
        for ln in reversed(recent):
            if "Processing slide" in ln:
                try:
                    seg = ln.split("Processing slide", 1)[1].strip()
                    a, b = seg.split("(")[0].split("/")
                    n_done = int(a.strip())
                    n_total = int(b.strip())
                    break
                except Exception:
                    pass
        if n_total:
            st.progress(min(0.999, n_done / n_total),
                        text=f"Slide {n_done}/{n_total}")
        else:
            st.progress(0.0, text="(starting up)")

        st.code("\n".join(recent) if recent else "(waiting for output...)",
                language="text")
        time.sleep(1.0)
        st.rerun()

    else:
        # PID gone: figure out exit status from the last log lines.
        done_line = next((l for l in recent if l.startswith("Elapsed time:")),
                         None)
        if done_line:
            kind = "success"
            msg = f"Done - {done_line.strip()} -> `{out_dir_active}`"
            _write_json(LAST_PATH, {
                "out_dir": str(out_dir_active),
                "finished_at": time.time(),
            })
        else:
            kind = "error"
            msg = (f"Run ended unexpectedly (no 'Elapsed time:' line). "
                   f"See `{log_path.relative_to(REPO).as_posix()}`.")
        st.session_state.last_status = (kind, msg)
        st.session_state.output_dir = out_dir_active
        _clear(MANIFEST_PATH)
        st.rerun()


# ---- Status banner ---------------------------------------------------------
if st.session_state.last_status is not None:
    kind, msg = st.session_state.last_status
    (st.success if kind == "success" else st.error)(msg)


# ---- Results tabs ----------------------------------------------------------
out_dir = st.session_state.output_dir
if out_dir is None:
    lc = _read_json(LAST_PATH)
    if lc and Path(lc.get("out_dir", "")).exists():
        out_dir = Path(lc["out_dir"])
        st.session_state.output_dir = out_dir

if out_dir and Path(out_dir).exists():
    results_csv = out_dir / "results.csv"
    phi3d_png = out_dir / "phi3d_hist.png"
    phi2d_png = out_dir / "phi2d_hist.png"
    phi3d_csv = out_dir / "phi3d_hist.csv"
    phi2d_csv = out_dir / "phi2d_hist.csv"
    shear_mat = out_dir / "shear_strength.mat"
    ba_shp = out_dir / "back_analysis.shp"
    poly_shp = out_dir / "2D_stability_polyline.shp"
    log_path = out_dir / "_run.log"

    tab_results, tab_hist, tab_files, tab_log = st.tabs(
        ["Results", "Histograms", "Files", "Log"])

    with tab_results:
        if results_csv.exists():
            import pandas as pd
            df = pd.read_csv(results_csv)
            n_total = len(df)
            n_ok = int((df.get("phi3d", pd.Series(dtype=float)) > 0).sum())
            n_skip = int((df.get("skip_reason", pd.Series(dtype=str))
                          .fillna("").astype(str).str.len() > 0).sum())

            m1, m2, m3 = st.columns(3)
            m1.metric("Total slides", f"{n_total}")
            m2.metric("Solved (phi3d > 0)", f"{n_ok}",
                       f"{100*n_ok/max(n_total,1):.1f}%")
            m3.metric("Skipped", f"{n_skip}")

            num_cols = [c for c in ("phi3d", "c3d", "rot3d", "phi2d", "c2d",
                                     "FS2D_by_phi3d_c3d") if c in df.columns]
            if num_cols:
                st.subheader("Summary statistics")
                st.dataframe(df[num_cols].describe().T, use_container_width=True)
            st.subheader("Per-slide results")
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("results.csv not found yet.")

    with tab_hist:
        cols = st.columns(2)
        if phi3d_png.exists():
            with cols[0]:
                st.image(str(phi3d_png), caption="phi3d distribution",
                          use_container_width=True)
        if phi2d_png.exists():
            with cols[1]:
                st.image(str(phi2d_png), caption="phi2d distribution",
                          use_container_width=True)
        if not (phi3d_png.exists() or phi2d_png.exists()):
            st.info("Histogram PNGs not found.")

    with tab_files:
        def _dl(path: Path, label: str | None = None, mime: str = "application/octet-stream"):
            if path.exists():
                st.download_button(
                    label or path.name,
                    data=path.read_bytes(), file_name=path.name,
                    mime=mime, use_container_width=True)

        st.subheader("CSVs")
        ccsv = st.columns(3)
        with ccsv[0]:
            _dl(results_csv, "results.csv", "text/csv")
        with ccsv[1]:
            _dl(phi3d_csv, "phi3d_hist.csv", "text/csv")
        with ccsv[2]:
            _dl(phi2d_csv, "phi2d_hist.csv", "text/csv")

        st.subheader("Shear-strength PMF (.mat)")
        st.caption(
            "Probability mass over (phi, c) pairs, in the schema consumed by "
            "USGS RegionGrow3D (prob, prob_phi [deg], prob_coh [kPa]).")
        _dl(shear_mat, "shear_strength.mat", "application/octet-stream")

        st.subheader("Shapefile bundles")
        st.caption("Shapefiles need their sidecar files (.shx, .dbf, .prj, "
                   ".cpg) - use Windows Explorer / zip to grab the full set "
                   "from the output folder.")
        for shp in (ba_shp, poly_shp):
            if shp.exists():
                st.code(str(shp), language="text")

        st.subheader("Output folder")
        st.code(str(out_dir), language="text")

    with tab_log:
        if log_path.exists():
            st.code(log_path.read_text(encoding="utf-8", errors="replace"),
                    language="text")
        else:
            st.info("No log file written yet.")

elif not start:
    st.info(
        "Configure the inputs and parameters in the left sidebar, then "
        "click **Start analysis**.\n\n"
        f"- Sample files are read from `{INPUT_DIR.relative_to(REPO).as_posix()}/`.\n"
        f"- Results are written under `{OUTPUT_DIR.relative_to(REPO).as_posix()}/` "
        "by default."
    )
