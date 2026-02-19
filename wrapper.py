# ----------------------------------------------------------------------
# 1. Imports
# ----------------------------------------------------------------------
import traceback
from pyetc_wst import WST
import os
import sys
import warnings
import multiprocessing
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt                     # noqa: F401  (kept: imported in notebook)
from astropy import units as u
from astropy.table import Table
from astropy.io import fits                         # noqa: F401  (kept: imported in notebook)

from utils.codebook import Obj_SED, Obj_Spat_Dis, MAG_FIL, MAG_SYS, INS, CH, TEMPLATE_FULL_OBS, AM, SED_Name, FLI

import time

warnings.filterwarnings("ignore")

# Optional progress bar (keep script runnable even if tqdm isn't installed)
try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(it, *args, **kwargs):  # type: ignore
        return it

# Reduce noisy per-row logs unless explicitly enabled
DEBUG = bool(int(os.environ.get("PYETC_WRAP_DEBUG", "0")))

# Output flags
CSV_OUTPUT = False
FITS_OUTPUT = True

# ----------------------------------------------------------------------
# 2. Worker initialiser – each process builds its own object
# ----------------------------------------------------------------------


def init_worker():
    """Initialise heavy objects once per worker process."""
    global obj
    obj = WST(log='ERROR', skip_dataload=False)
    # obj = VLT(skip_dataload=False)  # DEBUG


# ----------------------------------------------------------------------
# 3. Per-row computation wrapped in a picklable function
# ----------------------------------------------------------------------
def _snr_at_lam_ref(full_obs: Dict[str, Any]) -> Any:
    """
    Compute SNR at Lam_Ref for the provided full_obs (DIT must be set).

    Returns the same object type as the underlying ETC (often a float-like).
    """
    con, ob, spe, im, _ = obj.build_obs_full(full_obs)
    if full_obs.get("COADD_WL", 1) > 1:
        snr_spec = obj.snr_at_wave(con, im, spe, debug=False)[
            "snr_aperture_rebin"]
    else:
        snr_spec = obj.snr_at_wave(con, im, spe, debug=False)["snr_aperture"]
    return snr_spec


def _time_from_target_snr(
    full_obs: Dict[str, Any],
    *,
    target_snr: Any,
    fixed_ndit: Optional[Any],
) -> Tuple[Any, Dict[str, Any]]:
    """
    Compute exposure time (DIT) for a target SNR, optionally keeping NDIT fixed.

    Returns (dit_value, res_dict).
    """
    full_obs = full_obs.copy()
    full_obs["SNR"] = target_snr
    full_obs["DIT"] = np.nan
    if fixed_ndit is None:
        full_obs["NDIT"] = np.nan
        compute = "best"
    else:
        full_obs["NDIT"] = fixed_ndit
        compute = "dit"

    con, ob, spe, im, _ = obj.build_obs_full(full_obs)
    res = obj.time_from_source(con, im, spe, compute=compute)

    # pyetc uses lowercase keys ('dit', 'ndit', ...)
    dit_val = res.get("dit", res.get("DIT", np.nan))
    return dit_val, res


def process_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a FITS-row dictionary into full_obs, run ETC, return results.

    Any error is caught and stored in 'ERR' / 'ERR_MSG'.
    """

    # Debug info (printing from multiprocessing workers breaks tqdm output)
    if DEBUG:
        pid = os.getpid()
        try:
            cpu_id = multiprocessing.current_process()._identity[0]
        except Exception:
            cpu_id = "N/A"
        print(
            f"[DEBUG] Process PID={pid} CPU/Pool_ID={cpu_id} is processing ID={row.get('ID')}")

    full_obs = TEMPLATE_FULL_OBS.copy()
    full_obs.update(row)  # overwrite with row entries

    # Ensure output exposure-time columns exist.
    # We keep DIT only inside full_obs (ETC input); the FITS table should use T1/2/3.
    row.setdefault("T1", np.nan)
    row.setdefault("T2", np.nan)
    row.setdefault("T3", np.nan)

    # --- translate coded integers to strings --------------------------
    try:
        full_obs["Obj_SED"] = Obj_SED[row["Obj_SED"]]
        if row["Obj_SED"] == 1:
            full_obs["PL_Index"] = 0
        if row["Obj_SED"] == 4:
            full_obs["SED_Name"] = SED_Name[row["SED_Name"]]
    except Exception as exc:
        return {**row, "ERR": 1, "ERR_MSG": f"SED: {exc}"}

    try:
        full_obs["Obj_Spat_Dis"] = Obj_Spat_Dis[row["Obj_Spat_Dis"]]
        if row["Obj_Spat_Dis"] == 3:
            full_obs["IMA"] = "sersic"
    except Exception as exc:
        return {**row, "ERR": 1, "ERR_MSG": f"Obj_Spat_Dis: {exc}"}

    try:
        full_obs["MAG_FIL"] = None if pd.isna(
            row["MAG_FIL"]) else MAG_FIL[row["MAG_FIL"]]
    except Exception as exc:
        return {**row, "ERR": 1, "ERR_MSG": f"MAG_FIL: {exc}"}

    try:
        full_obs["MAG_SYS"] = None if pd.isna(
            row["MAG_SYS"]) else MAG_SYS[row["MAG_SYS"]]
    except Exception as exc:
        return {**row, "ERR": 1, "ERR_MSG": f"MAG_SYS: {exc}"}

    try:
        full_obs["INS"] = INS[row["INS"]]
    except Exception as exc:
        return {**row, "ERR": 1, "ERR_MSG": f"INS: {exc}"}

    try:
        full_obs["CH"] = CH[row["CH"]]
    except Exception as exc:
        return {**row, "ERR": 1, "ERR_MSG": f"CH: {exc}"}

    try:
        full_obs["AM"] = AM[row["AM"]]
    except Exception as exc:
        return {**row, "ERR": 1, "ERR_MSG": f"AM: {exc}"}

    try:
        fli_raw = row.get("FLI")
        if pd.isna(fli_raw):
            raise ValueError("Missing required code field 'FLI'")
        # tolerate floats like 1.0 coming from FITS/pandas
        fli_code = int(float(fli_raw))
        full_obs["FLI"] = FLI[fli_code]
    except Exception as exc:
        return {**row, "ERR": 1, "ERR_MSG": f"FLI: {exc}"}

    # ------------------------------------------------------------------
    # --- ETC computation ----------------------------------------------

    try:
        # if NDIT is given in input, keep it across all T1/T2/T3 computations
        fixed_ndit = None if pd.isna(row.get("NDIT")) else row.get("NDIT")

        # Decide what to compute based on whether SNR is given
        if pd.isna(row.get("SNR")):
            # ----------------------------------------------------------
            # Case exp time given: compute SNR for DIT(FLI) then compute
            # missing DIT's for darker skies using that SNR
            # ----------------------------------------------------------
            dit_key = f"T{fli_code}"
            dit_given = row.get(dit_key)
            if pd.isna(dit_given):
                return {**row, "ERR": 1, "ERR_MSG": f"Missing {dit_key} for exp-time case"}

            full_obs_snr = full_obs.copy()
            full_obs_snr["FLI"] = FLI[fli_code]
            # ETC needs 'DIT' key even though the FITS input has T1/2/3 only
            full_obs_snr["DIT"] = dit_given
            if fixed_ndit is not None:
                full_obs_snr["NDIT"] = fixed_ndit
            # ensure we are in "compute SNR" mode
            full_obs_snr["SNR"] = np.nan

            snr_val = _snr_at_lam_ref(full_obs_snr)
            row["SNR"] = snr_val

            # compute darker-sky exposure times that reach the same SNR
            for sky_code in range(1, fli_code):
                full_obs_i = full_obs.copy()
                full_obs_i["FLI"] = FLI[sky_code]
                dit_val, _ = _time_from_target_snr(
                    full_obs_i, target_snr=snr_val, fixed_ndit=fixed_ndit
                )
                row[f"T{sky_code}"] = dit_val

            row["ERR"] = 0
            return row

        # --------------------------------------------------------------
        # Case SNR given: compute t1/t2/t3 up to the requested FLI
        # --------------------------------------------------------------
        target_snr = row.get("SNR")
        last_res: Dict[str, Any] = {}
        for sky_code in range(1, fli_code + 1):
            full_obs_i = full_obs.copy()
            full_obs_i["FLI"] = FLI[sky_code]
            dit_val, res = _time_from_target_snr(
                full_obs_i, target_snr=target_snr, fixed_ndit=fixed_ndit
            )
            row[f"T{sky_code}"] = dit_val

        row["ERR"] = 0
        return row

    except Exception as exc:
        # print row and message
        print("Error in row\n", row, "\nError message: ", exc)
        print(full_obs)
        print("Full traceback:")
        traceback.print_exc()
        return {**row, "ERR": 1}


# ----------------------------------------------------------------------
# 4. Main driver
# ----------------------------------------------------------------------
def main() -> None:
    cpu_count = multiprocessing.cpu_count()
    print(f"Number of CPU available: {cpu_count}")
    # --- load FITS into pandas ----------------------------------------
    # PATH TO FITS FILE
    fits_path = ""
    tab = Table.read(fits_path, format="fits")
    df = tab.to_pandas()
    print("FITS file loaded")

    total_rows = len(df)
    # IMPORTANT: don't materialize all rows into a huge list; stream them instead
    input_rows = (row.to_dict() for _, row in df.iterrows())

    # --- multiprocessing ---------------------------------------------
    # with multiprocessing.Pool(processes=multiprocessing.cpu_count(), initializer=init_worker) as pool:
    print("Starting pool")
    # set the number of processes to the number of CPU available
    start_time = time.time()
    out_csv = ".csv"
    out_fits = ".fits"
    batch_size = 10000  # small batches keep memory bounded
    wrote_header = False
    wrote_fits = False

    if FITS_OUTPUT:
        # Create output FITS with EXACT same columns/dtypes as input,
        # then update rows in-place batch-by-batch.
        with fits.open(fits_path, memmap=True) as hdul_in:
            primary = hdul_in[0].copy()
            table_hdu = fits.BinTableHDU(
                data=hdul_in[1].data[:total_rows],
                header=hdul_in[1].header.copy(),
                name=hdul_in[1].name,
            )
            fits.HDUList([primary, table_hdu]).writeto(
                out_fits, overwrite=True)
        wrote_fits = True

    with multiprocessing.Pool(
        processes=cpu_count,  # Can be changed to the number of processes you want to use
        initializer=init_worker,
        maxtasksperchild=500,
    ) as pool:
        wrote_rows = 0
        batch = []
        for res in tqdm(
            pool.imap(process_row, input_rows, chunksize=10),
            total=total_rows,
            desc="Processing rows",
        ):
            batch.append(res)
            if len(batch) >= batch_size:
                out_df = pd.DataFrame(batch)
                if CSV_OUTPUT:
                    out_df.to_csv(
                        out_csv,
                        mode="a" if wrote_header else "w",
                        header=not wrote_header,
                        index=False,
                    )
                    wrote_header = True
                if FITS_OUTPUT:
                    # Update existing FITS table in-place (preserves exact column types)
                    start = wrote_rows
                    end = start + len(out_df)
                    with fits.open(out_fits, mode="update", memmap=True) as hdul:
                        data = hdul[1].data
                        for col in out_df.columns:
                            if col in data.names:
                                vals = out_df[col].to_numpy()
                                good = ~pd.isna(vals)
                                if good.all():
                                    data[col][start:end] = vals
                                elif good.any():
                                    seg = data[col][start:end]
                                    seg[good] = vals[good]
                                    data[col][start:end] = seg
                        hdul.flush()
                    wrote_rows = end
                batch.clear()

        if batch:
            out_df = pd.DataFrame(batch)
            if CSV_OUTPUT:
                out_df.to_csv(
                    out_csv,
                    mode="a" if wrote_header else "w",
                    header=not wrote_header,
                    index=False,
                )
                wrote_header = True
            if FITS_OUTPUT:
                start = wrote_rows
                end = start + len(out_df)
                with fits.open(out_fits, mode="update", memmap=True) as hdul:
                    data = hdul[1].data
                    for col in out_df.columns:
                        if col in data.names:
                            vals = out_df[col].to_numpy()
                            good = ~pd.isna(vals)
                            if good.all():
                                data[col][start:end] = vals
                            elif good.any():
                                seg = data[col][start:end]
                                seg[good] = vals[good]
                                data[col][start:end] = seg
                    hdul.flush()
                wrote_rows = end

    end_time = time.time()
    print(f"Time to process the rows: {end_time - start_time} seconds")

    if CSV_OUTPUT:
        print(f"✓ Results saved to {out_csv}")
    if FITS_OUTPUT:
        print(f"✓ Results saved to {out_fits}")


# ----------------------------------------------------------------------
# 5. Script guard
# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
