# ----------------------------------------------------------------------
# 1. Imports
# ----------------------------------------------------------------------
import traceback
from pyetc_wst import WST
import os
import sys
import warnings
import multiprocessing
from typing import Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt                     # noqa: F401  (kept: imported in notebook)
from astropy import units as u
from astropy.table import Table
from astropy.io import fits                         # noqa: F401  (kept: imported in notebook)

from utils.codebook import Obj_SED, Obj_Spat_Dis, MAG_FIL, MAG_SYS, INS, CH, TEMPLATE_FULL_OBS, AM, SED_Name, FLI

import time

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# 2. Worker initialiser – each process builds its own object
# ----------------------------------------------------------------------


def init_worker():
    """Initialise heavy objects once per worker process."""
    global obj
    obj = WST(log='DEBUG', skip_dataload=False)
    # obj = VLT(skip_dataload=False)  # DEBUG


# ----------------------------------------------------------------------
# 3. Per-row computation wrapped in a picklable function
# ----------------------------------------------------------------------
def process_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a FITS-row dictionary into full_obs, run ETC, return results.

    Any error is caught and stored in 'ERR' / 'ERR_MSG'.
    """

    # Debug info
    pid = os.getpid()
    try:
        cpu_id = multiprocessing.current_process()._identity[0]
    except Exception:
        cpu_id = "N/A"
    print(
        f"[DEBUG] Process PID={pid} CPU/Pool_ID={cpu_id} is processing ID={row.get('ID')}")

    full_obs = TEMPLATE_FULL_OBS.copy()
    full_obs.update(row)  # overwrite with row entries

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
        full_obs["FLI"] = FLI[row["FLI"]]
    except Exception as exc:
        return {**row, "ERR": 1, "ERR_MSG": f"FLI: {exc}"}

    # ------------------------------------------------------------------
    # --- ETC computation ----------------------------------------------

    try:
        DIT = full_obs["DIT"]
        SNR = full_obs["SNR"]

        print("Build obs pre")
        con, ob, spe, im, _ = obj.build_obs_full(full_obs)
        print(con)

        # Decide what to compute
        if np.isnan(SNR):
            if full_obs["COADD_WL"] > 1:
                res = {
                    "SNR": obj.snr_from_source(con, im, spe)["spec"][
                        "snr_rebin"
                    ].subspec(full_obs["Lam_Ref"], unit=u.angstrom)
                }
            else:
                res = {
                    "SNR": obj.snr_from_source(con, im, spe)["spec"]["snr"].subspec(
                        full_obs["Lam_Ref"], unit=u.angstrom
                    )
                }
        else:
            if np.isnan(DIT) and np.isnan(NDIT):
                print("Given SNR, find the best DIT and NDIT")
                res = obj.time_from_source(con, im, spe, compute="best")
            elif np.isnan(DIT):
                print("Given SNR, find the best DIT")
                res = obj.time_from_source(con, im, spe, compute="dit")
            elif np.isnan(NDIT):
                print("Given SNR, find the best NDIT")
                res = obj.time_from_source(con, im, spe, compute="ndit")

        # Add results (upper-case keys) to row dict
        for k, v in res.items():
            row[k.upper()] = v

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
    fits_path = "/Users/andre/Desktop/INAF/WST/pyetc_wrap/data/102_TESTCATALOG_v1225_1000.fits"
    tab = Table.read(fits_path, format="fits")
    df = tab.to_pandas()
    print("FITS file loaded")

    # Convert each row to a plain dict (pickle-safe)
    # calculate time to process the rows
    start_time = time.time()
    input_rows = [row.to_dict() for _, row in df.iterrows()]
    # input_rows = df.to_dict('records')
    end_time = time.time()
    print(f"Time to process the rows: {end_time - start_time} seconds")

    # --- multiprocessing ---------------------------------------------
    # with multiprocessing.Pool(processes=multiprocessing.cpu_count(), initializer=init_worker) as pool:
    print("Starting pool")
    with multiprocessing.Pool(processes=1, initializer=init_worker) as pool:
        results = pool.map(process_row, input_rows)

    # --- assemble results --------------------------------------------
    out_df = pd.DataFrame(results)
    print(out_df)
    # out_csv = "results_parallel.csv"
    # out_df.to_csv(out_csv, index=False)
    # print(f"✓ Results saved to {out_csv}")


# ----------------------------------------------------------------------
# 5. Script guard
# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
