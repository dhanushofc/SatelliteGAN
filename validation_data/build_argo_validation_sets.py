"""
build_argo_validation_sets.py  ─  Build aligned Argo/master/reanalysis CSVs

Outputs (CSV, identical schema + row order):
  - validation_data/argo_validation_tsfm.csv
  - validation_data/master_appended_tsfm.csv
  - validation_data/reanalysis_tsfm.csv

Selection rules:
  - Use `time` (profile time) for date mapping.
  - For each (platform_number, cycle_number, time), keep SST at min pressure.
  - Use temp_adjusted when temp_qc==1 and present; drop rows with temp_qc==4.
  - Map to nearest grid cell (0.25°). Master grid: 5.125..19.875N, 60.125..71.875E.
  - Reanalysis grid: 5..20N, 60..72E (0.25°), time in seconds since 1970-01-01.
  - Reanalysis sst is in Kelvin; convert to Celsius.
"""

from __future__ import annotations

import csv
import math
import re
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import xml.etree.ElementTree as ET
from netCDF4 import Dataset

ROOT = Path(__file__).resolve().parent
ARGO_XLSX = ROOT / "validation_data" / "Argo_validsation_TSFM.xlsx"
REANALYSIS_NC = ROOT / "validation_data" / "Argo_validsation_TSFM_reanalysis.nc"

MASTER_BASE = Path("D:/INCOIS-internship/data") / "baka's appended data"
MASTER_DATA = MASTER_BASE / "master_region_data_new.npy"
MASTER_ANOM = MASTER_BASE / "master_region_anomalies_new.npy"

OUT_ARGO = ROOT / "validation_data" / "argo_validation_tsfm.csv"
OUT_MASTER = ROOT / "validation_data" / "master_appended_tsfm.csv"
OUT_REAN = ROOT / "validation_data" / "reanalysis_tsfm.csv"

# Master grid
LAT_MIN, LAT_MAX = 5.125, 19.875
LON_MIN, LON_MAX = 60.125, 71.875
RES = 0.25
START_DATE = datetime(1981, 9, 1)


def col_to_idx(col_letters: str) -> int:
    idx = 0
    for ch in col_letters:
        idx = idx * 26 + (ord(ch) - 64)
    return idx - 1


def load_xlsx_sheet1(path: Path) -> Tuple[List[str], List[List[str]]]:
    z = zipfile.ZipFile(path)
    names = z.namelist()
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

    shared: List[str] = []
    if "xl/sharedStrings.xml" in names:
        ss = ET.fromstring(z.read("xl/sharedStrings.xml"))
        for si in ss.findall(".//a:si", ns):
            shared.append("".join([t.text or "" for t in si.findall(".//a:t", ns)]))

    sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
    dim = sheet.find(".//a:dimension", ns)
    dimref = dim.attrib.get("ref") if dim is not None else ""
    m = re.match(r"([A-Z]+)(\d+):([A-Z]+)(\d+)", dimref)
    maxcol = col_to_idx(m.group(3)) if m else 0
    maxrow = int(m.group(4)) if m else 0

    table = [["" for _ in range(maxcol + 1)] for __ in range(maxrow)]
    for row in sheet.findall(".//a:sheetData/a:row", ns):
        rnum = int(row.attrib.get("r", "0")) - 1
        for c in row.findall("a:c", ns):
            ref = c.attrib.get("r")
            if not ref:
                continue
            col_letters = "".join([ch for ch in ref if ch.isalpha()])
            col = col_to_idx(col_letters)
            v = c.find("a:v", ns)
            if v is None:
                val = ""
            else:
                t = c.attrib.get("t")
                if t == "s":
                    i = int(v.text)
                    val = shared[i] if i < len(shared) else v.text
                else:
                    val = v.text
            if 0 <= rnum < maxrow and 0 <= col <= maxcol:
                table[rnum][col] = val

    headers = table[0] if table else []
    rows = table[1:] if table else []
    return headers, rows


def parse_iso_date(value: str):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def parse_float(value: str):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    v = str(value).strip()
    if v == "" or v.lower() == "nan":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def map_to_master_grid(lat: float, lon: float, h: int, w: int):
    ri = int(round((lat - LAT_MIN) / RES))
    ci = int(round((lon - LON_MIN) / RES))
    if not (0 <= ri < h and 0 <= ci < w):
        return None
    grid_lat = LAT_MIN + ri * RES
    grid_lon = LON_MIN + ci * RES
    return ri, ci, grid_lat, grid_lon


def map_to_reanalysis_grid(lat: float, lon: float, lats: np.ndarray, lons: np.ndarray):
    lat_idx = int(np.argmin(np.abs(lats - lat)))
    lon_idx = int(np.argmin(np.abs(lons - lon)))
    return lat_idx, lon_idx, float(lats[lat_idx]), float(lons[lon_idx])


def main():
    if not ARGO_XLSX.exists():
        raise FileNotFoundError(f"Missing Argo XLSX: {ARGO_XLSX}")
    if not MASTER_DATA.exists():
        raise FileNotFoundError(f"Missing master data: {MASTER_DATA}")
    if not MASTER_ANOM.exists():
        raise FileNotFoundError(f"Missing master anomalies: {MASTER_ANOM}")
    if not REANALYSIS_NC.exists():
        raise FileNotFoundError(f"Missing reanalysis: {REANALYSIS_NC}")

    headers, rows = load_xlsx_sheet1(ARGO_XLSX)
    idx = {name: i for i, name in enumerate(headers)}

    # Build row records with QC filtering
    cleaned = []
    drop_counts: Dict[str, int] = {}
    for r in rows:
        def get(col):
            return r[idx[col]] if col in idx and idx[col] < len(r) else ""

        time_val = get("time")
        date = parse_iso_date(time_val)
        if date is None:
            drop_counts["bad_time"] = drop_counts.get("bad_time", 0) + 1
            continue

        lat = parse_float(get("latitude"))
        lon = parse_float(get("longitude"))
        pres = parse_float(get("pres"))
        if lat is None or lon is None or pres is None:
            drop_counts["bad_latlon_pres"] = drop_counts.get("bad_latlon_pres", 0) + 1
            continue

        temp_qc = str(get("temp_qc")).strip()
        temp_adjusted = parse_float(get("temp_adjusted"))
        temp_raw = parse_float(get("temp"))

        temp_used = None
        temp_source = ""
        if temp_qc == "1":
            if temp_adjusted is not None and not math.isnan(temp_adjusted):
                temp_used = temp_adjusted
                temp_source = "temp_adjusted"
            elif temp_raw is not None:
                temp_used = temp_raw
                temp_source = "temp"
        elif temp_qc == "4":
            drop_counts["temp_qc_4"] = drop_counts.get("temp_qc_4", 0) + 1
            continue
        else:
            if temp_raw is not None:
                temp_used = temp_raw
                temp_source = "temp"

        if temp_used is None:
            drop_counts["missing_temp"] = drop_counts.get("missing_temp", 0) + 1
            continue

        cleaned.append({
            "row": r,
            "date": date,
            "lat": lat,
            "lon": lon,
            "pres": pres,
            "temp_used": float(temp_used),
            "temp_source": temp_source,
            "platform_number": get("platform_number"),
            "cycle_number": get("cycle_number"),
            "time": time_val,
        })

    # Choose SST (min pressure) per (platform, cycle, time)
    grouped: Dict[Tuple[str, str, str], Dict] = {}
    for rec in cleaned:
        key = (rec["platform_number"], rec["cycle_number"], rec["time"])
        if key not in grouped or rec["pres"] < grouped[key]["pres"]:
            grouped[key] = rec

    selected = list(grouped.values())
    selected.sort(key=lambda x: (x["date"].isoformat(), x["platform_number"], x["cycle_number"]))

    # Load master arrays
    data = np.load(MASTER_DATA, mmap_mode="r")
    anom = np.load(MASTER_ANOM, mmap_mode="r")
    t_len, h, w = data.shape
    end_date = START_DATE + timedelta(days=t_len - 1)

    # Load reanalysis
    ds = Dataset(REANALYSIS_NC)
    r_times = ds.variables["valid_time"][:]
    r_lats = ds.variables["latitude"][:]
    r_lons = ds.variables["longitude"][:]
    r_sst = ds.variables["sst"]  # lazy access
    r_time_units = ds.variables["valid_time"].units

    # Build reanalysis time index (seconds since 1970-01-01)
    r_time_to_idx = {int(t): i for i, t in enumerate(r_times)}

    out_headers = [
        "key_id",
        "date",
        "lat",
        "lon",
        "pres",
        "temp_value",
        "temp_source",
        "master_t_idx",
        "master_row",
        "master_col",
        "master_lat",
        "master_lon",
    ]

    with OUT_ARGO.open("w", newline="", encoding="utf-8") as f_argo, \
         OUT_MASTER.open("w", newline="", encoding="utf-8") as f_master, \
         OUT_REAN.open("w", newline="", encoding="utf-8") as f_rean:

        w_argo = csv.writer(f_argo)
        w_master = csv.writer(f_master)
        w_rean = csv.writer(f_rean)
        w_argo.writerow(out_headers)
        w_master.writerow(out_headers + ["master_anom"])
        w_rean.writerow(out_headers)

        kept = 0
        dropped_master = 0
        dropped_rean = 0

        for i, rec in enumerate(selected, start=1):
            date = rec["date"]
            lat = rec["lat"]
            lon = rec["lon"]
            pres = rec["pres"]

            # Master mapping
            t_idx = (date - START_DATE.date()).days
            if t_idx < 0 or t_idx >= t_len:
                dropped_master += 1
                continue
            grid = map_to_master_grid(lat, lon, h, w)
            if grid is None:
                dropped_master += 1
                continue
            row_i, col_i, grid_lat, grid_lon = grid

            # Reanalysis mapping (daily, 00:00)
            ts = int(datetime(date.year, date.month, date.day, tzinfo=timezone.utc).timestamp())
            if ts not in r_time_to_idx:
                dropped_rean += 1
                continue
            rt_idx = r_time_to_idx[ts]
            r_lat_idx, r_lon_idx, r_lat, r_lon = map_to_reanalysis_grid(lat, lon, r_lats, r_lons)
            r_val = float(r_sst[rt_idx, r_lat_idx, r_lon_idx])
            r_val_c = r_val - 273.15

            key_id = f"{i:03d}"
            row_base = [
                key_id,
                date.isoformat(),
                f"{lat:.6f}",
                f"{lon:.6f}",
                f"{pres:.6f}",
            ]

            w_argo.writerow(row_base + [f"{rec['temp_used']:.6f}", rec["temp_source"],
                                         str(t_idx), str(row_i), str(col_i),
                                         f"{grid_lat:.6f}", f"{grid_lon:.6f}"])

            master_val = float(data[t_idx, row_i, col_i])
            master_anom = float(anom[t_idx, row_i, col_i])
            w_master.writerow(row_base + [f"{master_val:.6f}", "master_region_data_new",
                                           str(t_idx), str(row_i), str(col_i),
                                           f"{grid_lat:.6f}", f"{grid_lon:.6f}",
                                           f"{master_anom:.6f}"])

            w_rean.writerow(row_base + [f"{r_val_c:.6f}", "reanalysis_sst",
                                         str(t_idx), str(row_i), str(col_i),
                                         f"{grid_lat:.6f}", f"{grid_lon:.6f}"])

            kept += 1

    print(f"Argo rows cleaned: {len(cleaned)}")
    print(f"Selected SST rows: {len(selected)}")
    print(f"Master date range: {START_DATE.date()} -> {end_date.date()}")
    print(f"Wrote: {OUT_ARGO}")
    print(f"Wrote: {OUT_MASTER}")
    print(f"Wrote: {OUT_REAN}")
    print(f"Kept (aligned): {kept}")
    print(f"Dropped (master map): {dropped_master}")
    print(f"Dropped (reanalysis map): {dropped_rean}")
    if drop_counts:
        print("Drop counts (Argo QC):")
        for k in sorted(drop_counts):
            print(f"  {k}: {drop_counts[k]}")


if __name__ == "__main__":
    main()
