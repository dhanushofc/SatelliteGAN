"""
argo_filter_to_master.py  ─  Map Argo validation points to master_harry appended data

Reads:
  - validation_data/Argo_validsation_TSFM.xlsx
  - D:/INCOIS-internship/data/baka's appended data/master_region_data_new.npy
  - D:/INCOIS-internship/data/baka's appended data/master_region_anomalies_new.npy

Writes:
  - validation_data/Argo_validsation_TSFM_filtered_to_master.csv

Notes:
  - Uses temp_adjusted when temp_qc == 1 and temp_adjusted is present.
  - Drops rows with temp_qc == 4 (bad QC).
  - Maps to nearest grid cell using 0.25° resolution and START_DATE = 1981-09-01.
"""

from __future__ import annotations

import csv
import math
import re
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parent
ARGO_XLSX = ROOT / "validation_data" / "Argo_validsation_TSFM.xlsx"
OUT_CSV = ROOT / "validation_data" / "Argo_validsation_TSFM_filtered_to_master.csv"

MASTER_BASE = Path("D:/INCOIS-internship/data") / "baka's appended data"
MASTER_DATA = MASTER_BASE / "master_region_data_new.npy"
MASTER_ANOM = MASTER_BASE / "master_region_anomalies_new.npy"

# Grid + time conventions (from rolling/spatial scripts)
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
    """Load Sheet1 from XLSX into (headers, rows) with string values."""
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


def map_to_grid(lat: float, lon: float, h: int, w: int):
    ri = int(round((lat - LAT_MIN) / RES))
    ci = int(round((lon - LON_MIN) / RES))
    if not (0 <= ri < h and 0 <= ci < w):
        return None, None
    grid_lat = LAT_MIN + ri * RES
    grid_lon = LON_MIN + ci * RES
    return (ri, ci, grid_lat, grid_lon)


def main():
    if not ARGO_XLSX.exists():
        raise FileNotFoundError(f"Missing Argo XLSX: {ARGO_XLSX}")
    if not MASTER_DATA.exists():
        raise FileNotFoundError(f"Missing master data: {MASTER_DATA}")
    if not MASTER_ANOM.exists():
        raise FileNotFoundError(f"Missing master anomalies: {MASTER_ANOM}")

    headers, rows = load_xlsx_sheet1(ARGO_XLSX)
    idx = {name: i for i, name in enumerate(headers)}

    data = np.load(MASTER_DATA, mmap_mode="r")
    anom = np.load(MASTER_ANOM, mmap_mode="r")
    t_len, h, w = data.shape
    end_date = START_DATE + timedelta(days=t_len - 1)

    out_headers = headers + [
        "temp_used",
        "temp_source",
        "master_date",
        "master_t_idx",
        "master_row",
        "master_col",
        "master_lat",
        "master_lon",
        "master_sst",
        "master_anom",
        "diff_temp_vs_master",
        "abs_diff_temp_vs_master",
    ]

    kept = 0
    dropped = 0
    drop_reasons: Dict[str, int] = {}

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(out_headers)

        for r in rows:
            def get(col):
                return r[idx[col]] if col in idx and idx[col] < len(r) else ""

            time_val = get("time")
            date = parse_iso_date(time_val)
            if date is None:
                dropped += 1
                drop_reasons["bad_time"] = drop_reasons.get("bad_time", 0) + 1
                continue

            t_idx = (date - START_DATE.date()).days
            if t_idx < 0 or t_idx >= t_len:
                dropped += 1
                drop_reasons["time_out_of_range"] = drop_reasons.get("time_out_of_range", 0) + 1
                continue

            lat = parse_float(get("latitude"))
            lon = parse_float(get("longitude"))
            if lat is None or lon is None:
                dropped += 1
                drop_reasons["bad_latlon"] = drop_reasons.get("bad_latlon", 0) + 1
                continue

            grid = map_to_grid(lat, lon, h, w)
            if grid is None:
                dropped += 1
                drop_reasons["latlon_out_of_grid"] = drop_reasons.get("latlon_out_of_grid", 0) + 1
                continue
            row_i, col_i, grid_lat, grid_lon = grid

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
                dropped += 1
                drop_reasons["temp_qc_4"] = drop_reasons.get("temp_qc_4", 0) + 1
                continue
            else:
                if temp_raw is not None:
                    temp_used = temp_raw
                    temp_source = "temp"

            if temp_used is None:
                dropped += 1
                drop_reasons["missing_temp"] = drop_reasons.get("missing_temp", 0) + 1
                continue

            master_sst = float(data[t_idx, row_i, col_i])
            master_anom = float(anom[t_idx, row_i, col_i])
            diff = temp_used - master_sst

            out_row = list(r) + [
                f"{temp_used:.6f}",
                temp_source,
                date.isoformat(),
                str(t_idx),
                str(row_i),
                str(col_i),
                f"{grid_lat:.6f}",
                f"{grid_lon:.6f}",
                f"{master_sst:.6f}",
                f"{master_anom:.6f}",
                f"{diff:.6f}",
                f"{abs(diff):.6f}",
            ]
            writer.writerow(out_row)
            kept += 1

    print(f"Master data shape: {data.shape}")
    print(f"Date range: {START_DATE.date()} -> {end_date.date()}")
    print(f"Filtered rows written: {OUT_CSV}")
    print(f"Kept: {kept}  Dropped: {dropped}")
    for k in sorted(drop_reasons):
        print(f"  drop {k}: {drop_reasons[k]}")


if __name__ == "__main__":
    main()
