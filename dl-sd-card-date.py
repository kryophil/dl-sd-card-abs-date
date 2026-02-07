#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
dl-sd-card-date — Absolute Datierung der SD-Karten-Rohdaten gegen Influx-Export
--------------------------------------------------------------------------------
Refactored v2:
  - Vektorisierte I/O (kein iterrows)
  - float64 Quantisierung (kein Decimal)
  - Glatte Interpolation durch gebinnte Median-Offsets (kein Fenster-Stitching)
  - Alle Anker werden genutzt (keine 6h-Ausdünnung)
  - Vereinfachte Konfiguration (~10 Parameter statt 30+)
  - Gleiches Ausgabeformat (SD_absolute.csv, Segment_report.csv, etc.)

Konfig via YAML (optional): dl-sd-card-date.yaml im selben Ordner (oder via --config)
"""

from __future__ import annotations
import sys, os, math
from typing import List, Tuple, Dict, Optional
from pathlib import Path
from collections import defaultdict
from bisect import bisect_right

import pandas as pd
import numpy as np

# ====================
# Defaults (per YAML/CLI überschreibbar)
# ====================

INPUT_DIR  = "Input"
OUTPUT_DIR = "Output"

SD_GLOB      = "*_SDCard_raw_*.csv"
INFLUX_GLOB  = "Sensors_Raw_*.csv"

# Quantisierung — Nachkommastellen für exaktes Tripel-Matching
TEMP_DECIMALS = 10
RH_DECIMALS   = 8
BAT_DECIMALS  = 3

# Jitterfenster (laut Datenblatt: 0…8 s zufällige Verzögerung vor LoRa-TX)
J_MAX_SECONDS = 8.0

# Interpolation: Bin-Breite für Median-Offset-Glättung
BIN_HOURS = 6.0
MIN_ANCHORS_PER_BIN = 3

# Trimming: iteratives Entfernen von Ausreißern
N_TRIM_ITER       = 3
MAX_TRIM_FRACTION  = 0.02
MIN_ANCHORS_FOR_FIT = 30

# Qualitätsschwellen
RMSE_GOOD_S  = 12.0
RMSE_MED_S   = 20.0
MIN_ANCHORS_GOOD = 20

# Output-Dateien (unter OUTPUT_DIR)
OUT_SD_ABSOLUTE        = "SD_absolute.csv"
OUT_SEGMENT_REPORT     = "Segment_report.csv"
OUT_ANCHOR_REPORT      = "Anchors_report.csv"
OUT_PLAUSIBILITY_REPORT = "Plausibility_report.csv"


# ====================
# CLI & YAML
# ====================

def _apply_cli_overrides():
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default=None)
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    args, _ = parser.parse_known_args()
    if args.input_dir:
        globals()["INPUT_DIR"] = args.input_dir
    if args.output_dir:
        globals()["OUTPUT_DIR"] = args.output_dir
    return args


def _simple_yaml_load(text: str) -> dict:
    """Flacher Key:Value-Parser (kein PyYAML nötig)."""
    data = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Inline-Kommentare entfernen (nicht innerhalb Quotes)
        in_q = False
        for i, ch in enumerate(line):
            if ch in ("'", '"'):
                in_q = not in_q
            elif ch == "#" and not in_q:
                line = line[:i].rstrip()
                break
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        v = v.strip()
        # Typ-Erkennung
        if v == "" or v.lower() == "null":
            data[k.strip()] = None
        elif v.lower() in ("true", "false"):
            data[k.strip()] = v.lower() == "true"
        elif (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            data[k.strip()] = v[1:-1]
        else:
            try:
                data[k.strip()] = float(v) if ("." in v or "e" in v.lower()) else int(v)
            except ValueError:
                data[k.strip()] = v
    return data


def _apply_config_overrides(config_path: Optional[str]):
    if not config_path:
        script_dir = Path(os.path.abspath(os.path.dirname(__file__)))
        default_yaml = script_dir / "dl-sd-card-date.yaml"
        if default_yaml.exists():
            config_path = str(default_yaml)
        else:
            return
    p = Path(config_path)
    if not p.exists():
        print(f"[WARN] Config nicht gefunden: {config_path}", file=sys.stderr)
        return
    text = p.read_text(encoding="utf-8")
    try:
        import yaml
        data = yaml.safe_load(text)
    except Exception:
        data = _simple_yaml_load(text)
    if not isinstance(data, dict):
        return

    KNOWN = {
        "INPUT_DIR": str, "OUTPUT_DIR": str, "SD_GLOB": str, "INFLUX_GLOB": str,
        "TEMP_DECIMALS": int, "RH_DECIMALS": int, "BAT_DECIMALS": int,
        "J_MAX_SECONDS": float, "BIN_HOURS": float, "MIN_ANCHORS_PER_BIN": int,
        "N_TRIM_ITER": int, "MAX_TRIM_FRACTION": float, "MIN_ANCHORS_FOR_FIT": int,
        "RMSE_GOOD_S": float, "RMSE_MED_S": float, "MIN_ANCHORS_GOOD": int,
        "OUT_SD_ABSOLUTE": str, "OUT_SEGMENT_REPORT": str,
        "OUT_ANCHOR_REPORT": str, "OUT_PLAUSIBILITY_REPORT": str,
    }
    for name, cast in KNOWN.items():
        if name in data and data[name] is not None:
            try:
                globals()[name] = cast(data[name])
            except Exception:
                pass


# ====================
# Quantisierung (float64, kein Decimal)
# ====================

def _round_half_up(x: np.ndarray, decimals: int) -> np.ndarray:
    """ROUND_HALF_UP: 0.5 wird aufgerundet (wie kaufmännisches Runden)."""
    factor = 10.0 ** decimals
    return np.floor(x * factor + 0.5) / factor


def _format_val(val: float, decimals: int) -> str:
    return f"{val:.{decimals}f}"


def sd_raw_to_quantized(temp_raw: np.ndarray, rh_raw: np.ndarray, bat_raw: np.ndarray):
    """Vektorisierte Umrechnung: SD-Rohwerte → quantisierte physikalische Werte."""
    T = temp_raw * 175.0 / 65535.0 - 45.0
    RH = rh_raw * 100.0 / 65535.0
    U = bat_raw / 1000.0
    T_q = _round_half_up(T, TEMP_DECIMALS)
    RH_q = _round_half_up(RH, RH_DECIMALS)
    U_q = _round_half_up(U, BAT_DECIMALS)
    return T_q, RH_q, U_q


def influx_phys_to_quantized(T_vals: np.ndarray, RH_vals: np.ndarray, U_vals: np.ndarray):
    """Vektorisierte Quantisierung der Influx-Physikwerte."""
    T_q = _round_half_up(T_vals, TEMP_DECIMALS)
    RH_q = _round_half_up(RH_vals, RH_DECIMALS)
    U_q = _round_half_up(U_vals, BAT_DECIMALS)
    return T_q, RH_q, U_q


def _make_keys(T_q: np.ndarray, RH_q: np.ndarray, U_q: np.ndarray) -> np.ndarray:
    """Erzeugt String-Schlüssel für Tripel-Matching."""
    keys = np.empty(len(T_q), dtype=object)
    for i in range(len(T_q)):
        keys[i] = f"{T_q[i]:.{TEMP_DECIMALS}f}|{RH_q[i]:.{RH_DECIMALS}f}|{U_q[i]:.{BAT_DECIMALS}f}"
    return keys


# ====================
# I/O (vektorisiert)
# ====================

def resolve_paths() -> Tuple[List[Path], List[Path], Path]:
    script_dir = Path(os.path.abspath(os.path.dirname(__file__)))
    in_dir = Path(INPUT_DIR)
    out_dir = Path(OUTPUT_DIR)
    if not in_dir.is_absolute():
        in_dir = (script_dir / in_dir).resolve()
    if not out_dir.is_absolute():
        out_dir = (script_dir / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] INPUT_DIR:  {in_dir}")
    print(f"[INFO] OUTPUT_DIR: {out_dir}")
    # Glob case-insensitiv: probiere auch Großschreibung der Endung
    sd_files = sorted(in_dir.glob(SD_GLOB))
    if not sd_files:
        sd_files = sorted(in_dir.glob(SD_GLOB.replace('.csv', '.CSV')))
    influx_files = sorted(in_dir.glob(INFLUX_GLOB))
    if not influx_files:
        influx_files = sorted(in_dir.glob(INFLUX_GLOB.replace('.csv', '.CSV')))
    if not sd_files:
        print(f"[WARN] Keine SD-Dateien: {SD_GLOB} in {in_dir}", file=sys.stderr)
    if not influx_files:
        print(f"[WARN] Keine Influx-Dateien: {INFLUX_GLOB} in {in_dir}", file=sys.stderr)
    return sd_files, influx_files, out_dir


def read_sd_files(sd_paths: List[Path]) -> pd.DataFrame:
    """Liest SD-CSVs und erkennt Segmentgrenzen (negative Zeitsprünge).

    Returns DataFrame mit Spalten:
        global_idx, segment_id, idx_in_segment, t1024, t_rel_s, T_q, RH_q, U_q, key
    """
    frames = []
    for path in sd_paths:
        df = pd.read_csv(path, header=None,
                         names=["t1024", "temp_raw", "rh_raw", "bat_raw"],
                         encoding="utf-8-sig").dropna()
        df = df.astype({"t1024": "int64", "temp_raw": "int64",
                        "rh_raw": "int64", "bat_raw": "int64"})
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["t_rel_s"] = df["t1024"] / 1024.0

    # Segmente: wo die relative Zeit rückwärts springt
    dt = df["t_rel_s"].diff()
    segment_starts = (dt < 0).cumsum()
    df["segment_id"] = segment_starts
    df["idx_in_segment"] = df.groupby("segment_id").cumcount()
    df["global_idx"] = np.arange(len(df))

    # Vektorisierte Quantisierung
    T_q, RH_q, U_q = sd_raw_to_quantized(
        df["temp_raw"].values, df["rh_raw"].values, df["bat_raw"].values
    )
    df["T_q"] = T_q
    df["RH_q"] = RH_q
    df["U_q"] = U_q
    df["key"] = _make_keys(T_q, RH_q, U_q)

    return df


def _detect_influx_columns(df: pd.DataFrame) -> Tuple[str, str, str]:
    """Erkennt Temperatur-, Feuchte- und Batterie-Spalten automatisch."""
    cols = list(df.columns)
    bat_col = next((c for c in cols if "battery" in c.lower()), None)
    rh_col = next((c for c in cols if "humid" in c.lower()), None)
    t_col = next((c for c in cols if "temp" in c.lower()), None)
    if not (bat_col and rh_col and t_col):
        raise ValueError(f"Influx-Spalten nicht erkannt: {cols}")
    return t_col, rh_col, bat_col


def read_influx_files(influx_paths: List[Path]) -> pd.DataFrame:
    """Liest Influx-CSVs (vektorisiert).

    Returns DataFrame mit Spalten:
        influx_idx, ts_utc, ts_epoch, T_q, RH_q, U_q, key
    """
    frames = [pd.read_csv(p, encoding="utf-8-sig") for p in influx_paths]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    t_col, rh_col, bat_col = _detect_influx_columns(df)
    ts_col = df.columns[0]

    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.dropna(subset=[ts_col]).sort_values(by=ts_col).reset_index(drop=True)

    # Vektorisierte Quantisierung
    T_q, RH_q, U_q = influx_phys_to_quantized(
        df[t_col].astype(float).values,
        df[rh_col].astype(float).values,
        df[bat_col].astype(float).values,
    )
    # Epoch in Sekunden — pandas 3.0 nutzt datetime64[us], daher /1e6
    ts_values = df[ts_col].values
    ts_int = ts_values.astype("int64")
    resolution = np.datetime_data(ts_values.dtype)[0]
    divisor = {"ns": 1e9, "us": 1e6, "ms": 1e3, "s": 1.0}.get(resolution, 1e6)

    result = pd.DataFrame({
        "influx_idx": np.arange(len(df)),
        "ts_utc": ts_values,
        "ts_epoch": ts_int / divisor,
        "T_q": T_q,
        "RH_q": RH_q,
        "U_q": U_q,
    })
    result["key"] = _make_keys(T_q, RH_q, U_q)
    return result


# ====================
# Matching (greedy, ordnungserhaltend)
# ====================

def greedy_anchor_match(sd_df: pd.DataFrame, influx_df: pd.DataFrame) -> pd.DataFrame:
    """Ordnungserhaltendes Greedy-Matching über exakte Tripel-Gleichheit.

    Returns DataFrame mit Spalten:
        sd_global_idx, segment_id, idx_in_segment, t_rel_s,
        influx_idx, ts_utc, ts_epoch, T_q, RH_q, U_q
    """
    cols = ["sd_global_idx", "segment_id", "idx_in_segment", "t_rel_s",
            "influx_idx", "ts_utc", "ts_epoch", "T_q", "RH_q", "U_q"]
    if len(sd_df) == 0 or len(influx_df) == 0:
        return pd.DataFrame(columns=cols)

    # Index: key → sortierte Liste von global_idx
    sd_index: Dict[str, List[int]] = defaultdict(list)
    sd_keys = sd_df["key"].values
    for i, key in enumerate(sd_keys):
        sd_index[key].append(i)

    sd_global_idx = sd_df["global_idx"].values
    sd_segment_id = sd_df["segment_id"].values
    sd_idx_in_seg = sd_df["idx_in_segment"].values
    sd_t_rel_s = sd_df["t_rel_s"].values
    sd_T_q = sd_df["T_q"].values
    sd_RH_q = sd_df["RH_q"].values
    sd_U_q = sd_df["U_q"].values

    influx_keys = influx_df["key"].values
    influx_idx = influx_df["influx_idx"].values
    influx_ts_utc = influx_df["ts_utc"].values
    influx_ts_epoch = influx_df["ts_epoch"].values

    # Greedy Matching
    anchors = []
    current_pos = -1
    for j in range(len(influx_df)):
        key = influx_keys[j]
        positions = sd_index.get(key)
        if positions is None:
            continue
        # Nächste SD-Position nach current_pos
        insert_idx = bisect_right(positions, current_pos)
        if insert_idx >= len(positions):
            continue
        chosen = positions[insert_idx]
        anchors.append((
            sd_global_idx[chosen],
            sd_segment_id[chosen],
            sd_idx_in_seg[chosen],
            sd_t_rel_s[chosen],
            influx_idx[j],
            influx_ts_utc[j],
            influx_ts_epoch[j],
            sd_T_q[chosen],
            sd_RH_q[chosen],
            sd_U_q[chosen],
        ))
        current_pos = chosen

    if not anchors:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(anchors, columns=cols)


# ====================
# Zeitmodell: gebinnte Median-Offset-Interpolation
# ====================

def _centered_polyfit(x: np.ndarray, y: np.ndarray, deg: int = 1):
    """Numerisch stabile Variante von np.polyfit: zentriert x und y vor dem Fit.

    Bei großen Offset-Werten (~1.7e9) ist np.polyfit ohne Zentrierung instabil.
    Returns: (coeffs, x_mean, y_mean) wobei coeffs auf zentrierte Daten passen.
    """
    x_mean, y_mean = x.mean(), y.mean()
    coeffs = np.polyfit(x - x_mean, y - y_mean, deg)
    return coeffs, x_mean, y_mean


def _eval_centered_linear(x: np.ndarray, coeffs, x_mean: float, y_mean: float) -> np.ndarray:
    """Wertet einen zentrierten linearen Fit aus: y = y_mean + b*(x - x_mean) + a."""
    b, a = coeffs
    return y_mean + a + b * (x - x_mean)


def _trim_outliers(x: np.ndarray, offset: np.ndarray, J: float) -> np.ndarray:
    """Iteratives Trimming: entfernt Anker, deren Offset stark vom lokalen Trend abweicht.

    Returns: Boolean-Maske der behaltenen Anker.
    """
    kept = np.ones(len(x), dtype=bool)
    for iteration in range(N_TRIM_ITER):
        xk, ok = x[kept], offset[kept]
        if len(xk) < max(2, MIN_ANCHORS_FOR_FIT):
            break
        # Linearer Trend als Referenz (zentriert für numerische Stabilität)
        coeffs, xm, om = _centered_polyfit(xk, ok, 1)
        trend = _eval_centered_linear(xk, coeffs, xm, om)
        residuals = np.abs(ok - trend)
        # Maximal MAX_TRIM_FRACTION der Punkte entfernen
        k = max(1, int(len(xk) * MAX_TRIM_FRACTION))
        threshold = np.sort(residuals)[-k]
        if threshold <= J:
            break
        # Nur die schlimmsten entfernen
        worst = residuals >= threshold
        kept_indices = np.where(kept)[0]
        kept[kept_indices[worst]] = False
    return kept


def fit_segment(anchors_seg: pd.DataFrame, J: float):
    """Berechnet die Zeitabbildung für ein Segment mittels gebinnter Median-Interpolation.

    Returns: (x_grid, offset_grid, rmse, jitter_med, jitter_p95, quality, n_anchors)
        x_grid, offset_grid: Stützstellen für np.interp
        Oder None wenn zu wenige Anker.
    """
    x = anchors_seg["t_rel_s"].values.astype(float)
    T = anchors_seg["ts_epoch"].values.astype(float)
    raw_offset = T - x  # offset_i = T_influx_i - x_rel_i

    if len(x) < MIN_ANCHORS_FOR_FIT:
        return None

    # 1. Trimming: Ausreißer entfernen
    kept = _trim_outliers(x, raw_offset, J)
    x_clean = x[kept]
    offset_clean = raw_offset[kept]

    if len(x_clean) < MIN_ANCHORS_FOR_FIT:
        return None

    # 2. In Zeitbins aufteilen und Median berechnen
    bin_width_s = BIN_HOURS * 3600.0
    x_min, x_max = x_clean.min(), x_clean.max()
    bin_edges = np.arange(x_min, x_max + bin_width_s, bin_width_s)

    x_grid = []
    offset_grid = []
    for i in range(len(bin_edges) - 1):
        mask = (x_clean >= bin_edges[i]) & (x_clean < bin_edges[i + 1])
        if mask.sum() >= MIN_ANCHORS_PER_BIN:
            x_grid.append(np.median(x_clean[mask]))
            # Jitter-Korrektur: median(T - x) - J/2 ≈ wahrer Offset
            offset_grid.append(np.median(offset_clean[mask]) - J / 2.0)

    if len(x_grid) < 2:
        # Fallback: ein einziger linearer Fit über alle Anker (zentriert)
        if len(x_clean) >= 2:
            coeffs, xm, om = _centered_polyfit(x_clean, offset_clean - J / 2.0, 1)
            x_grid = np.array([x_min, x_max])
            offset_grid = np.array([
                _eval_centered_linear(np.array([x_min]), coeffs, xm, om)[0],
                _eval_centered_linear(np.array([x_max]), coeffs, xm, om)[0],
            ])
        else:
            return None
    else:
        x_grid = np.array(x_grid)
        offset_grid = np.array(offset_grid)

    # 3. Qualitätsmetriken berechnen (auf allen sauberen Ankern)
    tau_interp = x_clean + np.interp(x_clean, x_grid, offset_grid)
    T_clean = x_clean + offset_clean
    jitter = T_clean - tau_interp
    jitter_clipped = np.clip(jitter, 0.0, J)
    y_mid = T_clean - J / 2.0
    rmse = float(np.sqrt(np.mean((y_mid - tau_interp) ** 2)))
    jitter_med = float(np.median(jitter_clipped))
    jitter_p95 = float(np.percentile(jitter_clipped, 95))
    n_anchors = int(kept.sum())

    if rmse <= RMSE_GOOD_S and n_anchors >= MIN_ANCHORS_GOOD:
        quality = "good"
    elif rmse <= RMSE_MED_S:
        quality = "medium"
    else:
        quality = "poor"

    # Drift in ppm (aus zentriertem linearem Fit über die Grid-Punkte)
    if len(x_grid) >= 2:
        coeffs_drift, _, _ = _centered_polyfit(x_grid, offset_grid, 1)
        drift_ppm = float(coeffs_drift[0]) * 1e6
    else:
        drift_ppm = 0.0

    return {
        "x_grid": x_grid,
        "offset_grid": offset_grid,
        "rmse": rmse,
        "jitter_med": jitter_med,
        "jitter_p95": jitter_p95,
        "quality": quality,
        "n_anchors": n_anchors,
        "drift_ppm": drift_ppm,
        "x_min": float(x_min),
        "x_max": float(x_max),
    }


def apply_time_mapping(sd_seg: pd.DataFrame, fit_result) -> Tuple[np.ndarray, np.ndarray]:
    """Wendet die Zeitabbildung auf alle SD-Punkte eines Segments an.

    Returns: (t_abs_epoch, quality_flags) Arrays
    """
    x = sd_seg["t_rel_s"].values.astype(float)
    n = len(x)

    if fit_result is None:
        return np.full(n, np.nan), np.array(["no_abs_time"] * n, dtype=object)

    x_grid = fit_result["x_grid"]
    offset_grid = fit_result["offset_grid"]
    quality = fit_result["quality"]

    # np.interp extrapoliert mit den Randwerten (flat extrapolation) — genau richtig
    offsets = np.interp(x, x_grid, offset_grid)
    t_abs = x + offsets

    flags = np.array([quality] * n, dtype=object)
    return t_abs, flags


# ====================
# Reports / Output
# ====================

def make_outputs(sd_df: pd.DataFrame, t_abs_all: np.ndarray,
                 quality_all: np.ndarray, out_dir: Path) -> pd.DataFrame:
    """Schreibt SD_absolute.csv."""
    t_abs_utc = []
    for epoch in t_abs_all:
        if np.isnan(epoch):
            t_abs_utc.append("")
        else:
            t_abs_utc.append(pd.to_datetime(epoch, unit="s", utc=True).isoformat())

    df_out = pd.DataFrame({
        "segment_id": sd_df["segment_id"].values,
        "idx_sd_global": sd_df["global_idx"].values,
        "idx_in_segment": sd_df["idx_in_segment"].values,
        "t_rel_s": np.round(sd_df["t_rel_s"].values, 3),
        "t_abs_utc": t_abs_utc,
        "T_C": [_format_val(v, TEMP_DECIMALS) for v in sd_df["T_q"].values],
        "RH_pct": [_format_val(v, RH_DECIMALS) for v in sd_df["RH_q"].values],
        "U_V": [_format_val(v, BAT_DECIMALS) for v in sd_df["U_q"].values],
        "quality_flag": quality_all,
    })
    df_out.to_csv(out_dir / OUT_SD_ABSOLUTE, index=False)
    return df_out


def make_segment_report(sd_df: pd.DataFrame, fit_results: Dict,
                        out_dir: Path) -> pd.DataFrame:
    """Schreibt Segment_report.csv."""
    rows = []
    for seg_id in sorted(sd_df["segment_id"].unique()):
        n_points = int((sd_df["segment_id"] == seg_id).sum())
        fit = fit_results.get(seg_id)
        if fit is not None:
            rows.append({
                "segment_id": seg_id,
                "n_points": n_points,
                "n_grid_points": len(fit["x_grid"]),
                "rmse_to_mid_s_median": fit["rmse"],
                "jitter_median_s_overall": fit["jitter_med"],
                "jitter_p95_s_overall": fit["jitter_p95"],
                "drift_ppm_median": fit["drift_ppm"],
                "quality_flag": fit["quality"],
                "notes": f"anchors:{fit['n_anchors']}",
            })
        else:
            rows.append({
                "segment_id": seg_id,
                "n_points": n_points,
                "n_grid_points": 0,
                "rmse_to_mid_s_median": None,
                "jitter_median_s_overall": None,
                "jitter_p95_s_overall": None,
                "drift_ppm_median": None,
                "quality_flag": "no_abs_time",
                "notes": "no_anchors",
            })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / OUT_SEGMENT_REPORT, index=False)
    return df


def make_anchor_report(anchors_df: pd.DataFrame, fit_results: Dict,
                       out_dir: Path) -> pd.DataFrame:
    """Schreibt Anchors_report.csv."""
    rows = []
    for seg_id, group in anchors_df.groupby("segment_id"):
        fit = fit_results.get(seg_id)
        for _, a in group.iterrows():
            tau = None
            jitter = None
            if fit is not None:
                tau_epoch = a["t_rel_s"] + np.interp(
                    a["t_rel_s"], fit["x_grid"], fit["offset_grid"]
                )
                tau = pd.to_datetime(tau_epoch, unit="s", utc=True)
                jitter = a["ts_epoch"] - tau_epoch
            rows.append({
                "segment_id": seg_id,
                "idx_sd_global": a["sd_global_idx"],
                "idx_in_segment": a["idx_in_segment"],
                "sd_t_rel_s": round(a["t_rel_s"], 3),
                "influx_ts_utc": pd.to_datetime(a["ts_utc"], utc=True).isoformat(),
                "T_q": _format_val(a["T_q"], TEMP_DECIMALS),
                "RH_q": _format_val(a["RH_q"], RH_DECIMALS),
                "U_q": _format_val(a["U_q"], BAT_DECIMALS),
                "tau_abs_utc": tau.isoformat() if tau is not None else "",
                "jitter_s": jitter,
            })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / OUT_ANCHOR_REPORT, index=False)
    return df


def make_plausibility_report(sd_df: pd.DataFrame, influx_df: pd.DataFrame,
                             anchors_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Schreibt Plausibility_report.csv."""
    from collections import Counter
    sd_counts = Counter(sd_df["key"].values)
    in_counts = Counter(influx_df["key"].values)
    offenders = sum(1 for k, v in in_counts.items() if v > sd_counts.get(k, 0))
    coverage = len(anchors_df) / max(1, len(influx_df))
    df = pd.DataFrame([{
        "influx_count": len(influx_df),
        "sd_count": len(sd_df),
        "matched_influx_fraction": round(coverage, 6),
        "influx_values_missing_on_sd": offenders,
    }])
    df.to_csv(out_dir / OUT_PLAUSIBILITY_REPORT, index=False)
    return df


# ====================
# Main
# ====================

def main():
    cli_args = _apply_cli_overrides()
    _apply_config_overrides(cli_args.config)

    sd_paths, influx_paths, out_dir = resolve_paths()
    sd_df = read_sd_files(sd_paths)
    influx_df = read_influx_files(influx_paths)
    print(f"[INFO] SD-Punkte: {len(sd_df)} | Influx-Punkte: {len(influx_df)}")

    # Matching
    anchors_df = greedy_anchor_match(sd_df, influx_df)
    print(f"[INFO] Anker gesamt: {len(anchors_df)}")

    # Plausibilität
    make_plausibility_report(sd_df, influx_df, anchors_df, out_dir)

    # Pro Segment: Zeitabbildung berechnen
    fit_results: Dict = {}
    t_abs_all = np.full(len(sd_df), np.nan)
    quality_all = np.array(["no_abs_time"] * len(sd_df), dtype=object)

    for seg_id in sorted(sd_df["segment_id"].unique()):
        seg_anchors = anchors_df[anchors_df["segment_id"] == seg_id]
        seg_mask = sd_df["segment_id"] == seg_id

        print(f"[INFO] Segment {seg_id}: {seg_mask.sum()} Punkte, {len(seg_anchors)} Anker")

        fit = fit_segment(seg_anchors, J_MAX_SECONDS)
        fit_results[seg_id] = fit

        t_abs_seg, quality_seg = apply_time_mapping(sd_df[seg_mask], fit)
        t_abs_all[seg_mask] = t_abs_seg
        quality_all[seg_mask] = quality_seg

        if fit is not None:
            print(f"         → {fit['quality']} (RMSE={fit['rmse']:.2f}s, "
                  f"drift={fit['drift_ppm']:.1f}ppm, "
                  f"grid={len(fit['x_grid'])} Stützstellen)")
        else:
            print(f"         → no_abs_time")

    # Output schreiben
    make_outputs(sd_df, t_abs_all, quality_all, out_dir)
    make_segment_report(sd_df, fit_results, out_dir)
    make_anchor_report(anchors_df, fit_results, out_dir)

    print("=== dl-sd-card-date v2 ===")
    print("[INFO] Dateien geschrieben:")
    print(f"  SD_absolute:        {out_dir / OUT_SD_ABSOLUTE}")
    print(f"  Segment_report:     {out_dir / OUT_SEGMENT_REPORT}")
    print(f"  Anchors_report:     {out_dir / OUT_ANCHOR_REPORT}")
    print(f"  Plausibility_report: {out_dir / OUT_PLAUSIBILITY_REPORT}")


if __name__ == "__main__":
    main()
