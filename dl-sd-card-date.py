#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
dl-sd-card-date — Absolute Datierung der SD-Karten-Rohdaten gegen Influx-Export
-------------------------------------------------------------------------------
- Windows-freundliche I/O: Input\ (alle CSV-Inputs), Output\ (alle Resultate)
- 6h Fit-Anker-Selektion pro Segment (median/first/last) + Start/End-Pinning
- Stitching nur innerhalb der Fenster (konfigurierbar)
- Exakte Tripel-Matches mit Quantisierung (T=10, RH=8, U=3; ROUND_HALF_UP)
- Ordungserhaltendes Matching; Duplikat-Timestamps verengen Korridor nicht
- CLI-Overrides: --input-dir, --output-dir, --config

Konfig via YAML (optional): dl-sd-card-date.yaml im selben Ordner (oder via --config)
"""

from __future__ import annotations
import sys, os, math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from pathlib import Path
from collections import defaultdict
from bisect import bisect_right
from decimal import Decimal, ROUND_HALF_UP, getcontext

import pandas as pd
import numpy as np

# ====================
# Defaults (per YAML/CLI überschreibbar)
# ====================

INPUT_DIR  = "Input"
OUTPUT_DIR = "Output"

SD_GLOB      = "SGS_SDCard_raw_*.csv"     # Rohdaten SD-Card (ohne Header)
INFLUX_GLOB  = "Sensors_Raw_*.csv"        # Influx/Grafana Exporte (mit Header)

TIMEZONE = "UTC"
NOMINAL_INTERVAL_S = 120

# Quantisierung physikalischer Werte
TEMP_DECIMALS = 10
RH_DECIMALS = 8
BAT_DECIMALS = 3
ROUNDING = "ROUND_HALF_UP"

# Jitterfenster
J_MAX_SECONDS = 8.0

# Matching‑Robustheit
MAX_SD_CANDIDATES = 50
B_INIT_MIN = 0.5
B_INIT_MAX = 1.5
DT_DUP_EPS = 1.0

# Intervall‑Fit / Trimming
N_TRIM_ITER = 3
MAX_TRIM_FRACTION = 0.02
MIN_ANCHORS_FOR_FIT = 3

# Fensterung (innerhalb eines Segments)
WINDOW_DAYS = 21.0
WINDOW_OVERLAP_HOURS = 48.0
MIN_ANCHORS_PER_WINDOW = 30

# 6h Fit‑Anker‑Selektion (pro Segment)
FIT_ANCHOR_GRID_HOURS = 6.0
EDGE_ANCHOR_WINDOW_MIN = 60.0
ANCHOR_PICK = "median"   # "median" | "first" | "last"

# Fallback‑Kontrollen
WINDOW_FALLBACK_SEGMENT_FIT = True
MIN_ANCHORS_FOR_SEGMENT_FALLBACK = 30
MIN_FALLBACK_COVERAGE_FRAC = 0.5

# Qualitäts‑Schwellen
MIN_ANCHORS_GOOD = 20

# Segment-Fallback über gesamte Segmentbreite ausdehnen?
FALLBACK_EXTEND_TO_SEGMENT_BOUNDS = True
# Wenn mind. ein Fit existiert, dehne erste/letzte Fit-x-Grenzen auf Segment-Grenzen aus
EXTEND_FITS_TO_SEGMENT_BOUNDS = True

# Stitching
STITCH_WITHIN_ONLY = False 
STITCH_PAD_S = 1000000000000   # sehr groß, damit jeder Punkt einem Fit "am nächsten" zugeordnet wird


# --- Robustness knobs (added) ---
# If a 6h-fit window has too few fit-anchors, try ALL anchors in that window:
MIN_ANCHORS_PER_WINDOW_ALL = 10   # fallback threshold using all anchors in the window
# If a segment has too few fit-anchors overall, allow segment-wide fit using ALL anchors:
MIN_ANCHORS_FOR_SEGMENT_FALLBACK_ALL = 15

# Output-Dateien (unter OUTPUT_DIR)
OUT_SD_ABSOLUTE = "SD_absolute.csv"
OUT_SEGMENT_REPORT = "Segment_report.csv"
OUT_ANCHOR_REPORT = "Anchors_report.csv"
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
    args, _unknown = parser.parse_known_args()
    if args.input_dir:
        globals()["INPUT_DIR"] = args.input_dir
    if args.output_dir:
        globals()["OUTPUT_DIR"] = args.output_dir
    return args


def _simple_yaml_scalar(s: str):
    s = s.strip()
    if s == "" or s.lower() == "null": return None
    if s.lower() in ("true","false"): return s.lower()=="true"
    # strip quotes if present
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    # try int / float
    try:
        if "." in s or "e" in s.lower():
            return float(s)
        return int(s)
    except Exception:
        return s

def _simple_yaml_load(text: str):
    # very simple, flat key:value parser (for this project's config)
    data = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"): 
            continue
        # support inline comments: key: value  # comment
        if "#" in line:
            # keep '#' inside quotes
            in_s = False; in_d = False; idx = None
            for i,ch in enumerate(line):
                if ch == "'" and not in_d: in_s = not in_s
                elif ch == '"' and not in_s: in_d = not in_d
                elif ch == "#" and not in_s and not in_d:
                    idx = i; break
            if idx is not None:
                line = line[:idx].rstrip()
        if ":" not in line: 
            continue
        k, v = line.split(":", 1)
        key = k.strip()
        val = _simple_yaml_scalar(v)
        data[key] = val
    return data

def _apply_config_overrides(config_path: Optional[str]):
    # Determine config path
    if not config_path:
        script_dir = Path(os.path.abspath(os.path.dirname(__file__)))
        default_yaml = script_dir / "dl-sd-card-date.yaml"
        if default_yaml.exists():
            config_path = str(default_yaml)
        else:
            return
    p = Path(config_path)
    if not p.exists():
        print(f"[WARN] Config file not found: {config_path} — using built-in defaults.", file=sys.stderr)
        return
    text = p.read_text(encoding="utf-8")

    # Try YAML if extension is .yaml/.yml and PyYAML available; otherwise use simple parser.
    ext = p.suffix.lower()
    data = None
    if ext in (".yaml",".yml"):
        try:
            import yaml  # type: ignore
            data = yaml.safe_load(text)
        except Exception as e:
            print(f"[INFO] PyYAML not available or failed ({e}); using simple YAML parser.", file=sys.stderr)
            data = _simple_yaml_load(text)
    elif ext == ".json":
        import json
        data = json.loads(text)
    else:
        # Try YAML first, then JSON
        try:
            import yaml  # type: ignore
            data = yaml.safe_load(text)
        except Exception:
            try:
                import json
                data = json.loads(text)
            except Exception:
                print("[WARN] Could not parse config (neither YAML nor JSON). Using defaults.", file=sys.stderr)
                return

    if not isinstance(data, dict):
        print("[WARN] Config root must be a mapping/object. Ignoring.", file=sys.stderr); 
        return

    def ov(name, cast=None):
        if name in data:
            val = data[name]
            if cast is not None and val is not None:
                try: val = cast(val)
                except Exception: 
                    print(f"[WARN] could not cast {name} -> {val}", file=sys.stderr); 
                    return
            globals()[name] = val

    def _as_str(x): return str(x) if x is not None else None
    ov("INPUT_DIR", _as_str); ov("OUTPUT_DIR", _as_str)
    ov("SD_GLOB", _as_str); ov("INFLUX_GLOB", _as_str)
    ov("TIMEZONE", _as_str); ov("NOMINAL_INTERVAL_S", int)
    ov("TEMP_DECIMALS", int); ov("RH_DECIMALS", int); ov("BAT_DECIMALS", int); ov("ROUNDING", _as_str)
    ov("J_MAX_SECONDS", float)
    ov("MAX_SD_CANDIDATES", int); ov("B_INIT_MIN", float); ov("B_INIT_MAX", float); ov("DT_DUP_EPS", float)
    ov("N_TRIM_ITER", int); ov("MAX_TRIM_FRACTION", float); ov("MIN_ANCHORS_FOR_FIT", int)
    ov("WINDOW_DAYS", float); ov("WINDOW_OVERLAP_HOURS", float); ov("MIN_ANCHORS_PER_WINDOW", int)
    ov("FIT_ANCHOR_GRID_HOURS", float); ov("EDGE_ANCHOR_WINDOW_MIN", float); ov("ANCHOR_PICK", _as_str)
    ov("WINDOW_FALLBACK_SEGMENT_FIT", bool)
    ov("MIN_ANCHORS_FOR_SEGMENT_FALLBACK", int); ov("MIN_FALLBACK_COVERAGE_FRAC", float)
    ov("MIN_ANCHORS_GOOD", int)
    ov("STITCH_WITHIN_ONLY", bool); ov("STITCH_PAD_S", float)
    ov("OUT_SD_ABSOLUTE", _as_str); ov("OUT_SEGMENT_REPORT", _as_str); ov("OUT_ANCHOR_REPORT", _as_str); ov("OUT_PLAUSIBILITY_REPORT", _as_str)
    ov("FALLBACK_EXTEND_TO_SEGMENT_BOUNDS", bool);
    ov("MIN_ANCHORS_PER_WINDOW_ALL", int); ov("MIN_ANCHORS_FOR_SEGMENT_FALLBACK_ALL", int);
    ov("EXTEND_FITS_TO_SEGMENT_BOUNDS", bool)

# ====================
# Datenklassen
# ====================

@dataclass(frozen=True)
class SDTripleQ:
    T_q: str
    RH_q: str
    U_q: str

@dataclass
class SDPoint:
    global_idx: int
    segment_id: int
    idx_in_segment: int
    t1024: int
    t_rel_s: float
    triple_q: SDTripleQ

@dataclass
class InfluxPoint:
    idx: int
    ts_utc: pd.Timestamp
    triple_q: SDTripleQ

@dataclass
class Anchor:
    sd_global_idx: int
    sd_segment_id: int
    sd_idx_in_segment: int
    sd_t_rel_s: float
    influx_idx: int
    influx_ts_utc: pd.Timestamp
    triple_q: SDTripleQ

@dataclass
class WindowFit:
    seg_id: int
    start_ts: pd.Timestamp
    end_ts: pd.Timestamp
    x_min: float
    x_max: float
    x_center: float
    a: float
    b: float
    rmse_mid: Optional[float]
    jitter_med: Optional[float]
    jitter_p95: Optional[float]
    n_anchors: int
    quality: str
    notes: str

# ====================
# Hilfsfunktionen
# ====================

def _qunit(dp: int) -> Decimal: return Decimal(1).scaleb(-dp)
def _quantize_half_up(x: Decimal, dp: int) -> Decimal: return x.quantize(_qunit(dp), rounding=ROUND_HALF_UP)
def _format_dec(x: Decimal) -> str: return format(x, 'f')

def sd_raw_to_quantized(temp_raw: int, rh_raw: int, bat_raw: int,
                        dT: int, dRH: int, dU: int) -> SDTripleQ:
    T = (Decimal(temp_raw) * Decimal(175) / Decimal(65535)) - Decimal(45)
    RH = (Decimal(rh_raw) * Decimal(100) / Decimal(65535))
    U = (Decimal(bat_raw) / Decimal(1000))
    Tq = _quantize_half_up(T, dT); RHq = _quantize_half_up(RH, dRH); Uq = _quantize_half_up(U, dU)
    return SDTripleQ(_format_dec(Tq), _format_dec(RHq), _format_dec(Uq))

def influx_phys_to_quantized(T_val: str, RH_val: str, U_val: str,
                             dT: int, dRH: int, dU: int) -> SDTripleQ:
    T = _quantize_half_up(Decimal(T_val), TEMP_DECIMALS)
    RH = _quantize_half_up(Decimal(RH_val), RH_DECIMALS)
    U = _quantize_half_up(Decimal(U_val), BAT_DECIMALS)
    return SDTripleQ(_format_dec(T), _format_dec(RH), _format_dec(U))

def triple_key(q: SDTripleQ) -> str: return f"{q.T_q}|{q.RH_q}|{q.U_q}"

# ====================
# IO
# ====================

def resolve_paths() -> Tuple[List[Path], List[Path], Path]:
    """Find Input files under INPUT_DIR and ensure OUTPUT_DIR exists."""
    script_dir = Path(os.path.abspath(os.path.dirname(__file__)))
    in_dir = Path(INPUT_DIR)
    out_dir = Path(OUTPUT_DIR)
    # Falls relative Pfade: relativ zum Scriptordner interpretieren
    if not in_dir.is_absolute():
        in_dir = (script_dir / in_dir).resolve()
    if not out_dir.is_absolute():
        out_dir = (script_dir / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] INPUT_DIR:  {in_dir}")
    print(f"[INFO] OUTPUT_DIR: {out_dir}")

    sd_files = sorted(in_dir.glob(SD_GLOB))
    influx_files = sorted(in_dir.glob(INFLUX_GLOB))
    if not sd_files:
        print(f"[WARN] Keine SD-Dateien mit Muster {SD_GLOB} in {in_dir}", file=sys.stderr)
    if not influx_files:
        print(f"[WARN] Keine Influx-Dateien mit Muster {INFLUX_GLOB} in {in_dir}", file=sys.stderr)
    return sd_files, influx_files, out_dir

def read_sd_files(sd_paths: List[Path]) -> List[SDPoint]:
    sd_points: List[SDPoint] = []
    segment_id = 0; global_idx = 0; last_t_rel = None
    for path in sd_paths:
        df = pd.read_csv(path, header=None, names=["t1024","temp_raw","rh_raw","bat_raw"], encoding="utf-8-sig").dropna()
        df = df.astype({"t1024":"int64","temp_raw":"int64","rh_raw":"int64","bat_raw":"int64"})
        df["t_rel_s"] = df["t1024"] / 1024.0
        idx_in_segment = 0
        for _, row in df.iterrows():
            t_rel = float(row["t_rel_s"])
            if last_t_rel is not None and (t_rel - last_t_rel) < 0:
                segment_id += 1; idx_in_segment = 0
            tq = sd_raw_to_quantized(int(row["temp_raw"]), int(row["rh_raw"]), int(row["bat_raw"]),
                                     TEMP_DECIMALS, RH_DECIMALS, BAT_DECIMALS)
            sd_points.append(SDPoint(global_idx, segment_id, idx_in_segment,
                                     int(row["t1024"]), t_rel, tq))
            last_t_rel = t_rel; global_idx += 1; idx_in_segment += 1
    return sd_points

def detect_influx_columns(df: pd.DataFrame) -> Tuple[str, str, str, str, str]:
    cols = list(df.columns)
    ts_col = cols[0]; tz_col = cols[1] if len(cols) > 1 else None
    bat_col = next((c for c in cols if "battery" in c.lower()), None)
    rh_col  = next((c for c in cols if "humid" in c.lower()), None)
    t_col   = next((c for c in cols if "temp" in c.lower()), None)
    if not (bat_col and rh_col and t_col):
        raise ValueError(f"Influx CSV: konnte Spalten nicht erkennen. Spalten: {cols}")
    return ts_col, tz_col, bat_col, rh_col, t_col

def read_influx_files(influx_paths: List[Path]) -> List[InfluxPoint]:
    frames = [pd.read_csv(p, encoding="utf-8-sig") for p in influx_paths]
    if not frames:
        return []
    df = pd.concat(frames, ignore_index=True)
    ts_col, tz_col, bat_col, rh_col, t_col = detect_influx_columns(df)
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.dropna(subset=[ts_col]).sort_values(by=ts_col).reset_index(drop=True)
    df[bat_col] = df[bat_col].astype(str); df[rh_col] = df[rh_col].astype(str); df[t_col] = df[t_col].astype(str)
    pts: List[InfluxPoint] = []
    for i, row in df.iterrows():
        pts.append(InfluxPoint(i, row[ts_col],
                               influx_phys_to_quantized(row[t_col], row[rh_col], row[bat_col],
                                                        TEMP_DECIMALS, RH_DECIMALS, BAT_DECIMALS)))
    return pts

# ====================
# Matching (greedy, ordnungserhaltend)
# ====================

def build_sd_index(sd_points: List[SDPoint]) -> Dict[str, List[int]]:
    idx = defaultdict(list)
    for p in sd_points: idx[triple_key(p.triple_q)].append(p.global_idx)
    return idx

def prepare_sd_arrays(sd_points: List[SDPoint]):
    max_idx = max(p.global_idx for p in sd_points) if sd_points else -1
    seg_ids = np.empty(max_idx+1, dtype=np.int64)
    idx_in_seg = np.empty(max_idx+1, dtype=np.int64)
    t_rel = np.empty(max_idx+1, dtype=np.float64)
    triples = [None]*(max_idx+1)
    for p in sd_points:
        seg_ids[p.global_idx] = p.segment_id
        idx_in_seg[p.global_idx] = p.idx_in_segment
        t_rel[p.global_idx] = p.t_rel_s
        triples[p.global_idx] = p.triple_q
    return seg_ids, idx_in_seg, t_rel, triples


def robust_anchor_match(sd_points: List[SDPoint],
                        influx_points: List[InfluxPoint],
                        J: float) -> List[Anchor]:
    """
    Lenient, order-preserving greedy matching:
    - exact triple equality (after quantization) is required
    - picks the next SD occurrence after the previous SD position
    - no b/jitter gating; outliers handled later by fit_with_trim
    """
    sd_index = build_sd_index(sd_points)
    sd_seg_ids, sd_idx_in_seg, sd_t_rel, sd_triples = prepare_sd_arrays(sd_points)
    anchors: List[Anchor] = []
    current_sd_pos = -1

    for infl in influx_points:
        key = triple_key(infl.triple_q)
        positions = sd_index.get(key, [])
        if not positions:
            continue
        j0 = bisect_right(positions, current_sd_pos)
        if j0 >= len(positions):
            # no SD occurrence after current position
            continue
        chosen = positions[j0]
        anchors.append(Anchor(
            sd_global_idx=chosen,
            sd_segment_id=int(sd_seg_ids[chosen]),
            sd_idx_in_segment=int(sd_idx_in_seg[chosen]),
            sd_t_rel_s=float(sd_t_rel[chosen]),
            influx_idx=infl.idx,
            influx_ts_utc=infl.ts_utc,
            triple_q=infl.triple_q
        ))
        current_sd_pos = chosen
    return anchors

# ====================
# 6h Fit‑Anker‑Selektion
# ====================

def select_fit_anchors_for_segment(anchors_sorted: List[Anchor]) -> List[Anchor]:
    if not anchors_sorted: return []
    w0 = anchors_sorted[0].influx_ts_utc
    w1 = anchors_sorted[-1].influx_ts_utc
    grid = pd.Timedelta(hours=FIT_ANCHOR_GRID_HOURS)
    edge = pd.Timedelta(minutes=EDGE_ANCHOR_WINDOW_MIN)

    selected: List[Anchor] = []
    used_ids = set()

    # Startanker
    start_candidates = [a for a in anchors_sorted if a.influx_ts_utc <= w0 + edge]
    if start_candidates:
        a0 = _pick_in_window(start_candidates, ANCHOR_PICK)
        selected.append(a0); used_ids.add(id(a0))

    # Grid-Selection
    t = (w0.floor(f"{int(FIT_ANCHOR_GRID_HOURS)}H") if FIT_ANCHOR_GRID_HOURS >= 1.0 else w0)
    if t < w0: t = w0
    while t <= w1:
        t_next = t + grid
        block = [a for a in anchors_sorted if (a.influx_ts_utc >= t and a.influx_ts_utc < t_next)]
        if block:
            a_pick = _pick_in_window(block, ANCHOR_PICK)
            if id(a_pick) not in used_ids:
                selected.append(a_pick); used_ids.add(id(a_pick))
        t = t_next

    # Endanker
    end_candidates = [a for a in anchors_sorted if a.influx_ts_utc >= w1 - edge]
    if end_candidates:
        aE = _pick_in_window(end_candidates, ANCHOR_PICK)
        if id(aE) not in used_ids:
            selected.append(aE); used_ids.add(id(aE))

    selected.sort(key=lambda a: a.influx_ts_utc.value)
    return selected

def _pick_in_window(block: List[Anchor], policy: str) -> Anchor:
    if policy == "first":
        return block[0]
    if policy == "last":
        return block[-1]
    bsorted = sorted(block, key=lambda a: a.influx_ts_utc.value)
    return bsorted[len(bsorted)//2]

# ====================
# Intervall-Fit (mit Trimming)
# ====================

def _interval_bounds_for_a(b: float, x: np.ndarray, T: np.ndarray, J: float) -> Tuple[float, float]:
    A_lower = np.max(T - J - b * x)
    A_upper = np.min(T - b * x)
    return A_lower, A_upper

def _project_ls_to_feasible(a0: float, b0: float, x: np.ndarray, T: np.ndarray, J: float) -> Tuple[float, float, bool]:
    def try_range(center, rel):
        bs = np.linspace(center*(1-rel), center*(1+rel), 401) if abs(center) > 1e-12 else np.linspace(0.9, 1.1, 401)
        best = (math.inf, None, None); feasible_any = False
        for b in bs:
            A_lower, A_upper = _interval_bounds_for_a(b, x, T, J)
            if A_lower <= A_upper:
                feasible_any = True
                a_star = a0
                if a_star < A_lower: a_star = A_lower
                if a_star > A_upper: a_star = A_upper
                cost = (a_star - a0)**2 + (b - b0)**2
                if cost < best[0]: best = (cost, a_star, b)
        return best if feasible_any else None

    y_mid = T - J*0.5
    b0_hat, a0_hat = np.polyfit(x, y_mid, 1)
    out = try_range(b0_hat, 0.01) or try_range(1.0, 0.02) or try_range(1.0, 0.1)
    if not out: return (a0_hat, b0_hat, False)
    _, a_star, b_star = out
    return (float(a_star), float(b_star), True)

def fit_with_trim(anchors: List[Anchor], J: float, min_anchors: int):
    if len(anchors) < max(2, min_anchors):
        return (math.nan, math.nan, None, None, None, "no_abs_time", "too_few_anchors", [])
    T = np.array([a.influx_ts_utc.timestamp() for a in anchors], dtype=float)
    x = np.array([a.sd_t_rel_s for a in anchors], dtype=float)
    kept = np.arange(len(anchors)); notes = []
    for it in range(N_TRIM_ITER+1):
        xk = x[kept]; Tk = T[kept]
        y_mid = Tk - J*0.5
        b0, a0 = np.polyfit(xk, y_mid, 1)
        a, b, feasible = _project_ls_to_feasible(a0, b0, xk, Tk, J)
        if feasible:
            tau = a + b * xk
            rmse_mid = float(np.sqrt(np.mean((y_mid - tau)**2)))
            j = Tk - tau; j_clip = np.clip(j, 0.0, J)
            j_med = float(np.median(j_clip)); j_p95 = float(np.percentile(j_clip, 95))
            if rmse_mid <= 12.0: q = "good"
            elif rmse_mid <= 20.0: q = "medium"
            else: q = "poor"
            if q == "good" and len(kept) < MIN_ANCHORS_GOOD:
                q = "medium"
            return (a, b, rmse_mid, j_med, j_p95, q, ";".join(notes), kept.tolist())
        if it >= N_TRIM_ITER or len(kept) <= max(2, min_anchors):
            return (a0, b0, None, None, None, "no_abs_time", "no_feasible_after_trim", kept.tolist())
        tau0 = a0 + b0 * xk
        below = (Tk - J) - tau0
        above = tau0 - Tk
        viol = np.maximum(below, above)
        k = max(1, int(len(kept) * MAX_TRIM_FRACTION))
        worst_idx = np.argsort(viol)[-k:]
        notes.append(f"iter{it}_trim{len(worst_idx)}")
        kept = np.delete(kept, worst_idx)
    return (math.nan, math.nan, None, None, None, "no_abs_time", "internal_error", kept.tolist())

# ====================
# Fensterung & Stitching
# ====================
def _extend_fits_to_segment_bounds(fits: List[WindowFit], seg_bounds: SegmentBounds) -> List[WindowFit]:
    if not fits: 
        return fits
    # Kopie erzeugen (immutables vermeiden)
    out = list(fits)
    # Sortiert nach x_center (sollte schon segmentweit sein)
    out.sort(key=lambda f: f.x_center)
    # Ersten und letzten Fit auf Segmentgrenzen ausdehnen
    first = out[0]; last = out[-1]
    changed = False
    if EXTEND_FITS_TO_SEGMENT_BOUNDS:
        if first.x_min > seg_bounds.x_min:
            first.x_min = float(seg_bounds.x_min); changed = True
        if last.x_max < seg_bounds.x_max:
            last.x_max = float(seg_bounds.x_max); changed = True
    return out


@dataclass
class SegmentBounds:
    x_min: float
    x_max: float

def build_time_windows(anchors_sorted: List[Anchor], window_days: float, overlap_hours: float) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    if not anchors_sorted: return []
    first = anchors_sorted[0].influx_ts_utc
    last = anchors_sorted[-1].influx_ts_utc
    win = pd.Timedelta(days=window_days)
    step = win - pd.Timedelta(hours=overlap_hours)
    if step <= pd.Timedelta(0):
        step = pd.Timedelta(hours=1)
    windows = []
    t0 = first
    while t0 <= last:
        t1 = t0 + win
        windows.append((t0, t1))
        t0 = t0 + step
    return windows


def fit_windows_for_segment(seg_id: int, anchors_sorted: List[Anchor], fit_anchors_sorted: List[Anchor], seg_bounds: SegmentBounds) -> List[WindowFit]:
    windows = build_time_windows(anchors_sorted, WINDOW_DAYS, WINDOW_OVERLAP_HOURS)
    fits: List[WindowFit] = []
    for (w0, w1) in windows:
        # primary: use selected FIT anchors within window
        sub_fit = [a for a in fit_anchors_sorted if (a.influx_ts_utc >= w0 and a.influx_ts_utc <= w1)]
        anchors_for_this_window = None
        if len(sub_fit) >= MIN_ANCHORS_PER_WINDOW:
            anchors_for_this_window = sub_fit
        else:
            # fallback: use ALL anchors in this window if enough
            sub_all = [a for a in anchors_sorted if (a.influx_ts_utc >= w0 and a.influx_ts_utc <= w1)]
            if len(sub_all) >= MIN_ANCHORS_PER_WINDOW_ALL:
                anchors_for_this_window = sub_all

        if anchors_for_this_window is None:
            continue

        a, b, rmse, j_med, j_p95, q, notes, kept = fit_with_trim(anchors_for_this_window, J_MAX_SECONDS, MIN_ANCHORS_FOR_FIT)
        if q == "no_abs_time":
            continue
        xs = [a_.sd_t_rel_s for a_ in anchors_for_this_window]
        xm, xM = float(min(xs)), float(max(xs))
        xc = 0.5*(xm + xM)
        fits.append(WindowFit(seg_id, w0, w1, xm, xM, xc, a, b, rmse, j_med, j_p95, len(anchors_for_this_window), q, notes))

    # segment-wide fallbacks if no windows
    if not fits and WINDOW_FALLBACK_SEGMENT_FIT:
        # (a) try with FIT anchors across entire segment
        if len(fit_anchors_sorted) >= MIN_ANCHORS_FOR_SEGMENT_FALLBACK:
            xs = [a_.sd_t_rel_s for a_ in fit_anchors_sorted]
            if xs:
                xm, xM = float(min(xs)), float(max(xs))
                seg_span = (xM - xm)
                coverage_frac = 1.0 if seg_span > 0 else 0.0
                if coverage_frac >= MIN_FALLBACK_COVERAGE_FRAC:
                    a, b, rmse, j_med, j_p95, q, notes, kept = fit_with_trim(fit_anchors_sorted, J_MAX_SECONDS, MIN_ANCHORS_FOR_FIT)
                    if q != "no_abs_time":
                        xc = 0.5*(xm+xM)
                    if FALLBACK_EXTEND_TO_SEGMENT_BOUNDS:
                        xm, xM = float(seg_bounds.x_min), float(seg_bounds.x_max)
                        xc = 0.5*(xm+xM)
                    fits.append(WindowFit(seg_id, fit_anchors_sorted[0].influx_ts_utc, fit_anchors_sorted[-1].influx_ts_utc,
                                          xm, xM, xc, a, b, rmse, j_med, j_p95, len(fit_anchors_sorted), q, notes))
        # (b) if still no fits, try ALL anchors across segment
        if not fits and len(anchors_sorted) >= MIN_ANCHORS_FOR_SEGMENT_FALLBACK_ALL:
            xs = [a_.sd_t_rel_s for a_ in anchors_sorted]
            if xs:
                xm, xM = float(min(xs)), float(max(xs))
                if (xM - xm) > 0:
                    a, b, rmse, j_med, j_p95, q, notes, kept = fit_with_trim(anchors_sorted, J_MAX_SECONDS, MIN_ANCHORS_FOR_FIT)
                    if q != "no_abs_time":
                        xc = 0.5*(xm+xM)
                    if FALLBACK_EXTEND_TO_SEGMENT_BOUNDS:
                        xm, xM = float(seg_bounds.x_min), float(seg_bounds.x_max)
                        xc = 0.5*(xm+xM)
                    fits.append(WindowFit(seg_id, anchors_sorted[0].influx_ts_utc, anchors_sorted[-1].influx_ts_utc,
                                          xm, xM, xc, a, b, rmse, j_med, j_p95, len(anchors_sorted), q, notes))
    return fits


def choose_window_for_point(x: float, fits: List[WindowFit]) -> Optional[WindowFit]:
    if not fits: return None
    pad = STITCH_PAD_S if not STITCH_WITHIN_ONLY else 0.0
    candidates = [f for f in fits if (x >= f.x_min - pad and x <= f.x_max + pad)]
    if STITCH_WITHIN_ONLY:
        candidates = [f for f in candidates if (x >= f.x_min and x <= f.x_max)]
    if candidates:
        return min(candidates, key=lambda f: abs(x - f.x_center))
    return None

# ====================
# Reports / IO
# ====================

def make_outputs(sd_points: List[SDPoint],
                 fits_by_segment: Dict[int, List[WindowFit]],
                 out_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows_abs = []
    for p in sd_points:
        fits = fits_by_segment.get(p.segment_id, [])
        wf = choose_window_for_point(p.t_rel_s, fits)
        if wf is not None:
            t_abs_epoch = wf.a + wf.b * p.t_rel_s
            t_abs = pd.to_datetime(t_abs_epoch, unit="s", utc=True)
            q = wf.quality
        else:
            t_abs = pd.NaT; q = "no_abs_time"
        rows_abs.append({
            "segment_id": p.segment_id,
            "idx_sd_global": p.global_idx,
            "idx_in_segment": p.idx_in_segment,
            "t_rel_s": round(p.t_rel_s, 3),
            "t_abs_utc": t_abs.isoformat() if pd.notna(t_abs) else "",
            "T_C": p.triple_q.T_q,
            "RH_pct": p.triple_q.RH_q,
            "U_V": p.triple_q.U_q,
            "quality_flag": q
        })
    df_abs = pd.DataFrame(rows_abs)

    rows_seg = []
    seg_ids = sorted({p.segment_id for p in sd_points})
    for seg_id in seg_ids:
        fits = fits_by_segment.get(seg_id, [])
        if fits:
            rmse_vals = [f.rmse_mid for f in fits if f.rmse_mid is not None]
            jitter_meds = [f.jitter_med for f in fits if f.jitter_med is not None]
            jitter_p95s = [f.jitter_p95 for f in fits if f.jitter_p95 is not None]
            drift_ppm_vals = [ (f.b - 1.0)*1e6 for f in fits ]
            rmse_med = float(np.median(rmse_vals)) if rmse_vals else None
            jitter_med_all = float(np.median(jitter_meds)) if jitter_meds else None
            jitter_p95_all = float(np.median(jitter_p95s)) if jitter_p95s else None
            drift_ppm_med = float(np.median(drift_ppm_vals)) if drift_ppm_vals else None
            if rmse_med is not None and rmse_med <= 12.0 and np.median([f.n_anchors for f in fits]) >= MIN_ANCHORS_GOOD:
                qflag = "good"
            elif rmse_med is not None and rmse_med <= 20.0:
                qflag = "medium"
            else:
                qflag = "poor"
            notes = f"windows:{len(fits)}"
        else:
            qflag = "no_abs_time"; rmse_med = None; jitter_med_all = None; jitter_p95_all = None; drift_ppm_med = None; notes = "no_windows"
        n_points = sum(1 for p in sd_points if p.segment_id==seg_id)
        rows_seg.append({
            "segment_id": seg_id,
            "n_points": n_points,
            "n_windows": len(fits),
            "rmse_to_mid_s_median": rmse_med,
            "jitter_median_s_overall": jitter_med_all,
            "jitter_p95_s_overall": jitter_p95_all,
            "drift_ppm_median": drift_ppm_med,
            "quality_flag": qflag,
            "notes": notes
        })
    df_seg = pd.DataFrame(rows_seg)
    df_abs.to_csv(out_dir / OUT_SD_ABSOLUTE, index=False)
    df_seg.to_csv(out_dir / OUT_SEGMENT_REPORT, index=False)
    return df_abs, df_seg

def build_anchor_report(anchors_by_segment: Dict[int, List[Anchor]],
                        fits_by_segment: Dict[int, List[WindowFit]],
                        out_dir: Path) -> pd.DataFrame:
    rows = []
    for seg_id, anchors in anchors_by_segment.items():
        fits = fits_by_segment.get(seg_id, [])
        for a in anchors:
            wf = choose_window_for_point(a.sd_t_rel_s, fits)
            tau = None; jitter = None
            if wf is not None:
                tau_epoch = wf.a + wf.b * a.sd_t_rel_s
                tau = pd.to_datetime(tau_epoch, unit="s", utc=True)
                jitter = a.influx_ts_utc.timestamp() - tau_epoch
            rows.append({
                "segment_id": seg_id,
                "idx_sd_global": a.sd_global_idx,
                "idx_in_segment": a.sd_idx_in_segment,
                "sd_t_rel_s": round(a.sd_t_rel_s, 3),
                "influx_ts_utc": a.influx_ts_utc.isoformat(),
                "T_q": a.triple_q.T_q,
                "RH_q": a.triple_q.RH_q,
                "U_q": a.triple_q.U_q,
                "tau_abs_utc": tau.isoformat() if tau is not None else "",
                "jitter_s": jitter
            })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / OUT_ANCHOR_REPORT, index=False)
    return df

def plausibility_checks(sd_points: List[SDPoint], influx_points: List[InfluxPoint], out_dir: Path) -> pd.DataFrame:
    from collections import Counter
    sd_counts = Counter(triple_key(p.triple_q) for p in sd_points)
    in_counts = Counter(triple_key(p.triple_q) for p in influx_points)
    offenders = sum(1 for k,v in in_counts.items() if v > sd_counts.get(k,0))
    anchors_all = robust_anchor_match(sd_points, influx_points, J_MAX_SECONDS)
    coverage = len(anchors_all) / max(1, len(influx_points))
    df = pd.DataFrame([{
        "influx_count": len(influx_points),
        "sd_count": len(sd_points),
        "matched_influx_fraction": round(coverage, 6),
        "influx_values_missing_on_sd": offenders
    }])
    df.to_csv(out_dir / OUT_PLAUSIBILITY_REPORT, index=False)
    return df

# ====================
# Main
# ====================

def main():
    sd_paths, influx_paths, out_dir = resolve_paths()
    sd_points = read_sd_files(sd_paths)
    influx_points = read_influx_files(influx_paths)
    print(f"[INFO] Loaded SD points: {len(sd_points)} | Influx points: {len(influx_points)}")

    plaus = plausibility_checks(sd_points, influx_points, out_dir)

    anchors_all = robust_anchor_match(sd_points, influx_points, J_MAX_SECONDS)
    anchors_by_segment: Dict[int, List[Anchor]] = defaultdict(list)
    for a in anchors_all: anchors_by_segment[a.sd_segment_id].append(a)
    for seg in anchors_by_segment: anchors_by_segment[seg].sort(key=lambda x: x.influx_ts_utc)

    seg_bounds: Dict[int, Tuple[float,float]] = defaultdict(lambda: (float('inf'), float('-inf')))
    for p in sd_points:
        lo, hi = seg_bounds[p.segment_id]
        lo = min(lo, p.t_rel_s); hi = max(hi, p.t_rel_s)
        seg_bounds[p.segment_id] = (lo, hi)

    fit_anchors_by_segment: Dict[int, List[Anchor]] = {}
    for seg_id, anchors in anchors_by_segment.items():
        fit_anchors_by_segment[seg_id] = select_fit_anchors_for_segment(anchors)

    # Debug: per-segment anchor stats
    for seg_id, anc in anchors_by_segment.items():
        print(f"[INFO] Segment {seg_id}: anchors={len(anc)} fit_anchors={len(fit_anchors_by_segment.get(seg_id, []))}")

    fits_by_segment: Dict[int, List[WindowFit]] = {}
    for seg_id, anchors in anchors_by_segment.items():
        fit_anc = fit_anchors_by_segment.get(seg_id, [])
        bounds = seg_bounds[seg_id]
        sb = SegmentBounds(bounds[0], bounds[1])
        fits_by_segment[seg_id] = fit_windows_for_segment(seg_id, anchors, fit_anc, sb)
        fits_by_segment[seg_id] = _extend_fits_to_segment_bounds(fits_by_segment[seg_id], sb)
        print(f"[INFO] Segment {seg_id}: windows=" + str(len(fits_by_segment.get(seg_id, []))))

    df_abs, df_seg = make_outputs(sd_points, fits_by_segment, out_dir)
    df_anch = build_anchor_report(anchors_by_segment, fits_by_segment, out_dir)

    total_windows = sum(len(v) for v in fits_by_segment.values())
    print("=== dl-sd-card-date ===")
    print(f"[INFO] Anchors total: {sum(len(v) for v in anchors_by_segment.values())} | Windows: {total_windows}")
    print("[INFO] Wrote files:")
    print("  SD_absolute:", (out_dir / OUT_SD_ABSOLUTE))
    print("  Segment_report:", (out_dir / OUT_SEGMENT_REPORT))
    print("  Anchors_report:", (out_dir / OUT_ANCHOR_REPORT))
    print("  Plausibility_report:", (out_dir / OUT_PLAUSIBILITY_REPORT))

if __name__ == "__main__":
    main()
