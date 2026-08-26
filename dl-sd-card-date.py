#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
dl-sd-card-date — Absolute Datierung der SD-Karten-Rohdaten
============================================================
Weist SD-Karten-Messungen (nur relative Zeit) absolute UTC-Zeitstempel zu,
indem sie gegen LoRaWAN-Daten (mit Zeitstempel) gematcht werden.

Datenquellen für LoRaWAN-Referenz (in Prioritätsreihenfolge):
  1. Decentlab-API (automatisch, benötigt --readout-date)
  2. Influx-CSV-Dateien (offline, via --influx-dir)

Aufruf:
  # API-Modus (empfohlen):
  python dl-sd-card-date.py SD_Card.CSV --readout-date 2025-05-10

  # Offline-Modus:
  python dl-sd-card-date.py SD_Card.CSV --influx-dir Input/

  # Alte Kompatibilität (kein positionales Argument → INPUT_DIR aus Config):
  python dl-sd-card-date.py --config my.yaml

Konfig via YAML: dl-sd-card-date.yaml im selben Ordner (oder via --config)
"""

from __future__ import annotations
import sys, os, math, re, textwrap
from typing import List, Tuple, Dict, Optional
from pathlib import Path
from collections import defaultdict
from bisect import bisect_right
from datetime import datetime, timezone, timedelta

import pandas as pd
import numpy as np


# ====================
# Defaults (per YAML/CLI überschreibbar)
# ====================

# --- Pfade & Globs ---
INPUT_DIR  = "Input"
OUTPUT_DIR = "Output"
SD_GLOB    = "*_SDCard_raw_*.csv"
INFLUX_GLOB = "Sensors_Raw_*.csv"

# --- Gerätespezifisch: generische Spalten-Definition ---
# Jede Spalte: sd_index (0-basiert in CSV), sensor (Influx-Name), conversion, decimals
# Default = DL-SHT35
COLUMNS = [
    {"sd_index": 1, "sensor": "sensirion-sht35-temperature",
     "conversion": "175 * x / 65535 - 45", "decimals": 10, "label": "T_C"},
    {"sd_index": 2, "sensor": "sensirion-sht35-humidity",
     "conversion": "100 * x / 65535", "decimals": 8, "label": "RH_pct"},
    {"sd_index": 3, "sensor": "battery",
     "conversion": "x / 1000", "decimals": 3, "label": "U_V"},
]

# --- API ---
API_DOMAIN   = ""     # z.B. "meinserver.decentlab.com"
API_KEY      = ""     # Bearer-Token
DEVICE_ID    = ""     # z.B. "19057"
DATABASE     = "main"
READOUT_DATE = ""     # ISO-Datum, z.B. "2025-05-10"
TIME_MARGIN_DAYS = 1.0  # Sicherheitsreserve für API-Zeitfenster

# --- Jitter ---
J_MAX_SECONDS = 8.0

# --- Interpolation ---
BIN_HOURS = 6.0
MIN_ANCHORS_PER_BIN = 3

# --- Trimming ---
N_TRIM_ITER        = 3
MAX_TRIM_FRACTION  = 0.02
MIN_ANCHORS_FOR_FIT = 30

# --- Qualitätsschwellen ---
RMSE_GOOD_S  = 12.0
RMSE_MED_S   = 20.0
MIN_ANCHORS_GOOD = 20

# --- Output-Dateien ---
OUT_SD_ABSOLUTE         = "SD_absolute.csv"
OUT_SEGMENT_REPORT      = "Segment_report.csv"
OUT_ANCHOR_REPORT       = "Anchors_report.csv"
OUT_PLAUSIBILITY_REPORT = "Plausibility_report.csv"


# ====================
# CLI & YAML
# ====================

def _parse_cli():
    import argparse
    parser = argparse.ArgumentParser(
        description="Absolute Datierung von SD-Karten-Rohdaten gegen LoRaWAN-Referenz.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Beispiele:
              %(prog)s SD_Card.CSV --readout-date 2025-05-10
              %(prog)s SD_Card.CSV --influx-dir Input/
              %(prog)s --config my.yaml
        """),
    )
    parser.add_argument("sd_file", nargs="?", default=None,
                        help="Pfad zur SD-Karten-CSV-Datei")
    parser.add_argument("--readout-date", default=None,
                        help="Auslesedatum der SD-Karte (ISO, z.B. 2025-05-10)")
    parser.add_argument("--influx-dir", default=None,
                        help="Verzeichnis mit Influx-CSV-Dateien (Offline-Modus)")
    parser.add_argument("--multifile", default=None,
                        help="Verzeichnis mit mehreren SD-Dateien (alle SD_GLOB werden "
                             "verarbeitet; Auslesedatum je Datei aus dem Dateinamen)")
    parser.add_argument("--split-by-year", action="store_true",
                        help="SD_absolute pro Kalenderjahr (t_abs_utc) in separate "
                             "Dateien schreiben (SD_absolute_<Jahr>.csv)")
    parser.add_argument("--config", default=None,
                        help="Pfad zur YAML-Konfigurationsdatei")
    parser.add_argument("--output-dir", default=None,
                        help="Ausgabeverzeichnis")
    return parser.parse_args()


def _readout_date_from_name(path: Path) -> Optional[datetime]:
    """Extrahiert das Auslesedatum aus dem Dateinamen.

    Nimmt die letzte 8-stellige Zifferngruppe (YYYYMMDD), z.B.
    'SGS_SDCard_raw_20250816.csv' → 2025-08-16,
    'HIG_SDCard_raw_19057_20250510.CSV' → 2025-05-10.
    """
    matches = re.findall(r"\d{8}", path.stem)
    if not matches:
        return None
    try:
        return datetime.strptime(matches[-1], "%Y%m%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _simple_yaml_load(text: str) -> dict:
    """Flacher Key:Value-Parser (Fallback wenn kein PyYAML)."""
    data = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
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


def _load_yaml(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as e:
            print(f"[WARN] YAML-Syntaxfehler in {path}: {e}", file=sys.stderr)
            return {}
    except ImportError:
        # PyYAML nicht installiert → eingebauter Fallback (nur flache key: value)
        if "columns:" in text:
            print("[WARN] PyYAML nicht installiert: 'columns:'-Konfiguration aus YAML "
                  "wird ignoriert (eingebauter Parser unterstützt keine verschachtelten "
                  "Strukturen). Bitte 'pip install pyyaml' ausführen.", file=sys.stderr)
        data = _simple_yaml_load(text)
    return data if isinstance(data, dict) else {}


def _apply_config(config_path: Optional[str]):
    """Lädt Config und setzt globale Variablen."""
    if not config_path:
        script_dir = Path(os.path.abspath(os.path.dirname(__file__)))
        local_yaml = script_dir / "dl-sd-card-date.local.yaml"
        default_yaml = script_dir / "dl-sd-card-date.yaml"
        if local_yaml.exists():
            config_path = str(local_yaml)
        elif default_yaml.exists():
            config_path = str(default_yaml)
        else:
            return
    p = Path(config_path)
    if not p.exists():
        print(f"[WARN] Config nicht gefunden: {config_path}", file=sys.stderr)
        return
    data = _load_yaml(p)

    # Flache Schlüssel direkt übernehmen
    FLAT_KEYS = {
        "INPUT_DIR": str, "OUTPUT_DIR": str, "SD_GLOB": str, "INFLUX_GLOB": str,
        "J_MAX_SECONDS": float, "BIN_HOURS": float, "MIN_ANCHORS_PER_BIN": int,
        "N_TRIM_ITER": int, "MAX_TRIM_FRACTION": float, "MIN_ANCHORS_FOR_FIT": int,
        "RMSE_GOOD_S": float, "RMSE_MED_S": float, "MIN_ANCHORS_GOOD": int,
        "OUT_SD_ABSOLUTE": str, "OUT_SEGMENT_REPORT": str,
        "OUT_ANCHOR_REPORT": str, "OUT_PLAUSIBILITY_REPORT": str,
        "API_DOMAIN": str, "API_KEY": str, "DEVICE_ID": str, "DATABASE": str,
        "READOUT_DATE": str, "TIME_MARGIN_DAYS": float,
    }
    for name, cast in FLAT_KEYS.items():
        # Erlaube sowohl UPPER als auch lower-case Keys in YAML
        val = data.get(name) or data.get(name.lower())
        if val is not None:
            try:
                globals()[name] = cast(val)
            except Exception:
                pass

    # Spalten-Definition (Liste von Dicts)
    cols_raw = data.get("columns") or data.get("COLUMNS")
    if cols_raw and isinstance(cols_raw, list):
        parsed = []
        for c in cols_raw:
            if isinstance(c, dict) and "sensor" in c and "conversion" in c:
                parsed.append({
                    "sd_index": int(c.get("sd_index", len(parsed) + 1)),
                    "sensor": str(c["sensor"]),
                    "conversion": str(c["conversion"]),
                    "decimals": int(c.get("decimals", 10)),
                    "label": str(c.get("label", c["sensor"])),
                })
        if parsed:
            globals()["COLUMNS"] = parsed


# ====================
# Quantisierung (float64, generisch)
# ====================

def _round_half_up(x: np.ndarray, decimals: int) -> np.ndarray:
    """ROUND_HALF_UP: 0.5 wird aufgerundet (kaufmännisches Runden)."""
    factor = 10.0 ** decimals
    return np.floor(x * factor + 0.5) / factor


def _eval_conversion(raw: np.ndarray, formula: str) -> np.ndarray:
    """Wendet eine Konversionsformel auf Rohwerte an. Variable: x."""
    x = raw.astype(float)
    return eval(formula, {"__builtins__": {}}, {"x": x, "np": np, "math": math})


def _format_val(val: float, decimals: int) -> str:
    return f"{val:.{decimals}f}"


def _make_keys(quantized_arrays: List[np.ndarray], decimals_list: List[int]) -> np.ndarray:
    """Erzeugt String-Schlüssel für N-Tupel-Matching (generisch)."""
    n = len(quantized_arrays[0])
    keys = np.empty(n, dtype=object)
    for i in range(n):
        parts = []
        for arr, dec in zip(quantized_arrays, decimals_list):
            parts.append(f"{arr[i]:.{dec}f}")
        keys[i] = "|".join(parts)
    return keys


# ====================
# I/O: SD-Karte
# ====================

def read_sd_file(sd_path: Path) -> pd.DataFrame:
    """Liest eine SD-CSV und erkennt Segmentgrenzen (negative Zeitsprünge).

    Returns DataFrame mit Spalten:
        global_idx, segment_id, idx_in_segment, t1024, t_rel_s,
        val_q_0..N, key
    """
    df = pd.read_csv(sd_path, header=None, encoding="utf-8-sig").dropna()
    # Erste Spalte ist immer t1024
    n_cols = df.shape[1]
    col_names = ["t1024"] + [f"raw_{i}" for i in range(1, n_cols)]
    df.columns = col_names
    df = df.astype({c: "int64" for c in col_names})

    df["t_rel_s"] = df["t1024"] / 1024.0

    # Segmente: wo die relative Zeit rückwärts springt
    dt = df["t_rel_s"].diff()
    segment_starts = (dt < 0).cumsum()
    df["segment_id"] = segment_starts
    df["idx_in_segment"] = df.groupby("segment_id").cumcount()
    df["global_idx"] = np.arange(len(df))

    # Generische Quantisierung
    q_arrays = []
    dec_list = []
    for col_def in COLUMNS:
        raw_col = f"raw_{col_def['sd_index']}"
        if raw_col not in df.columns:
            raise ValueError(f"SD-Datei hat keine Spalte {raw_col} "
                             f"(sd_index={col_def['sd_index']}, nur {n_cols} Spalten)")
        phys = _eval_conversion(df[raw_col].values, col_def["conversion"])
        q = _round_half_up(phys, col_def["decimals"])
        col_name = f"val_q_{col_def['sd_index']}"
        df[col_name] = q
        q_arrays.append(q)
        dec_list.append(col_def["decimals"])

    df["key"] = _make_keys(q_arrays, dec_list)
    return df


def read_sd_files_from_dir(in_dir: Path) -> pd.DataFrame:
    """Legacy: liest alle SD-Dateien aus einem Verzeichnis."""
    sd_files = sorted(in_dir.glob(SD_GLOB))
    if not sd_files:
        sd_files = sorted(in_dir.glob(SD_GLOB.replace('.csv', '.CSV')))
    if not sd_files:
        print(f"[WARN] Keine SD-Dateien: {SD_GLOB} in {in_dir}", file=sys.stderr)
        return pd.DataFrame()
    frames = []
    seg_offset = 0
    idx_offset = 0
    for p in sd_files:
        f = read_sd_file(p)
        # Global eindeutige IDs über alle Dateien (sonst Kollision bei groupby)
        f["segment_id"] = f["segment_id"] + seg_offset
        f["global_idx"] = f["global_idx"] + idx_offset
        seg_offset = int(f["segment_id"].max()) + 1
        idx_offset = int(f["global_idx"].max()) + 1
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


# ====================
# I/O: Influx-CSV (Offline-Modus)
# ====================

def _detect_influx_columns(df: pd.DataFrame) -> Dict[str, str]:
    """Erkennt Influx-Spalten anhand der Sensor-Namen in COLUMNS-Config.

    Returns: dict {sensor_name: column_name_in_df}
    """
    cols = list(df.columns)
    mapping = {}
    for col_def in COLUMNS:
        sensor = col_def["sensor"]
        # Suche Spalte die den Sensor-Namen enthält
        match = next((c for c in cols if sensor in c.lower() or sensor in c), None)
        if match is None:
            # Fallback: allgemeinere Suche
            short = sensor.split("-")[-1]  # z.B. "temperature" aus "sensirion-sht35-temperature"
            match = next((c for c in cols if short in c.lower()), None)
        if match is None:
            raise ValueError(f"Influx-Spalte für Sensor '{sensor}' nicht gefunden in: {cols}")
        mapping[sensor] = match
    return mapping


def read_influx_files(influx_paths: List[Path]) -> pd.DataFrame:
    """Liest Influx-CSVs (Offline-Modus).

    Returns DataFrame mit Spalten:
        influx_idx, ts_utc, ts_epoch, val_q_0..N, key
    """
    frames = [pd.read_csv(p, encoding="utf-8-sig") for p in influx_paths]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)

    col_map = _detect_influx_columns(df)
    ts_col = df.columns[0]

    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.dropna(subset=[ts_col]).sort_values(by=ts_col).reset_index(drop=True)

    # Generische Quantisierung (Influx-Werte sind bereits physikalisch)
    q_arrays = []
    dec_list = []
    for col_def in COLUMNS:
        influx_col = col_map[col_def["sensor"]]
        vals = df[influx_col].astype(float).values
        q = _round_half_up(vals, col_def["decimals"])
        q_arrays.append(q)
        dec_list.append(col_def["decimals"])

    # Epoch in Sekunden — pandas 3.0 nutzt datetime64[us]
    ts_values = df[ts_col].values
    ts_int = ts_values.astype("int64")
    resolution = np.datetime_data(ts_values.dtype)[0]
    divisor = {"ns": 1e9, "us": 1e6, "ms": 1e3, "s": 1.0}.get(resolution, 1e6)

    result = pd.DataFrame({
        "influx_idx": np.arange(len(df)),
        "ts_utc": ts_values,
        "ts_epoch": ts_int / divisor,
    })
    for col_def, q in zip(COLUMNS, q_arrays):
        result[f"val_q_{col_def['sd_index']}"] = q
    result["key"] = _make_keys(q_arrays, dec_list)
    return result


def read_influx_from_dir(in_dir: Path) -> pd.DataFrame:
    """Legacy: liest alle Influx-Dateien aus einem Verzeichnis."""
    influx_files = sorted(in_dir.glob(INFLUX_GLOB))
    if not influx_files:
        influx_files = sorted(in_dir.glob(INFLUX_GLOB.replace('.csv', '.CSV')))
    if not influx_files:
        print(f"[WARN] Keine Influx-Dateien: {INFLUX_GLOB} in {in_dir}", file=sys.stderr)
    return read_influx_files(influx_files)


# ====================
# I/O: Decentlab-API
# ====================

def _estimate_segment_windows(sd_df: pd.DataFrame, readout_date: datetime
                              ) -> List[Tuple[int, datetime, datetime]]:
    """Schätzt API-Zeitfenster pro Segment rückwärts vom Auslesedatum.

    Annahme: Das letzte Segment endet (spätestens) am Auslesedatum.
    Dauer jedes Segments ergibt sich aus t_rel_s[last] - t_rel_s[first].
    Zwischen Segmenten liegt eine unbekannte Lücke — wir nehmen an, dass
    alle Segmente direkt aneinander liegen (konservativer Ansatz: ergibt
    etwas zu breite Fenster).

    Returns: [(segment_id, start_utc, end_utc), ...]
    """
    segments = sorted(sd_df["segment_id"].unique())
    if not segments:
        return []

    margin = timedelta(days=TIME_MARGIN_DAYS)

    # Sammle Segment-Dauern
    seg_durations = []  # (seg_id, duration_seconds)
    for seg_id in segments:
        seg = sd_df[sd_df["segment_id"] == seg_id]
        t_min = seg["t_rel_s"].min()
        t_max = seg["t_rel_s"].max()
        seg_durations.append((seg_id, t_max - t_min))

    # Rückwärts von readout_date
    windows = []
    cursor = readout_date  # Ende des letzten Segments
    for seg_id, dur in reversed(seg_durations):
        end = cursor + margin
        start = cursor - timedelta(seconds=dur) - margin
        windows.append((seg_id, start, end))
        cursor = cursor - timedelta(seconds=dur)

    windows.reverse()  # chronologisch sortieren
    return windows


def _influxql_time_filter(start: datetime, end: datetime) -> str:
    """Erzeugt InfluxQL time filter String."""
    s = start.strftime("%Y-%m-%dT%H:%M:%SZ")
    e = end.strftime("%Y-%m-%dT%H:%M:%SZ")
    return f"time >= '{s}' AND time <= '{e}'"


def query_api_for_segments(sd_df: pd.DataFrame, readout_date: datetime
                           ) -> pd.DataFrame:
    """Fragt die Decentlab-API pro Segment ab und vereinigt die Ergebnisse.

    Returns: DataFrame im selben Format wie read_influx_files().
    """
    try:
        import decentlab
    except ImportError:
        sys.exit("[FEHLER] decentlab.py nicht gefunden. "
                 "Bitte ins selbe Verzeichnis wie dl-sd-card-date.py legen.")

    windows = _estimate_segment_windows(sd_df, readout_date)
    if not windows:
        return pd.DataFrame()

    device_filter = f"/^{DEVICE_ID}$/"
    sensor_names = [c["sensor"] for c in COLUMNS]
    sensor_filter = "/^(" + "|".join(sensor_names) + ")$/"

    all_frames = []
    for seg_id, start, end in windows:
        tf = _influxql_time_filter(start, end)
        print(f"[API]  Segment {seg_id}: {start.isoformat()} -> {end.isoformat()}")
        try:
            api_df = decentlab.query(
                domain=API_DOMAIN,
                api_key=API_KEY,
                time_filter=tf,
                device=device_filter,
                sensor=sensor_filter,
                agg_func=None,       # Rohdaten, keine Aggregation!
                do_unstack=True,
                convert_timestamp=True,
                timezone="UTC",
                database=DATABASE,
            )
        except ValueError as e:
            print(f"[WARN] API-Abfrage für Segment {seg_id} lieferte keine Daten: {e}")
            continue

        if api_df.empty:
            print(f"[WARN] Keine API-Daten für Segment {seg_id}")
            continue

        all_frames.append(api_df)
        print(f"[API]  -> {len(api_df)} Datenpunkte empfangen")

    if not all_frames:
        return pd.DataFrame()

    # API-Ergebnis hat Spalten wie "19057.sensirion-sht35-temperature"
    combined = pd.concat(all_frames).sort_index()
    # Duplikate entfernen (überlappende Fenster)
    combined = combined[~combined.index.duplicated(keep="first")]

    return _api_df_to_standard(combined)


def _api_df_to_standard(api_df: pd.DataFrame) -> pd.DataFrame:
    """Konvertiert API-DataFrame ins Standard-Format (wie read_influx_files)."""
    # API liefert DatetimeIndex (tz-aware UTC)
    ts_values = api_df.index.values
    ts_int = ts_values.astype("int64")
    resolution = np.datetime_data(ts_values.dtype)[0]
    divisor = {"ns": 1e9, "us": 1e6, "ms": 1e3, "s": 1.0}.get(resolution, 1e6)

    result = pd.DataFrame({
        "influx_idx": np.arange(len(api_df)),
        "ts_utc": ts_values,
        "ts_epoch": ts_int / divisor,
    })

    # Generische Quantisierung
    q_arrays = []
    dec_list = []
    api_cols = list(api_df.columns)
    for col_def in COLUMNS:
        sensor = col_def["sensor"]
        # API-Spalten: "{device_id}.{sensor}"
        match = next((c for c in api_cols if sensor in c), None)
        if match is None:
            raise ValueError(f"API-Spalte für Sensor '{sensor}' nicht gefunden in: {api_cols}")
        vals = api_df[match].values.astype(float)
        q = _round_half_up(vals, col_def["decimals"])
        col_name = f"val_q_{col_def['sd_index']}"
        result[col_name] = q
        q_arrays.append(q)
        dec_list.append(col_def["decimals"])

    result["key"] = _make_keys(q_arrays, dec_list)

    # NaN-Zeilen entfernen (API kann lückenhafte Daten liefern)
    val_cols = [f"val_q_{c['sd_index']}" for c in COLUMNS]
    result = result.dropna(subset=val_cols).reset_index(drop=True)
    result["influx_idx"] = np.arange(len(result))

    return result


# ====================
# Matching (greedy, ordnungserhaltend)
# ====================

def greedy_anchor_match(sd_df: pd.DataFrame, influx_df: pd.DataFrame) -> pd.DataFrame:
    """Ordnungserhaltendes Greedy-Matching über exakte Tupel-Gleichheit.

    Returns DataFrame mit Spalten:
        sd_global_idx, segment_id, idx_in_segment, t_rel_s,
        influx_idx, ts_utc, ts_epoch, val_q_*
    """
    val_cols = [f"val_q_{c['sd_index']}" for c in COLUMNS]
    base_cols = ["sd_global_idx", "segment_id", "idx_in_segment", "t_rel_s",
                 "influx_idx", "ts_utc", "ts_epoch"] + val_cols
    if len(sd_df) == 0 or len(influx_df) == 0:
        return pd.DataFrame(columns=base_cols)

    # Index: key → sortierte Liste von Positionen
    sd_index: Dict[str, List[int]] = defaultdict(list)
    sd_keys = sd_df["key"].values
    for i, key in enumerate(sd_keys):
        sd_index[key].append(i)

    sd_global_idx = sd_df["global_idx"].values
    sd_segment_id = sd_df["segment_id"].values
    sd_idx_in_seg = sd_df["idx_in_segment"].values
    sd_t_rel_s    = sd_df["t_rel_s"].values
    sd_val_arrays = {vc: sd_df[vc].values for vc in val_cols if vc in sd_df.columns}

    influx_keys     = influx_df["key"].values
    influx_idx_arr  = influx_df["influx_idx"].values
    influx_ts_utc   = influx_df["ts_utc"].values
    influx_ts_epoch = influx_df["ts_epoch"].values

    # Greedy Matching
    anchors = []
    current_pos = -1
    for j in range(len(influx_df)):
        key = influx_keys[j]
        positions = sd_index.get(key)
        if positions is None:
            continue
        insert_idx = bisect_right(positions, current_pos)
        if insert_idx >= len(positions):
            continue
        chosen = positions[insert_idx]
        row = [
            sd_global_idx[chosen],
            sd_segment_id[chosen],
            sd_idx_in_seg[chosen],
            sd_t_rel_s[chosen],
            influx_idx_arr[j],
            influx_ts_utc[j],
            influx_ts_epoch[j],
        ]
        for vc in val_cols:
            if vc in sd_val_arrays:
                row.append(sd_val_arrays[vc][chosen])
        anchors.append(tuple(row))
        current_pos = chosen

    if not anchors:
        return pd.DataFrame(columns=base_cols)
    return pd.DataFrame(anchors, columns=base_cols)


# ====================
# Zeitmodell: gebinnte Median-Offset-Interpolation
# ====================

def _centered_polyfit(x: np.ndarray, y: np.ndarray, deg: int = 1):
    """Numerisch stabile Variante von np.polyfit: zentriert x und y."""
    x_mean, y_mean = x.mean(), y.mean()
    coeffs = np.polyfit(x - x_mean, y - y_mean, deg)
    return coeffs, x_mean, y_mean


def _eval_centered_linear(x: np.ndarray, coeffs, x_mean: float, y_mean: float) -> np.ndarray:
    b, a = coeffs
    return y_mean + a + b * (x - x_mean)


def _trim_outliers(x: np.ndarray, offset: np.ndarray, J: float) -> np.ndarray:
    """Iteratives Trimming: entfernt Anker deren Offset stark vom Trend abweicht."""
    kept = np.ones(len(x), dtype=bool)
    for iteration in range(N_TRIM_ITER):
        xk, ok = x[kept], offset[kept]
        if len(xk) < max(2, MIN_ANCHORS_FOR_FIT):
            break
        coeffs, xm, om = _centered_polyfit(xk, ok, 1)
        trend = _eval_centered_linear(xk, coeffs, xm, om)
        residuals = np.abs(ok - trend)
        k = max(1, int(len(xk) * MAX_TRIM_FRACTION))
        threshold = np.sort(residuals)[-k]
        if threshold <= J:
            break
        worst = residuals >= threshold
        kept_indices = np.where(kept)[0]
        kept[kept_indices[worst]] = False
    return kept


def fit_segment(anchors_seg: pd.DataFrame, J: float):
    """Berechnet Zeitabbildung für ein Segment mittels gebinnter Median-Interpolation.

    Returns dict mit x_grid, offset_grid, rmse, quality, ... oder None.
    """
    x = anchors_seg["t_rel_s"].values.astype(float)
    T = anchors_seg["ts_epoch"].values.astype(float)
    raw_offset = T - x

    if len(x) < MIN_ANCHORS_FOR_FIT:
        return None

    kept = _trim_outliers(x, raw_offset, J)
    x_clean = x[kept]
    offset_clean = raw_offset[kept]

    if len(x_clean) < MIN_ANCHORS_FOR_FIT:
        return None

    bin_width_s = BIN_HOURS * 3600.0
    x_min, x_max = x_clean.min(), x_clean.max()
    bin_edges = np.arange(x_min, x_max + bin_width_s, bin_width_s)

    x_grid = []
    offset_grid = []
    for i in range(len(bin_edges) - 1):
        mask = (x_clean >= bin_edges[i]) & (x_clean < bin_edges[i + 1])
        if mask.sum() >= MIN_ANCHORS_PER_BIN:
            x_grid.append(np.median(x_clean[mask]))
            offset_grid.append(np.median(offset_clean[mask]) - J / 2.0)

    if len(x_grid) < 2:
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

    # Qualitätsmetriken
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

    if len(x_grid) >= 2:
        coeffs_drift, _, _ = _centered_polyfit(x_grid, offset_grid, 1)
        drift_ppm = float(coeffs_drift[0]) * 1e6
    else:
        drift_ppm = 0.0

    return {
        "x_grid": x_grid, "offset_grid": offset_grid,
        "rmse": rmse, "jitter_med": jitter_med, "jitter_p95": jitter_p95,
        "quality": quality, "n_anchors": n_anchors, "drift_ppm": drift_ppm,
        "x_min": float(x_min), "x_max": float(x_max),
    }


def apply_time_mapping(sd_seg: pd.DataFrame, fit_result) -> Tuple[np.ndarray, np.ndarray]:
    """Wendet Zeitabbildung auf alle SD-Punkte eines Segments an."""
    x = sd_seg["t_rel_s"].values.astype(float)
    n = len(x)
    if fit_result is None:
        return np.full(n, np.nan), np.array(["no_abs_time"] * n, dtype=object)
    offsets = np.interp(x, fit_result["x_grid"], fit_result["offset_grid"])
    t_abs = x + offsets
    flags = np.array([fit_result["quality"]] * n, dtype=object)
    return t_abs, flags


# ====================
# Reports / Output
# ====================

def _write_sd_absolute_by_year(df_out: pd.DataFrame, out_dir: Path) -> List[str]:
    """Schreibt SD_absolute pro Kalenderjahr (aus t_abs_utc) in separate Dateien.

    Zeilen ohne absolute Zeit (leeres t_abs_utc) landen in
    'SD_absolute_undatiert.csv'.
    """
    stem, ext = os.path.splitext(OUT_SD_ABSOLUTE)
    years = df_out["t_abs_utc"].str.slice(0, 4)
    written = []
    for year, group in df_out.groupby(years):
        suffix = "undatiert" if year == "" else year
        name = f"{stem}_{suffix}{ext}"
        group.to_csv(out_dir / name, index=False)
        written.append(name)
    return written


def make_outputs(sd_df: pd.DataFrame, t_abs_all: np.ndarray,
                 quality_all: np.ndarray, out_dir: Path,
                 split_by_year: bool = False) -> pd.DataFrame:
    """Schreibt SD_absolute.csv (oder pro Jahr, wenn split_by_year)."""
    # Epoch → ISO-String vektorisiert (statt row-by-row pd.to_datetime-Aufruf)
    valid_mask = ~np.isnan(t_abs_all)
    t_abs_utc = np.empty(len(t_abs_all), dtype=object)
    t_abs_utc[~valid_mask] = ""
    if valid_mask.any():
        ts = pd.to_datetime(t_abs_all[valid_mask], unit="s", utc=True)
        t_abs_utc[valid_mask] = ts.strftime("%Y-%m-%dT%H:%M:%S.%f+00:00").values

    data = {
        "segment_id": sd_df["segment_id"].values,
        "idx_sd_global": sd_df["global_idx"].values,
        "idx_in_segment": sd_df["idx_in_segment"].values,
        "t_rel_s": np.round(sd_df["t_rel_s"].values, 3),
        "t_abs_utc": t_abs_utc,
    }
    # Physikwerte vektorisiert formatieren (statt list-comprehension)
    for col_def in COLUMNS:
        vc = f"val_q_{col_def['sd_index']}"
        data[col_def["label"]] = np.char.mod(
            f"%.{col_def['decimals']}f", sd_df[vc].values
        )
    data["quality_flag"] = quality_all

    df_out = pd.DataFrame(data)
    if split_by_year:
        _write_sd_absolute_by_year(df_out, out_dir)
    else:
        df_out.to_csv(out_dir / OUT_SD_ABSOLUTE, index=False)
    return df_out


def make_segment_report(sd_df: pd.DataFrame, fit_results: Dict,
                        out_dir: Path) -> pd.DataFrame:
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
                "segment_id": seg_id, "n_points": n_points,
                "n_grid_points": 0,
                "rmse_to_mid_s_median": None, "jitter_median_s_overall": None,
                "jitter_p95_s_overall": None, "drift_ppm_median": None,
                "quality_flag": "no_abs_time", "notes": "no_anchors",
            })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / OUT_SEGMENT_REPORT, index=False)
    return df


def make_anchor_report(anchors_df: pd.DataFrame, fit_results: Dict,
                       out_dir: Path) -> pd.DataFrame:
    val_cols = [f"val_q_{c['sd_index']}" for c in COLUMNS]

    if anchors_df.empty:
        df = pd.DataFrame(columns=[
            "segment_id", "idx_sd_global", "idx_in_segment", "sd_t_rel_s",
            "influx_ts_utc", "tau_abs_utc", "jitter_s",
        ] + [c["label"] for c in COLUMNS])
        df.to_csv(out_dir / OUT_ANCHOR_REPORT, index=False)
        return df

    # Pro Segment: tau und jitter vektorisiert berechnen (statt iterrows)
    seg_parts = []
    for seg_id, group in anchors_df.groupby("segment_id"):
        fit = fit_results.get(seg_id)
        t_rel = group["t_rel_s"].values.astype(np.float64)

        if fit is not None:
            tau_epochs = t_rel + np.interp(t_rel, fit["x_grid"], fit["offset_grid"])
            tau_utc_strs = pd.to_datetime(
                tau_epochs, unit="s", utc=True
            ).strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")
            jitter_s = group["ts_epoch"].values - tau_epochs
        else:
            tau_utc_strs = np.full(len(group), "", dtype=object)
            jitter_s = np.full(len(group), np.nan)

        influx_ts_strs = pd.to_datetime(
            group["ts_utc"], utc=True, errors="coerce"
        ).dt.strftime("%Y-%m-%dT%H:%M:%S.%f+00:00").fillna("")

        part = pd.DataFrame({
            "segment_id": seg_id,
            "idx_sd_global": group["sd_global_idx"].values,
            "idx_in_segment": group["idx_in_segment"].values,
            "sd_t_rel_s": np.round(t_rel, 3),
            "influx_ts_utc": influx_ts_strs.values,
            "tau_abs_utc": tau_utc_strs,
            "jitter_s": jitter_s,
        })
        for vc, col_def in zip(val_cols, COLUMNS):
            if vc in group.columns:
                part[col_def["label"]] = np.char.mod(
                    f"%.{col_def['decimals']}f", group[vc].values
                )
        seg_parts.append(part)

    df = pd.concat(seg_parts, ignore_index=True)
    df.to_csv(out_dir / OUT_ANCHOR_REPORT, index=False)
    return df


def make_plausibility_report(sd_df: pd.DataFrame, influx_df: pd.DataFrame,
                             anchors_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
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
# Pipeline pro SD-Datensatz
# ====================

def process_segments(sd_df: pd.DataFrame, influx_df: pd.DataFrame):
    """Matching + Zeitabbildung für einen SD-Datensatz gegen eine Referenz.

    Returns: (anchors_df, fit_results, t_abs_all, quality_all)
    t_abs_all/quality_all sind positionell zu sd_df ausgerichtet.
    """
    anchors_df = greedy_anchor_match(sd_df, influx_df)
    print(f"[INFO] Anker gesamt: {len(anchors_df)}")

    fit_results: Dict = {}
    t_abs_all = np.full(len(sd_df), np.nan)
    quality_all = np.array(["no_abs_time"] * len(sd_df), dtype=object)

    for seg_id in sorted(sd_df["segment_id"].unique()):
        seg_anchors = anchors_df[anchors_df["segment_id"] == seg_id]
        seg_mask = (sd_df["segment_id"] == seg_id).values

        print(f"[INFO] Segment {seg_id}: {int(seg_mask.sum())} Punkte, "
              f"{len(seg_anchors)} Anker")

        fit = fit_segment(seg_anchors, J_MAX_SECONDS)
        fit_results[seg_id] = fit

        t_abs_seg, quality_seg = apply_time_mapping(sd_df[seg_mask], fit)
        t_abs_all[seg_mask] = t_abs_seg
        quality_all[seg_mask] = quality_seg

        if fit is not None:
            print(f"         -> {fit['quality']} (RMSE={fit['rmse']:.2f}s, "
                  f"drift={fit['drift_ppm']:.1f}ppm, "
                  f"grid={len(fit['x_grid'])} Stützstellen)")
        else:
            print(f"         -> no_abs_time")

    return anchors_df, fit_results, t_abs_all, quality_all


def run_multifile(multi_dir: Path, out_dir: Path, influx_dir: Optional[str],
                  split_by_year: bool):
    """Verarbeitet alle SD-Dateien in einem Verzeichnis (je Datei eigenes
    Auslesedatum aus dem Dateinamen) und schreibt vereinte Reports."""
    sd_files = sorted(multi_dir.glob(SD_GLOB))
    if not sd_files:
        sd_files = sorted(multi_dir.glob(SD_GLOB.replace('.csv', '.CSV')))
    if not sd_files:
        sys.exit(f"[FEHLER] Keine SD-Dateien ({SD_GLOB}) in {multi_dir}")
    print(f"[INFO] Multifile-Modus: {len(sd_files)} Datei(en) in {multi_dir}")

    # Optionale gemeinsame Offline-Referenz (statt API)
    shared_influx = None
    if influx_dir:
        idir = Path(influx_dir)
        if not idir.is_absolute():
            idir = Path.cwd() / idir
        print(f"[INFO] Gemeinsame Influx-Referenz (offline): {idir}")
        shared_influx = read_influx_from_dir(idir)
    elif not (API_DOMAIN and API_KEY and DEVICE_ID):
        sys.exit("[FEHLER] Multifile-API-Modus braucht API_DOMAIN, API_KEY und "
                 "DEVICE_ID in der Config — oder --influx-dir für Offline-Referenz.")

    seg_offset = 0
    idx_offset = 0
    sd_parts: List[pd.DataFrame] = []
    influx_parts: List[pd.DataFrame] = []
    anchor_parts: List[pd.DataFrame] = []
    t_abs_chunks: List[np.ndarray] = []
    quality_chunks: List[np.ndarray] = []
    fit_results: Dict = {}

    for f in sd_files:
        print(f"\n[INFO] === Datei: {f.name} ===")
        sd_df = read_sd_file(f)
        # Global eindeutige IDs über alle Dateien
        sd_df["segment_id"] = sd_df["segment_id"] + seg_offset
        sd_df["global_idx"] = sd_df["global_idx"] + idx_offset
        seg_offset = int(sd_df["segment_id"].max()) + 1
        idx_offset = int(sd_df["global_idx"].max()) + 1

        if shared_influx is not None:
            influx_df = shared_influx
        else:
            readout = _readout_date_from_name(f)
            if readout is None and READOUT_DATE:
                readout = datetime.strptime(READOUT_DATE, "%Y-%m-%d").replace(
                    tzinfo=timezone.utc)
            if readout is None:
                print(f"[WARN] Kein Auslesedatum aus '{f.name}' ableitbar und kein "
                      f"READOUT_DATE gesetzt -> Datei ohne absolute Zeit.",
                      file=sys.stderr)
                influx_df = pd.DataFrame()
            else:
                print(f"[INFO] Auslesedatum (aus Dateiname): {readout.date()}")
                influx_df = query_api_for_segments(sd_df, readout)

        if influx_df.empty:
            anchors_df = greedy_anchor_match(sd_df, pd.DataFrame())
            t_abs = np.full(len(sd_df), np.nan)
            quality = np.array(["no_abs_time"] * len(sd_df), dtype=object)
            fits = {seg: None for seg in sd_df["segment_id"].unique()}
        else:
            anchors_df, fits, t_abs, quality = process_segments(sd_df, influx_df)
            influx_parts.append(influx_df)

        sd_parts.append(sd_df)
        anchor_parts.append(anchors_df)
        t_abs_chunks.append(t_abs)
        quality_chunks.append(quality)
        fit_results.update(fits)

    sd_all = pd.concat(sd_parts, ignore_index=True)
    anchors_all = (pd.concat(anchor_parts, ignore_index=True)
                   if any(len(a) for a in anchor_parts) else pd.DataFrame())
    influx_all = (pd.concat(influx_parts, ignore_index=True)
                  if influx_parts else pd.DataFrame())
    t_abs_all = np.concatenate(t_abs_chunks)
    quality_all = np.concatenate(quality_chunks)

    print(f"\n[INFO] Gesamt: {len(sd_all)} SD-Punkte, "
          f"{sd_all['segment_id'].nunique()} Segmente, {len(anchors_all)} Anker")
    print(f"[INFO] OUTPUT_DIR: {out_dir}")

    make_plausibility_report(sd_all, influx_all, anchors_all, out_dir)
    make_outputs(sd_all, t_abs_all, quality_all, out_dir, split_by_year=split_by_year)
    make_segment_report(sd_all, fit_results, out_dir)
    make_anchor_report(anchors_all, fit_results, out_dir)
    return out_dir


# ====================
# Main
# ====================

def main():
    args = _parse_cli()
    _apply_config(args.config)

    # CLI-Overrides haben Vorrang vor Config
    if args.output_dir:
        globals()["OUTPUT_DIR"] = args.output_dir
    if args.readout_date:
        globals()["READOUT_DATE"] = args.readout_date

    script_dir = Path(os.path.abspath(os.path.dirname(__file__)))
    out_dir = Path(OUTPUT_DIR)
    if not out_dir.is_absolute():
        out_dir = (script_dir / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Multifile-Modus: ganzes Verzeichnis SD-Dateien verarbeiten ---
    if args.multifile:
        multi_dir = Path(args.multifile)
        if not multi_dir.is_absolute():
            multi_dir = Path.cwd() / multi_dir
        if not multi_dir.is_dir():
            sys.exit(f"[FEHLER] Multifile-Pfad ist kein Verzeichnis: {multi_dir}")
        run_multifile(multi_dir, out_dir, args.influx_dir, args.split_by_year)
        print("=== dl-sd-card-date v3 ===")
        print(f"[INFO] Dateien geschrieben in: {out_dir}")
        return

    # --- SD-Karte lesen ---
    if args.sd_file:
        sd_path = Path(args.sd_file)
        if not sd_path.is_absolute():
            sd_path = Path.cwd() / sd_path
        if not sd_path.exists():
            sys.exit(f"[FEHLER] SD-Datei nicht gefunden: {sd_path}")
        print(f"[INFO] SD-Datei: {sd_path}")
        sd_df = read_sd_file(sd_path)
    else:
        # Legacy-Modus: INPUT_DIR
        in_dir = Path(INPUT_DIR)
        if not in_dir.is_absolute():
            in_dir = (script_dir / in_dir).resolve()
        print(f"[INFO] INPUT_DIR: {in_dir}")
        sd_df = read_sd_files_from_dir(in_dir)

    if sd_df.empty:
        sys.exit("[FEHLER] Keine SD-Daten geladen.")

    print(f"[INFO] SD-Punkte: {len(sd_df)}, "
          f"Segmente: {sd_df['segment_id'].nunique()}")

    # --- Referenzdaten (Influx) laden ---
    influx_df = pd.DataFrame()

    if args.influx_dir:
        # Offline-Modus: Influx-CSV-Dateien
        influx_dir = Path(args.influx_dir)
        if not influx_dir.is_absolute():
            influx_dir = Path.cwd() / influx_dir
        print(f"[INFO] Influx-Verzeichnis (offline): {influx_dir}")
        influx_df = read_influx_from_dir(influx_dir)
    elif READOUT_DATE and API_DOMAIN and API_KEY and DEVICE_ID:
        # API-Modus
        readout_dt = datetime.strptime(READOUT_DATE, "%Y-%m-%d").replace(
            tzinfo=timezone.utc)
        print(f"[INFO] API-Modus: {API_DOMAIN}, Gerät {DEVICE_ID}, "
              f"Auslesedatum {READOUT_DATE}")
        influx_df = query_api_for_segments(sd_df, readout_dt)
    else:
        # Legacy-Fallback: INPUT_DIR
        in_dir = Path(INPUT_DIR)
        if not in_dir.is_absolute():
            in_dir = (script_dir / in_dir).resolve()
        influx_df = read_influx_from_dir(in_dir)

    if influx_df.empty:
        sys.exit("[FEHLER] Keine Referenzdaten (Influx/API) geladen.")

    print(f"[INFO] Referenz-Punkte: {len(influx_df)}")
    print(f"[INFO] OUTPUT_DIR: {out_dir}")

    # --- Matching + Zeitabbildung pro Segment ---
    anchors_df, fit_results, t_abs_all, quality_all = process_segments(sd_df, influx_df)

    make_plausibility_report(sd_df, influx_df, anchors_df, out_dir)

    # --- Output ---
    make_outputs(sd_df, t_abs_all, quality_all, out_dir, split_by_year=args.split_by_year)
    make_segment_report(sd_df, fit_results, out_dir)
    make_anchor_report(anchors_df, fit_results, out_dir)

    print("=== dl-sd-card-date v3 ===")
    print("[INFO] Dateien geschrieben:")
    print(f"  SD_absolute:         {out_dir / OUT_SD_ABSOLUTE}")
    print(f"  Segment_report:      {out_dir / OUT_SEGMENT_REPORT}")
    print(f"  Anchors_report:      {out_dir / OUT_ANCHOR_REPORT}")
    print(f"  Plausibility_report: {out_dir / OUT_PLAUSIBILITY_REPORT}")


if __name__ == "__main__":
    main()
