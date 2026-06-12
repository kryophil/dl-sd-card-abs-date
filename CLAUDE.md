# CLAUDE.md — AI Assistant Guide for dl-sd-card-abs-date

## Project Overview

**dl-sd-card-abs-date** is a Python tool for absolute dating of SD card sensor data from Decentlab DL-SHT35 temperature/humidity sensors. It synchronizes relative timestamps from SD card raw data against UTC-timestamped InfluxDB exports by matching exact sensor value triples (temperature, humidity, battery voltage) as time anchors, then applies binned median offset interpolation to assign absolute UTC timestamps to all SD measurements.

Primary documentation (README.md) is written in **German**.

## Repository Structure

```
dl-sd-card-abs-date/
├── dl-sd-card-date.py          # Main script (~990 lines, single-file application)
├── dl-sd-card-date.yaml        # YAML configuration (all tunable parameters)
├── decentlab.py                # Decentlab API client (MIT License, Decentlab GmbH 2016)
├── requirements.txt            # Dependency pins (pandas, numpy, requests, PyYAML)
├── README.md                   # Comprehensive documentation (German)
├── .gitignore                  # Standard Python gitignore
├── Input/                      # Input data directory
│   ├── *_SDCard_raw_*.csv      # SD card raw data (no header, 4 int columns)
│   └── Sensors_Raw_*.csv       # InfluxDB/Grafana exports (with header)
└── Output/                     # Generated results
    ├── SD_absolute.csv         # Main output: SD data with UTC timestamps
    ├── Segment_report.csv      # Per-segment quality metrics
    ├── Anchors_report.csv      # Anchor point analysis
    └── Plausibility_report.csv # Data consistency checks
```

## Tech Stack & Dependencies

- **Python 3.10+** (tested with 3.11)
- **pandas** — CSV I/O and data manipulation
- **numpy** — numerical operations and linear algebra
- **requests** — HTTP client for Decentlab API (API mode only, via `decentlab.py`)
- **PyYAML** — required for `columns:` YAML config (nested list-of-dicts); the built-in fallback parser only handles flat key-value pairs
- Standard library: `pathlib`, `typing`, `argparse`, `bisect`, `collections`, `datetime`, `math`, `textwrap`

Install dependencies:
```bash
# All dependencies at once (includes optional requests + PyYAML):
pip install -r requirements.txt

# Or selectively:
pip install pandas numpy
# For API mode:
pip install requests
# For columns: configuration in YAML:
pip install pyyaml
```

A `requirements.txt` (dependency pins with comments) exists. No `setup.py` or
`pyproject.toml` exists — this is a single-file script, not a package.

## Running the Script

```bash
# API mode (recommended, requires decentlab.py and credentials in YAML)
python dl-sd-card-date.py SD_Card.CSV --readout-date 2025-05-10

# Offline mode (Influx CSVs already downloaded)
python dl-sd-card-date.py SD_Card.CSV --influx-dir Input/

# Legacy mode (no positional argument → INPUT_DIR from config)
python dl-sd-card-date.py --config my.yaml

# Override output directory
python dl-sd-card-date.py SD_Card.CSV --influx-dir Input/ --output-dir /tmp/out

# Multifile mode: process every *_SDCard_raw_*.csv in a directory.
# Each file is matched against the API using a readout date parsed from its
# filename (last YYYYMMDD group); --split-by-year writes one SD_absolute_<year>.csv
# per calendar year of t_abs_utc (undated rows → SD_absolute_undatiert.csv).
python dl-sd-card-date.py --multifile Input/ --split-by-year
```

Multifile assigns globally unique `segment_id`/`idx_sd_global` across files and
writes combined reports. `--split-by-year` is orthogonal and works in any mode.
`run_multifile()` / `process_segments()` factor the per-file pipeline; the
single-file path in `main()` reuses `process_segments()`.

Configuration precedence: CLI flags > YAML file > hardcoded defaults in script.

## Testing

There is **no automated test suite** (no pytest, unittest, or CI). Validation is done manually using sample data in `Input/` from real sensor deployments (Sägistalsee, Hintergräppelen).

## Linting & Formatting

No linting or formatting tools are configured. The `.gitignore` references `.ruff_cache/`, suggesting ruff may be used informally but there is no config file. When modifying code, match the existing style (see conventions below).

## Code Conventions

### Naming
- **Functions**: `snake_case`; private/internal functions prefixed with `_` (e.g., `_round_half_up`, `_apply_config`, `_detect_influx_columns`)
- **No dataclasses**: the codebase uses plain dicts and DataFrames for structured data, not `@dataclass`
- **Constants**: `ALL_CAPS` module-level globals (e.g., `J_MAX_SECONDS`, `BIN_HOURS`, `RMSE_GOOD_S`)

### Patterns
- Full type hints throughout (`from __future__ import annotations`)
- `np.floor(x * factor + 0.5) / factor` for `ROUND_HALF_UP` quantization (not the `decimal` module)
- `eval()` with restricted builtins for sensor conversion formulas
- Print-based logging (`[INFO]`, `[WARN]`, `[FEHLER]`); warnings/errors go to `stderr` via `print(..., file=sys.stderr)`
- `globals()` mutation in `_apply_config()` to update module-level configuration from YAML/CLI
- Algorithm: greedy order-preserving matching (bisect), binned median offset interpolation, iterative outlier trimming

### Documentation Style
- Minimal docstrings; code is self-documenting with descriptive variable names
- German comments in configuration and inline comments
- README.md is entirely in German

## Configuration Parameters (dl-sd-card-date.yaml)

Key parameter groups:
| Group | Examples | Purpose |
|-------|----------|---------|
| Paths | `INPUT_DIR`, `OUTPUT_DIR`, `SD_GLOB`, `INFLUX_GLOB` | File locations and glob patterns |
| API | `API_DOMAIN`, `API_KEY`, `DEVICE_ID`, `DATABASE`, `READOUT_DATE`, `TIME_MARGIN_DAYS` | Decentlab API credentials and timing |
| Columns | `columns:` (list of dicts with `sd_index`, `sensor`, `conversion`, `decimals`, `label`) | Sensor column definitions (requires PyYAML) |
| Jitter | `J_MAX_SECONDS=8.0` | Max allowed LoRaWAN transmission jitter |
| Interpolation | `BIN_HOURS=6.0`, `MIN_ANCHORS_PER_BIN=3` | Binned median offset smoothing |
| Fitting | `N_TRIM_ITER=3`, `MAX_TRIM_FRACTION=0.02`, `MIN_ANCHORS_FOR_FIT=30` | Outlier trimming in fits |
| Quality | `RMSE_GOOD_S=12.0`, `RMSE_MED_S=20.0`, `MIN_ANCHORS_GOOD=20` | Quality flag thresholds |
| Output | `OUT_SD_ABSOLUTE`, `OUT_SEGMENT_REPORT`, `OUT_ANCHOR_REPORT`, `OUT_PLAUSIBILITY_REPORT` | Output filenames |

## Data Formats

**SD card raw CSV** (no header):
```
time_t1024, temp_raw, rh_raw, bat_raw
```
Four integer columns: relative time (in 1/1024 s ticks since device reset), raw sensor readings.

**InfluxDB export CSV** (with header):
```
Timestamp, Timezone Offset, <device>.battery, <device>.sensirion-sht35-humidity, <device>.sensirion-sht35-temperature
```

## Key Algorithm Steps

1. **Load SD data** — detect segments at negative time jumps (`t_rel_s` backwards)
2. **Load reference data** — from Decentlab API (`query_api_for_segments`) or offline Influx CSV (`read_influx_files`)
3. **Quantize** physical values from both sources to matching precision (`_round_half_up`)
4. **Match** exact triples (temp, humidity, battery) as anchor points (`greedy_anchor_match`)
5. **Fit** time model per segment using binned median offset interpolation (`fit_segment`)
6. **Apply** time mapping to all SD points (`apply_time_mapping`)
7. **Output** absolute UTC timestamps with quality indicators (`make_outputs` + reports)

## Important Notes for AI Assistants

- This is a **scientific data processing tool** — correctness and numerical precision are critical
- `ROUND_HALF_UP` quantization uses `np.floor(x * factor + 0.5) / factor` — do not replace with `round()` or `decimal.Decimal`; the exact bitwise match between SD and Influx values depends on this
- The order-preserving greedy matching algorithm is a deliberate design choice
- `_simple_yaml_load` (the built-in YAML fallback) **cannot parse nested structures** like the `columns:` list — this is intentional (PyYAML is required for that feature)
- Do not add packaging infrastructure (setup.py, pyproject.toml) unless explicitly requested
- Preserve German comments and documentation language unless asked to translate
- The `Input/` directory contains real sensor data used for validation — do not modify or delete
- Windows path compatibility (backslash handling) is intentionally maintained
- `decentlab.py` is an external MIT-licensed file from Decentlab GmbH — prefer minimal modifications
