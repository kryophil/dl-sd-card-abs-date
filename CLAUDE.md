# CLAUDE.md — AI Assistant Guide for dl-sd-card-abs-date

## Project Overview

**dl-sd-card-abs-date** is a Python tool for absolute dating of SD card sensor data from Decentlab DL-SHT35 temperature/humidity sensors. It synchronizes relative timestamps from SD card raw data against UTC-timestamped InfluxDB exports by matching exact sensor value triples (temperature, humidity, battery voltage) as time anchors, then applies linear time-fit modeling to assign absolute UTC timestamps to all SD measurements.

Primary documentation (README.md) is written in **German**.

## Repository Structure

```
dl-sd-card-abs-date/
├── dl-sd-card-date.py          # Main script (~840 lines, single-file application)
├── dl-sd-card-date.yaml        # YAML configuration (all tunable parameters)
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
- **PyYAML** — optional (script includes a built-in YAML fallback parser)
- Standard library: `pathlib`, `dataclasses`, `typing`, `argparse`, `decimal`, `bisect`

Install dependencies:
```bash
pip install pandas numpy
# Optional:
pip install pyyaml
```

No `requirements.txt`, `setup.py`, or `pyproject.toml` exists. This is a single-file script, not a package.

## Running the Script

```bash
# Basic execution (uses defaults from YAML and script)
python dl-sd-card-date.py

# With CLI overrides
python dl-sd-card-date.py --input-dir /path/to/data --output-dir /path/to/results
python dl-sd-card-date.py --config custom-config.yaml
```

Configuration precedence: CLI flags > YAML file > hardcoded defaults in script.

## Testing

There is **no automated test suite** (no pytest, unittest, or CI). Validation is done manually using sample data in `Input/` from real sensor deployments (Sägistalsee, Hintergräppelen).

## Linting & Formatting

No linting or formatting tools are configured. The `.gitignore` references `.ruff_cache/`, suggesting ruff may be used informally but there is no config file. When modifying code, match the existing style (see conventions below).

## Code Conventions

### Naming
- **Functions**: `snake_case`; private/internal functions prefixed with `_` (e.g., `_quantize_half_up`, `_apply_cli_overrides`)
- **Classes**: `PascalCase` dataclasses (e.g., `SDPoint`, `WindowFit`, `Anchor`, `SegmentBounds`)
- **Constants**: `ALL_CAPS` module-level globals (e.g., `TEMP_DECIMALS`, `J_MAX_SECONDS`)

### Patterns
- Heavy use of `@dataclass` for structured data (7 dataclasses)
- Full type hints throughout (`from __future__ import annotations`)
- `decimal.Decimal` with `ROUND_HALF_UP` for precise quantization
- Print-based logging; warnings go to `stderr` via `print(..., file=sys.stderr)`
- Defensive None checks; try-except around YAML/JSON parsing
- Algorithm: greedy order-preserving matching, linear least-squares fitting, iterative outlier trimming, overlapping time windows, fallback cascade

### Documentation Style
- Minimal docstrings; code is self-documenting with descriptive variable names
- German comments in configuration and some inline comments
- README.md is entirely in German

## Configuration Parameters (dl-sd-card-date.yaml)

Key parameter groups:
| Group | Examples | Purpose |
|-------|----------|---------|
| Paths | `INPUT_DIR`, `OUTPUT_DIR`, `SD_GLOB`, `INFLUX_GLOB` | File locations and patterns |
| Quantization | `TEMP_DECIMALS=10`, `RH_DECIMALS=8`, `BAT_DECIMALS=3` | Sensor value precision |
| Jitter | `J_MAX_SECONDS=8.0` | Max allowed transmission jitter |
| Matching | `MAX_SD_CANDIDATES=50`, `B_INIT_MIN/MAX` | Anchor search parameters |
| Fitting | `N_TRIM_ITER=3`, `MAX_TRIM_FRACTION=0.02` | Outlier trimming in fits |
| Windowing | `WINDOW_DAYS=21`, `WINDOW_OVERLAP_HOURS=48` | Time window sizing |
| Quality | `MIN_ANCHORS_GOOD=20`, `MIN_ANCHORS_PER_WINDOW=30` | Quality thresholds |

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

1. **Quantize** physical values from both sources to matching precision
2. **Match** exact triples (temp, humidity, battery) as anchor points
3. **Segment** SD data by device resets
4. **Fit** linear time models (offset + rate) per segment using least-squares with trimming
5. **Window** fits for robustness over long time spans
6. **Stitch** fits together for continuous coverage
7. **Output** absolute UTC timestamps with quality indicators

## Important Notes for AI Assistants

- This is a **scientific data processing tool** — correctness and numerical precision are critical
- The `decimal` module with `ROUND_HALF_UP` is used intentionally for exact quantization matching; do not replace with float rounding
- The order-preserving greedy matching algorithm is a deliberate design choice
- Do not add packaging infrastructure (setup.py, pyproject.toml) unless explicitly requested
- Preserve German comments and documentation language unless asked to translate
- The `Input/` directory contains real sensor data used for validation — do not modify or delete
- Windows path compatibility (backslash handling) is intentionally maintained
