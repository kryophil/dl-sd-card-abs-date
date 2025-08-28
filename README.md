# dl-sd-card-date

Absolute Datierung von SD-Karten-Rohdaten (Decentlab DL-SHT35) anhand von Influx-Exports – robust gegen Jitter, Resets und Drift.

---

## Ziel

Sensorwerte (Temperatur, rel. Feuchte, Batteriespannung) werden auf der **SD-Karte** im Rohformat mit **relativen Zeiten seit letztem Reset** gespeichert. Eine **Teilmenge** dieser Werte wird (mit zufälligem Übermittlungs-Jitter) via LoRa an eine **Influx-Datenbank** übertragen und dort mit **absoluten UTC-Zeitstempeln** abgelegt.

Dieses Script datiert **alle SD-Rohwerte** absolut, indem es:
- **exakte Tripel-Übereinstimmungen** `(T, RH, U)` zwischen SD und Influx findet (nach definierter Stückelung),
- die SD-Zeitachse pro Segment (Reset-bis-Reset) per **linearem Fit** gegen die Influx-Zeitachse kalibriert,
- dabei **Jitter** und **Uhrdrift** berücksichtigt,
- und Lücken über **Fenster** (Windows) und **Fallbacks** robust schließt.

Ergebnis sind **UTC-Zeitstempel** für (nahezu) alle SD-Messpunkte.

---

## Datenformate

### SD-Karte: `*_SDCard_raw_*.csv` (ohne Header, Komma-getrennt)

Spalten:  
`time, temp_raw, rh_raw, bat_raw`

Umrechnungen:
```
t_rel_s   = time / 1024                             # Sekunden seit letztem Reset
T (°C)    = temp_raw * 175 / 65535 - 45
RH (%)    = rh_raw  * 100 / 65535
U (Volt)  = bat_raw / 1000
```

### Influx-Export: `Sensors_Raw_*.csv` (mit Header, Komma-getrennt)

Spalten (Bezeichner werden automatisch erkannt):
- erste Spalte: UTC-Zeitstempel (Format `YYYY-MM-DD HH:MM:SS`),
- zweite Spalte: Timezone-Offset (nicht genutzt, Daten sind UTC),
- weitere Spalten: `*battery*`, `*humid*`, `*temp*`.

**Wichtig:** Influx enthält **nur** die erfolgreich übertragenen Werte (Teilmenge der SD-Werte) und weist **zufälligen Übermittlungs-Jitter** auf.

---

## Konzept (Kurzfassung)

1. **Segmentierung:**  
   SD-Rohdaten werden in **Segmente** geschnitten, sobald `t_rel_s` **negativ springt** → Reset/Unterbruch/Testwechsel.

2. **Exakte Tripel-Anker (LCS-artig, ordnungserhaltend):**  
   - Physikalische Größen werden **quantisiert** mit `ROUND_HALF_UP`:  
     - Temperatur: **10** Nachkommastellen  
     - rel. Feuchte: **8** Nachkommastellen  
     - Spannung: **3** Nachkommastellen  
   - Ein SD-Punkt und ein Influx-Punkt bilden einen **Anker**, wenn ihre quantisierten Tripel **exakt gleich** sind.  
   - Greedy, **ordnungserhaltend**: Für eine Influx-Messung wird immer die **nächste** passende SD-Position nach der zuletzt verankerten gewählt.  
   - Dadurch sind **Duplikate/gleichzeitige Messungen** handhabbar, und die SD-Reihenfolge bleibt gewahrt.

3. **6-Stunden-Fitanker + Fenster (Windows):**  
   - Pro Segment werden aus der Ankerliste **Fit-Anker** im 6-h-Raster gewählt (Start/Ende gepinnt, Auswahl `median|first|last` konfigurierbar).  
   - Der Zeitraum wird in überlappende **Fenster** (z.B. 21 Tage, 48 h Überlapp) geteilt.  
   - Pro Fenster wird (mit Mindestanzahl Anker) ein **linearer Zeit-Fit** bestimmt.

4. **Zeit-Fit mit Jitter-Constraints + Trimming:**  
   Für Ankerpaare `(x_i = t_rel_s, T_i = t_abs_epoch)` wird
   ```
   tau_i = a + b * x_i
   ```
   bestimmt mit **Jitter-Korridor** (z. B. `J_MAX_SECONDS=8`):
   ```
   T_i − J ≤ tau_i ≤ T_i
   ```
   - Start mit least squares auf `T_i − J/2`,  
   - Projektion auf den zulässigen Intervallraum,  
   - mehrfache **Trimmung** der größten Verletzer (bis max. x % pro Iteration).  
   - Kennzahlen pro Fit: RMSE (gegen Mitte), Jitter-Median/95 %, `b`→Drift (ppm).

5. **Fallbacks & Stitching:**  
   - **Wenn zu wenig 6-h-Anker im Fenster:** Fit über **alle Anker** im Fenster (eigene Schwellwerte).  
   - **Wenn gar keine Fenster passen:** segmentweiter Fit (ggf. über **alle Anker** des Segments).  
   - **Abdeckung erweitern:** erste/letzte Fits können auf **gesamte Segmentbreite** ausgedehnt werden.  
   - **Stitching:** Jeder SD-Punkt erhält den **nächstgelegenen** Fit (konfigurierbar, auch außerhalb des Fit-Fensters), um Lücken zu schließen.

> **Warum kein Dynamic Time Warping?**  
> DTW ist hier unnötig/ungünstig: Die SD-Reihenfolge ist **strikt** (Monotonie), und wir haben **exakte** Wertanker. Der lineare Fit mit Jitter-Constraints nutzt diese Struktur direkter, reproduzierbar und schneller.

---

## Vorgehen & Quellen

Die Entwicklung des Scripts erfolgte mit Hilfe von **ChatGTP 5 Thinking**.

Für die Entwicklung wurden folgende externen Unterlagen/Daten bereitgestellt und berücksichtigt:
- **Datensheet Decentlab DL-SHT35:** <https://cdn.decentlab.com/download/datasheets/Decentlab-DL-SHT35-datasheet.pdf>  
- **SD-Card User Guide (Decentlab):** <https://cdn.decentlab.com/download/manuals/SD-card-user-guide.pdf>  
- **CSV-Rohdaten** des Geräts am **Sägistalsee** (SD-Karte)  
- **Datenbank-Exporte** (Influx) des gleichen End-Nodes  
- **Tests** zusätzlich mit analogen Daten des End-Nodes **Hintergräppelen**

---

## Umsetzung

### Repository-Struktur

```
.
├─ dl-sd-card-date.py       # Pipeline (Windows-tauglich)
├─ dl-sd-card-date.yaml     # Konfiguration (optional; YAML, flach)
├─ Input\                   # hier liegen alle CSV-Inputdateien
│  ├─ *_SDCard_raw_*.csv    # Files von der SD-Card des Gerätes
│  └─ Sensors_Raw_*.csv     # Downloads via Grafana von der Influx-Datenbank
└─ Output\                  # hier landen alle Resultatdateien
   ├─ SD_absolute.csv
   ├─ Segment_report.csv
   ├─ Anchors_report.csv
   └─ Plausibility_report.csv
```

### Abhängigkeiten

- Python **3.10+** (getestet mit 3.11)  
- `pandas`, `numpy`  
- **Optional:** `PyYAML` (YAML-Fallback ist eingebaut; nicht zwingend)

Installation:
```bash
pip install pandas numpy
# optional:
pip install pyyaml
```

---

## Ausführung

### 1) Dateien ablegen

- **SD-Rohdaten** in `Input\` → `*_SDCard_raw_*.csv` (ohne Header).  
- **Influx-Exporte** in `Input\` → `Sensors_Raw_*.csv` (mit Header).

### 2) Konfiguration (optional)

`dl-sd-card-date.yaml` (alle Werte sind optional; Defaults im Script):

```yaml
# Pfade (relativ zum Scriptordner oder absolut)
INPUT_DIR: "Input"
OUTPUT_DIR: "Output"

# Dateimuster
SD_GLOB: "*_SDCard_raw_*.csv"
INFLUX_GLOB: "Sensors_Raw_*.csv"

# Quantisierung der Physik (ROUND_HALF_UP)
TEMP_DECIMALS: 10
RH_DECIMALS: 8
BAT_DECIMALS: 3

# Jitter-Korridor (s)
J_MAX_SECONDS: 8.0

# Matching
MAX_SD_CANDIDATES: 50

# Fit / Trimming
N_TRIM_ITER: 3
MAX_TRIM_FRACTION: 0.02
MIN_ANCHORS_FOR_FIT: 3

# Fensterung
WINDOW_DAYS: 21.0
WINDOW_OVERLAP_HOURS: 48.0
MIN_ANCHORS_PER_WINDOW: 30
MIN_ANCHORS_PER_WINDOW_ALL: 10

# 6h Fit-Anker
FIT_ANCHOR_GRID_HOURS: 6.0
EDGE_ANCHOR_WINDOW_MIN: 60.0
ANCHOR_PICK: "median"        # "median" | "first" | "last"

# Fallbacks
WINDOW_FALLBACK_SEGMENT_FIT: true
MIN_ANCHORS_FOR_SEGMENT_FALLBACK: 30
MIN_ANCHORS_FOR_SEGMENT_FALLBACK_ALL: 15
MIN_FALLBACK_COVERAGE_FRAC: 0.5

# Qualität
MIN_ANCHORS_GOOD: 20

# Stitching (empfohlen für durchgehende Datierung)
STITCH_WITHIN_ONLY: false     # false = nimm "nächstgelegenen" Fit, auch außerhalb des Fitfensters
STITCH_PAD_S: 1000000000      # groß wählen, damit alle Punkte abgedeckt werden

# Abdeckung erweitern
FALLBACK_EXTEND_TO_SEGMENT_BOUNDS: true
EXTEND_FITS_TO_SEGMENT_BOUNDS: true

# Output-Dateinamen
OUT_SD_ABSOLUTE: "SD_absolute.csv"
OUT_SEGMENT_REPORT: "Segment_report.csv"
OUT_ANCHOR_REPORT: "Anchors_report.csv"
OUT_PLAUSIBILITY_REPORT: "Plausibility_report.csv"
```

### 3) Start (Windows, PowerShell)

Im Ordner des Scripts:
```powershell
# mit YAML neben dem Script
python .\dl-sd-card-date.py --config .\dl-sd-card-date.yaml

# oder Pfade explizit überschreiben:
python .\dl-sd-card-date.py `
  --input-dir "F:\Pfad\zu\Projekt\Input" `
  --output-dir "F:\Pfad\zu\Projekt\Output" `
  --config .\dl-sd-card-date.yaml
```

Beim Start werden **INPUT_DIR/OUTPUT_DIR** geloggt und am Ende die **vollständigen Output-Pfade** ausgegeben.

---

## Output-Dateien

### `Output\SD_absolute.csv`
Spalten (Auszug):
- `segment_id`, `idx_sd_global`, `idx_in_segment`
- `t_rel_s` – Sekunden seit letztem Reset (aus SD)
- `t_abs_utc` – **berechneter UTC-Zeitstempel** (ISO 8601)
- `T_C`, `RH_pct`, `U_V` – quantisierte Physik (10/8/3 Nachkommastellen)
- `quality_flag` – `good | medium | poor | no_abs_time`

### `Output\Segment_report.csv`
Pro Segment:
- `n_points`, `n_windows`
- `rmse_to_mid_s_median` – RMSE gegen mittlere Jitterlage (`T − J/2`)
- `jitter_median_s_overall`, `jitter_p95_s_overall`
- `drift_ppm_median` – `(b − 1) * 1e6`
- `quality_flag`, `notes`

### `Output\Anchors_report.csv`
Alle Anker:
- SD-Index, Influx-Zeit, Tripel (quantisiert), berechnetes `tau_abs_utc`, Jitter pro Anker.

### `Output\Plausibility_report.csv`
- Anzahl Influx/SD-Punkte, Anteil gematcht, Anzahl Influx-Tripel ohne SD-Gegenstück.

---

## Qualität & Grenzen

- **Qualitätsflag je Punkt** kommt vom zugeordneten Fit. Richtwerte:
  - `good`: RMSE ≤ 12 s **und** ausreichend Fit-Anker,
  - `medium`: RMSE ≤ 20 s,
  - `poor`: sonst,
  - `no_abs_time`: kein Fit verfügbar (z. B. Segment ohne Anker).
- **Initiale Segmente** (ganz zu Beginn) können leer bleiben, wenn keine Influx-Anker existieren. Ohne Anker wird **nicht spekuliert**.
- **Test-Mode** (viele Messungen in kurzer Zeit) braucht **keine** explizite Erkennung: das Matching ist ordnungserhaltend und der Fit nutzt die realen `t_rel_s`.

---

## Troubleshooting

- **„Anchors total = 0“ / „Windows = 0“**  
  - Passt die **Quantisierung** (10/8/3) zu deinen Influx-Werten?  
  - Stimmen **Dateimuster** und liegen die Files in `Input\`?
  - Schwellen lockern: `MIN_ANCHORS_PER_WINDOW`, `MIN_ANCHORS_PER_WINDOW_ALL`, `MIN_ANCHORS_FOR_SEGMENT_FALLBACK[_ALL]`.
  - `WINDOW_DAYS` verkleinern, `WINDOW_OVERLAP_HOURS` erhöhen, `FIT_ANCHOR_GRID_HOURS` senken.

- **Lücken in `t_abs_utc`**  
  - `STITCH_WITHIN_ONLY: false` und großes `STITCH_PAD_S` setzen → Punkte auch **außerhalb** eines Fit-Fensters dem **nächstgelegenen Fit** zuordnen.

- **YAML wird nicht gelesen / PyYAML fehlt**  
  - Script hat **eingebauten YAML-Fallback** (flache `key: value`-Konfiguration). PyYAML ist **optional**.

---

## Performance-Hinweise

- Große CSVs (10^5–10^6 Zeilen) brauchen Zeit/RAM.  
- Kleinere `WINDOW_DAYS` und niedrigere Mindestanker pro Fenster können helfen.

---

## Beitrag & Lizenz

- Pull Requests willkommen (Tests, Profiling, neue Reports).  
- Lizenz: tbd

