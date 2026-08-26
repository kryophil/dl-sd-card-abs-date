# dl-sd-card-date

Absolute Datierung von SD-Karten-Rohdaten (Decentlab DL-SHT35) anhand von LoRaWAN-Referenzdaten – robust gegen Jitter, Resets und Drift.

---

## Ziel

Sensorwerte (Temperatur, rel. Feuchte, Batteriespannung) werden auf der **SD-Karte** im Rohformat mit **relativen Zeiten seit letztem Reset** gespeichert. Eine **Teilmenge** dieser Werte wird (mit zufälligem Übermittlungs-Jitter) via LoRa an eine **Influx-Datenbank** übertragen und dort mit **absoluten UTC-Zeitstempeln** abgelegt.

Dieses Script datiert **alle SD-Rohwerte** absolut, indem es:
- **exakte Tripel-Übereinstimmungen** `(T, RH, U)` zwischen SD und Influx findet (nach definierter Quantisierung),
- die SD-Zeitachse pro Segment (Reset-bis-Reset) per **gebinnter Median-Offset-Interpolation** gegen die Influx-Zeitachse kalibriert,
- dabei **Jitter** und **Uhrdrift** berücksichtigt.

Ergebnis sind **UTC-Zeitstempel** für (nahezu) alle SD-Messpunkte.

---

## Datenformate

### SD-Karte: `*_SDCard_raw_*.csv` (ohne Header, Komma-getrennt)

Spalten:
`time, temp_raw, rh_raw, bat_raw`

Umrechnungen (für DL-SHT35, konfigurierbar):
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

**Wichtig:** Influx enthält **nur** die erfolgreich übertragenen Werte (Teilmenge der SD-Werte) und weist **zufälligen Übermittlungs-Jitter** auf (typisch 0–8 s).

---

## Konzept (Kurzfassung)

1. **Segmentierung:**
   SD-Rohdaten werden in **Segmente** geschnitten, sobald `t_rel_s` **negativ springt** → Reset/Unterbruch/Testwechsel.

2. **Exakte Tripel-Anker (ordnungserhaltend):**
   - Physikalische Größen werden **quantisiert** mit `ROUND_HALF_UP`:
     - Temperatur: **10** Nachkommastellen
     - rel. Feuchte: **8** Nachkommastellen
     - Spannung: **3** Nachkommastellen
   - Ein SD-Punkt und ein Influx-Punkt bilden einen **Anker**, wenn ihre quantisierten Tripel **exakt gleich** sind.
   - Greedy, **ordnungserhaltend** (via `bisect`): Für eine Influx-Messung wird immer die **nächste** passende SD-Position nach der zuletzt verankerten gewählt.

3. **Gebinnte Median-Offset-Interpolation:**
   - Die Anker werden in **6-Stunden-Bins** eingeteilt.
   - Pro Bin: **Median** des Zeitoffsets `(t_abs_epoch − t_rel_s)`, korrigiert um `J/2` (Jitter-Mittelpunkt).
   - Zwischen Bins: **lineare Interpolation** (`np.interp`), außerhalb: Extrapolation mit dem nächsten Randwert.
   - Vorher: iteratives **Ausreißer-Trimming** (bis `MAX_TRIM_FRACTION` pro Iteration, `N_TRIM_ITER` Durchläufe).

4. **Qualitätsbewertung:**
   - RMSE gegen Jitter-Mitte (`T − J/2`), Jitter-Median/95. Perzentil, Drift in ppm.
   - Flags: `good | medium | poor | no_abs_time`.

> **Warum kein Dynamic Time Warping?**
> DTW ist hier unnötig: Die SD-Reihenfolge ist **strikt monoton**, und wir haben **exakte** Wertanker. Der lineare Fit mit Jitter-Korrektur nutzt diese Struktur direkter, reproduzierbar und schneller.

---

## Vorgehen & Quellen

Die Entwicklung des Scripts erfolgte mit Hilfe von **ChatGPT** und **Claude**.

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
├─ dl-sd-card-date.yaml     # Konfiguration (optional; YAML)
├─ decentlab.py             # Decentlab-API-Client (MIT-Lizenz, Decentlab GmbH)
├─ Input/                   # Inputdateien
│  ├─ *_SDCard_raw_*.csv    # Files von der SD-Card des Gerätes
│  └─ Sensors_Raw_*.csv     # Downloads via Grafana von der Influx-Datenbank
└─ Output/                  # Resultatdateien
   ├─ SD_absolute.csv
   ├─ Segment_report.csv
   ├─ Anchors_report.csv
   └─ Plausibility_report.csv
```

### Abhängigkeiten

- Python **3.10+** (getestet mit 3.11)
- `pandas`, `numpy`
- `requests` — nur für API-Modus (via `decentlab.py`)
- **Optional:** `PyYAML` — der eingebaute YAML-Fallback-Parser unterstützt nur flache `key: value`-Strukturen; für die `columns:`-Konfiguration (Liste von Dicts) ist **PyYAML erforderlich**

Installation:
```bash
pip install pandas numpy
# für API-Modus:
pip install requests
# für columns:-Konfiguration via YAML:
pip install pyyaml
```

---

## Ausführung

### Betriebsmodi

**Wie das Skript die Datenquellen wählt:** Übergeordnet entscheidet `--multifile PATH`
— gesetzt, aktiviert es den Multifile-Modus, der die Einzeldatei-Logik pro Datei
umhüllt (siehe unten). Ohne `--multifile` bestimmen zwei unabhängige Weichen in
`main()` das Verhalten: die **SD-Quelle** (positionales Argument `sd_file`, sonst
`INPUT_DIR`-Fallback) und die **Influx-Quelle** (`--influx-dir` falls gesetzt, sonst
API falls `READOUT_DATE`/`API_DOMAIN`/`API_KEY`/`DEVICE_ID` vollständig konfiguriert
sind, sonst `INPUT_DIR`-Fallback). Die folgende Aufzählung zeigt die übliche
Kombination dieser Weichen — keine exklusive Prioritätsliste: `--influx-dir` ist
z. B. auch ohne positionales `sd_file` gültig.

**API-Modus** (empfohlen, wenn Decentlab-API verfügbar):
```bash
python dl-sd-card-date.py SD_Card.CSV --readout-date 2025-05-10
```
`decentlab.py` muss im selben Verzeichnis liegen. API-Zugangsdaten (`API_DOMAIN`, `API_KEY`, `DEVICE_ID`) werden aus der YAML-Konfiguration gelesen.

**Offline-Modus** (Influx-CSV bereits heruntergeladen):
```bash
python dl-sd-card-date.py SD_Card.CSV --influx-dir Input/
```

**Legacy-Modus** (kein positionales Argument → `INPUT_DIR` aus Config):
```bash
python dl-sd-card-date.py --config my.yaml
```

**Multifile-Modus** (mehrere SD-Dateien aus einem Verzeichnis):
```bash
# Alle *_SDCard_raw_*.csv aus dem Pfad, Referenz via API,
# Auslesedatum je Datei aus dem Dateinamen, Output pro Jahr:
python dl-sd-card-date.py --multifile Input/ --split-by-year
```
`--multifile PATH` verarbeitet **alle** Dateien, die auf `SD_GLOB`
(`*_SDCard_raw_*.csv`) passen. Jede Datei wird einzeln gegen die **Decentlab-API**
gematcht; das **Auslesedatum** wird aus der letzten `YYYYMMDD`-Gruppe im Dateinamen
abgeleitet (z. B. `SGS_SDCard_raw_20250816.csv` → 2025-08-16). `segment_id` und
`idx_sd_global` sind über alle Dateien hinweg eindeutig. Die Reports
(`Segment_/Anchors_/Plausibility_report`) werden **vereint** geschrieben.

Alternativ kann statt der API eine **gemeinsame Offline-Referenz** genutzt werden,
indem zusätzlich `--influx-dir` angegeben wird:
```bash
python dl-sd-card-date.py --multifile Input/ --influx-dir Input/ --split-by-year
```

**Jahres-Aufteilung** (`--split-by-year`, in jedem Modus nutzbar):
schreibt `SD_absolute_<Jahr>.csv` je Kalenderjahr von `t_abs_utc`
(z. B. `SD_absolute_2024.csv`); Zeilen ohne absolute Zeit landen in
`SD_absolute_undatiert.csv`. Ohne das Flag wird wie bisher ein einzelnes
`SD_absolute.csv` geschrieben.

### Modus-Übersicht: Welche Parameter sind wann relevant?

| Parameter / Quelle | API-Modus | Offline-Modus (`--influx-dir`) | Multifile-Modus (`--multifile`) | Legacy-Modus |
|---|---|---|---|---|
| `API_DOMAIN`, `API_KEY`, `DEVICE_ID` | **erforderlich** | ignoriert | **erforderlich** (ohne `--influx-dir`) | ignoriert |
| `READOUT_DATE` | **erforderlich** | ignoriert | aus Dateinamen (`YYYYMMDD`) | ignoriert |
| `TIME_MARGIN_DAYS` | aktiv | ignoriert | aktiv (API-Variante) | ignoriert |
| `--influx-dir` | — | **erforderlich** | optional (statt API) | — |
| `INPUT_DIR` | — | — | — | aktiv (SD + Influx) |
| Influx-CSV-Zeitraum | beliebig (API liefert) | muss SD-Zeitraum abdecken | beliebig / muss abdecken | muss SD-Zeitraum abdecken |

> **Hinweis Multifile / mehrere Auslesezeitpunkte:**  
> Im einfachen API-Modus gibt es nur ein `READOUT_DATE` — bei mehreren SD-Dateien mit unterschiedlichen Auslesezeitpunkten ist `--multifile` die richtige Wahl: das Datum wird automatisch aus dem Dateinamen abgeleitet. Im Offline-Modus (`--influx-dir`) entfällt das Problem ganz, da die Zeitachse aus den CSV-Zeitstempeln kommt.

### 1) Dateien ablegen

- **SD-Rohdaten:** `*_SDCard_raw_*.csv` (ohne Header) in `Input/` oder als Argument angeben.
- **Influx-Exporte:** `Sensors_Raw_*.csv` (mit Header) in `Input/` oder via `--influx-dir`.

### 2) Konfiguration (optional)

#### Lokale Konfiguration mit Credentials (nicht committen)

Das Script lädt automatisch `dl-sd-card-date.local.yaml` **vor** `dl-sd-card-date.yaml`, wenn
die Datei vorhanden ist. Diese Datei ist in `.gitignore` eingetragen und eignet sich für
API-Zugangsdaten, die nicht ins Repository gehören.

Einmalig anlegen:
```bash
cp dl-sd-card-date.yaml dl-sd-card-date.local.yaml
# Dann API_DOMAIN, API_KEY, DEVICE_ID in der .local.yaml eintragen
```

Ladereihenfolge: `--config`-Flag > `dl-sd-card-date.local.yaml` > `dl-sd-card-date.yaml`.

---

`dl-sd-card-date.yaml` (alle Werte optional; Defaults im Script):

```yaml
# Decentlab-API (für API-Modus)
# API_DOMAIN: "meinserver.decentlab.com"
# API_KEY: "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
# DEVICE_ID: "19057"
# DATABASE: "main"
# READOUT_DATE: "2025-05-10"
# TIME_MARGIN_DAYS: 1.0

# Spalten-Definition (generisch, pro Gerät anpassen)
# Erfordert PyYAML; ohne PyYAML gilt der Skript-Default (DL-SHT35)
columns:
  - sd_index: 1                              # 1-basierte Spaltennummer in SD-CSV
    sensor: "sensirion-sht35-temperature"    # Influx-/API-Sensorname
    conversion: "175 * x / 65535 - 45"      # Umrechnung Rohwert → Physikwert (Variable: x)
    decimals: 10                             # Nachkommastellen für Quantisierung
    label: "T_C"                             # Spaltenname in der Ausgabe
  - sd_index: 2
    sensor: "sensirion-sht35-humidity"
    conversion: "100 * x / 65535"
    decimals: 8
    label: "RH_pct"
  - sd_index: 3
    sensor: "battery"
    conversion: "x / 1000"
    decimals: 3
    label: "U_V"

# Pfade (relativ zum Scriptordner oder absolut)
INPUT_DIR: "Input"
OUTPUT_DIR: "Output"

# Dateimuster
SD_GLOB: "*_SDCard_raw_*.csv"
INFLUX_GLOB: "Sensors_Raw_*.csv"

# Jitter-Korridor (s) – laut Datenblatt: 0…8 s zufällige Verzögerung vor LoRa-TX
J_MAX_SECONDS: 8.0

# Interpolation: Bin-Breite für Median-Offset-Glättung
BIN_HOURS: 6.0
MIN_ANCHORS_PER_BIN: 3

# Trimming: iteratives Entfernen von Ausreißern
N_TRIM_ITER: 3
MAX_TRIM_FRACTION: 0.02
MIN_ANCHORS_FOR_FIT: 30

# Qualitätsschwellen
RMSE_GOOD_S: 12.0
RMSE_MED_S: 20.0
MIN_ANCHORS_GOOD: 20

# Output-Dateinamen (unter OUTPUT_DIR)
OUT_SD_ABSOLUTE: "SD_absolute.csv"
OUT_SEGMENT_REPORT: "Segment_report.csv"
OUT_ANCHOR_REPORT: "Anchors_report.csv"
OUT_PLAUSIBILITY_REPORT: "Plausibility_report.csv"
```

### 3) Start

```bash
# API-Modus
python dl-sd-card-date.py SD_Card.CSV --readout-date 2025-05-10

# Offline-Modus
python dl-sd-card-date.py SD_Card.CSV --influx-dir Input/

# Legacy-Modus (PowerShell/Windows)
python .\dl-sd-card-date.py --config .\dl-sd-card-date.yaml

# Ausgabeverzeichnis überschreiben
python dl-sd-card-date.py SD_Card.CSV --influx-dir Input/ --output-dir /tmp/results
```

Beim Start werden INPUT-/OUTPUT-Pfade geloggt, am Ende die vollständigen Output-Pfade ausgegeben.

---

## Output-Dateien

### `Output/SD_absolute.csv` (bzw. `SD_absolute_<Jahr>.csv` mit `--split-by-year`)
Spalten:
- `segment_id`, `idx_sd_global`, `idx_in_segment`
- `t_rel_s` – Sekunden seit letztem Reset (aus SD)
- `t_abs_utc` – **berechneter UTC-Zeitstempel** (ISO 8601, leer wenn kein Fit)
- `T_C`, `RH_pct`, `U_V` – quantisierte Physikwerte (10/8/3 Nachkommastellen)
- `quality_flag` – `good | medium | poor | no_abs_time`

### `Output/Segment_report.csv`
Pro Segment:
- `n_points`, `n_grid_points` (Anzahl Bin-Stützstellen im Fit)
- `rmse_to_mid_s_median` – RMSE gegen mittlere Jitterlage (`T − J/2`)
- `jitter_median_s_overall`, `jitter_p95_s_overall`
- `drift_ppm_median` – Drift der SD-Uhr in ppm
- `quality_flag`, `notes`

### `Output/Anchors_report.csv`
Alle Anker:
- SD-Index, Influx-Zeit, Tripel (quantisiert), berechnetes `tau_abs_utc`, Jitter pro Anker.

### `Output/Plausibility_report.csv`
- Anzahl Influx/SD-Punkte, Anteil gematchter Influx-Punkte, Anzahl Influx-Tripel ohne SD-Gegenstück.

---

## Qualität & Grenzen

- **Qualitätsflag je Punkt** kommt vom zugeordneten Fit. Richtwerte:
  - `good`: RMSE ≤ 12 s **und** ≥ 20 Anker,
  - `medium`: RMSE ≤ 20 s,
  - `poor`: sonst,
  - `no_abs_time`: kein Fit verfügbar (z. B. Segment ohne Anker).
- **Initiale Segmente** können leer bleiben, wenn keine Influx-Anker existieren. Ohne Anker wird **nicht spekuliert**.
- **Test-Mode** (viele Messungen in kurzer Zeit) braucht **keine** explizite Erkennung: das Matching ist ordnungserhaltend und der Fit nutzt die realen `t_rel_s`.

---

## Troubleshooting

- **„Anker gesamt = 0"**
  - Passt die **Quantisierung** (`decimals` in `columns:`) zu deinen Influx-Werten?
  - Stimmen **Dateimuster** (`SD_GLOB`, `INFLUX_GLOB`) und liegen die Files im richtigen Verzeichnis?
  - Im API-Modus: Sind `API_DOMAIN`, `API_KEY`, `DEVICE_ID` und `READOUT_DATE` korrekt gesetzt?

- **Lücken in `t_abs_utc`**
  - Segment hat zu wenige Anker für einen Fit (< `MIN_ANCHORS_FOR_FIT`).
  - `BIN_HOURS` verkleinern oder `MIN_ANCHORS_PER_BIN` reduzieren.

- **`columns:` aus YAML wird nicht übernommen**
  - PyYAML ist nicht installiert; der eingebaute Fallback-Parser unterstützt keine verschachtelten Strukturen.
  - Lösung: `pip install pyyaml`.

- **PyYAML fehlt / YAML wird nicht gelesen**
  - Script hat **eingebauten YAML-Fallback** für flache `key: value`-Konfiguration.
  - Die `columns:`-Konfiguration erfordert PyYAML (s. o.).

---

## Performance-Hinweise

- Große CSVs (10⁵–10⁶ Zeilen) benötigen entsprechend RAM (typisch 1–4 GB).
- `BIN_HOURS` erhöhen und `MIN_ANCHORS_PER_BIN` reduzieren, wenn zu wenige Bin-Stützstellen entstehen.

---

## Beitrag & Lizenz

- Pull Requests willkommen (Tests, Profiling, neue Reports).
- `decentlab.py`: MIT-Lizenz, Copyright 2016 Decentlab GmbH.
- Lizenz des übrigen Codes: tbd
