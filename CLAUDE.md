# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Jessie Fire Simulation: a Python Flask web server with a CUDA/C++ fire simulation backend and JavaScript canvas-based visualization frontend. The C++ simulator lives in the `fire_sim_lib` git submodule.

## Build & Run

### Python server

```bash
pip install -r requirements.txt
python views.py -h <machine-ip> -p 5000 --threaded
```

Flask runs in debug mode on 0.0.0.0:5000 by default.

### C++ simulator (fire_sim_lib submodule)

```bash
git submodule update --init --recursive
cd fire_sim_lib && mkdir -p build && cd build
cmake .. && make
```

Produces `fire_sim_lib/build/simulator`. The Python server invokes this binary via subprocess:
`./simulator [veg_file] [fire_file] [output.csv] [wind_x] [wind_y]`

Requires CUDA toolkit and GDAL.

## Architecture

**Data flow:**
```
Browser (Canvas/Google Maps)
  → Flask REST API (views.py)
  → File I/O + subprocess (util.py)
  → C++ CUDA simulator (fire_sim_lib)
  → CSV output
  → JSON API response
  → Canvas animation (static/js/fire_2D_map.js)
```

### Key files

- `views.py` — All Flask routes. REST API endpoints under `/api/` plus HTML-serving routes for upload and visualization.
- `util.py` — Data parsing (CSV→JSON), file writing, resolution conversion between DEM (642x906) and vegetation (203x287) grids, and subprocess execution of the C++ simulator. Log output goes to `log.txt`/`err_log.txt`.
- `static/js/fire_2D_map.js` (~3300 lines) — Core visualization: canvas rendering, Google Maps overlay, fire propagation animation, interactive editing tools, time slider.
- `fire_sim_lib/` — CUDA-accelerated fire spread simulation. Fire models: Behave (BD), Mark-3 Timber (MT), Improved MT (IMT), Sequential (SEQ). Entry point: `src/simulation.cu`.

### Data formats

- Output CSV: 8-line metadata header (bounding coords, grid dimensions, max/notset values) followed by comma-separated grid rows.
- Fire cell values: integer ignition timestamps; -1 = before start, 0 = not burning, 1 = burning.
- Vegetation values: integer fuel/cover type codes.
- Working data directory: `static/data/` (current simulation files and `existing/` archived datasets).

### Resolution mapping

DEM terrain grid is 642x906; vegetation grid is 203x287. `util.py:fit_high_resolution_into_low()` handles conversion between these scales.

## API Endpoints (summary)

| Method | Route | Purpose |
|--------|-------|---------|
| GET | `/api/veg_data` | Vegetation grid |
| GET | `/api/fire_data` | Full fire results |
| GET | `/api/fire_data/metadata` | Bounding coords + dimensions |
| GET | `/api/fire_frame_data/<start>/<end>` | Time-sliced fire data |
| POST | `/api/update_veg_file` | Update vegetation, re-run sim |
| POST | `/api/update_fire_file` | Update fire locations, re-run sim |
| POST | `/api/update_fire_info` | Update fire info from UI |
| GET | `/fire_vis/<folder_name>` | Load existing dataset |

Wind variants append `/<wind_x>/<wind_y>` to update/upload routes.
