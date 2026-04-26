# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Jessie Fire Simulation: wildfire spread simulation using the Rothermel model. Two execution paths:
1. **WebGPU (primary)** — entire simulation runs client-side on the GPU via compute shaders
2. **CUDA fallback** — server-side C++/CUDA binary (`fire_sim_lib` submodule) invoked via Flask subprocess

Flask web server hosts the visualization UI (Google Maps + canvas) and serves input data.

## Build & Run

```bash
pip install -r requirements.txt
python views.py -h <machine-ip> -p 5000 --threaded
```

For WebGPU: use Brave or Chrome 113+ with WebGPU enabled. Navigate to `/fire_vis/unr_aa_1`.

### Stress Tests

```bash
bun install
python views.py -p 5001 &
bun run test_stress.mjs --suite correctness
bun run test_stress.mjs --suite all
```

Suites: `scale`, `concurrency`, `numerical`, `correctness`, `perf`, `error`, `all`.

## Architecture

### WebGPU path (primary)
```
Browser → Flask /api/sim_data_manifest → fetch data files
       → JS: parse & pack inputs
       → GPU: rothermel.wgsl (one-time spread rate computation)
       → GPU: compact_active.wgsl → fim_solve.wgsl (FIM wavefront solver)
       → readback Float32Array of arrival times
       → Canvas animation (fire_2D_map.js)
```

### CUDA fallback
```
Browser → Flask API → subprocess(./simulator) → CSV → JSON → Canvas
```

### Key files

- `views.py` — Flask routes, REST API, data serving. Manifest endpoint at `/api/sim_data_manifest`.
- `util.py` — Data parsing (CSV→JSON), resolution mapping (203x287 veg → 906x642 DEM), subprocess.
- `static/js/webgpu_main.js` — WebGPU orchestration: fetch data, pack GPU inputs, run pipeline, convert output.
- `static/js/webgpu_sim.js` — GPU buffer management, compute pipelines, `initGPU()`, `runFIM()`, `readResults()`.
- `static/js/rothermel.js` — CPU Rothermel implementation (reference/fallback).
- `static/js/fuel_models.js` — Fuel model parsing (Anderson 13 / Scott-Burgan 40).
- `static/js/fire_2D_map.js` — Canvas visualization, Google Maps overlay, animation, editing tools.
- `static/js/shaders/*.wgsl` — WebGPU compute shaders (see below).
- `fire_sim_lib/` — C++/CUDA simulator submodule. Only BD (Burn Distance) model is used.

### Compute shaders

| Shader | Purpose |
|--------|---------|
| `rothermel.wgsl` | Full Rothermel model on GPU — spread rates, burn distances, crown params (10 bindings) |
| `fim_solve.wgsl` | FIM Eikonal solver — convergence-based wavefront fire propagation (7 bindings) |
| `compact_active.wgsl` | Atomic-append compaction of active cells + indirect dispatch args |
| `burn_dist.wgsl` | Legacy fixed-timestep BD propagation (replaced by FIM, kept for reference) |
| `accelerate.wgsl` | ROS acceleration toward maximum (BD path only) |
| `crown_rate.wgsl` | Crown fire escalation (BD path only) |

### Buffer layout conventions

- Spread rates: SoA — `rate[dir * simSize + cell]` (not AoS)
- Per-cell scalars packed as `vec4<f32>`: `[intensity_modifier, acc_const, I_o, RAC]`
- Ignition times: `atomic<u32>` with IEEE 754 bitcast for float atomics
- Burn distances: `f16` when `shader-f16` available, `f32` fallback (shader patched at load time)

### Data formats

- `.fmd`: Fuel model definitions (13 tokens per line: model# + 12 parameters)
- `.fms`: Fuel moisture (6 tokens per line: model# + 5 moisture percentages)
- `.asc`/`.fuel`: ASC grid format (key-value header + space-separated grid)
- Fire CSV: 8-line header (bounding coords, dims, notsetfire sentinel) + comma-separated grid
- Terrain `.bin`: raw Float32Array (3 floats per cell: slope, aspect, elevation)

### Grid dimensions

Default: 906 columns × 642 rows = 581,652 cells. Cell size: 30m. Vegetation grid (203×287) upsampled to match via `util.py:fit_high_resolution_into_low()`.

## API Endpoints

| Method | Route | Purpose |
|--------|-------|---------|
| GET | `/api/sim_data_manifest` | Input file paths + grid dimensions (supports `?mode=synthetic`) |
| GET | `/api/veg_data` | Vegetation grid |
| GET | `/api/fire_data` | Full fire results (server path) |
| GET | `/api/fire_data/metadata` | Bounding coords + dimensions |
| GET | `/api/fire_frame_data/<start>/<end>` | Time-sliced fire data |
| POST | `/api/update_veg_file` | Update vegetation, re-run sim |
| POST | `/api/update_fire_file` | Update fire locations, re-run sim |
| GET | `/fire_vis/<folder_name>` | Load existing dataset |

## Documentation

- `README.org` — Project overview, model description, parameters, architecture
- `WEBGPU.org` — CUDA-to-WebGPU porting details, shader documentation, buffer layouts
- `OPTIMIZE.org` — Performance optimization history with measurements
