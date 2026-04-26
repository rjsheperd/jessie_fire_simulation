// webgpu_main.js — Integration layer between WebGPU simulation and existing frontend
// If WebGPU is available, runs the fire simulation client-side.
// Otherwise, falls back to the existing server-side /api/fire_data flow.

var webgpuSimEnabled = false;
var webgpuSim = null;
// Cache parsed data for re-simulation
var _webgpuCache = null;

async function runWebGPUSimulation() {
    if (!navigator.gpu) {
        console.log('WebGPU not available, using server-side simulation');
        return null;
    }

    try {
        // Forward URL search params to manifest fetch for stress testing
        var manifestUrl = '/api/sim_data_manifest' + window.location.search;
        console.log('WebGPU: Fetching simulation data manifest...');
        const manifestResp = await fetch(manifestUrl);
        const manifest = await manifestResp.json();

        const SIM_COLS = manifest.sim_cols;
        const SIM_ROWS = manifest.sim_rows;
        const SIM_SIZE = SIM_COLS * SIM_ROWS;
        const FOLIAR_MOISTURE = manifest.foliar_moisture;

        var fuelTypes, slopeAspectElev, windField, I_o, RAC, bufs, models, moistures, fireIgn, cellSize;

        if (manifest.mode === 'synthetic' && typeof generateAllSyntheticData === 'function') {
            // Synthetic mode: generate data in-browser from URL params
            var urlParams = new URLSearchParams(window.location.search);
            var synConfig = {
                cols: SIM_COLS,
                rows: SIM_ROWS,
                fuelModel: parseInt(urlParams.get('fuel_model')) || 6,
                slopeDeg: parseFloat(urlParams.get('slope')) || 0,
                aspectDeg: parseFloat(urlParams.get('aspect')) || 0,
                windSpeed: parseFloat(urlParams.get('wind_speed')) || 0,
                windDir: parseFloat(urlParams.get('wind_dir')) || 0,
                ignitionPattern: urlParams.get('ignition') || 'center',
                ignitionParams: {
                    radius: parseInt(urlParams.get('ignition_radius')) || 3,
                    count: parseInt(urlParams.get('ignition_count')) || 10,
                    seed: parseInt(urlParams.get('ignition_seed')) || 42,
                },
                foliarMoisture: FOLIAR_MOISTURE,
                cellSize: parseFloat(urlParams.get('cell_size')) || 30.0,
            };
            // Parse custom moisture if provided
            if (urlParams.get('moisture_dead1h')) {
                synConfig.moisture = {
                    dead1h: parseFloat(urlParams.get('moisture_dead1h')),
                    dead10h: parseFloat(urlParams.get('moisture_dead10h')) || 0.02,
                    dead100h: parseFloat(urlParams.get('moisture_dead100h')) || 0.04,
                    liveH: parseFloat(urlParams.get('moisture_liveH')) || 0.85,
                    liveW: parseFloat(urlParams.get('moisture_liveW')) || 0.85,
                };
            }

            console.log('WebGPU: Generating synthetic data...', synConfig);
            var synData = generateAllSyntheticData(synConfig);
            fuelTypes = synData.fuelTypes;
            slopeAspectElev = synData.slopeAspectElev;
            windField = synData.windField;
            I_o = synData.I_o;
            RAC = synData.RAC;
            bufs = synData.bufs;
            models = synData.models;
            moistures = synData.moistures;
            fireIgn = synData.fireIgn;
            cellSize = synData.cellSize;
        } else {
            // Real mode: fetch and parse data files from server
            console.log('WebGPU: Loading simulation input files...');
            const [fuelText, fireText, terrainBin, canopyText, crownBaseText, crownBulkText,
                   windXText, windYText, fmdText, fmsText] = await Promise.all([
                fetch(manifest.fuel_file).then(r => r.text()),
                fetch(manifest.fire_file).then(r => r.text()),
                fetch(manifest.terrain_bin).then(r => r.arrayBuffer()),
                fetch(manifest.canopy_ht).then(r => r.text()),
                fetch(manifest.crown_base_ht).then(r => r.text()),
                fetch(manifest.crown_bulk_density).then(r => r.text()),
                fetch(manifest.wind_x).then(r => r.text()),
                fetch(manifest.wind_y).then(r => r.text()),
                fetch(manifest.fuel_models).then(r => r.text()),
                fetch(manifest.fuel_moisture).then(r => r.text()),
            ]);

            console.log('WebGPU: Parsing fuel models...');
            models = parseFuelModels(fmdText);
            moistures = parseFuelMoistures(fmsText);
            bufs = packFuelModelBuffers(models, moistures);

            const fuelGrid = parseAscGrid(fuelText);

            console.log('WebGPU: Upsampling fuel grid...');
            fuelTypes = new Int32Array(SIM_SIZE);
            for (let row = 0; row < SIM_ROWS; row++) {
                for (let col = 0; col < SIM_COLS; col++) {
                    const srcRow = Math.round(row * fuelGrid.nrows / SIM_ROWS);
                    const srcCol = Math.round(col * fuelGrid.ncols / SIM_COLS);
                    const clampRow = Math.min(srcRow, fuelGrid.nrows - 1);
                    const clampCol = Math.min(srcCol, fuelGrid.ncols - 1);
                    fuelTypes[row * SIM_COLS + col] = fuelGrid.data[clampRow * fuelGrid.ncols + clampCol];
                }
            }

            console.log('WebGPU: Loading terrain data...');
            slopeAspectElev = new Float32Array(terrainBin);

            const windXGrid = parseAscGrid(windXText);
            const windYGrid = parseAscGrid(windYText);

            windField = new Float32Array(SIM_SIZE * 2);
            for (let row = 0; row < SIM_ROWS; row++) {
                for (let col = 0; col < SIM_COLS; col++) {
                    const cell = row * SIM_COLS + col;
                    if (windXGrid.data.length === SIM_SIZE) {
                        windField[cell * 2 + 0] = windXGrid.data[cell];
                        windField[cell * 2 + 1] = windYGrid.data[cell];
                    } else {
                        const srcRow = Math.min(Math.round(row * windXGrid.nrows / SIM_ROWS), windXGrid.nrows - 1);
                        const srcCol = Math.min(Math.round(col * windXGrid.ncols / SIM_COLS), windXGrid.ncols - 1);
                        windField[cell * 2 + 0] = windXGrid.data[srcRow * windXGrid.ncols + srcCol];
                        windField[cell * 2 + 1] = windYGrid.data[srcRow * windYGrid.ncols + srcCol];
                    }
                }
            }

            const canopyGrid = parseAscGrid(canopyText);
            const crownBaseGrid = parseAscGrid(crownBaseText);
            const crownBulkGrid = parseAscGrid(crownBulkText);

            I_o = new Float32Array(SIM_SIZE);
            RAC = new Float32Array(SIM_SIZE);
            for (let i = 0; i < SIM_SIZE; i++) {
                let cbhVal, cbdVal;
                if (crownBaseGrid.data.length === SIM_SIZE) {
                    cbhVal = crownBaseGrid.data[i];
                    cbdVal = crownBulkGrid.data[i];
                } else {
                    const row = Math.floor(i / SIM_COLS);
                    const col = i % SIM_COLS;
                    const srcRow = Math.min(Math.round(row * crownBaseGrid.nrows / SIM_ROWS), crownBaseGrid.nrows - 1);
                    const srcCol = Math.min(Math.round(col * crownBaseGrid.ncols / SIM_COLS), crownBaseGrid.ncols - 1);
                    cbhVal = crownBaseGrid.data[srcRow * crownBaseGrid.ncols + srcCol];
                    cbdVal = crownBulkGrid.data[srcRow * crownBulkGrid.ncols + srcCol];
                }
                I_o[i] = Math.pow(0.01 * cbhVal * (460.0 + 25.9 * FOLIAR_MOISTURE), 1.5);
                RAC[i] = cbdVal > 0 ? 3.0 / cbdVal : 32767.0;
            }

            cellSize = fuelGrid.meta.cellsize || 30.0;
            fireIgn = parseFireIgnition(fireText);
        }

        // Pack fuel model data into GPU-friendly layout (13 vec4s per model)
        var rothT0 = performance.now();
        var numFuelModels = bufs.fuel_sav_accel_b.length / 2;
        // FuelModel struct: 13 vec4s = 52 floats per model
        var fuelModelData = new Float32Array(numFuelModels * 52);
        for (var fm = 0; fm < numFuelModels; fm++) {
            var base = fm * 52;
            // dead_1h (vec4)
            fuelModelData[base + 0] = bufs.dead_1h_b[fm*4+0];
            fuelModelData[base + 1] = bufs.dead_1h_b[fm*4+1];
            fuelModelData[base + 2] = bufs.dead_1h_b[fm*4+2];
            fuelModelData[base + 3] = bufs.dead_1h_b[fm*4+3];
            // dead_10h (vec4)
            fuelModelData[base + 4] = bufs.dead_10h_b[fm*4+0];
            fuelModelData[base + 5] = bufs.dead_10h_b[fm*4+1];
            fuelModelData[base + 6] = bufs.dead_10h_b[fm*4+2];
            fuelModelData[base + 7] = bufs.dead_10h_b[fm*4+3];
            // dead_100h (vec4)
            fuelModelData[base + 8] = bufs.dead_100h_b[fm*4+0];
            fuelModelData[base + 9] = bufs.dead_100h_b[fm*4+1];
            fuelModelData[base + 10] = bufs.dead_100h_b[fm*4+2];
            fuelModelData[base + 11] = bufs.dead_100h_b[fm*4+3];
            // live_h (vec4)
            fuelModelData[base + 12] = bufs.live_h_b[fm*4+0];
            fuelModelData[base + 13] = bufs.live_h_b[fm*4+1];
            fuelModelData[base + 14] = bufs.live_h_b[fm*4+2];
            fuelModelData[base + 15] = bufs.live_h_b[fm*4+3];
            // live_w (vec4)
            fuelModelData[base + 16] = bufs.live_w_b[fm*4+0];
            fuelModelData[base + 17] = bufs.live_w_b[fm*4+1];
            fuelModelData[base + 18] = bufs.live_w_b[fm*4+2];
            fuelModelData[base + 19] = bufs.live_w_b[fm*4+3];
            // dead_sav_b (vec4)
            fuelModelData[base + 20] = bufs.dead_sav_burnable_b[fm*4+0];
            fuelModelData[base + 21] = bufs.dead_sav_burnable_b[fm*4+1];
            fuelModelData[base + 22] = bufs.dead_sav_burnable_b[fm*4+2];
            fuelModelData[base + 23] = bufs.dead_sav_burnable_b[fm*4+3];
            // fded (vec4)
            fuelModelData[base + 24] = bufs.fine_dead_ext_density_b[fm*4+0];
            fuelModelData[base + 25] = bufs.fine_dead_ext_density_b[fm*4+1];
            fuelModelData[base + 26] = bufs.fine_dead_ext_density_b[fm*4+2];
            fuelModelData[base + 27] = bufs.fine_dead_ext_density_b[fm*4+3];
            // areas_react (vec4)
            fuelModelData[base + 28] = bufs.areas_reaction_b[fm*4+0];
            fuelModelData[base + 29] = bufs.areas_reaction_b[fm*4+1];
            fuelModelData[base + 30] = bufs.areas_reaction_b[fm*4+2];
            fuelModelData[base + 31] = bufs.areas_reaction_b[fm*4+3];
            // slope_wind (vec4)
            fuelModelData[base + 32] = bufs.slope_wind_b[fm*4+0];
            fuelModelData[base + 33] = bufs.slope_wind_b[fm*4+1];
            fuelModelData[base + 34] = bufs.slope_wind_b[fm*4+2];
            fuelModelData[base + 35] = bufs.slope_wind_b[fm*4+3];
            // resid_flux (vec4) — only x,y used, pad z,w
            fuelModelData[base + 36] = bufs.residence_flux_b[fm*4+0];
            fuelModelData[base + 37] = bufs.residence_flux_b[fm*4+1];
            fuelModelData[base + 38] = bufs.residence_flux_b[fm*4+2];
            fuelModelData[base + 39] = bufs.residence_flux_b[fm*4+3];
            // fuel_sav_acc (vec4) — fuelSAV, accelerationConstant, 0, 0
            fuelModelData[base + 40] = bufs.fuel_sav_accel_b[fm*2+0];
            fuelModelData[base + 41] = bufs.fuel_sav_accel_b[fm*2+1];
            fuelModelData[base + 42] = 0;
            fuelModelData[base + 43] = 0;
            // moistures (vec4) — dead1h, dead10h, dead100h, 0
            fuelModelData[base + 44] = bufs.dead_moistures_t[fm*3+0];
            fuelModelData[base + 45] = bufs.dead_moistures_t[fm*3+1];
            fuelModelData[base + 46] = bufs.dead_moistures_t[fm*3+2];
            fuelModelData[base + 47] = 0;
            // live_moist (vec4) — liveH, liveW, 0, 0
            fuelModelData[base + 48] = bufs.live_moistures_t[fm*2+0];
            fuelModelData[base + 49] = bufs.live_moistures_t[fm*2+1];
            fuelModelData[base + 50] = 0;
            fuelModelData[base + 51] = 0;
        }

        // Pack canopy I_o and RAC as interleaved pairs
        var canopyIoRac = new Float32Array(SIM_SIZE * 2);
        for (var i = 0; i < SIM_SIZE; i++) {
            canopyIoRac[i * 2 + 0] = I_o[i];
            canopyIoRac[i * 2 + 1] = RAC[i];
        }

        // Run GPU Rothermel + init all simulation buffers on GPU
        console.log('WebGPU: Initializing GPU + Rothermel...');
        webgpuSim = new FireSimWebGPU();
        await webgpuSim.initGPU(SIM_COLS, SIM_ROWS, {
            fuelModels: fuelModelData,
            fuelTypes: fuelTypes,
            slopeAspectElev: slopeAspectElev,
            windField: windField,
            fireIgnition: fireIgn.data,
            canopyIoRac: canopyIoRac,
            cellSize: cellSize,
            notsetfire: fireIgn.notsetfire,
            numFuelModels: numFuelModels,
        });
        console.log('WebGPU: GPU Rothermel + init complete in %.1f ms', performance.now() - rothT0);

        console.log('WebGPU: Running simulation (%d steps, timestep=%f)...', manifest.step_count, webgpuSim.timestep);
        const startTime = performance.now();
        await webgpuSim.run(manifest.step_count, true, function(done, total) {
            if (done % 100 === 0 || done === total) {
                console.log('WebGPU: Step %d/%d', done, total);
            }
        });
        const elapsed = performance.now() - startTime;
        console.log('WebGPU: Simulation complete in %.1f ms', elapsed);

        // Read results back
        const toaMap = await webgpuSim.readResults();

        // Convert to the same format fire_2D_map.js expects:
        // fireCurrent[row][col] as array of string arrays
        const fireData = [];
        let maxVal = 0;
        for (let row = 0; row < SIM_ROWS; row++) {
            const rowData = [];
            for (let col = 0; col < SIM_COLS; col++) {
                const val = Math.round(toaMap[row * SIM_COLS + col]);
                rowData.push(val.toString());
                if (val < fireIgn.notsetfire && val > maxVal) maxVal = val;
            }
            fireData.push(rowData);
        }

        // Expose stress test instrumentation before destroying
        window._stressTestResults = {
            timing: webgpuSim.getTimingReport(),
            memory: webgpuSim.getBufferMemoryUsage(),
            adapterLimits: webgpuSim.adapterLimits,
            deviceLostReason: webgpuSim.deviceLostReason,
            gridSize: SIM_COLS + 'x' + SIM_ROWS,
            cellCount: SIM_SIZE,
            stepCount: manifest.step_count,
            maxVal: maxVal,
            toaMap: toaMap,
        };

        webgpuSim.destroy();
        webgpuSim = null;

        // Cache parsed data for re-simulation
        _webgpuCache = {
            SIM_COLS: SIM_COLS, SIM_ROWS: SIM_ROWS, SIM_SIZE: SIM_SIZE,
            fuelTypes: fuelTypes, slopeAspectElev: slopeAspectElev,
            windField: windField, I_o: I_o, RAC: RAC, cellSize: cellSize,
            bufs: bufs, models: models, moistures: moistures,
            manifest: manifest, fireIgn: fireIgn,
            fuelModelData: fuelModelData, canopyIoRac: canopyIoRac,
            numFuelModels: numFuelModels,
        };

        webgpuSimEnabled = true;
        console.log('WebGPU: Returning results (maxVal=%d)', maxVal);

        return {
            let_top_lat: fireIgn.meta.left_top_lat || '61.8819050',
            left_top_long: fireIgn.meta.left_top_long || '40.17204368',
            right_bottom_lat: fireIgn.meta.right_bottom_lat || '62.7417377',
            right_bottom_long: fireIgn.meta.right_bottom_long || '40.11111514',
            num_rows: SIM_ROWS.toString(),
            num_cols: SIM_COLS.toString(),
            max_val: maxVal.toString(),
            notsetfire_val: fireIgn.notsetfire.toString(),
            fire_data: fireData
        };
    } catch (err) {
        console.error('WebGPU simulation failed, falling back to server:', err);
        return null;
    }
}

// Re-run simulation with modified fire ignition or vegetation data
// fireGrid2D: optional 2D array of fire ignition (1=fire, other=no fire)
// vegGrid2D: optional 2D array of vegetation types
// Returns the same format as runWebGPUSimulation() or null
async function rerunWebGPUSimulation(fireGrid2D, vegGrid2D) {
    if (!webgpuSimEnabled || !_webgpuCache) return null;
    if (!navigator.gpu) return null;

    try {
        var c = _webgpuCache;
        var fuelTypes = c.fuelTypes;

        // If vegetation was modified, re-upsample and recompute Rothermel
        if (vegGrid2D) {
            console.log('WebGPU: Recomputing with updated vegetation...');
            var vegRows = vegGrid2D.length;
            var vegCols = vegGrid2D[0].length;
            fuelTypes = new Int32Array(c.SIM_SIZE);
            for (var row = 0; row < c.SIM_ROWS; row++) {
                for (var col = 0; col < c.SIM_COLS; col++) {
                    var srcRow = Math.min(Math.round(row * vegRows / c.SIM_ROWS), vegRows - 1);
                    var srcCol = Math.min(Math.round(col * vegCols / c.SIM_COLS), vegCols - 1);
                    fuelTypes[row * c.SIM_COLS + col] = parseInt(vegGrid2D[srcRow][srcCol]) || 99;
                }
            }
        }

        // Rebuild ignition data
        var fireData;
        if (fireGrid2D) {
            fireData = new Int32Array(c.SIM_SIZE);
            fireData.fill(c.fireIgn.notsetfire);
            for (var row = 0; row < fireGrid2D.length; row++) {
                for (var col = 0; col < fireGrid2D[row].length; col++) {
                    if (parseInt(fireGrid2D[row][col]) === 2 || parseInt(fireGrid2D[row][col]) === 1) {
                        var vegRows = fireGrid2D.length;
                        var vegCols = fireGrid2D[0].length;
                        for (var dr = 0; dr < c.SIM_ROWS; dr++) {
                            for (var dc = 0; dc < c.SIM_COLS; dc++) {
                                var sr = Math.min(Math.round(dr * vegRows / c.SIM_ROWS), vegRows - 1);
                                var sc = Math.min(Math.round(dc * vegCols / c.SIM_COLS), vegCols - 1);
                                if (sr === row && sc === col) {
                                    fireData[dr * c.SIM_COLS + dc] = 1;
                                }
                            }
                        }
                    }
                }
            }
        } else {
            fireData = c.fireIgn.data;
        }

        // GPU Rothermel re-run
        var sim = new FireSimWebGPU();
        await sim.initGPU(c.SIM_COLS, c.SIM_ROWS, {
            fuelModels: c.fuelModelData,
            fuelTypes: fuelTypes,
            slopeAspectElev: c.slopeAspectElev,
            windField: c.windField,
            fireIgnition: fireData,
            canopyIoRac: c.canopyIoRac,
            cellSize: c.cellSize,
            notsetfire: c.fireIgn.notsetfire,
            numFuelModels: c.numFuelModels,
        });

        console.log('WebGPU: Re-running simulation...');
        var startTime = performance.now();
        await sim.run(c.manifest.step_count, true);
        console.log('WebGPU: Re-simulation complete in %.1f ms', performance.now() - startTime);

        var toaMap = await sim.readResults();
        sim.destroy();

        // Convert to frontend format
        var result = [];
        var maxVal = 0;
        for (var row = 0; row < c.SIM_ROWS; row++) {
            var rowData = [];
            for (var col = 0; col < c.SIM_COLS; col++) {
                var val = Math.round(toaMap[row * c.SIM_COLS + col]);
                rowData.push(val.toString());
                if (val < c.fireIgn.notsetfire && val > maxVal) maxVal = val;
            }
            result.push(rowData);
        }

        return {
            let_top_lat: c.fireIgn.meta.left_top_lat || '61.8819050',
            left_top_long: c.fireIgn.meta.left_top_long || '40.17204368',
            right_bottom_lat: c.fireIgn.meta.right_bottom_lat || '62.7417377',
            right_bottom_long: c.fireIgn.meta.right_bottom_long || '40.11111514',
            num_rows: c.SIM_ROWS.toString(),
            num_cols: c.SIM_COLS.toString(),
            max_val: maxVal.toString(),
            notsetfire_val: c.fireIgn.notsetfire.toString(),
            fire_data: result
        };
    } catch (err) {
        console.error('WebGPU re-simulation failed:', err);
        return null;
    }
}
