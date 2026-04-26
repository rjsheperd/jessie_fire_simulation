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

        // Run Rothermel spread computation (parallel via Web Workers if SharedArrayBuffer available)
        var rothT0 = performance.now();
        var useWorkers = typeof SharedArrayBuffer !== 'undefined' && typeof Worker !== 'undefined';
        var dirRates;

        if (useWorkers) {
            var numWorkers = Math.min(navigator.hardwareConcurrency || 4, 8);
            var chunkSize = Math.ceil(SIM_SIZE / numWorkers);
            console.log('WebGPU: Computing Rothermel spread rates (%d workers, SharedArrayBuffer)...', numWorkers);

            // Copy input data into SharedArrayBuffers for zero-copy worker access
            var sharedFuelTypes = new SharedArrayBuffer(fuelTypes.byteLength);
            new Int32Array(sharedFuelTypes).set(fuelTypes);
            var sharedSlopeAspectElev = new SharedArrayBuffer(slopeAspectElev.byteLength);
            new Float32Array(sharedSlopeAspectElev).set(slopeAspectElev);
            var sharedWindField = new SharedArrayBuffer(windField.byteLength);
            new Float32Array(sharedWindField).set(windField);
            var sharedFuelSavAccel = new SharedArrayBuffer(bufs.fuel_sav_accel_b.byteLength);
            new Float32Array(sharedFuelSavAccel).set(bufs.fuel_sav_accel_b);

            // Small bufs are cloned once (fuel-model-indexed, ~100KB total)
            var bufsSerialized = {};
            var bufKeys = ['dead_1h_b','dead_10h_b','dead_100h_b','live_h_b','live_w_b',
                           'dead_sav_burnable_b','fine_dead_ext_density_b','areas_reaction_b',
                           'slope_wind_b','residence_flux_b','dead_moistures_t','live_moistures_t'];
            for (var bk = 0; bk < bufKeys.length; bk++) {
                var key = bufKeys[bk];
                var src = bufs[key];
                var sab = new SharedArrayBuffer(src.byteLength);
                new Float32Array(sab).set(src);
                bufsSerialized[key] = sab;
            }

            // Dispatch workers
            var workerPromises = [];
            for (var w = 0; w < numWorkers; w++) {
                var startCell = w * chunkSize;
                var endCell = Math.min(startCell + chunkSize, SIM_SIZE);
                if (startCell >= SIM_SIZE) break;

                workerPromises.push(new Promise(function(resolve, reject) {
                    var worker = new Worker('/static/js/rothermel_worker.js');
                    worker.onmessage = function(e) {
                        worker.terminate();
                        resolve(e.data);
                    };
                    worker.onerror = function(e) {
                        worker.terminate();
                        reject(e);
                    };
                    worker.postMessage({
                        startCell: startCell,
                        endCell: endCell,
                        simCols: SIM_COLS,
                        simRows: SIM_ROWS,
                        cellSize: cellSize,
                        fuelTypesBuf: sharedFuelTypes,
                        slopeAspectElevBuf: sharedSlopeAspectElev,
                        windFieldBuf: sharedWindField,
                        fuelSavAccelBuf: sharedFuelSavAccel,
                        bufs: bufsSerialized,
                    });
                }));
            }

            // Collect and merge results
            var workerResults = await Promise.all(workerPromises);

            var maxspreadrate = new Float32Array(SIM_SIZE * 16);
            var curspreadrate = new Float32Array(SIM_SIZE * 16);
            var intensityMod = new Float32Array(SIM_SIZE);
            var accelConst = new Float32Array(SIM_SIZE);
            var maxSpreadGlobal = 0.0;

            for (var r = 0; r < workerResults.length; r++) {
                var res = workerResults[r];
                var rStart = res.startCell;
                var rEnd = res.endCell;
                var chunkMaxRate = new Float32Array(res.maxspreadrate);
                var chunkCurRate = new Float32Array(res.curspreadrate);
                // SoA layout: copy each direction plane's chunk [dir*SIM_SIZE+start, dir*SIM_SIZE+end)
                for (var d = 0; d < 16; d++) {
                    var planeOff = d * SIM_SIZE;
                    maxspreadrate.set(chunkMaxRate.subarray(planeOff + rStart, planeOff + rEnd), planeOff + rStart);
                    curspreadrate.set(chunkCurRate.subarray(planeOff + rStart, planeOff + rEnd), planeOff + rStart);
                }
                var chunkIM = new Float32Array(res.intensityModifier);
                intensityMod.set(chunkIM.subarray(rStart, rEnd), rStart);
                var chunkAC = new Float32Array(res.accelerationConstant);
                accelConst.set(chunkAC.subarray(rStart, rEnd), rStart);
                if (res.maxSpread > maxSpreadGlobal) maxSpreadGlobal = res.maxSpread;
            }

            var ortho = cellSize;
            var diag = cellSize * Math.SQRT2;
            var superDist = Math.sqrt(cellSize * cellSize + (2 * cellSize) * (2 * cellSize));
            var l_n = new Float32Array([
                ortho, diag, ortho, diag, ortho, diag, ortho, diag,
                superDist, superDist, superDist, superDist,
                superDist, superDist, superDist, superDist
            ]);
            var timestep = maxSpreadGlobal > 0 ? maxSpreadGlobal / cellSize : 1.0;
            dirRates = {
                maxspreadrate: maxspreadrate, curspreadrate: curspreadrate,
                intensityModifier: intensityMod, accelerationConstant: accelConst,
                l_n: l_n, timestep: timestep, maxSpread: maxSpreadGlobal,
            };
            console.log('WebGPU: Rothermel complete in %.1f ms (%d workers)', performance.now() - rothT0, numWorkers);
        } else {
            // Fallback: sequential Rothermel (no SharedArrayBuffer)
            console.log('WebGPU: Computing Rothermel spread rates (sequential)...');
            var rothData = updateSpreadData(SIM_COLS, SIM_ROWS, fuelTypes, slopeAspectElev, windField, bufs);
            dirRates = computeDirectionalRates(SIM_COLS, SIM_ROWS, rothData, fuelTypes, bufs.fuel_sav_accel_b, cellSize);
            console.log('WebGPU: Rothermel complete in %.1f ms (sequential)', performance.now() - rothT0);
        }

        // Initialize ignition times and burn distances
        const ignTimeIn = initIgnitionTimes(SIM_SIZE, fireIgn.data, fireIgn.notsetfire);
        const burnDist = initBurnDistances(SIM_SIZE, dirRates.l_n);

        // Create and run WebGPU simulation
        console.log('WebGPU: Initializing GPU...');
        webgpuSim = new FireSimWebGPU();
        await webgpuSim.init(SIM_COLS, SIM_ROWS, {
            ignTimeIn: ignTimeIn,
            burnDist: burnDist,
            maxspreadrate: dirRates.maxspreadrate,
            curspreadrate: dirRates.curspreadrate,
            intensityModifier: dirRates.intensityModifier,
            accelerationConstant: dirRates.accelerationConstant,
            I_o: I_o,
            RAC: RAC,
            timestep: dirRates.timestep,
        });

        console.log('WebGPU: Running simulation (%d steps, timestep=%f)...', manifest.step_count, dirRates.timestep);
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

        // Recompute Rothermel
        var rothData = updateSpreadData(c.SIM_COLS, c.SIM_ROWS, fuelTypes, c.slopeAspectElev, c.windField, c.bufs);
        var dirRates = computeDirectionalRates(c.SIM_COLS, c.SIM_ROWS, rothData, fuelTypes, c.bufs.fuel_sav_accel_b, c.cellSize);

        // Rebuild ignition times
        var fireData;
        if (fireGrid2D) {
            fireData = new Int32Array(c.SIM_SIZE);
            fireData.fill(c.fireIgn.notsetfire);
            for (var row = 0; row < fireGrid2D.length; row++) {
                for (var col = 0; col < fireGrid2D[row].length; col++) {
                    if (parseInt(fireGrid2D[row][col]) === 2 || parseInt(fireGrid2D[row][col]) === 1) {
                        // Map from veg grid coords to DEM coords
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

        var ignTimeIn = initIgnitionTimes(c.SIM_SIZE, fireData, c.fireIgn.notsetfire);
        var burnDist = initBurnDistances(c.SIM_SIZE, dirRates.l_n);

        // Run simulation
        var sim = new FireSimWebGPU();
        await sim.init(c.SIM_COLS, c.SIM_ROWS, {
            ignTimeIn: ignTimeIn,
            burnDist: burnDist,
            maxspreadrate: dirRates.maxspreadrate,
            curspreadrate: dirRates.curspreadrate,
            intensityModifier: dirRates.intensityModifier,
            accelerationConstant: dirRates.accelerationConstant,
            I_o: c.I_o,
            RAC: c.RAC,
            timestep: dirRates.timestep,
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
