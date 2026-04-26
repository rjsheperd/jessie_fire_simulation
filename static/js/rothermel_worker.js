// rothermel_worker.js — Web Worker for parallel Rothermel spread computation
// Receives a cell range and input data, computes rothData + directional rates for that chunk

importScripts('/static/js/fuel_models.js', '/static/js/rothermel.js');

self.onmessage = function(e) {
    var msg = e.data;
    var startCell = msg.startCell;
    var endCell = msg.endCell;
    var simCols = msg.simCols;
    var simRows = msg.simRows;
    var cellSize = msg.cellSize;

    // Reconstruct typed arrays from transferred buffers
    var fuelTypes = new Int32Array(msg.fuelTypesBuf);
    var slopeAspectElev = new Float32Array(msg.slopeAspectElevBuf);
    var windField = new Float32Array(msg.windFieldBuf);
    var fuelSavAccel = new Float32Array(msg.fuelSavAccelBuf);

    // Reconstruct bufs object from transferred buffers
    var bufs = {
        dead_1h_b: new Float32Array(msg.bufs.dead_1h_b),
        dead_10h_b: new Float32Array(msg.bufs.dead_10h_b),
        dead_100h_b: new Float32Array(msg.bufs.dead_100h_b),
        live_h_b: new Float32Array(msg.bufs.live_h_b),
        live_w_b: new Float32Array(msg.bufs.live_w_b),
        dead_sav_burnable_b: new Float32Array(msg.bufs.dead_sav_burnable_b),
        fine_dead_ext_density_b: new Float32Array(msg.bufs.fine_dead_ext_density_b),
        areas_reaction_b: new Float32Array(msg.bufs.areas_reaction_b),
        slope_wind_b: new Float32Array(msg.bufs.slope_wind_b),
        residence_flux_b: new Float32Array(msg.bufs.residence_flux_b),
        fuel_sav_accel_b: fuelSavAccel,
        dead_moistures_t: new Float32Array(msg.bufs.dead_moistures_t),
        live_moistures_t: new Float32Array(msg.bufs.live_moistures_t),
    };

    // Compute Rothermel spread data for our chunk
    var rothData = updateSpreadData(simCols, simRows, fuelTypes, slopeAspectElev, windField, bufs, startCell, endCell);

    // Compute directional rates for our chunk
    var dirRates = computeDirectionalRates(simCols, simRows, rothData, fuelTypes, fuelSavAccel, cellSize, startCell, endCell);

    // Send back the chunk results with transferable buffers for zero-copy
    self.postMessage({
        startCell: startCell,
        endCell: endCell,
        rothData: rothData.buffer,
        maxspreadrate: dirRates.maxspreadrate.buffer,
        curspreadrate: dirRates.curspreadrate.buffer,
        intensityModifier: dirRates.intensityModifier.buffer,
        accelerationConstant: dirRates.accelerationConstant.buffer,
        maxSpread: dirRates.maxSpread,
    }, [
        rothData.buffer,
        dirRates.maxspreadrate.buffer,
        dirRates.curspreadrate.buffer,
        dirRates.intensityModifier.buffer,
        dirRates.accelerationConstant.buffer,
    ]);
};
