// synthetic_data.js — Generate synthetic simulation inputs for stress testing
// Bypasses file fetches so we can test arbitrary grid sizes and conditions

// Built-in fuel model definitions (ENGLISH units, matching default.fmd format)
// Format: [load1h, load10h, load100h, loadLiveH, loadLiveW, SAV1h, SAV10h, SAVLiveH, depth, extinctionMoist, deadHeat, liveHeat]
var BUILTIN_FUEL_MODELS = {
    1:  [0.74, 0, 0, 0, 0, 3500, 1800, 1500, 1.0, 0.12, 8000, 8000],
    2:  [2.0, 1.0, 0.5, 0, 0.5, 3000, 1800, 1500, 1.0, 0.15, 8000, 8000],
    5:  [1.0, 0.5, 0, 0, 2.0, 2000, 1800, 1500, 2.0, 0.20, 8000, 8000],
    6:  [1.5, 2.5, 2.0, 0, 0, 1750, 1800, 1500, 2.5, 0.25, 8000, 8000],
    8:  [1.5, 1.0, 2.5, 0, 0, 2000, 1800, 1500, 0.2, 0.30, 8000, 8000],
    10: [3.01, 2.0, 5.01, 0, 2.0, 2000, 1800, 1500, 1.0, 0.25, 8000, 8000],
};

var DEFAULT_MOISTURE = { dead1h: 0.02, dead10h: 0.02, dead100h: 0.04, liveH: 0.85, liveW: 0.85 };

function generateSyntheticFuelGrid(cols, rows, fuelModel) {
    fuelModel = fuelModel || 6;
    var size = cols * rows;
    var data = new Int32Array(size);
    data.fill(fuelModel);
    return data;
}

function generateSyntheticTerrain(cols, rows, slopeDeg, aspectDeg) {
    slopeDeg = slopeDeg || 0;
    aspectDeg = aspectDeg || 0;
    var size = cols * rows;
    // slope/aspect/elevation interleaved (matching preprocess_terrain.py output)
    var sae = new Float32Array(size * 3);
    var slopeRad = slopeDeg * Math.PI / 180.0;
    var slopeVal = Math.tan(slopeRad); // slope as rise/run
    for (var i = 0; i < size; i++) {
        sae[i * 3 + 0] = slopeVal;
        sae[i * 3 + 1] = aspectDeg;
        sae[i * 3 + 2] = 1000.0; // elevation in meters
    }
    return sae;
}

function generateSyntheticWind(cols, rows, speedMph, directionDeg) {
    speedMph = speedMph || 0;
    directionDeg = directionDeg || 0;
    var size = cols * rows;
    var windField = new Float32Array(size * 2);
    // Convert speed + direction to x,y components (matching wind grid convention)
    var dirRad = directionDeg * Math.PI / 180.0;
    var wx = speedMph * Math.sin(dirRad);
    var wy = speedMph * Math.cos(dirRad);
    for (var i = 0; i < size; i++) {
        windField[i * 2 + 0] = wx;
        windField[i * 2 + 1] = wy;
    }
    return windField;
}

// Ignition patterns:
//   'center'     - single point at grid center
//   'multi'      - N random points (params.count)
//   'edge'       - full bottom edge
//   'ring'       - perimeter ring
//   'full'       - all cells ignited
function generateSyntheticIgnition(cols, rows, pattern, params) {
    pattern = pattern || 'center';
    params = params || {};
    var NOTSETFIRE = 32767;
    var size = cols * rows;
    var data = new Int32Array(size);
    data.fill(NOTSETFIRE);

    if (pattern === 'center') {
        var cr = Math.floor(rows / 2);
        var cc = Math.floor(cols / 2);
        var radius = params.radius || 3;
        for (var r = cr - radius; r <= cr + radius; r++) {
            for (var c = cc - radius; c <= cc + radius; c++) {
                if (r >= 0 && r < rows && c >= 0 && c < cols) {
                    data[r * cols + c] = 1;
                }
            }
        }
    } else if (pattern === 'multi') {
        var count = params.count || 10;
        var seed = params.seed || 42;
        // Simple seeded PRNG (xorshift32)
        var s = seed;
        for (var i = 0; i < count; i++) {
            s ^= s << 13; s ^= s >> 17; s ^= s << 5;
            var r = Math.abs(s) % rows;
            s ^= s << 13; s ^= s >> 17; s ^= s << 5;
            var c = Math.abs(s) % cols;
            var radius = params.radius || 2;
            for (var dr = -radius; dr <= radius; dr++) {
                for (var dc = -radius; dc <= radius; dc++) {
                    var rr = r + dr, rc = c + dc;
                    if (rr >= 0 && rr < rows && rc >= 0 && rc < cols) {
                        data[rr * cols + rc] = 1;
                    }
                }
            }
        }
    } else if (pattern === 'edge') {
        var edgeRow = rows - 1;
        for (var c = 0; c < cols; c++) {
            data[edgeRow * cols + c] = 1;
        }
    } else if (pattern === 'ring') {
        for (var c = 0; c < cols; c++) {
            data[c] = 1;                        // top
            data[(rows - 1) * cols + c] = 1;    // bottom
        }
        for (var r = 0; r < rows; r++) {
            data[r * cols] = 1;                  // left
            data[r * cols + cols - 1] = 1;       // right
        }
    } else if (pattern === 'full') {
        data.fill(1);
    }

    return {
        meta: {
            left_top_lat: '0.0', left_top_long: '0.0',
            right_bottom_lat: '1.0', right_bottom_long: '1.0',
            numrows: rows.toString(), numcols: cols.toString(),
            notsetfire: NOTSETFIRE.toString(),
        },
        numrows: rows,
        numcols: cols,
        notsetfire: NOTSETFIRE,
        data: data
    };
}

function generateSyntheticCanopy(cols, rows, cbhMeters, cbdKgM3, foliarMoisture) {
    cbhMeters = cbhMeters || 3.0;       // crown base height
    cbdKgM3 = cbdKgM3 || 0.15;         // crown bulk density
    foliarMoisture = foliarMoisture || 1.0;
    var size = cols * rows;
    var I_o = new Float32Array(size);
    var RAC = new Float32Array(size);
    for (var i = 0; i < size; i++) {
        I_o[i] = Math.pow(0.01 * cbhMeters * (460.0 + 25.9 * foliarMoisture), 1.5);
        RAC[i] = cbdKgM3 > 0 ? 3.0 / cbdKgM3 : 32767.0;
    }
    return { I_o: I_o, RAC: RAC };
}

// Build fuel models and moisture objects for a given fuel model number
// Returns { models, moistures } suitable for packFuelModelBuffers()
function generateSyntheticFuelModelsAndMoisture(fuelModelNum, moisture) {
    fuelModelNum = fuelModelNum || 6;
    moisture = moisture || DEFAULT_MOISTURE;
    var loadConversion = 0.0459136823;
    var fmDef = BUILTIN_FUEL_MODELS[fuelModelNum];
    if (!fmDef) {
        console.warn('Unknown fuel model ' + fuelModelNum + ', using model 6');
        fmDef = BUILTIN_FUEL_MODELS[6];
        fuelModelNum = 6;
    }

    // Build models array (parseFuelModels format)
    // ENGLISH units: multipliers are all 1.0
    var fmdText = fuelModelNum + '\t' + fmDef.join('\t') + '\n';
    var models = parseFuelModels(fmdText);

    // Build moistures array (parseFuelMoistures format)
    var fmsText = fuelModelNum + ' ' +
        (moisture.dead1h * 100) + ' ' + (moisture.dead10h * 100) + ' ' +
        (moisture.dead100h * 100) + ' ' + (moisture.liveH * 100) + ' ' +
        (moisture.liveW * 100) + '\n';
    var moistures = parseFuelMoistures(fmsText);

    return { models: models, moistures: moistures };
}

// Master function: generate all synthetic data for a given config
// Returns an object matching what runWebGPUSimulation() produces internally
function generateAllSyntheticData(config) {
    var cols = config.cols || 906;
    var rows = config.rows || 642;
    var fuelModel = config.fuelModel || 6;
    var slopeDeg = config.slopeDeg || 0;
    var aspectDeg = config.aspectDeg || 0;
    var windSpeed = config.windSpeed || 0;
    var windDir = config.windDir || 0;
    var ignitionPattern = config.ignitionPattern || 'center';
    var ignitionParams = config.ignitionParams || {};
    var moisture = config.moisture || DEFAULT_MOISTURE;
    var foliarMoisture = config.foliarMoisture || 1.0;
    var cellSize = config.cellSize || 30.0;

    var size = cols * rows;

    console.log('Synthetic: Generating data for %dx%d grid (%d cells)...', cols, rows, size);

    var fuelTypes = generateSyntheticFuelGrid(cols, rows, fuelModel);
    var slopeAspectElev = generateSyntheticTerrain(cols, rows, slopeDeg, aspectDeg);
    var windField = generateSyntheticWind(cols, rows, windSpeed, windDir);
    var fireIgn = generateSyntheticIgnition(cols, rows, ignitionPattern, ignitionParams);
    var canopy = generateSyntheticCanopy(cols, rows, 3.0, 0.15, foliarMoisture);
    var fm = generateSyntheticFuelModelsAndMoisture(fuelModel, moisture);
    var bufs = packFuelModelBuffers(fm.models, fm.moistures);

    return {
        SIM_COLS: cols,
        SIM_ROWS: rows,
        SIM_SIZE: size,
        fuelTypes: fuelTypes,
        slopeAspectElev: slopeAspectElev,
        windField: windField,
        I_o: canopy.I_o,
        RAC: canopy.RAC,
        cellSize: cellSize,
        bufs: bufs,
        models: fm.models,
        moistures: fm.moistures,
        fireIgn: fireIgn,
        foliarMoisture: foliarMoisture,
    };
}
