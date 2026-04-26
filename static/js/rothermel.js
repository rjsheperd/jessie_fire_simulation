// rothermel.js — Port of FireSim::UpdateSpreadData() and propagation.cu 16-direction decomposition
// Produces Float32Array buffers for WebGPU consumption

const INF = 32767;

const ANGLES_16 = [90, 45, 0, -45, -90, -115, 180, 115,
                   63.4349, 26.5651, -26.5651, -63.4349,
                   -116.5651, -153.4349, 153.4349, 116.5651];

function packFuelModelBuffers(models, moistures) {
    const n = models.length;
    const dead_1h_b = new Float32Array(n * 4);
    const dead_10h_b = new Float32Array(n * 4);
    const dead_100h_b = new Float32Array(n * 4);
    const live_h_b = new Float32Array(n * 4);
    const live_w_b = new Float32Array(n * 4);
    const dead_sav_burnable_b = new Float32Array(n * 4);
    const fine_dead_ext_density_b = new Float32Array(n * 4);
    const areas_reaction_b = new Float32Array(n * 4);
    const slope_wind_b = new Float32Array(n * 4);
    const residence_flux_b = new Float32Array(n * 4);
    const fuel_sav_accel_b = new Float32Array(n * 2);
    const dead_moistures_t = new Float32Array(n * 3);
    const live_moistures_t = new Float32Array(n * 2);

    for (let i = 0; i < n; i++) {
        const m = models[i];
        if (!m || !m.burnable) continue;

        dead_1h_b[i*4+0] = m.effectiveHeatingNumber[FuelClass.Dead1h];
        dead_1h_b[i*4+1] = m.load[FuelClass.Dead1h];
        dead_1h_b[i*4+2] = m.areaWeightingFactor[FuelClass.Dead1h];
        dead_1h_b[i*4+3] = m.fuelMoisture[FuelClass.Dead1h];

        dead_10h_b[i*4+0] = m.effectiveHeatingNumber[FuelClass.Dead10h];
        dead_10h_b[i*4+1] = m.load[FuelClass.Dead10h];
        dead_10h_b[i*4+2] = m.areaWeightingFactor[FuelClass.Dead10h];
        dead_10h_b[i*4+3] = m.fuelMoisture[FuelClass.Dead10h];

        dead_100h_b[i*4+0] = m.effectiveHeatingNumber[FuelClass.Dead100h];
        dead_100h_b[i*4+1] = m.load[FuelClass.Dead100h];
        dead_100h_b[i*4+2] = m.areaWeightingFactor[FuelClass.Dead100h];
        dead_100h_b[i*4+3] = m.fuelMoisture[FuelClass.Dead100h];

        live_h_b[i*4+0] = m.effectiveHeatingNumber[FuelClass.LiveH];
        live_h_b[i*4+1] = m.load[FuelClass.LiveH];
        live_h_b[i*4+2] = m.areaWeightingFactor[FuelClass.LiveH];
        live_h_b[i*4+3] = m.fuelMoisture[FuelClass.LiveH];

        live_w_b[i*4+0] = m.effectiveHeatingNumber[FuelClass.LiveW];
        live_w_b[i*4+1] = m.load[FuelClass.LiveW];
        live_w_b[i*4+2] = m.areaWeightingFactor[FuelClass.LiveW];
        live_w_b[i*4+3] = m.fuelMoisture[FuelClass.LiveW];

        dead_sav_burnable_b[i*4+0] = m.SAV[FuelClass.Dead1h];
        dead_sav_burnable_b[i*4+1] = m.SAV[FuelClass.Dead10h];
        dead_sav_burnable_b[i*4+2] = m.SAV[FuelClass.Dead100h];
        dead_sav_burnable_b[i*4+3] = 100.0;

        fine_dead_ext_density_b[i*4+0] = m.fineDeadRatio || 0;
        fine_dead_ext_density_b[i*4+1] = m.extinctionMoisture;
        fine_dead_ext_density_b[i*4+2] = m.liveExtinction || 0;
        fine_dead_ext_density_b[i*4+3] = m.fuelDensity;

        areas_reaction_b[i*4+0] = m.deadArea;
        areas_reaction_b[i*4+1] = m.liveArea;
        areas_reaction_b[i*4+2] = m.deadReactionFactor;
        areas_reaction_b[i*4+3] = m.liveReactionFactor;

        slope_wind_b[i*4+0] = m.slopeK;
        slope_wind_b[i*4+1] = m.windK;
        slope_wind_b[i*4+2] = m.windB;
        slope_wind_b[i*4+3] = m.windE;

        residence_flux_b[i*4+0] = m.residenceTime;
        residence_flux_b[i*4+1] = m.propagatingFlux;
        residence_flux_b[i*4+2] = m.SAV[FuelClass.LiveH];
        residence_flux_b[i*4+3] = m.SAV[FuelClass.LiveW];

        fuel_sav_accel_b[i*2+0] = m.fuelSAV;
        fuel_sav_accel_b[i*2+1] = m.accelerationConstant;
    }

    for (let i = 0; i < moistures.length && i < n; i++) {
        const mo = moistures[i];
        if (!mo) continue;
        dead_moistures_t[i*3+0] = mo.dead1h;
        dead_moistures_t[i*3+1] = mo.dead10h;
        dead_moistures_t[i*3+2] = mo.dead100h;
        live_moistures_t[i*2+0] = mo.liveH;
        live_moistures_t[i*2+1] = mo.liveW;
    }

    return {
        dead_1h_b, dead_10h_b, dead_100h_b, live_h_b, live_w_b,
        dead_sav_burnable_b, fine_dead_ext_density_b, areas_reaction_b,
        slope_wind_b, residence_flux_b, fuel_sav_accel_b,
        dead_moistures_t, live_moistures_t
    };
}

// Port of FireSim::UpdateSpreadData (FireSim.cpp:351-668)
// Returns per-cell roth_data: {x: maxSpreadRate, y: spreadDirection, z: eccentricity, w: intensityModifier}
// Optional startCell/endCell for chunk-parallel processing via Web Workers
function updateSpreadData(simCols, simRows, fuelTypes, slopeAspectElev, windField, bufs, startCell, endCell) {
    const simSize = simCols * simRows;
    if (startCell === undefined) startCell = 0;
    if (endCell === undefined) endCell = simSize;
    // roth_data: vec4 per cell (maxSpreadRate, spreadDirection, eccentricity, intensityModifier)
    const rothData = new Float32Array(simSize * 4);

    for (let cell = startCell; cell < endCell; cell++) {
            const fuelModel = fuelTypes[cell];

            if (fuelModel === 99) {
                rothData[cell*4+0] = 0.0;  // maxSpreadRate
                rothData[cell*4+1] = 1.0;  // spreadDirection
                rothData[cell*4+2] = 1.0;  // eccentricity
                rothData[cell*4+3] = 1.0;  // intensityModifier
                continue;
            }

            // Unpack fuel model buffers
            const dsb = { x: bufs.dead_sav_burnable_b[fuelModel*4+0],
                          y: bufs.dead_sav_burnable_b[fuelModel*4+1],
                          z: bufs.dead_sav_burnable_b[fuelModel*4+2],
                          w: bufs.dead_sav_burnable_b[fuelModel*4+3] };
            if (dsb.w < 50.0) continue;

            const d1h = { x: bufs.dead_1h_b[fuelModel*4+0], y: bufs.dead_1h_b[fuelModel*4+1],
                          z: bufs.dead_1h_b[fuelModel*4+2], w: bufs.dead_1h_b[fuelModel*4+3] };
            const d10h = { x: bufs.dead_10h_b[fuelModel*4+0], y: bufs.dead_10h_b[fuelModel*4+1],
                           z: bufs.dead_10h_b[fuelModel*4+2], w: bufs.dead_10h_b[fuelModel*4+3] };
            const d100h = { x: bufs.dead_100h_b[fuelModel*4+0], y: bufs.dead_100h_b[fuelModel*4+1],
                            z: bufs.dead_100h_b[fuelModel*4+2], w: bufs.dead_100h_b[fuelModel*4+3] };
            const lh = { x: bufs.live_h_b[fuelModel*4+0], y: bufs.live_h_b[fuelModel*4+1],
                         z: bufs.live_h_b[fuelModel*4+2], w: bufs.live_h_b[fuelModel*4+3] };
            const lw = { x: bufs.live_w_b[fuelModel*4+0], y: bufs.live_w_b[fuelModel*4+1],
                         z: bufs.live_w_b[fuelModel*4+2], w: bufs.live_w_b[fuelModel*4+3] };
            const fded = { x: bufs.fine_dead_ext_density_b[fuelModel*4+0],
                           y: bufs.fine_dead_ext_density_b[fuelModel*4+1],
                           z: bufs.fine_dead_ext_density_b[fuelModel*4+2],
                           w: bufs.fine_dead_ext_density_b[fuelModel*4+3] };
            const arf = { x: bufs.areas_reaction_b[fuelModel*4+0],
                          y: bufs.areas_reaction_b[fuelModel*4+1],
                          z: bufs.areas_reaction_b[fuelModel*4+2],
                          w: bufs.areas_reaction_b[fuelModel*4+3] };
            const swf = { x: bufs.slope_wind_b[fuelModel*4+0], y: bufs.slope_wind_b[fuelModel*4+1],
                          z: bufs.slope_wind_b[fuelModel*4+2], w: bufs.slope_wind_b[fuelModel*4+3] };
            const rfl = { x: bufs.residence_flux_b[fuelModel*4+0],
                          y: bufs.residence_flux_b[fuelModel*4+1] };
            const fuelSAV = bufs.fuel_sav_accel_b[fuelModel*2+0];

            const slope = slopeAspectElev[cell*3+0];
            const aspect = slopeAspectElev[cell*3+1];

            const windX = windField[cell*2+0];
            const windY = windField[cell*2+1];

            const deadMoist = { x: bufs.dead_moistures_t[fuelModel*3+0],
                                y: bufs.dead_moistures_t[fuelModel*3+1],
                                z: bufs.dead_moistures_t[fuelModel*3+2] };
            const liveMoist = { x: bufs.live_moistures_t[fuelModel*2+0],
                                y: bufs.live_moistures_t[fuelModel*2+1] };

            // Time lag class assignment
            const tlc = { x: 0, y: 0, z: 0 };
            tlc.x = dsb.x > 192.0 ? deadMoist.x : (dsb.x > 48.0 ? deadMoist.y : deadMoist.z);
            tlc.y = dsb.y > 192.0 ? deadMoist.x : (dsb.y > 48.0 ? deadMoist.y : deadMoist.z);
            tlc.z = dsb.z > 192.0 ? deadMoist.x : (dsb.z > 48.0 ? deadMoist.y : deadMoist.z);

            const weightedFuelModel = tlc.x * d1h.x * d1h.y +
                                      tlc.y * d10h.x * d10h.y +
                                      tlc.z * d100h.x * d100h.y;

            const fm = [tlc.x, tlc.y, tlc.z, liveMoist.x, liveMoist.y];

            // Live extinction moisture
            let liveExtinction = 0.0;
            if (lh.y > 0.0 || lw.y > 0.0) {
                let fineDeadMoisture = 0.0;
                if (fded.x > 0.0) fineDeadMoisture = weightedFuelModel / fded.x;
                liveExtinction = (fded.z * (1.0 - fineDeadMoisture / fded.y)) - 0.226;
                liveExtinction = Math.max(liveExtinction, fded.y);
            }

            // Heat of ignition
            let heatOfIgnition =
                arf.x * ((250.0 + 1116.0 * fm[0]) * d1h.z * d1h.x +
                         (250.0 + 1116.0 * fm[1]) * d10h.z * d10h.x +
                         (250.0 + 1116.0 * fm[2]) * d100h.z * d100h.x) +
                arf.y * ((250.0 + 1116.0 * fm[3]) * lh.z * lh.x +
                         (250.0 + 1116.0 * fm[4]) * lw.z * lw.x);
            heatOfIgnition *= fded.w;

            const liveMoistureW = lh.z * fm[3] + lw.z * fm[4];
            const deadMoistureW = d1h.z * fm[0] + d10h.z * fm[1] + d100h.z * fm[2];

            // Reaction intensity
            let reactionIntensity = 0.0;
            if (liveExtinction > 0.0) {
                const r = liveMoistureW / liveExtinction;
                if (r < 1.0)
                    reactionIntensity += arf.w * (1.0 - 2.59*r + 5.11*r*r - 3.52*r*r*r);
            }
            if (fded.y > 0.0) {
                const r = deadMoistureW / fded.y;
                if (r < 1.0)
                    reactionIntensity += arf.z * (1.0 - 2.59*r + 5.11*r*r - 3.52*r*r*r);
            }

            let baseSpreadRate = 0.0;
            if (heatOfIgnition > 0.0)
                baseSpreadRate = reactionIntensity * rfl.y / heatOfIgnition;

            // Slope and wind factors
            const slopeFactor = swf.x * slope * slope;
            let windFactor = 0.0;
            if (windX > 0.0) windFactor = swf.y * Math.pow(windX, swf.z);

            let spreadModifier = slopeFactor + windFactor;
            let upslope = aspect >= 180.0 ? aspect - 180.0 : aspect + 180.0;

            let maxSpreadRate = 0.0;
            let spreadDirection = 0.0;
            let ellipseEccentricity = 0.0;
            let checkEffectiveWindspeed = 0;
            let updateEffectiveWindspeed = 0;
            let effectiveWindspeed = 0.0;

            if (baseSpreadRate <= 0.0) {
                maxSpreadRate = 0.0;
                spreadDirection = 0.0;
            } else if (spreadModifier <= 0.0) {
                maxSpreadRate = baseSpreadRate;
                spreadDirection = 0.0;
            } else if (slope <= 0.0) {
                effectiveWindspeed = windX;
                maxSpreadRate = baseSpreadRate * (1.0 + spreadModifier);
                spreadDirection = windY;
                checkEffectiveWindspeed = 1;
            } else if (windX <= 0.0) {
                maxSpreadRate = baseSpreadRate * (1.0 + spreadModifier);
                spreadDirection = upslope;
                updateEffectiveWindspeed = 1;
                checkEffectiveWindspeed = 1;
            } else if (Math.abs(windY - upslope) < 0.000001) {
                maxSpreadRate = baseSpreadRate * (1.0 + spreadModifier);
                spreadDirection = upslope;
                updateEffectiveWindspeed = 1;
                checkEffectiveWindspeed = 1;
            } else {
                let angleDelta = upslope <= windY ? windY - upslope : 360.0 - upslope + windY;
                angleDelta *= Math.PI / 180.0;
                const slopeRate = baseSpreadRate * slopeFactor;
                const windRate = baseSpreadRate * windFactor;
                const x = slopeRate + windRate * Math.cos(angleDelta);
                const y = windRate * Math.sin(angleDelta);
                const addedSpeed = Math.sqrt(x * x + y * y);
                maxSpreadRate = baseSpreadRate + addedSpeed;

                spreadModifier = maxSpreadRate / baseSpreadRate - 1.0;
                if (spreadModifier > 0.0) updateEffectiveWindspeed = 1;
                checkEffectiveWindspeed = 1;

                let addedAngle = 0.0;
                if (addedSpeed > 0.0)
                    addedAngle = Math.asin(Math.min(1.0, Math.max(-1.0, Math.abs(y) / addedSpeed)));

                let angleOffset = 0.0;
                if (x >= 0.0) {
                    angleOffset = y >= 0.0 ? addedAngle : 2.0 * Math.PI - addedAngle;
                } else {
                    angleOffset = y >= 0.0 ? Math.PI + addedAngle : Math.PI - angleOffset;
                }
                spreadDirection = upslope + angleOffset * 180.0 / Math.PI;
                if (spreadDirection > 360.0) spreadDirection -= 360.0;
            }

            if (updateEffectiveWindspeed === 1) {
                effectiveWindspeed = Math.pow(spreadModifier * swf.w, 1.0 / swf.z);
            }
            if (checkEffectiveWindspeed === 1) {
                const maxWind = 0.9 * reactionIntensity;
                if (effectiveWindspeed > maxWind) {
                    if (maxWind <= 0.0)
                        spreadModifier = 0.0;
                    else
                        spreadModifier = swf.y * Math.pow(maxWind, swf.z);
                    maxSpreadRate = baseSpreadRate * (1.0 + spreadModifier);
                    effectiveWindspeed = maxWind;
                }
            }

            if (effectiveWindspeed > 0.0) {
                const lwr = 1.0 + 0.002840909 * effectiveWindspeed;
                ellipseEccentricity = Math.sqrt(lwr * lwr - 1.0) / lwr;
            }

            const intensityModifier =
                3.4613 * (384.0 * (reactionIntensity / 0.189275)) *
                (0.30480060960) / (60.0 * fuelSAV);

            rothData[cell*4+0] = maxSpreadRate;
            rothData[cell*4+1] = spreadDirection;
            rothData[cell*4+2] = ellipseEccentricity;
            rothData[cell*4+3] = intensityModifier;
    }
    return rothData;
}

// Port of propagation.cu:73-97 — decompose roth_data into 16 directional spread rates
// Optional startCell/endCell for chunk-parallel processing via Web Workers
function computeDirectionalRates(simCols, simRows, rothData, fuelTypes, fuelSavAccel, cellSize, startCell, endCell) {
    const simSize = simCols * simRows;
    if (startCell === undefined) startCell = 0;
    if (endCell === undefined) endCell = simSize;
    const maxspreadrate = new Float32Array(simSize * 16);
    const curspreadrate = new Float32Array(simSize * 16);
    const intensityModifier = new Float32Array(simSize);
    const accelerationConstant = new Float32Array(simSize);
    let maxSpread = 0.0;

    for (let cellIdx = startCell; cellIdx < endCell; cellIdx++) {
            const msr = rothData[cellIdx*4+0];
            const dir = rothData[cellIdx*4+1];
            const ecc = rothData[cellIdx*4+2];

            for (let i = 0; i < 16; i++) {
                let rate;
                if (msr < 0) {
                    rate = 0.0;
                } else {
                    const denom = 1.0 - ecc * Math.cos(dir * Math.PI / 180.0 - ANGLES_16[i] * Math.PI / 180.0);
                    rate = msr * (1.0 - ecc) / denom;
                }
                maxspreadrate[i * simSize + cellIdx] = rate;
                curspreadrate[i * simSize + cellIdx] = 0.0;
                if (rate > maxSpread) maxSpread = rate;
            }

            intensityModifier[cellIdx] = rothData[cellIdx*4+3];
            const fm = fuelTypes[cellIdx];
            accelerationConstant[cellIdx] = fuelSavAccel[fm * 2 + 1] || 0.115;
    }

    // Compute neighbor distances (l_n)
    const ortho = cellSize;
    const diag = cellSize * Math.SQRT2;
    const superDist = Math.sqrt(cellSize * cellSize + (2 * cellSize) * (2 * cellSize));
    const l_n = new Float32Array([
        ortho, diag, ortho, diag, ortho, diag, ortho, diag,
        superDist, superDist, superDist, superDist,
        superDist, superDist, superDist, superDist
    ]);

    // Timestep: max_spread / cell_size (from BD.cu:41)
    const timestep = maxSpread > 0 ? maxSpread / cellSize : 1.0;

    return { maxspreadrate, curspreadrate, intensityModifier, accelerationConstant, l_n, timestep, maxSpread };
}

// Initialize burn distances (from BD.cu:48-52)
function initBurnDistances(simSize, l_n) {
    const burnDist = new Float32Array(simSize * 8);
    for (let i = 0; i < simSize; i++) {
        for (let j = 0; j < 8; j++) {
            burnDist[i * 8 + j] = l_n[j];
        }
    }
    return burnDist;
}

// Initialize ignition time array from fire ignition data
function initIgnitionTimes(simSize, fireData, notsetfire) {
    const ignTime = new Float32Array(simSize);
    for (let i = 0; i < simSize; i++) {
        if (fireData[i] === 1) {
            ignTime[i] = 0.0;
        } else {
            ignTime[i] = notsetfire;
        }
    }
    return ignTime;
}
