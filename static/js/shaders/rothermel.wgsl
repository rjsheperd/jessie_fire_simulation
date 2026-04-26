// rothermel.wgsl — GPU Rothermel fire spread model
// Combines updateSpreadData + computeDirectionalRates + initBurnDistances + ignition init
// One thread per cell. Eliminates CPU Rothermel bottleneck.

enable f16;

// Fuel model lookup table entry (packed as 13 vec4s = 52 floats per model)
struct FuelModel {
    dead_1h:      vec4<f32>,   // effectiveHeatingNumber, load, areaWeightingFactor, fuelMoisture
    dead_10h:     vec4<f32>,
    dead_100h:    vec4<f32>,
    live_h:       vec4<f32>,
    live_w:       vec4<f32>,
    dead_sav_b:   vec4<f32>,   // SAV[Dead1h], SAV[Dead10h], SAV[Dead100h], 100.0
    fded:         vec4<f32>,   // fineDeadRatio, extinctionMoisture, liveExtinction, fuelDensity
    areas_react:  vec4<f32>,   // deadArea, liveArea, deadReactionFactor, liveReactionFactor
    slope_wind:   vec4<f32>,   // slopeK, windK, windB, windE
    resid_flux:   vec4<f32>,   // residenceTime, propagatingFlux, SAV[LiveH], SAV[LiveW]
    fuel_sav_acc: vec4<f32>,   // fuelSAV, accelerationConstant, 0, 0
    moistures:    vec4<f32>,   // dead1h, dead10h, dead100h, 0
    live_moist:   vec4<f32>,   // liveH, liveW, 0, 0
};

struct RothParams {
    sim_size:       u32,
    sim_cols:       u32,
    sim_rows:       u32,
    num_fuel_models: u32,
    cell_size:      f32,
    notsetfire:     f32,
    _pad0:          f32,
    _pad1:          f32,
};

// 16 decomposition angles (degrees) — same as ANGLES_16 in rothermel.js
const ANGLES_16 = array<f32, 16>(
    90.0, 45.0, 0.0, -45.0, -90.0, -115.0, 180.0, 115.0,
    63.4349, 26.5651, -26.5651, -63.4349,
    -116.5651, -153.4349, 153.4349, 116.5651
);

const PI: f32 = 3.14159265358979;
const DEG2RAD: f32 = 0.01745329251994;
const RAD2DEG: f32 = 57.29577951308232;

// Inputs (3 storage + 1 uniform)
@group(0) @binding(0)  var<storage, read>        fuel_models:       array<FuelModel>;
@group(0) @binding(1)  var<storage, read>        cell_ints:         array<vec2<i32>>;    // [fuelType, fireIgnition] per cell
@group(0) @binding(2)  var<storage, read>        cell_floats:       array<vec4<f32>>;    // 2 vec4s per cell: [slope,aspect,windX,windY], [I_o,RAC,0,0]
@group(0) @binding(3)  var<uniform>              params:            RothParams;
// Outputs (6 storage)
@group(0) @binding(4)  var<storage, read_write>  max_spread_rate:   array<f32>;          // SoA: [dir * simSize + cell]
@group(0) @binding(5)  var<storage, read_write>  cur_spread_rate:   array<f32>;          // SoA
@group(0) @binding(6)  var<storage, read_write>  packed_scalars:    array<vec4<f32>>;    // [IM, accConst, I_o, RAC]
@group(0) @binding(7)  var<storage, read_write>  burn_dist:         array<f16>;          // 8 per cell
@group(0) @binding(8)  var<storage, read_write>  ign_time:          array<atomic<u32>>;  // 1 per cell
@group(0) @binding(9)  var<storage, read_write>  max_spread_atomic: atomic<u32>;         // global max

// Atomic max on float via bitcast + CAS loop (positive floats only)
fn atomic_max_float(new_val: f32) {
    let new_bits = bitcast<u32>(new_val);
    loop {
        let old_bits = atomicLoad(&max_spread_atomic);
        let old_val = bitcast<f32>(old_bits);
        if (old_val >= new_val) {
            break;
        }
        let result = atomicCompareExchangeWeak(&max_spread_atomic, old_bits, new_bits);
        if (result.exchanged) {
            break;
        }
    }
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let cell = gid.x;
    if (cell >= params.sim_size) {
        return;
    }

    let sim_size = params.sim_size;
    let cell_size = params.cell_size;

    // Read packed per-cell data
    let ci = cell_ints[cell];
    let cf0 = cell_floats[cell * 2u];       // slope, aspect, windX, windY
    let cf1 = cell_floats[cell * 2u + 1u];  // I_o, RAC, 0, 0

    // ─── Ignition time init ─────────────────────────────────────────
    var ign_val: f32;
    if (ci.y == 1) {
        ign_val = 0.0;
    } else {
        ign_val = params.notsetfire;
    }
    atomicStore(&ign_time[cell], bitcast<u32>(ign_val));

    // ─── Fuel model lookup ──────────────────────────────────────────
    let fuel_model = ci.x;

    if (fuel_model == 99 || fuel_model < 0) {
        // Non-burnable
        for (var i = 0u; i < 16u; i++) {
            max_spread_rate[i * sim_size + cell] = 0.0;
            cur_spread_rate[i * sim_size + cell] = 0.0;
        }
        packed_scalars[cell] = vec4<f32>(1.0, 0.115, cf1.x, cf1.y);
        let ortho = cell_size;
        let diag = cell_size * 1.41421356;
        let dc = cell * 8u;
        burn_dist[dc + 0u] = f16(ortho); burn_dist[dc + 1u] = f16(diag);
        burn_dist[dc + 2u] = f16(ortho); burn_dist[dc + 3u] = f16(diag);
        burn_dist[dc + 4u] = f16(ortho); burn_dist[dc + 5u] = f16(diag);
        burn_dist[dc + 6u] = f16(ortho); burn_dist[dc + 7u] = f16(diag);
        return;
    }

    let fm = u32(fuel_model);
    let f = fuel_models[fm];

    // Check burnable (dead_sav_b.w >= 50)
    if (f.dead_sav_b.w < 50.0) {
        for (var i = 0u; i < 16u; i++) {
            max_spread_rate[i * sim_size + cell] = 0.0;
            cur_spread_rate[i * sim_size + cell] = 0.0;
        }
        packed_scalars[cell] = vec4<f32>(1.0, f.fuel_sav_acc.y, cf1.x, cf1.y);
        let ortho = cell_size;
        let diag = cell_size * 1.41421356;
        let dc = cell * 8u;
        burn_dist[dc + 0u] = f16(ortho); burn_dist[dc + 1u] = f16(diag);
        burn_dist[dc + 2u] = f16(ortho); burn_dist[dc + 3u] = f16(diag);
        burn_dist[dc + 4u] = f16(ortho); burn_dist[dc + 5u] = f16(diag);
        burn_dist[dc + 6u] = f16(ortho); burn_dist[dc + 7u] = f16(diag);
        return;
    }

    // ─── Per-cell terrain/wind ──────────────────────────────────────
    let slope = cf0.x;
    let aspect = cf0.y;
    let wind_x = cf0.z;
    let wind_y = cf0.w;

    // ─── Time lag class assignment ──────────────────────────────────
    var tlc: vec3<f32>;
    if (f.dead_sav_b.x > 192.0) { tlc.x = f.moistures.x; } else if (f.dead_sav_b.x > 48.0) { tlc.x = f.moistures.y; } else { tlc.x = f.moistures.z; }
    if (f.dead_sav_b.y > 192.0) { tlc.y = f.moistures.x; } else if (f.dead_sav_b.y > 48.0) { tlc.y = f.moistures.y; } else { tlc.y = f.moistures.z; }
    if (f.dead_sav_b.z > 192.0) { tlc.z = f.moistures.x; } else if (f.dead_sav_b.z > 48.0) { tlc.z = f.moistures.y; } else { tlc.z = f.moistures.z; }

    let weighted_fuel = tlc.x * f.dead_1h.x * f.dead_1h.y +
                        tlc.y * f.dead_10h.x * f.dead_10h.y +
                        tlc.z * f.dead_100h.x * f.dead_100h.y;

    // Fuel moisture array: [dead1h, dead10h, dead100h, liveH, liveW]
    let fm0 = tlc.x; let fm1 = tlc.y; let fm2 = tlc.z;
    let fm3 = f.live_moist.x; let fm4 = f.live_moist.y;

    // ─── Live extinction moisture ───────────────────────────────────
    var live_ext = 0.0;
    if (f.live_h.y > 0.0 || f.live_w.y > 0.0) {
        var fine_dead_moisture = 0.0;
        if (f.fded.x > 0.0) { fine_dead_moisture = weighted_fuel / f.fded.x; }
        live_ext = (f.fded.z * (1.0 - fine_dead_moisture / f.fded.y)) - 0.226;
        live_ext = max(live_ext, f.fded.y);
    }

    // ─── Heat of ignition ───────────────────────────────────────────
    var heat_ign = f.areas_react.x * (
        (250.0 + 1116.0 * fm0) * f.dead_1h.z * f.dead_1h.x +
        (250.0 + 1116.0 * fm1) * f.dead_10h.z * f.dead_10h.x +
        (250.0 + 1116.0 * fm2) * f.dead_100h.z * f.dead_100h.x
    ) + f.areas_react.y * (
        (250.0 + 1116.0 * fm3) * f.live_h.z * f.live_h.x +
        (250.0 + 1116.0 * fm4) * f.live_w.z * f.live_w.x
    );
    heat_ign *= f.fded.w;

    let live_moisture_w = f.live_h.z * fm3 + f.live_w.z * fm4;
    let dead_moisture_w = f.dead_1h.z * fm0 + f.dead_10h.z * fm1 + f.dead_100h.z * fm2;

    // ─── Reaction intensity ─────────────────────────────────────────
    var reaction_intensity = 0.0;
    if (live_ext > 0.0) {
        let r = live_moisture_w / live_ext;
        if (r < 1.0) {
            reaction_intensity += f.areas_react.w * (1.0 - 2.59 * r + 5.11 * r * r - 3.52 * r * r * r);
        }
    }
    if (f.fded.y > 0.0) {
        let r = dead_moisture_w / f.fded.y;
        if (r < 1.0) {
            reaction_intensity += f.areas_react.z * (1.0 - 2.59 * r + 5.11 * r * r - 3.52 * r * r * r);
        }
    }

    var base_spread_rate = 0.0;
    if (heat_ign > 0.0) {
        base_spread_rate = reaction_intensity * f.resid_flux.y / heat_ign;
    }

    // ─── Slope and wind factors ─────────────────────────────────────
    let slope_factor = f.slope_wind.x * slope * slope;
    var wind_factor = 0.0;
    if (wind_x > 0.0) { wind_factor = f.slope_wind.y * pow(wind_x, f.slope_wind.z); }

    var spread_modifier = slope_factor + wind_factor;
    var upslope: f32;
    if (aspect >= 180.0) { upslope = aspect - 180.0; } else { upslope = aspect + 180.0; }

    var max_sr = 0.0;
    var spread_dir = 0.0;
    var eccentricity = 0.0;
    var check_ews = false;
    var update_ews = false;
    var ews = 0.0;

    if (base_spread_rate <= 0.0) {
        max_sr = 0.0;
        spread_dir = 0.0;
    } else if (spread_modifier <= 0.0) {
        max_sr = base_spread_rate;
        spread_dir = 0.0;
    } else if (slope <= 0.0) {
        ews = wind_x;
        max_sr = base_spread_rate * (1.0 + spread_modifier);
        spread_dir = wind_y;
        check_ews = true;
    } else if (wind_x <= 0.0) {
        max_sr = base_spread_rate * (1.0 + spread_modifier);
        spread_dir = upslope;
        update_ews = true;
        check_ews = true;
    } else if (abs(wind_y - upslope) < 0.000001) {
        max_sr = base_spread_rate * (1.0 + spread_modifier);
        spread_dir = upslope;
        update_ews = true;
        check_ews = true;
    } else {
        var angle_delta: f32;
        if (upslope <= wind_y) { angle_delta = wind_y - upslope; } else { angle_delta = 360.0 - upslope + wind_y; }
        angle_delta *= DEG2RAD;
        let slope_rate = base_spread_rate * slope_factor;
        let wind_rate = base_spread_rate * wind_factor;
        let x = slope_rate + wind_rate * cos(angle_delta);
        let y = wind_rate * sin(angle_delta);
        let added_speed = sqrt(x * x + y * y);
        max_sr = base_spread_rate + added_speed;

        spread_modifier = max_sr / base_spread_rate - 1.0;
        if (spread_modifier > 0.0) { update_ews = true; }
        check_ews = true;

        var added_angle = 0.0;
        if (added_speed > 0.0) {
            added_angle = asin(clamp(abs(y) / added_speed, -1.0, 1.0));
        }

        var angle_offset = 0.0;
        if (x >= 0.0) {
            if (y >= 0.0) { angle_offset = added_angle; } else { angle_offset = 2.0 * PI - added_angle; }
        } else {
            if (y >= 0.0) { angle_offset = PI + added_angle; } else { angle_offset = PI; }
        }
        spread_dir = upslope + angle_offset * RAD2DEG;
        if (spread_dir > 360.0) { spread_dir -= 360.0; }
    }

    if (update_ews) {
        ews = pow(spread_modifier * f.slope_wind.w, 1.0 / f.slope_wind.z);
    }
    if (check_ews) {
        let max_wind = 0.9 * reaction_intensity;
        if (ews > max_wind) {
            if (max_wind <= 0.0) {
                spread_modifier = 0.0;
            } else {
                spread_modifier = f.slope_wind.y * pow(max_wind, f.slope_wind.z);
            }
            max_sr = base_spread_rate * (1.0 + spread_modifier);
            ews = max_wind;
        }
    }

    if (ews > 0.0) {
        let lwr = 1.0 + 0.002840909 * ews;
        eccentricity = sqrt(lwr * lwr - 1.0) / lwr;
    }

    let intensity_modifier = 3.4613 * (384.0 * (reaction_intensity / 0.189275)) *
                             0.30480060960 / (60.0 * f.fuel_sav_acc.x);

    // ─── 16-direction ellipse decomposition ─────────────────────────
    var local_max_spread = 0.0;
    for (var i = 0u; i < 16u; i++) {
        var rate: f32;
        if (max_sr <= 0.0) {
            rate = 0.0;
        } else {
            let denom = 1.0 - eccentricity * cos(spread_dir * DEG2RAD - ANGLES_16[i] * DEG2RAD);
            rate = max_sr * (1.0 - eccentricity) / denom;
        }
        max_spread_rate[i * sim_size + cell] = rate;
        cur_spread_rate[i * sim_size + cell] = 0.0;
        if (rate > local_max_spread) { local_max_spread = rate; }
    }

    // Global max spread (for timestep computation)
    if (local_max_spread > 0.0) {
        atomic_max_float(local_max_spread);
    }

    // ─── Packed scalars: [IM, accConst, I_o, RAC] ──────────────────
    let acc_const = select(f.fuel_sav_acc.y, 0.115, f.fuel_sav_acc.y == 0.0);
    packed_scalars[cell] = vec4<f32>(
        intensity_modifier,
        acc_const,
        cf1.x,
        cf1.y
    );

    // ─── Burn distances (8 neighbors) ───────────────────────────────
    let ortho = cell_size;
    let diag = cell_size * 1.41421356;
    let dc = cell * 8u;
    burn_dist[dc + 0u] = f16(ortho); burn_dist[dc + 1u] = f16(diag);
    burn_dist[dc + 2u] = f16(ortho); burn_dist[dc + 3u] = f16(diag);
    burn_dist[dc + 4u] = f16(ortho); burn_dist[dc + 5u] = f16(diag);
    burn_dist[dc + 6u] = f16(ortho); burn_dist[dc + 7u] = f16(diag);
}
