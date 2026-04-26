// crown_rate.wgsl — Port of kernel_common.cu TestCrownRate kernel
// Crown fire escalation
// SoA layout: rate[dir * sim_size + cell]

struct CrownParams {
    size: u32,     // sim_size (cells)
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read_write>  cur_spread_rate:    array<f32>;
@group(0) @binding(1) var<storage, read_write>  max_spread_rate:    array<f32>;
@group(0) @binding(2) var<storage, read>        intensity_modifier: array<f32>;
@group(0) @binding(3) var<storage, read>        I_o:                array<f32>;
@group(0) @binding(4) var<storage, read>        RAC:                array<f32>;
@group(0) @binding(5) var<uniform>              params:             CrownParams;

fn clamp_val(val: f32, flr: f32, ceiling: f32) -> f32 {
    return max(flr, min(val, ceiling));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let cell = gid.x;
    if (cell >= params.size) {
        return;
    }

    // Direction 0 base rate (SoA: [0 * sim_size + cell] = [cell])
    let cur_rate_base = cur_spread_rate[cell];
    let max_rate_base = max_spread_rate[cell];

    if (max_rate_base <= 0.0) {
        return;
    }

    let I_b = cur_rate_base * intensity_modifier[cell];
    if (I_b > I_o[cell]) {
        let R_max_crown = 3.34 * max_rate_base;
        let surface_fuel_consumption = I_o[cell] * cur_rate_base / I_b;
        let denom = 0.9 * (RAC[cell] - surface_fuel_consumption);
        if (denom > 0.0) {
            let crown_coeff = -log(0.1) / denom;
            var CFB = 1.0 - exp(-crown_coeff * (cur_rate_base - surface_fuel_consumption));
            CFB = clamp_val(CFB, 0.0, 1.0);
            let crown_rate = cur_rate_base + CFB * (R_max_crown - cur_rate_base);
            if (crown_rate >= RAC[cell]) {
                if (crown_rate > max_rate_base) {
                    max_spread_rate[cell] = crown_rate;
                }
            }
        }
    }
}
