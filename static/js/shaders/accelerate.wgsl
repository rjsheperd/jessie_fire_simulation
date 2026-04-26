// accelerate.wgsl — Port of kernel_common.cu Accelerate kernel
// Accelerates curspreadrate toward maxspreadrate
// SoA layout: rate[dir * sim_size + cell]

struct AccelParams {
    sim_size: u32,  // number of cells
    _pad0: u32,
    time_step: f32,
    _pad1: f32,
};

@group(0) @binding(0) var<storage, read_write>  cur_spread_rate:  array<f32>;
@group(0) @binding(1) var<storage, read>        max_spread_rate:  array<f32>;
@group(0) @binding(2) var<storage, read>        packed_scalars:   array<vec4<f32>>;  // [IM, accConst, I_o, RAC]
@group(0) @binding(3) var<uniform>              params:           AccelParams;

fn clamp_val(val: f32, flr: f32, ceiling: f32) -> f32 {
    return max(flr, min(val, ceiling));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let cell = gid.x;
    if (cell >= params.sim_size) {
        return;
    }

    // Check if cell is burnable (direction 0's max rate, SoA: [0 * sim_size + cell])
    let max_rate_base = max_spread_rate[cell];
    if (max_rate_base <= 0.0) {
        return;
    }

    let acceleration_constant = packed_scalars[cell].y;
    let time_step = params.time_step;
    let sim_size = params.sim_size;

    for (var i = 0u; i < 16u; i++) {
        let idx = i * sim_size + cell;
        let max_rate = max_spread_rate[idx];
        var current = cur_spread_rate[idx];

        // Clamp current to max
        if (max_rate < current) {
            current = max_rate_base;
        }

        let ratio = current / max_rate_base;
        cur_spread_rate[idx] = clamp_val(time_step - ratio, 0.0, 1.0) *
                               (max_rate_base - current) + current;
    }
}
