// burn_dist.wgsl — Port of kernel_BD.cu BurnDist kernel
// Fire propagation via burn distance method

struct SimParams {
    size: u32,
    row_size: u32,
    col_size: u32,
    _pad0: u32,
    time_step: f32,
    current_time: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var<storage, read_write>  ign_time_in:     array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write>  ign_time_out:    array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write>  burn_dist:       array<f32>;
@group(0) @binding(3) var<storage, read>        cur_spread_rate: array<f32>;
@group(0) @binding(4) var<uniform>              params:          SimParams;

const INF: f32 = 32767.0;

// Neighbor offsets: N, NE, E, SE, S, SW, W, NW
const N_COL = array<i32, 8>(0, 1, 1, 1, 0, -1, -1, -1);
const N_ROW = array<i32, 8>(1, 1, 0, -1, -1, -1, 0, 1);

// Atomic minimum on float via bitcast + CAS loop
// Exploits IEEE 754: for positive floats, bitwise ordering matches numeric ordering
fn atomic_min_float(idx: u32, new_val: f32) {
    let new_bits = bitcast<u32>(new_val);
    loop {
        let old_bits = atomicLoad(&ign_time_out[idx]);
        let old_val = bitcast<f32>(old_bits);
        if (old_val <= new_val) {
            break;
        }
        let result = atomicCompareExchangeWeak(&ign_time_out[idx], old_bits, new_bits);
        if (result.exchanged) {
            break;
        }
    }
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let cell = gid.x;
    if (cell >= params.size) {
        return;
    }

    let row = i32(cell / params.col_size);
    let col = i32(cell % params.col_size);

    let ign_time = bitcast<f32>(atomicLoad(&ign_time_in[cell]));
    if (ign_time >= INF) {
        return;
    }

    let dist_cell = cell * 8u;
    let roth_cell = cell * 16u;
    let t = params.current_time;
    let time_step = params.time_step;

    for (var n = 0u; n < 8u; n++) {
        let nrow = row + N_ROW[n];
        let ncol = col + N_COL[n];

        if (nrow < 0 || nrow >= i32(params.row_size) || ncol < 0 || ncol >= i32(params.col_size)) {
            continue;
        }

        let ncell = u32(ncol + nrow * i32(params.col_size));

        // Skip already-lit neighbors
        let ign_time_n = bitcast<f32>(atomicLoad(&ign_time_in[ncell]));
        if (ign_time_n < INF) {
            continue;
        }

        // Rate of spread in this direction
        let ROS = cur_spread_rate[roth_cell + n];

        // Update burn distance
        var dist = burn_dist[dist_cell + n];
        dist = dist - ROS * time_step;
        burn_dist[dist_cell + n] = dist;

        // Propagate fire when distance consumed
        if (dist <= 0.0) {
            let overshoot = -dist;
            var step_time = t;
            if (ROS > 0.0) {
                step_time = overshoot / ROS + t;
            }
            atomic_min_float(ncell, step_time);
        }
    }
}
