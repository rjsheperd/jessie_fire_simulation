// fim_solve.wgsl — Fast Iterative Method fire propagation
// Replaces the fixed-timestep BurnDist loop with direct arrival-time computation.
// Each active cell propagates fire to unignited neighbors using T = T_cell + dist / ROS.
// Newly reached neighbors are appended to the next active list for the next iteration.

struct FimParams {
    size:      u32,
    row_size:  u32,
    col_size:  u32,
    _pad0:     u32,
    cell_size: f32,
    diag_size: f32,
    _pad1:     f32,
    _pad2:     f32,
};

@group(0) @binding(0) var<storage, read_write> arrival_time:     array<atomic<u32>>;
@group(0) @binding(1) var<storage, read>       max_spread_rate:  array<f32>;
@group(0) @binding(2) var<storage, read>       active_in:        array<u32>;
@group(0) @binding(3) var<storage, read_write> active_out:       array<u32>;
@group(0) @binding(4) var<storage, read_write> active_out_count: atomic<u32>;
@group(0) @binding(5) var<storage, read_write> indirect_out:     array<atomic<u32>>;
@group(0) @binding(6) var<uniform>             params:           FimParams;

const INF: f32 = 32767.0;

// Neighbor offsets: N, NE, E, SE, S, SW, W, NW (same as burn_dist.wgsl)
const N_COL = array<i32, 8>(0, 1, 1, 1, 0, -1, -1, -1);
const N_ROW = array<i32, 8>(1, 1, 0, -1, -1, -1, 0, 1);

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    // First thread initializes indirect dispatch Y, Z for next iteration
    if (gid.x == 0u) {
        atomicStore(&indirect_out[1u], 1u);
        atomicStore(&indirect_out[2u], 1u);
    }

    let cell = active_in[gid.x];
    if (cell >= params.size) {
        return;
    }

    let row = i32(cell / params.col_size);
    let col = i32(cell % params.col_size);
    let t_cell = bitcast<f32>(atomicLoad(&arrival_time[cell]));
    if (t_cell >= INF) {
        return;
    }

    let sim_size = params.size;

    for (var n = 0u; n < 8u; n++) {
        let nrow = row + N_ROW[n];
        let ncol = col + N_COL[n];

        if (nrow < 0 || nrow >= i32(params.row_size) || ncol < 0 || ncol >= i32(params.col_size)) {
            continue;
        }

        let ncell = u32(ncol + nrow * i32(params.col_size));

        // Rate of spread FROM this cell IN direction n (SoA layout)
        let ros = max_spread_rate[n * sim_size + cell];
        if (ros <= 0.0) {
            continue;
        }

        // Distance: orthogonal for even directions, diagonal for odd
        let dist = select(params.cell_size, params.diag_size, n % 2u == 1u);
        let t_candidate = t_cell + dist / ros;

        // Atomic min with CAS loop — only append to active_out on successful update
        let new_bits = bitcast<u32>(t_candidate);
        loop {
            let old_bits = atomicLoad(&arrival_time[ncell]);
            let old_val = bitcast<f32>(old_bits);
            if (old_val <= t_candidate) {
                break;
            }
            let result = atomicCompareExchangeWeak(&arrival_time[ncell], old_bits, new_bits);
            if (result.exchanged) {
                // Successfully updated — add neighbor to next active list
                let slot = atomicAdd(&active_out_count, 1u);
                active_out[slot] = ncell;
                atomicMax(&indirect_out[0u], (slot + 1u + 255u) / 256u);
                break;
            }
        }
    }
}
