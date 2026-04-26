// mark_active.wgsl — Build active-cell bitmask for narrow-band optimization
// A cell is "active" if it has ignited AND at least one neighbor has not.
// This identifies exactly the fire front perimeter.

struct MarkParams {
    size: u32,
    row_size: u32,
    col_size: u32,
    _pad0: u32,
};

@group(0) @binding(0) var<storage, read_write>  ign_time:    array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write>  active_mask: array<atomic<u32>>;
@group(0) @binding(2) var<uniform>              params:      MarkParams;

const INF: f32 = 32767.0;

const N_COL = array<i32, 8>(0, 1, 1, 1, 0, -1, -1, -1);
const N_ROW = array<i32, 8>(1, 1, 0, -1, -1, -1, 0, 1);

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let cell = gid.x;
    if (cell >= params.size) {
        return;
    }

    let ign = bitcast<f32>(atomicLoad(&ign_time[cell]));

    // Not ignited → not active
    if (ign >= INF) {
        return;
    }

    // Check if any neighbor is unignited
    let row = i32(cell / params.col_size);
    let col = i32(cell % params.col_size);
    var has_unignited_neighbor = false;

    for (var n = 0u; n < 8u; n++) {
        let nrow = row + N_ROW[n];
        let ncol = col + N_COL[n];

        if (nrow < 0 || nrow >= i32(params.row_size) || ncol < 0 || ncol >= i32(params.col_size)) {
            continue;
        }

        let ncell = u32(ncol + nrow * i32(params.col_size));
        let nign = bitcast<f32>(atomicLoad(&ign_time[ncell]));
        if (nign >= INF) {
            has_unignited_neighbor = true;
            break;
        }
    }

    if (has_unignited_neighbor) {
        // Set this cell's bit in the bitmask via atomic OR
        let word = cell / 32u;
        let bit = cell % 32u;
        atomicOr(&active_mask[word], 1u << bit);
    }
}
