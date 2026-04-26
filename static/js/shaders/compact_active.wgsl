// compact_active.wgsl — Build compacted active-cell index + indirect dispatch args
// Replaces mark_active.wgsl: instead of a bitmask, produces a dense list of
// active cell IDs that BurnDist indexes into via dispatchWorkgroupsIndirect.

struct MarkParams {
    size: u32,
    row_size: u32,
    col_size: u32,
    _pad0: u32,
};

@group(0) @binding(0) var<storage, read_write>  ign_time:       array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write>  active_count:   atomic<u32>;       // single counter
@group(0) @binding(2) var<storage, read_write>  active_indices: array<u32>;        // compacted cell IDs
@group(0) @binding(3) var<storage, read_write>  indirect_args:  array<atomic<u32>>; // [workgroups_x, 1, 1]
@group(0) @binding(4) var<uniform>              params:         MarkParams;

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
        // Atomic append: get slot, write cell ID
        let slot = atomicAdd(&active_count, 1u);
        active_indices[slot] = cell;
        // Update indirect dispatch arg (workgroups_x = ceil(count / 256))
        // We write this after every append — wasteful but correct.
        // Only the final value matters; GPU executes indirect dispatch after this pass.
        atomicMax(&indirect_args[0u], (slot + 1u + 255u) / 256u);
    }
}
