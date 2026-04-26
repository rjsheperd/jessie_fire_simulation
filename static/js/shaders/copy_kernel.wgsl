// copy_kernel.wgsl — Port of kernel_BD.cu copyKernelBD
// Ping-pong: copy ign_time_out (atomic<u32>) → ign_time_in (f32)

@group(0) @binding(0) var<storage, read_write>  ign_time_in:  array<f32>;
@group(0) @binding(1) var<storage, read_write>  ign_time_out: array<atomic<u32>>;

struct CopyParams {
    size: u32,
};

@group(0) @binding(2) var<uniform> params: CopyParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let cell = gid.x;
    if (cell >= params.size) {
        return;
    }
    let val_bits = atomicLoad(&ign_time_out[cell]);
    ign_time_in[cell] = bitcast<f32>(val_bits);
}
