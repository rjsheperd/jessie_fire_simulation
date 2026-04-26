// webgpu_sim.js — WebGPU fire simulation orchestrator
// Manages GPU buffers, compute pipelines, and dispatch for the BD fire spread model

class FireSimWebGPU {
    constructor() {
        this.device = null;
        this.simSize = 0;
        this.simRows = 0;
        this.simCols = 0;
        // Instrumentation
        this.timing = {};
        this.adapterLimits = null;
        this.deviceLostReason = null;
        // Dual-buffer ping-pong state: 0 = A is read / B is write, 1 = B is read / A is write
        this._pingPong = 0;
    }

    async init(simCols, simRows, simData) {
        var t0 = performance.now();
        if (!navigator.gpu) throw new Error('WebGPU not supported');

        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) throw new Error('No WebGPU adapter found');

        // Capture adapter limits for stress test assertions
        this.adapterLimits = {
            maxBufferSize: adapter.limits.maxBufferSize,
            maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
            maxComputeWorkgroupsPerDimension: adapter.limits.maxComputeWorkgroupsPerDimension,
            maxComputeInvocationsPerWorkgroup: adapter.limits.maxComputeInvocationsPerWorkgroup,
        };

        // Request shader-f16 for half-precision burn_dist buffer
        var requiredFeatures = [];
        if (adapter.features.has('shader-f16')) {
            requiredFeatures.push('shader-f16');
        }
        this.hasF16 = requiredFeatures.indexOf('shader-f16') >= 0;

        this.device = await adapter.requestDevice({
            requiredFeatures: requiredFeatures,
            requiredLimits: {
                maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
                maxBufferSize: adapter.limits.maxBufferSize,
            }
        });

        // Device lost handler
        this.device.lost.then(function(info) {
            this.deviceLostReason = info.reason + ': ' + info.message;
            console.error('WebGPU device lost:', info.reason, info.message);
        }.bind(this));

        this.simCols = simCols;
        this.simRows = simRows;
        this.simSize = simCols * simRows;

        await this._createBuffers(simData);
        await this._createPipelines();
        await this._createBindGroups();
        this.timing.init_ms = performance.now() - t0;
    }

    getBufferMemoryUsage() {
        var s = this.simSize;
        var maskWords = Math.ceil(s / 32);
        var burnDistBytes = this.hasF16 ? s * 8 * 2 : s * 8 * 4;
        return {
            ignTimeA: s * 4,
            ignTimeB: s * 4,
            burnDist: burnDistBytes,
            maxSpreadRate: s * 16 * 4,
            curSpreadRate: s * 16 * 4,
            packedScalars: s * 16,
            activeIndices: s * 4,
            staging: s * 4,
            uniforms: 84,
            total: s * (4+4+64+64+16+4+4) + burnDistBytes + 16 + 84,
        };
    }

    getTimingReport() {
        return Object.assign({}, this.timing);
    }

    async _createBuffers(data) {
        const device = this.device;
        const simSize = this.simSize;

        // Dual ignition-time atomic buffers (both are atomic<u32>, no separate f32 buffer)
        const ignU32A = new Uint32Array(data.ignTimeIn.buffer.slice(0));
        this.ignTimeBufA = this._createStorageBuffer(ignU32A);
        const ignU32B = new Uint32Array(data.ignTimeIn.buffer.slice(0));
        this.ignTimeBufB = this._createStorageBuffer(ignU32B);

        // Burn distances (8 per cell, f16 if supported)
        if (this.hasF16) {
            var f16Data = this._packF32toF16(data.burnDist);
            this.burnDistBuf = this._createStorageBuffer(f16Data);
        } else {
            this.burnDistBuf = this._createStorageBuffer(data.burnDist);
        }

        // Spread rates (16 per cell)
        this.maxSpreadRateBuf = this._createStorageBuffer(data.maxspreadrate);
        this.curSpreadRateBuf = this._createStorageBuffer(data.curspreadrate);

        // Per-cell scalars packed as vec4: [IM, accConst, I_o, RAC]
        this.packedScalarsBuf = this._createStorageBuffer(data.packedScalars);

        // Compacted active-cell index buffer (worst case: all cells active)
        this.activeIndicesBuf = device.createBuffer({
            size: simSize * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // Atomic counter for compaction (single u32)
        this.activeCountBuf = device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // Indirect dispatch args buffer [workgroups_x, 1, 1]
        this.indirectArgsBuf = device.createBuffer({
            size: 12,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.INDIRECT,
        });

        // Simulation params uniform
        this.simParamsBuf = device.createBuffer({
            size: 32,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Compact-active params uniform (size, row_size, col_size, pad)
        this.compactParamsBuf = device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(this.compactParamsBuf, 0, new Uint32Array([simSize, this.simRows, this.simCols, 0]));

        // Accel params uniform
        this.accelParamsBuf = device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Crown params uniform
        this.crownParamsBuf = device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(this.crownParamsBuf, 0, new Uint32Array([simSize, 0, 0, 0]));

        // Staging buffer for readback
        this.stagingBuf = device.createBuffer({
            size: simSize * 4,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });

        this.timestep = data.timestep;
    }

    _createStorageBuffer(typedArray) {
        const buf = this.device.createBuffer({
            size: typedArray.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true,
        });
        if (typedArray instanceof Float32Array) {
            new Float32Array(buf.getMappedRange()).set(typedArray);
        } else if (typedArray instanceof Uint16Array) {
            new Uint16Array(buf.getMappedRange()).set(typedArray);
        } else {
            new Uint32Array(buf.getMappedRange()).set(typedArray);
        }
        buf.unmap();
        return buf;
    }

    // Pack Float32Array into Uint16Array of IEEE 754 half-precision floats
    _packF32toF16(f32arr) {
        var u16 = new Uint16Array(f32arr.length);
        var dv = new DataView(new ArrayBuffer(4));
        for (var i = 0; i < f32arr.length; i++) {
            dv.setFloat32(0, f32arr[i], true);
            var bits = dv.getUint32(0, true);
            var sign = (bits >> 31) & 1;
            var exp = (bits >> 23) & 0xFF;
            var frac = bits & 0x7FFFFF;
            var h;
            if (exp === 0) {
                h = (sign << 15); // zero or subnormal → zero
            } else if (exp === 0xFF) {
                h = (sign << 15) | 0x7C00 | (frac ? 1 : 0); // inf or NaN
            } else {
                var newExp = exp - 127 + 15;
                if (newExp >= 31) {
                    h = (sign << 15) | 0x7C00; // overflow → inf
                } else if (newExp <= 0) {
                    h = (sign << 15); // underflow → zero
                } else {
                    h = (sign << 15) | (newExp << 10) | (frac >> 13);
                }
            }
            u16[i] = h;
        }
        return u16;
    }

    async _loadShader(path) {
        const resp = await fetch(path);
        return await resp.text();
    }

    async _createPipelines() {
        const device = this.device;

        var [burnDistSrc, accelSrc, crownSrc, compactSrc] = await Promise.all([
            this._loadShader('/static/js/shaders/burn_dist.wgsl'),
            this._loadShader('/static/js/shaders/accelerate.wgsl'),
            this._loadShader('/static/js/shaders/crown_rate.wgsl'),
            this._loadShader('/static/js/shaders/compact_active.wgsl'),
        ]);

        // If f16 not supported, patch the shader to use f32 for burn_dist
        if (!this.hasF16) {
            burnDistSrc = burnDistSrc
                .replace('enable f16;\n', '')
                .replace(/array<f16>/g, 'array<f32>')
                .replace(/f16\(/g, '(')
                .replace(/f32\(burn_dist/g, 'burn_dist');
        }

        // BurnDist pipeline
        this.burnDistPipeline = device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: device.createShaderModule({ code: burnDistSrc }),
                entryPoint: 'main',
            },
        });

        // Compact-active pipeline
        this.compactPipeline = device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: device.createShaderModule({ code: compactSrc }),
                entryPoint: 'main',
            },
        });

        // Accelerate pipeline
        this.accelPipeline = device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: device.createShaderModule({ code: accelSrc }),
                entryPoint: 'main',
            },
        });

        // Crown rate pipeline
        this.crownPipeline = device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: device.createShaderModule({ code: crownSrc }),
                entryPoint: 'main',
            },
        });
    }

    _createBindGroups() {
        const device = this.device;

        // BurnDist bind groups — two variants for ping-pong swap, with compacted active indices
        // Variant AB: A=read, B=write
        this.burnDistBindGroupAB = device.createBindGroup({
            layout: this.burnDistPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.ignTimeBufA } },
                { binding: 1, resource: { buffer: this.ignTimeBufB } },
                { binding: 2, resource: { buffer: this.burnDistBuf } },
                { binding: 3, resource: { buffer: this.curSpreadRateBuf } },
                { binding: 4, resource: { buffer: this.simParamsBuf } },
                { binding: 5, resource: { buffer: this.activeIndicesBuf } },
            ],
        });
        // Variant BA: B=read, A=write
        this.burnDistBindGroupBA = device.createBindGroup({
            layout: this.burnDistPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.ignTimeBufB } },
                { binding: 1, resource: { buffer: this.ignTimeBufA } },
                { binding: 2, resource: { buffer: this.burnDistBuf } },
                { binding: 3, resource: { buffer: this.curSpreadRateBuf } },
                { binding: 4, resource: { buffer: this.simParamsBuf } },
                { binding: 5, resource: { buffer: this.activeIndicesBuf } },
            ],
        });

        // Compact-active bind groups — two variants matching which ign_time buffer to read
        // Variant A: read from buffer A
        this.compactBindGroupA = device.createBindGroup({
            layout: this.compactPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.ignTimeBufA } },
                { binding: 1, resource: { buffer: this.activeCountBuf } },
                { binding: 2, resource: { buffer: this.activeIndicesBuf } },
                { binding: 3, resource: { buffer: this.indirectArgsBuf } },
                { binding: 4, resource: { buffer: this.compactParamsBuf } },
            ],
        });
        // Variant B: read from buffer B
        this.compactBindGroupB = device.createBindGroup({
            layout: this.compactPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.ignTimeBufB } },
                { binding: 1, resource: { buffer: this.activeCountBuf } },
                { binding: 2, resource: { buffer: this.activeIndicesBuf } },
                { binding: 3, resource: { buffer: this.indirectArgsBuf } },
                { binding: 4, resource: { buffer: this.compactParamsBuf } },
            ],
        });

        // Accelerate bind group (packed_scalars replaces acc_const)
        this.accelBindGroup = device.createBindGroup({
            layout: this.accelPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.curSpreadRateBuf } },
                { binding: 1, resource: { buffer: this.maxSpreadRateBuf } },
                { binding: 2, resource: { buffer: this.packedScalarsBuf } },
                { binding: 3, resource: { buffer: this.accelParamsBuf } },
            ],
        });

        // Crown rate bind group (packed_scalars replaces IM, I_o, RAC)
        this.crownBindGroup = device.createBindGroup({
            layout: this.crownPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.curSpreadRateBuf } },
                { binding: 1, resource: { buffer: this.maxSpreadRateBuf } },
                { binding: 2, resource: { buffer: this.packedScalarsBuf } },
                { binding: 3, resource: { buffer: this.crownParamsBuf } },
            ],
        });
    }

    async run(stepCount, crowningEnabled, onProgress) {
        var t0 = performance.now();
        stepCount = stepCount || 600;
        crowningEnabled = crowningEnabled !== false;

        const device = this.device;
        const simSize = this.simSize;
        const cellWorkgroups = Math.ceil(simSize / 256);
        const timestep = this.timestep;

        // Update accel params once (static for the whole run)
        // SoA layout: sim_size is cell count (not * 16)
        const accelData = new ArrayBuffer(16);
        new Uint32Array(accelData, 0, 2).set([simSize, 0]);
        new Float32Array(accelData, 8, 2).set([timestep, 0]);
        device.queue.writeBuffer(this.accelParamsBuf, 0, accelData);

        // Pre-bake static sim params (everything except step_index / current_time)
        const paramTemplate = new ArrayBuffer(32);
        const paramU32 = new Uint32Array(paramTemplate);
        const paramF32 = new Float32Array(paramTemplate);
        paramU32[0] = simSize;       // size
        paramU32[1] = this.simRows;  // row_size
        paramU32[2] = this.simCols;  // col_size
        paramF32[4] = timestep;      // time_step
        paramF32[6] = 0;             // pad
        paramF32[7] = 0;             // pad

        // Zero buffers for clearing compaction state each step
        const zeroCount = new Uint32Array([0]);
        const zeroIndirect = new Uint32Array([0, 1, 1]);

        // Crown fire frequency: run every CROWN_INTERVAL steps
        const CROWN_INTERVAL = 10;

        // Accelerate frequency: run every ACCEL_INTERVAL steps
        const ACCEL_INTERVAL = 5;

        // Submit all steps without per-batch GPU sync
        const BATCH_SIZE = 50;
        for (let step = 0; step < stepCount; step += BATCH_SIZE) {
            const batchEnd = Math.min(step + BATCH_SIZE, stepCount);
            const encoder = device.createCommandEncoder();

            for (let s = step; s < batchEnd; s++) {
                // Update step_index and current_time
                paramU32[3] = s;
                paramF32[5] = s * timestep;
                device.queue.writeBuffer(this.simParamsBuf, 0, paramTemplate);

                // Clear compaction counter and indirect args before compact pass
                device.queue.writeBuffer(this.activeCountBuf, 0, zeroCount);
                device.queue.writeBuffer(this.indirectArgsBuf, 0, zeroIndirect);

                // Pass 0: Compact active cells (builds active_indices + indirect args)
                var compactBG = (this._pingPong === 0) ? this.compactBindGroupA : this.compactBindGroupB;
                const pass0 = encoder.beginComputePass();
                pass0.setPipeline(this.compactPipeline);
                pass0.setBindGroup(0, compactBG);
                pass0.dispatchWorkgroups(cellWorkgroups);
                pass0.end();

                // Pass 1: BurnDist with indirect dispatch (only active cells)
                var burnDistBG = (this._pingPong === 0) ? this.burnDistBindGroupAB : this.burnDistBindGroupBA;
                const pass1 = encoder.beginComputePass();
                pass1.setPipeline(this.burnDistPipeline);
                pass1.setBindGroup(0, burnDistBG);
                pass1.dispatchWorkgroupsIndirect(this.indirectArgsBuf, 0);
                pass1.end();

                // Swap ping-pong: output becomes next step's input
                this._pingPong = 1 - this._pingPong;

                // Pass 2: Crown rate (every CROWN_INTERVAL steps)
                if (crowningEnabled && (s % CROWN_INTERVAL === 0)) {
                    const pass2 = encoder.beginComputePass();
                    pass2.setPipeline(this.crownPipeline);
                    pass2.setBindGroup(0, this.crownBindGroup);
                    pass2.dispatchWorkgroups(cellWorkgroups);
                    pass2.end();
                }

                // Pass 3: Accelerate (every ACCEL_INTERVAL steps)
                if (s % ACCEL_INTERVAL === 0) {
                    const pass3 = encoder.beginComputePass();
                    pass3.setPipeline(this.accelPipeline);
                    pass3.setBindGroup(0, this.accelBindGroup);
                    pass3.dispatchWorkgroups(cellWorkgroups);
                    pass3.end();
                }
            }

            device.queue.submit([encoder.finish()]);

            if (onProgress) {
                onProgress(Math.min(batchEnd, stepCount), stepCount);
            }
        }

        // Single GPU sync at end of all steps
        await device.queue.onSubmittedWorkDone();
        this.timing.run_ms = performance.now() - t0;
    }

    async readResults() {
        var t0 = performance.now();
        const device = this.device;

        // _pingPong=0 means A is read (has latest data), =1 means B is read
        var resultBuf = (this._pingPong === 0) ? this.ignTimeBufA : this.ignTimeBufB;

        const encoder = device.createCommandEncoder();
        encoder.copyBufferToBuffer(resultBuf, 0, this.stagingBuf, 0, this.simSize * 4);
        device.queue.submit([encoder.finish()]);

        await this.stagingBuf.mapAsync(GPUMapMode.READ);
        const result = new Float32Array(this.stagingBuf.getMappedRange().slice(0));
        this.stagingBuf.unmap();
        this.timing.readback_ms = performance.now() - t0;
        return result;
    }

    destroy() {
        if (this.ignTimeBufA) this.ignTimeBufA.destroy();
        if (this.ignTimeBufB) this.ignTimeBufB.destroy();
        if (this.burnDistBuf) this.burnDistBuf.destroy();
        if (this.maxSpreadRateBuf) this.maxSpreadRateBuf.destroy();
        if (this.curSpreadRateBuf) this.curSpreadRateBuf.destroy();
        if (this.packedScalarsBuf) this.packedScalarsBuf.destroy();
        if (this.activeIndicesBuf) this.activeIndicesBuf.destroy();
        if (this.activeCountBuf) this.activeCountBuf.destroy();
        if (this.indirectArgsBuf) this.indirectArgsBuf.destroy();
        if (this.simParamsBuf) this.simParamsBuf.destroy();
        if (this.compactParamsBuf) this.compactParamsBuf.destroy();
        if (this.accelParamsBuf) this.accelParamsBuf.destroy();
        if (this.crownParamsBuf) this.crownParamsBuf.destroy();
        if (this.stagingBuf) this.stagingBuf.destroy();
        this.device = null;
    }
}
