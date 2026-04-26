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

        this.device = await adapter.requestDevice({
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
        return {
            ignTimeA: s * 4,
            ignTimeB: s * 4,
            burnDist: s * 8 * 4,
            maxSpreadRate: s * 16 * 4,
            curSpreadRate: s * 16 * 4,
            intensityMod: s * 4,
            accConst: s * 4,
            I_o: s * 4,
            RAC: s * 4,
            activeMask: maskWords * 4,
            staging: s * 4,
            uniforms: 84,
            total: s * (4+4+32+64+64+4+4+4+4+4) + maskWords * 4 + 84,
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

        // Burn distances (8 per cell)
        this.burnDistBuf = this._createStorageBuffer(data.burnDist);

        // Spread rates (16 per cell)
        this.maxSpreadRateBuf = this._createStorageBuffer(data.maxspreadrate);
        this.curSpreadRateBuf = this._createStorageBuffer(data.curspreadrate);

        // Per-cell scalars
        this.intensityModBuf = this._createStorageBuffer(data.intensityModifier);
        this.accConstBuf = this._createStorageBuffer(data.accelerationConstant);
        this.ioBuf = this._createStorageBuffer(data.I_o);
        this.racBuf = this._createStorageBuffer(data.RAC);

        // Active-cell bitmask (1 bit per cell, packed into u32 words)
        this.maskWords = Math.ceil(simSize / 32);
        this.activeMaskBuf = device.createBuffer({
            size: this.maskWords * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // Simulation params uniform
        this.simParamsBuf = device.createBuffer({
            size: 32,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Mark-active params uniform (size, row_size, col_size, pad)
        this.markParamsBuf = device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(this.markParamsBuf, 0, new Uint32Array([simSize, this.simRows, this.simCols, 0]));

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
        } else {
            new Uint32Array(buf.getMappedRange()).set(typedArray);
        }
        buf.unmap();
        return buf;
    }

    async _loadShader(path) {
        const resp = await fetch(path);
        return await resp.text();
    }

    async _createPipelines() {
        const device = this.device;

        const [burnDistSrc, accelSrc, crownSrc, markActiveSrc] = await Promise.all([
            this._loadShader('/static/js/shaders/burn_dist.wgsl'),
            this._loadShader('/static/js/shaders/accelerate.wgsl'),
            this._loadShader('/static/js/shaders/crown_rate.wgsl'),
            this._loadShader('/static/js/shaders/mark_active.wgsl'),
        ]);

        // BurnDist pipeline
        this.burnDistPipeline = device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: device.createShaderModule({ code: burnDistSrc }),
                entryPoint: 'main',
            },
        });

        // Mark-active pipeline
        this.markActivePipeline = device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: device.createShaderModule({ code: markActiveSrc }),
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

        // BurnDist bind groups — two variants for ping-pong swap, now with active_mask
        // Variant AB: A=read, B=write
        this.burnDistBindGroupAB = device.createBindGroup({
            layout: this.burnDistPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.ignTimeBufA } },
                { binding: 1, resource: { buffer: this.ignTimeBufB } },
                { binding: 2, resource: { buffer: this.burnDistBuf } },
                { binding: 3, resource: { buffer: this.curSpreadRateBuf } },
                { binding: 4, resource: { buffer: this.simParamsBuf } },
                { binding: 5, resource: { buffer: this.activeMaskBuf } },
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
                { binding: 5, resource: { buffer: this.activeMaskBuf } },
            ],
        });

        // Mark-active bind groups — two variants matching which ign_time buffer to read
        // Variant A: read from buffer A
        this.markActiveBindGroupA = device.createBindGroup({
            layout: this.markActivePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.ignTimeBufA } },
                { binding: 1, resource: { buffer: this.activeMaskBuf } },
                { binding: 2, resource: { buffer: this.markParamsBuf } },
            ],
        });
        // Variant B: read from buffer B
        this.markActiveBindGroupB = device.createBindGroup({
            layout: this.markActivePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.ignTimeBufB } },
                { binding: 1, resource: { buffer: this.activeMaskBuf } },
                { binding: 2, resource: { buffer: this.markParamsBuf } },
            ],
        });

        // Accelerate bind group
        this.accelBindGroup = device.createBindGroup({
            layout: this.accelPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.curSpreadRateBuf } },
                { binding: 1, resource: { buffer: this.maxSpreadRateBuf } },
                { binding: 2, resource: { buffer: this.accConstBuf } },
                { binding: 3, resource: { buffer: this.accelParamsBuf } },
            ],
        });

        // Crown rate bind group
        this.crownBindGroup = device.createBindGroup({
            layout: this.crownPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.curSpreadRateBuf } },
                { binding: 1, resource: { buffer: this.maxSpreadRateBuf } },
                { binding: 2, resource: { buffer: this.intensityModBuf } },
                { binding: 3, resource: { buffer: this.ioBuf } },
                { binding: 4, resource: { buffer: this.racBuf } },
                { binding: 5, resource: { buffer: this.crownParamsBuf } },
            ],
        });
    }

    async run(stepCount, crowningEnabled, onProgress) {
        var t0 = performance.now();
        stepCount = stepCount || 600;
        crowningEnabled = crowningEnabled !== false;

        const device = this.device;
        const simSize = this.simSize;
        const workgroups = Math.ceil(simSize / 256);
        const cellWorkgroups = Math.ceil(simSize / 256);
        const maskWorkgroups = Math.ceil(simSize / 256); // one thread per cell for mark_active
        const timestep = this.timestep;

        // Update accel params once (static for the whole run)
        const accelData = new ArrayBuffer(16);
        new Uint32Array(accelData, 0, 2).set([simSize * 16, 0]);
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

        // Zero buffer for clearing the bitmask each step
        const zeroMask = new Uint32Array(this.maskWords);

        // Crown fire frequency: run every CROWN_INTERVAL steps
        const CROWN_INTERVAL = 10;

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

                // Clear the active bitmask to zero before marking
                device.queue.writeBuffer(this.activeMaskBuf, 0, zeroMask);

                // Pass 0: Mark active cells (reads current read-buffer)
                var markBG = (this._pingPong === 0) ? this.markActiveBindGroupA : this.markActiveBindGroupB;
                const pass0 = encoder.beginComputePass();
                pass0.setPipeline(this.markActivePipeline);
                pass0.setBindGroup(0, markBG);
                pass0.dispatchWorkgroups(maskWorkgroups);
                pass0.end();

                // Pass 1: BurnDist with ping-pong bind group + active mask
                var burnDistBG = (this._pingPong === 0) ? this.burnDistBindGroupAB : this.burnDistBindGroupBA;
                const pass1 = encoder.beginComputePass();
                pass1.setPipeline(this.burnDistPipeline);
                pass1.setBindGroup(0, burnDistBG);
                pass1.dispatchWorkgroups(workgroups);
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

                // Pass 3: Accelerate
                const pass3 = encoder.beginComputePass();
                pass3.setPipeline(this.accelPipeline);
                pass3.setBindGroup(0, this.accelBindGroup);
                pass3.dispatchWorkgroups(cellWorkgroups);
                pass3.end();
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
        if (this.intensityModBuf) this.intensityModBuf.destroy();
        if (this.accConstBuf) this.accConstBuf.destroy();
        if (this.ioBuf) this.ioBuf.destroy();
        if (this.racBuf) this.racBuf.destroy();
        if (this.activeMaskBuf) this.activeMaskBuf.destroy();
        if (this.simParamsBuf) this.simParamsBuf.destroy();
        if (this.markParamsBuf) this.markParamsBuf.destroy();
        if (this.accelParamsBuf) this.accelParamsBuf.destroy();
        if (this.crownParamsBuf) this.crownParamsBuf.destroy();
        if (this.stagingBuf) this.stagingBuf.destroy();
        this.device = null;
    }
}
