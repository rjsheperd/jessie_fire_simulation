// fuel_models.js — Port of FuelModel.h and FuelMoisture.h
// Rothermel fuel model definitions and parsers

const FuelClass = { Dead1h: 0, Dead10h: 1, Dead100h: 2, LiveH: 3, LiveW: 4, NumClasses: 5 };

class FuelModel {
    constructor(modelNumber, dead1h, dead10h, dead100h, liveHerbaceous, liveWoody,
                dead1hSAV, liveHerbaceousSAV, liveWoodySAV, fuelDepth,
                extinctionMoisture, deadHeatContent, liveHeatContent) {

        const NC = FuelClass.NumClasses;
        if (modelNumber === undefined) {
            this.modelNumber = -9999;
            this.burnable = false;
            this.load = new Float32Array(NC);
            this.SAV = new Float32Array(NC);
            this.density = new Float32Array(NC);
            this.heatContent = new Float32Array(NC);
            this.totalSilica = new Float32Array(NC);
            this.effectiveSilica = new Float32Array(NC);
            this.surfaceArea = new Float32Array(NC);
            this.effectiveHeatingNumber = new Float32Array(NC);
            this.areaWeightingFactor = new Float32Array(NC);
            this.fuelMoisture = new Float32Array(NC);
            return;
        }

        this.burnable = true;
        this.modelNumber = modelNumber;
        this.fuelDepth = fuelDepth;
        this.extinctionMoisture = extinctionMoisture;
        this.rosAdjustment = 0.0;

        this.load = new Float32Array(NC);
        this.SAV = new Float32Array(NC);
        this.density = new Float32Array(NC);
        this.heatContent = new Float32Array(NC);
        this.totalSilica = new Float32Array(NC);
        this.effectiveSilica = new Float32Array(NC);
        this.surfaceArea = new Float32Array(NC);
        this.effectiveHeatingNumber = new Float32Array(NC);
        this.areaWeightingFactor = new Float32Array(NC);
        this.fuelMoisture = new Float32Array(NC);

        this.load[FuelClass.Dead1h] = dead1h;
        this.load[FuelClass.Dead10h] = dead10h;
        this.load[FuelClass.Dead100h] = dead100h;
        this.load[FuelClass.LiveH] = liveHerbaceous;
        this.load[FuelClass.LiveW] = liveWoody;

        this.SAV[FuelClass.Dead1h] = dead1hSAV;
        this.SAV[FuelClass.Dead10h] = 109.0;
        this.SAV[FuelClass.Dead100h] = 30.0;
        this.SAV[FuelClass.LiveH] = liveHerbaceousSAV;
        this.SAV[FuelClass.LiveW] = liveWoodySAV;

        for (let i = 0; i < NC; i++) {
            this.density[i] = 32.0;
            this.totalSilica[i] = 0.0555;
            this.effectiveSilica[i] = 0.0100;
        }

        this.heatContent[FuelClass.Dead1h] = deadHeatContent;
        this.heatContent[FuelClass.Dead10h] = deadHeatContent;
        this.heatContent[FuelClass.Dead100h] = deadHeatContent;
        this.heatContent[FuelClass.LiveH] = liveHeatContent;
        this.heatContent[FuelClass.LiveW] = liveHeatContent;

        this.numFuelParticles = 0;
        for (let i = 0; i < NC; i++) {
            if (this.load[i] > 0.0) this.numFuelParticles++;
        }

        for (let i = 0; i < NC; i++) {
            if (this.density[i] > 0.0)
                this.surfaceArea[i] = this.load[i] * this.SAV[i] / this.density[i];
            else
                this.surfaceArea[i] = 0.0;

            if (this.SAV[i] > 0.0)
                this.effectiveHeatingNumber[i] = Math.exp(-138.0 / this.SAV[i]);
            else
                this.effectiveHeatingNumber[i] = 0.0;

            this.areaWeightingFactor[i] = 0.0;
        }

        this.liveArea = this.surfaceArea[FuelClass.LiveH] + this.surfaceArea[FuelClass.LiveW];
        this.deadArea = this.surfaceArea[FuelClass.Dead1h] + this.surfaceArea[FuelClass.Dead10h] + this.surfaceArea[FuelClass.Dead100h];

        if (this.liveArea > 0.0) {
            this.areaWeightingFactor[FuelClass.LiveH] = this.surfaceArea[FuelClass.LiveH] / this.liveArea;
            this.areaWeightingFactor[FuelClass.LiveW] = this.surfaceArea[FuelClass.LiveW] / this.liveArea;
        }
        if (this.deadArea > 0.0) {
            this.areaWeightingFactor[FuelClass.Dead1h] = this.surfaceArea[FuelClass.Dead1h] / this.deadArea;
            this.areaWeightingFactor[FuelClass.Dead10h] = this.surfaceArea[FuelClass.Dead10h] / this.deadArea;
            this.areaWeightingFactor[FuelClass.Dead100h] = this.surfaceArea[FuelClass.Dead100h] / this.deadArea;
        }

        const totalArea = this.liveArea + this.deadArea;
        this.liveArea /= totalArea;
        this.deadArea /= totalArea;

        this.fuelDensity = 0.0;
        if (this.fuelDepth > 0.0) {
            this.fuelDensity = this.load[FuelClass.LiveH] + this.load[FuelClass.LiveW] +
                               this.load[FuelClass.Dead1h] + this.load[FuelClass.Dead10h] + this.load[FuelClass.Dead100h];
            this.fuelDensity /= this.fuelDepth;
        }

        const liveSAV = this.areaWeightingFactor[FuelClass.LiveH] * this.SAV[FuelClass.LiveH] +
                         this.areaWeightingFactor[FuelClass.LiveW] * this.SAV[FuelClass.LiveW];
        const deadSAV = this.areaWeightingFactor[FuelClass.Dead1h] * this.SAV[FuelClass.Dead1h] +
                         this.areaWeightingFactor[FuelClass.Dead10h] * this.SAV[FuelClass.Dead10h] +
                         this.areaWeightingFactor[FuelClass.Dead100h] * this.SAV[FuelClass.Dead100h];
        this.fuelSAV = liveSAV * this.liveArea + deadSAV * this.deadArea;

        this.packingRatio = 0.0;
        for (let i = 0; i < NC; i++) {
            if (this.density[i] > 0.0)
                this.packingRatio += this.load[i] / this.density[i];
        }
        if (this.fuelDepth > 0.0)
            this.packingRatio /= this.fuelDepth;

        const liveLoad = this.areaWeightingFactor[FuelClass.LiveH] * this.load[FuelClass.LiveH] *
                              (1.0 - this.totalSilica[FuelClass.LiveH]) +
                          this.areaWeightingFactor[FuelClass.LiveW] * this.load[FuelClass.LiveW] *
                              (1.0 - this.totalSilica[FuelClass.LiveW]);
        const deadLoad = this.areaWeightingFactor[FuelClass.Dead1h] * this.load[FuelClass.Dead1h] *
                              (1.0 - this.totalSilica[FuelClass.Dead1h]) +
                          this.areaWeightingFactor[FuelClass.Dead10h] * this.load[FuelClass.Dead10h] *
                              (1.0 - this.totalSilica[FuelClass.Dead10h]) +
                          this.areaWeightingFactor[FuelClass.Dead100h] * this.load[FuelClass.Dead100h] *
                              (1.0 - this.totalSilica[FuelClass.Dead100h]);

        const liveHC = this.areaWeightingFactor[FuelClass.LiveH] * this.heatContent[FuelClass.LiveH] +
                        this.areaWeightingFactor[FuelClass.LiveW] * this.heatContent[FuelClass.LiveW];
        const deadHC = this.areaWeightingFactor[FuelClass.Dead1h] * this.heatContent[FuelClass.Dead1h] +
                        this.areaWeightingFactor[FuelClass.Dead10h] * this.heatContent[FuelClass.Dead10h] +
                        this.areaWeightingFactor[FuelClass.Dead100h] * this.heatContent[FuelClass.Dead100h];

        const liveEffSilica = this.areaWeightingFactor[FuelClass.LiveH] * this.effectiveSilica[FuelClass.LiveH] +
                              this.areaWeightingFactor[FuelClass.LiveW] * this.effectiveSilica[FuelClass.LiveW];
        const deadEffSilica = this.areaWeightingFactor[FuelClass.Dead1h] * this.effectiveSilica[FuelClass.Dead1h] +
                              this.areaWeightingFactor[FuelClass.Dead10h] * this.effectiveSilica[FuelClass.Dead10h] +
                              this.areaWeightingFactor[FuelClass.Dead100h] * this.effectiveSilica[FuelClass.Dead100h];

        let liveSilicaFactor = 1.0;
        if (liveEffSilica > 0.0)
            liveSilicaFactor = Math.min(1.0, 0.174 / Math.pow(liveEffSilica, 0.19));
        let deadSilicaFactor = 1.0;
        if (deadEffSilica > 0.0)
            deadSilicaFactor = Math.min(1.0, 0.174 / Math.pow(deadEffSilica, 0.19));

        this.liveReactionFactor = liveLoad * liveHC * liveSilicaFactor;
        this.deadReactionFactor = deadLoad * deadHC * deadSilicaFactor;

        this.residenceTime = 384.0 / this.fuelSAV;
        this.propagatingFlux = Math.exp((0.792 + 0.681 * Math.sqrt(this.fuelSAV)) *
                               (this.packingRatio + 0.1)) /
                               (192.0 + 0.2595 * this.fuelSAV);

        const betaOpt = 3.348 / Math.pow(this.fuelSAV, 0.8189);
        const ratio = this.packingRatio / betaOpt;
        const aa = 133.0 / Math.pow(this.fuelSAV, 0.7913);
        const sigma15 = Math.pow(this.fuelSAV, 1.5);
        const maxGamma = sigma15 / (495.0 + 0.0594 * sigma15);
        const gamma = maxGamma * Math.pow(ratio, aa) * Math.exp(aa * (1.0 - ratio));

        this.liveReactionFactor *= gamma;
        this.deadReactionFactor *= gamma;

        this.slopeK = 5.275 * Math.pow(this.packingRatio, -0.3);
        this.windB = 0.02526 * Math.pow(this.fuelSAV, 0.54);
        const c = 7.47 * Math.exp(-0.133 * Math.pow(this.fuelSAV, 0.55));
        const e = 0.715 * Math.exp(-0.000359 * this.fuelSAV);
        this.windK = c * Math.pow(ratio, -e);
        this.windE = Math.pow(ratio, e) / c;

        let flive = 0.0;
        if (this.SAV[FuelClass.LiveH] > 0.0)
            flive += this.load[FuelClass.LiveH] * Math.exp(-500.0 / this.SAV[FuelClass.LiveH]);
        if (this.SAV[FuelClass.LiveW] > 0.0)
            flive += this.load[FuelClass.LiveW] * Math.exp(-500.0 / this.SAV[FuelClass.LiveW]);

        this.liveExtinction = 0.0;
        if (liveLoad > 0.0) {
            this.fineDeadRatio = this.load[FuelClass.Dead1h] * this.effectiveHeatingNumber[FuelClass.Dead1h] +
                                 this.load[FuelClass.Dead10h] * this.effectiveHeatingNumber[FuelClass.Dead10h] +
                                 this.load[FuelClass.Dead100h] * this.effectiveHeatingNumber[FuelClass.Dead100h];
            if (flive > 0.0)
                this.liveExtinction = 2.9 * this.fineDeadRatio / flive;
        } else {
            this.fineDeadRatio = 0.0;
        }

        for (let i = 0; i < NC; i++) this.fuelMoisture[i] = 0.0;
        this.accelerationConstant = 0.115;
    }
}

function parseFuelModels(text) {
    const models = [];
    let loadMultiplier = 1.0, SAVMultiplier = 1.0, depthMultiplier = 1.0, heatContentMultiplier = 1.0;
    const loadConversion = 0.0459136823;

    for (const rawLine of text.split('\n')) {
        const line = rawLine.trim();
        if (!line || line.startsWith('#')) continue;
        if (line === 'ENGLISH') continue;
        if (line === 'METRIC') {
            loadMultiplier = 0.020481614;
            SAVMultiplier = 30.499999954;
            depthMultiplier = 0.032808399;
            heatContentMultiplier = 0.429922614;
            continue;
        }

        const tokens = line.split(/[\s=,\t]+/).filter(t => t.length > 0);
        if (tokens.length < 13) continue;

        const fuelModel = parseInt(tokens[0]);
        const v = tokens.slice(1, 13).map(parseFloat);

        while (models.length <= fuelModel) models.push(new FuelModel());

        models[fuelModel] = new FuelModel(
            fuelModel,
            v[0] * loadMultiplier * loadConversion,
            v[1] * loadMultiplier * loadConversion,
            v[2] * loadMultiplier * loadConversion,
            v[3] * loadMultiplier * loadConversion,
            v[4] * loadMultiplier * loadConversion,
            v[5] * SAVMultiplier,
            v[6] * SAVMultiplier,
            v[7] * SAVMultiplier,
            v[8] * depthMultiplier,
            v[9],
            v[10] * heatContentMultiplier,
            v[11] * heatContentMultiplier
        );
    }
    return models;
}

function parseFuelMoistures(text) {
    const moistures = [];
    for (const rawLine of text.split('\n')) {
        const line = rawLine.trim();
        if (!line || line.startsWith('#')) continue;

        const tokens = line.split(/[\s=,\t]+/).filter(t => t.length > 0);
        if (tokens.length < 6) continue;

        const fuelModel = parseInt(tokens[0]);
        const v = tokens.slice(1, 6).map(t => parseFloat(t) / 100.0);

        while (moistures.length <= fuelModel)
            moistures.push({ fuelModel: 0, dead1h: 0, dead10h: 0, dead100h: 0, liveH: 0, liveW: 0 });

        moistures[fuelModel] = {
            fuelModel: fuelModel,
            dead1h: v[0], dead10h: v[1], dead100h: v[2], liveH: v[3], liveW: v[4]
        };
    }
    return moistures;
}

function parseAscGrid(text) {
    const lines = text.split('\n');
    const meta = {};
    let dataStart = 0;
    for (let i = 0; i < Math.min(lines.length, 10); i++) {
        const parts = lines[i].trim().split(/\s+/);
        if (parts.length === 2 && isNaN(parseFloat(parts[0]))) {
            meta[parts[0].toLowerCase()] = parseFloat(parts[1]);
            dataStart = i + 1;
        } else {
            break;
        }
    }
    const ncols = meta.ncols | 0;
    const nrows = meta.nrows | 0;
    const data = new Int32Array(nrows * ncols);
    let idx = 0;
    for (let i = dataStart; i < lines.length && idx < data.length; i++) {
        const vals = lines[i].trim().split(/\s+/);
        for (let j = 0; j < vals.length && idx < data.length; j++) {
            data[idx++] = parseInt(vals[j]) | 0;
        }
    }
    return { meta, ncols, nrows, data };
}

function parseFireIgnition(text) {
    const lines = text.split('\n');
    const meta = {};
    // 8-line header: key:value format
    for (let i = 0; i < 8 && i < lines.length; i++) {
        const parts = lines[i].split(':');
        if (parts.length >= 2) {
            meta[parts[0].trim()] = parts[1].trim();
        }
    }
    const numrows = parseInt(meta.numrows) || 642;
    const numcols = parseInt(meta.numcols) || 906;
    const notsetfire = parseInt(meta.notsetfire) || 32767;
    const data = new Int32Array(numrows * numcols);
    data.fill(notsetfire);

    let idx = 0;
    for (let i = 8; i < lines.length && idx < data.length; i++) {
        const line = lines[i].trim();
        if (!line) continue;
        const vals = line.split(',');
        for (let j = 0; j < vals.length && idx < data.length; j++) {
            const v = vals[j].trim();
            if (v.length > 0) data[idx++] = parseInt(v) | 0;
        }
    }
    return { meta, numrows, numcols, notsetfire, data };
}
