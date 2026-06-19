import { describe, it, expect } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import { UnsupportedLayerError } from 'tensorflow-web';
import { buildModel, assertExportable, summarize, estimateArenaBytes } from '../src/models/builder';
import { imageCnnPreset, mlpPreset } from '../src/models/presets';

describe('buildModel', () => {
  it('builds an exportable CNN from the image preset', () => {
    const model = buildModel(imageCnnPreset(3, 48), [48, 48, 1]);
    try {
      // The preset uses only supported layers, so the export guard passes.
      expect(() => assertExportable(model)).not.toThrow();
      const summary = summarize(model);
      expect(summary.paramCount).toBeGreaterThan(0);
      expect(summary.estimatedWeightBytes).toBeGreaterThan(0);
      expect(summary.layerCount).toBeGreaterThan(3);
      expect(estimateArenaBytes(model)).toBeGreaterThan(0);
      // The output layer is sized to the class count.
      const out = model.outputs[0]!.shape;
      expect(out[out.length - 1]).toBe(3);
    } finally {
      model.dispose();
    }
  });

  it('builds an exportable MLP from the mlp preset', () => {
    const model = buildModel(mlpPreset(4), [64]);
    try {
      expect(() => assertExportable(model)).not.toThrow();
      expect(model.outputs[0]!.shape.at(-1)).toBe(4);
    } finally {
      model.dispose();
    }
  });

  it('rejects an empty spec', () => {
    expect(() => buildModel({ layers: [] }, [8])).toThrow();
  });
});

describe('assertExportable', () => {
  it('throws UnsupportedLayerError naming a layer with no converter', () => {
    const model = tf.sequential();
    model.add(tf.layers.dense({ inputShape: [4], units: 4 }));
    // Batch normalization has no converter in the library, so export must refuse.
    model.add(tf.layers.batchNormalization());
    try {
      expect(() => assertExportable(model)).toThrow(UnsupportedLayerError);
      expect(() => assertExportable(model)).toThrow(/BatchNormalization/);
    } finally {
      model.dispose();
    }
  });
});
