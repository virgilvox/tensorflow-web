import { describe, it, expect } from 'vitest';
import { encodeBundle, parseBundle } from '../src/lib/modelBundle';
import type { FeatureConfig } from '../src/features';

const imageConfig: FeatureConfig = { kind: 'image', image: { size: 48, channels: 1, normalize: true } };

describe('modelBundle', () => {
  it('round trips a model and its config through encode and parse', () => {
    const bytes = new Uint8Array([0, 1, 2, 250, 251, 255]);
    const json = encodeBundle({
      name: 'gestures',
      modality: 'image',
      classes: ['up', 'down', 'Neither'],
      featureConfig: imageConfig,
      inputShape: [48, 48, 1],
      bytes,
    });
    const parsed = parseBundle(JSON.parse(json));
    expect(parsed.name).toBe('gestures');
    expect(parsed.modality).toBe('image');
    expect(parsed.classes).toEqual(['up', 'down', 'Neither']);
    expect(parsed.inputShape).toEqual([48, 48, 1]);
    expect(Array.from(parsed.bytes)).toEqual([0, 1, 2, 250, 251, 255]);
    expect(parsed.featureConfig).toEqual(imageConfig);
  });

  it('carries the audio clip length when present', () => {
    const json = encodeBundle({
      name: 'words',
      modality: 'audio',
      classes: ['yes', 'no'],
      featureConfig: { kind: 'audio', audio: { sampleRate: 16000, fftSize: 512, hopSize: 256, numMel: 32, mode: 'mel', numCoeffs: 13, targetFrames: 32, clipSeconds: 1 } },
      inputShape: [32, 32, 1],
      audioSeconds: 2,
      bytes: new Uint8Array([9, 9]),
    });
    expect(parseBundle(JSON.parse(json)).audioSeconds).toBe(2);
  });

  it('rejects a file without the format marker', () => {
    expect(() => parseBundle({ version: 1, modality: 'image' })).toThrow(/not a recognized/i);
  });

  it('rejects a feature config that does not match the modality', () => {
    const bad = {
      format: 'tfwebstudio-model',
      version: 1,
      name: 'x',
      modality: 'audio',
      classes: ['a'],
      featureConfig: imageConfig,
      inputShape: [1],
      tfliteBase64: 'AAA=',
    };
    expect(() => parseBundle(bad)).toThrow(/does not match the modality/i);
  });

  it('rejects a text bundle with no vocabulary', () => {
    const bad = {
      format: 'tfwebstudio-model',
      version: 1,
      name: 'x',
      modality: 'text',
      classes: ['a', 'b'],
      featureConfig: { kind: 'text' },
      inputShape: [10],
      tfliteBase64: 'AAA=',
    };
    expect(() => parseBundle(bad)).toThrow(/vocabulary/i);
  });

  it('rejects an empty input shape', () => {
    const bad = {
      format: 'tfwebstudio-model',
      version: 1,
      name: 'x',
      modality: 'image',
      classes: ['a'],
      featureConfig: imageConfig,
      inputShape: [],
      tfliteBase64: 'AAA=',
    };
    expect(() => parseBundle(bad)).toThrow(/non-empty/i);
  });

  it('rejects an unknown modality', () => {
    const bad = {
      format: 'tfwebstudio-model',
      version: 1,
      name: 'x',
      modality: 'video',
      classes: [],
      featureConfig: { kind: 'video' },
      inputShape: [1],
      tfliteBase64: 'AAA=',
    };
    expect(() => parseBundle(bad)).toThrow(/unknown modality/i);
  });
});
