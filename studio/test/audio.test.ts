import { describe, it, expect } from 'vitest';
import {
  audioToFeatures,
  audioTensorShape,
  DEFAULT_AUDIO_CONFIG,
} from '../src/features/audio';

/** A pure sine clip. */
function sine(freq: number, n: number, rate: number): Float32Array {
  const x = new Float32Array(n);
  for (let i = 0; i < n; i++) x[i] = Math.sin((2 * Math.PI * freq * i) / rate);
  return x;
}

describe('audioToFeatures', () => {
  it('produces a fixed size grid normalized to 0..1', () => {
    const clip = sine(1000, 16000, 16000);
    const f = audioToFeatures(clip, 16000, DEFAULT_AUDIO_CONFIG);
    expect(f.length).toBe(32 * 32);
    let min = Infinity;
    let max = -Infinity;
    for (const v of f) {
      min = Math.min(min, v);
      max = Math.max(max, v);
    }
    expect(min).toBeGreaterThanOrEqual(0);
    expect(max).toBeLessThanOrEqual(1);
    expect(max).toBeGreaterThan(0.5); // a tone has real structure, not a flat grid
  });

  it('resamples a clip captured at a different rate to a fixed grid', () => {
    const clip = sine(1000, 48000, 48000);
    const f = audioToFeatures(clip, 48000, DEFAULT_AUDIO_CONFIG);
    expect(f.length).toBe(32 * 32);
  });

  it('keeps a quiet clip distinct from a loud one, not stretched to full scale', () => {
    // Regression: per-clip min-max normalization made a quiet clip look as loud as
    // a full scale one, so a keyword model predicted the keyword for near-silence.
    // The same tone at 1/100th amplitude must map clearly lower, not to full range.
    const loud = sine(1000, 16000, 16000);
    const quiet = new Float32Array(16000);
    for (let i = 0; i < quiet.length; i++) quiet[i] = 0.01 * Math.sin((2 * Math.PI * 1000 * i) / 16000);
    const fLoud = audioToFeatures(loud, 16000, DEFAULT_AUDIO_CONFIG);
    const fQuiet = audioToFeatures(quiet, 16000, DEFAULT_AUDIO_CONFIG);
    const maxLoud = Math.max(...fLoud);
    const maxQuiet = Math.max(...fQuiet);
    expect(maxLoud).toBeGreaterThan(0.7);
    expect(maxQuiet).toBeLessThan(maxLoud - 0.15);
  });

  it('maps a silent clip to all zeros, not raw log domain values', () => {
    const silence = new Float32Array(16000);
    const f = audioToFeatures(silence, 16000, DEFAULT_AUDIO_CONFIG);
    expect(f.length).toBe(32 * 32);
    expect(Array.from(f).every((v) => v === 0)).toBe(true);
  });

  it('reports the single channel tensor shape', () => {
    expect(audioTensorShape(DEFAULT_AUDIO_CONFIG)).toEqual([32, 32, 1]);
    expect(audioTensorShape({ ...DEFAULT_AUDIO_CONFIG, mode: 'mfcc', numCoeffs: 13 })).toEqual([
      32, 13, 1,
    ]);
  });
});
