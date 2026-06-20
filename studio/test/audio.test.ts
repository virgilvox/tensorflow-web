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
