import { describe, it, expect } from 'vitest';
import {
  motionToFeatures,
  motionTensorShape,
  DEFAULT_MOTION_CONFIG,
} from '../src/features/motion';

/** Builds an interleaved 3 axis window from per axis series. */
function interleave(x: number[], y: number[], z: number[]): Float32Array {
  const out = new Float32Array(x.length * 3);
  for (let i = 0; i < x.length; i++) {
    out[i * 3] = x[i]!;
    out[i * 3 + 1] = y[i]!;
    out[i * 3 + 2] = z[i]!;
  }
  return out;
}

describe('motionToFeatures', () => {
  it('produces a fixed length vector normalized per axis to 0..1', () => {
    const n = 50;
    const ramp = Array.from({ length: n }, (_, i) => i); // 0..49
    const data = interleave(ramp, ramp, ramp);
    const f = motionToFeatures(data, 3, DEFAULT_MOTION_CONFIG);
    expect(f.length).toBe(DEFAULT_MOTION_CONFIG.targetLen * 3);
    // A monotonic ramp normalizes to 0 at the start and 1 at the end of each axis.
    expect(f[0]).toBeCloseTo(0, 5);
    expect(f[DEFAULT_MOTION_CONFIG.targetLen - 1]).toBeCloseTo(1, 5);
  });

  it('maps a flat axis to zeros (no range to normalize)', () => {
    const n = 20;
    const flat = new Array(n).fill(5);
    const data = interleave(flat, flat, flat);
    const f = motionToFeatures(data, 3, DEFAULT_MOTION_CONFIG);
    expect(Array.from(f).every((v) => v === 0)).toBe(true);
  });

  it('reports the flattened window shape', () => {
    expect(motionTensorShape({ targetLen: 32, axes: 3 })).toEqual([96]);
    expect(motionTensorShape({ targetLen: 16, axes: 2 })).toEqual([32]);
  });
});
