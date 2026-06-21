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

const cfg = (scale: number) => ({ ...DEFAULT_MOTION_CONFIG, scales: [scale, scale, scale] });

describe('motionToFeatures', () => {
  it('produces a fixed length vector in 0..1 against a fixed scale', () => {
    const n = 50;
    const ramp = Array.from({ length: n }, (_, i) => i - 25); // -25..24
    const data = interleave(ramp, ramp, ramp);
    const f = motionToFeatures(data, 3, cfg(25));
    expect(f.length).toBe(DEFAULT_MOTION_CONFIG.targetLen * 3);
    for (const v of f) {
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(1);
    }
    // Zero maps to the 0.5 midpoint, -scale to 0; the ramp starts at -scale.
    expect(f[0]).toBeCloseTo(0, 5);
  });

  it('does not stretch a small (idle) window to full range', () => {
    // Regression: per-axis min-max stretched a near-still window to full contrast,
    // so an Idle class looked like a gesture. With a fixed scale a small window
    // stays compressed near the midpoint while a large one spans the range.
    const n = 32;
    const big = Array.from({ length: n }, (_, i) => 10 * Math.sin(i));
    const small = Array.from({ length: n }, (_, i) => 0.1 * Math.sin(i));
    const zero = new Array(n).fill(0);
    const L = DEFAULT_MOTION_CONFIG.targetLen;
    const fBig = motionToFeatures(interleave(big, zero, zero), 3, cfg(10));
    const fSmall = motionToFeatures(interleave(small, zero, zero), 3, cfg(10));
    const spread = (a: Float32Array) => {
      const axis = Array.from(a.slice(0, L));
      return Math.max(...axis) - Math.min(...axis);
    };
    expect(spread(fBig)).toBeGreaterThan(0.5);
    expect(spread(fSmall)).toBeLessThan(0.1);
  });

  it('reports the flattened window shape', () => {
    expect(motionTensorShape({ targetLen: 32, axes: 3, scales: [1, 1, 1] })).toEqual([96]);
    expect(motionTensorShape({ targetLen: 16, axes: 2, scales: [1, 1] })).toEqual([32]);
  });
});
