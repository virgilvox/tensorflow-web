/**
 * Unit tests for the pure int8 quantization math. Every expected value is hand
 * computed from the formulas in DESIGN.md so a regression in the arithmetic
 * fails here, before any graph or flatbuffer is involved.
 */
import { describe, expect, it } from 'vitest';
import {
  quantizeActivation,
  quantizeBias,
  quantizeWeightsPerChannel,
  quantizeWeightsPerTensor,
} from '../src/quant/math';

describe('quantizeActivation', () => {
  it('maps a [0, 1] range with zero point at the bottom', () => {
    // rmin=0, rmax=1, scale=1/255, zeroPoint=round(-128 - 0) = -128.
    const { scale, zeroPoint } = quantizeActivation(0, 1);
    expect(scale).toBeCloseTo(1 / 255, 12);
    expect(zeroPoint).toBe(-128);
  });

  it('maps a [-1, 0] range with zero point at the top', () => {
    // scale=1/255, zeroPoint=round(-128 - (-1)/(1/255)) = round(-128 + 255) = 127.
    const { scale, zeroPoint } = quantizeActivation(-1, 0);
    expect(scale).toBeCloseTo(1 / 255, 12);
    expect(zeroPoint).toBe(127);
  });

  it('maps an asymmetric [-2, 6] range', () => {
    // scale=8/255, zeroPoint=round(-128 - (-2)/(8/255)) = round(-128 + 63.75) = -64.
    const { scale, zeroPoint } = quantizeActivation(-2, 6);
    expect(scale).toBeCloseTo(8 / 255, 12);
    expect(zeroPoint).toBe(-64);
  });

  it('widens a strictly positive range to include zero', () => {
    // min=2, max=6 widens to rmin=0, rmax=6. scale=6/255, zeroPoint=round(-128)= -128.
    const { scale, zeroPoint } = quantizeActivation(2, 6);
    expect(scale).toBeCloseTo(6 / 255, 12);
    expect(zeroPoint).toBe(-128);
  });

  it('handles a degenerate all zero range safely', () => {
    const { scale, zeroPoint } = quantizeActivation(0, 0);
    expect(scale).toBe(1);
    expect(zeroPoint).toBe(0);
  });

  it('clamps the zero point into [-128, 127]', () => {
    const { zeroPoint } = quantizeActivation(0, 1);
    expect(zeroPoint).toBeGreaterThanOrEqual(-128);
    expect(zeroPoint).toBeLessThanOrEqual(127);
  });
});

describe('quantizeWeightsPerChannel', () => {
  it('computes one scale per channel along axis 0', () => {
    // shape [2, 2], axis 0. Channel 0 = [1, -2] maxAbs 2, channel 1 = [4, 4] maxAbs 4.
    const values = new Float32Array([1, -2, 4, 4]);
    const { q, scales } = quantizeWeightsPerChannel(values, [2, 2], 0);
    expect(scales[0]).toBeCloseTo(2 / 127, 12);
    expect(scales[1]).toBeCloseTo(4 / 127, 12);
    // q = round(w / scale_c), clamped to [-127, 127].
    // ch0: round(1/(2/127)) = round(63.5) = 64; round(-2/(2/127)) = -127.
    // ch1: round(4/(4/127)) = 127; round(4/(4/127)) = 127.
    expect(Array.from(q)).toEqual([64, -127, 127, 127]);
  });

  it('uses scale 1 for an all zero channel', () => {
    const values = new Float32Array([0, 0, 3, -3]);
    const { q, scales } = quantizeWeightsPerChannel(values, [2, 2], 0);
    expect(scales[0]).toBe(1);
    expect(scales[1]).toBeCloseTo(3 / 127, 12);
    expect(Array.from(q)).toEqual([0, 0, 127, -127]);
  });

  it('quantizes along axis 3 for depthwise layout', () => {
    // shape [1, 1, 1, 2], axis 3, two channels interleaved as the last axis.
    const values = new Float32Array([2, 10]);
    const { q, scales } = quantizeWeightsPerChannel(values, [1, 1, 1, 2], 3);
    expect(scales[0]).toBeCloseTo(2 / 127, 12);
    expect(scales[1]).toBeCloseTo(10 / 127, 12);
    expect(Array.from(q)).toEqual([127, 127]);
  });

  it('throws on a channelAxis out of bounds', () => {
    expect(() => quantizeWeightsPerChannel(new Float32Array([1]), [1], 1)).toThrow(/out of bounds/);
  });

  it('throws when values length disagrees with shape', () => {
    expect(() => quantizeWeightsPerChannel(new Float32Array([1, 2]), [3], 0)).toThrow(/does not match/);
  });
});

describe('quantizeWeightsPerTensor', () => {
  it('computes a single scale from the global maxAbs', () => {
    const values = new Float32Array([1, -2, 4, -8]);
    const { q, scale } = quantizeWeightsPerTensor(values);
    expect(scale).toBeCloseTo(8 / 127, 12);
    // round(1/(8/127))=round(15.875)=16; round(-2/(8/127))=round(-31.75)=-32;
    // round(4/(8/127))=round(63.5)=64; round(-8/(8/127))=-127.
    expect(Array.from(q)).toEqual([16, -32, 64, -127]);
  });

  it('uses scale 1 when every value is zero', () => {
    const { q, scale } = quantizeWeightsPerTensor(new Float32Array([0, 0, 0]));
    expect(scale).toBe(1);
    expect(Array.from(q)).toEqual([0, 0, 0]);
  });
});

describe('quantizeBias', () => {
  it('quantizes per channel with biasScale = inputScale * weightScale', () => {
    // inputScale 0.5, weightScales [0.1, 0.2]. biasScale = [0.05, 0.1].
    // bias [1, -1]: round(1/0.05)=20; round(-1/0.1)=-10.
    const { q, scales } = quantizeBias(new Float32Array([1, -1]), 0.5, [0.1, 0.2]);
    expect(scales[0]).toBeCloseTo(0.05, 12);
    expect(scales[1]).toBeCloseTo(0.1, 12);
    expect(Array.from(q)).toEqual([20, -10]);
  });

  it('throws on int32 bias overflow rather than silently clamping', () => {
    // A tiny bias scale would push the quantized bias past the int32 range.
    // Clamping there silently corrupts the channel, so the quantizer must fail.
    expect(() => quantizeBias(new Float32Array([1e12]), 1e-9, [1])).toThrow(/overflow/);
  });

  it('quantizes a bias well inside the int32 range without throwing', () => {
    const { q } = quantizeBias(new Float32Array([2]), 1e-9, [1]);
    expect(q[0]).toBe(2_000_000_000);
  });

  it('throws when bias and weight channel counts differ', () => {
    expect(() => quantizeBias(new Float32Array([1, 2]), 0.5, [0.1])).toThrow(/does not match/);
  });
});
