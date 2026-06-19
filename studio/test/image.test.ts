import { describe, it, expect } from 'vitest';
import {
  resizeRgba,
  imageToFeatures,
  imageTensorShape,
  type RgbaFrame,
} from '../src/features/image';

/** Builds a flat RGBA frame from per pixel [r,g,b,a] tuples. */
function frame(width: number, height: number, pixels: number[][]): RgbaFrame {
  const data = new Uint8ClampedArray(width * height * 4);
  pixels.forEach((px, i) => data.set(px, i * 4));
  return { width, height, data };
}

describe('resizeRgba', () => {
  it('averages a 2x2 down to 1x1 at the center', () => {
    const src = frame(2, 2, [
      [0, 0, 0, 255],
      [100, 100, 100, 255],
      [100, 100, 100, 255],
      [200, 200, 200, 255],
    ]);
    const out = resizeRgba(src, 1, 1);
    // Bilinear at the single center samples all four corners equally: mean 100.
    expect(out[0]).toBe(100);
    expect(out[3]).toBe(255);
  });

  it('keeps a uniform image uniform when upscaled', () => {
    const src = frame(1, 1, [[50, 60, 70, 255]]);
    const out = resizeRgba(src, 3, 3);
    for (let p = 0; p < 9; p++) {
      expect(out[p * 4]).toBe(50);
      expect(out[p * 4 + 1]).toBe(60);
      expect(out[p * 4 + 2]).toBe(70);
    }
  });
});

describe('imageToFeatures', () => {
  it('produces a normalized grayscale value via luma weights', () => {
    const white = frame(1, 1, [[255, 255, 255, 255]]);
    const f = imageToFeatures(white, { size: 1, channels: 1, normalize: true });
    expect(f.length).toBe(1);
    expect(f[0]).toBeCloseTo(1, 5);
  });

  it('keeps three channels when configured, normalized', () => {
    const px = frame(1, 1, [[255, 0, 0, 255]]);
    const f = imageToFeatures(px, { size: 1, channels: 3, normalize: true });
    expect(Array.from(f)).toEqual([1, 0, 0]);
  });

  it('skips normalization when disabled', () => {
    const px = frame(1, 1, [[128, 128, 128, 255]]);
    const f = imageToFeatures(px, { size: 1, channels: 1, normalize: false });
    // Luma of an equal gray is the gray value itself.
    expect(f[0]).toBeCloseTo(128, 4);
  });

  it('reports the tensor shape a config produces', () => {
    expect(imageTensorShape({ size: 48, channels: 1, normalize: true })).toEqual([48, 48, 1]);
    expect(imageTensorShape({ size: 32, channels: 3, normalize: true })).toEqual([32, 32, 3]);
  });
});
