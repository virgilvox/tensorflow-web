/**
 * Pure image feature extraction. Turns a raw captured frame (RGBA pixels at any
 * size) into the normalized tensor data a small CNN trains on. No canvas, no
 * Vue, no library, so every step is unit testable against hand computed values
 * and can run in a worker. The Data stage stores raw frames; this recomputes the
 * features whenever the feature config changes.
 */

/** A raw RGBA frame, the shape getImageData and the dataset payload both use. */
export interface RgbaFrame {
  width: number;
  height: number;
  /** Row major RGBA, four bytes per pixel, length width*height*4. */
  data: Uint8ClampedArray;
}

/** The image preprocessing config, auto chosen and editable from Standard up. */
export interface ImageFeatureConfig {
  /** Output side length in pixels. The tensor is size by size. */
  size: number;
  /** 1 for grayscale, 3 for RGB. */
  channels: 1 | 3;
  /** Scale pixel values from 0..255 to 0..1 when true. */
  normalize: boolean;
}

/** The default image feature config: a small grayscale square, normalized. */
export const DEFAULT_IMAGE_CONFIG: ImageFeatureConfig = {
  size: 48,
  channels: 1,
  normalize: true,
};

/** The tensor shape [height, width, channels] a config produces, batch aside. */
export function imageTensorShape(config: ImageFeatureConfig): [number, number, number] {
  return [config.size, config.size, config.channels];
}

/** Rec. 601 luma weights, the standard grayscale conversion. */
const LUMA_R = 0.299;
const LUMA_G = 0.587;
const LUMA_B = 0.114;

/**
 * Bilinearly resamples an RGBA frame to a new size. Bilinear keeps small targets
 * legible where nearest neighbour would alias the shape away.
 *
 * @returns a new RGBA buffer of length dw*dh*4.
 */
export function resizeRgba(frame: RgbaFrame, dw: number, dh: number): Uint8ClampedArray {
  const { width: sw, height: sh, data } = frame;
  const out = new Uint8ClampedArray(dw * dh * 4);
  // Map destination pixel centers back into source space.
  const xRatio = sw / dw;
  const yRatio = sh / dh;
  for (let y = 0; y < dh; y++) {
    const sy = Math.min(sh - 1, Math.max(0, (y + 0.5) * yRatio - 0.5));
    const y0 = Math.floor(sy);
    const y1 = Math.min(sh - 1, y0 + 1);
    const wy = sy - y0;
    for (let x = 0; x < dw; x++) {
      const sx = Math.min(sw - 1, Math.max(0, (x + 0.5) * xRatio - 0.5));
      const x0 = Math.floor(sx);
      const x1 = Math.min(sw - 1, x0 + 1);
      const wx = sx - x0;
      const o = (y * dw + x) * 4;
      for (let c = 0; c < 4; c++) {
        const p00 = data[(y0 * sw + x0) * 4 + c]!;
        const p10 = data[(y0 * sw + x1) * 4 + c]!;
        const p01 = data[(y1 * sw + x0) * 4 + c]!;
        const p11 = data[(y1 * sw + x1) * 4 + c]!;
        const top = p00 + (p10 - p00) * wx;
        const bottom = p01 + (p11 - p01) * wx;
        out[o + c] = Math.round(top + (bottom - top) * wy);
      }
    }
  }
  return out;
}

/**
 * Extracts the model input from a raw RGBA frame: resize to the configured
 * square, reduce to the configured channel count, and optionally normalize to
 * 0..1. The output is row major [size, size, channels], the layout a tfjs
 * tensor4d expects once a batch dimension is added.
 *
 * @returns a Float32Array of length size*size*channels.
 */
export function imageToFeatures(frame: RgbaFrame, config: ImageFeatureConfig): Float32Array {
  const { size, channels, normalize } = config;
  const resized =
    frame.width === size && frame.height === size ? frame.data : resizeRgba(frame, size, size);
  const out = new Float32Array(size * size * channels);
  const scale = normalize ? 1 / 255 : 1;
  for (let p = 0; p < size * size; p++) {
    const r = resized[p * 4]!;
    const g = resized[p * 4 + 1]!;
    const b = resized[p * 4 + 2]!;
    if (channels === 1) {
      out[p] = (LUMA_R * r + LUMA_G * g + LUMA_B * b) * scale;
    } else {
      out[p * 3] = r * scale;
      out[p * 3 + 1] = g * scale;
      out[p * 3 + 2] = b * scale;
    }
  }
  return out;
}
