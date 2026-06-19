/**
 * Pure motion feature extraction. Turns a windowed accelerometer signal into a
 * fixed length vector a small MLP trains on: each axis is resampled to a fixed
 * number of steps and normalized to 0..1, then the axes are concatenated. This
 * keeps the gesture's shape across time while bounding the range for a clean int8
 * input boundary. No Vue, no browser.
 */

/** The motion preprocessing config. */
export interface MotionFeatureConfig {
  /** Resampled time steps per axis. */
  targetLen: number;
  /** Number of axes captured (3 for accelerometer x, y, z). */
  axes: number;
}

/** The default motion config: 32 steps over three axes. */
export const DEFAULT_MOTION_CONFIG: MotionFeatureConfig = { targetLen: 32, axes: 3 };

/** The tensor shape [targetLen * axes] a config produces, batch aside. */
export function motionTensorShape(config: MotionFeatureConfig): [number] {
  return [config.targetLen * config.axes];
}

/** Linearly resamples a series to a fixed length. */
function resampleSeries(series: number[], targetLen: number): Float32Array {
  const out = new Float32Array(targetLen);
  if (series.length === 0) return out;
  if (series.length === 1) {
    out.fill(series[0]!);
    return out;
  }
  for (let i = 0; i < targetLen; i++) {
    const src = (i / (targetLen - 1)) * (series.length - 1);
    const i0 = Math.floor(src);
    const i1 = Math.min(series.length - 1, i0 + 1);
    const w = src - i0;
    out[i] = series[i0]! * (1 - w) + series[i1]! * w;
  }
  return out;
}

/**
 * Extracts the model input from an interleaved motion window. The window is
 * laid out [t0a0, t0a1, t0a2, t1a0, ...] with `axes` channels per time step.
 * Each axis is deinterleaved, resampled to the configured length, and min max
 * normalized to 0..1.
 *
 * @param data interleaved axis samples.
 * @param axes number of axes in the data.
 * @param config the motion feature config.
 * @returns a Float32Array of length config.targetLen * config.axes.
 */
export function motionToFeatures(
  data: Float32Array,
  axes: number,
  config: MotionFeatureConfig,
): Float32Array {
  const steps = Math.floor(data.length / axes);
  const out = new Float32Array(config.targetLen * config.axes);
  const useAxes = Math.min(axes, config.axes);
  for (let a = 0; a < useAxes; a++) {
    const series: number[] = new Array(steps);
    for (let t = 0; t < steps; t++) series[t] = data[t * axes + a] ?? 0;
    const resampled = resampleSeries(series, config.targetLen);
    let min = Infinity;
    let max = -Infinity;
    for (const v of resampled) {
      if (v < min) min = v;
      if (v > max) max = v;
    }
    const range = max - min;
    const base = a * config.targetLen;
    for (let i = 0; i < config.targetLen; i++) {
      out[base + i] = range > 1e-9 ? (resampled[i]! - min) / range : 0;
    }
  }
  return out;
}
