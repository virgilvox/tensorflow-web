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
  /**
   * Fixed full-scale acceleration per axis the window is divided by, in the data's
   * own units. Set once from the training set (each axis's largest magnitude, with
   * a floor so a globally quiet axis is not amplified), not from each window, so a
   * near-still Idle window stays compressed near the midpoint instead of being
   * stretched to full range and looking like a gesture, while an active axis keeps
   * its full contrast.
   */
  scales: number[];
}

/** The default motion config: 32 steps over three axes, unit scale per axis. */
export const DEFAULT_MOTION_CONFIG: MotionFeatureConfig = { targetLen: 32, axes: 3, scales: [1, 1, 1] };

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
  // Divide each axis by its FIXED per-axis scale and centre at 0.5, rather than a
  // per-window min-max. Zero maps to 0.5, +scale to 1, -scale to 0. A still window
  // stays a flat line near the midpoint; only real motion spreads across the
  // range, so an Idle class is not stretched to look like a gesture, while an
  // active axis keeps its full contrast.
  for (let a = 0; a < useAxes; a++) {
    const s = config.scales[a];
    const scale = s !== undefined && s > 1e-9 ? s : 1;
    const series: number[] = new Array(steps);
    for (let t = 0; t < steps; t++) series[t] = data[t * axes + a] ?? 0;
    const resampled = resampleSeries(series, config.targetLen);
    const base = a * config.targetLen;
    for (let i = 0; i < config.targetLen; i++) {
      out[base + i] = Math.min(1, Math.max(0, 0.5 + resampled[i]! / (2 * scale)));
    }
  }
  return out;
}
