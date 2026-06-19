/**
 * Pure quantization math for int8 post training quantization. Every function
 * here is small, side effect free, and unit tested against hand computed
 * expected values. The graph level quantizer in src/quantize.ts composes these
 * helpers; it owns no arithmetic of its own.
 *
 * Activations are asymmetric int8 (per tensor). Weights are symmetric int8
 * (per channel or per tensor). Bias is symmetric int32 with scale equal to the
 * input scale times the weight scale for that channel. See DESIGN.md.
 */

/** Smallest representable signed 32 bit integer. */
export const INT32_MIN = -2147483648;
/** Largest representable signed 32 bit integer. */
export const INT32_MAX = 2147483647;

/** Asymmetric int8 quantization parameters for one activation tensor. */
export interface ActivationQuant {
  scale: number;
  zeroPoint: number;
}

function clamp(value: number, lo: number, hi: number): number {
  if (value < lo) return lo;
  if (value > hi) return hi;
  return value;
}

/**
 * Computes asymmetric int8 scale and zero point for a calibrated range.
 *
 * The range is widened to include zero so the zero point is representable, then
 * scale = (max - min) / 255 and zeroPoint = clamp(round(-128 - min / scale),
 * -128, 127). A degenerate range where min equals max (including the all zero
 * case) maps to scale 1 and zero point 0, which keeps the math well defined and
 * the boundary QUANTIZE/DEQUANTIZE pair exact.
 *
 * @param min observed minimum of the activation across calibration data
 * @param max observed maximum of the activation across calibration data
 * @returns the int8 scale and zero point for this tensor
 */
export function quantizeActivation(min: number, max: number): ActivationQuant {
  // Widen the range to include zero so the real value 0 is representable.
  let rmin = Math.min(min, 0);
  let rmax = Math.max(max, 0);

  // Degenerate range. After widening this only happens when both ends are 0.
  if (rmin === rmax) {
    return { scale: 1, zeroPoint: 0 };
  }

  const scale = (rmax - rmin) / 255;
  const zeroPoint = clamp(Math.round(-128 - rmin / scale), -128, 127);
  return { scale, zeroPoint };
}

/** Symmetric per channel int8 weights with one scale per channel. */
export interface PerChannelWeights {
  q: Int8Array;
  scales: number[];
}

/**
 * Quantizes weights symmetrically to int8 with one scale per channel along
 * channelAxis. For channel c, scale_c = maxAbs(W_c) / 127 (or 1 when the channel
 * is all zero, to avoid division by zero) and each value is
 * clamp(round(w / scale_c), -127, 127). Zero point is implicitly 0.
 *
 * @param values flattened weight values in row major order for shape
 * @param shape the logical shape of the weight tensor
 * @param channelAxis the axis whose entries each get their own scale
 * @returns the int8 values and one scale per channel
 * @throws RangeError if channelAxis is out of bounds for shape, or if values
 *   does not contain exactly the number of elements shape describes
 */
export function quantizeWeightsPerChannel(
  values: Float32Array,
  shape: number[],
  channelAxis: number,
): PerChannelWeights {
  if (channelAxis < 0 || channelAxis >= shape.length) {
    throw new RangeError(`channelAxis ${channelAxis} out of bounds for rank ${shape.length}`);
  }
  const total = shape.reduce((a, b) => a * b, 1);
  if (values.length !== total) {
    throw new RangeError(`values length ${values.length} does not match shape product ${total}`);
  }

  const channels = shape[channelAxis] as number;
  // Stride of one step along channelAxis in the flattened array: product of the
  // dimensions after channelAxis.
  let inner = 1;
  for (let a = channelAxis + 1; a < shape.length; a++) inner *= shape[a] as number;

  // First pass: maxAbs per channel.
  const maxAbs = new Float64Array(channels);
  for (let i = 0; i < values.length; i++) {
    const c = Math.floor(i / inner) % channels;
    const a = Math.abs(values[i] as number);
    if (a > (maxAbs[c] as number)) maxAbs[c] = a;
  }

  const scales: number[] = new Array(channels);
  for (let c = 0; c < channels; c++) {
    const m = maxAbs[c] as number;
    scales[c] = m === 0 ? 1 : m / 127;
  }

  const q = new Int8Array(values.length);
  for (let i = 0; i < values.length; i++) {
    const c = Math.floor(i / inner) % channels;
    q[i] = clamp(Math.round((values[i] as number) / (scales[c] as number)), -127, 127);
  }

  return { q, scales };
}

/** Symmetric per tensor int8 weights with a single scale. */
export interface PerTensorWeights {
  q: Int8Array;
  scale: number;
}

/**
 * Quantizes weights symmetrically to int8 with a single scale for the whole
 * tensor. scale = maxAbs(W) / 127 (or 1 when every value is zero) and each value
 * is clamp(round(w / scale), -127, 127). Zero point is implicitly 0.
 *
 * @param values flattened weight values
 * @returns the int8 values and the single scale
 */
export function quantizeWeightsPerTensor(values: Float32Array): PerTensorWeights {
  let maxAbs = 0;
  for (let i = 0; i < values.length; i++) {
    const a = Math.abs(values[i] as number);
    if (a > maxAbs) maxAbs = a;
  }
  const scale = maxAbs === 0 ? 1 : maxAbs / 127;

  const q = new Int8Array(values.length);
  for (let i = 0; i < values.length; i++) {
    q[i] = clamp(Math.round((values[i] as number) / scale), -127, 127);
  }
  return { q, scale };
}

/** Symmetric per channel int32 bias with one scale per channel. */
export interface QuantizedBias {
  q: Int32Array;
  scales: number[];
}

/**
 * Quantizes a bias vector to int32, per channel, with zero point 0. The bias
 * scale for channel c is inputScale times weightScales[c], and each value is
 * clamp(round(b / biasScale_c), INT32_MIN, INT32_MAX).
 *
 * @param values the float bias values, one per channel
 * @param inputScale the asymmetric scale of the op's input activation
 * @param weightScales the per channel weight scales for this op
 * @returns the int32 bias values and the per channel bias scales
 * @throws RangeError if values and weightScales have different lengths
 */
export function quantizeBias(
  values: Float32Array,
  inputScale: number,
  weightScales: number[],
): QuantizedBias {
  if (values.length !== weightScales.length) {
    throw new RangeError(
      `bias length ${values.length} does not match weight channel count ${weightScales.length}`,
    );
  }
  const scales: number[] = new Array(values.length);
  const q = new Int32Array(values.length);
  for (let c = 0; c < values.length; c++) {
    const biasScale = inputScale * (weightScales[c] as number);
    scales[c] = biasScale;
    // A zero bias scale only arises if inputScale is 0, which the activation
    // quantizer never produces for a real range. Guard anyway: 0 maps to 0.
    const scaled = biasScale === 0 ? 0 : (values[c] as number) / biasScale;
    const rounded = Math.round(scaled);
    // Clamping a bias is never correct; it silently corrupts the channel. Fail
    // loud instead. This arises when the activation range feeding a layer is
    // very narrow relative to the bias magnitude.
    if (rounded < INT32_MIN || rounded > INT32_MAX) {
      throw new RangeError(
        `int32 bias overflow on channel ${c}: ${values[c]} / ${biasScale} rounds to ${rounded}, ` +
          `outside the int32 range. The activation feeding this layer is too narrow for the bias magnitude.`,
      );
    }
    q[c] = rounded;
  }
  return { q, scales };
}
