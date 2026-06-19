/**
 * Int8 post training quantization. Transforms a float graph IR plus a
 * calibration into an int8 graph IR: symmetric per channel weights, asymmetric
 * per tensor activations, int32 bias, and float boundaries via QUANTIZE and
 * DEQUANTIZE. See DESIGN.md for the exact math.
 *
 * quantizeGraph is the pure, unit tested core. quantize is the public wrapper
 * that lowers a model first. All arithmetic lives in src/quant/math.ts; this
 * module only walks the graph and threads tensors through.
 */
import type { Calibration, Model, QuantizeOptions, QuantizedModel } from './types';
import type { IRGraph, IROp, IRTensor, OpKind, Quantization } from './ir';
import { GraphBuilder } from './ir';
import { buildFloatGraph } from './convert/build-graph';
import { bytesToF32, i8ToBytes, i32ToBytes } from './dtype';
import {
  quantizeActivation,
  quantizeBias,
  quantizeWeightsPerChannel,
  quantizeWeightsPerTensor,
} from './quant/math';

/** Ops that carry a weight constant at input[1] and a bias constant at input[2]. */
const WEIGHTED_OPS: ReadonlySet<OpKind> = new Set<OpKind>([
  'CONV_2D',
  'DEPTHWISE_CONV_2D',
  'FULLY_CONNECTED',
]);

/**
 * Some int8 TFLite kernels mandate a fixed output scale and zero point and
 * reject anything else at prepare time. The output of these ops must use the
 * value below instead of its calibrated range. These are the documented
 * requirements of the reference kernels: SOFTMAX and LOGISTIC produce values in
 * [0, 1] at scale 1/256 with zero point -128, and TANH produces values in
 * [-1, 1] at scale 1/128 with zero point 0.
 */
const FIXED_OUTPUT_QUANT: Partial<Record<OpKind, { scale: number; zeroPoint: number }>> = {
  SOFTMAX: { scale: 1 / 256, zeroPoint: -128 },
  LOGISTIC: { scale: 1 / 256, zeroPoint: -128 },
  TANH: { scale: 1 / 128, zeroPoint: 0 },
};

/**
 * Ops that do not requantize between input and output: MAX_POOL_2D takes the
 * max of the raw int8 codes, and RESHAPE and PAD move codes unchanged. The
 * TFLite int8 kernels assume the input and output share one scale and zero
 * point, so the output must carry the input's quantization rather than its own
 * calibrated range. AVERAGE_POOL_2D and MEAN are not here: they accumulate and
 * requantize, so an independent output scale is correct for them.
 */
const SCALE_PRESERVING_OPS: ReadonlySet<OpKind> = new Set<OpKind>([
  'MAX_POOL_2D',
  'RESHAPE',
  'PAD',
]);

/**
 * The per channel axis for a weighted op's weight tensor. CONV_2D and
 * FULLY_CONNECTED quantize along axis 0; DEPTHWISE_CONV_2D along axis 3. See
 * DESIGN.md, "The per channel axis depends on the operator".
 */
function weightChannelAxis(kind: OpKind): number {
  return kind === 'DEPTHWISE_CONV_2D' ? 3 : 0;
}

function defaultWeightScheme(options?: QuantizeOptions): 'per-channel' | 'per-tensor' {
  return options?.weights ?? 'per-channel';
}

/** Builds the int8 quantization record for an activation from its range. */
function activationQuant(min: number, max: number): Quantization {
  const { scale, zeroPoint } = quantizeActivation(min, max);
  return { scale: [scale], zeroPoint: [zeroPoint], quantizedDimension: 0, min: [min], max: [max] };
}

/**
 * Quantizes a float graph IR to an int8 graph IR.
 *
 * Weights default to per channel and activations to per tensor. The graph input
 * stays float32 and feeds a QUANTIZE op into an int8 copy; the final int8
 * activation feeds a DEQUANTIZE op into a float32 graph output. Every interior
 * activation becomes int8 using its calibrated range, looked up by tensor name.
 * Weighted ops (CONV_2D, DEPTHWISE_CONV_2D, FULLY_CONNECTED) get per channel
 * int8 weights and int32 bias.
 *
 * @param float the float32 graph produced by buildFloatGraph
 * @param calibration ranges keyed by activation tensor name; the model input is
 *   keyed by 'input'
 * @param options weight and activation schemes; defaults per channel weights
 * @returns a new int8 IRGraph ready for the serializer
 * @throws Error if the graph is not a single input single output linear graph,
 *   if a required calibration range is missing, or if a weighted op is missing
 *   its weight or bias constant
 */
export function quantizeGraph(
  float: IRGraph,
  calibration: Calibration,
  options?: QuantizeOptions,
): IRGraph {
  if (float.inputs.length !== 1) {
    throw new Error(`quantizeGraph supports one graph input, got ${float.inputs.length}`);
  }
  if (float.outputs.length !== 1) {
    throw new Error(`quantizeGraph supports one graph output, got ${float.outputs.length}`);
  }

  const weightScheme = defaultWeightScheme(options);
  const builder = new GraphBuilder(float.description);

  const rangeOf = (name: string) => {
    const r = calibration.ranges[name];
    if (!r) throw new Error(`No calibration range for activation "${name}"`);
    return r;
  };

  // Activations that are the output of a kernel with a fixed output scale.
  // Their quantization is dictated by the kernel, not by calibration.
  const fixedByOld = new Map<number, { scale: number; zeroPoint: number }>();
  for (const op of float.ops) {
    const fixed = FIXED_OUTPUT_QUANT[op.kind];
    if (fixed) for (const out of op.outputs) fixedByOld.set(out, fixed);
  }

  // Maps an old activation tensor index to its new int8 tensor index. Each
  // activation is created once, the first time it is referenced.
  const int8ByOld = new Map<number, number>();

  const makeInt8 = (oldIndex: number, quant: Quantization): number => {
    const existing = int8ByOld.get(oldIndex);
    if (existing !== undefined) return existing;
    const t = tensorAt(float, oldIndex);
    const created = builder.addActivation(t.name, t.shape, 'int8', quant);
    int8ByOld.set(oldIndex, created);
    return created;
  };

  // The quantization an activation gets from its own calibration, unless a
  // kernel fixes its output scale (softmax, logistic, tanh).
  const quantFor = (oldIndex: number): Quantization => {
    const fixed = fixedByOld.get(oldIndex);
    if (fixed) return { scale: [fixed.scale], zeroPoint: [fixed.zeroPoint], quantizedDimension: 0 };
    const r = rangeOf(tensorAt(float, oldIndex).name);
    return activationQuant(r.min, r.max);
  };

  const int8Activation = (oldIndex: number): number => makeInt8(oldIndex, quantFor(oldIndex));

  // Constant operands (reshape target shape, mean reduction axes, pad amounts)
  // are not activations and carry no calibration range. They pass through
  // unchanged, created once and reused if shared.
  const constByOld = new Map<number, number>();
  const passthroughConst = (oldIndex: number): number => {
    const existing = constByOld.get(oldIndex);
    if (existing !== undefined) return existing;
    const t = tensorAt(float, oldIndex);
    if (!t.data) throw new Error(`Expected constant data on operand "${t.name}"`);
    const created = builder.addConst(t.name, t.shape, t.dtype, t.data, t.quantization);
    constByOld.set(oldIndex, created);
    return created;
  };

  // The float graph input. Keep it float32 and make it the graph input.
  const oldInput = float.inputs[0] as number;
  const inputTensor = tensorAt(float, oldInput);
  const floatInput = builder.addActivation(`${inputTensor.name}_float`, inputTensor.shape, 'float32');
  builder.setInputs([floatInput]);

  // QUANTIZE the float input into the int8 input activation the first op reads.
  const int8Input = int8Activation(oldInput);
  builder.addOp({ kind: 'QUANTIZE', inputs: [floatInput], outputs: [int8Input] });

  for (const op of float.ops) {
    if (SCALE_PRESERVING_OPS.has(op.kind)) {
      const inputs = op.inputs.map((i) =>
        i === -1 ? -1 : float.tensors[i]?.data !== undefined ? passthroughConst(i) : int8Activation(i),
      );
      const actInOld = op.inputs.find((i) => i !== -1 && float.tensors[i]?.data === undefined);
      if (actInOld === undefined) throw new Error(`${op.kind} has no activation input`);
      const inQuant = builder.getTensor(int8Activation(actInOld)).quantization;
      if (!inQuant) throw new Error(`${op.kind} input has no quantization`);
      const outputs = op.outputs.map((o) =>
        makeInt8(o, {
          scale: [...inQuant.scale],
          zeroPoint: [...inQuant.zeroPoint],
          quantizedDimension: inQuant.quantizedDimension,
        }),
      );
      builder.addOp({ ...op, inputs, outputs } as IROp);
    } else {
      rewriteOp(op, float, builder, int8Activation, passthroughConst, weightScheme);
    }
  }

  // DEQUANTIZE the final int8 activation into a float32 graph output.
  const oldOutput = float.outputs[0] as number;
  const int8Output = int8Activation(oldOutput);
  const outTensor = tensorAt(float, oldOutput);
  const floatOutput = builder.addActivation(`${outTensor.name}_float`, outTensor.shape, 'float32');
  builder.addOp({ kind: 'DEQUANTIZE', inputs: [int8Output], outputs: [floatOutput] });
  builder.setOutputs([floatOutput]);

  return builder.build();
}

/**
 * Rewrites one float op into its int8 form, appending any quantized constants
 * and the op itself to builder. Activation inputs and outputs are threaded
 * through int8Activation so shared tensors are created exactly once.
 */
function rewriteOp(
  op: IROp,
  float: IRGraph,
  builder: GraphBuilder,
  int8Activation: (oldIndex: number) => number,
  passthroughConst: (oldIndex: number) => number,
  weightScheme: 'per-channel' | 'per-tensor',
): void {
  if (WEIGHTED_OPS.has(op.kind)) {
    rewriteWeightedOp(op, float, builder, int8Activation, weightScheme);
    return;
  }

  // Ops with no weights mix activation operands with constant operands (a
  // reshape's target shape, a mean's axes). Constants pass through unchanged;
  // activations are quantized. -1 marks an absent optional operand.
  const inputs = op.inputs.map((i) => {
    if (i === -1) return -1;
    const t = float.tensors[i];
    return t && t.data !== undefined ? passthroughConst(i) : int8Activation(i);
  });
  const outputs = op.outputs.map((i) => int8Activation(i));
  builder.addOp({ ...op, inputs, outputs } as IROp);
}

/** Rewrites a CONV_2D, DEPTHWISE_CONV_2D, or FULLY_CONNECTED op. */
function rewriteWeightedOp(
  op: IROp,
  float: IRGraph,
  builder: GraphBuilder,
  int8Activation: (oldIndex: number) => number,
  weightScheme: 'per-channel' | 'per-tensor',
): void {
  const actIn = op.inputs[0];
  const weightIn = op.inputs[1];
  const biasIn = op.inputs[2];
  if (actIn === undefined) {
    throw new Error(`${op.kind} is missing its input activation`);
  }
  if (weightIn === undefined || weightIn < 0) {
    throw new Error(`${op.kind} is missing its weight constant`);
  }

  const int8In = int8Activation(actIn);
  const inputScale = builder.getTensor(int8In).quantization!.scale[0] as number;

  const weightTensor = tensorAt(float, weightIn);
  const weightFloats = readFloatConst(weightTensor);

  // FULLY_CONNECTED weights are quantized per tensor. The TFLite int8 fully
  // connected kernel reads a single weight scale; per channel weights are
  // misinterpreted and corrupt the logits. CONV_2D and DEPTHWISE_CONV_2D use
  // per channel quantization along their channel axis.
  const scheme = op.kind === 'FULLY_CONNECTED' ? 'per-tensor' : weightScheme;
  const axis = scheme === 'per-tensor' ? 0 : weightChannelAxis(op.kind);

  let weightInt8: Int8Array;
  let weightScales: number[];
  if (scheme === 'per-tensor') {
    const { q, scale } = quantizeWeightsPerTensor(weightFloats);
    weightInt8 = q;
    weightScales = [scale];
  } else {
    const { q, scales } = quantizeWeightsPerChannel(weightFloats, weightTensor.shape, axis);
    weightInt8 = q;
    weightScales = scales;
  }

  const weightZeros = weightScales.map(() => 0);
  const newWeight = builder.addConst(
    weightTensor.name,
    weightTensor.shape,
    'int8',
    i8ToBytes(weightInt8),
    { scale: weightScales, zeroPoint: weightZeros, quantizedDimension: axis },
  );

  const newInputs: number[] = [int8In, newWeight];

  // Bias is optional. When present it is int32 with scale inputScale * weightScale_c.
  if (biasIn !== undefined && biasIn >= 0) {
    const biasTensor = tensorAt(float, biasIn);
    const biasFloats = readFloatConst(biasTensor);
    // Per tensor weights give one scale; broadcast it across every bias channel.
    const perChannelWeightScales =
      weightScales.length === biasFloats.length
        ? weightScales
        : biasFloats.length > 0 && weightScales.length === 1
          ? new Array<number>(biasFloats.length).fill(weightScales[0] as number)
          : weightScales;
    const { q, scales } = quantizeBias(biasFloats, inputScale, perChannelWeightScales);
    const newBias = builder.addConst(biasTensor.name, biasTensor.shape, 'int32', i32ToBytes(q), {
      scale: scales,
      zeroPoint: scales.map(() => 0),
      quantizedDimension: 0,
    });
    newInputs.push(newBias);
  } else if (biasIn === -1) {
    newInputs.push(-1);
  }

  const outputs = op.outputs.map((i) => int8Activation(i));
  builder.addOp({ ...op, inputs: newInputs, outputs } as IROp);
}

/** Returns the tensor at index, throwing if the index is out of range. */
function tensorAt(graph: IRGraph, index: number): IRTensor {
  const t = graph.tensors[index];
  if (!t) throw new RangeError(`No tensor at index ${index}`);
  return t;
}

/** Reads a float32 constant tensor's bytes into a Float32Array. */
function readFloatConst(tensor: IRTensor): Float32Array {
  if (!tensor.data) {
    throw new Error(`Expected constant data on tensor "${tensor.name}"`);
  }
  return bytesToF32(tensor.data);
}

export function quantize(
  model: Model,
  calibration: Calibration,
  options?: QuantizeOptions,
): QuantizedModel {
  return { graph: quantizeGraph(buildFloatGraph(model), calibration, options), scheme: 'int8' };
}
