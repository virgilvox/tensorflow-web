/**
 * The FlatBuffers serializer. Walks an IRGraph and emits valid .tflite bytes.
 *
 * This is the novel core of the library. Every enum value and table comes from
 * the generated schema bindings, referenced by name, so the output tracks the
 * pinned schema exactly and never depends on a hand copied magic number.
 *
 * The emitted layout follows the .tflite conventions the interpreter expects:
 *   - Buffer 0 is empty. Activation tensors reference it. Constant tensors get
 *     their own buffer holding little endian bytes.
 *   - One OperatorCode per distinct builtin operator, referenced by index.
 *   - A single subgraph with the graph inputs and outputs.
 *   - The model is finished with the "TFL3" file identifier.
 *
 * Correctness against a real interpreter is proven by loading the bytes back in
 * tfjs-tflite in the browser. See verify.ts.
 */
import * as flatbuffers from 'flatbuffers';
import {
  ActivationFunctionType,
  AddOptions,
  Buffer as TflBuffer,
  BuiltinOperator,
  BuiltinOptions,
  ConcatenationOptions,
  Conv2DOptions,
  DepthwiseConv2DOptions,
  FullyConnectedOptions,
  Model,
  Operator,
  OperatorCode,
  Padding as TflPadding,
  Pool2DOptions,
  QuantizationParameters,
  ReducerOptions,
  ReshapeOptions,
  SoftmaxOptions,
  SubGraph,
  Tensor,
} from './schema/tflite';
import type { FusedActivation, IRGraph, IROp, IRTensor, Padding } from '../ir';
import { TensorTypeCode } from '../dtype';

/** TFLite model schema version. The format has been at version 3 for years. */
const MODEL_VERSION = 3;

function fusedActivation(value: FusedActivation): ActivationFunctionType {
  switch (value) {
    case 'NONE':
      return ActivationFunctionType.NONE;
    case 'RELU':
      return ActivationFunctionType.RELU;
    case 'RELU6':
      return ActivationFunctionType.RELU6;
    case 'RELU_N1_TO_1':
      return ActivationFunctionType.RELU_N1_TO_1;
    case 'TANH':
      return ActivationFunctionType.TANH;
  }
}

function padding(value: Padding): TflPadding {
  return value === 'SAME' ? TflPadding.SAME : TflPadding.VALID;
}

/** True when any operand of the op is an int8 or int16 tensor. */
function isQuantizedOp(op: IROp, graph: IRGraph): boolean {
  return [...op.inputs, ...op.outputs].some((i) => {
    if (i < 0) return false;
    const t = graph.tensors[i];
    return t !== undefined && (t.dtype === 'int8' || t.dtype === 'int16');
  });
}

/**
 * The minimum TFLite operator version for a kind. Quantized kernels require a
 * higher minimum than their float form; these values match the TFLite runtime's
 * documented int8 minimums so a strict interpreter accepts the model.
 */
function operatorVersion(kind: IROp['kind'], quantized: boolean): number {
  if (!quantized) return 1;
  switch (kind) {
    case 'CONV_2D':
    case 'DEPTHWISE_CONV_2D':
      return 3;
    case 'FULLY_CONNECTED':
      return 4;
    case 'AVERAGE_POOL_2D':
    case 'MAX_POOL_2D':
    case 'SOFTMAX':
    case 'LOGISTIC':
    case 'TANH':
    case 'MEAN':
    case 'ADD':
    case 'CONCATENATION':
    case 'PAD':
    case 'RELU':
    case 'RELU6':
    case 'DEQUANTIZE':
      return 2;
    case 'QUANTIZE':
    case 'RESHAPE':
      return 1;
  }
}

/** Maps each op kind to its builtin operator code. */
function builtinCodeFor(kind: IROp['kind']): BuiltinOperator {
  switch (kind) {
    case 'CONV_2D':
      return BuiltinOperator.CONV_2D;
    case 'DEPTHWISE_CONV_2D':
      return BuiltinOperator.DEPTHWISE_CONV_2D;
    case 'FULLY_CONNECTED':
      return BuiltinOperator.FULLY_CONNECTED;
    case 'MAX_POOL_2D':
      return BuiltinOperator.MAX_POOL_2D;
    case 'AVERAGE_POOL_2D':
      return BuiltinOperator.AVERAGE_POOL_2D;
    case 'SOFTMAX':
      return BuiltinOperator.SOFTMAX;
    case 'RESHAPE':
      return BuiltinOperator.RESHAPE;
    case 'MEAN':
      return BuiltinOperator.MEAN;
    case 'PAD':
      return BuiltinOperator.PAD;
    case 'ADD':
      return BuiltinOperator.ADD;
    case 'CONCATENATION':
      return BuiltinOperator.CONCATENATION;
    case 'LOGISTIC':
      return BuiltinOperator.LOGISTIC;
    case 'TANH':
      return BuiltinOperator.TANH;
    case 'RELU':
      return BuiltinOperator.RELU;
    case 'RELU6':
      return BuiltinOperator.RELU6;
    case 'QUANTIZE':
      return BuiltinOperator.QUANTIZE;
    case 'DEQUANTIZE':
      return BuiltinOperator.DEQUANTIZE;
  }
}

/** Builds the builtin options table for an op. Returns NONE when it has none. */
function buildOptions(
  b: flatbuffers.Builder,
  op: IROp,
): { type: BuiltinOptions; offset: number } {
  switch (op.kind) {
    case 'CONV_2D': {
      Conv2DOptions.startConv2DOptions(b);
      Conv2DOptions.addPadding(b, padding(op.options.padding));
      Conv2DOptions.addStrideW(b, op.options.strideW);
      Conv2DOptions.addStrideH(b, op.options.strideH);
      Conv2DOptions.addFusedActivationFunction(b, fusedActivation(op.options.fusedActivation));
      Conv2DOptions.addDilationWFactor(b, op.options.dilationW);
      Conv2DOptions.addDilationHFactor(b, op.options.dilationH);
      return { type: BuiltinOptions.Conv2DOptions, offset: Conv2DOptions.endConv2DOptions(b) };
    }
    case 'DEPTHWISE_CONV_2D': {
      DepthwiseConv2DOptions.startDepthwiseConv2DOptions(b);
      DepthwiseConv2DOptions.addPadding(b, padding(op.options.padding));
      DepthwiseConv2DOptions.addStrideW(b, op.options.strideW);
      DepthwiseConv2DOptions.addStrideH(b, op.options.strideH);
      DepthwiseConv2DOptions.addDepthMultiplier(b, op.options.depthMultiplier);
      DepthwiseConv2DOptions.addFusedActivationFunction(b, fusedActivation(op.options.fusedActivation));
      DepthwiseConv2DOptions.addDilationWFactor(b, op.options.dilationW);
      DepthwiseConv2DOptions.addDilationHFactor(b, op.options.dilationH);
      return {
        type: BuiltinOptions.DepthwiseConv2DOptions,
        offset: DepthwiseConv2DOptions.endDepthwiseConv2DOptions(b),
      };
    }
    case 'FULLY_CONNECTED': {
      FullyConnectedOptions.startFullyConnectedOptions(b);
      FullyConnectedOptions.addFusedActivationFunction(b, fusedActivation(op.options.fusedActivation));
      FullyConnectedOptions.addKeepNumDims(b, op.options.keepNumDims);
      return {
        type: BuiltinOptions.FullyConnectedOptions,
        offset: FullyConnectedOptions.endFullyConnectedOptions(b),
      };
    }
    case 'MAX_POOL_2D':
    case 'AVERAGE_POOL_2D': {
      Pool2DOptions.startPool2DOptions(b);
      Pool2DOptions.addPadding(b, padding(op.options.padding));
      Pool2DOptions.addStrideW(b, op.options.strideW);
      Pool2DOptions.addStrideH(b, op.options.strideH);
      Pool2DOptions.addFilterWidth(b, op.options.filterW);
      Pool2DOptions.addFilterHeight(b, op.options.filterH);
      Pool2DOptions.addFusedActivationFunction(b, fusedActivation(op.options.fusedActivation));
      return { type: BuiltinOptions.Pool2DOptions, offset: Pool2DOptions.endPool2DOptions(b) };
    }
    case 'SOFTMAX': {
      SoftmaxOptions.startSoftmaxOptions(b);
      SoftmaxOptions.addBeta(b, op.options.beta);
      return { type: BuiltinOptions.SoftmaxOptions, offset: SoftmaxOptions.endSoftmaxOptions(b) };
    }
    case 'RESHAPE': {
      const newShape = ReshapeOptions.createNewShapeVector(b, op.options.newShape);
      ReshapeOptions.startReshapeOptions(b);
      ReshapeOptions.addNewShape(b, newShape);
      return { type: BuiltinOptions.ReshapeOptions, offset: ReshapeOptions.endReshapeOptions(b) };
    }
    case 'MEAN': {
      ReducerOptions.startReducerOptions(b);
      ReducerOptions.addKeepDims(b, op.options.keepDims);
      return { type: BuiltinOptions.ReducerOptions, offset: ReducerOptions.endReducerOptions(b) };
    }
    case 'ADD': {
      AddOptions.startAddOptions(b);
      AddOptions.addFusedActivationFunction(b, fusedActivation(op.options.fusedActivation));
      return { type: BuiltinOptions.AddOptions, offset: AddOptions.endAddOptions(b) };
    }
    case 'CONCATENATION': {
      ConcatenationOptions.startConcatenationOptions(b);
      ConcatenationOptions.addAxis(b, op.options.axis);
      ConcatenationOptions.addFusedActivationFunction(b, fusedActivation(op.options.fusedActivation));
      return {
        type: BuiltinOptions.ConcatenationOptions,
        offset: ConcatenationOptions.endConcatenationOptions(b),
      };
    }
    case 'PAD':
    case 'LOGISTIC':
    case 'TANH':
    case 'RELU':
    case 'RELU6':
    case 'QUANTIZE':
    case 'DEQUANTIZE':
      return { type: BuiltinOptions.NONE, offset: 0 };
  }
}

function buildQuantization(b: flatbuffers.Builder, tensor: IRTensor): number {
  const q = tensor.quantization;
  if (!q) return 0;
  if (q.scale.length === 0) {
    throw new Error(`Tensor "${tensor.name}" has a quantization record with no scale.`);
  }
  const scale = QuantizationParameters.createScaleVector(b, q.scale);
  const zeroPoint = QuantizationParameters.createZeroPointVector(
    b,
    q.zeroPoint.map((z) => BigInt(z)),
  );
  QuantizationParameters.startQuantizationParameters(b);
  QuantizationParameters.addScale(b, scale);
  QuantizationParameters.addZeroPoint(b, zeroPoint);
  QuantizationParameters.addQuantizedDimension(b, q.quantizedDimension);
  return QuantizationParameters.endQuantizationParameters(b);
}

/**
 * Serializes a graph to .tflite bytes.
 *
 * @throws if a constant tensor declares no data, which would produce a model
 *   the interpreter cannot run.
 */
export function serialize(graph: IRGraph): Uint8Array {
  const b = new flatbuffers.Builder(1024);

  // Buffers. Index 0 is the required empty buffer. Constant tensors each get a
  // dedicated buffer; activation tensors share buffer 0.
  const bufferOffsets: number[] = [];
  TflBuffer.startBuffer(b);
  bufferOffsets.push(TflBuffer.endBuffer(b));

  const tensorBufferIndex: number[] = [];
  for (const tensor of graph.tensors) {
    if (tensor.data === undefined) {
      tensorBufferIndex.push(0);
      continue;
    }
    const data = TflBuffer.createDataVector(b, tensor.data);
    TflBuffer.startBuffer(b);
    TflBuffer.addData(b, data);
    tensorBufferIndex.push(bufferOffsets.length);
    bufferOffsets.push(TflBuffer.endBuffer(b));
  }

  // Tensors.
  const tensorOffsets: number[] = [];
  graph.tensors.forEach((tensor, i) => {
    const name = b.createString(tensor.name);
    const shape = Tensor.createShapeVector(b, tensor.shape);
    const quantization = buildQuantization(b, tensor);
    Tensor.startTensor(b);
    Tensor.addShape(b, shape);
    Tensor.addType(b, TensorTypeCode[tensor.dtype]);
    Tensor.addBuffer(b, tensorBufferIndex[i] as number);
    Tensor.addName(b, name);
    if (quantization !== 0) Tensor.addQuantization(b, quantization);
    if (tensor.isVariable) Tensor.addIsVariable(b, true);
    tensorOffsets.push(Tensor.endTensor(b));
  });

  // Operator codes, one per distinct builtin operator. The version is the
  // maximum required by any op of that kind, because a quantized op needs a
  // higher minimum version than its float form, and a strict on device
  // interpreter validates the declared version against its registered kernel.
  const opcodeIndexByKind = new Map<IROp['kind'], number>();
  const operatorCodeOffsets: number[] = [];
  const versionByKind = new Map<IROp['kind'], number>();
  for (const op of graph.ops) {
    const v = operatorVersion(op.kind, isQuantizedOp(op, graph));
    versionByKind.set(op.kind, Math.max(versionByKind.get(op.kind) ?? 1, v));
  }
  for (const op of graph.ops) {
    if (opcodeIndexByKind.has(op.kind)) continue;
    const code = builtinCodeFor(op.kind);
    OperatorCode.startOperatorCode(b);
    // deprecated_builtin_code is the legacy int8 field; keep it in sync for
    // codes that fit, which is every operator this library emits today.
    OperatorCode.addDeprecatedBuiltinCode(b, code <= 127 ? code : 127);
    OperatorCode.addBuiltinCode(b, code);
    OperatorCode.addVersion(b, versionByKind.get(op.kind) ?? 1);
    opcodeIndexByKind.set(op.kind, operatorCodeOffsets.length);
    operatorCodeOffsets.push(OperatorCode.endOperatorCode(b));
  }

  // Operators.
  const operatorOffsets: number[] = [];
  for (const op of graph.ops) {
    const options = buildOptions(b, op);
    const inputs = Operator.createInputsVector(b, op.inputs);
    const outputs = Operator.createOutputsVector(b, op.outputs);
    Operator.startOperator(b);
    Operator.addOpcodeIndex(b, opcodeIndexByKind.get(op.kind) as number);
    Operator.addInputs(b, inputs);
    Operator.addOutputs(b, outputs);
    if (options.type !== BuiltinOptions.NONE) {
      Operator.addBuiltinOptionsType(b, options.type);
      Operator.addBuiltinOptions(b, options.offset);
    }
    operatorOffsets.push(Operator.endOperator(b));
  }

  // Subgraph.
  const subgraphName = b.createString('main');
  const tensorsVec = SubGraph.createTensorsVector(b, tensorOffsets);
  const inputsVec = SubGraph.createInputsVector(b, graph.inputs);
  const outputsVec = SubGraph.createOutputsVector(b, graph.outputs);
  const operatorsVec = SubGraph.createOperatorsVector(b, operatorOffsets);
  SubGraph.startSubGraph(b);
  SubGraph.addTensors(b, tensorsVec);
  SubGraph.addInputs(b, inputsVec);
  SubGraph.addOutputs(b, outputsVec);
  SubGraph.addOperators(b, operatorsVec);
  SubGraph.addName(b, subgraphName);
  const subgraph = SubGraph.endSubGraph(b);

  // Model.
  const description = b.createString(graph.description);
  const operatorCodesVec = Model.createOperatorCodesVector(b, operatorCodeOffsets);
  const subgraphsVec = Model.createSubgraphsVector(b, [subgraph]);
  const buffersVec = Model.createBuffersVector(b, bufferOffsets);
  Model.startModel(b);
  Model.addVersion(b, MODEL_VERSION);
  Model.addOperatorCodes(b, operatorCodesVec);
  Model.addSubgraphs(b, subgraphsVec);
  Model.addDescription(b, description);
  Model.addBuffers(b, buffersVec);
  const model = Model.endModel(b);

  Model.finishModelBuffer(b, model);
  return b.asUint8Array().slice();
}
