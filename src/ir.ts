/**
 * The internal graph intermediate representation.
 *
 * Converters produce an IRGraph. The serializer consumes one and emits a
 * .tflite. The IR speaks in semantic op kinds and plain numbers; it never
 * imports the generated FlatBuffers schema. That mapping lives only in the
 * serializer, so converters and the quantizer stay decoupled from schema
 * details and remain unit testable without building a flatbuffer.
 */
import type { DType } from './types';

export type FusedActivation = 'NONE' | 'RELU' | 'RELU6' | 'RELU_N1_TO_1' | 'TANH';
export type Padding = 'SAME' | 'VALID';

/**
 * Quantization parameters for a tensor. For per-tensor quantization the arrays
 * hold a single element. For per-channel weights they hold one element per
 * output channel and quantizedDimension names the channel axis.
 */
export interface Quantization {
  scale: number[];
  zeroPoint: number[];
  quantizedDimension: number;
  /** Recorded calibration range, carried for inspection and debugging. */
  min?: number[];
  max?: number[];
}

export interface IRTensor {
  name: string;
  shape: number[];
  dtype: DType;
  /** Raw little endian bytes for a constant tensor. Undefined for activations. */
  data?: Uint8Array;
  quantization?: Quantization;
  isVariable?: boolean;
}

export interface Conv2DOpts {
  padding: Padding;
  strideW: number;
  strideH: number;
  fusedActivation: FusedActivation;
  dilationW: number;
  dilationH: number;
}

export interface DepthwiseConv2DOpts extends Conv2DOpts {
  depthMultiplier: number;
}

export interface FullyConnectedOpts {
  fusedActivation: FusedActivation;
  keepNumDims: boolean;
}

export interface Pool2DOpts {
  padding: Padding;
  strideW: number;
  strideH: number;
  filterW: number;
  filterH: number;
  fusedActivation: FusedActivation;
}

export interface SoftmaxOpts {
  beta: number;
}

export interface ReshapeOpts {
  newShape: number[];
}

export interface ReducerOpts {
  keepDims: boolean;
}

export interface ConcatOpts {
  axis: number;
  fusedActivation: FusedActivation;
}

export interface AddOpts {
  fusedActivation: FusedActivation;
}

/**
 * A single operator. Discriminated on kind so the serializer narrows the
 * options type exactly. Inputs and outputs are tensor indices into the graph.
 * A value of -1 in inputs marks an optional operand that is intentionally
 * absent (for example a convolution with no bias).
 */
export type IROp =
  | { kind: 'CONV_2D'; inputs: number[]; outputs: number[]; options: Conv2DOpts }
  | { kind: 'DEPTHWISE_CONV_2D'; inputs: number[]; outputs: number[]; options: DepthwiseConv2DOpts }
  | { kind: 'FULLY_CONNECTED'; inputs: number[]; outputs: number[]; options: FullyConnectedOpts }
  | { kind: 'MAX_POOL_2D'; inputs: number[]; outputs: number[]; options: Pool2DOpts }
  | { kind: 'AVERAGE_POOL_2D'; inputs: number[]; outputs: number[]; options: Pool2DOpts }
  | { kind: 'SOFTMAX'; inputs: number[]; outputs: number[]; options: SoftmaxOpts }
  | { kind: 'RESHAPE'; inputs: number[]; outputs: number[]; options: ReshapeOpts }
  | { kind: 'MEAN'; inputs: number[]; outputs: number[]; options: ReducerOpts }
  | { kind: 'PAD'; inputs: number[]; outputs: number[] }
  | { kind: 'ADD'; inputs: number[]; outputs: number[]; options: AddOpts }
  | { kind: 'CONCATENATION'; inputs: number[]; outputs: number[]; options: ConcatOpts }
  | { kind: 'LOGISTIC'; inputs: number[]; outputs: number[] }
  | { kind: 'TANH'; inputs: number[]; outputs: number[] }
  | { kind: 'RELU'; inputs: number[]; outputs: number[] }
  | { kind: 'RELU6'; inputs: number[]; outputs: number[] }
  | { kind: 'QUANTIZE'; inputs: number[]; outputs: number[] }
  | { kind: 'DEQUANTIZE'; inputs: number[]; outputs: number[] };

export type OpKind = IROp['kind'];

export interface IRGraph {
  tensors: IRTensor[];
  ops: IROp[];
  inputs: number[];
  outputs: number[];
  description: string;
}

/**
 * Accumulates tensors and operators into an IRGraph. Tensor names are made
 * unique so two layers that happen to produce the same name do not collide,
 * which keeps the emitted model valid and easier to inspect.
 */
export class GraphBuilder {
  private readonly tensors: IRTensor[] = [];
  private readonly ops: IROp[] = [];
  private inputs: number[] = [];
  private outputs: number[] = [];
  private readonly usedNames = new Set<string>();

  constructor(private readonly description = 'tensorflow-web') {}

  private uniqueName(name: string): string {
    if (!this.usedNames.has(name)) {
      this.usedNames.add(name);
      return name;
    }
    let i = 1;
    let candidate = `${name}_${i}`;
    while (this.usedNames.has(candidate)) {
      i += 1;
      candidate = `${name}_${i}`;
    }
    this.usedNames.add(candidate);
    return candidate;
  }

  /** Adds a tensor and returns its index. */
  addTensor(spec: IRTensor): number {
    const tensor: IRTensor = { ...spec, name: this.uniqueName(spec.name) };
    this.tensors.push(tensor);
    return this.tensors.length - 1;
  }

  /** Adds an activation tensor with no constant data. */
  addActivation(name: string, shape: number[], dtype: DType, quantization?: Quantization): number {
    return this.addTensor(quantization ? { name, shape, dtype, quantization } : { name, shape, dtype });
  }

  /** Adds a constant tensor backed by raw little endian bytes. */
  addConst(
    name: string,
    shape: number[],
    dtype: DType,
    data: Uint8Array,
    quantization?: Quantization,
  ): number {
    return this.addTensor(
      quantization ? { name, shape, dtype, data, quantization } : { name, shape, dtype, data },
    );
  }

  addOp(op: IROp): void {
    this.ops.push(op);
  }

  setInputs(indices: number[]): void {
    this.inputs = [...indices];
  }

  setOutputs(indices: number[]): void {
    this.outputs = [...indices];
  }

  getTensor(index: number): IRTensor {
    const tensor = this.tensors[index];
    if (!tensor) throw new RangeError(`No tensor at index ${index}`);
    return tensor;
  }

  get tensorCount(): number {
    return this.tensors.length;
  }

  build(): IRGraph {
    if (this.inputs.length === 0) throw new Error('Graph has no inputs');
    if (this.outputs.length === 0) throw new Error('Graph has no outputs');
    return {
      tensors: this.tensors,
      ops: this.ops,
      inputs: this.inputs,
      outputs: this.outputs,
      description: this.description,
    };
  }
}
