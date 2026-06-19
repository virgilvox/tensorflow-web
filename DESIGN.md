# Design

This document is the internal contract between modules. It defines the data flow
and the exact interfaces so each module can be built and tested on its own and
still compose into one pipeline. Read it before implementing any module.

## Data flow

```
tf.LayersModel
   |  train()            optional, thin wrapper over model.fit
   v
tf.LayersModel (trained)
   |  buildGraph()       walk layers, dispatch to op converters
   v
IRGraph (float32)        weights as float bytes, activations named by layer
   |  calibrate()        run float model over representative data, record ranges
   v
Calibration              { ranges: name -> {min,max} }
   |  quantize()         float IRGraph + Calibration -> int8 IRGraph
   v
IRGraph (int8)           per-channel int8 weights, int32 bias, QUANTIZE/DEQUANTIZE at the boundary
   |  serialize()        already built and tested
   v
Uint8Array (.tflite)
   |  verify()           load in tfjs-tflite (browser), compare to float reference
   v
VerifyReport
```

`convert()` chains buildGraph, calibrate, quantize, serialize, and verify into
one call.

## The IR (already built, see src/ir.ts)

`GraphBuilder` accumulates `IRTensor` and `IROp` values. A tensor is either an
activation (no `data`) or a constant (raw little endian `data`, from src/dtype.ts
helpers). Ops are a discriminated union keyed by `kind`. The serializer is the
only code that touches the generated schema.

## buildGraph (src/convert/build-graph.ts)

Walks `model.layers` in order. The library supports Sequential models and linear
functional models in v1 (one input, one output, no branching). Branching fails
loudly with a clear message.

For each layer:
1. Look up a converter in the registry by `layer.getClassName()`.
2. If none exists, throw `UnsupportedLayerError` naming the class. Never emit a
   partial or guessed op.
3. Call the converter with the input tensor indices and a `ConvertContext`.
4. The converter appends constants and one op, and returns the output tensor
   index or indices.

Tensor naming: the input activation is named `input`. Each layer's output
activation is named exactly `layer.name`. TensorFlow.js guarantees unique layer
names, so these names are stable keys that calibrate and quantize rely on. Do not
rename them.

Output shapes come from `layer.outputShape` with a null batch dimension replaced
by 1.

## Converter interface (src/ops/types.ts)

```ts
export interface ConvertContext {
  builder: GraphBuilder;
  /** Adds a float32 constant from a tfjs tensor, returns its IR index. */
  addFloatConst(name: string, tensor: tf.Tensor): number;
  /** Adds a constant int32 vector (for shape, axis, paddings inputs). */
  addInt32Const(name: string, values: number[], shape?: number[]): number;
}

export interface LayerConverter {
  readonly layerClass: string;
  convert(layer: tf.layers.Layer, inputs: number[], ctx: ConvertContext): number[];
}
```

Converters read configuration with `layer.getConfig()` and weights with
`layer.getWeights()`. They never read calibration or quantization data; the float
graph carries float weights only. Quantization happens later.

### Weight layout, the part that must be exactly right

TensorFlow.js stores weights in TensorFlow layout. TFLite expects its own layout.
The converter transposes:

- Conv2D kernel: tfjs `[kh, kw, inC, outC]` to TFLite `[outC, kh, kw, inC]`
  (OHWI). Bias `[outC]` unchanged. Output activation `[1, oh, ow, outC]`.
- DepthwiseConv2D kernel: tfjs `[kh, kw, inC, mult]` to TFLite
  `[1, kh, kw, inC * mult]`. Set `depthMultiplier = mult`. Output channels are
  `inC * mult`.
- Dense kernel: tfjs `[inDim, units]` to TFLite `[units, inDim]`. Bias `[units]`.

Each transpose has a focused unit test that checks the permutation on a small
known tensor. Permutation correctness is verifiable without any flatbuffer.

### Fused activation

If a Conv2D, DepthwiseConv2D, or Dense layer config has `activation` of `relu`,
`relu6`, or linear/none, fold it into the op's `fusedActivation`. `softmax`,
`sigmoid`, `tanh`, and any unfused activation become their own op (`SOFTMAX`,
`LOGISTIC`, `TANH`). A standalone `Activation` or `ReLU` layer becomes its own op.

### Layers to support in v1

Conv2D, DepthwiseConv2D, Dense, MaxPooling2D, AveragePooling2D,
GlobalAveragePooling2D (emit `MEAN` over the spatial axes with a const axis input
and `keepDims = false`), Flatten and Reshape (emit `RESHAPE` with a const int32
shape input and the `newShape` option), Softmax, Activation, ReLU, Add.

## calibrate (src/calibrate.ts)

Builds a tfjs model that outputs every layer's activation, runs it over the
representative data, and records per-tensor `{min, max}` keyed by `layer.name`,
plus the model input keyed by `input`. v1 implements `minmax`. The range is
widened to include zero so the zero point is representable.

## quantize (src/quantize.ts)

Transforms a float `IRGraph` plus a `Calibration` into an int8 `IRGraph`.

Activation quantization is asymmetric int8, per tensor. For a calibrated range
`[rmin, rmax]` widened to include 0:

```
scale = (rmax - rmin) / 255
zeroPoint = clamp(round(-128 - rmin / scale), -128, 127)
```

Weight quantization is symmetric int8. Per channel (default) computes one scale
per output channel along `quantizedDimension = 0`:

```
scale_c = maxAbs(W_c) / 127          // if maxAbs is 0, use 1 to avoid div by zero
q = clamp(round(w / scale_c), -127, 127)
zeroPoint_c = 0
```

Per tensor weight quantization uses a single scale `maxAbs(W) / 127`.

The per channel axis depends on the operator, because the weight layouts differ:

- CONV_2D weights are `[outC, kh, kw, inC]`, channel axis 0, `quantizedDimension = 0`.
- FULLY_CONNECTED weights are `[units, inDim]`, channel axis 0, `quantizedDimension = 0`.
- DEPTHWISE_CONV_2D weights are `[1, kh, kw, outC]`, channel axis 3,
  `quantizedDimension = 3`.

The bias length and the weight channel count must match, and the bias scale for
channel c uses that same channel's weight scale.

Two non obvious kernel requirements the quantizer must honor:

- FULLY_CONNECTED weights are quantized per tensor, not per channel. The TFLite
  int8 fully connected kernel reads a single weight scale; per channel scales are
  misread and corrupt the logits. CONV_2D and DEPTHWISE_CONV_2D are per channel.
- MAX_POOL_2D, RESHAPE, and PAD do not requantize between input and output, so
  the output activation must carry the input's exact scale and zero point rather
  than its own calibrated range. AVERAGE_POOL_2D and MEAN do requantize, so an
  independent output scale is correct for them.
- SOFTMAX and LOGISTIC outputs are fixed at scale 1/256, zero point -128; TANH at
  scale 1/128, zero point 0. The kernels reject any other output scale.

Operator versions are not all 1. A quantized kernel needs a higher minimum
version than its float form (for example int8 CONV_2D is version 3, int8
FULLY_CONNECTED is version 4), and a strict on device interpreter validates the
declared version. The serializer computes the version per op from its kind and
whether its operands are int8.

Bias is int32, per channel, with zero point 0:

```
biasScale_c = inputScale * weightScale_c
qBias_c = clamp(round(bias_c / biasScale_c), INT32_MIN, INT32_MAX)
```

The graph boundary stays float for ease of use and for parity checking against
the float reference: insert a `QUANTIZE` op from the float input activation to an
int8 tensor, and a `DEQUANTIZE` op from the final int8 tensor to a float output.
Every interior activation becomes int8 with the scale and zero point from
calibration.

All of this math is pure and unit tested with hand computed expected values
before any model is built.

## train (src/train.ts)

A thin wrapper over `model.fit` that accepts `{xs, ys}` or a `tf.data.Dataset`,
forwards `onEpoch` and `onBatch`, supports an `AbortSignal`, and returns
`{ model, history }`. It does not capture calibration data; calibrate is its own
explicit call.

## verify (src/verify.ts)

The trust layer. It loads the emitted `.tflite` in `@tensorflow/tfjs-tflite`,
runs it on the test inputs, and compares against the float reference model. It
reports `maxAbsError`, the parity boolean within a tolerance, and, when labels
exist, float and int8 accuracy and a confusion matrix.

`@tensorflow/tfjs-tflite` is a browser WASM build. verify therefore runs in the
browser (the test app), not in Node. The module loads the interpreter through a
dynamic import so Node unit tests can import the rest of the library without
pulling in WASM. A separate Node side structural check (parse the bytes, confirm
tensors and ops are well formed) lives next to it and is unit tested.

## emitC (src/emitC.ts), optional

Exports weights, scales, and a C array for a bare metal runtime. Off the main
line. Build last, behind the same op registry.
