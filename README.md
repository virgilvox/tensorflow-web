# tensorflow-web

Train a model with TensorFlow.js in a browser tab and emit a verified `.tflite` file. TensorFlow.js does the training. `tensorflow-web` adds the half TensorFlow.js does not have: it calibrates the trained model over representative data, quantizes it to int8 with post training quantization, serializes a real TensorFlow Lite FlatBuffer, and then verifies that the emitted file reproduces the float model's numbers in an actual interpreter. It is a headless library. It has no user interface of its own, returns plain data and promises, and leaves all presentation to the host application.

## The one constraint to understand first

Training is general. Export is bounded by the op registry.

TensorFlow.js will happily train any model you build. `tensorflow-web` can only export the layers it has a converter for. A model that uses a layer outside the registry trains fine and then fails loudly on export with an `UnsupportedLayerError` that names the offending layer class. It never writes a silently wrong file. This is deliberate. A `.tflite` that produces different numbers than the float model is worse than no file at all, because it fails quietly. The registry refuses to guess.

Adding a layer is registering one converter. The registry grows one converter at a time.

The set of supported layers is whatever converters are registered at runtime. You can read the current set with `supportedLayers()`. The v1 target set, as described in `DESIGN.md`, is:

Conv2D, DepthwiseConv2D, Dense, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, Flatten, Reshape, Softmax, Activation, ReLU, Add.

All of these converters are registered today. `supportedLayers()` always returns the live set. See [Status](#status) below for what is verified and where.

## Install

```sh
npm install tensorflow-web
```

TensorFlow.js is a peer dependency. The host application provides it, so the library carries no second copy of tfjs:

```sh
npm install @tensorflow/tfjs
```

The int8 parity verification step loads the emitted file back into the TensorFlow Lite WASM interpreter. That interpreter ships in a separate optional peer dependency. Install it only if you intend to call `verify` or `convert` with verification:

```sh
npm install @tensorflow/tfjs-tflite
```

## Usage

The pipeline is a sequence of small, explicit stages. You can run them one at a time, or chain them with the `convert` one shot.

### Train

`train` is a thin wrapper over `model.fit` and `model.fitDataset`. The model must already be compiled by you. It forwards progress callbacks and supports cooperative cancellation through an `AbortSignal`.

```ts
import { train } from 'tensorflow-web';

const { model, history } = await train(compiledModel, {
  data: { xs, ys },        // or a tf.data.Dataset
  epochs: 10,
  batchSize: 32,
  onEpoch: (epoch, logs) => updateChart(epoch, logs),
  // signal: abortController.signal,
});
```

### Calibrate

`calibrate` runs the trained float model over representative inputs and records the observed `{ min, max }` range of every layer activation, keyed by layer name. Labels are not needed. The `minmax` method is implemented today; `percentile` is planned and throws if requested.

```ts
import { calibrate } from 'tensorflow-web';

const calibration = await calibrate(model, representativeData, { method: 'minmax' });
```

### Quantize

`quantize` lowers the model to the graph IR and produces an int8 `QuantizedModel`: symmetric per channel int8 weights, asymmetric per tensor int8 activations, int32 bias, and `QUANTIZE` / `DEQUANTIZE` ops at the graph boundary so the input and output stay float for easy parity checking.

```ts
import { quantize } from 'tensorflow-web';

const quantized = quantize(model, calibration, { weights: 'per-channel' });
```

If you already hold a float graph IR, `quantizeGraph(floatGraph, calibration, options)` is the pure function underneath, and it is unit tested against hand computed values.

### Serialize to .tflite

`toTFLite` serializes a `QuantizedModel` to TensorFlow Lite FlatBuffer bytes.

```ts
import { toTFLite } from 'tensorflow-web';

const bytes: Uint8Array = await toTFLite(quantized);
```

### Verify

`verify` is the trust layer. It loads the emitted bytes back into the real interpreter, runs both the emitted model and the float reference over the same test inputs, and reports the largest absolute error between them, a `parity` boolean within a tolerance, and, when labels are present, float and int8 accuracy plus a confusion matrix.

```ts
import { verify } from 'tensorflow-web';

const report = await verify(bytes, model, testData, { tolerance: 1e-2 });
// report.parity, report.maxAbsError, report.floatAcc, report.int8Acc
```

This step is a browser step. `@tensorflow/tfjs-tflite` is a WebAssembly build, so `verify` loads the interpreter through a dynamic import and runs in the browser, not in a plain Node process. Calling it in Node rejects when the WASM runtime cannot be found.

For a Node side sanity pass that does not touch WASM, `structuralCheck(bytes)` parses the FlatBuffer and confirms the model is well formed: the `TFL3` file identifier is present, there is exactly one subgraph, and every operator and tensor index is in range. It returns a report and never throws on a malformed buffer.

```ts
import { structuralCheck } from 'tensorflow-web';

const check = structuralCheck(bytes); // { ok, issues, tensorCount, operatorCount }
```

### Convert: the one shot path

`convert` chains the build, calibrate, quantize, serialize, and verify stages into a single call: a trained model in, a verified `.tflite` out.

```ts
import { convert } from 'tensorflow-web';

const { tflite, report } = await convert(model, {
  representativeData,
  testData,
  quantize: 'int8',
  tolerance: 1e-2,
});
```

When `testData` is omitted, `convert` returns just the bytes and runs entirely in Node. When `testData` is given, it also runs `verify`, which needs the browser interpreter (see above).

## Verification, and why it exists

Nothing the library emits is trusted until it is checked. A quantized `.tflite` can look structurally valid and still compute the wrong numbers. The only honest test is to load the emitted file into the same interpreter that will run it in production and compare its output, element by element, against the float model the user actually trained. That is what `verify` does. `parity` is true only when the largest absolute elementwise error stays inside the tolerance. If it does not, the report says so, and you do not ship the file.

This is the spine of the project, not an afterthought.

## Status

Early. Pre 1.0, but the core promise is verified end to end. Two test layers cover it:

- Node unit tests (`npm test`): the serializer, the graph IR, the dtype helpers, the quantization math, `train`, `calibrate`, `quantize`, `buildFloatGraph`, `toTFLite`, `convert`, and `structuralCheck`, plus a full pipeline test that trains a small model, quantizes it to int8, and serializes a structurally valid `.tflite`.
- Browser parity (`npm run test:parity`): trains four practical models (a CNN, a depthwise separable net, an average pooling plus sigmoid net, and a functional model with a residual skip connection), emits each as a float32 and an int8 `.tflite`, loads them back into the real TensorFlow Lite WASM interpreter through Playwright, and checks the numbers. The float exports match TensorFlow.js to floating point noise (max absolute error near 1e-7), and the int8 exports stay inside the error budget while keeping the models' accuracy with no measurable drop.

Together these verify the whole op registry numerically against the real interpreter: CONV_2D with valid and same padding, DEPTHWISE_CONV_2D, pointwise CONV_2D, FULLY_CONNECTED, MAX_POOL_2D, AVERAGE_POOL_2D, global average pooling (MEAN), RESHAPE, ADD, SOFTMAX, LOGISTIC, and fused ReLU. The op registry grows one converter at a time, and the public types are stable.

## Testing

```sh
npm test            # Node unit tests (Vitest): math, serialization, pipeline
npm run test:parity # browser parity against the real TFLite interpreter
npm run typecheck   # strict TypeScript, library and tooling
npm run build       # bundle to dist (ESM plus type declarations)
```

`npm run test:parity` starts the Vite dev server, drives `app/parity.html` in a headless Chromium through Playwright, and loads the emitted `.tflite` files into the actual TensorFlow Lite WASM interpreter. It needs a local Chrome or Chromium and network access for the interpreter's WASM. The unit tests need neither and run anywhere Node does.

## Development

```sh
npm install
npm run gen:schema  # regenerate FlatBuffers bindings from schema/schema.fbs (needs flatc)
npm run app         # run the test app at http://localhost:5173
```

Adding support for a new layer is a small, isolated change: write a converter under `src/ops/`, register it, and add a parity case. See [`DESIGN.md`](./DESIGN.md) for the architecture and the weight layout rules.

## License

Apache License 2.0. See [`LICENSE`](./LICENSE) and [`NOTICE`](./NOTICE).

Apache 2.0 matches every dependency this library builds on, it is permissive enough for wide adoption, and it carries an explicit patent grant that matters for a model format and quantization library.

## Credits

This library stands on work done by others. It is the thin layer between these projects, not a replacement for any of them.

- [TensorFlow](https://github.com/tensorflow/tensorflow) and [TensorFlow.js](https://github.com/tensorflow/tfjs), which build, train, and run the float model. The library adds only the export half that TensorFlow.js does not provide.
- The [TensorFlow Lite WASM interpreter (`@tensorflow/tfjs-tflite`)](https://github.com/tensorflow/tfjs/tree/master/tfjs-tflite), used unmodified as the parity oracle in `verify`. The library does not write a second interpreter; it trusts the real one.
- [Google FlatBuffers](https://github.com/google/flatbuffers) and the `flatc` compiler, which generate the serialization bindings and provide the runtime the serializer uses.
- [Playwright](https://github.com/microsoft/playwright), which drives a real headless browser so the int8 parity check runs against the actual WASM interpreter rather than a stand in.
- The [TensorFlow Lite FlatBuffers schema](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/mlir/lite/schema/schema.fbs) (`file_identifier` `TFL3`, `root_type` `Model`), pinned and committed in this repository. See [`schema/SCHEMA_SOURCE.md`](./schema/SCHEMA_SOURCE.md) for provenance and the regeneration procedure.
- The documented failure mode where a quantized `.tflite` is silently broken by a `flatc` JSON round trip, the conversion to text JSON and back losing or corrupting the quantization scales and buffers. That case is precisely why this library serializes through the FlatBuffers binary runtime directly and never round trips quantized models through JSON, and it is why `verify` exists: the emitted bytes are checked against the float reference in a real interpreter before they are trusted.

TensorFlow, TensorFlow.js, `tfjs-tflite`, the TFLite schema, and FlatBuffers are each licensed under Apache 2.0. They are peer or development dependencies and are not redistributed as part of this library.
