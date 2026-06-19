/**
 * One shot convenience: a train ready model in, a verified .tflite out. Chains
 * the pipeline stages into a single call. See DESIGN.md.
 */
import type { ConvertOptions, ConvertResult, Model, QuantizedModel } from './types';
import { calibrate } from './calibrate';
import { quantize } from './quantize';
import { buildFloatGraph } from './convert/build-graph';
import { toTFLite } from './emit';
import { verify } from './verify';

/**
 * Lowers, optionally quantizes, serializes, and optionally verifies a model.
 *
 * With `quantize: 'int8'` (the default) the model is calibrated over
 * `representativeData` and quantized to int8. With `quantize: 'float'` the model
 * is serialized as float32 and `representativeData` is ignored.
 *
 * Verification runs only when `testData` is provided. It loads the emitted file
 * in the tfjs-tflite WASM interpreter and so requires a browser; in a non browser
 * host, pass no `testData` and verify separately where an interpreter exists.
 *
 * @returns the emitted bytes and, when `testData` is given, a parity report.
 * @throws UnsupportedLayerError if the model uses a layer with no converter.
 */
export async function convert(model: Model, options: ConvertOptions): Promise<ConvertResult> {
  const scheme = options.quantize ?? 'int8';

  let lowered: QuantizedModel;
  if (scheme === 'float') {
    lowered = { graph: buildFloatGraph(model), scheme: 'float' };
  } else {
    const calibration = await calibrate(model, options.representativeData, options.calibrate);
    lowered = quantize(model, calibration, options.quantizeOptions);
  }

  const tflite = await toTFLite(lowered);

  if (!options.testData) return { tflite };

  const report = await verify(tflite, model, options.testData, { tolerance: options.tolerance });
  return { tflite, report };
}
