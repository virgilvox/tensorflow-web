/**
 * Calibrates and quantizes a trained model to int8 and emits the .tflite bytes.
 * A thin wrapper over the library's calibrate, quantize, and toTFLite so the
 * Export stage reads as three plain steps. Representative data drives the
 * activation ranges; no labels are needed for it.
 */
import * as tf from '@tensorflow/tfjs';
import { calibrate, quantize, toTFLite, type Model } from 'tensorflow-web';

export function useQuantizer() {
  /**
   * Runs the int8 path end to end.
   *
   * @param model the trained float reference model.
   * @param representative input tensors that cover the expected range.
   * @returns the emitted .tflite bytes.
   * @throws if the model uses a layer the export registry does not support.
   */
  async function toInt8Tflite(model: Model, representative: tf.Tensor): Promise<Uint8Array> {
    const calibration = await calibrate(model, representative, { method: 'minmax' });
    const quantized = quantize(model, calibration);
    return toTFLite(quantized);
  }

  return { toInt8Tflite };
}
