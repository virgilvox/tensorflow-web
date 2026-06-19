/**
 * Emits a real .tflite from a lowered model. This is a thin async wrapper over
 * the serializer so the public surface reads naturally and stays open to future
 * asynchronous work (for example streaming large buffers).
 */
import { serialize } from './serialize';
import type { QuantizedModel } from './types';

export async function toTFLite(model: QuantizedModel): Promise<Uint8Array> {
  return serialize(model.graph);
}
