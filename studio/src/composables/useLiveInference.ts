/**
 * Live inference. Loads the emitted .tflite once and runs it on fresh input one
 * sample at a time, as the model's fixed batch of one requires. Returns plain
 * class score arrays so the Test stage can draw confidence bars without holding a
 * tensor. The loaded model is shared across callers so the Test stage and any
 * preview use the same interpreter instance.
 */
import { ref } from 'vue';
import * as tf from '@tensorflow/tfjs';
import { useInterpreter, type LoadedModel } from './useInterpreter';

// Shared so loading once in one place serves every consumer this session.
const loaded = ref(false);
let model: LoadedModel | null = null;

/** Coerces an interpreter predict result to a single output tensor. */
function asTensor(out: unknown): tf.Tensor {
  if (Array.isArray(out)) return out[0] as tf.Tensor;
  if (out && typeof (out as tf.Tensor).dataSync === 'function') return out as tf.Tensor;
  return Object.values(out as Record<string, tf.Tensor>)[0] as tf.Tensor;
}

export function useLiveInference() {
  const interpreter = useInterpreter();

  /** Loads bytes into the interpreter for subsequent predict calls. */
  async function load(bytes: Uint8Array): Promise<void> {
    model = await interpreter.loadModel(bytes);
    loaded.value = true;
  }

  /** Drops the loaded model. */
  function reset(): void {
    model = null;
    loaded.value = false;
  }

  /**
   * Runs one sample through the loaded model.
   *
   * @param features the flat feature vector for one sample.
   * @param featureShape its shape without the batch dimension.
   * @returns the class score array.
   * @throws if no model is loaded.
   */
  async function predict(features: Float32Array, featureShape: number[]): Promise<number[]> {
    if (!model) throw new Error('No model loaded for live inference.');
    const input = tf.tensor(features, [1, ...featureShape]);
    try {
      const out = asTensor(model.predict(input));
      const data = (await out.data()) as Float32Array;
      out.dispose();
      return Array.from(data);
    } finally {
      input.dispose();
    }
  }

  return { loaded, load, reset, predict };
}
