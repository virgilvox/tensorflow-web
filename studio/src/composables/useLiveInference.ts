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

/** Coerces a predict result to one output tensor, disposing any others. */
function asTensor(out: unknown): tf.Tensor {
  if (Array.isArray(out)) {
    const first = out[0] as tf.Tensor;
    for (let i = 1; i < out.length; i++) (out[i] as tf.Tensor)?.dispose?.();
    return first;
  }
  if (out && typeof (out as tf.Tensor).dataSync === 'function') return out as tf.Tensor;
  const values = Object.values(out as Record<string, tf.Tensor>);
  for (let i = 1; i < values.length; i++) values[i]?.dispose?.();
  return values[0] as tf.Tensor;
}

/** Releases a loaded interpreter model's WASM memory, if it exposes dispose. */
function disposeModel(m: LoadedModel | null): void {
  (m as (LoadedModel & { dispose?: () => void }) | null)?.dispose?.();
}

export function useLiveInference() {
  const interpreter = useInterpreter();

  /** Loads bytes into the interpreter for subsequent predict calls. */
  async function load(bytes: Uint8Array): Promise<void> {
    disposeModel(model); // release the previous interpreter before replacing it
    model = await interpreter.loadModel(bytes);
    loaded.value = true;
  }

  /** Drops the loaded model and releases its interpreter memory. */
  function reset(): void {
    disposeModel(model);
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
