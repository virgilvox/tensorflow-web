/**
 * Shared TensorFlow.js setup for Node tests. The CPU backend is pure JavaScript
 * and runs small models fine, which is all the unit tests need. Call useCpu()
 * before touching tf so every test file initializes the backend the same way.
 */
import * as tf from '@tensorflow/tfjs';

let ready: Promise<void> | null = null;

export async function useCpu(): Promise<typeof tf> {
  if (!ready) {
    ready = (async () => {
      await tf.setBackend('cpu');
      await tf.ready();
    })();
  }
  await ready;
  return tf;
}

export { tf };
