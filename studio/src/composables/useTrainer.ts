/**
 * Wraps the library's train() with the studio's progress store and a cancel
 * control. Forwards per epoch metrics into the training store for the gauges and
 * the loss curve, and exposes an AbortSignal backed cancel. It does not build the
 * model or the data; those are assembled by the caller and passed in.
 */
import * as tf from '@tensorflow/tfjs';
import { train, type Model } from 'tensorflow-web';
import { useTrainingStore } from '../stores/training';

export interface TrainRun {
  model: Model;
  data: { xs: tf.Tensor; ys: tf.Tensor };
  validationData?: { xs: tf.Tensor; ys: tf.Tensor };
  epochs: number;
  batchSize?: number;
}

/** Reads a loss or accuracy value out of a tfjs logs object, which is loosely typed. */
function num(value: unknown): number | undefined {
  return typeof value === 'number' && !Number.isNaN(value) ? value : undefined;
}

export function useTrainer() {
  const store = useTrainingStore();
  let controller: AbortController | null = null;

  /**
   * Runs a training job, streaming metrics into the training store.
   *
   * @returns nothing; read the result from the training store and the model.
   */
  async function run(opts: TrainRun): Promise<void> {
    store.begin(opts.epochs);
    controller = new AbortController();
    try {
      await train(opts.model, {
        data: opts.data,
        validationData: opts.validationData,
        epochs: opts.epochs,
        batchSize: opts.batchSize ?? 16,
        shuffle: true,
        signal: controller.signal,
        onEpoch: (epoch, logs) => {
          store.pushEpoch({
            epoch: epoch + 1,
            loss: num(logs.loss) ?? 0,
            acc: num(logs.acc) ?? num((logs as Record<string, unknown>).accuracy),
            valLoss: num(logs.val_loss),
            valAcc: num(logs.val_acc) ?? num((logs as Record<string, unknown>).val_accuracy),
          });
        },
      });
      store.finish(controller.signal.aborted ? 'cancelled' : 'done', 'trained');
    } catch (err) {
      store.finish('error', err instanceof Error ? err.message : String(err));
      throw err;
    } finally {
      controller = null;
    }
  }

  /** Requests cooperative cancellation of the running job. */
  function cancel(): void {
    controller?.abort();
  }

  return { run, cancel };
}
