/**
 * Playground inference. Runs a model live against fresh input, independent of the
 * training project. It loads from three sources: the current session model, a
 * self-contained bundle (.tfwsmodel.json), or a bare .tflite the caller
 * configures by hand. It holds its OWN interpreter instance, separate from the
 * Test stage's shared live inference, so loading a model here never disturbs the
 * project's model. Feature extraction reuses the same pure extractors training
 * uses, so a studio model behaves identically here.
 */
import { ref } from 'vue';
import * as tf from '@tensorflow/tfjs';
import { useInterpreter, type LoadedModel } from './useInterpreter';
import { imageToFeatures, type RgbaFrame } from '../features/image';
import { audioToFeatures } from '../features/audio';
import { motionToFeatures } from '../features/motion';
import { textToFeatures } from '../features/text';
import type { FeatureConfig } from '../features';
import type { Modality } from '../types';
import { parseBundle } from '../lib/modelBundle';

/** Where a loaded model came from, shown so the user knows what is running. */
export type ModelSource = 'current' | 'bundle' | 'tflite';

/** A prediction, one per class. */
export interface Prediction {
  name: string;
  score: number;
}

/** Everything needed to run a model live and label its outputs. */
export interface ActiveModel {
  name: string;
  source: ModelSource;
  modality: Modality;
  classes: string[];
  featureConfig: FeatureConfig;
  /** Per sample input shape, batch aside. */
  inputShape: number[];
  /** Microphone clip length for audio capture, if the bundle carried one. */
  audioSeconds?: number;
}

/**
 * Coerces an interpreter predict result to a single output tensor, disposing any
 * other output tensors a multi-output model returns so they do not leak on every
 * prediction.
 */
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

export function usePlayground() {
  const interpreter = useInterpreter();
  const active = ref<ActiveModel | null>(null);
  // The loaded interpreter model, held locally so it never touches the shared
  // live inference singleton the Test stage uses.
  let model: LoadedModel | null = null;

  /** True when the interpreter global is present. */
  function ready(): boolean {
    return interpreter.ready();
  }

  /** Loads bytes into the interpreter and marks the descriptor active. */
  async function loadInto(bytes: Uint8Array, descriptor: ActiveModel): Promise<void> {
    model = await interpreter.loadModel(bytes);
    active.value = descriptor;
  }

  /** Loads the current session model from pipeline state. */
  async function loadCurrent(input: {
    name: string;
    modality: Modality;
    classes: string[];
    featureConfig: FeatureConfig;
    inputShape: number[];
    audioSeconds?: number;
    bytes: Uint8Array;
  }): Promise<void> {
    await loadInto(input.bytes, {
      name: input.name,
      source: 'current',
      modality: input.modality,
      classes: input.classes,
      featureConfig: input.featureConfig,
      inputShape: input.inputShape,
      audioSeconds: input.audioSeconds,
    });
  }

  /**
   * Loads a self-contained bundle from its JSON text.
   *
   * @throws if the text is not a valid bundle.
   */
  async function loadBundle(jsonText: string): Promise<void> {
    const b = parseBundle(JSON.parse(jsonText));
    await loadInto(b.bytes, {
      name: b.name,
      source: 'bundle',
      modality: b.modality,
      classes: b.classes,
      featureConfig: b.featureConfig,
      inputShape: b.inputShape,
      audioSeconds: b.audioSeconds,
    });
  }

  /**
   * Loads bare .tflite bytes into the interpreter for inspection, without
   * activating a descriptor yet. Returns the input shape the interpreter
   * reports, if it exposes one, so the caller can prefill the manual setup.
   */
  async function inspectTflite(bytes: Uint8Array): Promise<number[] | null> {
    model = await interpreter.loadModel(bytes);
    active.value = null;
    const m = model as unknown as { inputs?: { shape?: number[] }[] };
    const shape = m?.inputs?.[0]?.shape;
    // Drop a leading batch dimension of 1 so the hint matches a per sample shape.
    if (!Array.isArray(shape)) return null;
    return shape[0] === 1 ? shape.slice(1) : shape;
  }

  /**
   * Activates a bare .tflite that inspectTflite already loaded, with the modality,
   * classes, and feature config the caller built from the manual setup.
   *
   * @throws if no bare model has been inspected.
   */
  function activateBare(descriptor: Omit<ActiveModel, 'source'>): void {
    if (!model) throw new Error('Load a .tflite first.');
    active.value = { ...descriptor, source: 'tflite' };
  }

  /** Unloads the active model. */
  function clear(): void {
    model = null;
    active.value = null;
  }

  /** Runs one flat feature vector through the loaded model. */
  async function run(features: Float32Array): Promise<number[]> {
    if (!model || !active.value) throw new Error('No model loaded.');
    const input = tf.tensor(features, [1, ...active.value.inputShape]);
    try {
      const out = asTensor(model.predict(input));
      const data = (await out.data()) as Float32Array;
      out.dispose();
      return Array.from(data);
    } finally {
      input.dispose();
    }
  }

  /** Maps a score array to named, per class predictions. */
  function toResults(scores: number[]): Prediction[] {
    const names = active.value?.classes ?? [];
    return scores.map((score, i) => ({ name: names[i] ?? `class ${i}`, score }));
  }

  /** Predicts on one camera frame. @throws if the active model is not image. */
  async function predictImage(frame: RgbaFrame): Promise<Prediction[]> {
    if (active.value?.featureConfig.kind !== 'image') throw new Error('Loaded model is not an image model.');
    return toResults(await run(imageToFeatures(frame, active.value.featureConfig.image)));
  }

  /** Predicts on one audio clip. @throws if the active model is not audio. */
  async function predictAudio(samples: Float32Array, sampleRate: number): Promise<Prediction[]> {
    if (active.value?.featureConfig.kind !== 'audio') throw new Error('Loaded model is not an audio model.');
    return toResults(await run(audioToFeatures(samples, sampleRate, active.value.featureConfig.audio)));
  }

  /** Predicts on one motion window. @throws if the active model is not motion. */
  async function predictMotion(data: Float32Array, axes: number): Promise<Prediction[]> {
    if (active.value?.featureConfig.kind !== 'motion') throw new Error('Loaded model is not a motion model.');
    return toResults(await run(motionToFeatures(data, axes, active.value.featureConfig.motion)));
  }

  /** Predicts on one text string. @throws if the active model is not text. */
  async function predictText(text: string): Promise<Prediction[]> {
    if (active.value?.featureConfig.kind !== 'text') throw new Error('Loaded model is not a text model.');
    return toResults(await run(textToFeatures(text, active.value.featureConfig.text)));
  }

  return {
    active,
    ready,
    loadCurrent,
    loadBundle,
    inspectTflite,
    activateBare,
    clear,
    predictImage,
    predictAudio,
    predictMotion,
    predictText,
  };
}
