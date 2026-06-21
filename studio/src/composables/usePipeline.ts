/**
 * The pipeline orchestrator. Shared, session wide state that carries the trained
 * model and the emitted bytes across the Train, Export, and Test stages, and the
 * high level actions those stages call: train, export, and live predict. It
 * composes the single purpose composables (trainer, quantizer, verifier, budget,
 * live inference) and the pure feature, model, split, and tensor helpers. The
 * stores hold display state; the heavy artifacts (the model, the tensors, the
 * bytes) live here, disposed deliberately on each new run.
 */
import { shallowRef, ref } from 'vue';
import * as tf from '@tensorflow/tfjs';
import type { Model } from 'tensorflow-web';
import { useProjectStore } from '../stores/project';
import { useTrainingStore } from '../stores/training';
import { useSettingsStore } from '../stores/settings';
import { useTrainer } from './useTrainer';
import { useQuantizer } from './useQuantizer';
import { useVerifier } from './useVerifier';
import { useDeviceBudget } from './useDeviceBudget';
import { useLiveInference } from './useLiveInference';
import {
  buildFeatureConfig,
  featureExtractor,
  featureShape,
  type FeatureConfig,
} from '../features';
import { imageToFeatures, type RgbaFrame } from '../features/image';
import { audioToFeatures } from '../features/audio';
import { motionToFeatures } from '../features/motion';
import { textToFeatures } from '../features/text';
import { buildBatchedData } from '../lib/tensors';
import { sessionAwareSplit } from '../lib/split';
import { buildModel, summarize, describeLayers, type LayerInfo } from '../models/builder';
import { imageCnnPreset, mlpPreset } from '../models/presets';
import type { ModelSpec, ModelCapacity } from '../models/types';
import type { Modality } from '../types';
import { resolverCallsForLayers } from '../lib/cformat';
import type { Sample } from '../types';
import type { ModelSummary } from '../models/types';

/** The minimum samples per class before training is allowed. */
export const MIN_PER_CLASS = 5;

/** Modalities whose full pipeline is wired today. */
const SUPPORTED: ReadonlySet<Modality> = new Set<Modality>(['image', 'audio', 'motion', 'text']);

/**
 * Chooses an architecture for a feature shape: a small CNN for a 2D grid (image
 * or spectrogram), an MLP for a 1D feature vector (motion summary or text).
 */
function presetFor(shape: number[], classCount: number, capacity: ModelCapacity): ModelSpec {
  if (shape.length === 3) return imageCnnPreset(classCount, Math.min(shape[0]!, shape[1]!), capacity);
  return mlpPreset(classCount, capacity);
}

// Shared singleton state: one trained model and one emitted artifact per session.
const trainedModel = shallowRef<Model | null>(null);
const bytes = ref<Uint8Array | null>(null);
const classOrder = ref<string[]>([]);
const classNames = ref<string[]>([]);
const featureCfg = ref<FeatureConfig | null>(null);
const fShape = ref<number[]>([]);
const hasModel = ref(false);
// True while train() or exportModel() is running. A single flag both check so a
// second job cannot start mid-run and so reset() (New project, Import, modality
// switch) cannot free tensors out from under an in-flight fit or export.
const busy = ref(false);

// Tensors kept alive between train and export, disposed on the next run.
let trainXs: tf.Tensor | null = null;
let testXs: tf.Tensor | null = null;
let testYs: tf.Tensor | null = null;

function disposeTensors(): void {
  trainXs?.dispose();
  testXs?.dispose();
  testYs?.dispose();
  trainXs = testXs = testYs = null;
}

/**
 * Disposes a model and its optimizer. tfjs frees the optimizer on dispose() only
 * when it was compiled from a string; the builder compiles with an Adam instance,
 * which leaves the per weight slot variables (m, v) for us to free, so they are
 * released explicitly here. Safe to call with null.
 */
function disposeModel(model: Model | null): void {
  if (!model) return;
  const optimizer = (model as Model & { optimizer?: { dispose?: () => void } }).optimizer;
  model.dispose();
  optimizer?.dispose?.();
}

export function usePipeline() {
  const project = useProjectStore();
  const training = useTrainingStore();
  const settings = useSettingsStore();
  const trainer = useTrainer();
  const quantizer = useQuantizer();
  const verifier = useVerifier();
  const budget = useDeviceBudget();
  const live = useLiveInference();

  /**
   * Samples that can actually train the current model: their class still exists
   * and their payload matches the current modality. Filtering here keeps a stale
   * sample (an orphan, or one left from a different modality) from reaching the
   * tensor builder, where it would throw.
   */
  function usableSamples(): Sample[] {
    const known = new Set(project.classes.map((c) => c.id));
    return project.samples.filter(
      (s) => known.has(s.classId) && s.payload.kind === project.modality,
    );
  }

  /** True when the usable dataset clears the minimum bar to train. */
  function canTrain(): boolean {
    if (project.classes.length < 2) return false;
    const counts = new Map<string, number>();
    for (const s of usableSamples()) counts.set(s.classId, (counts.get(s.classId) ?? 0) + 1);
    return project.classes.every((c) => (counts.get(c.id) ?? 0) >= MIN_PER_CLASS);
  }

  /** The Standard feature knobs, read from settings, passed to buildFeatureConfig. */
  function featureOptions() {
    return {
      audioSeconds: settings.audioSeconds,
      imageSize: settings.imageSize,
      imageChannels: settings.imageChannels,
      audioMode: settings.audioMode,
      audioBands: settings.audioBands,
      motionSteps: settings.motionSteps,
      textVocabCap: settings.textVocabCap,
    };
  }

  /** Sample count per class that can actually train the current model. */
  function usableCounts(): Record<string, number> {
    const counts: Record<string, number> = {};
    for (const c of project.classes) counts[c.id] = 0;
    for (const s of usableSamples()) counts[s.classId] = (counts[s.classId] ?? 0) + 1;
    return counts;
  }

  /** Builds a preset model just to report its size for the Model stage. */
  function previewSummary(): ModelSummary | null {
    if (!SUPPORTED.has(project.modality)) return null;
    const cfg = buildFeatureConfig(project.modality, project.samples, featureOptions());
    const shape = featureShape(cfg);
    // A data derived shape (text vocabulary) can be empty before any data; the
    // size estimate only makes sense once the input has a width.
    if (shape.some((d) => d <= 0)) return null;
    const classCount = Math.max(2, project.classes.length);
    const model = buildModel(presetFor(shape, classCount, settings.modelCapacity), shape);
    try {
      return summarize(model);
    } finally {
      model.dispose();
    }
  }

  /** Describes the preset model's layers for the Expert operator inspector. */
  function inspectLayers(): LayerInfo[] {
    if (!SUPPORTED.has(project.modality)) return [];
    const cfg = buildFeatureConfig(project.modality, project.samples, featureOptions());
    const shape = featureShape(cfg);
    if (shape.some((d) => d <= 0)) return [];
    const model = buildModel(presetFor(shape, Math.max(2, project.classes.length), settings.modelCapacity), shape);
    try {
      return describeLayers(model);
    } finally {
      model.dispose();
    }
  }

  /**
   * Trains a model on the current dataset. Splits by capture session, builds the
   * tensors and the preset architecture, and runs training with live metrics.
   *
   * @throws if the modality is not yet implemented or the data is insufficient.
   */
  async function train(epochs: number, batchSize = 16): Promise<void> {
    if (!SUPPORTED.has(project.modality)) {
      throw new Error(`Training for ${project.modality} lands with its phase.`);
    }
    if (!canTrain()) throw new Error('Not enough data to train yet.');
    if (busy.value) throw new Error('A job is already running.');
    busy.value = true;
    try {

    // Reset any prior run, freeing the previous model, its optimizer, and tensors.
    disposeTensors();
    disposeModel(trainedModel.value);
    trainedModel.value = null;
    bytes.value = null;
    hasModel.value = false;
    live.reset();

    const ids = project.classes.map((c) => c.id);

    // Session aware split over the usable samples only, with a random per class
    // fallback if every class has a single session (which cannot be split without
    // leakage).
    let { train: trainSamples, test: testSamples } = sessionAwareSplit(usableSamples(), 0.25);
    if (testSamples.length === 0) {
      ({ trainSamples, testSamples } = randomHoldout(usableSamples(), 0.25));
    }

    // Build the feature config from the train split only, so a text vocabulary
    // never sees the held out test set.
    const cfg = buildFeatureConfig(project.modality, trainSamples, featureOptions());
    const shape = featureShape(cfg);
    if (shape.some((d) => d <= 0)) throw new Error('No usable features; add more varied data.');
    const extract = featureExtractor(cfg);
    const classCount = ids.length;

    // Allocate the tensors and the model inside a guarded block so a failure at
    // any step frees everything it created instead of orphaning tensors. The
    // module refs are assigned as soon as each tensor exists, so disposeTensors
    // can reach them on the error path.
    let model: Model | null = null;
    let trainYs: tf.Tensor | null = null;
    try {
      const trainData = buildBatchedData(trainSamples, ids, extract, shape);
      trainXs = trainData.xs;
      trainYs = trainData.ys;
      const testData = buildBatchedData(testSamples, ids, extract, shape);
      testXs = testData.xs;
      testYs = testData.ys;

      model = buildModel(presetFor(shape, classCount, settings.modelCapacity), shape);

      classOrder.value = ids;
      classNames.value = project.classes.map((c) => c.name);
      featureCfg.value = cfg;
      fShape.value = shape;

      await trainer.run({
        model,
        data: { xs: trainXs, ys: trainYs },
        validationData: { xs: testXs, ys: testYs },
        epochs,
        batchSize,
      });

      trainedModel.value = model;
      model = null; // ownership transferred to the store; do not dispose below
    } catch (err) {
      disposeTensors();
      disposeModel(model);
      throw err;
    } finally {
      // The training labels are not needed past the fit (or after a failure);
      // the inputs stay for calibration at export time.
      trainYs?.dispose();
    }
    } finally {
      busy.value = false;
    }
  }

  /**
   * Calibrates, quantizes, emits, verifies, and budgets the trained model. The
   * emitted bytes become shippable (downloadable and loaded for live inference)
   * only when verify reports parity: a .tflite that does not match the float
   * reference is worthless, so it is never exposed. The verify report is always
   * stored so the failure is shown, not hidden.
   *
   * @throws if no model is trained, or the interpreter is unavailable. On any
   *   failure the pipeline is left with no shippable artifact.
   */
  async function exportModel(): Promise<void> {
    const model = trainedModel.value;
    if (!model || !trainXs || !testXs || !testYs) {
      throw new Error('Train a model before exporting.');
    }
    if (busy.value) throw new Error('A job is already running.');
    busy.value = true;
    try {
      const emitted = await quantizer.toInt8Tflite(model, trainXs);
      const report = await verifier.run(emitted, model, { xs: testXs, ys: testYs }, 0.05);
      const result = budget.evaluate(emitted, model, settings.target, report.arenaBytes);
      // Show the result either way, including a parity failure.
      training.setArtifacts(emitted.length, report, result);

      if (report.parity) {
        bytes.value = emitted;
        await live.load(emitted);
        hasModel.value = true;
      } else {
        // Parity failed: do not expose the artifact for download or live use.
        bytes.value = null;
        hasModel.value = false;
        live.reset();
      }
    } catch (err) {
      bytes.value = null;
      hasModel.value = false;
      live.reset();
      throw err;
    } finally {
      busy.value = false;
    }
  }

  /** Maps a raw score array to named, per class results. */
  function toResults(scores: number[]): { name: string; score: number }[] {
    return scores.map((score, i) => ({ name: classNames.value[i] ?? `class ${i}`, score }));
  }

  /** Runs one captured frame through the live model, returning per class scores. */
  async function predictImage(frame: RgbaFrame): Promise<{ name: string; score: number }[]> {
    if (featureCfg.value?.kind !== 'image') throw new Error('No image model; train an image project first.');
    const features = imageToFeatures(frame, featureCfg.value.image);
    return toResults(await live.predict(features, fShape.value));
  }

  /** Runs one captured audio clip through the live model, returning per class scores. */
  async function predictAudio(
    samples: Float32Array,
    sampleRate: number,
  ): Promise<{ name: string; score: number }[]> {
    if (featureCfg.value?.kind !== 'audio') throw new Error('No audio model; train an audio project first.');
    const features = audioToFeatures(samples, sampleRate, featureCfg.value.audio);
    return toResults(await live.predict(features, fShape.value));
  }

  /** Runs one captured motion window through the live model. */
  async function predictMotion(
    data: Float32Array,
    axes: number,
  ): Promise<{ name: string; score: number }[]> {
    if (featureCfg.value?.kind !== 'motion') throw new Error('No motion model; train a motion project first.');
    const features = motionToFeatures(data, axes, featureCfg.value.motion);
    return toResults(await live.predict(features, fShape.value));
  }

  /** Runs one text string through the live model. */
  async function predictText(text: string): Promise<{ name: string; score: number }[]> {
    if (featureCfg.value?.kind !== 'text') throw new Error('No text model; train a text project first.');
    const features = textToFeatures(text, featureCfg.value.text);
    return toResults(await live.predict(features, fShape.value));
  }

  /**
   * Clears every trained artifact and resets the live model and the training
   * store. Called whenever the dataset changes out from under the model (a new
   * project, an imported project, or a modality switch) so a model trained on
   * one dataset can never be tested or downloaded against another. Disposes the
   * model, its optimizer, and the retained tensors.
   */
  function reset(): void {
    // Never tear down artifacts mid-run; callers guard on busy too, this is the
    // backstop so an in-flight fit or export keeps its tensors and model.
    if (busy.value) return;
    disposeTensors();
    disposeModel(trainedModel.value);
    trainedModel.value = null;
    bytes.value = null;
    classOrder.value = [];
    classNames.value = [];
    featureCfg.value = null;
    fShape.value = [];
    hasModel.value = false;
    live.reset();
    training.$reset();
  }

  /** The op resolver calls and input shape a sketch export needs. */
  function exportOps(): { resolverCalls: string[]; inputShape: number[] } {
    const layers = trainedModel.value?.layers ?? [];
    const classes = layers.map((l) => l.getClassName());
    // Derive the softmax head from the real model's last layer activation rather
    // than assuming it, so the resolver list reflects the actual graph.
    const last = layers[layers.length - 1];
    const activation = last
      ? (last.getConfig() as { activation?: string }).activation
      : undefined;
    const resolverCalls = resolverCallsForLayers(classes, activation === 'softmax');
    const inputShape = [1, ...fShape.value];
    return { resolverCalls, inputShape };
  }

  return {
    trainedModel,
    bytes,
    classNames,
    featureCfg,
    fShape,
    hasModel,
    busy,
    canTrain,
    usableCounts,
    previewSummary,
    inspectLayers,
    train,
    cancel: trainer.cancel,
    exportModel,
    reset,
    predictImage,
    predictAudio,
    predictMotion,
    predictText,
    exportOps,
  };
}

/** A random per class holdout, used only when a session split is impossible. */
function randomHoldout(samples: Sample[], fraction: number): { trainSamples: Sample[]; testSamples: Sample[] } {
  const byClass = new Map<string, Sample[]>();
  for (const s of samples) {
    const arr = byClass.get(s.classId) ?? [];
    arr.push(s);
    byClass.set(s.classId, arr);
  }
  const trainSamples: Sample[] = [];
  const testSamples: Sample[] = [];
  for (const arr of byClass.values()) {
    const cut = Math.max(1, Math.round(arr.length * fraction));
    testSamples.push(...arr.slice(0, cut));
    trainSamples.push(...arr.slice(cut));
  }
  return { trainSamples, testSamples };
}
