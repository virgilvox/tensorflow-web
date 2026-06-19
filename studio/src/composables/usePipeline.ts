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
import { defaultFeatureConfig, featureExtractor, featureShape, type FeatureConfig } from '../features';
import { imageToFeatures, type RgbaFrame } from '../features/image';
import { audioToFeatures } from '../features/audio';
import { buildBatchedData } from '../lib/tensors';
import { sessionAwareSplit } from '../lib/split';
import { buildModel, summarize } from '../models/builder';
import { imageCnnPreset, mlpPreset } from '../models/presets';
import type { ModelSpec } from '../models/types';
import type { Modality } from '../types';
import { resolverCallsForLayers } from '../lib/cformat';
import type { Sample } from '../types';
import type { ModelSummary } from '../models/types';

/** The minimum samples per class before training is allowed. */
export const MIN_PER_CLASS = 5;

/** Modalities whose full pipeline is wired today. */
const SUPPORTED: ReadonlySet<Modality> = new Set<Modality>(['image', 'audio']);

/**
 * Chooses an architecture for a feature shape: a small CNN for a 2D grid (image
 * or spectrogram), an MLP for a 1D feature vector (motion summary or text).
 */
function presetFor(shape: number[], classCount: number): ModelSpec {
  if (shape.length === 3) return imageCnnPreset(classCount, Math.min(shape[0]!, shape[1]!));
  return mlpPreset(classCount);
}

// Shared singleton state: one trained model and one emitted artifact per session.
const trainedModel = shallowRef<Model | null>(null);
const bytes = ref<Uint8Array | null>(null);
const classOrder = ref<string[]>([]);
const classNames = ref<string[]>([]);
const featureCfg = ref<FeatureConfig | null>(null);
const fShape = ref<number[]>([]);
const hasModel = ref(false);

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

export function usePipeline() {
  const project = useProjectStore();
  const training = useTrainingStore();
  const settings = useSettingsStore();
  const trainer = useTrainer();
  const quantizer = useQuantizer();
  const verifier = useVerifier();
  const budget = useDeviceBudget();
  const live = useLiveInference();

  /** True when the dataset clears the minimum bar to train. */
  function canTrain(): boolean {
    if (project.classes.length < 2) return false;
    return project.classes.every((c) => (project.countsByClass[c.id] ?? 0) >= MIN_PER_CLASS);
  }

  /** Builds a preset model just to report its size for the Model stage. */
  function previewSummary(): ModelSummary | null {
    if (!SUPPORTED.has(project.modality)) return null;
    const cfg = defaultFeatureConfig(project.modality);
    const shape = featureShape(cfg);
    const classCount = Math.max(2, project.classes.length);
    const model = buildModel(presetFor(shape, classCount), shape);
    try {
      return summarize(model);
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

    // Reset any prior run.
    disposeTensors();
    trainedModel.value?.dispose();
    trainedModel.value = null;
    bytes.value = null;
    hasModel.value = false;
    live.reset();

    const cfg = defaultFeatureConfig(project.modality);
    const shape = featureShape(cfg);
    const extract = featureExtractor(cfg);
    const ids = project.classes.map((c) => c.id);

    // Session aware split, with a random per class fallback if every class has a
    // single session (which cannot be split without leakage).
    let { train: trainSamples, test: testSamples } = sessionAwareSplit(project.samples, 0.25);
    if (testSamples.length === 0) {
      ({ trainSamples, testSamples } = randomHoldout(project.samples, 0.25));
    }

    const trainData = buildBatchedData(trainSamples, ids, extract, shape);
    const testData = buildBatchedData(testSamples, ids, extract, shape);

    const classCount = ids.length;
    const model = buildModel(presetFor(shape, classCount), shape);

    trainXs = trainData.xs;
    testXs = testData.xs;
    testYs = testData.ys;
    classOrder.value = ids;
    classNames.value = project.classes.map((c) => c.name);
    featureCfg.value = cfg;
    fShape.value = shape;

    try {
      await trainer.run({
        model,
        data: { xs: trainXs, ys: trainData.ys },
        validationData: { xs: testXs, ys: testYs },
        epochs,
        batchSize,
      });
    } finally {
      // The training labels are not needed past the fit; the inputs stay for
      // calibration at export time.
      trainData.ys.dispose();
    }

    trainedModel.value = model;
  }

  /**
   * Calibrates, quantizes, emits, verifies, and budgets the trained model, then
   * loads the result for live inference.
   *
   * @returns nothing; results land in the training store and this pipeline.
   * @throws if no model is trained, or the interpreter is unavailable.
   */
  async function exportModel(): Promise<void> {
    const model = trainedModel.value;
    if (!model || !trainXs || !testXs || !testYs) {
      throw new Error('Train a model before exporting.');
    }
    const emitted = await quantizer.toInt8Tflite(model, trainXs);
    bytes.value = emitted;

    const report = await verifier.run(emitted, model, { xs: testXs, ys: testYs }, 0.05);
    const result = budget.evaluate(emitted, model, settings.target, report.arenaBytes);
    training.setArtifacts(emitted.length, report, result);

    await live.load(emitted);
    hasModel.value = true;
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

  /** The op resolver calls and input shape a sketch export needs. */
  function exportOps(): { resolverCalls: string[]; inputShape: number[] } {
    const model = trainedModel.value;
    const classes = model?.layers.map((l) => l.getClassName()) ?? [];
    const resolverCalls = resolverCallsForLayers(classes, true);
    const inputShape = [1, ...fShape.value];
    return { resolverCalls, inputShape };
  }

  return {
    trainedModel,
    bytes,
    classNames,
    hasModel,
    canTrain,
    previewSummary,
    train,
    cancel: trainer.cancel,
    exportModel,
    predictImage,
    predictAudio,
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
