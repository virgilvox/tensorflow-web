/**
 * Feature dispatch. Maps a sample to the flat feature vector its modality
 * defines, given that modality's feature config, and reports the tensor shape a
 * config produces. Each modality's extractor lives in its own folder and is a
 * pure function; this module only routes to the right one. Text builds its
 * vocabulary from the data, so configs can be derived from the training set with
 * buildFeatureConfig; the others have a static default.
 */
import type { Modality, Sample } from '../types';
import {
  imageToFeatures,
  imageTensorShape,
  DEFAULT_IMAGE_CONFIG,
  type ImageFeatureConfig,
  type RgbaFrame,
} from './image';
import {
  audioToFeatures,
  audioTensorShape,
  DEFAULT_AUDIO_CONFIG,
  type AudioFeatureConfig,
} from './audio';
import {
  motionToFeatures,
  motionTensorShape,
  DEFAULT_MOTION_CONFIG,
  type MotionFeatureConfig,
} from './motion';
import {
  textToFeatures,
  textTensorShape,
  buildVocabulary,
  type TextFeatureConfig,
} from './text';

/** The feature config for a modality. */
export type FeatureConfig =
  | { kind: 'image'; image: ImageFeatureConfig }
  | { kind: 'audio'; audio: AudioFeatureConfig }
  | { kind: 'motion'; motion: MotionFeatureConfig }
  | { kind: 'text'; text: TextFeatureConfig };

/** The default feature config for a modality. Text starts with an empty vocab. */
export function defaultFeatureConfig(modality: Modality): FeatureConfig {
  switch (modality) {
    case 'image':
      return { kind: 'image', image: { ...DEFAULT_IMAGE_CONFIG } };
    case 'audio':
      return { kind: 'audio', audio: { ...DEFAULT_AUDIO_CONFIG } };
    case 'motion':
      return { kind: 'motion', motion: { ...DEFAULT_MOTION_CONFIG } };
    case 'text':
      return { kind: 'text', text: { vocab: [] } };
  }
}

/**
 * Options that tune a built feature config beyond what the samples imply. These
 * are the Standard-altitude knobs; every field defaults to the auto value, so an
 * empty options object reproduces the Guided pipeline exactly.
 */
export interface FeatureConfigOptions {
  /** Audio clip length in seconds the spectrogram grid is anchored to. */
  audioSeconds?: number;
  /** Image output side length in pixels. */
  imageSize?: number;
  /** Image channels: 1 grayscale, 3 RGB. */
  imageChannels?: 1 | 3;
  /** Audio output: a mel spectrogram or MFCCs. */
  audioMode?: 'mel' | 'mfcc';
  /** Number of mel bands in the spectrogram analysis. */
  audioBands?: number;
  /** Motion resampled steps per axis. */
  motionSteps?: number;
  /** Maximum text vocabulary size. */
  textVocabCap?: number;
}

/**
 * Builds a feature config for a modality from training samples and the Standard
 * knobs. Text builds its vocabulary from the supplied samples (pass the train
 * split to keep the test set out of it); audio anchors its grid to the clip
 * length; motion fixes a per-axis scale from the data. Unset options fall back to
 * the auto defaults.
 */
export function buildFeatureConfig(
  modality: Modality,
  samples: Sample[],
  options: FeatureConfigOptions = {},
): FeatureConfig {
  if (modality === 'text') {
    const texts = samples
      .map((s) => (s.payload.kind === 'text' ? s.payload.text : ''))
      .filter((t) => t.length > 0);
    return { kind: 'text', text: { vocab: buildVocabulary(texts, options.textVocabCap) } };
  }
  if (modality === 'image') {
    return {
      kind: 'image',
      image: {
        ...DEFAULT_IMAGE_CONFIG,
        size: options.imageSize ?? DEFAULT_IMAGE_CONFIG.size,
        channels: options.imageChannels ?? DEFAULT_IMAGE_CONFIG.channels,
      },
    };
  }
  if (modality === 'audio') {
    const clipSeconds = options.audioSeconds ?? DEFAULT_AUDIO_CONFIG.clipSeconds;
    const mode = options.audioMode ?? DEFAULT_AUDIO_CONFIG.mode;
    const numMel = options.audioBands ?? DEFAULT_AUDIO_CONFIG.numMel;
    return { kind: 'audio', audio: { ...DEFAULT_AUDIO_CONFIG, clipSeconds, mode, numMel } };
  }
  if (modality === 'motion') {
    // Fix a per-axis acceleration scale from the training set: each axis's largest
    // magnitude, floored at a fraction of the overall max so a globally quiet axis
    // is not amplified. Every window is then divided by this fixed reference, not
    // its own range, so an active axis keeps full contrast while a near-still
    // window stays flat near the midpoint.
    const axisCount = DEFAULT_MOTION_CONFIG.axes;
    const axisMax = new Array<number>(axisCount).fill(0);
    for (const s of samples) {
      if (s.payload.kind !== 'motion') continue;
      const d = s.payload.data;
      const ax = s.payload.axes || axisCount;
      for (let i = 0; i < d.length; i++) {
        const a = i % ax;
        if (a < axisCount) {
          const v = Math.abs(d[i]!);
          if (v > axisMax[a]!) axisMax[a] = v;
        }
      }
    }
    const globalMax = Math.max(...axisMax, 1e-6);
    const floor = 0.25 * globalMax;
    const scales = axisMax.map((m) => Math.max(m, floor, 1e-6));
    const targetLen = options.motionSteps ?? DEFAULT_MOTION_CONFIG.targetLen;
    return { kind: 'motion', motion: { ...DEFAULT_MOTION_CONFIG, targetLen, scales } };
  }
  return defaultFeatureConfig(modality);
}

/** The per sample tensor shape a feature config produces, batch aside. */
export function featureShape(config: FeatureConfig): number[] {
  switch (config.kind) {
    case 'image':
      return imageTensorShape(config.image);
    case 'audio':
      return audioTensorShape(config.audio);
    case 'motion':
      return motionTensorShape(config.motion);
    case 'text':
      return textTensorShape(config.text);
  }
}

/**
 * Returns a closure that turns one sample into its flat feature vector for the
 * given config.
 *
 * @throws if the sample payload does not match the config kind.
 */
export function featureExtractor(config: FeatureConfig): (sample: Sample) => Float32Array {
  switch (config.kind) {
    case 'image':
      return (sample) => {
        if (sample.payload.kind !== 'image') throw new Error(`Expected image, got ${sample.payload.kind}`);
        const frame: RgbaFrame = {
          width: sample.payload.width,
          height: sample.payload.height,
          data: sample.payload.data,
        };
        return imageToFeatures(frame, config.image);
      };
    case 'audio':
      return (sample) => {
        if (sample.payload.kind !== 'audio') throw new Error(`Expected audio, got ${sample.payload.kind}`);
        return audioToFeatures(sample.payload.data, sample.payload.sampleRate, config.audio);
      };
    case 'motion':
      return (sample) => {
        if (sample.payload.kind !== 'motion') throw new Error(`Expected motion, got ${sample.payload.kind}`);
        return motionToFeatures(sample.payload.data, sample.payload.axes, config.motion);
      };
    case 'text':
      return (sample) => {
        if (sample.payload.kind !== 'text') throw new Error(`Expected text, got ${sample.payload.kind}`);
        return textToFeatures(sample.payload.text, config.text);
      };
  }
}
