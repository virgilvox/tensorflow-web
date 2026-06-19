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
 * Builds a feature config for a modality from training samples. Only text needs
 * the data (to build its vocabulary); the others return their static default.
 * The vocabulary is built from the supplied samples only, so passing the train
 * split keeps the test set out of the vocabulary.
 */
export function buildFeatureConfig(modality: Modality, samples: Sample[]): FeatureConfig {
  if (modality !== 'text') return defaultFeatureConfig(modality);
  const texts = samples
    .map((s) => (s.payload.kind === 'text' ? s.payload.text : ''))
    .filter((t) => t.length > 0);
  return { kind: 'text', text: { vocab: buildVocabulary(texts) } };
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
