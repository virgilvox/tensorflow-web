/**
 * Feature dispatch. Maps a sample to the flat feature vector its modality
 * defines, given that modality's feature config, and reports the tensor shape a
 * config produces. Each modality's extractor lives in its own folder and is a
 * pure function; this module only routes to the right one. Image is implemented
 * for the flagship flow; the other modalities land with their phases and throw a
 * clear error until then.
 */
import type { Modality, Sample } from '../types';
import {
  imageToFeatures,
  imageTensorShape,
  DEFAULT_IMAGE_CONFIG,
  type ImageFeatureConfig,
  type RgbaFrame,
} from './image';

/** The feature config for a modality. Extended as each modality lands. */
export type FeatureConfig = { kind: 'image'; image: ImageFeatureConfig };

/** The default feature config for a modality. */
export function defaultFeatureConfig(modality: Modality): FeatureConfig {
  if (modality === 'image') return { kind: 'image', image: { ...DEFAULT_IMAGE_CONFIG } };
  throw new Error(`Feature config for ${modality} lands with its phase.`);
}

/** The per sample tensor shape a feature config produces, batch aside. */
export function featureShape(config: FeatureConfig): number[] {
  return imageTensorShape(config.image);
}

/**
 * Returns a closure that turns one sample into its flat feature vector for the
 * given config. Closing over the config keeps the hot path allocation free of
 * config lookups.
 *
 * @throws if the sample payload does not match the config kind.
 */
export function featureExtractor(config: FeatureConfig): (sample: Sample) => Float32Array {
  if (config.kind === 'image') {
    return (sample: Sample) => {
      if (sample.payload.kind !== 'image') {
        throw new Error(`Expected an image sample, got ${sample.payload.kind}`);
      }
      const frame: RgbaFrame = {
        width: sample.payload.width,
        height: sample.payload.height,
        data: sample.payload.data,
      };
      return imageToFeatures(frame, config.image);
    };
  }
  throw new Error('Unsupported feature config kind.');
}
