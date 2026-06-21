/**
 * The self-contained model bundle: a single JSON file that carries everything
 * the Playground needs to run a trained model live, with no project open. It
 * holds the .tflite bytes (base64) alongside the modality, the class names, the
 * feature config, and the input shape, so a studio export can be reloaded and
 * run on fresh camera, microphone, motion, or text input directly. The raw
 * .tflite stays the artifact you flash; this bundle is the artifact you replay.
 */
import type { FeatureConfig } from '../features';
import type { Modality } from '../types';

const MODALITIES: ReadonlySet<Modality> = new Set<Modality>(['image', 'audio', 'motion', 'text']);

/** The on disk bundle shape. Versioned so a future format can migrate. */
export interface ModelBundle {
  format: 'tfwebstudio-model';
  version: 1;
  name: string;
  modality: Modality;
  classes: string[];
  featureConfig: FeatureConfig;
  inputShape: number[];
  /** The microphone clip length the model was trained at, for audio live capture. */
  audioSeconds?: number;
  /** The .tflite bytes, base64 encoded. */
  tfliteBase64: string;
}

/** A parsed bundle with the bytes decoded back to a typed array. */
export interface ParsedBundle {
  name: string;
  modality: Modality;
  classes: string[];
  featureConfig: FeatureConfig;
  inputShape: number[];
  audioSeconds?: number;
  bytes: Uint8Array;
}

/** Encodes bytes to a base64 string in chunks, avoiding a huge call stack. */
function bytesToBase64(bytes: Uint8Array): string {
  let binary = '';
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}

/** Decodes a base64 string back to bytes. */
function base64ToBytes(b64: string): Uint8Array {
  const binary = atob(b64);
  const out = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) out[i] = binary.charCodeAt(i);
  return out;
}

/** The error thrown for any file that is not a valid model bundle. */
function invalid(reason: string): never {
  throw new Error(`Not a recognized TF Web Studio model bundle: ${reason}.`);
}

/**
 * Serializes a trained model and its run config into a bundle JSON string.
 *
 * @returns the bundle as a pretty printed JSON string.
 */
export function encodeBundle(input: {
  name: string;
  modality: Modality;
  classes: string[];
  featureConfig: FeatureConfig;
  inputShape: number[];
  audioSeconds?: number;
  bytes: Uint8Array;
}): string {
  const bundle: ModelBundle = {
    format: 'tfwebstudio-model',
    version: 1,
    name: input.name,
    modality: input.modality,
    classes: input.classes,
    featureConfig: input.featureConfig,
    inputShape: input.inputShape,
    ...(input.audioSeconds !== undefined ? { audioSeconds: input.audioSeconds } : {}),
    tfliteBase64: bytesToBase64(input.bytes),
  };
  return JSON.stringify(bundle, null, 2);
}

/**
 * Parses and validates a bundle, decoding the .tflite bytes.
 *
 * @throws if the value is not a valid bundle (wrong format, version, modality,
 *   feature config kind, or missing bytes).
 */
export function parseBundle(value: unknown): ParsedBundle {
  if (!value || typeof value !== 'object') invalid('not an object');
  const b = value as Record<string, unknown>;
  if (b.format !== 'tfwebstudio-model') invalid('missing format marker');
  if (b.version !== 1) invalid(`unsupported version ${String(b.version)}`);
  if (typeof b.modality !== 'string' || !MODALITIES.has(b.modality as Modality)) {
    invalid(`unknown modality ${String(b.modality)}`);
  }
  const modality = b.modality as Modality;
  if (!Array.isArray(b.classes) || !b.classes.every((c) => typeof c === 'string')) {
    invalid('classes must be a string array');
  }
  if (!b.featureConfig || typeof b.featureConfig !== 'object') invalid('missing feature config');
  const fc = b.featureConfig as Record<string, unknown>;
  if (fc.kind !== modality) invalid('feature config does not match the modality');
  // Validate the sub-config the matching extractor will read, so a malformed
  // bundle fails here at load rather than deep in feature extraction at predict.
  if (modality === 'text') {
    const text = fc.text as { vocab?: unknown } | undefined;
    if (!text || !Array.isArray(text.vocab)) invalid('text config needs a vocabulary');
  } else if (!fc[modality] || typeof fc[modality] !== 'object') {
    invalid(`${modality} config is missing`);
  }
  if (
    !Array.isArray(b.inputShape) ||
    b.inputShape.length === 0 ||
    !b.inputShape.every((n) => typeof n === 'number' && n > 0)
  ) {
    invalid('inputShape must be a non-empty array of positive numbers');
  }
  if (typeof b.tfliteBase64 !== 'string' || b.tfliteBase64.length === 0) {
    invalid('missing model bytes');
  }
  if (b.audioSeconds !== undefined && (typeof b.audioSeconds !== 'number' || !(b.audioSeconds > 0))) {
    invalid('audioSeconds must be a positive number');
  }
  return {
    name: typeof b.name === 'string' ? b.name : 'model',
    modality,
    classes: b.classes as string[],
    featureConfig: b.featureConfig as FeatureConfig,
    inputShape: b.inputShape as number[],
    audioSeconds: b.audioSeconds as number | undefined,
    bytes: base64ToBytes(b.tfliteBase64 as string),
  };
}
