/**
 * Pure audio feature extraction. Turns a one second PCM clip into a fixed size
 * log mel spectrogram (or MFCC), treated as a single channel image a small CNN
 * trains on. The grid is padded or truncated to a fixed frame count so the model
 * input shape is constant, and normalized to 0..1 so the float input boundary
 * quantizes cleanly. No Vue, no browser; the DSP lives in lib/dsp.
 */
import { logMelSpectrogram, mfcc, type SpectrogramOptions } from '../../lib/dsp';

/** The audio preprocessing config. Mel for general sounds, MFCC for keywords. */
export interface AudioFeatureConfig {
  sampleRate: number;
  fftSize: number;
  hopSize: number;
  numMel: number;
  /** mel for a spectrogram, mfcc for cepstral coefficients. */
  mode: 'mel' | 'mfcc';
  /** Number of MFCCs kept when mode is mfcc. */
  numCoeffs: number;
  /** Fixed number of time frames the grid is padded or truncated to. */
  targetFrames: number;
}

/** The default audio config: a 32 by 32 log mel grid at 16 kHz. */
export const DEFAULT_AUDIO_CONFIG: AudioFeatureConfig = {
  sampleRate: 16000,
  fftSize: 512,
  hopSize: 256,
  numMel: 32,
  mode: 'mel',
  numCoeffs: 13,
  targetFrames: 32,
};

/** Number of feature bands a config yields per frame. */
function bandsOf(config: AudioFeatureConfig): number {
  return config.mode === 'mfcc' ? config.numCoeffs : config.numMel;
}

/** The tensor shape [frames, bands, 1] a config produces, batch aside. */
export function audioTensorShape(config: AudioFeatureConfig): [number, number, number] {
  return [config.targetFrames, bandsOf(config), 1];
}

/** Linearly resamples a signal to a target sample rate, if it differs. */
function resample(signal: Float32Array, from: number, to: number): Float32Array {
  if (from === to) return signal;
  const ratio = to / from;
  const out = new Float32Array(Math.round(signal.length * ratio));
  for (let i = 0; i < out.length; i++) {
    const src = i / ratio;
    const i0 = Math.floor(src);
    const i1 = Math.min(signal.length - 1, i0 + 1);
    const w = src - i0;
    out[i] = (signal[i0] ?? 0) * (1 - w) + (signal[i1] ?? 0) * w;
  }
  return out;
}

/**
 * Extracts the model input from a PCM clip: resample to the configured rate,
 * compute the log mel spectrogram or MFCC, fix the frame count, and normalize
 * the whole grid to 0..1.
 *
 * @param samples mono PCM samples.
 * @param sampleRate the rate the samples were captured at.
 * @param config the audio feature config.
 * @returns a Float32Array of length targetFrames * bands.
 */
export function audioToFeatures(
  samples: Float32Array,
  sampleRate: number,
  config: AudioFeatureConfig,
): Float32Array {
  const pcm = resample(samples, sampleRate, config.sampleRate);
  const opts: SpectrogramOptions = {
    sampleRate: config.sampleRate,
    fftSize: config.fftSize,
    hopSize: config.hopSize,
    numMel: config.numMel,
    fmin: 20,
    fmax: config.sampleRate / 2,
  };
  const spectral = config.mode === 'mfcc' ? mfcc(pcm, opts, config.numCoeffs) : logMelSpectrogram(pcm, opts);
  const bands = bandsOf(config);
  const out = new Float32Array(config.targetFrames * bands);

  // Copy up to targetFrames; remaining frames stay at the array's zero fill.
  const copyFrames = Math.min(config.targetFrames, spectral.frames);
  out.set(spectral.data.subarray(0, copyFrames * bands), 0);

  // Normalize the populated region to 0..1 so the int8 input boundary is stable.
  let min = Infinity;
  let max = -Infinity;
  const populated = copyFrames * bands;
  for (let i = 0; i < populated; i++) {
    const v = out[i]!;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const range = max - min;
  if (range > 1e-9) {
    for (let i = 0; i < populated; i++) out[i] = (out[i]! - min) / range;
  }
  return out;
}
