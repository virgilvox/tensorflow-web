/**
 * Pure audio feature extraction. Turns a one second PCM clip into a fixed size
 * log mel spectrogram (or MFCC), treated as a single channel image a small CNN
 * trains on. The grid is padded or truncated to a fixed frame count so the model
 * input shape is constant, and normalized to 0..1 so the float input boundary
 * quantizes cleanly. No Vue, no browser; the DSP lives in lib/dsp.
 */
import { logMelSpectrogram, mfcc, type SpectrogramOptions } from '../../lib/dsp';

/**
 * Fixed log mel range mapped to 0..1. The floor matches the 1e-6 log floor in the
 * spectrogram (a silent frame), and the ceiling is the largest power a full scale
 * frame can produce for the default FFT (|X| <= sum of the window <= fftSize/2, so
 * power <= (fftSize/2)^2). Mapping against this fixed reference, not each clip's
 * own min and max, is deliberate: a per-clip min-max stretches a quiet clip (room
 * noise, a Background Noise sample) to full contrast, so it looks as strong as a
 * spoken keyword and the model predicts the keyword for near-silence. A fixed
 * reference keeps quiet inputs low and loud inputs high, preserving the amplitude
 * difference between a keyword and the background.
 */
const LOG_FLOOR = Math.log(1e-6);
const LOG_CEIL = Math.log(65536);

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
  /**
   * Clip length in seconds the grid is anchored to. Every clip is padded or
   * truncated to this many samples before the spectrogram, and the hop is derived
   * from it, so the same sound always maps to the same grid regardless of the
   * recorded length. This is what keeps training and live inference consistent.
   */
  clipSeconds: number;
}

/** The default audio config: a 32 by 32 log mel grid over a one second clip. */
export const DEFAULT_AUDIO_CONFIG: AudioFeatureConfig = {
  sampleRate: 16000,
  fftSize: 512,
  hopSize: 256,
  numMel: 32,
  mode: 'mel',
  numCoeffs: 13,
  targetFrames: 32,
  clipSeconds: 1,
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
  // Anchor the analysis to a fixed clip length so the time axis is the same for
  // every clip, whatever its recorded duration. Pad short clips with silence and
  // truncate long ones to clipSeconds, then derive a single hop from that fixed
  // sample count. Because the window length and hop depend only on the config,
  // not on this clip's length, the same sound always maps to the same grid, so
  // training and live inference stay consistent and a longer configured clip
  // simply spans more time across the same fixed grid.
  const fixedLen = Math.max(config.fftSize, Math.round(config.clipSeconds * config.sampleRate));
  const win = new Float32Array(fixedLen);
  win.set(pcm.subarray(0, Math.min(pcm.length, fixedLen)));
  const hopSize = Math.max(1, Math.floor((fixedLen - config.fftSize) / Math.max(1, config.targetFrames - 1)));
  const opts: SpectrogramOptions = {
    sampleRate: config.sampleRate,
    fftSize: config.fftSize,
    hopSize,
    numMel: config.numMel,
    fmin: 20,
    fmax: config.sampleRate / 2,
  };
  const spectral = config.mode === 'mfcc' ? mfcc(win, opts, config.numCoeffs) : logMelSpectrogram(win, opts);
  const bands = bandsOf(config);
  const out = new Float32Array(config.targetFrames * bands);

  // Copy up to targetFrames; remaining frames stay at the array's zero fill.
  const copyFrames = Math.min(config.targetFrames, spectral.frames);
  out.set(spectral.data.subarray(0, copyFrames * bands), 0);

  const populated = copyFrames * bands;
  if (config.mode === 'mfcc') {
    // MFCC coefficients are not log energies, so a fixed energy reference does not
    // apply; keep a per-clip range here. The default pipeline is mel, below.
    let min = Infinity;
    let max = -Infinity;
    for (let i = 0; i < populated; i++) {
      const v = out[i]!;
      if (v < min) min = v;
      if (v > max) max = v;
    }
    const range = max - min;
    if (range > 1e-9) {
      for (let i = 0; i < populated; i++) out[i] = (out[i]! - min) / range;
    } else {
      for (let i = 0; i < populated; i++) out[i] = 0;
    }
  } else {
    // Map log mel energies to 0..1 against the fixed reference, not a per-clip
    // min-max. A silent frame sits at LOG_FLOOR and maps to 0; a loud frame maps
    // high. The amplitude difference between a keyword and quiet background is
    // preserved, so the model can tell them apart instead of treating every
    // non-silent input as the keyword.
    const span = LOG_CEIL - LOG_FLOOR;
    for (let i = 0; i < populated; i++) {
      out[i] = Math.min(1, Math.max(0, (out[i]! - LOG_FLOOR) / span));
    }
  }
  // The unpopulated tail (when fewer than targetFrames frames were produced) stays
  // at the array's zero fill, which is the 0 = silence value in this scale.
  return out;
}
