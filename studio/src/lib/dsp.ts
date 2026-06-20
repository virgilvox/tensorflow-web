/**
 * Digital signal processing for the audio modality. A thin wrapper over a proven
 * FFT (fft.js), plus a hand written, unit tested mel filterbank, log mel
 * spectrogram, and MFCC via a DCT. Pure functions, no Vue, no browser, so the
 * whole chain is testable against known values and can run in a worker. We do
 * not reinvent the FFT; everything else is small and verified.
 */
import FFT from 'fft.js';

/** Options shared by the spectrogram and MFCC extractors. */
export interface SpectrogramOptions {
  sampleRate: number;
  /** FFT size, a power of two. */
  fftSize: number;
  /** Hop between frames in samples. */
  hopSize: number;
  /** Number of mel bands. */
  numMel: number;
  /** Lowest mel frequency in Hz. */
  fmin: number;
  /** Highest mel frequency in Hz; defaults to the Nyquist frequency. */
  fmax: number;
}

/** A Hann window of length n, the standard taper that limits spectral leakage. */
export function hannWindow(n: number): Float32Array {
  const w = new Float32Array(n);
  for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (n - 1));
  return w;
}

/** Converts a frequency in Hz to the mel scale (HTK formula). */
export function hzToMel(hz: number): number {
  return 2595 * Math.log10(1 + hz / 700);
}

/** Converts a mel value back to Hz. */
export function melToHz(mel: number): number {
  return 700 * (10 ** (mel / 2595) - 1);
}

/**
 * Builds a triangular mel filterbank as numMel filters over the fftSize/2+1
 * magnitude bins. Each row is one filter; the filters tile the mel axis evenly
 * and overlap at their half power points, the standard construction.
 *
 * @returns numMel filters, each a Float32Array of length fftSize/2+1.
 */
export function melFilterbank(opts: SpectrogramOptions): Float32Array[] {
  const { sampleRate, fftSize, numMel, fmin, fmax } = opts;
  const bins = fftSize / 2 + 1;
  const melMin = hzToMel(fmin);
  const melMax = hzToMel(fmax);
  // numMel + 2 points define numMel triangles (each spans three points).
  const points = new Array<number>(numMel + 2);
  for (let i = 0; i < points.length; i++) {
    const mel = melMin + ((melMax - melMin) * i) / (numMel + 1);
    // Map the mel point to the nearest FFT bin.
    points[i] = Math.floor(((fftSize + 1) * melToHz(mel)) / sampleRate);
  }
  const filters: Float32Array[] = [];
  for (let m = 1; m <= numMel; m++) {
    const filter = new Float32Array(bins);
    const left = points[m - 1]!;
    const center = points[m]!;
    const right = points[m + 1]!;
    for (let k = left; k < center; k++) {
      if (center !== left) filter[k] = (k - left) / (center - left);
    }
    for (let k = center; k < right; k++) {
      if (right !== center) filter[k] = (right - k) / (right - center);
    }
    filters.push(filter);
  }
  return filters;
}

/** Power spectrum (magnitude squared) of one windowed frame, length fftSize/2+1. */
function powerSpectrum(frame: Float32Array, fft: FFT, scratch: number[]): Float32Array {
  fft.realTransform(scratch, frame as unknown as number[]);
  fft.completeSpectrum(scratch);
  const bins = frame.length / 2 + 1;
  const power = new Float32Array(bins);
  for (let i = 0; i < bins; i++) {
    const re = scratch[2 * i]!;
    const im = scratch[2 * i + 1]!;
    power[i] = re * re + im * im;
  }
  return power;
}

/** The result of a framed spectral transform: row major [frames, bands]. */
export interface SpectralResult {
  data: Float32Array;
  frames: number;
  bands: number;
}

/**
 * Computes a log mel spectrogram: frame the signal with a Hann window, take the
 * power spectrum, apply the mel filterbank, and take the log. Returns row major
 * [frames, numMel].
 *
 * @throws if fftSize is not a power of two.
 */
export function logMelSpectrogram(signal: Float32Array, opts: SpectrogramOptions): SpectralResult {
  const { fftSize, hopSize, numMel } = opts;
  if ((fftSize & (fftSize - 1)) !== 0) throw new Error('fftSize must be a power of two.');
  const fft = new FFT(fftSize);
  const scratch = fft.createComplexArray();
  const window = hannWindow(fftSize);
  const filters = melFilterbank(opts);

  const frames = signal.length >= fftSize ? 1 + Math.floor((signal.length - fftSize) / hopSize) : 0;
  const data = new Float32Array(frames * numMel);
  const frame = new Float32Array(fftSize);

  for (let t = 0; t < frames; t++) {
    const start = t * hopSize;
    for (let i = 0; i < fftSize; i++) frame[i] = (signal[start + i] ?? 0) * window[i]!;
    const power = powerSpectrum(frame, fft, scratch);
    for (let m = 0; m < numMel; m++) {
      const filter = filters[m]!;
      let sum = 0;
      for (let k = 0; k < filter.length; k++) sum += filter[k]! * power[k]!;
      // Log with a small floor so silence maps to a finite, very negative value.
      data[t * numMel + m] = Math.log(sum + 1e-6);
    }
  }
  return { data, frames, bands: numMel };
}

/**
 * Type II DCT of a single vector, returning the first numCoeffs coefficients.
 * Used to turn a log mel frame into MFCCs, which decorrelate the bands.
 */
export function dct2(input: Float32Array, numCoeffs: number): Float32Array {
  const n = input.length;
  const out = new Float32Array(numCoeffs);
  for (let k = 0; k < numCoeffs; k++) {
    let sum = 0;
    for (let i = 0; i < n; i++) sum += input[i]! * Math.cos((Math.PI * k * (2 * i + 1)) / (2 * n));
    out[k] = sum;
  }
  return out;
}

/**
 * Computes MFCCs: a log mel spectrogram followed by a per frame DCT, keeping the
 * first numCoeffs coefficients. Returns row major [frames, numCoeffs].
 */
export function mfcc(signal: Float32Array, opts: SpectrogramOptions, numCoeffs: number): SpectralResult {
  const mel = logMelSpectrogram(signal, opts);
  const out = new Float32Array(mel.frames * numCoeffs);
  const frameBuf = new Float32Array(opts.numMel);
  for (let t = 0; t < mel.frames; t++) {
    for (let m = 0; m < opts.numMel; m++) frameBuf[m] = mel.data[t * opts.numMel + m]!;
    const coeffs = dct2(frameBuf, numCoeffs);
    out.set(coeffs, t * numCoeffs);
  }
  return { data: out, frames: mel.frames, bands: numCoeffs };
}
