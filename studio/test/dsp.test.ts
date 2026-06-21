import { describe, it, expect } from 'vitest';
import {
  hzToMel,
  melToHz,
  hannWindow,
  melFilterbank,
  logMelSpectrogram,
  dct2,
  mfcc,
  type SpectrogramOptions,
} from '../src/lib/dsp';

const OPTS: SpectrogramOptions = {
  sampleRate: 16000,
  fftSize: 512,
  hopSize: 256,
  numMel: 32,
  fmin: 20,
  fmax: 8000,
};

/** A pure sine of the given frequency, n samples at the option's rate. */
function sine(freq: number, n: number, rate: number): Float32Array {
  const x = new Float32Array(n);
  for (let i = 0; i < n; i++) x[i] = Math.sin((2 * Math.PI * freq * i) / rate);
  return x;
}

describe('mel scale', () => {
  it('round trips Hz through mel', () => {
    expect(melToHz(hzToMel(1000))).toBeCloseTo(1000, 3);
    expect(hzToMel(0)).toBe(0);
  });
});

describe('hannWindow', () => {
  it('is zero at the ends and one in the middle', () => {
    const w = hannWindow(9);
    expect(w[0]).toBeCloseTo(0, 6);
    expect(w[8]).toBeCloseTo(0, 6);
    expect(w[4]).toBeCloseTo(1, 6);
  });
});

describe('melFilterbank', () => {
  it('builds the right number of non negative filters over the bins', () => {
    const filters = melFilterbank(OPTS);
    expect(filters.length).toBe(32);
    for (const f of filters) {
      expect(f.length).toBe(OPTS.fftSize / 2 + 1);
      expect(Math.min(...f)).toBeGreaterThanOrEqual(0);
    }
    // The bank as a whole covers energy: total weight is positive.
    const total = filters.reduce((s, f) => s + f.reduce((a, b) => a + b, 0), 0);
    expect(total).toBeGreaterThan(0);
  });

  it('has no collapsed all-zero filter at the default config', () => {
    // Invariant: at the studio's 32 bands / 512 FFT / 20 Hz config the mel points
    // are already strictly increasing, so every filter carries weight. This guards
    // against a future config change that would round adjacent points to the same
    // bin and leave a band a constant for every clip.
    const filters = melFilterbank(OPTS);
    for (const f of filters) {
      expect(f.reduce((a, b) => a + b, 0)).toBeGreaterThan(0);
    }
  });
});

describe('dct2', () => {
  it('puts all energy of a constant signal in coefficient zero', () => {
    const input = new Float32Array(8).fill(2);
    const out = dct2(input, 4);
    expect(out[0]).toBeCloseTo(16, 5); // sum of 2 over 8 samples
    expect(out[1]).toBeCloseTo(0, 5);
    expect(out[2]).toBeCloseTo(0, 5);
    expect(out[3]).toBeCloseTo(0, 5);
  });
});

describe('logMelSpectrogram', () => {
  it('produces the expected frame and band counts', () => {
    const s = sine(1000, 16000, OPTS.sampleRate);
    const result = logMelSpectrogram(s, OPTS);
    const expectedFrames = 1 + Math.floor((16000 - OPTS.fftSize) / OPTS.hopSize);
    expect(result.frames).toBe(expectedFrames);
    expect(result.bands).toBe(32);
    expect(result.data.length).toBe(expectedFrames * 32);
  });

  it('shows more energy for a tone than for silence', () => {
    const tone = logMelSpectrogram(sine(1000, 8192, OPTS.sampleRate), OPTS);
    const silence = logMelSpectrogram(new Float32Array(8192), OPTS);
    const maxOf = (a: Float32Array) => a.reduce((m, v) => Math.max(m, v), -Infinity);
    expect(maxOf(tone.data)).toBeGreaterThan(maxOf(silence.data));
  });
});

describe('mfcc', () => {
  it('returns the requested coefficient count per frame', () => {
    const result = mfcc(sine(440, 8192, OPTS.sampleRate), OPTS, 13);
    expect(result.bands).toBe(13);
    expect(result.data.length).toBe(result.frames * 13);
  });
});
