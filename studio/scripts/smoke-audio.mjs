/**
 * Phase 3 headless smoke: the audio keyword spotting flow, in Guided mode.
 * Generates three classes of distinct one second tones as WAV files, imports
 * them through the real audio import path (two sessions per class), trains the
 * spectrogram CNN, verifies parity in the real interpreter, and confirms a live
 * prediction on a held out clip. A real microphone is not available headless, so
 * the smoke uses the file import path; the DSP itself is covered by unit tests.
 *
 * Run from studio/: node scripts/smoke-audio.mjs (needs a local Chrome and the
 * tfjs and tfjs-tflite CDN scripts).
 */
import { spawn } from 'node:child_process';
import { chromium } from 'playwright-core';

const root = new URL('..', import.meta.url).pathname;
const RATE = 16000;
const CLASSES = [
  { name: 'low', freq: 350 },
  { name: 'mid', freq: 1200 },
  { name: 'high', freq: 3500 },
];

/** Encodes mono Float32 PCM as a 16 bit WAV buffer. */
function encodeWav(samples, rate) {
  const dataLen = samples.length * 2;
  const buf = Buffer.alloc(44 + dataLen);
  buf.write('RIFF', 0);
  buf.writeUInt32LE(36 + dataLen, 4);
  buf.write('WAVE', 8);
  buf.write('fmt ', 12);
  buf.writeUInt32LE(16, 16);
  buf.writeUInt16LE(1, 20); // PCM
  buf.writeUInt16LE(1, 22); // mono
  buf.writeUInt32LE(rate, 24);
  buf.writeUInt32LE(rate * 2, 28);
  buf.writeUInt16LE(2, 32);
  buf.writeUInt16LE(16, 34);
  buf.write('data', 36);
  buf.writeUInt32LE(dataLen, 40);
  for (let i = 0; i < samples.length; i++) {
    const v = Math.max(-1, Math.min(1, samples[i]));
    buf.writeInt16LE(Math.round(v * 32767), 44 + i * 2);
  }
  return buf;
}

/** A one second tone for a class with light noise and per sample variation. */
function tone(freq, seed) {
  let s = (seed * 2654435761) >>> 0;
  const rand = () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0), s / 2 ** 32);
  const amp = 0.6 + rand() * 0.3;
  const phase = rand() * Math.PI;
  const n = RATE;
  const x = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    x[i] = amp * Math.sin((2 * Math.PI * freq * i) / RATE + phase) + (rand() - 0.5) * 0.05;
  }
  return x;
}

function batch(classIndex, session, count) {
  const { name, freq } = CLASSES[classIndex];
  return Array.from({ length: count }, (_, i) => ({
    name: `${name}_${session}_${i}.wav`,
    mimeType: 'audio/wav',
    buffer: encodeWav(tone(freq, classIndex * 1000 + session * 100 + i), RATE),
  }));
}

function startVite() {
  const proc = spawn('npx', ['vite', '--port', '5186', '--strictPort'], {
    cwd: root,
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  const url = new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('Vite did not start within 40s')), 40000);
    const onData = (chunk) => {
      const m = chunk.toString().match(/localhost:(\d+)/);
      if (m) {
        clearTimeout(timer);
        resolve(`http://localhost:${m[1]}`);
      }
    };
    proc.stdout.on('data', onData);
    proc.stderr.on('data', onData);
  });
  return { proc, url };
}

const checks = [];
function check(name, ok, detail = '') {
  checks.push({ name, ok });
  console.log(`  ${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? `  (${detail})` : ''}`);
}

const { proc: vite, url: urlPromise } = startVite();
let exitCode = 0;

try {
  const base = await urlPromise;
  const browser = await chromium.launch({ channel: 'chrome', headless: true });
  try {
    const page = await browser.newPage();
    page.on('pageerror', (e) => console.log('[pageerror]', e.message));
    page.on('console', (m) => {
      if (m.type() === 'error') console.log('[console]', m.text());
    });

    await page.goto(base, { waitUntil: 'networkidle' });
    await page.waitForSelector('.topbar');

    // Switch to the audio modality.
    await page.locator('.modcard', { hasText: 'Audio' }).click();
    await page.waitForTimeout(80);

    // Add three keyword classes.
    for (const c of CLASSES) {
      await page.fill('.addrow input', c.name);
      await page.getByRole('button', { name: /Add class/i }).click();
      await page.waitForTimeout(60);
    }

    // Import two sessions of clips into each class. Decoding is async, so wait
    // for the bench rail sample count to reach the full set before continuing.
    const perBatch = 8;
    const expected = CLASSES.length * 2 * perBatch;
    for (let c = 0; c < CLASSES.length; c++) {
      await page.locator('.selbtn', { hasText: CLASSES[c].name }).first().click();
      for (const session of [0, 1]) {
        await page.setInputFiles('input[data-test="import-audio"]', batch(c, session, perBatch));
        await page.waitForTimeout(120);
      }
    }
    await page.waitForFunction(
      (n) => {
        const t = document.querySelector('.bench')?.textContent ?? '';
        const m = t.match(/Samples\s*(\d+)/);
        return !!m && Number(m[1]) >= n;
      },
      expected,
      { timeout: 60000 },
    );
    check('imported an audio dataset', true, `${expected} clips`);

    // Train.
    await page.getByRole('link', { name: /Train/ }).click();
    await page.getByRole('button', { name: /Train model/i }).click();
    await page.getByRole('button', { name: /Go to Export/i }).waitFor({ timeout: 180000 });
    check('training completed', true);

    // Export and verify.
    await page.getByRole('button', { name: /Go to Export/i }).click();
    await page.getByRole('button', { name: /Quantize and verify/i }).click();
    await page.getByText(/parity holds|parity failed/).first().waitFor({ timeout: 120000 });
    const exportText = (await page.textContent('section')) ?? '';
    check('parity holds against the real interpreter', /parity holds/.test(exportText));
    const int8 = Number(exportText.match(/int8\s+(\d+)%/)?.[1] ?? 0);
    check('int8 accuracy learned the keywords', int8 >= 75, `${int8}%`);
    const fitBench = (await page.textContent('.bench')) ?? '';
    check('device budget reports a fit', /fits/.test(fitBench));

    // Live: a held out low tone should predict "low".
    await page.getByRole('link', { name: /Test/ }).click();
    await page.setInputFiles('[data-test="predict-audio-file"]', {
      name: 'probe_low.wav',
      mimeType: 'audio/wav',
      buffer: encodeWav(tone(CLASSES[0].freq, 88888), RATE),
    });
    await page.locator('[data-test="top-prediction"]').waitFor({ timeout: 15000 });
    const top = await page.locator('[data-test="top-prediction"]').textContent();
    check('live prediction is correct', top?.trim() === 'low', `predicted "${top}"`);

    exitCode = checks.every((c) => c.ok) ? 0 : 1;
  } finally {
    await browser.close();
  }
} catch (err) {
  console.error('Audio smoke failed:', err);
  exitCode = 1;
} finally {
  vite.kill('SIGTERM');
}

console.log(
  `\n${exitCode === 0 ? 'AUDIO SMOKE OK' : 'AUDIO SMOKE FAILED'} (${checks.filter((c) => c.ok).length}/${checks.length} checks)`,
);
process.exit(exitCode);
