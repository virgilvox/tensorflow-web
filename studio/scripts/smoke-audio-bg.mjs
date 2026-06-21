/**
 * Audio energy-discrimination smoke. Reproduces the real failure: a loud keyword
 * class ("hello") and a quiet "Background Noise" class. It trains, verifies parity,
 * then probes with a loud keyword clip (must predict "hello") AND a quiet clip
 * (must predict "Background Noise", NOT the keyword). With per-clip min-max
 * normalization the quiet clip was stretched to full contrast and predicted the
 * keyword; this smoke guards the fixed-reference scaling that fixes that, so a
 * keyword model can no longer call near-silence the keyword.
 *
 * Run from studio/: node scripts/smoke-audio-bg.mjs (needs a local Chrome and the
 * tfjs and tfjs-tflite CDN scripts).
 */
import { spawn } from 'node:child_process';
import { chromium } from 'playwright-core';

const root = new URL('..', import.meta.url).pathname;
const RATE = 16000;
const KEYWORD = 'hello';
const BACKGROUND = 'Background Noise';

function encodeWav(samples, rate) {
  const dataLen = samples.length * 2;
  const buf = Buffer.alloc(44 + dataLen);
  buf.write('RIFF', 0);
  buf.writeUInt32LE(36 + dataLen, 4);
  buf.write('WAVE', 8);
  buf.write('fmt ', 12);
  buf.writeUInt32LE(16, 16);
  buf.writeUInt16LE(1, 20);
  buf.writeUInt16LE(1, 22);
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

function rng(seed) {
  let s = (seed * 2654435761) >>> 0;
  return () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0), s / 2 ** 32);
}

/** A loud spoken-keyword stand-in: a couple of strong formant tones, full amplitude. */
function keyword(seed) {
  const rand = rng(seed);
  const amp = 0.7 + rand() * 0.2;
  const phase = rand() * Math.PI;
  const x = new Float32Array(RATE);
  for (let i = 0; i < RATE; i++) {
    const t = i / RATE;
    x[i] =
      amp * (0.6 * Math.sin(2 * Math.PI * 700 * t + phase) + 0.4 * Math.sin(2 * Math.PI * 1800 * t)) +
      (rand() - 0.5) * 0.05;
  }
  return x;
}

/** Quiet background: low amplitude broadband noise, the rest state. */
function background(seed) {
  const rand = rng(seed);
  const amp = 0.015 + rand() * 0.015;
  const x = new Float32Array(RATE);
  for (let i = 0; i < RATE; i++) x[i] = (rand() - 0.5) * 2 * amp;
  return x;
}

function batch(label, make, session, count, base) {
  return Array.from({ length: count }, (_, i) => ({
    name: `${label.replace(/\s+/g, '_')}_${session}_${i}.wav`,
    mimeType: 'audio/wav',
    buffer: encodeWav(make(base + session * 100 + i), RATE),
  }));
}

function startVite() {
  const proc = spawn('npx', ['vite', '--port', '5188', '--strictPort'], {
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

async function probe(page, make, seed) {
  await page.setInputFiles('[data-test="predict-audio-file"]', {
    name: 'probe.wav',
    mimeType: 'audio/wav',
    buffer: encodeWav(make(seed), RATE),
  });
  await page.locator('[data-test="top-prediction"]').waitFor({ timeout: 15000 });
  // Re-read until it reflects this probe (the bars update in place).
  await page.waitForTimeout(300);
  return (await page.locator('[data-test="top-prediction"]').textContent())?.trim();
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

    await page.locator('.modcard', { hasText: 'Audio' }).click();
    await page.waitForTimeout(80);

    for (const name of [KEYWORD, BACKGROUND]) {
      await page.fill('.addrow input', name);
      await page.getByRole('button', { name: /Add class/i }).click();
      await page.waitForTimeout(60);
    }

    const perBatch = 10;
    const expected = 2 * 2 * perBatch;
    const plan = [
      { name: KEYWORD, make: keyword, base: 1000 },
      { name: BACKGROUND, make: background, base: 5000 },
    ];
    for (const p of plan) {
      await page.locator('.selbtn', { hasText: p.name }).first().click();
      for (const session of [0, 1]) {
        await page.setInputFiles('input[data-test="import-audio"]', batch(p.name, p.make, session, perBatch, p.base));
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
    check('imported a loud keyword and a quiet background class', true, `${expected} clips`);

    await page.getByRole('link', { name: /Train/ }).click();
    await page.getByRole('button', { name: /Train model/i }).click();
    await page.getByRole('button', { name: /Go to Export/i }).waitFor({ timeout: 180000 });

    await page.getByRole('button', { name: /Go to Export/i }).click();
    await page.getByRole('button', { name: /Quantize and verify/i }).click();
    await page.getByText(/parity holds|parity failed/).first().waitFor({ timeout: 120000 });
    const exportText = (await page.textContent('section')) ?? '';
    check('parity holds against the real interpreter', /parity holds/.test(exportText));
    const int8 = Number(exportText.match(/int8\s+(\d+)%/)?.[1] ?? 0);
    check('int8 accuracy separated keyword from background', int8 >= 75, `${int8}%`);

    await page.getByRole('link', { name: /Test/ }).click();

    const loudTop = await probe(page, keyword, 91111);
    check('a loud keyword clip predicts the keyword', loudTop === KEYWORD, `predicted "${loudTop}"`);

    const quietTop = await probe(page, background, 92222);
    check(
      'a quiet clip predicts background, not the keyword',
      quietTop === BACKGROUND && quietTop !== KEYWORD,
      `predicted "${quietTop}"`,
    );

    exitCode = checks.every((c) => c.ok) ? 0 : 1;
  } finally {
    await browser.close();
  }
} catch (err) {
  console.error('Audio background smoke failed:', err);
  exitCode = 1;
} finally {
  vite.kill('SIGTERM');
}

console.log(
  `\n${exitCode === 0 ? 'AUDIO BG SMOKE OK' : 'AUDIO BG SMOKE FAILED'} (${checks.filter((c) => c.ok).length}/${checks.length} checks)`,
);
process.exit(exitCode);
