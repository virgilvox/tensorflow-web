/**
 * Phase 2 headless smoke: the flagship image flow end to end, in Guided mode.
 * Starts the Vite dev server, opens the studio in the system Chrome, builds a
 * three class dataset from generated, clearly separable images imported through
 * the real file import path (two capture sessions per class), trains, quantizes,
 * verifies parity in the real TFLite interpreter, checks the device budget, and
 * confirms a live prediction on a held out image.
 *
 * Run from studio/: node scripts/smoke-image.mjs (needs a local Chrome and network
 * for the tfjs and tfjs-tflite CDN scripts the interpreter loads).
 */
import { spawn } from 'node:child_process';
import { deflateSync } from 'node:zlib';
import { chromium } from 'playwright-core';

const root = new URL('..', import.meta.url).pathname;
const SIDE = 64;
const CLASSES = ['top', 'bottom', 'center'];

// ---- image generation ---------------------------------------------------

/** CRC32 for PNG chunks. */
function crc32(buf) {
  let crc = ~0;
  for (let i = 0; i < buf.length; i++) {
    crc ^= buf[i];
    for (let k = 0; k < 8; k++) crc = (crc >>> 1) ^ (0xedb88320 & -(crc & 1));
  }
  return ~crc >>> 0;
}

/** Encodes an 8 bit grayscale image (Uint8Array, length side*side) as a PNG. */
function grayPng(side, pixels) {
  const sig = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
  const chunk = (type, data) => {
    const len = Buffer.alloc(4);
    len.writeUInt32BE(data.length);
    const td = Buffer.concat([Buffer.from(type), data]);
    const crc = Buffer.alloc(4);
    crc.writeUInt32BE(crc32(td));
    return Buffer.concat([len, td, crc]);
  };
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(side, 0);
  ihdr.writeUInt32BE(side, 4);
  ihdr[8] = 8; // bit depth
  ihdr[9] = 0; // color type grayscale
  // each scanline is prefixed with a filter byte (0 = none)
  const raw = Buffer.alloc(side * (side + 1));
  for (let y = 0; y < side; y++) {
    raw[y * (side + 1)] = 0;
    for (let x = 0; x < side; x++) raw[y * (side + 1) + 1 + x] = pixels[y * side + x];
  }
  const idat = deflateSync(raw);
  return Buffer.concat([sig, chunk('IHDR', ihdr), chunk('IDAT', idat), chunk('IEND', Buffer.alloc(0))]);
}

/** A deterministic per class pattern with light noise, as PNG bytes. */
function classImage(classIndex, seed) {
  let s = (seed * 2654435761) >>> 0;
  const rand = () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0), s / 2 ** 32);
  const px = new Uint8Array(SIDE * SIDE);
  const half = SIDE / 2;
  const q = SIDE / 4;
  for (let y = 0; y < SIDE; y++) {
    for (let x = 0; x < SIDE; x++) {
      let on = false;
      if (classIndex === 0) on = y < half; // bright top
      else if (classIndex === 1) on = y >= half; // bright bottom
      else on = x >= q && x < SIDE - q && y >= q && y < SIDE - q; // bright center box
      const base = on ? 235 : 20;
      px[y * SIDE + x] = Math.max(0, Math.min(255, base + (rand() - 0.5) * 30));
    }
  }
  return grayPng(SIDE, px);
}

/** A batch of Playwright file payloads for one class and session. */
function batch(classIndex, session, count) {
  return Array.from({ length: count }, (_, i) => ({
    name: `${CLASSES[classIndex]}_${session}_${i}.png`,
    mimeType: 'image/png',
    buffer: classImage(classIndex, classIndex * 1000 + session * 100 + i),
  }));
}

// ---- harness ------------------------------------------------------------

function startVite() {
  const proc = spawn('npx', ['vite', '--port', '5185', '--strictPort'], {
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

    // Data stage: add three classes and import two sessions of images into each.
    for (const name of CLASSES) {
      await page.fill('.addrow input', name);
      await page.getByRole('button', { name: /Add class/i }).click();
      await page.waitForTimeout(60);
    }

    for (let c = 0; c < CLASSES.length; c++) {
      await page.locator('.classbtn', { hasText: CLASSES[c] }).first().click();
      for (const session of [0, 1]) {
        await page.setInputFiles('input[data-test="import-file"]', batch(c, session, 9));
        await page.waitForTimeout(150);
      }
    }
    const totalText = await page.textContent('.bench');
    check('imported a dataset', /5[0-9]|[6-9][0-9]/.test(totalText ?? ''), 'samples in bench rail');

    // Train stage.
    await page.getByRole('link', { name: /Train/ }).click();
    await page.getByRole('button', { name: /Train model/i }).click();
    await page.getByRole('button', { name: /Go to Export/i }).waitFor({ timeout: 120000 });
    const trainAcc = await page.locator('.gauges').textContent();
    check('training completed', true, trainAcc?.replace(/\s+/g, ' ').trim().slice(0, 60));

    // Export stage: quantize, verify, budget.
    await page.getByRole('button', { name: /Go to Export/i }).click();
    await page.getByRole('button', { name: /Quantize and verify/i }).click();
    await page.getByText(/parity holds|parity failed/).first().waitFor({ timeout: 120000 });

    const exportText = (await page.textContent('section')) ?? '';
    const parity = /parity holds/.test(exportText);
    check('parity holds against the real interpreter', parity);

    const benchText = (await page.textContent('.bench')) ?? '';
    const fits = /fits ESP32 S3/.test(benchText) && !/over ESP32/.test(benchText);
    check('device budget reports a fit for ESP32 S3', fits, benchText.replace(/\s+/g, ' ').slice(-40));

    const int8Match = exportText.match(/int8\s+(\d+)%/);
    const int8Acc = int8Match ? Number(int8Match[1]) : 0;
    check('int8 accuracy learned the task', int8Acc >= 75, `${int8Acc}%`);

    // The verified .tflite actually downloads.
    const [download] = await Promise.all([
      page.waitForEvent('download', { timeout: 15000 }),
      page.getByRole('button', { name: /\.tflite/ }).click(),
    ]);
    const dlPath = await download.path();
    const fs = await import('node:fs');
    const dlSize = dlPath ? fs.statSync(dlPath).size : 0;
    check('downloads a .tflite file', /\.tflite$/.test(download.suggestedFilename()) && dlSize > 0, `${dlSize} bytes`);

    // Test stage: a held out top image should predict "top".
    await page.getByRole('link', { name: /Test/ }).click();
    await page.setInputFiles('[data-test="predict-file"]', {
      name: 'probe_top.png',
      mimeType: 'image/png',
      buffer: classImage(0, 99999),
    });
    await page.locator('[data-test="top-prediction"]').waitFor({ timeout: 15000 });
    const top = await page.locator('[data-test="top-prediction"]').textContent();
    check('live prediction is correct', top?.trim() === 'top', `predicted "${top}"`);

    exitCode = checks.every((c) => c.ok) ? 0 : 1;
  } finally {
    await browser.close();
  }
} catch (err) {
  console.error('Image smoke failed:', err);
  exitCode = 1;
} finally {
  vite.kill('SIGTERM');
}

console.log(
  `\n${exitCode === 0 ? 'IMAGE SMOKE OK' : 'IMAGE SMOKE FAILED'} (${checks.filter((c) => c.ok).length}/${checks.length} checks)`,
);
process.exit(exitCode);
