/**
 * Playground smoke. Builds and exports a verified image model, then exercises the
 * standalone Playground two ways: running the current session model live, and
 * reloading a self-contained bundle (.tfwsmodel.json) after a full page reload
 * (when the in-memory session model is gone) and running that. Both must predict
 * a held out image correctly, proving the Playground runs a model with no project
 * and that the bundle carries everything live inference needs.
 *
 * Run from studio/: node scripts/smoke-playground.mjs (needs a local Chrome and
 * network for the tfjs and tfjs-tflite CDN scripts the interpreter loads).
 */
import { spawn } from 'node:child_process';
import { deflateSync } from 'node:zlib';
import { readFileSync } from 'node:fs';
import { chromium } from 'playwright-core';

const root = new URL('..', import.meta.url).pathname;
const SIDE = 64;
const CLASSES = ['top', 'bottom', 'center'];

// ---- image generation (same scheme as the image smoke) ------------------

function crc32(buf) {
  let crc = ~0;
  for (let i = 0; i < buf.length; i++) {
    crc ^= buf[i];
    for (let k = 0; k < 8; k++) crc = (crc >>> 1) ^ (0xedb88320 & -(crc & 1));
  }
  return ~crc >>> 0;
}

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
  ihdr[8] = 8;
  ihdr[9] = 0;
  const raw = Buffer.alloc(side * (side + 1));
  for (let y = 0; y < side; y++) {
    raw[y * (side + 1)] = 0;
    for (let x = 0; x < side; x++) raw[y * (side + 1) + 1 + x] = pixels[y * side + x];
  }
  const idat = deflateSync(raw);
  return Buffer.concat([sig, chunk('IHDR', ihdr), chunk('IDAT', idat), chunk('IEND', Buffer.alloc(0))]);
}

function classImage(classIndex, seed) {
  let s = (seed * 2654435761) >>> 0;
  const rand = () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0), s / 2 ** 32);
  const px = new Uint8Array(SIDE * SIDE);
  const half = SIDE / 2;
  const q = SIDE / 4;
  for (let y = 0; y < SIDE; y++) {
    for (let x = 0; x < SIDE; x++) {
      let on = false;
      if (classIndex === 0) on = y < half;
      else if (classIndex === 1) on = y >= half;
      else on = x >= q && x < SIDE - q && y >= q && y < SIDE - q;
      const base = on ? 235 : 20;
      px[y * SIDE + x] = Math.max(0, Math.min(255, base + (rand() - 0.5) * 30));
    }
  }
  return grayPng(SIDE, px);
}

function batch(classIndex, session, count) {
  return Array.from({ length: count }, (_, i) => ({
    name: `${CLASSES[classIndex]}_${session}_${i}.png`,
    mimeType: 'image/png',
    buffer: classImage(classIndex, classIndex * 1000 + session * 100 + i),
  }));
}

// ---- harness ------------------------------------------------------------

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

async function predictImageInPlayground(page, classIndex) {
  await page.setInputFiles('[data-test="pg-predict-image"]', {
    name: `probe_${classIndex}.png`,
    mimeType: 'image/png',
    buffer: classImage(classIndex, 99999),
  });
  await page.locator('[data-test="top-prediction"]').waitFor({ timeout: 15000 });
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

    // Build and export a verified image model.
    for (const name of CLASSES) {
      await page.fill('.addrow input', name);
      await page.getByRole('button', { name: /Add class/i }).click();
      await page.waitForTimeout(60);
    }
    for (let c = 0; c < CLASSES.length; c++) {
      await page.locator('.selbtn', { hasText: CLASSES[c] }).first().click();
      for (const session of [0, 1]) {
        await page.setInputFiles('input[data-test="import-file"]', batch(c, session, 9));
        await page.waitForTimeout(150);
      }
    }
    await page.getByRole('link', { name: /Train/ }).click();
    await page.getByRole('button', { name: /Train model/i }).click();
    await page.getByRole('button', { name: /Go to Export/i }).waitFor({ timeout: 120000 });
    await page.getByRole('button', { name: /Go to Export/i }).click();
    await page.getByRole('button', { name: /Quantize and verify/i }).click();
    await page.getByText(/parity holds|parity failed/).first().waitFor({ timeout: 120000 });
    const parity = /parity holds/.test((await page.textContent('section')) ?? '');
    check('built a parity verified model to run', parity);

    // Capture the Playground bundle download for the reload test later.
    const [bundleDownload] = await Promise.all([
      page.waitForEvent('download', { timeout: 15000 }),
      page.locator('[data-test="dl-bundle"]').click(),
    ]);
    const bundlePath = await bundleDownload.path();
    const bundleName = bundleDownload.suggestedFilename();
    const bundleBuffer = bundlePath ? readFileSync(bundlePath) : Buffer.alloc(0);
    check(
      'exports a playground bundle',
      /\.tfwsmodel\.json$/.test(bundleName) && bundleBuffer.length > 0,
      `${bundleName}, ${bundleBuffer.length} bytes`,
    );

    // Playground, path 1: run the current session model.
    await page.getByRole('link', { name: /Playground/ }).click();
    await page.locator('[data-test="use-current"]').click();
    await page.getByText('Running', { exact: false }).first().waitFor({ timeout: 8000 });
    const currentTop = await predictImageInPlayground(page, 0);
    check('runs the current model and predicts correctly', currentTop === 'top', `predicted "${currentTop}"`);

    // Reload: the in-memory session model is gone, the project persists.
    await page.reload({ waitUntil: 'networkidle' });
    await page.waitForSelector('.topbar');
    await page.getByRole('link', { name: /Playground/ }).click();
    const useCurrentDisabled = await page.locator('[data-test="use-current"]').isDisabled();
    check('after reload there is no session model to run', useCurrentDisabled);

    // Playground, path 2: load the bundle file and run it standalone.
    await page.setInputFiles('[data-test="load-model-file"]', {
      name: bundleName,
      mimeType: 'application/json',
      buffer: bundleBuffer,
    });
    await page.getByText('Running', { exact: false }).first().waitFor({ timeout: 8000 });
    const bundleTop = await predictImageInPlayground(page, 1);
    check('runs a loaded bundle and predicts correctly', bundleTop === 'bottom', `predicted "${bundleTop}"`);

    const noLoadError = (await page.locator('[data-test="load-error"]').count()) === 0;
    check('bundle loaded without error', noLoadError);

    exitCode = checks.every((c) => c.ok) ? 0 : 1;
  } finally {
    await browser.close();
  }
} catch (err) {
  console.error('Playground smoke failed:', err);
  exitCode = 1;
} finally {
  vite.kill('SIGTERM');
}

console.log(
  `\n${exitCode === 0 ? 'PLAYGROUND SMOKE OK' : 'PLAYGROUND SMOKE FAILED'} (${checks.filter((c) => c.ok).length}/${checks.length} checks)`,
);
process.exit(exitCode);
