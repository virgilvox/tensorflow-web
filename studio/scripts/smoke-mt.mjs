/**
 * Phase 4 headless smoke: the motion and text flows, in Guided mode. One Vite
 * server, two isolated browser contexts (so IndexedDB does not carry between the
 * two projects). Text is driven entirely through the typed example UI; motion
 * uses the JSON window import path, since no sensor is available headless. Each
 * flow trains, verifies parity in the real interpreter, and confirms a live
 * prediction.
 *
 * Run from studio/: node scripts/smoke-mt.mjs (needs a local Chrome and the tfjs
 * and tfjs-tflite CDN scripts).
 */
import { spawn } from 'node:child_process';
import { chromium } from 'playwright-core';

const root = new URL('..', import.meta.url).pathname;

// ---- generators ---------------------------------------------------------

// Each class keeps strong, repeated class tokens so a bag of words model
// generalizes to the held out split rather than memorizing unique phrases.
const TEXT_CLASSES = {
  greeting: [
    'hello there', 'hi there friend', 'hey there', 'hello friend', 'hi how are you',
    'hey how are you', 'hello good morning', 'hi there hello', 'hey hello friend', 'hello hello there',
  ],
  weather: [
    'is it raining', 'rain today', 'sunny outside', 'very sunny day', 'cold and windy',
    'cold rain today', 'will it snow', 'snow tonight', 'cloudy and cold', 'hot and sunny',
  ],
  farewell: [
    'goodbye now', 'goodbye friend', 'bye for now', 'goodbye see you later', 'later friend bye',
    'farewell now', 'farewell my friend', 'good night bye', 'goodbye good night', 'take care bye now',
  ],
};

const MOTION_CLASSES = ['shake', 'wave', 'tap'];

/** A distinct motion window per class as a JSON file buffer. */
function motionFile(classIndex, seed) {
  let s = (seed * 2654435761) >>> 0;
  const rand = () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0), s / 2 ** 32);
  const steps = 100;
  const data = [];
  for (let t = 0; t < steps; t++) {
    const p = t / steps;
    let x = 0;
    let y = 0;
    let z = 0;
    if (classIndex === 0) {
      x = Math.sin(2 * Math.PI * 8 * p); // fast shake on x
    } else if (classIndex === 1) {
      x = Math.sin(2 * Math.PI * 2 * p); // slow wave, x and y in quadrature
      y = Math.sin(2 * Math.PI * 2 * p + Math.PI / 2);
    } else {
      x = t < 12 ? 1 : 0; // a tap spike then rest
      z = t < 12 ? 1 : 0;
    }
    data.push(x + (rand() - 0.5) * 0.08, y + (rand() - 0.5) * 0.08, z + (rand() - 0.5) * 0.08);
  }
  return {
    name: `${MOTION_CLASSES[classIndex]}_${seed}.json`,
    mimeType: 'application/json',
    buffer: Buffer.from(JSON.stringify({ hz: 50, axes: 3, data })),
  };
}

// ---- harness ------------------------------------------------------------

function startVite() {
  const proc = spawn('npx', ['vite', '--port', '5187', '--strictPort'], {
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

async function waitSamples(page, n) {
  await page.waitForFunction(
    (need) => {
      const t = document.querySelector('.bench')?.textContent ?? '';
      const m = t.match(/Samples\s*(\d+)/);
      return !!m && Number(m[1]) >= need;
    },
    n,
    { timeout: 60000 },
  );
}

async function trainExportVerify(page, label) {
  await page.getByRole('link', { name: /Train/ }).click();
  await page.getByRole('button', { name: /Train model/i }).click();
  await page.getByRole('button', { name: /Go to Export/i }).waitFor({ timeout: 180000 });
  check(`${label}: training completed`, true);

  await page.getByRole('button', { name: /Go to Export/i }).click();
  await page.getByRole('button', { name: /Quantize and verify/i }).click();
  await page.getByText(/parity holds|parity failed/).first().waitFor({ timeout: 120000 });
  const text = (await page.textContent('section')) ?? '';
  check(`${label}: parity holds`, /parity holds/.test(text));
  const int8 = Number(text.match(/int8\s+(\d+)%/)?.[1] ?? 0);
  check(`${label}: int8 accuracy learned the task`, int8 >= 75, `${int8}%`);
}

const { proc: vite, url: urlPromise } = startVite();
let exitCode = 0;

try {
  const base = await urlPromise;
  const browser = await chromium.launch({ channel: 'chrome', headless: true });
  try {
    // ---- TEXT ----
    {
      const ctx = await browser.newContext();
      const page = await ctx.newPage();
      page.on('pageerror', (e) => console.log('[text pageerror]', e.message));
      await page.goto(base, { waitUntil: 'networkidle' });
      await page.waitForSelector('.topbar');
      await page.locator('.modcard', { hasText: 'Text' }).click();

      for (const name of Object.keys(TEXT_CLASSES)) {
        await page.fill('.addrow input', name);
        await page.getByRole('button', { name: /Add class/i }).click();
        await page.waitForTimeout(50);
      }
      let total = 0;
      for (const [name, examples] of Object.entries(TEXT_CLASSES)) {
        await page.locator('.classbtn', { hasText: name }).first().click();
        for (const ex of examples) {
          await page.fill('[data-test="text-input"]', ex);
          await page.getByRole('button', { name: /Add example/i }).click();
          total += 1;
          await page.waitForTimeout(30);
        }
      }
      await waitSamples(page, total);
      check('text: typed a dataset', true, `${total} examples`);

      await trainExportVerify(page, 'text');

      await page.getByRole('link', { name: /Test/ }).click();
      await page.fill('[data-test="text-predict-input"]', 'hello there my good friend');
      await page.getByRole('button', { name: /^Predict$/ }).click();
      await page.locator('[data-test="top-prediction"]').waitFor({ timeout: 15000 });
      const top = await page.locator('[data-test="top-prediction"]').textContent();
      check('text: live prediction is correct', top?.trim() === 'greeting', `predicted "${top}"`);
      await ctx.close();
    }

    // ---- MOTION ----
    {
      const ctx = await browser.newContext();
      const page = await ctx.newPage();
      page.on('pageerror', (e) => console.log('[motion pageerror]', e.message));
      await page.goto(base, { waitUntil: 'networkidle' });
      await page.waitForSelector('.topbar');
      await page.locator('.modcard', { hasText: 'Motion' }).click();

      for (const name of MOTION_CLASSES) {
        await page.fill('.addrow input', name);
        await page.getByRole('button', { name: /Add class/i }).click();
        await page.waitForTimeout(50);
      }
      const perBatch = 6;
      const expected = MOTION_CLASSES.length * 2 * perBatch;
      for (let c = 0; c < MOTION_CLASSES.length; c++) {
        await page.locator('.classbtn', { hasText: MOTION_CLASSES[c] }).first().click();
        for (const session of [0, 1]) {
          const files = Array.from({ length: perBatch }, (_, i) => motionFile(c, c * 1000 + session * 100 + i));
          await page.setInputFiles('input[data-test="import-motion"]', files);
          await page.waitForTimeout(120);
        }
      }
      await waitSamples(page, expected);
      check('motion: imported windows', true, `${expected} windows`);

      await trainExportVerify(page, 'motion');

      await page.getByRole('link', { name: /Test/ }).click();
      await page.setInputFiles('[data-test="predict-motion-file"]', motionFile(0, 77777));
      await page.locator('[data-test="top-prediction"]').waitFor({ timeout: 15000 });
      const top = await page.locator('[data-test="top-prediction"]').textContent();
      check('motion: live prediction is correct', top?.trim() === 'shake', `predicted "${top}"`);
      await ctx.close();
    }

    exitCode = checks.every((c) => c.ok) ? 0 : 1;
  } finally {
    await browser.close();
  }
} catch (err) {
  console.error('Motion/text smoke failed:', err);
  exitCode = 1;
} finally {
  vite.kill('SIGTERM');
}

console.log(
  `\n${exitCode === 0 ? 'MOTION/TEXT SMOKE OK' : 'MOTION/TEXT SMOKE FAILED'} (${checks.filter((c) => c.ok).length}/${checks.length} checks)`,
);
process.exit(exitCode);
