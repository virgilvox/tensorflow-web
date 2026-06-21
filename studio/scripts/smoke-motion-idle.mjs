/**
 * Motion energy-discrimination smoke. The motion counterpart to smoke-audio-bg:
 * a large "shake" class and a near-still "Idle" class. It trains, verifies parity,
 * then probes with a shake window (must predict "shake") AND a still window (must
 * predict "Idle", NOT shake). With per-axis min-max normalization a still window's
 * tiny jitter was stretched to full range and looked like a gesture, so Idle was
 * misread as shake; this guards the fixed per-axis scaling that fixes that.
 *
 * Run from studio/: node scripts/smoke-motion-idle.mjs (needs a local Chrome and
 * the tfjs and tfjs-tflite CDN scripts).
 */
import { spawn } from 'node:child_process';
import { chromium } from 'playwright-core';

const root = new URL('..', import.meta.url).pathname;
const SHAKE = 'shake';
const IDLE = 'Idle';
const STEPS = 100;
const HZ = 50;

function rng(seed) {
  let s = (seed * 2654435761) >>> 0;
  return () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0), s / 2 ** 32);
}

/** A vigorous shake: large x oscillation, gravity on z. Units are m/s^2 like. */
function shake(seed) {
  const rand = rng(seed);
  const amp = 13 + rand() * 4;
  const data = [];
  for (let t = 0; t < STEPS; t++) {
    const p = t / STEPS;
    const x = amp * Math.sin(2 * Math.PI * 8 * p) + (rand() - 0.5) * 0.4;
    const y = (rand() - 0.5) * 0.4;
    const z = 9.8 + (rand() - 0.5) * 0.4;
    data.push(x, y, z);
  }
  return data;
}

/** A still hand: only gravity and tiny jitter, the rest state. */
function idle(seed) {
  const rand = rng(seed);
  const data = [];
  for (let t = 0; t < STEPS; t++) {
    data.push((rand() - 0.5) * 0.2, (rand() - 0.5) * 0.2, 9.8 + (rand() - 0.5) * 0.2);
  }
  return data;
}

function motionFile(label, make, seed) {
  return {
    name: `${label}_${seed}.json`,
    mimeType: 'application/json',
    buffer: Buffer.from(JSON.stringify({ hz: HZ, axes: 3, data: make(seed) })),
  };
}

function startVite() {
  const proc = spawn('npx', ['vite', '--port', '5189', '--strictPort'], {
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
  await page.setInputFiles('[data-test="predict-motion-file"]', motionFile('probe', make, seed));
  await page.locator('[data-test="top-prediction"]').waitFor({ timeout: 15000 });
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

    await page.locator('.modcard', { hasText: 'Motion' }).click();
    await page.waitForTimeout(80);

    for (const name of [SHAKE, IDLE]) {
      await page.fill('.addrow input', name);
      await page.getByRole('button', { name: /Add class/i }).click();
      await page.waitForTimeout(60);
    }

    const perClass = 14;
    const plan = [
      { name: SHAKE, make: shake, base: 1000 },
      { name: IDLE, make: idle, base: 5000 },
    ];
    for (const p of plan) {
      await page.locator('.selbtn', { hasText: p.name }).first().click();
      await page.setInputFiles(
        'input[data-test="import-motion"]',
        Array.from({ length: perClass }, (_, i) => motionFile(p.name, p.make, p.base + i)),
      );
      await page.waitForTimeout(150);
    }
    await page.waitForFunction(
      (n) => {
        const t = document.querySelector('.bench')?.textContent ?? '';
        const m = t.match(/Samples\s*(\d+)/);
        return !!m && Number(m[1]) >= n;
      },
      perClass * 2,
      { timeout: 60000 },
    );
    check('imported a shake and a near-still idle class', true, `${perClass * 2} windows`);

    await page.getByRole('link', { name: /Train/ }).click();
    await page.getByRole('button', { name: /Train model/i }).click();
    await page.getByRole('button', { name: /Go to Export/i }).waitFor({ timeout: 180000 });

    await page.getByRole('button', { name: /Go to Export/i }).click();
    await page.getByRole('button', { name: /Quantize and verify/i }).click();
    await page.getByText(/parity holds|parity failed/).first().waitFor({ timeout: 120000 });
    const exportText = (await page.textContent('section')) ?? '';
    check('parity holds against the real interpreter', /parity holds/.test(exportText));
    const int8 = Number(exportText.match(/int8\s+(\d+)%/)?.[1] ?? 0);
    check('int8 accuracy separated shake from idle', int8 >= 75, `${int8}%`);

    await page.getByRole('link', { name: /Test/ }).click();

    const shakeTop = await probe(page, shake, 91111);
    check('a shake window predicts shake', shakeTop === SHAKE, `predicted "${shakeTop}"`);

    const idleTop = await probe(page, idle, 92222);
    check(
      'a still window predicts idle, not shake',
      idleTop === IDLE && idleTop !== SHAKE,
      `predicted "${idleTop}"`,
    );

    exitCode = checks.every((c) => c.ok) ? 0 : 1;
  } finally {
    await browser.close();
  }
} catch (err) {
  console.error('Motion idle smoke failed:', err);
  exitCode = 1;
} finally {
  vite.kill('SIGTERM');
}

console.log(
  `\n${exitCode === 0 ? 'MOTION IDLE SMOKE OK' : 'MOTION IDLE SMOKE FAILED'} (${checks.filter((c) => c.ok).length}/${checks.length} checks)`,
);
process.exit(exitCode);
