/**
 * Phase 5 headless smoke: expert depth and project save and load. Confirms the
 * Expert altitude exposes the operator inspector, that an expert completes a full
 * project (train, quantize, verify), and that a project exports to a local file
 * and imports back into a fresh context with its classes and samples intact.
 *
 * Run from studio/: node scripts/smoke-expert.mjs (needs a local Chrome and the
 * tfjs and tfjs-tflite CDN scripts).
 */
import { spawn } from 'node:child_process';
import { deflateSync } from 'node:zlib';
import { readFileSync } from 'node:fs';
import { chromium } from 'playwright-core';

const root = new URL('..', import.meta.url).pathname;
const SIDE = 64;
const CLASSES = ['top', 'bottom', 'center'];

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
      px[y * SIDE + x] = Math.max(0, Math.min(255, (on ? 235 : 20) + (rand() - 0.5) * 30));
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

const { proc: vite, url: urlPromise } = startVite();
let exitCode = 0;

try {
  const base = await urlPromise;
  const browser = await chromium.launch({ channel: 'chrome', headless: true });
  let savedPath = null;
  try {
    // ---- Build, inspect, train, export, and save in an Expert context ----
    const ctx = await browser.newContext({ acceptDownloads: true });
    const page = await ctx.newPage();
    page.on('pageerror', (e) => console.log('[pageerror]', e.message));
    await page.goto(base, { waitUntil: 'networkidle' });
    await page.waitForSelector('.topbar');

    // Name the project, raise to Expert.
    await page.fill('.pname', 'bench-parts');
    await page.getByRole('radio', { name: 'Expert' }).click();

    for (const name of CLASSES) {
      await page.fill('.addrow input', name);
      await page.getByRole('button', { name: /Add class/i }).click();
      await page.waitForTimeout(50);
    }
    for (let c = 0; c < CLASSES.length; c++) {
      await page.locator('.classbtn', { hasText: CLASSES[c] }).first().click();
      for (const session of [0, 1]) {
        await page.setInputFiles('input[data-test="import-file"]', batch(c, session, 7));
        await page.waitForTimeout(120);
      }
    }
    check('imported a dataset', true);

    // Expert exposes the operator inspector on the Model stage.
    await page.getByRole('link', { name: /Model/ }).click();
    await page.waitForSelector('.ops', { timeout: 8000 });
    const opsText = (await page.textContent('.ops')) ?? '';
    check('Expert operator inspector lists real operators', /Conv2D/.test(opsText) && /Dense/.test(opsText));

    // An expert completes the project.
    await page.getByRole('link', { name: /Train/ }).click();
    await page.getByRole('button', { name: /Train model/i }).click();
    await page.getByRole('button', { name: /Go to Export/i }).waitFor({ timeout: 120000 });
    await page.getByRole('button', { name: /Go to Export/i }).click();
    await page.getByRole('button', { name: /Quantize and verify/i }).click();
    await page.getByText(/parity holds|parity failed/).first().waitFor({ timeout: 120000 });
    const exportText = (await page.textContent('section')) ?? '';
    check('expert completes a verified project', /parity holds/.test(exportText));

    // Save the project to a local file.
    await page.getByRole('link', { name: /Data/ }).click();
    const [download] = await Promise.all([
      page.waitForEvent('download', { timeout: 15000 }),
      page.getByRole('button', { name: /Export project/i }).click(),
    ]);
    savedPath = await download.path();
    // Read the bytes now: the download temp file is removed when its context
    // closes, so the import below uses an in memory buffer, not the path.
    const savedBuffer = savedPath ? readFileSync(savedPath) : Buffer.alloc(0);
    const savedName = download.suggestedFilename();
    check('exports a project file', /\.viseproj\.json$/.test(savedName) && savedBuffer.length > 0, `${savedBuffer.length} bytes`);
    await ctx.close();

    // ---- Import the saved project into a fresh context ----
    const ctx2 = await browser.newContext();
    const page2 = await ctx2.newPage();
    page2.on('pageerror', (e) => console.log('[pageerror2]', e.message));
    await page2.goto(base, { waitUntil: 'networkidle' });
    await page2.waitForSelector('.topbar');
    // A fresh context starts empty.
    const emptyBench = (await page2.textContent('.bench')) ?? '';
    const emptyOk = /Samples\s*0/.test(emptyBench);

    await page2.setInputFiles('[data-test="import-project"]', {
      name: savedName,
      mimeType: 'application/json',
      buffer: savedBuffer,
    });
    await page2.waitForFunction(
      () => {
        const t = document.querySelector('.bench')?.textContent ?? '';
        const m = t.match(/Samples\s*(\d+)/);
        return !!m && Number(m[1]) >= 42;
      },
      undefined,
      { timeout: 15000 },
    );
    await page2.waitForTimeout(300);
    const nameVal = await page2.inputValue('.pname');
    const dataText = (await page2.textContent('main')) ?? '';
    const namesOk = CLASSES.every((c) => dataText.includes(c));
    const restored = nameVal === 'bench-parts' && namesOk;
    check('imports the project with classes and samples restored', emptyOk && restored, `name "${nameVal}"`);

    await ctx2.close();
    exitCode = checks.every((c) => c.ok) ? 0 : 1;
  } finally {
    await browser.close();
  }
} catch (err) {
  console.error('Expert smoke failed:', err);
  exitCode = 1;
} finally {
  vite.kill('SIGTERM');
}

console.log(
  `\n${exitCode === 0 ? 'EXPERT SMOKE OK' : 'EXPERT SMOKE FAILED'} (${checks.filter((c) => c.ok).length}/${checks.length} checks)`,
);
process.exit(exitCode);
