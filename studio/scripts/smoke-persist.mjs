/**
 * Persistence regression smoke. The earlier per-modality smokes never reloaded
 * the page after capturing, so a bug where the class list failed to persist (the
 * meta record held Vue reactive proxies that IndexedDB could not structured
 * clone) went unseen: on reload the classes vanished and every sample became an
 * orphan, crashing Train with "Sample references unknown class". This drives a
 * capture, a reload, and a train to prove classes and samples survive a refresh,
 * including after removing a class.
 *
 * Run from studio/: node scripts/smoke-persist.mjs (needs a local Chrome).
 */
import { spawn } from 'node:child_process';
import { chromium } from 'playwright-core';

const root = new URL('..', import.meta.url).pathname;

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
const check = (name, ok, detail = '') => {
  checks.push({ name, ok });
  console.log(`  ${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? `  (${detail})` : ''}`);
};

async function addText(page, classes) {
  await page.locator('.modcard', { hasText: 'Text' }).click();
  for (const n of classes) {
    await page.fill('.addrow input', n);
    await page.getByRole('button', { name: /Add class/i }).click();
    await page.waitForTimeout(50);
  }
  for (const n of classes) {
    await page.locator('.selbtn', { hasText: n }).first().click();
    for (let i = 0; i < 6; i++) {
      await page.fill('[data-test="text-input"]', `${n} example phrase variant ${i}`);
      await page.getByRole('button', { name: /Add example/i }).click();
      await page.waitForTimeout(20);
    }
  }
}

async function trainReaches(page) {
  await page.getByRole('link', { name: /Train/ }).click();
  const btn = page.getByRole('button', { name: /Train model/i });
  if (await btn.isDisabled()) return false;
  let pageErr = null;
  page.once('pageerror', (e) => (pageErr = e.message));
  await btn.click();
  try {
    await page.getByRole('button', { name: /Go to Export/i }).waitFor({ timeout: 60000 });
  } catch {
    return false;
  }
  return !pageErr;
}

const { proc: vite, url: urlPromise } = startVite();
let exitCode = 0;
try {
  const base = await urlPromise;
  const browser = await chromium.launch({ channel: 'chrome', headless: true });
  try {
    // Capture, reload, and confirm classes and samples both survived.
    {
      const ctx = await browser.newContext();
      const page = await ctx.newPage();
      page.on('dialog', (d) => d.accept());
      await page.goto(base, { waitUntil: 'networkidle' });
      await page.waitForSelector('.topbar');
      await addText(page, ['alpha', 'beta']);
      await page.reload({ waitUntil: 'networkidle' });
      await page.waitForSelector('.topbar');
      const bench = (await page.textContent('.bench')) ?? '';
      check('classes and samples survive a reload', /Classes\s*2/.test(bench) && /Samples\s*12/.test(bench), bench.replace(/\s+/g, ' ').match(/Classes\s*\d+\s*.*?Samples\s*\d+|Samples\s*\d+.*?Classes\s*\d+/)?.[0]);
      check('train runs after a reload without an orphan error', await trainReaches(page));
      await ctx.close();
    }
    // Remove a class, reload, and train (the deletion path stays consistent).
    {
      const ctx = await browser.newContext();
      const page = await ctx.newPage();
      page.on('dialog', (d) => d.accept());
      await page.goto(base, { waitUntil: 'networkidle' });
      await page.waitForSelector('.topbar');
      await addText(page, ['alpha', 'beta', 'gamma']);
      await page.locator('.classrow', { hasText: 'gamma' }).first().locator('.del').click();
      await page.waitForTimeout(300);
      await page.reload({ waitUntil: 'networkidle' });
      await page.waitForSelector('.topbar');
      const bench = (await page.textContent('.bench')) ?? '';
      check('removing a class then reloading leaves a consistent dataset', /Classes\s*2/.test(bench) && /Samples\s*12/.test(bench));
      check('train runs after remove and reload', await trainReaches(page));
      await ctx.close();
    }
    exitCode = checks.every((c) => c.ok) ? 0 : 1;
  } finally {
    await browser.close();
  }
} catch (err) {
  console.error('Persistence smoke failed:', err);
  exitCode = 1;
} finally {
  vite.kill('SIGTERM');
}
console.log(`\n${exitCode === 0 ? 'PERSIST SMOKE OK' : 'PERSIST SMOKE FAILED'} (${checks.filter((c) => c.ok).length}/${checks.length} checks)`);
process.exit(exitCode);
