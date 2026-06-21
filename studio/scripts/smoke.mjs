/**
 * Phase 1 headless smoke. Starts the Vite dev server, opens the studio in the
 * system Chrome through Playwright, and confirms the shell renders in the TF Web Studio
 * language and that the altitude control switches a visible disclosure state:
 * the Features and Model stages are hidden in Guided and appear from Standard up.
 *
 * Run from studio/: node scripts/smoke.mjs (needs a local Chrome).
 */
import { spawn } from 'node:child_process';
import { chromium } from 'playwright-core';

const root = new URL('..', import.meta.url).pathname;

function startVite() {
  const proc = spawn('npx', ['vite', '--port', '5180', '--strictPort'], {
    cwd: root,
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  const url = new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('Vite did not start within 40s')), 40000);
    const onData = (chunk) => {
      const match = chunk.toString().match(/localhost:(\d+)/);
      if (match) {
        clearTimeout(timer);
        resolve(`http://localhost:${match[1]}`);
      }
    };
    proc.stdout.on('data', onData);
    proc.stderr.on('data', onData);
  });
  return { proc, url };
}

const checks = [];
function check(name, ok, detail = '') {
  checks.push({ name, ok, detail });
  console.log(`  ${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? `  (${detail})` : ''}`);
}

const { proc: vite, url: urlPromise } = startVite();
let exitCode = 0;

try {
  const base = await urlPromise;
  const browser = await chromium.launch({ channel: 'chrome', headless: true });
  try {
    const page = await browser.newPage();
    const consoleErrors = [];
    page.on('console', (m) => {
      if (m.type() === 'error') consoleErrors.push(m.text());
    });
    page.on('pageerror', (err) => consoleErrors.push(err.message));

    await page.goto(base, { waitUntil: 'networkidle' });
    await page.waitForSelector('.topbar', { timeout: 15000 });

    // Renders in the studio shell language: the wordmark and the local only badge.
    const wordmark = await page.textContent('.topbar .word');
    check('TF Web wordmark renders', wordmark?.trim() === 'TF Web', `got "${wordmark}"`);

    const localBadge = await page.textContent('.topbar .local');
    check('local only indicator present', /local only/i.test(localBadge ?? ''));

    // The steel surface: the body background resolves to the void token.
    const bg = await page.evaluate(() => getComputedStyle(document.body).backgroundColor);
    check('steel surface applied', bg === 'rgb(10, 12, 15)', bg);

    // The jaw grip clamps the work panel.
    const jaws = await page.locator('.jaw').count();
    check('jaw grip present', jaws >= 2, `${jaws} jaws`);

    // Disclosure, the heart of the exit test. Guided hides Features and Model.
    const railSel = 'nav[aria-label="Workflow stages"] a';
    const guidedStages = await page.locator(railSel).allTextContents();
    const guidedCount = guidedStages.length;
    const guidedHasConfig = guidedStages.some((t) => /Features|Model/.test(t));
    check('Guided hides config stages', guidedCount === 4 && !guidedHasConfig, `${guidedCount} stages`);

    // Switch the altitude control to Standard.
    await page.getByRole('radio', { name: 'Standard' }).click();
    await page.waitForTimeout(150);
    const stdStages = await page.locator(railSel).allTextContents();
    const stdCount = stdStages.length;
    const stdHasConfig =
      stdStages.some((t) => /Features/.test(t)) && stdStages.some((t) => /Model/.test(t));
    check('Standard reveals Features and Model', stdCount === 6 && stdHasConfig, `${stdCount} stages`);

    // Disclosure inside a stage: Expert exposes an expert-only card on Model.
    // Navigate by clicking the rail link so the SPA route and the Pinia altitude
    // state both persist (a full reload would reset the store to Guided).
    await page.getByRole('radio', { name: 'Expert' }).click();
    await page.getByRole('link', { name: /Model/ }).click();
    await page.getByText('Operator inspector', { exact: true }).first().waitFor({ timeout: 8000 });
    const modelText = await page.textContent('main');
    check('Expert exposes the operator inspector on Model', /Operator inspector/i.test(modelText ?? ''));

    // The supported layer list is read live from the library.
    check('library layer set surfaced', /Conv2D/.test(modelText ?? '') && /Dense/.test(modelText ?? ''));

    // Standard feature controls are real, not placeholder text: changing the image
    // size changes the live input shape preview.
    await page.getByRole('link', { name: /Features/ }).click();
    await page.waitForSelector('[data-test="feature-shape"]');
    const shapeBefore = await page.textContent('[data-test="feature-shape"]');
    await page.selectOption('select[aria-label="Image size"]', '32');
    await page.waitForTimeout(100);
    const shapeAfter = await page.textContent('[data-test="feature-shape"]');
    check(
      'Standard feature controls change the input shape',
      shapeBefore !== shapeAfter && /32/.test(shapeAfter ?? ''),
      `${shapeBefore?.trim()} -> ${shapeAfter?.trim()}`,
    );

    // Data stage works: add a class and see it listed.
    await page.getByRole('link', { name: /Data/ }).click();
    await page.waitForSelector('.modgrid');
    await page.fill('.addrow input', 'cup');
    await page.getByRole('button', { name: /Add class/i }).click();
    await page.waitForTimeout(120);
    const classList = await page.textContent('.classlist');
    check('Data stage adds a class', /cup/.test(classList ?? ''));

    // The styled confirm dialog guards destructive actions, replacing window.confirm.
    await page.getByRole('button', { name: 'New project' }).click();
    await page.waitForSelector('[data-test="confirm-dialog"]', { timeout: 5000 });
    await page.locator('[data-test="confirm-cancel"]').click();
    await page.waitForTimeout(120);
    const afterCancel = (await page.textContent('.classlist')) ?? '';
    const dialogGone = (await page.locator('[data-test="confirm-dialog"]').count()) === 0;
    check('confirm dialog cancel keeps the data', /cup/.test(afterCancel) && dialogGone);

    await page.getByRole('button', { name: 'New project' }).click();
    await page.waitForSelector('[data-test="confirm-dialog"]');
    await page.locator('[data-test="confirm-ok"]').click();
    await page.waitForTimeout(150);
    check('confirm dialog confirm clears the data', (await page.locator('.classlist').count()) === 0);

    check('no console errors', consoleErrors.length === 0, consoleErrors.slice(0, 3).join(' | '));

    exitCode = checks.every((c) => c.ok) ? 0 : 1;
  } finally {
    await browser.close();
  }
} catch (err) {
  console.error('Smoke runner failed:', err);
  exitCode = 1;
} finally {
  vite.kill('SIGTERM');
}

console.log(`\n${exitCode === 0 ? 'SMOKE OK' : 'SMOKE FAILED'} (${checks.filter((c) => c.ok).length}/${checks.length} checks)`);
process.exit(exitCode);
