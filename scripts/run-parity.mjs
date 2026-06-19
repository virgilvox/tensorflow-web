/**
 * Runs the browser parity check end to end. Starts the Vite dev server, opens
 * app/parity.html in a real Chromium through Playwright, waits for the in page
 * check to finish, prints the report, and exits non zero if parity fails.
 *
 * tfjs-tflite is a browser WASM build, so this is how the emitted .tflite is
 * validated against the real interpreter. Run with: npm run test:parity
 * (requires a local Chrome or Chromium; uses the system Chrome by default).
 */
import { spawn } from 'node:child_process';
import { chromium } from 'playwright-core';

const root = new URL('..', import.meta.url).pathname;

function startVite() {
  const proc = spawn('npm', ['run', 'app'], { cwd: root, stdio: ['ignore', 'pipe', 'pipe'] });
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

const { proc: vite, url: urlPromise } = startVite();

let exitCode = 0;
try {
  const base = await urlPromise;
  const browser = await chromium.launch({ channel: 'chrome', headless: true });
  try {
    const page = await browser.newPage();
    page.on('console', (msg) => console.log('[browser]', msg.text()));
    page.on('pageerror', (err) => console.log('[pageerror]', err.message));
    page.on('requestfailed', (req) => console.log('[reqfail]', req.url()));
    page.on('response', (res) => {
      if (res.status() >= 400) console.log('[http]', res.status(), res.url());
    });

    await page.goto(`${base}/parity.html`, { waitUntil: 'load' });
    await page.waitForFunction(() => globalThis.__parity && globalThis.__parity.done, {
      timeout: 240000,
    });
    const result = await page.evaluate(() => globalThis.__parity);

    console.log('\n=== parity result ===');
    console.log(JSON.stringify(result, null, 2));

    if (result.error) {
      console.log('\nPARITY ERROR (page threw):', result.error);
      exitCode = 1;
    } else {
      // A model is only meaningful if it actually learned its task, and int8 is
      // only correct if it keeps that accuracy. These checks fail loudly on a
      // degenerate int8 model that collapses to one class, even within budget.
      const MIN_FLOAT_ACC = 0.8;
      const MIN_INT8_ACC = 0.75;
      const MAX_ACC_DROP = 0.15;
      for (const name of ['cnn', 'separable', 'mixed', 'residual']) {
        const m = result[name];
        if (!m) {
          console.log(`\n[${name}] FAIL: no result`);
          exitCode = 1;
          continue;
        }
        console.log(`\n[${name}]`);
        if (m.floatError) {
          console.log('  FLOAT ERROR:', m.floatError.split('\n')[0]);
          exitCode = 1;
        } else {
          const f = m.float ?? {};
          console.log(`  FLOAT parity=${f.parity} maxAbsError=${f.maxAbsError} acc=${f.floatAcc}`);
          if (!f.parity) {
            console.log('  FAIL: float export does not match the reference interpreter.');
            exitCode = 1;
          }
          if ((f.floatAcc ?? 0) < MIN_FLOAT_ACC) {
            console.log(`  FAIL: model did not learn its task (acc ${f.floatAcc} < ${MIN_FLOAT_ACC}).`);
            exitCode = 1;
          }
        }
        if (m.int8Error) {
          console.log('  INT8 ERROR:', m.int8Error.split('\n')[0]);
          exitCode = 1;
        } else {
          const q = m.int8 ?? {};
          const drop = (m.float?.floatAcc ?? 0) - (q.int8Acc ?? 0);
          console.log(
            `  INT8  parity=${q.parity} maxAbsError=${q.maxAbsError} acc=${q.int8Acc} drop=${drop.toFixed(3)}`,
          );
          if (!q.parity) {
            console.log('  FAIL: int8 export exceeds the error budget.');
            exitCode = 1;
          }
          if (drop > MAX_ACC_DROP) {
            console.log(`  FAIL: int8 accuracy dropped ${drop.toFixed(3)} > ${MAX_ACC_DROP}.`);
            exitCode = 1;
          }
          if ((q.int8Acc ?? 0) < MIN_INT8_ACC) {
            console.log(`  FAIL: int8 accuracy ${q.int8Acc} < floor ${MIN_INT8_ACC}.`);
            exitCode = 1;
          }
        }
      }
      if (exitCode === 0) {
        console.log('\nPARITY OK (float + int8, all models, accuracy preserved)');
      }
    }
  } finally {
    await browser.close();
  }
} catch (err) {
  console.error('Parity runner failed:', err);
  exitCode = 1;
} finally {
  vite.kill('SIGTERM');
}

process.exit(exitCode);
