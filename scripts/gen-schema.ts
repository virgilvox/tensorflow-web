/**
 * Generates the TypeScript FlatBuffers bindings from the pinned TFLite schema.
 *
 * The schema is committed at schema/schema.fbs and pinned to a specific upstream
 * revision (see schema/SCHEMA_SOURCE.md). Regenerating the bindings is a
 * deliberate, reviewed act, never a silent one. Run with: npm run gen:schema
 *
 * Requires the flatc compiler on PATH. Install with: brew install flatbuffers
 * (macOS) or see https://github.com/google/flatbuffers/releases.
 */
import { execFileSync } from 'node:child_process';
import { existsSync, rmSync, mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '..');
const schema = resolve(root, 'schema', 'schema.fbs');
const outDir = resolve(root, 'src', 'serialize', 'schema');

function flatcAvailable(): boolean {
  try {
    execFileSync('flatc', ['--version'], { stdio: 'ignore' });
    return true;
  } catch {
    return false;
  }
}

if (!flatcAvailable()) {
  console.error(
    'flatc not found on PATH. Install it (brew install flatbuffers) and retry.',
  );
  process.exit(1);
}

if (!existsSync(schema)) {
  console.error(`Schema not found at ${schema}.`);
  process.exit(1);
}

rmSync(outDir, { recursive: true, force: true });
mkdirSync(outDir, { recursive: true });

const version = execFileSync('flatc', ['--version']).toString().trim();
console.log(`Using ${version}`);

execFileSync(
  'flatc',
  ['--ts', '--gen-all', '--ts-no-import-ext', '-o', outDir, schema],
  { stdio: 'inherit' },
);

console.log(`Generated TypeScript bindings into ${outDir}`);
