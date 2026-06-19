import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';

// The studio imports the live library source through this alias so changes to
// ../src show up without a build step. The library publishes only dist, so the
// studio (and this alias) never ship with it.
const libraryEntry = fileURLToPath(new URL('../src/index.ts', import.meta.url));

export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      'tensorflow-web': libraryEntry,
    },
    // The studio and the aliased library source both import @tensorflow/tfjs.
    // Without deduping, the bundler would load two copies (one from studio
    // node_modules, one from the repo root), giving two unrelated tf instances
    // and two nominal Tensor types. Dedupe only the meta package so both resolve
    // to the studio copy; its nested tfjs-core, layers, and converter stay
    // version matched (deduping those mixes versions and breaks internal imports).
    dedupe: ['@tensorflow/tfjs'],
  },
  // @tensorflow/tfjs-tflite is a WASM build whose module layout the Vite
  // dependency optimizer cannot process. It is loaded from a CDN script tag (see
  // index.html), and verify()/live inference pick up the global it exposes.
  // Excluding it keeps the dev optimizer away from the dynamic import fallback
  // inside the library; marking it external keeps the production build from
  // trying to bundle the package and its broken internal imports. The dynamic
  // import is only a fallback and never runs while the CDN global is present.
  optimizeDeps: {
    exclude: ['@tensorflow/tfjs-tflite'],
  },
  build: {
    rollupOptions: {
      external: ['@tensorflow/tfjs-tflite'],
    },
  },
  test: {
    environment: 'node',
    include: ['test/**/*.test.ts'],
  },
});
