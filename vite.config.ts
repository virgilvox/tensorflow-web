import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';

// The test app is never published. It imports the live library source through
// the 'tensorflow-web' alias so changes to src show up without a build step.
const srcIndex = fileURLToPath(new URL('./src/index.ts', import.meta.url));

// @tensorflow/tfjs-tflite is an older package whose internal module layout the
// Vite dependency optimizer cannot process. It is built to be loaded from a
// script tag, which is how the app provides it (see app/index.html). Excluding
// it here keeps the optimizer away from it; verify() picks up the global it
// exposes. The dynamic import in verify stays as a fallback for other hosts.
export default defineConfig({
  root: 'app',
  resolve: {
    alias: {
      'tensorflow-web': srcIndex,
    },
  },
  optimizeDeps: {
    exclude: ['@tensorflow/tfjs-tflite'],
  },
});
