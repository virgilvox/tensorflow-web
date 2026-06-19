import { defineConfig } from 'vitest/config';

// Unit tests run in Node. They cover the parts that are deterministic and
// runtime independent: quantization math, FlatBuffers structure, and schema
// round trips. End to end parity against the real TFLite interpreter runs in
// the browser through the test app, because tfjs-tflite is a browser WASM
// build and cannot load here. See verify.ts and the README.
export default defineConfig({
  test: {
    environment: 'node',
    include: ['test/**/*.test.ts', 'src/**/*.test.ts'],
    coverage: {
      provider: 'v8',
      include: ['src/**/*.ts'],
      exclude: ['src/serialize/schema/**', 'src/**/*.test.ts'],
    },
  },
});
