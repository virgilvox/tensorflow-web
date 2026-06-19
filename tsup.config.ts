import { defineConfig } from 'tsup';

// The library ships as ESM with type declarations. flatbuffers is force bundled
// (tsup externalizes dependencies by default) so the published file loads in a
// plain browser from a CDN without the consumer mapping that internal runtime.
// TensorFlow.js stays external; it is a peer dependency the host provides.
export default defineConfig({
  entry: ['src/index.ts'],
  format: ['esm'],
  dts: true,
  sourcemap: true,
  clean: true,
  treeshake: true,
  target: 'es2021',
  platform: 'browser',
  external: ['@tensorflow/tfjs', '@tensorflow/tfjs-tflite'],
  noExternal: ['flatbuffers'],
});
