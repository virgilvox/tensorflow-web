import { defineConfig } from 'tsup';

// The library ships as ESM with type declarations. flatbuffers is bundled
// because it is a small runtime dependency. TensorFlow.js stays external; it
// is a peer dependency that the host application provides.
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
});
