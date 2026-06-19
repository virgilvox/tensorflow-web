/// <reference types="vite/client" />

// Single component shape declaration so TypeScript understands .vue imports.
declare module '*.vue' {
  import type { DefineComponent } from 'vue';
  const component: DefineComponent<Record<string, unknown>, Record<string, unknown>, unknown>;
  export default component;
}

// The TFLite WASM interpreter is loaded from a CDN as a global script (see
// index.html). It wraps its outputs with a global tf, also loaded as a script.
// The library's verify() detects this global and uses it instead of bundling.
declare global {
  interface Window {
    tflite?: {
      setWasmPath?: (path: string) => void;
      loadTFLiteModel?: (model: ArrayBuffer | string) => Promise<unknown>;
    };
  }
}

export {};
