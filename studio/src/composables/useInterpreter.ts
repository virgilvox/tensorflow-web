/**
 * Access to the TFLite WASM interpreter. The interpreter is loaded as a global
 * script from the alpha.8 CDN in index.html, because the alpha.10 npm build omits
 * the WASM binaries and Vite cannot bundle the package. The library's verify()
 * detects the same global; this composable points it at the alpha.8 dist before
 * verify runs and loads emitted models for live inference. See PLAN section 20.
 */

/** The dist that actually ships the WASM, matching the loaded global script. */
const WASM_DIST = 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite@0.0.1-alpha.8/dist/';

let wasmPathSet = false;

/** A loaded interpreter, narrowed to the predict method live inference calls. */
export interface LoadedModel {
  predict(input: unknown): unknown;
}

export function useInterpreter() {
  /** True when the global interpreter script has loaded and exposes its API. */
  function ready(): boolean {
    return typeof window !== 'undefined' && !!window.tflite?.loadTFLiteModel;
  }

  /** Points the interpreter at the alpha.8 WASM dist, once. Safe to call often. */
  function ensureWasmPath(): void {
    if (wasmPathSet) return;
    window.tflite?.setWasmPath?.(WASM_DIST);
    wasmPathSet = true;
  }

  /**
   * Loads emitted .tflite bytes into a fresh interpreter for live inference.
   *
   * @returns the loaded model.
   * @throws if the interpreter global is not present, with guidance on the CDN
   *   script, or if the bytes fail to load.
   */
  async function loadModel(bytes: Uint8Array): Promise<LoadedModel> {
    if (!ready()) {
      throw new Error(
        'The TFLite interpreter is not loaded. Confirm the tfjs and tfjs-tflite CDN scripts in index.html are reachable.',
      );
    }
    ensureWasmPath();
    const buffer = bytes.buffer.slice(
      bytes.byteOffset,
      bytes.byteOffset + bytes.byteLength,
    ) as ArrayBuffer;
    return (await window.tflite!.loadTFLiteModel!(buffer)) as LoadedModel;
  }

  return { ready, ensureWasmPath, loadModel };
}
