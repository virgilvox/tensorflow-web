/**
 * Export downloads. Triggers browser downloads of the emitted .tflite, the C
 * array, and the TFLite Micro sketch, plus a small JSON metadata file with the
 * classes and the feature config. Everything is generated in the browser from
 * the bytes already in hand; no network, no account.
 */
import { toCArray, toCIdentifier, toTFLMSketch, type SketchOps } from '../lib/cformat';
import { encodeBundle } from '../lib/modelBundle';
import type { FeatureConfig } from '../features';
import type { Modality } from '../types';

/** Triggers a browser download of arbitrary bytes or text under a file name. */
function download(filename: string, content: Uint8Array | string, mime: string): void {
  const blob =
    typeof content === 'string'
      ? new Blob([content], { type: mime })
      : new Blob([content.slice().buffer], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

export interface ExportMeta {
  name: string;
  classes: string[];
  inputShape: number[];
  arenaBytes: number;
  ops: SketchOps;
}

export function useExport() {
  /** Downloads the raw .tflite. */
  function downloadTflite(bytes: Uint8Array, name: string): void {
    download(`${toCIdentifier(name)}.tflite`, bytes, 'application/octet-stream');
  }

  /** Downloads the C source array. */
  function downloadCArray(bytes: Uint8Array, name: string): void {
    const id = toCIdentifier(name);
    download(`${id}_model.h`, toCArray(bytes, name), 'text/plain');
  }

  /** Downloads the TFLite Micro sketch wired with the op resolver and arena. */
  function downloadSketch(bytes: Uint8Array, meta: ExportMeta): void {
    const sketch = toTFLMSketch({
      varName: meta.name,
      arenaBytes: meta.arenaBytes,
      inputShape: meta.inputShape,
      classCount: meta.classes.length,
      ops: meta.ops,
    });
    download(`${toCIdentifier(meta.name)}_sketch.ino`, sketch, 'text/plain');
    // The sketch references the model header, so it is handy alongside it.
    download(`${toCIdentifier(meta.name)}_model.h`, toCArray(bytes, meta.name), 'text/plain');
  }

  /** Downloads a small JSON describing the classes and feature config. */
  function downloadMetadata(meta: ExportMeta): void {
    const json = JSON.stringify(
      { name: meta.name, classes: meta.classes, inputShape: meta.inputShape, arenaBytes: meta.arenaBytes },
      null,
      2,
    );
    download(`${toCIdentifier(meta.name)}_metadata.json`, json, 'application/json');
  }

  /**
   * Downloads a single self-contained bundle (.tfwsmodel.json) carrying the
   * .tflite bytes plus the modality, classes, feature config, and input shape,
   * so the Playground can load and run the model with one file and no project.
   */
  function downloadBundle(input: {
    name: string;
    modality: Modality;
    classes: string[];
    featureConfig: FeatureConfig;
    inputShape: number[];
    audioSeconds?: number;
    bytes: Uint8Array;
  }): void {
    download(`${toCIdentifier(input.name)}.tfwsmodel.json`, encodeBundle(input), 'application/json');
  }

  return { downloadTflite, downloadCArray, downloadSketch, downloadMetadata, downloadBundle };
}
