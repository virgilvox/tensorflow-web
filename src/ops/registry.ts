/**
 * The converter registry. This is the roadmap, not a wall: supporting a new
 * layer is registering one converter. A layer with no converter fails loudly on
 * export, naming the offending class, rather than emitting a silently wrong file.
 */
import type { LayerConverter } from './types';

export class UnsupportedLayerError extends Error {
  constructor(public readonly layerClass: string, supported: readonly string[]) {
    super(
      `tensorflow-web has no converter for the layer "${layerClass}". ` +
        `Training a model with this layer works, but exporting it to .tflite does not yet. ` +
        `Supported layers: ${supported.join(', ')}. ` +
        `Adding a converter is a small, isolated change; see src/ops and DESIGN.md.`,
    );
    this.name = 'UnsupportedLayerError';
  }
}

const registry = new Map<string, LayerConverter>();

export function registerConverter(converter: LayerConverter): void {
  registry.set(converter.layerClass, converter);
}

export function hasConverter(layerClass: string): boolean {
  return registry.has(layerClass);
}

export function getConverter(layerClass: string): LayerConverter {
  const converter = registry.get(layerClass);
  if (!converter) throw new UnsupportedLayerError(layerClass, supportedLayers());
  return converter;
}

export function supportedLayers(): string[] {
  return [...registry.keys()].sort();
}
