/**
 * Per modality default architectures, auto sized from the input shape and the
 * class count. Each returns a ModelSpec built only from exportable layers. The
 * presets are deliberately small: these models target microcontrollers, and the
 * library trains them from scratch rather than by transfer learning.
 */
import type { LayerSpec, ModelSpec, ModelCapacity } from './types';

/** Multiplier on the hidden filter and unit counts per capacity lever. */
const CAPACITY_SCALE: Record<ModelCapacity, number> = { compact: 0.5, standard: 1, large: 2 };

/** Scales a hidden width by the capacity, never below a sensible floor. */
function scaled(width: number, capacity: ModelCapacity): number {
  return Math.max(4, Math.round(width * CAPACITY_SCALE[capacity]));
}

/**
 * A small convolutional classifier for images and spectrograms: two conv and
 * pool blocks, a flatten, a dense, and a softmax head. The filter counts stay
 * low so the int8 model fits a microcontroller.
 *
 * @param classCount number of output classes, sets the softmax width.
 * @param size input side length, used to keep deeper stacks off tiny inputs.
 * @param capacity scales the hidden widths; 'standard' is the auto size.
 */
export function imageCnnPreset(classCount: number, size: number, capacity: ModelCapacity = 'standard'): ModelSpec {
  const layers: LayerSpec[] = [
    { type: 'conv2d', filters: scaled(8, capacity), kernelSize: 3, activation: 'relu', padding: 'same' },
    { type: 'maxPool2d', poolSize: 2 },
    { type: 'conv2d', filters: scaled(16, capacity), kernelSize: 3, activation: 'relu', padding: 'same' },
    { type: 'maxPool2d', poolSize: 2 },
  ];
  // A third block only earns its keep on larger inputs.
  if (size >= 64) {
    layers.push({ type: 'conv2d', filters: scaled(32, capacity), kernelSize: 3, activation: 'relu', padding: 'same' });
    layers.push({ type: 'maxPool2d', poolSize: 2 });
  }
  layers.push({ type: 'flatten' });
  layers.push({ type: 'dense', units: scaled(16, capacity), activation: 'relu' });
  layers.push({ type: 'dense', units: classCount, activation: 'softmax' });
  return { layers };
}

/**
 * A small multilayer perceptron for motion summary features and bag of words
 * text: one or two dense hidden layers and a softmax head.
 *
 * @param classCount number of output classes.
 * @param capacity scales the hidden widths; 'standard' is the auto size.
 */
export function mlpPreset(classCount: number, capacity: ModelCapacity = 'standard'): ModelSpec {
  return {
    layers: [
      { type: 'dense', units: scaled(32, capacity), activation: 'relu' },
      { type: 'dense', units: scaled(16, capacity), activation: 'relu' },
      { type: 'dense', units: classCount, activation: 'softmax' },
    ],
  };
}
