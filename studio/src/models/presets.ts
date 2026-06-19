/**
 * Per modality default architectures, auto sized from the input shape and the
 * class count. Each returns a ModelSpec built only from exportable layers. The
 * presets are deliberately small: these models target microcontrollers, and the
 * library trains them from scratch rather than by transfer learning.
 */
import type { LayerSpec, ModelSpec } from './types';

/**
 * A small convolutional classifier for images and spectrograms: two conv and
 * pool blocks, a flatten, a dense, and a softmax head. The filter counts stay
 * low so the int8 model fits a microcontroller.
 *
 * @param classCount number of output classes, sets the softmax width.
 * @param size input side length, used to keep deeper stacks off tiny inputs.
 */
export function imageCnnPreset(classCount: number, size: number): ModelSpec {
  const layers: LayerSpec[] = [
    { type: 'conv2d', filters: 8, kernelSize: 3, activation: 'relu', padding: 'same' },
    { type: 'maxPool2d', poolSize: 2 },
    { type: 'conv2d', filters: 16, kernelSize: 3, activation: 'relu', padding: 'same' },
    { type: 'maxPool2d', poolSize: 2 },
  ];
  // A third block only earns its keep on larger inputs.
  if (size >= 64) {
    layers.push({ type: 'conv2d', filters: 32, kernelSize: 3, activation: 'relu', padding: 'same' });
    layers.push({ type: 'maxPool2d', poolSize: 2 });
  }
  layers.push({ type: 'flatten' });
  layers.push({ type: 'dense', units: 16, activation: 'relu' });
  layers.push({ type: 'dense', units: classCount, activation: 'softmax' });
  return { layers };
}

/**
 * A small multilayer perceptron for motion summary features and bag of words
 * text: one or two dense hidden layers and a softmax head.
 *
 * @param classCount number of output classes.
 */
export function mlpPreset(classCount: number): ModelSpec {
  return {
    layers: [
      { type: 'dense', units: 32, activation: 'relu' },
      { type: 'dense', units: 16, activation: 'relu' },
      { type: 'dense', units: classCount, activation: 'softmax' },
    ],
  };
}
