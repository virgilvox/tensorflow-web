/**
 * Emits a standalone activation operator. Used both by the Activation and ReLU
 * layer converters and by conv or dense converters whose configured activation
 * cannot be fused (softmax, sigmoid, tanh).
 */
import type { ConvertContext } from './types';

/**
 * Appends one activation op reading `input` and writing a new float32
 * activation named `outName` with shape `outShape`.
 * Returns the index of the produced activation tensor.
 * Throws if the activation name is not a supported standalone activation.
 */
export function emitActivation(
  ctx: ConvertContext,
  activation: string,
  input: number,
  outName: string,
  outShape: number[],
): number {
  const out = ctx.builder.addActivation(outName, outShape, 'float32');
  switch (activation) {
    case 'relu':
      ctx.builder.addOp({ kind: 'RELU', inputs: [input], outputs: [out] });
      break;
    case 'relu6':
      ctx.builder.addOp({ kind: 'RELU6', inputs: [input], outputs: [out] });
      break;
    case 'sigmoid':
      ctx.builder.addOp({ kind: 'LOGISTIC', inputs: [input], outputs: [out] });
      break;
    case 'tanh':
      ctx.builder.addOp({ kind: 'TANH', inputs: [input], outputs: [out] });
      break;
    case 'softmax':
      ctx.builder.addOp({ kind: 'SOFTMAX', inputs: [input], outputs: [out], options: { beta: 1 } });
      break;
    default:
      throw new Error(`Unsupported standalone activation "${activation}".`);
  }
  return out;
}
