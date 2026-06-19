/**
 * Walks a TensorFlow.js model and lowers it to a float32 graph IR using the
 * converter registry. See DESIGN.md for the layer walk and weight layout rules.
 */
import * as tf from '@tensorflow/tfjs';
import '../ops';
import type { Model } from '../types';
import { GraphBuilder, type IRGraph } from '../ir';
import { getConverter } from '../ops/registry';
import type { ConvertContext } from '../ops/types';
import { f32ToBytes, i32ToBytes } from '../dtype';

/** Concrete ConvertContext backed by a GraphBuilder and the dtype byte helpers. */
class BuildContext implements ConvertContext {
  constructor(readonly builder: GraphBuilder) {}

  /**
   * Adds a float32 constant from a tfjs tensor, reading its data synchronously
   * and storing it as little endian bytes. Returns the new tensor's IR index.
   */
  addFloatConst(name: string, tensor: tf.Tensor): number {
    const data = tensor.dataSync() as Float32Array;
    return this.builder.addConst(name, [...tensor.shape], 'float32', f32ToBytes(data));
  }

  /**
   * Adds a constant int32 vector used for shape, axis, or paddings inputs.
   * Defaults the tensor shape to a 1D vector of the values' length.
   * Returns the new tensor's IR index.
   */
  addInt32Const(name: string, values: number[], shape?: number[]): number {
    return this.builder.addConst(name, shape ?? [values.length], 'int32', i32ToBytes(values));
  }
}

/** A symbolic tensor that exposes the id tfjs uses to wire layers together. */
interface SymbolicLike {
  name: string;
  id: number;
}

/** Returns the symbolic output tensors of a layer as a flat array. */
function layerOutputs(layer: tf.layers.Layer): SymbolicLike[] {
  const out = layer.output as unknown;
  return (Array.isArray(out) ? out : [out]) as SymbolicLike[];
}

/**
 * Lowers a Sequential or linear functional model to a float32 IRGraph.
 *
 * The walk supports one input and one output with no branching. The input
 * activation is named "input" and each layer's output activation is named after
 * the layer. Weights are transposed to TFLite layout by the registered
 * converters.
 *
 * Returns the built IRGraph.
 * Throws if the model has more than one input or output, if a layer reuses an
 * output across multiple nodes (branching), or if a layer has no converter
 * (UnsupportedLayerError).
 */
export function buildFloatGraph(model: Model): IRGraph {
  if (model.inputs.length !== 1) {
    throw new Error(
      `buildFloatGraph supports one model input; this model has ${model.inputs.length}.`,
    );
  }
  if (model.outputs.length !== 1) {
    throw new Error(
      `buildFloatGraph supports one model output; this model has ${model.outputs.length}.`,
    );
  }

  const builder = new GraphBuilder();
  const ctx = new BuildContext(builder);

  // Map a symbolic tensor id to the IR tensor index that carries its value.
  const tensorIndexById = new Map<number, number>();

  const modelInput = model.inputs[0] as unknown as SymbolicLike & { shape: number[] };
  const inputShape = modelInput.shape.map((d, i) =>
    i === 0 && (d == null || d < 0) ? 1 : (d as number),
  );
  const inputIdx = builder.addActivation('input', inputShape, 'float32');
  tensorIndexById.set(modelInput.id, inputIdx);
  builder.setInputs([inputIdx]);

  let lastOutputIdx = inputIdx;

  for (const layer of model.layers) {
    const className = layer.getClassName();
    if (className === 'InputLayer') {
      // The functional input layer carries no op; its output is the graph input.
      const outs = layerOutputs(layer);
      if (outs.length === 1) tensorIndexById.set((outs[0] as SymbolicLike).id, inputIdx);
      continue;
    }

    if (layer.inboundNodes.length > 1) {
      throw new Error(
        `Layer "${layer.name}" is used by ${layer.inboundNodes.length} nodes; ` +
          `buildFloatGraph supports linear models without branching.`,
      );
    }

    const node = layer.inboundNodes[0] as { inputTensors?: unknown } | undefined;
    const inboundTensors = ((node?.inputTensors as unknown as SymbolicLike[]) ?? []);
    const inputIndices = inboundTensors.map((t) => {
      const idx = tensorIndexById.get(t.id);
      if (idx === undefined) {
        throw new Error(
          `Layer "${layer.name}" reads tensor "${t.name}" that no prior layer produced; ` +
            `buildFloatGraph supports linear models only.`,
        );
      }
      return idx;
    });
    // Sequential models report no inbound input tensors; feed the running output.
    const resolvedInputs = inputIndices.length > 0 ? inputIndices : [lastOutputIdx];

    const converter = getConverter(className);
    const outputs = converter.convert(layer, resolvedInputs, ctx);
    if (outputs.length !== 1) {
      throw new Error(
        `Layer "${layer.name}" produced ${outputs.length} outputs; ` +
          `buildFloatGraph supports one output per layer.`,
      );
    }

    const producedIdx = outputs[0] as number;
    const symbolicOuts = layerOutputs(layer);
    if (symbolicOuts.length === 1) {
      tensorIndexById.set((symbolicOuts[0] as SymbolicLike).id, producedIdx);
    }
    lastOutputIdx = producedIdx;
  }

  builder.setOutputs([lastOutputIdx]);
  return builder.build();
}
