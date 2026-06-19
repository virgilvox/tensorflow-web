/**
 * Public API for tensorflow-web.
 *
 * TensorFlow.js trains the model. This library calibrates it, quantizes it to
 * int8, serializes a real .tflite, and verifies the result against the float
 * reference. Everything here is headless: functions return plain data or
 * promises, and the host application owns all presentation.
 */
export type {
  Model,
  DType,
  DataSource,
  RepresentativeData,
  Logs,
  TrainOptions,
  TrainResult,
  CalibrationMethod,
  CalibrateOptions,
  TensorRange,
  Calibration,
  WeightQuantScheme,
  ActivationQuantScheme,
  QuantizeOptions,
  QuantizedModel,
  VerifyReport,
  ConvertOptions,
  ConvertResult,
} from './types';

export type {
  IRGraph,
  IRTensor,
  IROp,
  OpKind,
  Quantization,
  FusedActivation,
  Padding,
} from './ir';
export { GraphBuilder } from './ir';

// Pipeline stages.
export { train } from './train';
export { calibrate } from './calibrate';
export { quantize, quantizeGraph } from './quantize';
export { buildFloatGraph } from './convert/build-graph';
export { toTFLite } from './emit';
export { verify, structuralCheck } from './verify';
export type { StructuralCheck, VerifyOptions } from './verify';
export { convert } from './convert';

// Serialization and the converter registry.
export { serialize } from './serialize';
export {
  registerConverter,
  getConverter,
  hasConverter,
  supportedLayers,
  UnsupportedLayerError,
} from './ops/registry';
export type { LayerConverter, ConvertContext } from './ops/types';
