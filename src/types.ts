/**
 * Public types for tensorflow-web.
 *
 * These describe the data that crosses the library boundary. They never depend
 * on the generated FlatBuffers schema; that coupling lives only in the
 * serializer. TensorFlow.js types are imported as types only, so the library
 * carries no runtime dependency on tfjs (it is a peer dependency the host
 * application provides).
 */
import type * as tf from '@tensorflow/tfjs';

/** A trained or untrained Keras style model from TensorFlow.js. */
export type Model = tf.LayersModel;

/** Numeric storage types a tensor can take in the emitted graph. */
export type DType = 'float32' | 'int8' | 'int32' | 'uint8' | 'int16' | 'bool';

/** A dataset the library can consume: either a tfjs Dataset or raw tensors. */
export type DataSource =
  | tf.data.Dataset<{ xs: tf.Tensor; ys: tf.Tensor }>
  | { xs: tf.Tensor; ys: tf.Tensor };

/** Representative input data used for calibration. Labels are not required. */
export type RepresentativeData =
  | tf.data.Dataset<tf.Tensor>
  | tf.data.Dataset<{ xs: tf.Tensor }>
  | tf.Tensor
  | tf.Tensor[];

/** Per epoch and per batch metrics forwarded straight from tfjs. */
export type Logs = tf.Logs;

export interface TrainOptions {
  data: DataSource;
  epochs: number;
  batchSize?: number;
  validationSplit?: number;
  validationData?: DataSource;
  shuffle?: boolean;
  /** Called after every epoch. Drive a progress bar or a loss curve from here. */
  onEpoch?: (epoch: number, logs: Logs) => void | Promise<void>;
  /** Called after every batch. Optional, higher frequency than onEpoch. */
  onBatch?: (batch: number, logs: Logs) => void | Promise<void>;
  /** Abort training cooperatively. The returned model holds the weights so far. */
  signal?: AbortSignal;
}

export interface TrainResult {
  model: Model;
  history: tf.History;
}

export type CalibrationMethod = 'minmax' | 'percentile';

export interface CalibrateOptions {
  /** minmax is implemented today. percentile and kl are planned. */
  method?: CalibrationMethod;
  /** Percentile in (0, 100] used when method is percentile. */
  percentile?: number;
  batchSize?: number;
}

/** Observed numeric range of a single tensor across the representative set. */
export interface TensorRange {
  min: number;
  max: number;
}

export interface Calibration {
  /** Ranges keyed by the activation tensor name in the float graph. */
  ranges: Record<string, TensorRange>;
  method: CalibrationMethod;
  sampleCount: number;
}

export type WeightQuantScheme = 'per-channel' | 'per-tensor';
export type ActivationQuantScheme = 'per-tensor';

export interface QuantizeOptions {
  weights?: WeightQuantScheme;
  activations?: ActivationQuantScheme;
}

export interface VerifyReport {
  /** True when every output element is within the tolerance of the reference. */
  parity: boolean;
  /** Largest absolute elementwise error between reference and emitted model. */
  maxAbsError: number;
  /** Accuracy of the float reference model on the test set, when labels exist. */
  floatAcc?: number;
  /** Accuracy of the emitted int8 model on the test set, when labels exist. */
  int8Acc?: number;
  /** Row indexed confusion matrix for classification, when labels exist. */
  confusion?: number[][];
  /** Estimated runtime arena size in bytes for the emitted model. */
  arenaBytes?: number;
  /** Number of samples evaluated. */
  sampleCount: number;
}

export interface ConvertOptions {
  representativeData: RepresentativeData;
  testData?: DataSource;
  /** int8 runs the full quantization path. float emits a float32 .tflite. */
  quantize?: 'int8' | 'float';
  calibrate?: CalibrateOptions;
  quantizeOptions?: QuantizeOptions;
  /** Tolerance for the parity check in verify. Defaults are scheme dependent. */
  tolerance?: number;
}

export interface ConvertResult {
  tflite: Uint8Array;
  report?: VerifyReport;
}

/**
 * A model lowered to the graph IR and ready to serialize. `scheme` records
 * whether it carries int8 quantized weights and activations or stays float32.
 * The `graph` is the internal representation the serializer consumes.
 */
export interface QuantizedModel {
  readonly graph: import('./ir').IRGraph;
  readonly scheme: 'int8' | 'float';
}
