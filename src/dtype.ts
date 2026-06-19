/**
 * Numeric type helpers shared across the IR and the serializer.
 *
 * Tensor constant data is stored as raw little endian bytes, because that is
 * what the .tflite format requires regardless of the host machine. The helpers
 * here write little endian explicitly through a DataView rather than aliasing a
 * typed array buffer, so the output is correct on a big endian host too.
 */
import type { DType } from './types';

/** TFLite TensorType enum values. Mirrors schema/tflite/tensor-type.ts. */
export const TensorTypeCode: Record<DType, number> = {
  float32: 0,
  int32: 2,
  uint8: 3,
  bool: 6,
  int16: 7,
  int8: 9,
};

/** Bytes occupied by one element of each dtype. */
export const DTypeByteWidth: Record<DType, number> = {
  float32: 4,
  int32: 4,
  int16: 2,
  int8: 1,
  uint8: 1,
  bool: 1,
};

export function f32ToBytes(values: ArrayLike<number>): Uint8Array {
  const out = new Uint8Array(values.length * 4);
  const view = new DataView(out.buffer);
  for (let i = 0; i < values.length; i++) view.setFloat32(i * 4, values[i] as number, true);
  return out;
}

export function i32ToBytes(values: ArrayLike<number>): Uint8Array {
  const out = new Uint8Array(values.length * 4);
  const view = new DataView(out.buffer);
  for (let i = 0; i < values.length; i++) view.setInt32(i * 4, values[i] as number, true);
  return out;
}

export function i8ToBytes(values: ArrayLike<number>): Uint8Array {
  const out = new Uint8Array(values.length);
  for (let i = 0; i < values.length; i++) out[i] = (values[i] as number) & 0xff;
  return out;
}

export function bytesToF32(bytes: Uint8Array): Float32Array {
  const out = new Float32Array(bytes.length / 4);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let i = 0; i < out.length; i++) out[i] = view.getFloat32(i * 4, true);
  return out;
}

export function bytesToI8(bytes: Uint8Array): Int8Array {
  return new Int8Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
}

/** Total element count for a shape. An empty shape is a scalar with one element. */
export function numElements(shape: readonly number[]): number {
  return shape.reduce((a, b) => a * b, 1);
}
