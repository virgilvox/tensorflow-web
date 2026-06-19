/**
 * Shared helpers for layer converters. Keeps padding and activation mapping in
 * one place so every converter agrees on the TFLite spelling of these enums.
 */
import type { FusedActivation, Padding } from '../ir';

/**
 * Maps a tfjs padding string to the TFLite padding enum.
 * Returns 'SAME' or 'VALID'.
 * Throws if the padding is neither 'same' nor 'valid'.
 */
export function toPadding(padding: unknown): Padding {
  if (padding === 'same') return 'SAME';
  if (padding === 'valid') return 'VALID';
  throw new Error(`Unsupported padding "${String(padding)}". Use "same" or "valid".`);
}

/**
 * Maps a tfjs activation name to a fused TFLite activation, or null when the
 * activation cannot be fused into a conv or fully connected op.
 * Returns 'NONE' for linear or absent activations, 'RELU' for relu, 'RELU6' for
 * relu6, and null for any activation that must become its own op.
 */
export function toFusedActivation(activation: unknown): FusedActivation | null {
  if (activation == null || activation === 'linear') return 'NONE';
  if (activation === 'relu') return 'RELU';
  if (activation === 'relu6') return 'RELU6';
  return null;
}

/** Returns a [strideH, strideW] pair from a tfjs strides config value. */
export function toStridePair(strides: unknown, fallback = 1): [number, number] {
  if (Array.isArray(strides)) return [strides[0] ?? fallback, strides[1] ?? fallback];
  if (typeof strides === 'number') return [strides, strides];
  return [fallback, fallback];
}
