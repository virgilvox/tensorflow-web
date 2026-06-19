/**
 * Imports a motion window from a JSON file, the counterpart to importing image
 * or audio files. The file is a JSON object with the sampling rate, the axis
 * count, and the interleaved samples. This makes the motion flow usable without
 * a sensor and testable headless. Pure parsing, no device access.
 */
import type { MotionWindow } from '../composables/useMotion';

interface MotionFileShape {
  hz?: number;
  axes?: number;
  data?: number[];
}

/**
 * Parses a JSON motion file into a window.
 *
 * @param file a JSON file shaped { hz, axes, data: number[] } where data is
 *   interleaved per axis.
 * @returns the parsed window.
 * @throws if the file is not valid JSON or is missing the sample data.
 */
export async function fileToMotion(file: File): Promise<MotionWindow> {
  const text = await file.text();
  let parsed: MotionFileShape;
  try {
    parsed = JSON.parse(text) as MotionFileShape;
  } catch {
    throw new Error(`${file.name} is not valid JSON.`);
  }
  if (!Array.isArray(parsed.data) || parsed.data.length === 0) {
    throw new Error(`${file.name} has no motion samples.`);
  }
  const axes = parsed.axes ?? 3;
  return { hz: parsed.hz ?? 50, axes, data: Float32Array.from(parsed.data) };
}
