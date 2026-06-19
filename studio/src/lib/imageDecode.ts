/**
 * Decodes an image file into a raw RGBA frame at a fixed square size, center
 * cropping the longer axis. Used by the drag and drop and file picker import
 * path in the Data stage. Browser only (it uses createImageBitmap and a canvas),
 * so it lives outside the pure feature code.
 */
import type { RgbaFrame } from '../features/image';

/**
 * Decodes a file to a square RGBA frame.
 *
 * @param file an image file (PNG, JPEG, or anything the browser can decode).
 * @param size the output side length in pixels.
 * @returns the decoded frame.
 * @throws if the file cannot be decoded as an image.
 */
export async function fileToFrame(file: File, size: number): Promise<RgbaFrame> {
  const bitmap = await createImageBitmap(file);
  try {
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) throw new Error('Canvas 2D context unavailable.');
    const side = Math.min(bitmap.width, bitmap.height);
    const sx = (bitmap.width - side) / 2;
    const sy = (bitmap.height - side) / 2;
    ctx.drawImage(bitmap, sx, sy, side, side, 0, 0, size, size);
    const data = ctx.getImageData(0, 0, size, size).data;
    return { width: size, height: size, data };
  } finally {
    bitmap.close();
  }
}
