/**
 * Webcam capture. Opens a getUserMedia stream, binds it to a video element, and
 * grabs square frames into raw RGBA buffers the dataset stores. Frames are center
 * cropped and downscaled to a modest capture size so a project stays small in
 * IndexedDB while the feature stage can still resize down further. Devices are
 * touched only here, never in a store or a view.
 */
import { ref, onScopeDispose } from 'vue';
import type { RgbaFrame } from '../features/image';

/** The square side, in pixels, captured frames are stored at. */
export const CAPTURE_SIZE = 96;

export function useCamera() {
  const active = ref(false);
  const error = ref<string | null>(null);
  let stream: MediaStream | null = null;
  let video: HTMLVideoElement | null = null;
  const canvas = typeof document !== 'undefined' ? document.createElement('canvas') : null;

  /**
   * Starts the stream and binds it to the given video element.
   *
   * @throws if the browser denies camera access or has no camera.
   */
  async function start(el: HTMLVideoElement): Promise<void> {
    error.value = null;
    // navigator.mediaDevices is undefined on an insecure origin; surface a clear
    // message instead of a raw "cannot read getUserMedia of undefined".
    if (!navigator.mediaDevices?.getUserMedia) {
      error.value = 'Camera needs a secure context (https or localhost) and an available camera.';
      throw new Error(error.value);
    }
    // Release any stream already held so a re-entrant start does not leak the
    // previous MediaStream and leave the camera on.
    stop();
    video = el;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false });
      el.srcObject = stream;
      await el.play();
      active.value = true;
    } catch (err) {
      error.value = err instanceof Error ? err.message : String(err);
      throw err;
    }
  }

  /** Stops the stream and releases the camera. */
  function stop(): void {
    stream?.getTracks().forEach((t) => t.stop());
    stream = null;
    active.value = false;
    if (video) video.srcObject = null;
  }

  /**
   * Captures the current video frame as a square RGBA buffer at CAPTURE_SIZE,
   * center cropping the longer axis.
   *
   * @returns the captured frame, or null if the stream is not ready yet.
   */
  function capture(): RgbaFrame | null {
    if (!video || !canvas || !active.value) return null;
    const vw = video.videoWidth;
    const vh = video.videoHeight;
    if (vw === 0 || vh === 0) return null;
    const side = Math.min(vw, vh);
    const sx = (vw - side) / 2;
    const sy = (vh - side) / 2;
    canvas.width = CAPTURE_SIZE;
    canvas.height = CAPTURE_SIZE;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) return null;
    ctx.drawImage(video, sx, sy, side, side, 0, 0, CAPTURE_SIZE, CAPTURE_SIZE);
    const imageData = ctx.getImageData(0, 0, CAPTURE_SIZE, CAPTURE_SIZE);
    return { width: CAPTURE_SIZE, height: CAPTURE_SIZE, data: imageData.data };
  }

  onScopeDispose(stop);

  return { active, error, start, stop, capture };
}
