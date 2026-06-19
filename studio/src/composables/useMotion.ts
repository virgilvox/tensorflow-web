/**
 * Device motion capture. Listens to the accelerometer through DeviceMotion
 * events, keeps a short rolling trace so the gesture is visibly distinct during
 * capture, and records fixed length windows the dataset stores. Some platforms
 * require a permission gesture, which start requests. Devices are touched only
 * here.
 */
import { ref, onScopeDispose } from 'vue';

/** Default capture window length in seconds. */
export const MOTION_SECONDS = 2;
/** Number of axes captured: accelerometer x, y, z. */
export const MOTION_AXES = 3;

export interface MotionWindow {
  hz: number;
  axes: number;
  /** Interleaved [x0, y0, z0, x1, y1, z1, ...]. */
  data: Float32Array;
}

interface DeviceMotionEventStatic {
  requestPermission?: () => Promise<'granted' | 'denied'>;
}

export function useMotion() {
  const active = ref(false);
  const error = ref<string | null>(null);
  // A rolling trace of recent samples for the live capture preview.
  const trace = ref<Array<[number, number, number]>>([]);
  let buffer: Array<[number, number, number]> = [];
  let collecting = false;
  let listener: ((e: DeviceMotionEvent) => void) | null = null;

  /** True when the DeviceMotion API exists at all. */
  function supported(): boolean {
    return typeof window !== 'undefined' && 'DeviceMotionEvent' in window;
  }

  /**
   * Starts listening to motion events, requesting permission where required.
   *
   * @throws if the API is unavailable or permission is denied.
   */
  async function start(): Promise<void> {
    error.value = null;
    if (!supported()) {
      error.value = 'This device or browser does not expose motion sensors.';
      throw new Error(error.value);
    }
    const ctor = window.DeviceMotionEvent as unknown as DeviceMotionEventStatic;
    if (typeof ctor.requestPermission === 'function') {
      const result = await ctor.requestPermission();
      if (result !== 'granted') throw new Error('Motion permission denied.');
    }
    listener = (e: DeviceMotionEvent) => {
      const a = e.accelerationIncludingGravity ?? e.acceleration;
      const sample: [number, number, number] = [a?.x ?? 0, a?.y ?? 0, a?.z ?? 0];
      if (collecting) buffer.push(sample);
      trace.value.push(sample);
      if (trace.value.length > 120) trace.value.shift();
    };
    window.addEventListener('devicemotion', listener);
    active.value = true;
  }

  /** Stops listening. */
  function stop(): void {
    if (listener) window.removeEventListener('devicemotion', listener);
    listener = null;
    active.value = false;
  }

  /**
   * Records one window of the given length.
   *
   * @returns the interleaved samples, the axis count, and the observed rate.
   */
  function record(seconds = MOTION_SECONDS): Promise<MotionWindow> {
    buffer = [];
    collecting = true;
    return new Promise<MotionWindow>((resolve) => {
      window.setTimeout(() => {
        collecting = false;
        const samples = buffer;
        const data = new Float32Array(samples.length * MOTION_AXES);
        samples.forEach((s, i) => data.set(s, i * MOTION_AXES));
        resolve({ hz: samples.length / seconds, axes: MOTION_AXES, data });
      }, seconds * 1000);
    });
  }

  onScopeDispose(stop);

  return { active, error, trace, supported, start, stop, record };
}
