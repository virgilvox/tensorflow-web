/**
 * Microphone capture. Opens a getUserMedia audio stream and keeps a rolling ring
 * buffer of the most recent audio while active, at the audio context's native
 * sample rate. Two ways to read it: record() resolves one fixed length clip (used
 * by the Data and Test stages), and latestWindow() returns the most recent window
 * synchronously for continuous live inference (used by the Playground, so it can
 * keep listening instead of recording one clip at a time). Devices are touched
 * only here.
 */
import { ref, onScopeDispose } from 'vue';

/** Default clip length in seconds. */
export const CLIP_SECONDS = 1;

/** Seconds of audio retained in the ring buffer; bounds the longest clip/window. */
const RING_SECONDS = 12;

export interface AudioClip {
  data: Float32Array;
  sampleRate: number;
}

export function useMicrophone() {
  const active = ref(false);
  const error = ref<string | null>(null);
  let stream: MediaStream | null = null;
  let ctx: AudioContext | null = null;
  let source: MediaStreamAudioSourceNode | null = null;
  let processor: ScriptProcessorNode | null = null;

  // Rolling ring buffer of the most recent audio at the native sample rate.
  let ring: Float32Array | null = null;
  let ringCap = 0;
  let writePos = 0;
  let filled = 0;
  let sampleRate = 0;
  // Called after each captured block, so a pending record() can check progress.
  let onBlock: (() => void) | null = null;
  // Settles an in-flight record() if the graph is torn down mid capture.
  let abortRecord: (() => void) | null = null;

  /**
   * Opens the microphone stream and the audio graph and starts filling the ring.
   *
   * @throws if the browser denies microphone access.
   */
  async function start(): Promise<void> {
    error.value = null;
    // navigator.mediaDevices is undefined on an insecure origin; surface a clear
    // message instead of a raw "cannot read getUserMedia of undefined".
    if (!navigator.mediaDevices?.getUserMedia) {
      error.value = 'Microphone needs a secure context (https or localhost) and an available microphone.';
      throw new Error(error.value);
    }
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
      ctx = new AudioContext();
      sampleRate = ctx.sampleRate;
      ringCap = Math.ceil(RING_SECONDS * sampleRate);
      ring = new Float32Array(ringCap);
      writePos = 0;
      filled = 0;
      source = ctx.createMediaStreamSource(stream);
      // ScriptProcessorNode is deprecated but universally supported and the
      // lightest way to pull raw PCM; an AudioWorklet would be heavier here.
      processor = ctx.createScriptProcessor(4096, 1, 1);
      processor.onaudioprocess = (e) => {
        const ch = e.inputBuffer.getChannelData(0);
        for (let i = 0; i < ch.length; i++) {
          ring![writePos] = ch[i]!;
          writePos = (writePos + 1) % ringCap;
        }
        filled = Math.min(ringCap, filled + ch.length);
        onBlock?.();
      };
      source.connect(processor);
      processor.connect(ctx.destination);
      // Resume in case the context starts suspended under the autoplay policy,
      // otherwise no audio blocks ever fire and record() would hang.
      await ctx.resume?.();
      active.value = true;
    } catch (err) {
      error.value = err instanceof Error ? err.message : String(err);
      throw err;
    }
  }

  /** Stops the stream and tears down the audio graph. */
  function stop(): void {
    // Settle a pending record so its awaiter does not hang after teardown.
    abortRecord?.();
    abortRecord = null;
    onBlock = null;
    processor?.disconnect();
    source?.disconnect();
    stream?.getTracks().forEach((t) => t.stop());
    void ctx?.close();
    processor = null;
    source = null;
    stream = null;
    ctx = null;
    ring = null;
    ringCap = 0;
    writePos = 0;
    filled = 0;
    sampleRate = 0;
    active.value = false;
  }

  /** Copies the most recent `need` samples out of the ring in order. */
  function readLast(need: number): Float32Array {
    const out = new Float32Array(need);
    let start = (writePos - need) % ringCap;
    if (start < 0) start += ringCap;
    for (let i = 0; i < need; i++) out[i] = ring![(start + i) % ringCap]!;
    return out;
  }

  /**
   * Returns the most recent window of audio, or null if not enough has been
   * captured yet. For continuous live inference: poll this on an interval.
   *
   * @param seconds window length; clamped to the ring capacity.
   */
  function latestWindow(seconds = CLIP_SECONDS): AudioClip | null {
    if (!ring || sampleRate === 0) return null;
    const need = Math.min(ringCap, Math.round(sampleRate * seconds));
    if (need <= 0 || filled < need) return null;
    return { data: readLast(need), sampleRate };
  }

  /**
   * Records one clip of the given length, resolving once that many fresh samples
   * have been captured since the call.
   *
   * @returns the mono PCM samples and the capture sample rate.
   * @throws if the microphone is not started.
   */
  function record(seconds = CLIP_SECONDS): Promise<AudioClip> {
    if (!ctx || !ring) throw new Error('Start the microphone before recording.');
    const need = Math.min(ringCap, Math.round(sampleRate * seconds));
    const startFilled = filled;
    return new Promise<AudioClip>((resolve, reject) => {
      abortRecord = () => {
        onBlock = null;
        abortRecord = null;
        reject(new Error('Recording stopped before it finished.'));
      };
      onBlock = () => {
        if (filled - startFilled >= need) {
          onBlock = null;
          abortRecord = null;
          resolve({ data: readLast(need), sampleRate });
        }
      };
    });
  }

  onScopeDispose(stop);

  return { active, error, start, stop, record, latestWindow };
}
