/**
 * Microphone capture. Opens a getUserMedia audio stream and records fixed length
 * mono PCM clips the dataset stores. Capture is at the audio context's native
 * sample rate; the audio feature extractor resamples to its working rate, so the
 * native rate is stored alongside the samples. Devices are touched only here.
 */
import { ref, onScopeDispose } from 'vue';

/** Default clip length in seconds. */
export const CLIP_SECONDS = 1;

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
  // Settles an in-flight record() if the graph is torn down mid capture, so the
  // promise never hangs.
  let abortRecord: (() => void) | null = null;

  /**
   * Opens the microphone stream and the audio graph.
   *
   * @throws if the browser denies microphone access.
   */
  async function start(): Promise<void> {
    error.value = null;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
      ctx = new AudioContext();
      source = ctx.createMediaStreamSource(stream);
      // ScriptProcessorNode is deprecated but universally supported and the
      // lightest way to pull raw PCM; an AudioWorklet would be heavier here.
      processor = ctx.createScriptProcessor(4096, 1, 1);
      source.connect(processor);
      processor.connect(ctx.destination);
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
    processor?.disconnect();
    source?.disconnect();
    stream?.getTracks().forEach((t) => t.stop());
    void ctx?.close();
    processor = null;
    source = null;
    stream = null;
    ctx = null;
    active.value = false;
  }

  /**
   * Records one clip of the given length.
   *
   * @returns the mono PCM samples and the capture sample rate.
   * @throws if the microphone is not started.
   */
  function record(seconds = CLIP_SECONDS): Promise<AudioClip> {
    if (!ctx || !processor) throw new Error('Start the microphone before recording.');
    const sampleRate = ctx.sampleRate;
    const need = Math.round(sampleRate * seconds);
    const chunks: Float32Array[] = [];
    let got = 0;
    return new Promise<AudioClip>((resolve, reject) => {
      abortRecord = () => {
        if (processor) processor.onaudioprocess = null;
        abortRecord = null;
        reject(new Error('Recording stopped before it finished.'));
      };
      processor!.onaudioprocess = (e) => {
        const ch = e.inputBuffer.getChannelData(0);
        chunks.push(new Float32Array(ch));
        got += ch.length;
        if (got >= need) {
          processor!.onaudioprocess = null;
          abortRecord = null;
          const data = new Float32Array(need);
          let offset = 0;
          for (const chunk of chunks) {
            if (offset >= need) break;
            data.set(chunk.subarray(0, need - offset), offset);
            offset += chunk.length;
          }
          resolve({ data, sampleRate });
        }
      };
    });
  }

  onScopeDispose(stop);

  return { active, error, start, stop, record };
}
