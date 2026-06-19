/**
 * Decodes an audio file (WAV and anything the browser can decode) into mono PCM
 * samples and its sample rate. Used by the audio import path in the Data stage,
 * the counterpart to importing image files. Browser only; it uses a single
 * shared AudioContext, since browsers cap the number of live contexts and a
 * batch import would otherwise exhaust them.
 */
import type { AudioClip } from '../composables/useMicrophone';

let sharedCtx: AudioContext | null = null;

/** Lazily creates and reuses one decode context for the whole session. */
function decodeContext(): AudioContext {
  if (!sharedCtx) sharedCtx = new AudioContext();
  return sharedCtx;
}

/**
 * Decodes a file to mono PCM, mixing down any extra channels.
 *
 * @param file an audio file.
 * @returns the samples and the decoded sample rate.
 * @throws if the file cannot be decoded as audio.
 */
export async function fileToAudio(file: File): Promise<AudioClip> {
  const bytes = await file.arrayBuffer();
  // decodeAudioData detaches the buffer, so each call gets its own copy.
  const buffer = await decodeContext().decodeAudioData(bytes);
  const channels = buffer.numberOfChannels;
  const length = buffer.length;
  const out = new Float32Array(length);
  for (let c = 0; c < channels; c++) {
    const data = buffer.getChannelData(c);
    for (let i = 0; i < length; i++) out[i] += data[i]! / channels;
  }
  return { data: out, sampleRate: buffer.sampleRate };
}
