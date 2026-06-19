/**
 * Domain types shared across the studio. These describe the project, its data,
 * and the configuration that flows from the stores into the composables that
 * call the tensorflow-web library. They carry no Vue and no library coupling.
 */

/** The four lead modalities plus bag of words text. */
export type Modality = 'image' | 'audio' | 'motion' | 'text';

/** Progressive disclosure level. Drives what every stage shows. */
export type AltitudeLevel = 'guided' | 'standard' | 'expert';

/** A microcontroller target the device budget meter checks the model against. */
export type TargetDeviceId = 'esp32' | 'esp32s3' | 'cortex-m4' | 'cortex-m7';

/** A deployment target with the flash and runtime arena budget it offers. */
export interface TargetDevice {
  id: TargetDeviceId;
  name: string;
  /** Usable flash for the model, in bytes. */
  flashBytes: number;
  /** Usable RAM for the runtime tensor arena, in bytes. */
  ramBytes: number;
}

/** The six workflow stages, in nav rail order. */
export type StageId = 'data' | 'features' | 'model' | 'train' | 'test' | 'export';

/** A class the model learns. The negative class is marked so the flow can scaffold it. */
export interface ClassDef {
  id: string;
  name: string;
  /** True for the scaffolded Neither / Background Noise / Idle / Other class. */
  negative: boolean;
}

/**
 * One captured example. The raw payload is kept so features can be recomputed
 * when the feature config changes. The shape of payload depends on the modality.
 */
export interface Sample {
  id: string;
  classId: string;
  /** Whole capture sessions are assigned to train or test together, never split. */
  sessionId: string;
  createdAt: number;
  /** Raw payload: image pixels, audio samples, a motion window, or text. */
  payload: SamplePayload;
}

export type SamplePayload =
  | { kind: 'image'; width: number; height: number; data: Uint8ClampedArray }
  | { kind: 'audio'; sampleRate: number; data: Float32Array }
  | { kind: 'motion'; hz: number; axes: number; data: Float32Array }
  | { kind: 'text'; text: string };
