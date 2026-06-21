/**
 * Settings store: the altitude level that drives progressive disclosure across
 * every stage, and the selected target device the budget meter checks against.
 * These two choices live in the top bar and are read everywhere.
 */
import { defineStore } from 'pinia';
import type { AltitudeLevel, TargetDevice, TargetDeviceId } from '../types';
import type { ModelCapacity } from '../models/types';

/** The microcontroller targets the studio knows how to size a model against. */
export const TARGET_DEVICES: readonly TargetDevice[] = [
  // Budgets are deliberately conservative: usable space after a TFLite Micro
  // runtime and an application, not the raw chip totals.
  { id: 'esp32s3', name: 'ESP32 S3', flashBytes: 4 * 1024 * 1024, ramBytes: 320 * 1024 },
  { id: 'esp32', name: 'ESP32', flashBytes: 2 * 1024 * 1024, ramBytes: 160 * 1024 },
  { id: 'cortex-m7', name: 'Cortex M7', flashBytes: 1 * 1024 * 1024, ramBytes: 256 * 1024 },
  { id: 'cortex-m4', name: 'Cortex M4', flashBytes: 512 * 1024, ramBytes: 128 * 1024 },
];

/** Bounds on the audio clip length, in seconds. Longer clips mean a larger
 *  spectrogram span and a heavier model, so the upper bound stays modest. */
export const AUDIO_SECONDS_MIN = 0.25;
export const AUDIO_SECONDS_MAX = 10;
/** Quick-pick clip lengths offered in the capture UI, in seconds. */
export const AUDIO_SECONDS_PRESETS: readonly number[] = [1, 2, 4];

interface SettingsState {
  altitude: AltitudeLevel;
  targetId: TargetDeviceId;
  /** Microphone clip length for new audio captures, in seconds. */
  audioSeconds: number;
  // Feature knobs, editable from Standard up. Defaults match the auto pipeline,
  // so leaving them alone reproduces the Guided behavior exactly.
  imageSize: number;
  imageChannels: 1 | 3;
  audioMode: 'mel' | 'mfcc';
  audioBands: number;
  motionSteps: number;
  textVocabCap: number;
  /** Model size lever. */
  modelCapacity: ModelCapacity;
}

export const useSettingsStore = defineStore('settings', {
  state: (): SettingsState => ({
    altitude: 'guided',
    targetId: 'esp32s3',
    audioSeconds: 1,
    imageSize: 48,
    imageChannels: 1,
    audioMode: 'mel',
    audioBands: 32,
    motionSteps: 32,
    textVocabCap: 200,
    modelCapacity: 'standard',
  }),
  getters: {
    /** True when configuration stages (Features, Model) are visible at all. */
    showsConfigStages: (s): boolean => s.altitude !== 'guided',
    /** True when the main knobs on each stage are editable. */
    editable: (s): boolean => s.altitude !== 'guided',
    /** True when every expert lever and the layer editor are exposed. */
    expert: (s): boolean => s.altitude === 'expert',
    /** The resolved target device record for the current selection. */
    target(s): TargetDevice {
      return TARGET_DEVICES.find((d) => d.id === s.targetId) ?? TARGET_DEVICES[0]!;
    },
  },
  actions: {
    setAltitude(level: AltitudeLevel): void {
      this.altitude = level;
    },
    setTarget(id: TargetDeviceId): void {
      this.targetId = id;
    },
    /** Sets the audio clip length, clamped to the supported range. */
    setAudioSeconds(seconds: number): void {
      if (!Number.isFinite(seconds)) return;
      this.audioSeconds = Math.min(AUDIO_SECONDS_MAX, Math.max(AUDIO_SECONDS_MIN, seconds));
    },
  },
});
