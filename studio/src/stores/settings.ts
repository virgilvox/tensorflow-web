/**
 * Settings store: the altitude level that drives progressive disclosure across
 * every stage, and the selected target device the budget meter checks against.
 * These two choices live in the top bar and are read everywhere.
 */
import { defineStore } from 'pinia';
import type { AltitudeLevel, TargetDevice, TargetDeviceId } from '../types';

/** The microcontroller targets the studio knows how to size a model against. */
export const TARGET_DEVICES: readonly TargetDevice[] = [
  // Budgets are deliberately conservative: usable space after a TFLite Micro
  // runtime and an application, not the raw chip totals.
  { id: 'esp32s3', name: 'ESP32 S3', flashBytes: 4 * 1024 * 1024, ramBytes: 320 * 1024 },
  { id: 'esp32', name: 'ESP32', flashBytes: 2 * 1024 * 1024, ramBytes: 160 * 1024 },
  { id: 'cortex-m7', name: 'Cortex M7', flashBytes: 1 * 1024 * 1024, ramBytes: 256 * 1024 },
  { id: 'cortex-m4', name: 'Cortex M4', flashBytes: 512 * 1024, ramBytes: 128 * 1024 },
];

interface SettingsState {
  altitude: AltitudeLevel;
  targetId: TargetDeviceId;
}

export const useSettingsStore = defineStore('settings', {
  state: (): SettingsState => ({
    altitude: 'guided',
    targetId: 'esp32s3',
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
  },
});
