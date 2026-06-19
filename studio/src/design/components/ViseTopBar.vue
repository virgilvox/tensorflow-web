<script setup lang="ts">
/**
 * The fixed top bar. Carries the VISE mark, the altitude control that drives
 * progressive disclosure across every stage, the active project name, the target
 * device selector the budget meter checks against, and the loud local only
 * indicator stating that nothing leaves the browser.
 */
import { computed, useTemplateRef } from 'vue';
import ViseSegmented from './ViseSegmented.vue';
import ViseIcon from './ViseIcon.vue';
import { useSettingsStore, TARGET_DEVICES } from '../../stores/settings';
import { useProjectStore } from '../../stores/project';
import { useProjectFile } from '../../composables/useProjectFile';
import { useDataset } from '../../composables/useDataset';
import type { AltitudeLevel, TargetDeviceId } from '../../types';

const settings = useSettingsStore();
const project = useProjectStore();
const projectFile = useProjectFile();
const dataset = useDataset();
const importInput = useTemplateRef<HTMLInputElement>('importInput');

async function onRename(event: Event): Promise<void> {
  project.name = (event.target as HTMLInputElement).value || 'untitled';
  await dataset.persistMeta();
}

async function onImport(event: Event): Promise<void> {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];
  if (file) await projectFile.importFromFile(file);
  input.value = '';
}

const ALTITUDES: ReadonlyArray<{ value: AltitudeLevel; label: string }> = [
  { value: 'guided', label: 'Guided' },
  { value: 'standard', label: 'Standard' },
  { value: 'expert', label: 'Expert' },
];

const altitude = computed<string>({
  get: () => settings.altitude,
  set: (v) => settings.setAltitude(v as AltitudeLevel),
});

function onTargetChange(event: Event): void {
  settings.setTarget((event.target as HTMLSelectElement).value as TargetDeviceId);
}
</script>

<template>
  <header class="topbar">
    <div class="mark">
      <span class="jawmark" aria-hidden="true"><i></i><i></i></span>
      <span class="word">VISE</span>
      <span class="sub">Studio</span>
    </div>

    <div class="altitude">
      <span class="lab cap">Altitude</span>
      <ViseSegmented v-model="altitude" :options="ALTITUDES" aria-label="Altitude level" />
    </div>

    <div class="right">
      <div class="project">
        <span class="lab cap">Project</span>
        <div class="prow">
          <input
            class="pname"
            :value="project.name"
            aria-label="Project name"
            spellcheck="false"
            @change="onRename"
          />
          <button class="picon" type="button" aria-label="Export project to a file" title="Export project" @click="projectFile.exportToFile()">
            <ViseIcon name="save" :size="14" />
          </button>
          <button class="picon" type="button" aria-label="Import project from a file" title="Import project" @click="importInput?.click()">
            <ViseIcon name="open" :size="14" />
          </button>
          <input ref="importInput" type="file" accept=".json,application/json" class="hidden" data-test="import-project" @change="onImport" />
        </div>
      </div>

      <label class="device">
        <span class="lab cap">Target</span>
        <select :value="settings.targetId" aria-label="Target device" @change="onTargetChange">
          <option v-for="d in TARGET_DEVICES" :key="d.id" :value="d.id">{{ d.name }}</option>
        </select>
      </label>

      <span class="local" title="All data stays in this browser. Nothing is uploaded.">
        <ViseIcon name="lock" :size="13" />
        Local only
      </span>
    </div>
  </header>
</template>

<style scoped>
.topbar {
  display: flex;
  align-items: center;
  gap: var(--s-6);
  background: var(--soot);
  border-bottom: 1px solid var(--seam);
  padding: var(--s-2) var(--s-5);
  flex: none;
}
.mark {
  display: inline-flex;
  align-items: baseline;
  gap: var(--s-2);
}
.jawmark {
  position: relative;
  display: inline-block;
  width: 16px;
  height: 16px;
  align-self: center;
}
.jawmark i {
  position: absolute;
  top: 50%;
  transform: translateY(-50%);
  width: 4px;
  height: 12px;
  background: var(--live);
}
.jawmark i:first-child {
  left: 0;
}
.jawmark i:last-child {
  right: 0;
}
.word {
  font-family: var(--f-display);
  font-weight: 700;
  font-size: var(--t-lg);
  letter-spacing: 0.08em;
  color: var(--chalk);
}
.sub {
  font-family: var(--f-label);
  text-transform: uppercase;
  letter-spacing: 0.16em;
  font-size: 9px;
  color: var(--ash);
}
.altitude {
  display: inline-flex;
  align-items: center;
  gap: var(--s-3);
}
.cap {
  white-space: nowrap;
}
.right {
  margin-left: auto;
  display: inline-flex;
  align-items: center;
  gap: var(--s-5);
}
.project,
.device {
  display: inline-flex;
  flex-direction: column;
  gap: 3px;
}
.prow {
  display: inline-flex;
  align-items: center;
  gap: 4px;
}
.pname {
  font-family: var(--f-mono);
  font-size: var(--t-sm);
  color: var(--chalk);
  background: var(--gunmetal);
  border: 1px solid var(--seam);
  padding: 3px 7px;
  width: 13ch;
}
.pname:focus {
  outline: none;
  border-color: var(--live);
  box-shadow: 0 0 0 3px var(--live-glow);
}
.picon {
  display: inline-flex;
  background: none;
  border: 1px solid var(--seam);
  color: var(--ash);
  padding: 4px;
  cursor: pointer;
  transition: color var(--fast), border-color var(--fast);
}
.picon:hover {
  color: var(--live);
  border-color: var(--live-dim);
}
.hidden {
  display: none;
}
.device select {
  background: var(--gunmetal);
  border: 1px solid var(--seam);
  color: var(--chalk);
  font-family: var(--f-mono);
  font-size: var(--t-xs);
  padding: 5px 8px;
}
.device select:focus {
  outline: none;
  border-color: var(--live);
  box-shadow: 0 0 0 3px var(--live-glow);
}
.local {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-family: var(--f-label);
  text-transform: uppercase;
  letter-spacing: 0.12em;
  font-size: 9.5px;
  color: var(--live);
  border: 1px solid var(--live-dim);
  padding: 5px 9px;
  background: var(--live-glow);
}
</style>
