<script setup lang="ts">
/**
 * The bench rail. Persistent live state across the bottom of the bench: the
 * training job status, the total and per class sample counts, the emitted model
 * size, and the device fit meter against the selected target. It reads the
 * stores and renders read only; every stage writes the state it shows.
 */
import { computed } from 'vue';
import TwStatus from './TwStatus.vue';
import TwHazard from './TwHazard.vue';
import { useProjectStore } from '../../stores/project';
import { useTrainingStore } from '../../stores/training';
import { useSettingsStore } from '../../stores/settings';
import { formatBytes, formatFixed } from '../../lib/format';

const project = useProjectStore();
const training = useTrainingStore();
const settings = useSettingsStore();

const jobState = computed(() => {
  switch (training.status) {
    case 'running':
      return { state: 'run' as const, label: `training ${training.epoch}/${training.totalEpochs}` };
    case 'done':
      return { state: 'pass' as const, label: 'done' };
    case 'error':
      return { state: 'fault' as const, label: 'fault' };
    case 'cancelled':
      return { state: 'warn' as const, label: 'cancelled' };
    default:
      return { state: 'hold' as const, label: 'idle' };
  }
});

const modelSize = computed(() =>
  training.tfliteBytes === null ? '·' : formatBytes(training.tfliteBytes),
);

const fit = computed(() => {
  if (!training.budget) return { state: 'hold' as const, label: 'no model' };
  return training.budget.fits
    ? { state: 'pass' as const, label: `fits ${settings.target.name}` }
    : { state: 'fault' as const, label: `over ${settings.target.name}` };
});
</script>

<template>
  <footer class="bench">
    <TwHazard v-if="training.running" :height="4" />
    <div class="readouts">
      <div class="cell job">
        <span class="k">Job</span>
        <TwStatus :state="jobState.state">{{ jobState.label }}</TwStatus>
      </div>

      <div class="cell">
        <span class="k">Samples</span>
        <span class="v mono-num">{{ project.totalSamples }}</span>
      </div>

      <div class="cell">
        <span class="k">Classes</span>
        <span class="v mono-num">{{ project.classes.length }}</span>
      </div>

      <div class="cell">
        <span class="k">Last loss</span>
        <span class="v mono-num">{{ formatFixed(training.latest?.loss) }}</span>
      </div>

      <div class="cell">
        <span class="k">Model</span>
        <span class="v mono-num">{{ modelSize }}</span>
      </div>

      <div class="cell">
        <span class="k">Fit</span>
        <TwStatus :state="fit.state">{{ fit.label }}</TwStatus>
      </div>

      <a class="credit" href="https://hack.build" target="_blank" rel="noopener noreferrer">
        made by hack.build
      </a>
    </div>
  </footer>
</template>

<style scoped>
.bench {
  background: var(--soot);
  border-top: 1px solid var(--seam);
  flex: none;
}
.readouts {
  display: flex;
  align-items: center;
  gap: var(--s-6);
  padding: var(--s-2) var(--s-5);
  overflow-x: auto;
}
.cell {
  display: inline-flex;
  align-items: center;
  gap: var(--s-2);
  white-space: nowrap;
}
.k {
  font-family: var(--f-label);
  text-transform: uppercase;
  letter-spacing: 0.12em;
  font-size: 8.5px;
  color: var(--ash);
}
.v {
  font-family: var(--f-display);
  font-weight: 600;
  font-size: 13px;
  color: var(--chalk);
}
.credit {
  margin-left: auto;
  white-space: nowrap;
  font-family: var(--f-label);
  text-transform: uppercase;
  letter-spacing: 0.12em;
  font-size: 8.5px;
  color: var(--ash);
  transition: color var(--fast);
}
.credit:hover {
  color: var(--live);
}
</style>
