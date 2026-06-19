<script setup lang="ts">
/**
 * Train stage. One button training with live loss and accuracy gauges, a loss
 * curve, the hazard rail while running, an epoch readout, and cancel. The button
 * lights up only when the minimum data is met. The training run itself is wired
 * to the library with the flagship image flow; this stage owns the controls and
 * the live readouts that every modality shares.
 */
import { ref, computed } from 'vue';
import ViseSectionHead from '../design/components/ViseSectionHead.vue';
import ViseButton from '../design/components/ViseButton.vue';
import ViseGauge from '../design/components/ViseGauge.vue';
import ViseLossChart from '../design/components/ViseLossChart.vue';
import ViseStatus from '../design/components/ViseStatus.vue';
import ViseSlider from '../design/components/ViseSlider.vue';
import ViseIcon from '../design/components/ViseIcon.vue';
import { useProjectStore } from '../stores/project';
import { useTrainingStore } from '../stores/training';
import { useSettingsStore } from '../stores/settings';
import { formatFixed, formatPercent } from '../lib/format';

const project = useProjectStore();
const training = useTrainingStore();
const settings = useSettingsStore();

const epochs = ref(12);

/** Minimum bar to train: at least two classes with at least a few samples each. */
const MIN_PER_CLASS = 5;
const ready = computed(() => {
  if (project.classes.length < 2) return false;
  return project.classes.every((c) => (project.countsByClass[c.id] ?? 0) >= MIN_PER_CLASS);
});

const blockReason = computed(() => {
  if (project.classes.length < 2) return 'Add at least two classes in the Data stage.';
  return `Collect at least ${MIN_PER_CLASS} samples per class. Capture lands with each modality.`;
});

const lossSeries = computed(() => training.metrics.map((m) => m.loss));
const accSeries = computed(() =>
  training.metrics.map((m) => m.acc ?? 0).filter(() => training.metrics.some((m) => m.acc != null)),
);

const latestAcc = computed(() => training.latest?.acc);
</script>

<template>
  <section>
    <ViseSectionHead index="04" title="Train" note="local and unlimited, runs in this tab" />

    <div class="head">
      <ViseButton :disabled="!ready || training.running">
        <ViseIcon name="play" :size="14" /> Train model
      </ViseButton>
      <ViseButton v-if="training.running" variant="danger" size="sm">
        <ViseIcon name="stop" :size="13" /> Cancel
      </ViseButton>
      <ViseStatus :state="training.running ? 'run' : training.status === 'done' ? 'pass' : 'hold'">
        {{ training.running ? `epoch ${training.epoch}/${training.totalEpochs}` : training.status }}
      </ViseStatus>
    </div>

    <p v-if="!ready" class="block">{{ blockReason }}</p>

    <div v-if="settings.editable" class="knob">
      <ViseSlider v-model="epochs" label="Epochs" :min="1" :max="60" :step="1" />
    </div>

    <div class="gauges">
      <ViseGauge label="Progress" :fraction="training.progress" :value="formatPercent(training.progress)" />
      <ViseGauge
        label="Loss"
        :fraction="training.latest ? Math.max(0, 1 - training.latest.loss) : 0"
        :value="formatFixed(training.latest?.loss, 3)"
        cool
      />
      <ViseGauge
        label="Accuracy"
        :fraction="latestAcc ?? 0"
        :value="latestAcc === undefined ? '—' : formatPercent(latestAcc)"
      />
    </div>

    <div class="chart">
      <ViseLossChart :loss="lossSeries" :acc="accSeries.length ? accSeries : undefined" />
    </div>
  </section>
</template>

<style scoped>
.head {
  display: flex;
  align-items: center;
  gap: var(--s-4);
  margin-bottom: var(--s-4);
}
.block {
  font-size: 12px;
  color: var(--amber);
  margin: 0 0 var(--s-4);
}
.knob {
  max-width: 320px;
  margin-bottom: var(--s-5);
}
.gauges {
  display: flex;
  flex-wrap: wrap;
  gap: var(--s-3);
  margin-bottom: var(--s-5);
}
.chart {
  max-width: 640px;
}
</style>
