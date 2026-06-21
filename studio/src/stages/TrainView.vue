<script setup lang="ts">
/**
 * Train stage. One button training with live loss and accuracy gauges, a loss
 * curve, the hazard rail while running, an epoch readout, and cancel. The button
 * lights up only when the minimum data is met. The training run itself is wired
 * to the library with the flagship image flow; this stage owns the controls and
 * the live readouts that every modality shares.
 */
import { ref, computed } from 'vue';
import { useRouter } from 'vue-router';
import TwSectionHead from '../design/components/TwSectionHead.vue';
import TwButton from '../design/components/TwButton.vue';
import TwGauge from '../design/components/TwGauge.vue';
import TwLossChart from '../design/components/TwLossChart.vue';
import TwStatus from '../design/components/TwStatus.vue';
import TwSlider from '../design/components/TwSlider.vue';
import TwIcon from '../design/components/TwIcon.vue';
import { useProjectStore } from '../stores/project';
import { useTrainingStore } from '../stores/training';
import { useSettingsStore } from '../stores/settings';
import { usePipeline, MIN_PER_CLASS } from '../composables/usePipeline';
import { formatFixed, formatPercent } from '../lib/format';

const project = useProjectStore();
const training = useTrainingStore();
const settings = useSettingsStore();
const pipeline = usePipeline();
const router = useRouter();

const epochs = ref(12);
const error = ref<string | null>(null);

const ready = computed(() => pipeline.canTrain());

const blockReason = computed(() => {
  if (project.classes.length < 2) return 'Add at least two classes in the Data stage.';
  return `Collect at least ${MIN_PER_CLASS} samples per class in the Data stage.`;
});

async function startTraining(): Promise<void> {
  error.value = null;
  try {
    await pipeline.train(epochs.value);
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err);
  }
}

function goExport(): void {
  void router.push('/export');
}

const lossSeries = computed(() => training.metrics.map((m) => m.loss));
const accSeries = computed(() =>
  training.metrics.map((m) => m.acc ?? 0).filter(() => training.metrics.some((m) => m.acc != null)),
);

const latestAcc = computed(() => training.latest?.acc);
</script>

<template>
  <section>
    <TwSectionHead index="04" title="Train" note="local and unlimited, runs in this tab" />

    <div class="head">
      <TwButton :disabled="!ready || training.running" @click="startTraining">
        <TwIcon name="play" :size="14" /> Train model
      </TwButton>
      <TwButton v-if="training.running" variant="danger" size="sm" @click="pipeline.cancel()">
        <TwIcon name="stop" :size="13" /> Cancel
      </TwButton>
      <TwButton v-if="training.status === 'done'" variant="ghost" size="sm" @click="goExport">
        <TwIcon name="export" :size="13" /> Go to Export
      </TwButton>
      <TwStatus :state="training.running ? 'run' : training.status === 'done' ? 'pass' : 'hold'">
        {{ training.running ? `epoch ${training.epoch}/${training.totalEpochs}` : training.status }}
      </TwStatus>
    </div>

    <p v-if="!ready" class="block">{{ blockReason }}</p>
    <p v-if="error" class="block err">{{ error }}</p>

    <div v-if="settings.editable" class="knob">
      <TwSlider v-model="epochs" label="Epochs" :min="1" :max="60" :step="1" />
    </div>

    <div class="gauges">
      <TwGauge label="Progress" :fraction="training.progress" :value="formatPercent(training.progress)" />
      <TwGauge
        label="Loss"
        :fraction="training.latest ? Math.max(0, 1 - training.latest.loss) : 0"
        :value="formatFixed(training.latest?.loss, 3)"
        cool
      />
      <TwGauge
        label="Accuracy"
        :fraction="latestAcc ?? 0"
        :value="latestAcc === undefined ? '·' : formatPercent(latestAcc)"
      />
    </div>

    <div class="chart">
      <TwLossChart :loss="lossSeries" :acc="accSeries.length ? accSeries : undefined" />
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
.block.err {
  color: var(--rust);
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
