<script setup lang="ts">
/**
 * Export stage. Calibrate, quantize to int8, emit the .tflite, verify parity in
 * the real interpreter, show the float versus int8 accuracy delta and the device
 * fit, then offer the named downloads. The heavy lifting is the pipeline's; this
 * stage drives it and renders the trust readout, the budget meter, and the
 * download surface.
 */
import { ref, computed } from 'vue';
import ViseSectionHead from '../design/components/ViseSectionHead.vue';
import ViseCard from '../design/components/ViseCard.vue';
import ViseButton from '../design/components/ViseButton.vue';
import ViseStatus from '../design/components/ViseStatus.vue';
import ViseBadge from '../design/components/ViseBadge.vue';
import ViseIcon from '../design/components/ViseIcon.vue';
import { useTrainingStore } from '../stores/training';
import { useSettingsStore } from '../stores/settings';
import { useProjectStore } from '../stores/project';
import { usePipeline } from '../composables/usePipeline';
import { useExport } from '../composables/useExport';
import { formatBytes, formatPercent } from '../lib/format';

const training = useTrainingStore();
const settings = useSettingsStore();
const project = useProjectStore();
const pipeline = usePipeline();
const exporter = useExport();

const busy = ref(false);
const error = ref<string | null>(null);

const report = computed(() => training.report);
const hasBytes = computed(() => pipeline.bytes.value !== null);
const hasTrained = computed(() => pipeline.trainedModel.value !== null);

const delta = computed(() => {
  if (!report.value || report.value.floatAcc == null || report.value.int8Acc == null) return null;
  return report.value.floatAcc - report.value.int8Acc;
});

async function runExport(): Promise<void> {
  error.value = null;
  busy.value = true;
  try {
    await pipeline.exportModel();
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err);
  } finally {
    busy.value = false;
  }
}

function meta() {
  const ops = pipeline.exportOps();
  return {
    name: project.name,
    classes: pipeline.classNames.value,
    inputShape: ops.inputShape,
    arenaBytes: training.budget?.arenaBytes ?? 0,
    ops: { resolverCalls: ops.resolverCalls },
  };
}

function dlTflite(): void {
  if (pipeline.bytes.value) exporter.downloadTflite(pipeline.bytes.value, project.name);
}
function dlCArray(): void {
  if (pipeline.bytes.value) exporter.downloadCArray(pipeline.bytes.value, project.name);
}
function dlSketch(): void {
  if (pipeline.bytes.value) exporter.downloadSketch(pipeline.bytes.value, meta());
}
</script>

<template>
  <section>
    <ViseSectionHead index="06" title="Export" note="verify, then ship" />

    <div class="actionbar">
      <ViseButton :disabled="!hasTrained || busy" @click="runExport">
        <ViseIcon name="bolt" :size="14" /> {{ busy ? 'Working...' : 'Quantize and verify' }}
      </ViseButton>
      <ViseStatus :state="busy ? 'run' : report ? (report.parity ? 'pass' : 'fault') : 'hold'">
        {{ busy ? 'running' : report ? (report.parity ? 'parity holds' : 'parity failed') : 'idle' }}
      </ViseStatus>
    </div>
    <p v-if="!hasTrained" class="hint">Train a model first, then quantize and verify it here.</p>
    <p v-if="error" class="err">{{ error }}</p>

    <ViseCard title="Verified output" meta="trust signal" accent>
      <div class="row">
        <ViseStatus :state="report?.parity ? 'pass' : hasBytes ? 'warn' : 'hold'">
          {{ report ? (report.parity ? 'parity holds' : 'parity failed') : 'not exported yet' }}
        </ViseStatus>
        <ViseBadge v-if="report" variant="why" :title="`max abs error ${report.maxAbsError}`">
          why
        </ViseBadge>
      </div>
      <p class="reason">
        The emitted .tflite is loaded back into the real TFLite interpreter and checked against the
        float reference before it is trusted.
      </p>
    </ViseCard>

    <div class="grid">
      <ViseCard title="Accuracy delta" meta="float vs int8">
        <div v-if="report" class="delta">
          <span class="mono-num">float {{ formatPercent(report.floatAcc ?? 0) }}</span>
          <span class="mono-num">int8 {{ formatPercent(report.int8Acc ?? 0) }}</span>
          <span v-if="delta !== null" class="drop mono-num">drop {{ formatPercent(delta) }}</span>
        </div>
        <p v-else class="reason">Quantization loss is shown here, never hidden.</p>
      </ViseCard>

      <ViseCard title="Device fit" :meta="settings.target.name">
        <div v-if="training.budget" class="fit">
          <ViseStatus :state="training.budget.fits ? 'pass' : 'fault'">
            {{ training.budget.fits ? 'fits' : 'over budget' }}
          </ViseStatus>
          <span class="mono-num">flash {{ formatBytes(training.budget.flashBytes) }}</span>
          <span class="mono-num">arena {{ formatBytes(training.budget.arenaBytes) }}</span>
        </div>
        <p v-else class="reason">
          Flash is the exact .tflite byte count; the arena is estimated from the model and checked
          against {{ settings.target.name }}.
        </p>
      </ViseCard>
    </div>

    <p class="subhead">Downloads</p>
    <p v-if="report && !report.parity" class="err">
      Parity failed: the int8 model does not match the float reference within tolerance, so it is
      not offered for download. Try more calibration data or retrain.
    </p>
    <div class="downloads">
      <ViseButton :disabled="!hasBytes" @click="dlTflite"><ViseIcon name="download" :size="14" /> .tflite</ViseButton>
      <ViseButton variant="ghost" :disabled="!hasBytes" @click="dlCArray">
        <ViseIcon name="chip" :size="14" /> C array
      </ViseButton>
      <ViseButton variant="ghost" :disabled="!hasBytes" @click="dlSketch">
        <ViseIcon name="bolt" :size="14" /> TFLite Micro sketch
      </ViseButton>
    </div>
    <p class="reason mt">
      The C array and the TFLite Micro sketch are generated in the browser from the .tflite bytes,
      with the op resolver and the arena size pre filled, so the train to ESP32 story is real.
    </p>
  </section>
</template>

<style scoped>
.actionbar {
  display: flex;
  align-items: center;
  gap: var(--s-4);
  margin-bottom: var(--s-3);
}
.hint,
.err {
  font-size: 12px;
  margin: 0 0 var(--s-4);
}
.hint {
  color: var(--ash);
}
.err {
  color: var(--rust);
}
.row {
  display: flex;
  align-items: center;
  gap: var(--s-3);
  margin-bottom: var(--s-3);
}
.reason {
  font-size: 12px;
  color: var(--ash);
  margin: 0;
  line-height: 1.6;
}
.grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: var(--s-4);
  margin: var(--s-4) 0;
}
@media (max-width: 760px) {
  .grid {
    grid-template-columns: 1fr;
  }
}
.delta,
.fit {
  display: flex;
  flex-wrap: wrap;
  gap: var(--s-4);
  align-items: center;
  font-size: 12px;
  color: var(--steam);
}
.drop {
  color: var(--amber);
}
.subhead {
  font-family: var(--f-label);
  text-transform: uppercase;
  letter-spacing: 0.14em;
  font-size: 10px;
  color: var(--ash);
  margin: var(--s-5) 0 14px;
  display: flex;
  align-items: center;
  gap: 8px;
}
.subhead::before {
  content: '';
  width: 14px;
  height: 2px;
  background: var(--live);
}
.downloads {
  display: flex;
  flex-wrap: wrap;
  gap: var(--s-3);
}
.mt {
  margin-top: var(--s-4);
}
</style>
