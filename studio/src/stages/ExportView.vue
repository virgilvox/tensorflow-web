<script setup lang="ts">
/**
 * Export stage. Calibrate, quantize to int8, emit the .tflite, verify parity in
 * the real interpreter, show the float versus int8 accuracy delta and the device
 * fit, then offer the named downloads. The pipeline is wired with the flagship
 * image flow; this stage owns the verify readout, the budget meter, and the
 * download surface every modality shares.
 */
import { computed } from 'vue';
import ViseSectionHead from '../design/components/ViseSectionHead.vue';
import ViseCard from '../design/components/ViseCard.vue';
import ViseButton from '../design/components/ViseButton.vue';
import ViseStatus from '../design/components/ViseStatus.vue';
import ViseBadge from '../design/components/ViseBadge.vue';
import ViseIcon from '../design/components/ViseIcon.vue';
import { useTrainingStore } from '../stores/training';
import { useSettingsStore } from '../stores/settings';
import { formatBytes, formatPercent } from '../lib/format';

const training = useTrainingStore();
const settings = useSettingsStore();

const report = computed(() => training.report);
const hasModel = computed(() => training.tfliteBytes !== null);

const delta = computed(() => {
  if (!report.value || report.value.floatAcc == null || report.value.int8Acc == null) return null;
  return report.value.floatAcc - report.value.int8Acc;
});
</script>

<template>
  <section>
    <ViseSectionHead index="06" title="Export" note="verify, then ship" />

    <ViseCard title="Verified output" meta="trust signal" accent>
      <div class="row">
        <ViseStatus :state="report?.parity ? 'pass' : hasModel ? 'warn' : 'hold'">
          {{ report ? (report.parity ? 'parity holds' : 'parity failed') : 'not exported yet' }}
        </ViseStatus>
        <ViseBadge v-if="report" variant="why" :title="`max abs error ${report.maxAbsError}`">
          why
        </ViseBadge>
      </div>
      <p class="reason">
        The emitted .tflite is loaded back into the real TFLite interpreter and checked against the
        float reference before it is trusted. No competitor shows this.
      </p>
    </ViseCard>

    <div class="grid">
      <ViseCard title="Accuracy delta" meta="float vs int8">
        <div v-if="report" class="delta">
          <span class="mono-num">float {{ formatPercent(report.floatAcc ?? 0) }}</span>
          <span class="mono-num">int8 {{ formatPercent(report.int8Acc ?? 0) }}</span>
          <span class="drop mono-num" v-if="delta !== null">drop {{ formatPercent(delta) }}</span>
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
    <div class="downloads">
      <ViseButton :disabled="!hasModel"><ViseIcon name="download" :size="14" /> .tflite</ViseButton>
      <ViseButton variant="ghost" :disabled="!hasModel">
        <ViseIcon name="chip" :size="14" /> C array
      </ViseButton>
      <ViseButton variant="ghost" :disabled="!hasModel">
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
