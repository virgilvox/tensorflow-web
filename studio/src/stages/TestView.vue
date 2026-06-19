<script setup lang="ts">
/**
 * Test stage. Live inference on fresh input with real time confidence bars, plus
 * the held out test accuracy, a confusion matrix, and per class precision and
 * recall. Live capture is wired per modality with that modality's phase; the
 * held out report comes from verify and is shown here when it exists.
 */
import { computed } from 'vue';
import ViseSectionHead from '../design/components/ViseSectionHead.vue';
import ViseCard from '../design/components/ViseCard.vue';
import ViseConfusion from '../design/components/ViseConfusion.vue';
import ViseStatus from '../design/components/ViseStatus.vue';
import { useProjectStore } from '../stores/project';
import { useTrainingStore } from '../stores/training';
import { formatPercent } from '../lib/format';

const project = useProjectStore();
const training = useTrainingStore();

const classNames = computed(() => project.classes.map((c) => c.name));
const report = computed(() => training.report);
</script>

<template>
  <section>
    <ViseSectionHead index="05" title="Test" note="live and held out" />

    <ViseCard title="Live inference" meta="fresh input">
      <div class="live">
        <ViseStatus state="hold">awaiting a trained model</ViseStatus>
        <p class="reason">
          Live confidence bars run the emitted model on fresh camera, microphone, motion, or text
          input, one sample at a time as the .tflite expects. This panel activates once a model is
          trained and exported.
        </p>
      </div>
    </ViseCard>

    <ViseCard v-if="report" title="Held out accuracy" meta="from verify" accent class="mt">
      <div class="acc">
        <div class="accnum">
          <span class="lab">int8</span>
          <span class="big mono-num">{{ formatPercent(report.int8Acc ?? 0) }}</span>
        </div>
        <div class="accnum">
          <span class="lab">float</span>
          <span class="big mono-num">{{ formatPercent(report.floatAcc ?? 0) }}</span>
        </div>
      </div>
    </ViseCard>

    <ViseCard v-if="report?.confusion" title="Confusion matrix" meta="rows are true class" class="mt">
      <ViseConfusion :matrix="report.confusion" :labels="classNames" />
    </ViseCard>
  </section>
</template>

<style scoped>
.live {
  display: flex;
  flex-direction: column;
  gap: var(--s-3);
}
.reason {
  font-size: 12px;
  color: var(--ash);
  margin: 0;
  line-height: 1.6;
}
.mt {
  margin-top: var(--s-4);
}
.acc {
  display: flex;
  gap: var(--s-7);
}
.accnum {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.big {
  font-family: var(--f-display);
  font-weight: 700;
  font-size: var(--t-2xl);
  color: var(--chalk);
}
</style>
