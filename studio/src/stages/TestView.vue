<script setup lang="ts">
/**
 * Test stage. Live inference on fresh input with real time confidence bars, plus
 * the held out test accuracy and the confusion matrix from verify. Live capture
 * runs the emitted .tflite one sample at a time, as its fixed batch of one
 * requires. An image can come from the webcam or a file, so the flow is testable
 * without a camera.
 */
import { ref, computed, useTemplateRef } from 'vue';
import ViseSectionHead from '../design/components/ViseSectionHead.vue';
import ViseCard from '../design/components/ViseCard.vue';
import ViseConfusion from '../design/components/ViseConfusion.vue';
import ViseStatus from '../design/components/ViseStatus.vue';
import ViseButton from '../design/components/ViseButton.vue';
import ViseIcon from '../design/components/ViseIcon.vue';
import { useProjectStore } from '../stores/project';
import { useTrainingStore } from '../stores/training';
import { usePipeline } from '../composables/usePipeline';
import { useCamera, CAPTURE_SIZE } from '../composables/useCamera';
import { fileToFrame } from '../lib/imageDecode';
import { formatPercent } from '../lib/format';

const project = useProjectStore();
const training = useTrainingStore();
const pipeline = usePipeline();
const camera = useCamera();

const video = useTemplateRef<HTMLVideoElement>('video');
const fileInput = useTemplateRef<HTMLInputElement>('fileInput');
const predictions = ref<{ name: string; score: number }[]>([]);
const error = ref<string | null>(null);

const classNames = computed(() => project.classes.map((c) => c.name));
const report = computed(() => training.report);
const live = computed(() => pipeline.hasModel.value);
const top = computed(() =>
  predictions.value.reduce<{ name: string; score: number } | null>(
    (best, p) => (!best || p.score > best.score ? p : best),
    null,
  ),
);

async function startCamera(): Promise<void> {
  if (video.value) await camera.start(video.value);
}

async function predictFromCamera(): Promise<void> {
  error.value = null;
  const frame = camera.capture();
  if (!frame) return;
  try {
    predictions.value = await pipeline.predictImage(frame);
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err);
  }
}

async function predictFromFile(event: Event): Promise<void> {
  error.value = null;
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];
  if (!file) return;
  try {
    const frame = await fileToFrame(file, CAPTURE_SIZE);
    predictions.value = await pipeline.predictImage(frame);
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err);
  } finally {
    input.value = '';
  }
}
</script>

<template>
  <section>
    <ViseSectionHead index="05" title="Test" note="live and held out" />

    <ViseCard title="Live inference" meta="fresh input">
      <div v-if="!live" class="live">
        <ViseStatus state="hold">awaiting a trained and exported model</ViseStatus>
        <p class="reason">
          Live confidence bars run the emitted model on fresh input. Train a model, then quantize
          and verify it in the Export stage to activate this panel.
        </p>
      </div>

      <div v-else class="livegrid">
        <div class="cam">
          <video ref="video" class="preview" playsinline muted></video>
          <div class="camrow">
            <ViseButton v-if="!camera.active.value" size="sm" @click="startCamera">
              <ViseIcon name="camera" :size="13" /> Start camera
            </ViseButton>
            <ViseButton v-else size="sm" @click="predictFromCamera">
              <ViseIcon name="play" :size="13" /> Predict frame
            </ViseButton>
            <input
              ref="fileInput"
              type="file"
              accept="image/*"
              class="hidden-input"
              data-test="predict-file"
              @change="predictFromFile"
            />
            <ViseButton variant="ghost" size="sm" @click="fileInput?.click()">
              <ViseIcon name="image" :size="13" /> Predict an image
            </ViseButton>
          </div>
          <p v-if="error" class="err">{{ error }}</p>
        </div>

        <div class="bars" data-test="predictions">
          <div v-if="top" class="topline">
            <span class="lab">prediction</span>
            <span class="topname" data-test="top-prediction">{{ top.name }}</span>
          </div>
          <div v-for="p in predictions" :key="p.name" class="bar">
            <span class="bn">{{ p.name }}</span>
            <span class="track"><span class="fill" :style="{ width: `${Math.round(p.score * 100)}%` }"></span></span>
            <span class="bv mono-num">{{ formatPercent(p.score) }}</span>
          </div>
          <p v-if="!predictions.length" class="reason">Capture or pick an image to see confidence.</p>
        </div>
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
.livegrid {
  display: grid;
  grid-template-columns: auto 1fr;
  gap: var(--s-5);
  align-items: start;
}
@media (max-width: 620px) {
  .livegrid {
    grid-template-columns: 1fr;
  }
}
.preview {
  width: 192px;
  height: 192px;
  background: var(--void);
  border: 1px solid var(--seam);
  object-fit: cover;
  display: block;
}
.camrow {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: var(--s-2);
  margin-top: var(--s-3);
}
.hidden-input {
  display: none;
}
.reason {
  font-size: 12px;
  color: var(--ash);
  margin: 0;
  line-height: 1.6;
}
.err {
  font-size: 11px;
  color: var(--rust);
  margin: var(--s-2) 0 0;
}
.bars {
  display: flex;
  flex-direction: column;
  gap: var(--s-2);
}
.topline {
  display: flex;
  align-items: baseline;
  gap: var(--s-3);
  margin-bottom: var(--s-2);
}
.topname {
  font-family: var(--f-display);
  font-weight: 700;
  font-size: var(--t-xl);
  color: var(--live);
}
.bar {
  display: grid;
  grid-template-columns: 90px 1fr 44px;
  align-items: center;
  gap: var(--s-3);
}
.bn {
  font-family: var(--f-mono);
  font-size: 12px;
  color: var(--steam);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.track {
  height: 10px;
  background: var(--steel);
  border: 1px solid var(--seam);
}
.fill {
  display: block;
  height: 100%;
  background: var(--live);
  transition: width var(--fast) var(--ease);
}
.bv {
  font-family: var(--f-label);
  font-size: 10px;
  color: var(--ash);
  text-align: right;
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
