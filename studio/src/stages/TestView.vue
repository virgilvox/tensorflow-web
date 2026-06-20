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
import { useMicrophone } from '../composables/useMicrophone';
import { useMotion } from '../composables/useMotion';
import { fileToFrame } from '../lib/imageDecode';
import { fileToAudio } from '../lib/audioDecode';
import { fileToMotion } from '../lib/motionImport';
import { formatPercent } from '../lib/format';

const project = useProjectStore();
const training = useTrainingStore();
const pipeline = usePipeline();
const camera = useCamera();
const mic = useMicrophone();
const motion = useMotion();

const video = useTemplateRef<HTMLVideoElement>('video');
const fileInput = useTemplateRef<HTMLInputElement>('fileInput');
const audioInput = useTemplateRef<HTMLInputElement>('audioInput');
const motionInput = useTemplateRef<HTMLInputElement>('motionInput');
const predictions = ref<{ name: string; score: number }[]>([]);
const error = ref<string | null>(null);
const recording = ref(false);
const textValue = ref('');

// The confusion matrix is indexed by the class order frozen at train time, so it
// must be labelled with the frozen names from the pipeline, not the live store
// order (which the user can change after exporting).
const classNames = computed(() =>
  pipeline.classNames.value.length
    ? pipeline.classNames.value
    : project.classes.map((c) => c.name),
);
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

async function startMic(): Promise<void> {
  await mic.start();
}

async function recordAndPredict(): Promise<void> {
  error.value = null;
  recording.value = true;
  try {
    const clip = await mic.record();
    predictions.value = await pipeline.predictAudio(clip.data, clip.sampleRate);
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err);
  } finally {
    recording.value = false;
  }
}

async function predictAudioFromFile(event: Event): Promise<void> {
  error.value = null;
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];
  if (!file) return;
  try {
    const clip = await fileToAudio(file);
    predictions.value = await pipeline.predictAudio(clip.data, clip.sampleRate);
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err);
  } finally {
    input.value = '';
  }
}

async function startMotion(): Promise<void> {
  await motion.start();
}

async function recordMotionPredict(): Promise<void> {
  error.value = null;
  recording.value = true;
  try {
    const window = await motion.record();
    predictions.value = await pipeline.predictMotion(window.data, window.axes);
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err);
  } finally {
    recording.value = false;
  }
}

async function predictMotionFromFile(event: Event): Promise<void> {
  error.value = null;
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];
  if (!file) return;
  try {
    const window = await fileToMotion(file);
    predictions.value = await pipeline.predictMotion(window.data, window.axes);
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err);
  } finally {
    input.value = '';
  }
}

async function predictTextLive(): Promise<void> {
  error.value = null;
  const text = textValue.value.trim();
  if (!text) return;
  try {
    predictions.value = await pipeline.predictText(text);
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err);
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
        <div v-if="project.modality === 'image'" class="cam">
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

        <div v-else-if="project.modality === 'audio'" class="cam">
          <div class="camrow">
            <ViseButton v-if="!mic.active.value" size="sm" @click="startMic">
              <ViseIcon name="mic" :size="13" /> Start microphone
            </ViseButton>
            <ViseButton v-else size="sm" :disabled="recording" @click="recordAndPredict">
              <ViseIcon name="mic" :size="13" /> {{ recording ? 'Listening...' : 'Record and predict' }}
            </ViseButton>
            <input
              ref="audioInput"
              type="file"
              accept="audio/*"
              class="hidden-input"
              data-test="predict-audio-file"
              @change="predictAudioFromFile"
            />
            <ViseButton variant="ghost" size="sm" @click="audioInput?.click()">
              <ViseIcon name="mic" :size="13" /> Predict a clip
            </ViseButton>
          </div>
          <p v-if="error" class="err">{{ error }}</p>
        </div>

        <div v-else-if="project.modality === 'motion'" class="cam">
          <div class="camrow">
            <ViseButton v-if="!motion.active.value" size="sm" @click="startMotion">
              <ViseIcon name="motion" :size="13" /> Start sensor
            </ViseButton>
            <ViseButton v-else size="sm" :disabled="recording" @click="recordMotionPredict">
              <ViseIcon name="motion" :size="13" /> {{ recording ? 'Reading...' : 'Record and predict' }}
            </ViseButton>
            <input
              ref="motionInput"
              type="file"
              accept="application/json,.json"
              class="hidden-input"
              data-test="predict-motion-file"
              @change="predictMotionFromFile"
            />
            <ViseButton variant="ghost" size="sm" @click="motionInput?.click()">
              <ViseIcon name="motion" :size="13" /> Predict a window
            </ViseButton>
          </div>
          <p v-if="error" class="err">{{ error }}</p>
        </div>

        <div v-else class="cam textpredict">
          <textarea
            v-model="textValue"
            class="textarea"
            rows="2"
            placeholder="Type a phrase to classify."
            data-test="text-predict-input"
            @keydown.enter.exact.prevent="predictTextLive"
          ></textarea>
          <ViseButton size="sm" :disabled="!textValue.trim()" @click="predictTextLive">
            <ViseIcon name="play" :size="13" /> Predict
          </ViseButton>
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
.textpredict {
  gap: var(--s-3);
  min-width: 240px;
}
.textarea {
  width: 100%;
  background: var(--gunmetal);
  border: 1px solid var(--seam);
  color: var(--chalk);
  font-family: var(--f-mono);
  font-size: 12.5px;
  padding: 9px 11px;
  resize: vertical;
}
.textarea:focus {
  outline: none;
  border-color: var(--live);
  box-shadow: 0 0 0 3px var(--live-glow);
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
