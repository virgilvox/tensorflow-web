<script setup lang="ts">
/**
 * Playground. Runs a model live against fresh camera, microphone, motion, or
 * text input, independent of the training project. It loads the current session
 * model, a self-contained bundle (.tfwsmodel.json), or a bare .tflite the user
 * configures by hand. This is where you confirm an exported model actually works
 * before you flash it, or replay an old one without reopening its project.
 */
import { ref, computed, onUnmounted, useTemplateRef } from 'vue';
import TwSectionHead from '../design/components/TwSectionHead.vue';
import TwCard from '../design/components/TwCard.vue';
import TwButton from '../design/components/TwButton.vue';
import TwStatus from '../design/components/TwStatus.vue';
import TwBadge from '../design/components/TwBadge.vue';
import TwIcon from '../design/components/TwIcon.vue';
import { usePlayground } from '../composables/usePlayground';
import { usePipeline } from '../composables/usePipeline';
import { useProjectStore } from '../stores/project';
import { useSettingsStore } from '../stores/settings';
import { useCamera, CAPTURE_SIZE } from '../composables/useCamera';
import { useMicrophone } from '../composables/useMicrophone';
import { useMotion } from '../composables/useMotion';
import { fileToFrame } from '../lib/imageDecode';
import { fileToAudio } from '../lib/audioDecode';
import { fileToMotion } from '../lib/motionImport';
import { MODALITIES } from '../lib/modalities';
import { DEFAULT_AUDIO_CONFIG, audioTensorShape } from '../features/audio';
import { DEFAULT_MOTION_CONFIG, motionTensorShape } from '../features/motion';
import { imageTensorShape } from '../features/image';
import type { FeatureConfig } from '../features';
import type { Modality } from '../types';

const playground = usePlayground();
const pipeline = usePipeline();
const project = useProjectStore();
const settings = useSettingsStore();
const camera = useCamera();
const mic = useMicrophone();
const motion = useMotion();

const video = useTemplateRef<HTMLVideoElement>('video');
const fileInput = useTemplateRef<HTMLInputElement>('fileInput');
const imgInput = useTemplateRef<HTMLInputElement>('imgInput');
const audioInput = useTemplateRef<HTMLInputElement>('audioInput');
const motionInput = useTemplateRef<HTMLInputElement>('motionInput');

const predictions = ref<{ name: string; score: number }[]>([]);
const error = ref<string | null>(null);
const loadError = ref<string | null>(null);
const recording = ref(false);
const textValue = ref('');
// Continuous live modes: keep predicting on a rolling window instead of one shot.
const listening = ref(false);
const liveCam = ref(false);
const loading = ref(false);
let audioTimer: ReturnType<typeof setTimeout> | null = null;
let camTimer: ReturnType<typeof setTimeout> | null = null;
let audioInFlight = false;
let camInFlight = false;

const active = computed(() => playground.active.value);
const ready = computed(() => playground.ready());
const canUseCurrent = computed(() => pipeline.hasModel.value && pipeline.featureCfg.value !== null);
const recSeconds = computed(() => active.value?.audioSeconds ?? settings.audioSeconds);
const top = computed(() =>
  predictions.value.reduce<{ name: string; score: number } | null>(
    (best, p) => (!best || p.score > best.score ? p : best),
    null,
  ),
);

// Bare .tflite setup. A foreign model carries no labels or preprocessing, so the
// user supplies them; text needs a vocabulary, so it is bundle only.
const bareModalities: readonly Modality[] = ['image', 'audio', 'motion'];
const showBare = ref(false);
const bareHint = ref<number[] | null>(null);
const bareModality = ref<Modality>('image');
const bareLabels = ref('');
const bareImageSize = ref(48);
const bareImageChannels = ref<1 | 3>(1);

/** Stops both continuous loops without touching the loaded model. */
function stopLoops(): void {
  listening.value = false;
  liveCam.value = false;
  if (audioTimer) {
    clearTimeout(audioTimer);
    audioTimer = null;
  }
  if (camTimer) {
    clearTimeout(camTimer);
    camTimer = null;
  }
}

function resetCapture(): void {
  stopLoops();
  predictions.value = [];
  error.value = null;
  camera.stop();
  mic.stop();
  motion.stop();
}

onUnmounted(stopLoops);

async function useCurrent(): Promise<void> {
  const cfg = pipeline.featureCfg.value;
  if (!pipeline.bytes.value || !cfg || loading.value) return;
  loadError.value = null;
  resetCapture();
  loading.value = true;
  try {
    await playground.loadCurrent({
      name: project.name,
      modality: project.modality,
      classes: pipeline.classNames.value,
      featureConfig: cfg,
      inputShape: pipeline.fShape.value,
      audioSeconds: project.modality === 'audio' ? settings.audioSeconds : undefined,
      bytes: pipeline.bytes.value,
    });
  } catch (err) {
    loadError.value = err instanceof Error ? err.message : String(err);
  } finally {
    loading.value = false;
  }
}

async function onModelFile(event: Event): Promise<void> {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];
  input.value = '';
  if (!file || loading.value) return;
  loadError.value = null;
  showBare.value = false;
  bareHint.value = null;
  resetCapture();
  loading.value = true;
  try {
    if (file.name.toLowerCase().endsWith('.json')) {
      await playground.loadBundle(await file.text());
    } else {
      const bytes = new Uint8Array(await file.arrayBuffer());
      bareHint.value = await playground.inspectTflite(bytes);
      prefillBareForm();
      showBare.value = true;
    }
  } catch (err) {
    loadError.value = err instanceof Error ? err.message : String(err);
  } finally {
    loading.value = false;
  }
}

/** Seeds the bare setup from the interpreter's reported input shape. */
function prefillBareForm(): void {
  const hint = bareHint.value;
  if (hint && hint.length === 3) {
    bareModality.value = 'image';
    bareImageSize.value = hint[0]!;
    bareImageChannels.value = hint[2] === 3 ? 3 : 1;
  } else if (hint && hint.length === 1) {
    bareModality.value = 'motion';
  }
}

function bareLabelList(): string[] {
  return bareLabels.value
    .split(',')
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
}

function confirmBare(): void {
  loadError.value = null;
  let featureConfig: FeatureConfig;
  let inputShape: number[];
  if (bareModality.value === 'image') {
    const image = { size: bareImageSize.value, channels: bareImageChannels.value, normalize: true };
    featureConfig = { kind: 'image', image };
    inputShape = imageTensorShape(image);
  } else if (bareModality.value === 'audio') {
    const audio = { ...DEFAULT_AUDIO_CONFIG };
    featureConfig = { kind: 'audio', audio };
    inputShape = audioTensorShape(audio);
  } else {
    const motionCfg = { ...DEFAULT_MOTION_CONFIG };
    featureConfig = { kind: 'motion', motion: motionCfg };
    inputShape = motionTensorShape(motionCfg);
  }
  // If the interpreter reported its input shape, the configured preprocessing must
  // produce exactly that, or the model would throw on the first predict. Refuse up
  // front with a clear message instead of failing late, and point at the bundle
  // path which carries matching preprocessing.
  const hint = bareHint.value;
  if (hint && (hint.length !== inputShape.length || hint.some((d, i) => d !== inputShape[i]))) {
    loadError.value = `This model expects input [${hint.join(', ')}], but the ${bareModality.value} preprocessing here produces [${inputShape.join(', ')}]. Export a bundle from the training project, which carries matching preprocessing.`;
    return;
  }
  try {
    playground.activateBare({
      name: 'uploaded model',
      modality: bareModality.value,
      classes: bareLabelList(),
      featureConfig,
      inputShape,
    });
    showBare.value = false;
  } catch (err) {
    loadError.value = err instanceof Error ? err.message : String(err);
  }
}

function unload(): void {
  resetCapture();
  playground.clear();
}

/** Toggles continuous camera prediction: start the camera and predict every frame. */
async function toggleLive(): Promise<void> {
  if (liveCam.value) {
    stopLive();
    return;
  }
  error.value = null;
  try {
    if (!camera.active.value && video.value) await camera.start(video.value);
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err);
    return;
  }
  liveCam.value = true;
  const tick = async (): Promise<void> => {
    if (!liveCam.value) return;
    if (!camInFlight) {
      camInFlight = true;
      try {
        const frame = camera.capture();
        if (frame) predictions.value = await playground.predictImage(frame);
      } catch (err) {
        error.value = err instanceof Error ? err.message : String(err);
      } finally {
        camInFlight = false;
      }
    }
    if (liveCam.value) camTimer = setTimeout(() => void tick(), 200);
  };
  void tick();
}
function stopLive(): void {
  liveCam.value = false;
  if (camTimer) {
    clearTimeout(camTimer);
    camTimer = null;
  }
  camera.stop();
}
async function predictImageFile(event: Event): Promise<void> {
  error.value = null;
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];
  if (!file) return;
  try {
    predictions.value = await playground.predictImage(await fileToFrame(file, CAPTURE_SIZE));
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err);
  } finally {
    input.value = '';
  }
}

/** Toggles continuous listening: keep the mic open and predict a rolling window. */
async function toggleListen(): Promise<void> {
  if (listening.value) {
    stopListening();
    return;
  }
  error.value = null;
  try {
    if (!mic.active.value) await mic.start();
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err);
    return;
  }
  listening.value = true;
  const tick = async (): Promise<void> => {
    if (!listening.value) return;
    if (!audioInFlight) {
      audioInFlight = true;
      try {
        const win = mic.latestWindow(recSeconds.value);
        if (win) predictions.value = await playground.predictAudio(win.data, win.sampleRate);
      } catch (err) {
        error.value = err instanceof Error ? err.message : String(err);
      } finally {
        audioInFlight = false;
      }
    }
    if (listening.value) audioTimer = setTimeout(() => void tick(), 250);
  };
  void tick();
}
function stopListening(): void {
  listening.value = false;
  if (audioTimer) {
    clearTimeout(audioTimer);
    audioTimer = null;
  }
  mic.stop();
}
async function predictAudioFile(event: Event): Promise<void> {
  error.value = null;
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];
  if (!file) return;
  try {
    const clip = await fileToAudio(file);
    predictions.value = await playground.predictAudio(clip.data, clip.sampleRate);
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
    predictions.value = await playground.predictMotion(window.data, window.axes);
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err);
  } finally {
    recording.value = false;
  }
}
async function predictMotionFile(event: Event): Promise<void> {
  error.value = null;
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];
  if (!file) return;
  try {
    const window = await fileToMotion(file);
    predictions.value = await playground.predictMotion(window.data, window.axes);
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
    predictions.value = await playground.predictText(text);
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err);
  }
}
</script>

<template>
  <section>
    <TwSectionHead title="Playground" note="run a model live on fresh input" />

    <TwCard v-if="!ready" title="Interpreter offline" meta="needs the CDN">
      The TFLite interpreter is not loaded. Confirm the tfjs and tfjs-tflite CDN scripts in
      index.html are reachable, then reload.
    </TwCard>

    <template v-else>
      <TwCard v-if="!active" title="Load a model" meta="three ways in">
        <div class="sources">
          <div class="src">
            <p class="srch">Current project</p>
            <TwButton size="sm" :disabled="!canUseCurrent || loading" data-test="use-current" @click="useCurrent">
              <TwIcon name="play" :size="13" /> Use the trained model
            </TwButton>
            <p class="srchint">
              {{ canUseCurrent ? 'Run the model you just trained and exported.' : 'Train and export a model to run it here.' }}
            </p>
          </div>

          <div class="src">
            <p class="srch">A file</p>
            <input
              ref="fileInput"
              type="file"
              accept=".json,.tflite,application/json,application/octet-stream"
              class="hidden-input"
              data-test="load-model-file"
              @change="onModelFile"
            />
            <TwButton size="sm" variant="ghost" :disabled="loading" @click="fileInput?.click()">
              <TwIcon name="open" :size="13" /> Load a bundle or .tflite
            </TwButton>
            <p class="srchint">
              A studio bundle (.tfwsmodel.json) runs straightaway. A bare .tflite needs a quick
              setup below.
            </p>
          </div>
        </div>
        <p v-if="loadError" class="err" data-test="load-error">{{ loadError }}</p>

        <div v-if="showBare" class="bare" data-test="bare-setup">
          <p class="srch">Set up the .tflite</p>
          <p class="srchint">
            A bare model carries no labels or preprocessing.
            <template v-if="bareHint">The interpreter reports an input shape of [{{ bareHint.join(', ') }}].</template>
            Text models need a bundle for their vocabulary.
          </p>
          <div class="barerow">
            <span class="blab">Modality</span>
            <div class="chips">
              <button
                v-for="m in bareModalities"
                :key="m"
                type="button"
                class="chip"
                :class="{ on: bareModality === m }"
                :aria-pressed="bareModality === m"
                @click="bareModality = m"
              >
                <TwIcon :name="MODALITIES[m].icon" :size="13" /> {{ MODALITIES[m].label }}
              </button>
            </div>
          </div>
          <div v-if="bareModality === 'image'" class="barerow">
            <span class="blab">Image input</span>
            <input v-model.number="bareImageSize" class="num mono-num" type="number" min="8" max="224" aria-label="Image size" />
            <span class="x">px,</span>
            <select v-model.number="bareImageChannels" class="sel" aria-label="Channels">
              <option :value="1">grayscale</option>
              <option :value="3">RGB</option>
            </select>
          </div>
          <div class="barerow">
            <span class="blab">Class labels</span>
            <input
              v-model="bareLabels"
              class="txt"
              type="text"
              placeholder="comma separated, in output order"
              aria-label="Class labels"
            />
          </div>
          <TwButton size="sm" data-test="confirm-bare" @click="confirmBare">
            <TwIcon name="check" :size="13" /> Load this model
          </TwButton>
        </div>
      </TwCard>

      <template v-else>
        <TwCard title="Running" meta="loaded model" accent>
          <div class="actv">
            <div class="actvinfo">
              <span class="aname">{{ active.name }}</span>
              <TwBadge variant="auto">{{ active.modality }}</TwBadge>
              <TwBadge variant="ver-cur">{{ active.source }}</TwBadge>
            </div>
            <div class="actvmeta">
              <span class="mono-num">input [{{ active.inputShape.join(', ') }}]</span>
              <span>{{ active.classes.length || 'unlabeled' }} {{ active.classes.length === 1 ? 'class' : 'classes' }}</span>
            </div>
            <TwButton size="sm" variant="ghost" data-test="unload" @click="unload">
              <TwIcon name="x" :size="13" /> Unload
            </TwButton>
          </div>
        </TwCard>

        <TwCard title="Live inference" meta="fresh input" class="mt">
          <div class="livegrid">
            <div v-if="active.modality === 'image'" class="cam">
              <video ref="video" class="preview" playsinline muted></video>
              <div class="camrow">
                <TwButton v-if="!liveCam" size="sm" @click="toggleLive">
                  <TwIcon name="camera" :size="13" /> Go live
                </TwButton>
                <TwButton v-else size="sm" variant="ghost" @click="toggleLive">
                  <TwIcon name="stop" :size="13" /> Stop
                </TwButton>
                <TwStatus v-if="liveCam" state="run">live</TwStatus>
                <input ref="imgInput" type="file" accept="image/*" class="hidden-input" data-test="pg-predict-image" @change="predictImageFile" />
                <TwButton variant="ghost" size="sm" @click="imgInput?.click()">
                  <TwIcon name="image" :size="13" /> Predict an image
                </TwButton>
              </div>
              <p class="reason">Goes live and predicts every frame. Or predict a single image.</p>
              <p v-if="camera.error.value" class="err">{{ camera.error.value }}</p>
            </div>

            <div v-else-if="active.modality === 'audio'" class="cam">
              <div class="camrow">
                <TwButton v-if="!listening" size="sm" data-test="pg-listen" @click="toggleListen">
                  <TwIcon name="mic" :size="13" /> Listen
                </TwButton>
                <TwButton v-else size="sm" variant="ghost" @click="toggleListen">
                  <TwIcon name="stop" :size="13" /> Stop
                </TwButton>
                <TwStatus v-if="listening" state="run">listening</TwStatus>
                <input ref="audioInput" type="file" accept="audio/*" class="hidden-input" data-test="pg-predict-audio" @change="predictAudioFile" />
                <TwButton variant="ghost" size="sm" @click="audioInput?.click()">
                  <TwIcon name="mic" :size="13" /> Predict a clip
                </TwButton>
              </div>
              <p class="reason">Listens continuously and predicts a rolling {{ recSeconds }}s window. Or predict a single clip.</p>
              <p v-if="mic.error.value" class="err">{{ mic.error.value }}</p>
            </div>

            <div v-else-if="active.modality === 'motion'" class="cam">
              <div class="camrow">
                <TwButton v-if="!motion.active.value" size="sm" @click="startMotion">
                  <TwIcon name="motion" :size="13" /> Start sensor
                </TwButton>
                <TwButton v-else size="sm" :disabled="recording" @click="recordMotionPredict">
                  <TwIcon name="motion" :size="13" /> {{ recording ? 'Reading...' : 'Record and predict' }}
                </TwButton>
                <input ref="motionInput" type="file" accept="application/json,.json" class="hidden-input" @change="predictMotionFile" />
                <TwButton variant="ghost" size="sm" @click="motionInput?.click()">
                  <TwIcon name="motion" :size="13" /> Predict a window
                </TwButton>
              </div>
              <p v-if="motion.error.value" class="err">{{ motion.error.value }}</p>
            </div>

            <div v-else class="cam textpredict">
              <textarea
                v-model="textValue"
                class="textarea"
                rows="2"
                placeholder="Type a phrase to classify."
                @keydown.enter.exact.prevent="predictTextLive"
              ></textarea>
              <TwButton size="sm" :disabled="!textValue.trim()" @click="predictTextLive">
                <TwIcon name="play" :size="13" /> Predict
              </TwButton>
              <p v-if="error" class="err">{{ error }}</p>
            </div>

            <div class="bars" data-test="predictions">
              <div v-if="top" class="topline">
                <span class="lab">prediction</span>
                <span class="topname" data-test="top-prediction">{{ top.name }}</span>
              </div>
              <div v-for="(p, i) in predictions" :key="i" class="bar">
                <span class="bn">{{ p.name }}</span>
                <span class="track"><span class="fill" :style="{ width: `${Math.round(p.score * 100)}%` }"></span></span>
                <span class="bv mono-num">{{ Math.round(p.score * 100) }}%</span>
              </div>
              <p v-if="!predictions.length" class="reason">Capture or pick input to see confidence.</p>
              <p v-if="error && active.modality !== 'text'" class="err">{{ error }}</p>
            </div>
          </div>
        </TwCard>
      </template>
    </template>
  </section>
</template>

<style scoped>
.sources {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: var(--s-5);
}
@media (max-width: 620px) {
  .sources {
    grid-template-columns: 1fr;
  }
}
.src {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: var(--s-2);
}
.srch {
  font-family: var(--f-label);
  text-transform: uppercase;
  letter-spacing: 0.12em;
  font-size: 9.5px;
  color: var(--ash);
  margin: 0;
}
.srchint {
  font-size: 11px;
  color: var(--ash);
  line-height: 1.5;
  margin: 0;
}
.hidden-input {
  display: none;
}
.bare {
  margin-top: var(--s-5);
  border-top: 1px solid var(--seam);
  padding-top: var(--s-4);
  display: flex;
  flex-direction: column;
  gap: var(--s-3);
  align-items: flex-start;
}
.barerow {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: var(--s-3);
}
.blab {
  font-family: var(--f-label);
  font-size: 10px;
  letter-spacing: 0.08em;
  color: var(--steam);
  min-width: 80px;
}
.chips {
  display: inline-flex;
  gap: 6px;
}
.chip {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  font-family: var(--f-mono);
  font-size: 11px;
  color: var(--steam);
  background: var(--gunmetal);
  border: 1px solid var(--seam);
  padding: 4px 9px;
  cursor: pointer;
}
.chip.on {
  border-color: var(--live);
  color: var(--live);
}
.num {
  width: 6ch;
}
.num,
.sel,
.txt {
  background: var(--gunmetal);
  border: 1px solid var(--seam);
  color: var(--chalk);
  font-size: 11px;
  padding: 4px 7px;
}
.txt {
  min-width: 240px;
}
.num:focus,
.sel:focus,
.txt:focus {
  outline: none;
  border-color: var(--live);
  box-shadow: 0 0 0 3px var(--live-glow);
}
.x {
  font-size: 11px;
  color: var(--ash);
}
.actv {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: var(--s-4);
}
.actvinfo {
  display: inline-flex;
  align-items: center;
  gap: var(--s-2);
}
.aname {
  font-family: var(--f-mono);
  font-size: 13px;
  color: var(--chalk);
}
.actvmeta {
  display: inline-flex;
  gap: var(--s-4);
  font-size: 11px;
  color: var(--ash);
  margin-left: auto;
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
.lab {
  font-family: var(--f-label);
  text-transform: uppercase;
  letter-spacing: 0.12em;
  font-size: 9px;
  color: var(--ash);
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
</style>
