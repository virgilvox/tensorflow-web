<script setup lang="ts">
/**
 * Data stage. Choose the modality, manage the class list including the
 * scaffolded negative class, capture from the webcam or import image files, and
 * see per class counts. The split is by capture session, so each camera session
 * and each import batch gets its own session id and is never split across train
 * and test. Capture for the other modalities lands with their phases.
 */
import { ref, computed, onMounted, useTemplateRef } from 'vue';
import { useRouter } from 'vue-router';
import TwSectionHead from '../design/components/TwSectionHead.vue';
import TwCard from '../design/components/TwCard.vue';
import TwButton from '../design/components/TwButton.vue';
import TwBadge from '../design/components/TwBadge.vue';
import TwField from '../design/components/TwField.vue';
import TwIcon from '../design/components/TwIcon.vue';
import TwStatus from '../design/components/TwStatus.vue';
import { useProjectStore } from '../stores/project';
import {
  useSettingsStore,
  AUDIO_SECONDS_PRESETS,
  AUDIO_SECONDS_MIN,
  AUDIO_SECONDS_MAX,
} from '../stores/settings';
import { useDataset } from '../composables/useDataset';
import { usePipeline, MIN_PER_CLASS } from '../composables/usePipeline';
import { useCamera, CAPTURE_SIZE } from '../composables/useCamera';
import { useMicrophone } from '../composables/useMicrophone';
import { useMotion } from '../composables/useMotion';
import { fileToFrame } from '../lib/imageDecode';
import { fileToAudio } from '../lib/audioDecode';
import { fileToMotion } from '../lib/motionImport';
import { MODALITIES, MODALITY_ORDER } from '../lib/modalities';
import type { Modality } from '../types';

const project = useProjectStore();
const settings = useSettingsStore();
const dataset = useDataset();
const pipeline = usePipeline();
const router = useRouter();
const camera = useCamera();
const mic = useMicrophone();
const motion = useMotion();

const newClassName = ref('');
const activeClassId = ref<string | null>(null);
const importing = ref(false);
const recording = ref(false);
const textValue = ref('');
const cameraSession = ref<string | null>(null);
const micSession = ref<string | null>(null);
const motionSession = ref<string | null>(null);
const info = computed(() => MODALITIES[project.modality]);
const video = useTemplateRef<HTMLVideoElement>('video');
const fileInput = useTemplateRef<HTMLInputElement>('fileInput');
const audioInput = useTemplateRef<HTMLInputElement>('audioInput');
const motionInput = useTemplateRef<HTMLInputElement>('motionInput');

onMounted(async () => {
  await dataset.init();
  if (!activeClassId.value && project.classes.length) {
    activeClassId.value = project.classes[0]!.id;
  }
});

function newSessionId(): string {
  return typeof crypto !== 'undefined' && 'randomUUID' in crypto
    ? `session-${crypto.randomUUID()}`
    : `session-${Date.now()}`;
}

async function pickModality(m: Modality): Promise<void> {
  if (m === project.modality) return;
  // Do not switch (and clear) while a model is training or exporting.
  if (pipeline.busy.value) return;
  // A project is single modality: its samples and classes belong to one. Switching
  // with data present clears it, after confirmation, so no sample is ever left
  // attached to the wrong modality.
  if (project.samples.length > 0 || project.classes.length > 0) {
    const ok = window.confirm(
      `Switching modality clears the current ${project.classes.length} class(es) and ${project.totalSamples} sample(s). Continue?`,
    );
    if (!ok) return;
    await dataset.clearAll();
    // A model trained on the old modality must not survive the switch.
    pipeline.reset();
    activeClassId.value = null;
  }
  project.setModality(m);
  await dataset.persistMeta();
}

async function addClass(): Promise<void> {
  const name = newClassName.value.trim();
  if (!name) return;
  const cls = project.addClass(name);
  activeClassId.value = cls.id;
  newClassName.value = '';
  await dataset.persistMeta();
}

async function ensureNegative(): Promise<void> {
  const cls = project.ensureNegativeClass();
  activeClassId.value = cls.id;
  await dataset.persistMeta();
}

async function startCamera(): Promise<void> {
  if (!video.value) return;
  await camera.start(video.value);
  cameraSession.value = newSessionId();
}

async function captureFrame(): Promise<void> {
  if (!activeClassId.value || !cameraSession.value) return;
  const frame = camera.capture();
  if (frame) await dataset.addImageSample(activeClassId.value, cameraSession.value, frame);
}

async function onFiles(event: Event): Promise<void> {
  const input = event.target as HTMLInputElement;
  const files = input.files ? Array.from(input.files) : [];
  if (!files.length || !activeClassId.value) return;
  importing.value = true;
  const session = newSessionId();
  try {
    for (const file of files) {
      const frame = await fileToFrame(file, CAPTURE_SIZE);
      await dataset.addImageSample(activeClassId.value, session, frame);
    }
  } finally {
    importing.value = false;
    input.value = '';
  }
}

async function startMic(): Promise<void> {
  await mic.start();
  micSession.value = newSessionId();
}

async function recordClip(): Promise<void> {
  if (!activeClassId.value || !micSession.value || recording.value) return;
  recording.value = true;
  try {
    const clip = await mic.record(settings.audioSeconds);
    await dataset.addAudioSample(activeClassId.value, micSession.value, clip);
  } finally {
    recording.value = false;
  }
}

/** Formats the clip length for a button label, trimming a trailing .0. */
const clipLabel = computed<string>(() => {
  const s = settings.audioSeconds;
  return Number.isInteger(s) ? `${s}s` : `${s.toFixed(1)}s`;
});

function onCustomSeconds(event: Event): void {
  const input = event.target as HTMLInputElement;
  settings.setAudioSeconds(Number(input.value));
  // Reflect clamping or rejection of a bad value back into the field, so the
  // input never disagrees with the stored clip length.
  input.value = String(settings.audioSeconds);
}

async function onAudioFiles(event: Event): Promise<void> {
  const input = event.target as HTMLInputElement;
  const files = input.files ? Array.from(input.files) : [];
  if (!files.length || !activeClassId.value) return;
  importing.value = true;
  const session = newSessionId();
  try {
    for (const file of files) {
      const clip = await fileToAudio(file);
      await dataset.addAudioSample(activeClassId.value, session, clip);
    }
  } finally {
    importing.value = false;
    input.value = '';
  }
}

async function startMotion(): Promise<void> {
  await motion.start();
  motionSession.value = newSessionId();
}

async function recordMotion(): Promise<void> {
  if (!activeClassId.value || !motionSession.value || recording.value) return;
  recording.value = true;
  try {
    const window = await motion.record();
    await dataset.addMotionSample(activeClassId.value, motionSession.value, window);
  } finally {
    recording.value = false;
  }
}

async function onMotionFiles(event: Event): Promise<void> {
  const input = event.target as HTMLInputElement;
  const files = input.files ? Array.from(input.files) : [];
  if (!files.length || !activeClassId.value) return;
  importing.value = true;
  const session = newSessionId();
  try {
    for (const file of files) {
      const window = await fileToMotion(file);
      await dataset.addMotionSample(activeClassId.value, session, window);
    }
  } finally {
    importing.value = false;
    input.value = '';
  }
}

async function addTextExample(): Promise<void> {
  const text = textValue.value.trim();
  if (!text || !activeClassId.value) return;
  // Each typed example is its own capture session, so the split distributes them
  // cleanly across train and test without leakage.
  await dataset.addTextSample(activeClassId.value, newSessionId(), text);
  textValue.value = '';
}

async function removeClass(id: string): Promise<void> {
  await dataset.removeClass(id);
  if (activeClassId.value === id) activeClassId.value = project.classes[0]?.id ?? null;
}

const activeClassName = computed(
  () => project.classes.find((c) => c.id === activeClassId.value)?.name ?? null,
);

// Train readiness, surfaced on the Data stage so it is obvious when enough has
// been collected. A class is ready at MIN_PER_CLASS samples; training needs at
// least two classes, each clearing that bar.
const classProgress = computed(() => {
  // Count only samples that can actually train this modality's model, the same
  // basis canTrain() uses, so the cue never says "ready" while Train is disabled.
  const usable = pipeline.usableCounts();
  return project.classes.map((c) => {
    const count = usable[c.id] ?? 0;
    return { id: c.id, name: c.name, count, enough: count >= MIN_PER_CLASS };
  });
});
const ready = computed(() => pipeline.canTrain());
const readyMessage = computed<string>(() => {
  if (project.classes.length < 2) return 'Add at least two classes, then collect samples for each.';
  const short = classProgress.value.filter((c) => !c.enough);
  if (short.length === 0) return `Every class has at least ${MIN_PER_CLASS} samples. You can train.`;
  return `${short.map((c) => `${c.name} needs ${MIN_PER_CLASS - c.count} more`).join(', ')}.`;
});

function goToTrain(): void {
  router.push('/train');
}
</script>

<template>
  <section>
    <TwSectionHead index="01" title="Data" note="collect and label, nothing leaves the browser" />

    <p class="subhead">Modality</p>
    <div class="modgrid">
      <button
        v-for="m in MODALITY_ORDER"
        :key="m"
        type="button"
        class="modcard"
        :class="{ on: project.modality === m }"
        :aria-pressed="project.modality === m"
        @click="pickModality(m)"
      >
        <TwIcon :name="MODALITIES[m].icon" :size="22" />
        <span class="ml">{{ MODALITIES[m].label }}</span>
        <span class="mc">{{ MODALITIES[m].capture }}</span>
      </button>
    </div>

    <div class="cols">
      <div class="left">
        <p class="subhead">Classes</p>

        <div class="addrow">
          <TwField
            v-model="newClassName"
            label="New class name"
            placeholder="e.g. thumbs up"
            @keyup.enter="addClass"
          />
          <TwButton size="sm" :disabled="!newClassName.trim()" @click="addClass">
            <TwIcon name="plus" :size="13" /> Add class
          </TwButton>
        </div>

        <div v-if="!project.negativeClass" class="scaffold">
          <span>
            A {{ info.negative }} class catches everything that is none of your classes. It is a
            first class step, not an afterthought.
          </span>
          <TwButton size="sm" variant="ghost" @click="ensureNegative">
            Add {{ info.negative }}
          </TwButton>
        </div>

        <ul v-if="project.classes.length" class="classlist">
          <li v-for="c in project.classes" :key="c.id">
            <TwCard :accent="c.id === activeClassId">
              <div class="classrow">
                <button
                  type="button"
                  class="selbtn"
                  :class="{ active: c.id === activeClassId }"
                  :aria-pressed="c.id === activeClassId"
                  @click="activeClassId = c.id"
                >
                  <span class="cname">
                    {{ c.name }}
                    <TwBadge v-if="c.negative" variant="auto">negative</TwBadge>
                    <TwBadge v-if="c.id === activeClassId" variant="ver-cur">capturing</TwBadge>
                  </span>
                </button>
                <span class="ccount mono-num">{{ project.countsByClass[c.id] ?? 0 }} samples</span>
                <button
                  type="button"
                  class="del"
                  :aria-label="`Remove ${c.name}`"
                  @click="removeClass(c.id)"
                >
                  <TwIcon name="trash" :size="14" />
                </button>
              </div>
            </TwCard>
          </li>
        </ul>
        <p v-else class="empty">No classes yet. Name your first class above.</p>

        <div v-if="project.classes.length" class="ready" :class="{ go: ready }">
          <div class="rhead">
            <TwStatus :state="ready ? 'pass' : 'hold'">
              {{ ready ? 'Ready to train' : 'Keep collecting' }}
            </TwStatus>
            <TwButton size="sm" :disabled="!ready" data-test="go-train" @click="goToTrain">
              <TwIcon name="train" :size="13" /> Train
            </TwButton>
          </div>
          <p class="rmsg">{{ readyMessage }}</p>
          <ul class="rprog">
            <li v-for="c in classProgress" :key="c.id" :class="{ done: c.enough }">
              <span class="rdot"><TwIcon v-if="c.enough" name="check" :size="11" /></span>
              <span class="rpn">{{ c.name }}</span>
              <span class="rpc mono-num">{{ Math.min(c.count, MIN_PER_CLASS) }}/{{ MIN_PER_CLASS }}</span>
            </li>
          </ul>
        </div>

        <template v-if="project.modality === 'image'">
          <p class="subhead capture-head">
            Capture
            <span v-if="activeClassName" class="into">into {{ activeClassName }}</span>
          </p>
          <TwCard v-if="!activeClassId" title="Pick a class first" meta="capture target">
            Select a class above, then capture from the webcam or import image files into it.
          </TwCard>
          <div v-else class="capture">
            <div class="cam">
              <video ref="video" class="preview" playsinline muted></video>
              <div class="camrow">
                <TwButton v-if="!camera.active.value" size="sm" @click="startCamera">
                  <TwIcon name="camera" :size="13" /> Start camera
                </TwButton>
                <template v-else>
                  <TwButton size="sm" @click="captureFrame">
                    <TwIcon name="plus" :size="13" /> Capture
                  </TwButton>
                  <TwButton size="sm" variant="ghost" @click="camera.stop()">Stop</TwButton>
                  <TwStatus state="run">live</TwStatus>
                </template>
              </div>
              <p v-if="camera.error.value" class="camerr">{{ camera.error.value }}</p>
            </div>

            <div class="import">
              <input
                ref="fileInput"
                type="file"
                accept="image/*"
                multiple
                class="hidden-input"
                data-test="import-file"
                @change="onFiles"
              />
              <TwButton variant="ghost" size="sm" :disabled="importing" @click="fileInput?.click()">
                <TwIcon name="image" :size="13" /> {{ importing ? 'Importing...' : 'Import images' }}
              </TwButton>
              <p class="hint">Each import batch is one capture session.</p>
            </div>
          </div>
        </template>
        <template v-else-if="project.modality === 'audio'">
          <p class="subhead capture-head">
            Capture
            <span v-if="activeClassName" class="into">into {{ activeClassName }}</span>
          </p>
          <TwCard v-if="!activeClassId" title="Pick a class first" meta="capture target">
            Select a class above, then record clips from the microphone or import audio
            files into it. Add a Background Noise class so the model has a rest state.
          </TwCard>
          <div v-else class="capture">
            <div class="cam">
              <div class="cliplen">
                <span class="clab">Clip length</span>
                <div class="cliprow">
                  <button
                    v-for="p in AUDIO_SECONDS_PRESETS"
                    :key="p"
                    type="button"
                    class="chip"
                    :class="{ on: settings.audioSeconds === p }"
                    :aria-pressed="settings.audioSeconds === p"
                    @click="settings.setAudioSeconds(p)"
                  >
                    {{ p }}s
                  </button>
                  <input
                    class="clipnum mono-num"
                    type="number"
                    :min="AUDIO_SECONDS_MIN"
                    :max="AUDIO_SECONDS_MAX"
                    step="0.25"
                    :value="settings.audioSeconds"
                    aria-label="Custom clip length in seconds"
                    @change="onCustomSeconds"
                  />
                  <span class="csuffix">s</span>
                </div>
              </div>
              <div class="camrow">
                <TwButton v-if="!mic.active.value" size="sm" @click="startMic">
                  <TwIcon name="mic" :size="13" /> Start microphone
                </TwButton>
                <template v-else>
                  <TwButton size="sm" :disabled="recording" @click="recordClip">
                    <TwIcon name="mic" :size="13" /> {{ recording ? 'Recording...' : `Record ${clipLabel}` }}
                  </TwButton>
                  <TwButton size="sm" variant="ghost" @click="mic.stop()">Stop</TwButton>
                  <TwStatus :state="recording ? 'run' : 'pass'">{{ recording ? 'recording' : 'ready' }}</TwStatus>
                </template>
              </div>
              <p v-if="mic.error.value" class="camerr">{{ mic.error.value }}</p>
            </div>
            <div class="import">
              <input
                ref="audioInput"
                type="file"
                accept="audio/*"
                multiple
                class="hidden-input"
                data-test="import-audio"
                @change="onAudioFiles"
              />
              <TwButton variant="ghost" size="sm" :disabled="importing" @click="audioInput?.click()">
                <TwIcon name="mic" :size="13" /> {{ importing ? 'Importing...' : 'Import clips' }}
              </TwButton>
              <p class="hint">Each import batch is one capture session.</p>
            </div>
          </div>
        </template>

        <template v-else-if="project.modality === 'motion'">
          <p class="subhead capture-head">
            Capture
            <span v-if="activeClassName" class="into">into {{ activeClassName }}</span>
          </p>
          <TwCard v-if="!activeClassId" title="Pick a class first" meta="capture target">
            Select a class above, then record two second gesture windows from the device motion
            sensor or import recorded windows. Add an Idle class for the rest state.
          </TwCard>
          <div v-else class="capture">
            <div class="cam">
              <svg v-if="motion.trace.value.length" class="trace" viewBox="0 0 120 48" preserveAspectRatio="none">
                <polyline
                  v-for="(axis, ai) in [0, 1, 2]"
                  :key="ai"
                  :points="motion.trace.value.map((s, i) => `${(i / 120) * 120},${24 - (s[axis] ?? 0) * 2}`).join(' ')"
                  :class="`ax ax-${ai}`"
                />
              </svg>
              <div class="camrow">
                <TwButton v-if="!motion.active.value" size="sm" @click="startMotion">
                  <TwIcon name="motion" :size="13" /> Start sensor
                </TwButton>
                <template v-else>
                  <TwButton size="sm" :disabled="recording" @click="recordMotion">
                    <TwIcon name="motion" :size="13" /> {{ recording ? 'Recording...' : 'Record 2s' }}
                  </TwButton>
                  <TwButton size="sm" variant="ghost" @click="motion.stop()">Stop</TwButton>
                </template>
              </div>
              <p v-if="motion.error.value" class="camerr">{{ motion.error.value }}</p>
            </div>
            <div class="import">
              <input
                ref="motionInput"
                type="file"
                accept="application/json,.json"
                multiple
                class="hidden-input"
                data-test="import-motion"
                @change="onMotionFiles"
              />
              <TwButton variant="ghost" size="sm" :disabled="importing" @click="motionInput?.click()">
                <TwIcon name="motion" :size="13" /> {{ importing ? 'Importing...' : 'Import windows' }}
              </TwButton>
              <p class="hint">JSON windows, one batch is one session.</p>
            </div>
          </div>
        </template>

        <template v-else-if="project.modality === 'text'">
          <p class="subhead capture-head">
            Examples
            <span v-if="activeClassName" class="into">into {{ activeClassName }}</span>
          </p>
          <TwCard v-if="!activeClassId" title="Pick a class first" meta="capture target">
            Select a class above, then type or paste example strings into it. Keep classes balanced.
            Add an Other class for everything else.
          </TwCard>
          <div v-else class="textcap">
            <textarea
              v-model="textValue"
              class="textarea"
              rows="2"
              placeholder="Type an example, then Add. One phrase per example."
              data-test="text-input"
              @keydown.enter.exact.prevent="addTextExample"
            ></textarea>
            <TwButton size="sm" :disabled="!textValue.trim()" @click="addTextExample">
              <TwIcon name="plus" :size="13" /> Add example
            </TwButton>
          </div>
        </template>

        <p v-else class="note">Capture for {{ info.label }} is not available.</p>
      </div>

      <aside class="right">
        <TwCard title="Coaching" meta="good data wins">
          <ul class="coach">
            <li v-for="(tip, i) in info.coaching" :key="i">{{ tip }}</li>
          </ul>
        </TwCard>

        <TwCard v-if="settings.expert" title="Expert" meta="data tools" accent>
          Per session review, leakage inspection, and class rebalancing surface here at the Expert
          altitude as the dataset grows.
        </TwCard>
      </aside>
    </div>
  </section>
</template>

<style scoped>
.subhead {
  font-family: var(--f-label);
  text-transform: uppercase;
  letter-spacing: 0.14em;
  font-size: 10px;
  color: var(--ash);
  margin: 0 0 14px;
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
.capture-head {
  margin-top: var(--s-6);
}
.into {
  color: var(--live);
  letter-spacing: 0.08em;
}
.modgrid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
  gap: var(--s-3);
  margin-bottom: var(--s-6);
}
.modcard {
  display: grid;
  grid-template-columns: 22px 1fr;
  grid-template-rows: auto auto;
  gap: 4px 10px;
  text-align: left;
  background: var(--graphite);
  border: 1px solid var(--seam);
  color: var(--steam);
  padding: var(--s-4);
  cursor: pointer;
  transition: transform var(--fast), border-color var(--fast);
}
.modcard:hover {
  transform: translateY(-2px);
  border-color: var(--edge);
}
.modcard.on {
  border-color: var(--live);
  color: var(--chalk);
  box-shadow: var(--cut-live);
}
.modcard .ml {
  font-family: var(--f-display);
  font-weight: 600;
  font-size: 15px;
}
.modcard .mc {
  grid-column: 2;
  font-size: 11px;
  color: var(--ash);
  line-height: 1.5;
}
.cols {
  display: grid;
  grid-template-columns: 1fr 280px;
  gap: var(--s-6);
  align-items: start;
}
@media (max-width: 900px) {
  .cols {
    grid-template-columns: 1fr;
  }
}
.addrow {
  display: grid;
  grid-template-columns: 1fr auto;
  gap: var(--s-3);
  align-items: end;
  margin-bottom: var(--s-4);
}
.addrow :deep(.field) {
  margin-bottom: 0;
}
.scaffold {
  display: flex;
  align-items: center;
  gap: var(--s-4);
  background: var(--iron);
  border: 1px dashed var(--edge);
  padding: var(--s-3) var(--s-4);
  font-size: 12px;
  color: var(--steam);
  margin-bottom: var(--s-4);
}
.classlist {
  list-style: none;
  margin: 0 0 var(--s-2);
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: var(--s-2);
}
.selbtn {
  flex: 1;
  min-width: 0;
  text-align: left;
  background: none;
  border: none;
  padding: 0;
  cursor: pointer;
}
.selbtn:focus-visible {
  outline: 2px solid var(--live);
  outline-offset: 2px;
}
.classrow {
  display: flex;
  align-items: center;
  gap: var(--s-4);
}
.cname {
  font-family: var(--f-mono);
  font-size: 13px;
  color: var(--chalk);
  display: inline-flex;
  align-items: center;
  gap: var(--s-2);
}
.ccount {
  margin-left: auto;
  font-family: var(--f-label);
  font-size: 9.5px;
  letter-spacing: 0.08em;
  color: var(--ash);
}
.del {
  color: var(--ash);
  cursor: pointer;
  padding: 4px;
  display: inline-flex;
  background: none;
  border: none;
}
.del:hover {
  color: var(--rust);
}
.del:focus-visible {
  outline: 2px solid var(--rust);
  outline-offset: 2px;
}
.ready {
  border: 1px solid var(--seam);
  border-left: 2px solid var(--ash);
  background: var(--graphite);
  padding: var(--s-3) var(--s-4);
  margin-bottom: var(--s-4);
}
.ready.go {
  border-left-color: var(--live);
  box-shadow: var(--cut-live);
}
.rhead {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--s-3);
}
.rmsg {
  font-size: 11.5px;
  color: var(--steam);
  line-height: 1.5;
  margin: var(--s-2) 0 0;
}
.rprog {
  list-style: none;
  margin: var(--s-3) 0 0;
  padding: 0;
  display: flex;
  flex-wrap: wrap;
  gap: var(--s-2) var(--s-4);
}
.rprog li {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 11px;
  color: var(--ash);
}
.rprog li.done {
  color: var(--live);
}
.rdot {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 14px;
  height: 14px;
  border: 1px solid var(--edge);
  border-radius: 50%;
  color: var(--live);
}
.rprog li.done .rdot {
  border-color: var(--live);
}
.rpn {
  font-family: var(--f-mono);
}
.rpc {
  font-family: var(--f-label);
  font-size: 9.5px;
  letter-spacing: 0.06em;
}
.empty,
.note {
  font-size: 12px;
  color: var(--ash);
  line-height: 1.6;
}
.capture {
  display: grid;
  grid-template-columns: 1fr auto;
  gap: var(--s-5);
  align-items: start;
}
@media (max-width: 620px) {
  .capture {
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
  align-items: center;
  gap: var(--s-3);
  margin-top: var(--s-3);
}
.cliplen {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.clab {
  font-family: var(--f-label);
  text-transform: uppercase;
  letter-spacing: 0.12em;
  font-size: 9px;
  color: var(--ash);
}
.cliprow {
  display: inline-flex;
  align-items: center;
  gap: 6px;
}
.chip {
  font-family: var(--f-mono);
  font-size: 11px;
  color: var(--steam);
  background: var(--gunmetal);
  border: 1px solid var(--seam);
  padding: 4px 9px;
  cursor: pointer;
  transition: color var(--fast), border-color var(--fast);
}
.chip:hover {
  border-color: var(--edge);
  color: var(--chalk);
}
.chip.on {
  border-color: var(--live);
  color: var(--live);
}
.clipnum {
  width: 5ch;
  background: var(--gunmetal);
  border: 1px solid var(--seam);
  color: var(--chalk);
  font-size: 11px;
  padding: 4px 6px;
}
.clipnum:focus {
  outline: none;
  border-color: var(--live);
  box-shadow: 0 0 0 3px var(--live-glow);
}
.csuffix {
  font-family: var(--f-label);
  font-size: 10px;
  color: var(--ash);
}
.camerr {
  font-size: 11px;
  color: var(--rust);
  margin: var(--s-2) 0 0;
  max-width: 192px;
}
.hidden-input {
  display: none;
}
.hint {
  font-size: 10px;
  color: var(--ash);
  margin: var(--s-2) 0 0;
  max-width: 150px;
}
.trace {
  width: 220px;
  height: 88px;
  background: var(--void);
  border: 1px solid var(--seam);
  display: block;
  margin-bottom: var(--s-3);
}
.trace .ax {
  fill: none;
  stroke-width: 1;
  vector-effect: non-scaling-stroke;
}
.trace .ax-0 {
  stroke: var(--live);
}
.trace .ax-1 {
  stroke: var(--patina);
}
.trace .ax-2 {
  stroke: var(--amber);
}
.textcap {
  display: flex;
  flex-direction: column;
  gap: var(--s-3);
  align-items: flex-start;
  max-width: 460px;
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
.coach {
  margin: 0;
  padding-left: 18px;
  display: flex;
  flex-direction: column;
  gap: 6px;
}
</style>
