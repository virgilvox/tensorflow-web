<script setup lang="ts">
/**
 * Data stage. Choose the modality, manage the class list including the
 * scaffolded negative class, capture from the webcam or import image files, and
 * see per class counts. The split is by capture session, so each camera session
 * and each import batch gets its own session id and is never split across train
 * and test. Capture for the other modalities lands with their phases.
 */
import { ref, computed, onMounted, useTemplateRef } from 'vue';
import ViseSectionHead from '../design/components/ViseSectionHead.vue';
import ViseCard from '../design/components/ViseCard.vue';
import ViseButton from '../design/components/ViseButton.vue';
import ViseBadge from '../design/components/ViseBadge.vue';
import ViseField from '../design/components/ViseField.vue';
import ViseIcon from '../design/components/ViseIcon.vue';
import ViseStatus from '../design/components/ViseStatus.vue';
import { useProjectStore } from '../stores/project';
import { useSettingsStore } from '../stores/settings';
import { useDataset } from '../composables/useDataset';
import { useCamera, CAPTURE_SIZE } from '../composables/useCamera';
import { fileToFrame } from '../lib/imageDecode';
import { MODALITIES, MODALITY_ORDER } from '../lib/modalities';
import type { Modality } from '../types';

const project = useProjectStore();
const settings = useSettingsStore();
const dataset = useDataset();
const camera = useCamera();

const newClassName = ref('');
const activeClassId = ref<string | null>(null);
const importing = ref(false);
const cameraSession = ref<string | null>(null);
const info = computed(() => MODALITIES[project.modality]);
const video = useTemplateRef<HTMLVideoElement>('video');
const fileInput = useTemplateRef<HTMLInputElement>('fileInput');

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

function pickModality(m: Modality): void {
  project.setModality(m);
  void dataset.persistMeta();
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

async function removeClass(id: string): Promise<void> {
  await dataset.removeClass(id);
  if (activeClassId.value === id) activeClassId.value = project.classes[0]?.id ?? null;
}

const activeClassName = computed(
  () => project.classes.find((c) => c.id === activeClassId.value)?.name ?? null,
);
</script>

<template>
  <section>
    <ViseSectionHead index="01" title="Data" note="collect and label, nothing leaves the browser" />

    <p class="subhead">Modality</p>
    <div class="modgrid">
      <button
        v-for="m in MODALITY_ORDER"
        :key="m"
        type="button"
        class="modcard"
        :class="{ on: project.modality === m }"
        @click="pickModality(m)"
      >
        <ViseIcon :name="MODALITIES[m].icon" :size="22" />
        <span class="ml">{{ MODALITIES[m].label }}</span>
        <span class="mc">{{ MODALITIES[m].capture }}</span>
      </button>
    </div>

    <div class="cols">
      <div class="left">
        <p class="subhead">Classes</p>

        <div class="addrow">
          <ViseField
            v-model="newClassName"
            label="New class name"
            placeholder="e.g. thumbs up"
            @keyup.enter="addClass"
          />
          <ViseButton size="sm" :disabled="!newClassName.trim()" @click="addClass">
            <ViseIcon name="plus" :size="13" /> Add class
          </ViseButton>
        </div>

        <div v-if="!project.negativeClass" class="scaffold">
          <span>
            A {{ info.negative }} class catches everything that is none of your classes. It is a
            first class step, not an afterthought.
          </span>
          <ViseButton size="sm" variant="ghost" @click="ensureNegative">
            Add {{ info.negative }}
          </ViseButton>
        </div>

        <ul v-if="project.classes.length" class="classlist">
          <li v-for="c in project.classes" :key="c.id">
            <button
              type="button"
              class="classbtn"
              :class="{ active: c.id === activeClassId }"
              @click="activeClassId = c.id"
            >
              <ViseCard :accent="c.id === activeClassId">
                <div class="classrow">
                  <span class="cname">
                    {{ c.name }}
                    <ViseBadge v-if="c.negative" variant="auto">negative</ViseBadge>
                    <ViseBadge v-if="c.id === activeClassId" variant="ver-cur">capturing</ViseBadge>
                  </span>
                  <span class="ccount mono-num">{{ project.countsByClass[c.id] ?? 0 }} samples</span>
                  <span class="del" role="button" :aria-label="`Remove ${c.name}`" @click.stop="removeClass(c.id)">
                    <ViseIcon name="trash" :size="14" />
                  </span>
                </div>
              </ViseCard>
            </button>
          </li>
        </ul>
        <p v-else class="empty">No classes yet. Name your first class above.</p>

        <template v-if="project.modality === 'image'">
          <p class="subhead capture-head">
            Capture
            <span v-if="activeClassName" class="into">into {{ activeClassName }}</span>
          </p>
          <ViseCard v-if="!activeClassId" title="Pick a class first" meta="capture target">
            Select a class above, then capture from the webcam or import image files into it.
          </ViseCard>
          <div v-else class="capture">
            <div class="cam">
              <video ref="video" class="preview" playsinline muted></video>
              <div class="camrow">
                <ViseButton v-if="!camera.active.value" size="sm" @click="startCamera">
                  <ViseIcon name="camera" :size="13" /> Start camera
                </ViseButton>
                <template v-else>
                  <ViseButton size="sm" @click="captureFrame">
                    <ViseIcon name="plus" :size="13" /> Capture
                  </ViseButton>
                  <ViseButton size="sm" variant="ghost" @click="camera.stop()">Stop</ViseButton>
                  <ViseStatus state="run">live</ViseStatus>
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
              <ViseButton variant="ghost" size="sm" :disabled="importing" @click="fileInput?.click()">
                <ViseIcon name="image" :size="13" /> {{ importing ? 'Importing...' : 'Import images' }}
              </ViseButton>
              <p class="hint">Each import batch is one capture session.</p>
            </div>
          </div>
        </template>
        <p v-else class="note">
          Capture and import for {{ info.label }} arrive with the {{ info.label }} flow. The split is
          by capture session, never a random row.
        </p>
      </div>

      <aside class="right">
        <ViseCard title="Coaching" meta="good data wins">
          <ul class="coach">
            <li v-for="(tip, i) in info.coaching" :key="i">{{ tip }}</li>
          </ul>
        </ViseCard>

        <ViseCard v-if="settings.expert" title="Expert" meta="data tools" accent>
          Per session review, leakage inspection, and class rebalancing surface here at the Expert
          altitude as the dataset grows.
        </ViseCard>
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
.classbtn {
  display: block;
  width: 100%;
  text-align: left;
  background: none;
  border: none;
  padding: 0;
  cursor: pointer;
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
}
.del:hover {
  color: var(--rust);
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
.coach {
  margin: 0;
  padding-left: 18px;
  display: flex;
  flex-direction: column;
  gap: 6px;
}
</style>
