<script setup lang="ts">
/**
 * Features stage. The preprocessing that turns raw samples into tensors, auto
 * chosen per modality and shown with a one line reason. From Standard up the main
 * knobs become real, editable controls bound to the settings store, with a live
 * preview of the resulting tensor shape. Changes apply the next time you train.
 */
import { computed } from 'vue';
import TwSectionHead from '../design/components/TwSectionHead.vue';
import TwCard from '../design/components/TwCard.vue';
import TwBadge from '../design/components/TwBadge.vue';
import { useProjectStore } from '../stores/project';
import {
  useSettingsStore,
  AUDIO_SECONDS_PRESETS,
  AUDIO_SECONDS_MIN,
  AUDIO_SECONDS_MAX,
} from '../stores/settings';
import { buildFeatureConfig, featureShape } from '../features';
import { MODALITIES } from '../lib/modalities';

const project = useProjectStore();
const settings = useSettingsStore();
const info = computed(() => MODALITIES[project.modality]);

const IMAGE_SIZES = [32, 48, 64, 96];
const AUDIO_BANDS = [16, 32, 40];
const MOTION_STEPS = [16, 32, 64];
const VOCAB_CAPS = [50, 100, 200, 500];

// Live preview of the input tensor shape the current knobs produce. Text depends
// on the collected vocabulary, so it reads "needs data" until samples exist.
const shapeLabel = computed<string>(() => {
  const cfg = buildFeatureConfig(project.modality, project.samples, {
    audioSeconds: settings.audioSeconds,
    imageSize: settings.imageSize,
    imageChannels: settings.imageChannels,
    audioMode: settings.audioMode,
    audioBands: settings.audioBands,
    motionSteps: settings.motionSteps,
    textVocabCap: settings.textVocabCap,
  });
  const s = featureShape(cfg);
  return s.length > 0 && s.every((d) => d > 0) ? `[${s.join(', ')}]` : 'needs data';
});

function onClipSeconds(event: Event): void {
  const input = event.target as HTMLInputElement;
  settings.setAudioSeconds(Number(input.value));
  input.value = String(settings.audioSeconds);
}
</script>

<template>
  <section>
    <TwSectionHead index="02" title="Features" :note="`auto chosen for ${info.label.toLowerCase()}`" />

    <TwCard accent>
      <template #header>
        <span class="ct">Pipeline</span>
        <TwBadge variant="auto">auto</TwBadge>
        <TwBadge variant="why" :title="info.featureReason">why</TwBadge>
      </template>
      <p class="line">{{ info.feature }}</p>
      <p class="reason">{{ info.featureReason }}</p>
    </TwCard>

    <TwCard v-if="settings.editable" title="Main knobs" meta="standard" class="mt">
      <div class="knobs">
        <template v-if="project.modality === 'image'">
          <div class="ctl">
            <span class="klab">Image size</span>
            <select v-model.number="settings.imageSize" class="sel mono-num" aria-label="Image size">
              <option v-for="s in IMAGE_SIZES" :key="s" :value="s">{{ s }} px</option>
            </select>
          </div>
          <div class="ctl">
            <span class="klab">Color</span>
            <div class="chips">
              <button type="button" class="chip" :class="{ on: settings.imageChannels === 1 }" :aria-pressed="settings.imageChannels === 1" @click="settings.imageChannels = 1">Grayscale</button>
              <button type="button" class="chip" :class="{ on: settings.imageChannels === 3 }" :aria-pressed="settings.imageChannels === 3" @click="settings.imageChannels = 3">RGB</button>
            </div>
          </div>
        </template>

        <template v-else-if="project.modality === 'audio'">
          <div class="ctl">
            <span class="klab">Clip length</span>
            <div class="chips">
              <button v-for="p in AUDIO_SECONDS_PRESETS" :key="p" type="button" class="chip" :class="{ on: settings.audioSeconds === p }" :aria-pressed="settings.audioSeconds === p" @click="settings.setAudioSeconds(p)">{{ p }}s</button>
              <input class="num mono-num" type="number" :min="AUDIO_SECONDS_MIN" :max="AUDIO_SECONDS_MAX" step="0.25" :value="settings.audioSeconds" aria-label="Custom clip length" @change="onClipSeconds" />
              <span class="suffix">s</span>
            </div>
          </div>
          <div class="ctl">
            <span class="klab">Output</span>
            <div class="chips">
              <button type="button" class="chip" :class="{ on: settings.audioMode === 'mel' }" :aria-pressed="settings.audioMode === 'mel'" @click="settings.audioMode = 'mel'">Mel spectrogram</button>
              <button type="button" class="chip" :class="{ on: settings.audioMode === 'mfcc' }" :aria-pressed="settings.audioMode === 'mfcc'" @click="settings.audioMode = 'mfcc'">MFCC</button>
            </div>
          </div>
          <div class="ctl">
            <span class="klab">Mel bands</span>
            <select v-model.number="settings.audioBands" class="sel mono-num" aria-label="Mel bands">
              <option v-for="b in AUDIO_BANDS" :key="b" :value="b">{{ b }}</option>
            </select>
          </div>
        </template>

        <template v-else-if="project.modality === 'motion'">
          <div class="ctl">
            <span class="klab">Window steps</span>
            <select v-model.number="settings.motionSteps" class="sel mono-num" aria-label="Window steps">
              <option v-for="s in MOTION_STEPS" :key="s" :value="s">{{ s }}</option>
            </select>
          </div>
        </template>

        <template v-else>
          <div class="ctl">
            <span class="klab">Vocabulary cap</span>
            <select v-model.number="settings.textVocabCap" class="sel mono-num" aria-label="Vocabulary cap">
              <option v-for="c in VOCAB_CAPS" :key="c" :value="c">{{ c }} words</option>
            </select>
          </div>
        </template>

        <div class="ctl">
          <span class="klab">Input shape</span>
          <span class="shape mono-num" data-test="feature-shape">{{ shapeLabel }}</span>
        </div>
      </div>
      <p class="reason mt2">Changes apply the next time you train.</p>
    </TwCard>

    <p v-if="!settings.showsConfigStages" class="guided">
      In Guided the feature pipeline is automatic and stays out of your way. Raise the altitude to
      Standard to see and adjust it.
    </p>
  </section>
</template>

<style scoped>
.ct {
  font-family: var(--f-display);
  font-weight: 600;
  font-size: 14px;
  letter-spacing: 0.03em;
}
.line {
  font-family: var(--f-mono);
  font-size: 13px;
  color: var(--chalk);
  margin: 0 0 8px;
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
.mt2 {
  margin-top: var(--s-4);
}
.knobs {
  display: flex;
  flex-direction: column;
  gap: var(--s-3);
}
.ctl {
  display: grid;
  grid-template-columns: 120px 1fr;
  align-items: center;
  gap: var(--s-4);
}
.klab {
  font-family: var(--f-label);
  text-transform: uppercase;
  letter-spacing: 0.1em;
  font-size: 9.5px;
  color: var(--ash);
}
.chips {
  display: inline-flex;
  flex-wrap: wrap;
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
.sel,
.num {
  background: var(--gunmetal);
  border: 1px solid var(--seam);
  color: var(--chalk);
  font-size: 11px;
  padding: 4px 7px;
}
.num {
  width: 5ch;
}
.sel:focus,
.num:focus {
  outline: none;
  border-color: var(--live);
  box-shadow: 0 0 0 3px var(--live-glow);
}
.suffix {
  font-family: var(--f-label);
  font-size: 10px;
  color: var(--ash);
}
.shape {
  font-size: 12px;
  color: var(--live);
}
.guided {
  font-size: 12px;
  color: var(--ash);
  margin-top: var(--s-5);
  line-height: 1.6;
}
</style>
