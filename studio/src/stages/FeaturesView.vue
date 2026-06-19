<script setup lang="ts">
/**
 * Features stage. The preprocessing that turns raw samples into tensors, auto
 * chosen per modality and shown read only with a one line reason. The main knobs
 * become editable from Standard up; the full pipeline opens at Expert. A live
 * preview of the processed result lands with each modality's phase.
 */
import { computed } from 'vue';
import ViseSectionHead from '../design/components/ViseSectionHead.vue';
import ViseCard from '../design/components/ViseCard.vue';
import ViseBadge from '../design/components/ViseBadge.vue';
import { useProjectStore } from '../stores/project';
import { useSettingsStore } from '../stores/settings';
import { MODALITIES } from '../lib/modalities';

const project = useProjectStore();
const settings = useSettingsStore();
const info = computed(() => MODALITIES[project.modality]);
</script>

<template>
  <section>
    <ViseSectionHead index="02" title="Features" :note="`auto chosen for ${info.label.toLowerCase()}`" />

    <ViseCard accent>
      <template #header>
        <span class="ct">Pipeline</span>
        <ViseBadge variant="auto">auto</ViseBadge>
        <ViseBadge variant="why" :title="info.featureReason">why</ViseBadge>
      </template>
      <p class="line">{{ info.feature }}</p>
      <p class="reason">{{ info.featureReason }}</p>
    </ViseCard>

    <ViseCard v-if="settings.editable" title="Main knobs" meta="standard" class="mt">
      Image size, channels, and normalization become editable here. The live preview shows the
      processed tensor as you change them.
    </ViseCard>

    <ViseCard v-if="settings.expert" title="Full pipeline" meta="expert" accent class="mt">
      Augmentation, the spectrogram parameters, MFCC versus mel, windowing, and the vocabulary cap
      open at the Expert altitude, each with the same live preview.
    </ViseCard>

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
.guided {
  font-size: 12px;
  color: var(--ash);
  margin-top: var(--s-5);
  line-height: 1.6;
}
</style>
