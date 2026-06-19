<script setup lang="ts">
/**
 * Model stage. The architecture, auto sized from the modality and data, shown
 * with its summary and the layer set the export path supports. The layer editor
 * opens at Expert and is restricted to the operators the library can export; an
 * unsupported layer fails loud on export with its name. This stage reads the
 * supported layer list straight from the library so the constraint is honest.
 */
import { computed, ref, watchEffect } from 'vue';
import { supportedLayers } from 'tensorflow-web';
import ViseSectionHead from '../design/components/ViseSectionHead.vue';
import ViseCard from '../design/components/ViseCard.vue';
import ViseBadge from '../design/components/ViseBadge.vue';
import { useProjectStore } from '../stores/project';
import { useSettingsStore } from '../stores/settings';
import { usePipeline } from '../composables/usePipeline';
import { MODALITIES } from '../lib/modalities';
import { formatBytes } from '../lib/format';
import type { ModelSummary } from '../models/types';

const project = useProjectStore();
const settings = useSettingsStore();
const pipeline = usePipeline();
const info = computed(() => MODALITIES[project.modality]);

// The twelve exportable layers, read from the library registry, not hard coded.
const layers = supportedLayers().slice().sort();

// A live size estimate from the preset model, rebuilt as the class count changes.
const summary = ref<ModelSummary | null>(null);
watchEffect(() => {
  // Reference the class count so the estimate refreshes when classes change.
  void project.classes.length;
  try {
    summary.value = pipeline.previewSummary();
  } catch {
    summary.value = null;
  }
});
</script>

<template>
  <section>
    <ViseSectionHead index="03" title="Model" :note="`auto sized for ${info.label.toLowerCase()}`" />

    <ViseCard accent>
      <template #header>
        <span class="ct">Architecture</span>
        <ViseBadge variant="auto">auto</ViseBadge>
        <ViseBadge variant="why" :title="info.modelReason">why</ViseBadge>
      </template>
      <p class="line">{{ info.model }}</p>
      <p class="reason">{{ info.modelReason }}</p>
      <div v-if="summary" class="stats">
        <span class="stat"><span class="lab">params</span><span class="sv mono-num">{{ summary.paramCount.toLocaleString() }}</span></span>
        <span class="stat"><span class="lab">int8 weights</span><span class="sv mono-num">{{ formatBytes(summary.estimatedWeightBytes) }}</span></span>
        <span class="stat"><span class="lab">layers</span><span class="sv mono-num">{{ summary.layerCount }}</span></span>
      </div>
    </ViseCard>

    <ViseCard title="Exportable layers" meta="library constraint" class="mt">
      <p class="reason">
        The library trains from scratch and exports int8. Only these layers serialize to a .tflite;
        anything else trains fine and then fails loud on export with its name.
      </p>
      <div class="chips">
        <ViseBadge v-for="l in layers" :key="l" variant="default">{{ l }}</ViseBadge>
      </div>
    </ViseCard>

    <ViseCard v-if="settings.expert" title="Layer editor" meta="expert" accent class="mt">
      The Expert layer editor builds the network from the supported operators above and refuses to
      add any other, mirroring the export guard. It arrives with the flagship image flow.
    </ViseCard>

    <p v-if="!settings.showsConfigStages" class="guided">
      In Guided the architecture is chosen for you and sized to your data and target. Raise the
      altitude to Standard to see it, or to Expert to edit it.
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
.chips {
  display: flex;
  flex-wrap: wrap;
  gap: var(--s-2);
  margin-top: var(--s-3);
}
.stats {
  display: flex;
  flex-wrap: wrap;
  gap: var(--s-5);
  margin-top: var(--s-4);
  padding-top: var(--s-3);
  border-top: 1px solid var(--seam);
}
.stat {
  display: flex;
  flex-direction: column;
  gap: 3px;
}
.sv {
  font-family: var(--f-display);
  font-weight: 600;
  font-size: 16px;
  color: var(--chalk);
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
