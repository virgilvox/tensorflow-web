<script setup lang="ts">
/**
 * Model stage. The architecture, auto sized from the modality and data, shown
 * with its summary and the layer set the export path supports. The layer editor
 * opens at Expert and is restricted to the operators the library can export; an
 * unsupported layer fails loud on export with its name. This stage reads the
 * supported layer list straight from the library so the constraint is honest.
 */
import { computed } from 'vue';
import { supportedLayers } from 'tensorflow-web';
import ViseSectionHead from '../design/components/ViseSectionHead.vue';
import ViseCard from '../design/components/ViseCard.vue';
import ViseBadge from '../design/components/ViseBadge.vue';
import { useProjectStore } from '../stores/project';
import { useSettingsStore } from '../stores/settings';
import { MODALITIES } from '../lib/modalities';

const project = useProjectStore();
const settings = useSettingsStore();
const info = computed(() => MODALITIES[project.modality]);

// The twelve exportable layers, read from the library registry, not hard coded.
const layers = supportedLayers().slice().sort();
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
