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
import TwSectionHead from '../design/components/TwSectionHead.vue';
import TwCard from '../design/components/TwCard.vue';
import TwBadge from '../design/components/TwBadge.vue';
import { useProjectStore } from '../stores/project';
import { useSettingsStore } from '../stores/settings';
import { usePipeline } from '../composables/usePipeline';
import { MODALITIES } from '../lib/modalities';
import { formatBytes } from '../lib/format';
import type { ModelSummary } from '../models/types';
import type { LayerInfo } from '../models/builder';

const project = useProjectStore();
const settings = useSettingsStore();
const pipeline = usePipeline();
const info = computed(() => MODALITIES[project.modality]);

// The twelve exportable layers, read from the library registry, not hard coded.
const layers = supportedLayers().slice().sort();

// A live size estimate and operator list from the preset model, rebuilt as the
// class count and the data change.
const summary = ref<ModelSummary | null>(null);
const opLayers = ref<LayerInfo[]>([]);
watchEffect(() => {
  // Reference the data so the estimate refreshes when classes or samples change.
  void project.classes.length;
  void project.samples.length;
  try {
    summary.value = pipeline.previewSummary();
    // The operator inspector is Expert only, so only build its model there.
    opLayers.value = settings.expert ? pipeline.inspectLayers() : [];
  } catch {
    summary.value = null;
    opLayers.value = [];
  }
});
</script>

<template>
  <section>
    <TwSectionHead index="03" title="Model" :note="`auto sized for ${info.label.toLowerCase()}`" />

    <TwCard accent>
      <template #header>
        <span class="ct">Architecture</span>
        <TwBadge variant="auto">auto</TwBadge>
        <TwBadge variant="why" :title="info.modelReason">why</TwBadge>
      </template>
      <p class="line">{{ info.model }}</p>
      <p class="reason">{{ info.modelReason }}</p>
      <div v-if="summary" class="stats">
        <span class="stat"><span class="lab">params</span><span class="sv mono-num">{{ summary.paramCount.toLocaleString() }}</span></span>
        <span class="stat"><span class="lab">int8 weights</span><span class="sv mono-num">{{ formatBytes(summary.estimatedWeightBytes) }}</span></span>
        <span class="stat"><span class="lab">layers</span><span class="sv mono-num">{{ summary.layerCount }}</span></span>
      </div>
    </TwCard>

    <TwCard title="Exportable layers" meta="library constraint" class="mt">
      <p class="reason">
        The library trains from scratch and exports int8. Only these layers serialize to a .tflite;
        anything else trains fine and then fails loud on export with its name.
      </p>
      <div class="chips">
        <TwBadge v-for="l in layers" :key="l" variant="default">{{ l }}</TwBadge>
      </div>
    </TwCard>

    <TwCard v-if="settings.expert && opLayers.length" title="Operator inspector" meta="expert" accent class="mt">
      <table class="ops">
        <thead>
          <tr><th>#</th><th>operator</th><th>output shape</th><th>params</th></tr>
        </thead>
        <tbody>
          <tr v-for="(l, i) in opLayers" :key="l.name">
            <td class="mono-num">{{ i + 1 }}</td>
            <td>{{ l.className }}</td>
            <td class="mono-num">{{ l.outputShape }}</td>
            <td class="mono-num">{{ l.params.toLocaleString() }}</td>
          </tr>
        </tbody>
      </table>
      <p class="reason mt">
        Every operator above is one the export registry supports. The auto architecture is sized to
        your data and target; a full layer editor that builds only from these operators is the
        deferred Expert stretch.
      </p>
    </TwCard>

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
.ops {
  width: 100%;
  border-collapse: collapse;
  font-family: var(--f-mono);
  font-size: 11.5px;
}
.ops th {
  text-align: left;
  font-family: var(--f-label);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-size: 8.5px;
  color: var(--ash);
  border-bottom: 1px solid var(--seam);
  padding: 6px 10px 6px 0;
}
.ops td {
  color: var(--steam);
  border-bottom: 1px solid var(--seam);
  padding: 6px 10px 6px 0;
}
.guided {
  font-size: 12px;
  color: var(--ash);
  margin-top: var(--s-5);
  line-height: 1.6;
}
</style>
