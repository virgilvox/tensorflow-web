<script setup lang="ts">
/**
 * Data stage. Choose the modality, manage the class list including the
 * scaffolded negative class, and see per class counts. Capture and import for
 * each modality land with that modality's phase; this stage owns the class model
 * and the coaching that frames good data collection.
 */
import { ref, computed } from 'vue';
import ViseSectionHead from '../design/components/ViseSectionHead.vue';
import ViseCard from '../design/components/ViseCard.vue';
import ViseButton from '../design/components/ViseButton.vue';
import ViseBadge from '../design/components/ViseBadge.vue';
import ViseField from '../design/components/ViseField.vue';
import ViseIcon from '../design/components/ViseIcon.vue';
import { useProjectStore } from '../stores/project';
import { useSettingsStore } from '../stores/settings';
import { MODALITIES, MODALITY_ORDER } from '../lib/modalities';
import type { Modality } from '../types';

const project = useProjectStore();
const settings = useSettingsStore();

const newClassName = ref('');
const info = computed(() => MODALITIES[project.modality]);

function pickModality(m: Modality): void {
  project.setModality(m);
}

function addClass(): void {
  const name = newClassName.value.trim();
  if (!name) return;
  project.addClass(name);
  newClassName.value = '';
}
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
          <ViseButton size="sm" variant="ghost" @click="project.ensureNegativeClass()">
            Add {{ info.negative }}
          </ViseButton>
        </div>

        <ul v-if="project.classes.length" class="classlist">
          <li v-for="c in project.classes" :key="c.id">
            <ViseCard :accent="c.negative">
              <div class="classrow">
                <span class="cname">
                  {{ c.name }}
                  <ViseBadge v-if="c.negative" variant="auto">negative</ViseBadge>
                </span>
                <span class="ccount mono-num">{{ project.countsByClass[c.id] ?? 0 }} samples</span>
                <button class="del" type="button" :aria-label="`Remove ${c.name}`" @click="project.removeClass(c.id)">
                  <ViseIcon name="trash" :size="14" />
                </button>
              </div>
            </ViseCard>
          </li>
        </ul>
        <p v-else class="empty">No classes yet. Name your first class above.</p>

        <p class="note">
          Capture and import for {{ info.label }} arrive with the {{ info.label }} flow. The split is
          by capture session, never a random row, so the same subject does not leak across train and
          test.
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
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: var(--s-2);
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
  background: none;
  border: none;
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
.note {
  margin-top: var(--s-5);
}
.coach {
  margin: 0;
  padding-left: 18px;
  display: flex;
  flex-direction: column;
  gap: 6px;
}
</style>
