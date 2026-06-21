<script setup lang="ts">
/**
 * The nav rail. One entry per workflow stage in plain words, the active stage
 * marked with the live left border. The two configuration stages, Features and
 * Model, are hidden in Guided and appear from Standard up, which is how altitude
 * drives progressive disclosure of the rail itself.
 */
import { computed } from 'vue';
import TwIcon from './TwIcon.vue';
import { STAGES } from '../../router';
import { useSettingsStore } from '../../stores/settings';

const settings = useSettingsStore();

// Configuration stages drop out of the rail in Guided. Numbering still reflects
// the real process order of the visible stages.
const visibleStages = computed(() =>
  STAGES.filter((s) => settings.showsConfigStages || !s.configuration),
);
</script>

<template>
  <div class="rail">
    <nav aria-label="Workflow stages">
      <RouterLink
        v-for="(stage, i) in visibleStages"
        :key="stage.id"
        :to="`/${stage.id}`"
        class="ri"
        active-class="on"
      >
        <span class="n mono-num">{{ String(i + 1).padStart(2, '0') }}</span>
        <span class="ic"><TwIcon :name="stage.id" :size="18" /></span>
        <span class="txt">
          <span class="lbl">{{ stage.label }}</span>
          <span class="cap">{{ stage.caption }}</span>
        </span>
      </RouterLink>
    </nav>

    <span class="sep" aria-hidden="true"></span>

    <nav aria-label="Tools">
      <RouterLink to="/playground" class="ri" active-class="on">
        <span class="n mono-num">&middot;</span>
        <span class="ic"><TwIcon name="play" :size="18" /></span>
        <span class="txt">
          <span class="lbl">Playground</span>
          <span class="cap">run any model</span>
        </span>
      </RouterLink>
    </nav>
  </div>
</template>

<style scoped>
.rail {
  display: flex;
  flex-direction: column;
  background: var(--soot);
  border-right: 1px solid var(--seam);
  padding: var(--s-4) 0;
  width: 168px;
  flex: none;
  overflow-y: auto;
}
.ri {
  display: grid;
  grid-template-columns: 22px 22px 1fr;
  align-items: center;
  gap: var(--s-2);
  padding: 11px 14px;
  border-left: 2px solid transparent;
  color: var(--steam);
  transition: color var(--fast), background var(--fast);
}
.ri:hover {
  color: var(--chalk);
  background: var(--iron);
}
.ri.on {
  border-left-color: var(--live);
  color: var(--live);
  background: var(--iron);
}
.sep {
  height: 1px;
  background: var(--seam);
  margin: var(--s-3) 14px;
}
.n {
  font-family: var(--f-display);
  font-weight: 700;
  font-size: 12px;
  color: var(--ghost);
}
.ri.on .n {
  color: var(--live-dim);
}
.ic {
  display: inline-flex;
}
.txt {
  display: flex;
  flex-direction: column;
  line-height: 1.2;
  min-width: 0;
}
.lbl {
  font-family: var(--f-display);
  font-weight: 600;
  font-size: 13px;
  letter-spacing: 0.04em;
}
.cap {
  font-family: var(--f-label);
  font-size: 8px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--ash);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
</style>
