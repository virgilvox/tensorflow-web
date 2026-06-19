<script setup lang="ts">
/**
 * ViseGauge renders a semicircle instrument gauge. A steel background arc sits
 * behind a foreground arc that sweeps from the left end to a point set by the
 * given fraction. The formatted value is shown large below the arc, with an
 * optional small unit after it.
 */

import { computed } from 'vue'

const props = defineProps<{
  /** Caption shown above the arc, rendered in uppercase label type. */
  label: string
  /** Sweep amount from 0 to 1, clamped. Drives how far the foreground arc fills. */
  fraction: number
  /** Preformatted number shown large below the arc, for example "94". */
  value: string
  /** Optional small unit drawn directly after the value, for example "%". */
  unit?: string
  /** When true, use the cool patina stroke for the foreground arc instead of live. */
  cool?: boolean
}>()

/** Total length of the semicircle arc path, used for the dash math. */
const ARC_LENGTH = 106.8

/** Fraction held within the 0..1 range so the arc never under or overfills. */
const clampedFraction = computed(() => Math.min(1, Math.max(0, props.fraction)))

/** Dash offset that hides the unswept tail of the foreground arc. */
const dashOffset = computed(() => ARC_LENGTH * (1 - clampedFraction.value))
</script>

<template>
  <div class="gauge">
    <div class="gl">{{ label }}</div>
    <svg viewBox="0 0 86 52">
      <path class="ga-bg" d="M9 48 A 34 34 0 0 1 77 48" />
      <path
        class="ga-fg"
        :class="{ cool }"
        d="M9 48 A 34 34 0 0 1 77 48"
        :style="{ strokeDasharray: ARC_LENGTH, strokeDashoffset: dashOffset }"
      />
    </svg>
    <div class="gv mono-num">{{ value }}<small v-if="unit">{{ unit }}</small></div>
  </div>
</template>

<style scoped>
.gauge {
  background: var(--graphite);
  border: 1px solid var(--seam);
  padding: 14px 20px;
  text-align: center;
}
.gauge .gl {
  font-family: var(--f-label);
  font-size: 8.5px;
  letter-spacing: .1em;
  text-transform: uppercase;
  color: var(--ash);
  margin-bottom: 7px;
}
.gauge svg {
  width: 86px;
  height: 52px;
}
.ga-bg {
  fill: none;
  stroke: var(--steel);
  stroke-width: 7;
}
.ga-fg {
  fill: none;
  stroke: var(--live);
  stroke-width: 7;
  transition: stroke-dashoffset .6s var(--ease);
}
.ga-fg.cool {
  stroke: var(--patina);
}
.gauge .gv {
  font-family: var(--f-display);
  font-weight: 700;
  font-size: 20px;
  margin-top: 3px;
  font-variant-numeric: tabular-nums;
}
.gauge .gv small {
  font-size: 11px;
  color: var(--ash);
  font-weight: 400;
}
</style>
