<script setup lang="ts">
/**
 * ViseLossChart draws a hand built SVG line chart of training metrics
 * across epochs without any charting library. It plots the loss curve as
 * a polyline normalized so the largest loss sits at the top of the box and
 * zero sits at the bottom. An optional accuracy series is plotted on a fixed
 * 0..1 scale mapped across the full height. When no data exists yet it shows
 * an empty framed box with a quiet placeholder label.
 */
import { computed } from 'vue'

const props = defineProps<{
  /** Loss values, one per recorded epoch. Drives the primary curve and the
   *  vertical normalization (max loss maps to the top, zero to the bottom). */
  loss: number[]
  /** Optional accuracy values in the 0..1 range. May be the same length as
   *  loss or shorter. When present a second curve and legend are drawn. */
  acc?: number[]
}>()

// Fixed drawing surface. The SVG scales non uniformly via preserveAspectRatio
// "none", so all stroke widths use vector-effect to stay crisp.
const VIEW_W = 300
const VIEW_H = 120

/** True when there is at least one loss sample to draw. */
const hasData = computed(() => props.loss.length > 0)

/** True when an accuracy series with samples is supplied. */
const hasAcc = computed(() => Array.isArray(props.acc) && props.acc.length > 0)

/**
 * Map an array of values to an SVG points string. Each value is placed at an
 * evenly spaced x position. The y mapping is supplied by the caller so loss
 * and accuracy can use different scales. Single point arrays are pinned to the
 * left edge to avoid a divide by zero on the x step.
 */
function toPoints(values: number[], yOf: (v: number) => number): string {
  const n = values.length
  if (n === 0) return ''
  const step = n > 1 ? VIEW_W / (n - 1) : 0
  return values.map((v, i) => `${(i * step).toFixed(2)},${yOf(v).toFixed(2)}`).join(' ')
}

/** Largest loss value, used to normalize the loss curve. Guards an all zero
 *  or empty series by falling back to 1 so the curve stays on the baseline. */
const maxLoss = computed(() => {
  if (!hasData.value) return 1
  const m = Math.max(...props.loss)
  return m > 0 ? m : 1
})

/** Loss polyline points. Max loss maps to y = 0 (top), zero maps to the
 *  bottom edge. */
const lossPoints = computed(() =>
  toPoints(props.loss, (v) => VIEW_H - (v / maxLoss.value) * VIEW_H),
)

/** Accuracy polyline points on a fixed 0..1 scale. 1 maps to the top, 0 to
 *  the bottom. Values are clamped so out of range inputs stay inside the box. */
const accPoints = computed(() => {
  if (!hasAcc.value) return ''
  return toPoints(props.acc as number[], (v) => {
    const c = Math.min(1, Math.max(0, v))
    return VIEW_H - c * VIEW_H
  })
})

/** Y positions for the faint baseline grid lines (one third and two thirds). */
const gridLines = [VIEW_H / 3, (VIEW_H / 3) * 2]
</script>

<template>
  <div class="chart">
    <div v-if="hasAcc" class="legend">
      <span class="chip chip-loss">loss</span>
      <span class="chip chip-acc">acc</span>
    </div>

    <div class="frame">
      <svg
        v-if="hasData"
        class="plot"
        :viewBox="`0 0 ${VIEW_W} ${VIEW_H}`"
        preserveAspectRatio="none"
        role="img"
        aria-label="Training loss across epochs"
      >
        <line
          v-for="(y, i) in gridLines"
          :key="i"
          class="grid"
          x1="0"
          :y1="y"
          :x2="VIEW_W"
          :y2="y"
        />
        <polyline class="line-loss" :points="lossPoints" />
        <polyline v-if="hasAcc" class="line-acc" :points="accPoints" />
      </svg>

      <div v-else class="empty">no data yet</div>
    </div>
  </div>
</template>

<style scoped>
.chart {
  display: flex;
  flex-direction: column;
  gap: var(--s-2);
}

/* Box: background var(--graphite), border 1px solid var(--seam),
   padding var(--s-3). */
.frame {
  background: var(--graphite);
  border: 1px solid var(--seam);
  padding: var(--s-3);
}

.plot {
  display: block;
  width: 100%;
  height: 120px;
}

.grid {
  stroke: var(--seam);
  stroke-width: 1;
  vector-effect: non-scaling-stroke;
}

/* Polylines: fill none, stroke-width 1.6, vector-effect non-scaling-stroke so
   the line stays crisp under the non-uniform viewBox scale. */
.line-loss,
.line-acc {
  fill: none;
  stroke-width: 1.6;
  vector-effect: non-scaling-stroke;
}

.line-loss {
  stroke: var(--live);
}

.line-acc {
  stroke: var(--patina);
}

.empty {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 120px;
  color: var(--ash);
  font-family: var(--f-label);
  font-size: var(--t-xs);
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

/* Legend chips in var(--f-label) uppercase 9px. */
.legend {
  display: flex;
  gap: var(--s-2);
}

.chip {
  display: inline-flex;
  align-items: center;
  gap: var(--s-1);
  font-family: var(--f-label);
  font-size: 9px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--ash);
}

.chip::before {
  content: '';
  display: inline-block;
  width: 10px;
  height: 2px;
}

.chip-loss::before {
  background: var(--live);
}

.chip-acc::before {
  background: var(--patina);
}
</style>
