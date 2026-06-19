<script setup lang="ts">
/**
 * ViseBadge renders a compact metadata badge for marking up model and dataset
 * details. It carries no logic of its own, the visual treatment is selected by
 * the variant prop and the label text comes from the default slot.
 */

type Variant =
  | 'default'
  | 'auto'
  | 'why'
  | 'ver'
  | 'ver-cur'
  | 'type-text'
  | 'type-img'
  | 'type-traj'

const props = withDefaults(
  defineProps<{
    /**
     * Visual variant. 'auto' is a filled caution chip for machine set values,
     * 'why' is a dim outline with a help cursor for explanatory hints, 'ver'
     * and 'ver-cur' tag versions with the current one highlighted, and the
     * 'type-*' variants color code the modality of a dataset.
     */
    variant?: Variant
    /**
     * Native tooltip text. Used mainly by the 'why' variant to surface the
     * explanation on hover.
     */
    title?: string
  }>(),
  {
    variant: 'default',
    title: undefined,
  },
)

const variantMap: Record<Variant, string> = {
  default: '',
  auto: 'auto',
  why: 'why',
  ver: 'ver',
  'ver-cur': 'ver cur',
  'type-text': 'type-text',
  'type-img': 'type-img',
  'type-traj': 'type-traj',
}

const variantClass = variantMap[props.variant]
</script>

<template>
  <span class="badge" :class="variantClass" :title="title"><slot /></span>
</template>

<style scoped>
.badge {
  font-family: var(--f-label);
  font-size: 8.5px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  padding: 3px 8px;
  border: 1px solid var(--edge);
  color: var(--steam);
}
.badge.auto {
  background: var(--live);
  color: var(--live-ink);
  border-color: var(--live);
  font-weight: 700;
}
.badge.why {
  border-color: var(--live-dim);
  color: var(--live-dim);
  cursor: help;
}
.badge.ver {
  background: var(--gunmetal);
  border-color: var(--seam);
}
.badge.ver.cur {
  border-color: var(--live-dim);
  color: var(--live);
}
.badge.type-text {
  border-color: var(--patina);
  color: var(--patina);
}
.badge.type-img {
  border-color: var(--live-dim);
  color: var(--live);
}
.badge.type-traj {
  border-color: var(--amber);
  color: var(--amber);
}
</style>
