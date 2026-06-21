<script setup lang="ts">
/**
 * TwCard renders a surface card with an optional header row and a body.
 * The header shows a title on the left and right-aligned meta text. A named
 * "header" slot can replace the title/meta layout entirely. The accent variant
 * adds a 3px live left border to flag the active item.
 */

defineProps<{
  /** Header title text. When omitted, the title/meta header row is not rendered. */
  title?: string
  /** Right-aligned meta text. Only rendered when both meta and title are present. */
  meta?: string
  /** Flag the card as the active item with a live left border. */
  accent?: boolean
}>()
</script>

<template>
  <div class="card" :class="{ accent }">
    <div v-if="$slots.header" class="ch">
      <slot name="header" />
    </div>
    <div v-else-if="title" class="ch">
      <span class="ct">{{ title }}</span>
      <span v-if="meta" class="cm">{{ meta }}</span>
    </div>
    <div class="cb">
      <slot />
    </div>
  </div>
</template>

<style scoped>
.card {
  background: var(--graphite);
  border: 1px solid var(--seam);
  padding: 16px;
}
.card.accent {
  border-left: 3px solid var(--live);
}
.card .ch {
  display: flex;
  align-items: center;
  gap: 9px;
  border-bottom: 1px solid var(--seam);
  padding-bottom: 11px;
  margin-bottom: 11px;
}
.card .ch .ct {
  font-family: var(--f-display);
  font-weight: 600;
  font-size: 14px;
  letter-spacing: 0.03em;
}
.card .ch .cm {
  margin-left: auto;
  font-family: var(--f-label);
  font-size: 8.5px;
  letter-spacing: 0.08em;
  color: var(--ash);
}
.card .cb {
  font-size: 12px;
  color: var(--steam);
  line-height: 1.6;
}
</style>
