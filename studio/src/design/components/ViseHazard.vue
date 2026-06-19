<script setup lang="ts">
/**
 * ViseHazard renders a thin diagonal caution hazard stripe bar.
 * Use it to mark a running job rail or a live boundary. It is decorative
 * and carries no semantic meaning, so it is hidden from assistive tech.
 * Keep it as a narrow strip. It is never meant to be a large fill area.
 */

const props = withDefaults(
  defineProps<{
    /**
     * Height of the stripe bar in pixels. Defaults to 14 to read as a
     * thin rail rather than a block of fill.
     */
    height?: number
  }>(),
  {
    height: 14,
  },
)
</script>

<template>
  <div class="hazbar" :style="{ height: `${props.height}px` }" aria-hidden="true" />
</template>

<style scoped>
.hazbar {
  height: 14px;
  background: var(--hazard);
  opacity: 0.7;
}
</style>
