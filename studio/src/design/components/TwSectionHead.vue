<script setup lang="ts">
/**
 * TwSectionHead
 * An index-tagged section header. Renders an optional accent number, a display
 * title, a rule that fills the remaining width of the row, and an optional mono
 * note aligned to the right. Used to separate major sections of a panel.
 */

defineProps<{
  /** Section index tag such as "01". Optional; the number slot is omitted when absent. */
  index?: string
  /** The section title text, shown uppercase in the display face. */
  title: string
  /** Optional short mono note shown at the far right of the row. Omitted when absent. */
  note?: string
}>()
</script>

<template>
  <div class="sechead">
    <span v-if="index" class="n">{{ index }}</span>
    <span class="t">{{ title }}</span>
    <span class="ln"></span>
    <span v-if="note" class="note">{{ note }}</span>
  </div>
</template>

<style scoped>
.sechead {
  display: flex;
  align-items: baseline;
  gap: 14px;
  margin-bottom: 30px;
}
.sechead .n {
  font-family: var(--f-display);
  font-weight: 700;
  font-size: 20px;
  color: var(--live-dim);
}
.sechead .t {
  font-family: var(--f-display);
  font-weight: 700;
  font-size: 18px;
  letter-spacing: .08em;
  text-transform: uppercase;
}
.sechead .ln {
  flex: 1;
  height: 1px;
  background: var(--seam);
  align-self: center;
}
.sechead .note {
  font-family: var(--f-label);
  font-size: 9.5px;
  color: var(--ash);
  letter-spacing: .06em;
}
</style>
