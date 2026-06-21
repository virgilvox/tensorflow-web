<script setup lang="ts">
/**
 * Segmented control. The altitude control in the top bar is its primary use:
 * Guided, Standard, Expert. Generic over a list of options bound through
 * v-model. The active segment carries the caution fill.
 */
defineProps<{
  /** The bound value. Use with v-model. */
  modelValue: string;
  /** Options to show, in order. value is emitted, label is shown. */
  options: ReadonlyArray<{ value: string; label: string }>;
  /** Accessible group label. */
  ariaLabel?: string;
}>();

const emit = defineEmits<{ 'update:modelValue': [value: string] }>();
</script>

<template>
  <div class="seg" role="radiogroup" :aria-label="ariaLabel">
    <button
      v-for="opt in options"
      :key="opt.value"
      type="button"
      role="radio"
      :aria-checked="opt.value === modelValue"
      :class="{ on: opt.value === modelValue }"
      @click="emit('update:modelValue', opt.value)"
    >
      {{ opt.label }}
    </button>
  </div>
</template>

<style scoped>
.seg {
  display: inline-flex;
  border: 1px solid var(--seam);
  background: var(--graphite);
}
.seg button {
  background: none;
  border: none;
  border-right: 1px solid var(--seam);
  padding: 8px 16px;
  font-family: var(--f-label);
  text-transform: uppercase;
  letter-spacing: 0.12em;
  font-size: 10px;
  color: var(--ash);
  cursor: pointer;
  transition: all var(--fast);
}
.seg button:last-child {
  border-right: none;
}
.seg button.on {
  background: var(--live);
  color: var(--live-ink);
  font-weight: 700;
}
.seg button:not(.on):hover {
  color: var(--chalk);
  background: var(--gunmetal);
}
</style>
