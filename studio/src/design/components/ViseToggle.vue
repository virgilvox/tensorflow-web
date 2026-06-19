<script setup lang="ts">
/**
 * ViseToggle is a labelled on/off switch.
 * The track fills with the caution-dim tone and the knob slides right and
 * turns live when the value is on. Renders as a button with switch semantics
 * so assistive technology reports the checked state.
 */

const props = defineProps<{
  /** Current on/off state. Drives the track fill and knob position. */
  modelValue: boolean
  /** Text shown next to the switch describing what it controls. */
  label: string
}>()

const emit = defineEmits<{
  /** Fired on click with the negated state. */
  (e: 'update:modelValue', value: boolean): void
}>()

/** Flip the state and notify the parent. */
function toggle(): void {
  emit('update:modelValue', !props.modelValue)
}
</script>

<template>
  <button
    type="button"
    role="switch"
    :aria-checked="modelValue"
    class="toggle"
    :class="{ on: modelValue }"
    @click="toggle"
  >
    <span class="sw2"></span>
    <span class="tl">{{ label }}</span>
  </button>
</template>

<style scoped>
.toggle {
  display: inline-flex;
  align-items: center;
  gap: 11px;
  padding: 10px 14px;
  background: var(--graphite);
  border: 1px solid var(--seam);
  cursor: pointer;
}
.toggle .sw2 {
  width: 34px;
  height: 18px;
  background: var(--steel);
  border: 1px solid var(--edge);
  position: relative;
  flex: none;
  transition: background var(--fast);
}
.toggle .sw2::after {
  content: '';
  position: absolute;
  top: 1px;
  left: 1px;
  width: 14px;
  height: 14px;
  background: var(--ash);
  transition: all var(--fast);
}
.toggle.on .sw2 {
  background: var(--live-dim);
}
.toggle.on .sw2::after {
  left: 17px;
  background: var(--live);
}
.toggle .tl {
  font-size: 11.5px;
  color: var(--chalk);
}
</style>
