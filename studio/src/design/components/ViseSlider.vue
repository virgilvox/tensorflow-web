<script setup lang="ts">
/**
 * ViseSlider
 *
 * A range slider with an uppercase label on the left and the live current
 * value on the right. Wraps a native range input and reports numeric changes
 * through v-model.
 */

const props = withDefaults(
  defineProps<{
    /** Current numeric value, bound through v-model. */
    modelValue: number
    /** Uppercase descriptor shown on the left of the header row. */
    label: string
    /** Lowest selectable value. */
    min: number
    /** Highest selectable value. */
    max: number
    /** Increment between selectable values. Defaults to whole numbers. */
    step?: number
    /** Optional unit appended after the value in the display, for example "px". */
    unit?: string
  }>(),
  {
    step: 1,
    unit: '',
  },
)

const emit = defineEmits<{
  (e: 'update:modelValue', value: number): void
}>()

/** Parse the input value to a number and emit it for v-model binding. */
function onInput(event: Event) {
  const target = event.target as HTMLInputElement
  emit('update:modelValue', Number(target.value))
}
</script>

<template>
  <div class="slr">
    <div class="slr-head">
      <span>{{ label }}</span>
      <span class="v">{{ modelValue }}{{ unit ? ' ' + unit : '' }}</span>
    </div>
    <input
      type="range"
      :min="min"
      :max="max"
      :step="step"
      :value="modelValue"
      @input="onInput"
    />
  </div>
</template>

<style scoped>
input[type=range] {
  width: 100%;
  -webkit-appearance: none;
  appearance: none;
  height: 4px;
  background: var(--steel);
  outline: none;
}
input[type=range]::-webkit-slider-thumb {
  -webkit-appearance: none;
  width: 14px;
  height: 14px;
  background: var(--live);
  border: 2px solid var(--void);
  cursor: pointer;
}
input[type=range]::-moz-range-thumb {
  width: 14px;
  height: 14px;
  background: var(--live);
  border: 2px solid var(--void);
  cursor: pointer;
}
.slr-head {
  display: flex;
  justify-content: space-between;
  font-family: var(--f-label);
  font-size: 9px;
  letter-spacing: .1em;
  text-transform: uppercase;
  color: var(--ash);
  margin-bottom: 6px;
}
.slr-head .v {
  color: var(--live);
  font-family: var(--f-display);
  font-weight: 600;
}
</style>
