<script setup lang="ts">
/**
 * Labelled single line input. The label sits above the input in uppercase
 * label-type. The input takes a live focus ring on focus. Use with v-model.
 * For type 'number' the emitted value is a Number, for 'text' it is the string.
 */
import { useId } from 'vue';

const props = withDefaults(
  defineProps<{
    /** The bound value. Use with v-model. */
    modelValue: string | number;
    /** Uppercase label-type caption rendered above the input. */
    label: string;
    /** Input mode. 'number' emits a Number, 'text' emits the string. */
    type?: 'text' | 'number';
    /** Placeholder shown when the input is empty. */
    placeholder?: string;
    /** Minimum value for type 'number'. */
    min?: number;
    /** Maximum value for type 'number'. */
    max?: number;
    /** Step increment for type 'number'. */
    step?: number;
  }>(),
  {
    type: 'text',
    placeholder: undefined,
    min: undefined,
    max: undefined,
    step: undefined,
  },
);

const emit = defineEmits<{ 'update:modelValue': [value: string | number] }>();

// Unique id ties the label to the input through :for and :id.
const id = useId();

function onInput(event: Event): void {
  const value = (event.target as HTMLInputElement).value;
  emit('update:modelValue', props.type === 'number' ? Number(value) : value);
}
</script>

<template>
  <div class="field">
    <label :for="id">{{ label }}</label>
    <input
      :id="id"
      :type="type"
      :value="modelValue"
      :placeholder="placeholder"
      :min="min"
      :max="max"
      :step="step"
      @input="onInput"
    />
  </div>
</template>

<style scoped>
.field {
  margin-bottom: 14px;
}
.field label {
  display: block;
  font-family: var(--f-label);
  font-size: 9px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--ash);
  margin-bottom: 6px;
}
.field input {
  width: 100%;
  background: var(--gunmetal);
  border: 1px solid var(--seam);
  color: var(--chalk);
  font-family: var(--f-mono);
  font-size: 12.5px;
  padding: 9px 11px;
  transition: border-color var(--fast), box-shadow var(--fast);
}
.field input:focus {
  outline: none;
  border-color: var(--live);
  box-shadow: 0 0 0 3px var(--live-glow);
}
.field input::placeholder {
  color: var(--ghost);
}
</style>
