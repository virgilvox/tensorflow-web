<script setup lang="ts">
/**
 * TwSelect renders a labelled dropdown styled to match the field input.
 * The label sits above the select, wired to it by a unique id for
 * accessibility, and the select shows a live focus ring when active.
 * Two way binding flows through modelValue and update:modelValue.
 */

interface SelectOption {
  value: string
  label: string
}

const props = defineProps<{
  /** Current selected option value, bound with v-model. */
  modelValue: string
  /** Field caption shown above the select, kept short and uppercased by style. */
  label: string
  /** Options to render, each an object with a value and a display label. */
  options: ReadonlyArray<SelectOption>
}>()

const emit = defineEmits<{
  (event: 'update:modelValue', value: string): void
}>()

/** Unique id so the label points at exactly this select instance. */
const selectId = `tw-select-${Math.random().toString(36).slice(2, 10)}`

/** Forward native change events to the parent as a v-model update. */
function onChange(event: Event): void {
  const target = event.target as HTMLSelectElement
  emit('update:modelValue', target.value)
}
</script>

<template>
  <div class="field">
    <label :for="selectId">{{ props.label }}</label>
    <select :id="selectId" :value="props.modelValue" @change="onChange">
      <option v-for="option in props.options" :key="option.value" :value="option.value">
        {{ option.label }}
      </option>
    </select>
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

.field select {
  width: 100%;
  background: var(--gunmetal);
  border: 1px solid var(--seam);
  color: var(--chalk);
  font-family: var(--f-mono);
  font-size: 12.5px;
  padding: 9px 11px;
  transition: border-color var(--fast), box-shadow var(--fast);
}

.field select:focus {
  outline: none;
  border-color: var(--live);
  box-shadow: 0 0 0 3px var(--live-glow);
}
</style>
