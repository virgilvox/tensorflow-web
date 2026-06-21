<script setup lang="ts">
/**
 * The TF Web Studio button. Caution yellow fill by default, a ghost outline variant, a
 * small size, and a danger variant for cancel and destroy actions. Square
 * corners, uppercase label type, a one pixel translateY lift on hover.
 */
withDefaults(
  defineProps<{
    variant?: 'primary' | 'ghost' | 'danger';
    size?: 'md' | 'sm';
    type?: 'button' | 'submit';
    disabled?: boolean;
  }>(),
  { variant: 'primary', size: 'md', type: 'button', disabled: false },
);
</script>

<template>
  <button
    class="btn"
    :class="[variant, { sm: size === 'sm' }]"
    :type="type"
    :disabled="disabled"
  >
    <slot />
  </button>
</template>

<style scoped>
.btn {
  display: inline-flex;
  align-items: center;
  gap: var(--s-2);
  padding: 10px 18px;
  background: var(--live);
  color: var(--live-ink);
  border: 1px solid var(--live);
  font-family: var(--f-label);
  text-transform: uppercase;
  letter-spacing: 0.1em;
  font-size: 10.5px;
  font-weight: 700;
  cursor: pointer;
  transition: transform var(--fast);
}
.btn:hover:not(:disabled) {
  transform: translateY(-1px);
}
.btn:active:not(:disabled) {
  transform: translateY(0);
}
.btn:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}
.btn.ghost {
  background: transparent;
  color: var(--steam);
  border-color: var(--edge);
}
.btn.ghost:hover:not(:disabled) {
  color: var(--chalk);
  border-color: var(--live);
}
.btn.danger {
  background: var(--rust);
  border-color: var(--rust);
  color: #fff;
}
.btn.sm {
  padding: 7px 12px;
  font-size: 9.5px;
}
</style>
