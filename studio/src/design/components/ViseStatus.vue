<script setup lang="ts">
/**
 * ViseStatus renders a small status indicator made of a colored square and an
 * uppercase label. The square color encodes the state. The running state pulses
 * to signal active work, every other state is static. The label text is supplied
 * through the default slot so callers control the wording.
 */

defineProps<{
  /**
   * The current condition being reported. Drives both the square color and the
   * label color, and decides whether the square pulses. 'run' is active work and
   * pulses, 'pass' is a good result, 'hold' is idle or paused, 'warn' needs
   * attention, and 'fault' is an error.
   */
  state: 'run' | 'pass' | 'hold' | 'warn' | 'fault'
}>()
</script>

<template>
  <span class="status" :class="state">
    <span class="sq"></span>
    <slot />
  </span>
</template>

<style scoped>
.status {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  font-family: var(--f-label);
  font-size: 9.5px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
}

.status .sq {
  width: 9px;
  height: 9px;
  flex: none;
}

.status.run {
  color: var(--live);
}

.status.run .sq {
  background: var(--live);
  border-radius: 50%;
  animation: pulse 1.3s infinite;
}

.status.pass {
  color: var(--pass);
}

.status.pass .sq {
  background: var(--pass);
}

.status.hold {
  color: var(--ash);
}

.status.hold .sq {
  background: var(--ash);
}

.status.warn {
  color: var(--amber);
}

.status.warn .sq {
  background: var(--amber);
}

.status.fault {
  color: var(--rust);
}

.status.fault .sq {
  background: var(--rust);
}

@keyframes pulse {
  0%,
  100% {
    opacity: 1;
    box-shadow: 0 0 0 0 var(--live-glow);
  }
  50% {
    opacity: 0.5;
    box-shadow: 0 0 0 5px rgba(204, 242, 62, 0);
  }
}
</style>
