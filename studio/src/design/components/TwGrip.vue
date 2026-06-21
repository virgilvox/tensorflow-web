<script setup lang="ts">
/**
 * The jaw grip. Two steel jaw brackets clamp the work panel at its vertical
 * midpoint, each with a live tooth where it bites. This is the one bold move in
 * the design system, used once per screen on the primary work panel. The slotted
 * content scrolls inside the clamped panel.
 */
</script>

<template>
  <div class="grip-shell">
    <span class="jaw jaw-l" aria-hidden="true"><i></i></span>
    <div class="work">
      <slot />
    </div>
    <span class="jaw jaw-r" aria-hidden="true"><i></i></span>
  </div>
</template>

<style scoped>
.grip-shell {
  position: relative;
  height: 100%;
  min-height: 0;
  padding: 0 26px;
}
.work {
  background: var(--iron);
  border: 1px solid var(--seam);
  box-shadow: var(--cut);
  height: 100%;
  min-height: 0;
  overflow: auto;
  padding: var(--s-6);
}
/* The clamp jaws, gripping the panel at the vertical midpoint. */
.jaw {
  position: absolute;
  top: 50%;
  transform: translateY(-50%);
  width: 24px;
  height: 64%;
  z-index: 2;
}
.jaw::before,
.jaw::after {
  content: '';
  position: absolute;
  left: 0;
  width: 100%;
  height: 4px;
  background: var(--edge);
}
.jaw::before {
  top: 0;
}
.jaw::after {
  bottom: 0;
}
.jaw i {
  position: absolute;
  top: 50%;
  transform: translateY(-50%);
  height: 34%;
  width: 5px;
  background: var(--live);
}
.jaw-l {
  left: 0;
}
.jaw-l i {
  left: 0;
}
.jaw-r {
  right: 0;
}
.jaw-r i {
  right: 0;
}
</style>
