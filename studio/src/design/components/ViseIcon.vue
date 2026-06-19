<script setup lang="ts">
/**
 * Inline SVG icon set. The design system forbids emoji used as icons, so every
 * glyph the studio needs is a real stroked path drawn in currentColor. Add new
 * icons to the PATHS map; the component never renders raster art or an emoji.
 */
import { computed } from 'vue';

const props = withDefaults(
  defineProps<{
    /** Name of the icon to draw. Unknown names render nothing. */
    name: string;
    /** Pixel size of the square viewport. */
    size?: number;
  }>(),
  { size: 16 },
);

// 24x24 viewBox path data. Stroked, round joins, no fill, so they read at any
// size and inherit the surrounding text color.
const PATHS: Record<string, string> = {
  data: 'M4 7c0-1.7 3.6-3 8-3s8 1.3 8 3-3.6 3-8 3-8-1.3-8-3zM4 7v10c0 1.7 3.6 3 8 3s8-1.3 8-3V7M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3',
  features: 'M3 12h4l3-7 4 14 3-7h4',
  model: 'M6 4h12M6 20h12M9 4v4l-3 4 3 4v4M15 4v4l3 4-3 4v4',
  train: 'M5 19V9M10 19V5M15 19v-7M20 19v-4',
  test: 'M11 4a7 7 0 1 0 0 14 7 7 0 0 0 0-14zM16 16l4 4',
  export: 'M12 3v12M8 11l4 4 4-4M5 21h14',
  camera: 'M3 7h3l2-2h8l2 2h3v12H3zM12 16a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7z',
  mic: 'M12 3a3 3 0 0 1 3 3v5a3 3 0 0 1-6 0V6a3 3 0 0 1 3-3zM5 11a7 7 0 0 0 14 0M12 18v3',
  motion: 'M3 12h3l2-6 4 12 2-6h7',
  text: 'M4 6h16M4 6V5M8 6v13M8 19H6M8 19h2',
  play: 'M7 4l13 8-13 8z',
  stop: 'M6 6h12v12H6z',
  download: 'M12 3v12M7 10l5 5 5-5M4 21h16',
  check: 'M5 13l4 4L19 7',
  x: 'M6 6l12 12M18 6L6 18',
  plus: 'M12 5v14M5 12h14',
  trash: 'M4 7h16M9 7V5h6v2M6 7l1 13h10l1-13',
  chip: 'M8 8h8v8H8zM9 3v3M15 3v3M9 18v3M15 18v3M3 9h3M3 15h3M18 9h3M18 15h3',
  lock: 'M6 11h12v9H6zM9 11V8a3 3 0 0 1 6 0v3',
  bolt: 'M13 3L5 13h6l-1 8 8-10h-6z',
  image: 'M3 5h18v14H3zM3 16l5-5 4 4 3-3 6 6',
  gauge: 'M4 18a8 8 0 0 1 16 0M12 18l4-5',
  warn: 'M12 4l9 16H3zM12 10v4M12 17v.5',
};

const path = computed(() => PATHS[props.name] ?? '');
</script>

<template>
  <svg
    v-if="path"
    :width="size"
    :height="size"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    stroke-width="1.6"
    stroke-linecap="round"
    stroke-linejoin="round"
    aria-hidden="true"
    focusable="false"
  >
    <path :d="path" />
  </svg>
</template>
