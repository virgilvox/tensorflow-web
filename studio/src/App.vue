<script setup lang="ts">
/**
 * The bench shell. Four zones from the VISE layout anatomy: the fixed top bar,
 * the nav rail, the jaw gripped work panel that holds the active stage, and the
 * persistent bench rail. The router fills the work panel; the altitude control
 * in the top bar drives disclosure across whatever stage is mounted.
 */
import ViseTopBar from './design/components/ViseTopBar.vue';
import ViseNavRail from './design/components/ViseNavRail.vue';
import ViseBenchRail from './design/components/ViseBenchRail.vue';
import ViseGrip from './design/components/ViseGrip.vue';
</script>

<template>
  <ViseTopBar />
  <div class="body">
    <ViseNavRail />
    <main class="work-area">
      <ViseGrip>
        <RouterView v-slot="{ Component }">
          <component :is="Component" />
        </RouterView>
      </ViseGrip>
    </main>
  </div>
  <ViseBenchRail />
</template>

<style scoped>
.body {
  flex: 1;
  display: flex;
  min-height: 0;
}
.work-area {
  flex: 1;
  min-width: 0;
  min-height: 0;
  padding: var(--s-4) var(--s-2);
}
</style>
