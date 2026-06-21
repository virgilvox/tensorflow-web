<script setup lang="ts">
/**
 * The bench shell. Four zones from the TF Web Studio layout anatomy: the fixed top bar,
 * the nav rail, the jaw gripped work panel that holds the active stage, and the
 * persistent bench rail. The router fills the work panel; the altitude control
 * in the top bar drives disclosure across whatever stage is mounted.
 */
import TwTopBar from './design/components/TwTopBar.vue';
import TwNavRail from './design/components/TwNavRail.vue';
import TwBenchRail from './design/components/TwBenchRail.vue';
import TwGrip from './design/components/TwGrip.vue';
import TwConfirm from './design/components/TwConfirm.vue';
</script>

<template>
  <TwTopBar />
  <div class="body">
    <TwNavRail />
    <main class="work-area">
      <TwGrip>
        <RouterView v-slot="{ Component }">
          <component :is="Component" />
        </RouterView>
      </TwGrip>
    </main>
  </div>
  <TwBenchRail />
  <TwConfirm />
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
