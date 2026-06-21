<script setup lang="ts">
/**
 * The in-app confirmation dialog. Mounted once in the shell; it reads the shared
 * confirm state and renders a focus-managed modal over the work area. Escape and
 * the backdrop cancel; the confirm button is styled destructive when the action
 * is. Replaces the native window.confirm so destructive guards are styled,
 * keyboard accessible, and consistent with the rest of the studio.
 */
import { ref, watch, nextTick, onBeforeUnmount } from 'vue';
import { useConfirm } from '../../composables/useConfirm';

const { state, settle } = useConfirm();
const confirmBtn = ref<HTMLButtonElement | null>(null);
const cancelBtn = ref<HTMLButtonElement | null>(null);
let lastFocused: HTMLElement | null = null;

function onKeydown(e: KeyboardEvent): void {
  if (!state.open) return;
  if (e.key === 'Escape') {
    e.preventDefault();
    settle(false);
    return;
  }
  if (e.key === 'Tab') {
    // Trap focus inside the dialog. With two buttons, cycling on Tab and Shift+Tab
    // both just move between them; the modulo wraps and an index of -1 (focus on
    // neither) lands on the first button.
    e.preventDefault();
    const order = [cancelBtn.value, confirmBtn.value].filter((b): b is HTMLButtonElement => !!b);
    if (order.length === 0) return;
    const idx = order.indexOf(document.activeElement as HTMLButtonElement);
    const dir = e.shiftKey ? -1 : 1;
    order[(idx + dir + order.length) % order.length]!.focus();
  }
}

/** Focuses the default button: cancel for a destructive action, else confirm. */
function focusDefault(): void {
  (state.danger ? cancelBtn.value : confirmBtn.value)?.focus();
}

watch(
  () => state.open,
  (open) => {
    if (open) {
      lastFocused = document.activeElement as HTMLElement | null;
      window.addEventListener('keydown', onKeydown, true);
    } else {
      window.removeEventListener('keydown', onKeydown, true);
      // Only restore focus if the trigger is still in the document; after a New
      // project / modality switch the triggering control may have re-rendered.
      if (lastFocused && document.contains(lastFocused)) lastFocused.focus();
      lastFocused = null;
    }
  },
);

// Re-apply the default focus on every request (seq), so an overlapping confirm()
// that reuses the open dialog still focuses the right button.
watch(
  () => state.seq,
  async () => {
    if (!state.open) return;
    await nextTick();
    focusDefault();
  },
);

onBeforeUnmount(() => window.removeEventListener('keydown', onKeydown, true));
</script>

<template>
  <Teleport to="body">
    <div
      v-if="state.open"
      class="backdrop"
      data-test="confirm-dialog"
      @click.self="settle(false)"
    >
      <div
        class="dialog"
        role="alertdialog"
        aria-modal="true"
        aria-labelledby="confirm-title"
        aria-describedby="confirm-message"
      >
        <h2 id="confirm-title" class="title">{{ state.title }}</h2>
        <p id="confirm-message" class="message">{{ state.message }}</p>
        <div class="actions">
          <button ref="cancelBtn" type="button" class="btn ghost" data-test="confirm-cancel" @click="settle(false)">
            {{ state.cancelLabel }}
          </button>
          <button
            ref="confirmBtn"
            type="button"
            class="btn"
            :class="{ danger: state.danger }"
            data-test="confirm-ok"
            @click="settle(true)"
          >
            {{ state.confirmLabel }}
          </button>
        </div>
      </div>
    </div>
  </Teleport>
</template>

<style scoped>
.backdrop {
  position: fixed;
  inset: 0;
  z-index: 100;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(5, 7, 9, 0.66);
  padding: var(--s-5);
}
.dialog {
  width: 100%;
  max-width: 420px;
  background: var(--graphite);
  border: 1px solid var(--seam);
  border-top: 2px solid var(--live);
  box-shadow: var(--cut-strong, 0 16px 40px rgba(0, 0, 0, 0.5));
  padding: var(--s-5);
}
.title {
  font-family: var(--f-display);
  font-weight: 700;
  font-size: var(--t-lg);
  letter-spacing: 0.04em;
  color: var(--chalk);
  margin: 0 0 var(--s-3);
}
.message {
  font-size: 12.5px;
  color: var(--steam);
  line-height: 1.6;
  margin: 0 0 var(--s-5);
}
.actions {
  display: flex;
  justify-content: flex-end;
  gap: var(--s-3);
}
/* Native buttons, so the dialog can take and trap real focus. Styled to match
   TwButton's small primary, ghost, and danger variants. */
.btn {
  display: inline-flex;
  align-items: center;
  gap: var(--s-2);
  padding: 7px 12px;
  background: var(--live);
  color: var(--live-ink);
  border: 1px solid var(--live);
  font-family: var(--f-label);
  text-transform: uppercase;
  letter-spacing: 0.1em;
  font-size: 9.5px;
  font-weight: 700;
  cursor: pointer;
  transition: transform var(--fast);
}
.btn:hover {
  transform: translateY(-1px);
}
.btn:focus-visible {
  outline: 2px solid var(--chalk);
  outline-offset: 2px;
}
.btn.ghost {
  background: transparent;
  color: var(--steam);
  border-color: var(--edge);
}
.btn.ghost:hover {
  color: var(--chalk);
  border-color: var(--live);
}
.btn.danger {
  background: var(--rust);
  border-color: var(--rust);
  color: #fff;
}
</style>
