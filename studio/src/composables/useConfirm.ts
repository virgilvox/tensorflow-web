/**
 * In-app confirmation dialog, replacing the browser's native window.confirm. A
 * single shared dialog is mounted once (TwConfirm in App.vue); callers anywhere
 * call confirm() and await a boolean. The state is module level so the prompt and
 * the dialog share one instance without prop plumbing. Used for destructive
 * actions (New project, switching modality) so the guard is styled, focus
 * managed, and keyboard dismissible inside the app.
 */
import { reactive, readonly } from 'vue';

interface ConfirmOptions {
  title: string;
  message: string;
  confirmLabel?: string;
  cancelLabel?: string;
  /** Styles the confirm button as destructive when true. */
  danger?: boolean;
}

interface ConfirmState {
  open: boolean;
  title: string;
  message: string;
  confirmLabel: string;
  cancelLabel: string;
  danger: boolean;
  /** Bumped on every confirm() so the dialog can re-apply focus even when it was
   *  already open (an overlapping request reuses the same dialog element). */
  seq: number;
}

const state = reactive<ConfirmState>({
  open: false,
  title: '',
  message: '',
  confirmLabel: 'Confirm',
  cancelLabel: 'Cancel',
  danger: false,
  seq: 0,
});

let resolver: ((value: boolean) => void) | null = null;

export function useConfirm() {
  /**
   * Opens the confirmation dialog and resolves true if confirmed, false if
   * cancelled or dismissed. Only one dialog is shown at a time; a second call
   * while one is open resolves the first as cancelled.
   */
  function confirm(options: ConfirmOptions): Promise<boolean> {
    if (resolver) {
      resolver(false);
      resolver = null;
    }
    state.title = options.title;
    state.message = options.message;
    state.confirmLabel = options.confirmLabel ?? 'Confirm';
    state.cancelLabel = options.cancelLabel ?? 'Cancel';
    state.danger = options.danger ?? false;
    state.seq += 1;
    state.open = true;
    return new Promise<boolean>((resolve) => {
      resolver = resolve;
    });
  }

  /** Closes the dialog and resolves the pending promise. Used by the dialog only. */
  function settle(value: boolean): void {
    state.open = false;
    resolver?.(value);
    resolver = null;
  }

  return { state: readonly(state), confirm, settle };
}
