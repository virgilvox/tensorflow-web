/**
 * Device budget. Flash is the exact .tflite byte count; the runtime arena is the
 * model's estimate, refined by the interpreter's report when verify provides one.
 * Both are checked against the selected target so the Export stage can say plainly
 * whether the model fits. See PLAN section 8.
 */
import type { Model } from 'tensorflow-web';
import { estimateArenaBytes } from '../models/builder';
import type { BudgetResult } from '../stores/training';
import type { TargetDevice } from '../types';

export function useDeviceBudget() {
  /**
   * Computes the budget result for a model against a target.
   *
   * @param bytes the emitted .tflite, whose length is the flash figure.
   * @param model the float reference, used to estimate the arena.
   * @param target the device whose flash and RAM budgets are checked.
   * @param reportedArena an arena size verify obtained from the interpreter, if any.
   * @returns the flash and arena figures and whether both fit the target.
   */
  function evaluate(
    bytes: Uint8Array,
    model: Model,
    target: TargetDevice,
    reportedArena?: number,
  ): BudgetResult {
    const flashBytes = bytes.length;
    const arenaBytes = reportedArena && reportedArena > 0 ? reportedArena : estimateArenaBytes(model);
    const fits = flashBytes <= target.flashBytes && arenaBytes <= target.ramBytes;
    return { flashBytes, arenaBytes, fits };
  }

  return { evaluate };
}
