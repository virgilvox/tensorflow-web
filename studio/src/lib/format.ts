/**
 * Number and size formatting helpers. Pure functions, unit testable, no Vue.
 */

/**
 * Formats a byte count as a short human readable size using binary units.
 *
 * @returns a string like "4.0 KB" or "1.2 MB". Bytes under 1024 are shown as a
 *   plain integer with a "B" suffix.
 */
export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) return '—';
  if (bytes < 1024) return `${Math.round(bytes)} B`;
  const units = ['KB', 'MB', 'GB'];
  let value = bytes / 1024;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(value >= 100 ? 0 : 1)} ${units[unit]}`;
}

/**
 * Formats a 0..1 fraction as a percent string with no decimals.
 *
 * @returns a string like "94%", or an em space placeholder for non finite input.
 */
export function formatPercent(fraction: number): string {
  if (!Number.isFinite(fraction)) return '—';
  return `${Math.round(fraction * 100)}%`;
}

/**
 * Formats a number to a fixed number of decimals, with a placeholder for missing
 * or NaN values.
 *
 * @returns the fixed string, or an em space placeholder.
 */
export function formatFixed(value: number | undefined, digits = 4): string {
  if (value === undefined || Number.isNaN(value)) return '—';
  return value.toFixed(digits);
}
