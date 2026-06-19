import { describe, it, expect } from 'vitest';
import { formatBytes, formatPercent, formatFixed } from '../src/lib/format';

describe('formatBytes', () => {
  it('shows plain bytes under a kilobyte', () => {
    expect(formatBytes(0)).toBe('0 B');
    expect(formatBytes(512)).toBe('512 B');
  });

  it('scales to binary units', () => {
    expect(formatBytes(1024)).toBe('1.0 KB');
    expect(formatBytes(1536)).toBe('1.5 KB');
    expect(formatBytes(1024 * 1024)).toBe('1.0 MB');
    expect(formatBytes(4 * 1024 * 1024)).toBe('4.0 MB');
  });

  it('drops the decimal at or above 100 in a unit', () => {
    expect(formatBytes(150 * 1024)).toBe('150 KB');
  });

  it('returns a placeholder for invalid input', () => {
    expect(formatBytes(-1)).toBe('—');
    expect(formatBytes(NaN)).toBe('—');
  });
});

describe('formatPercent', () => {
  it('rounds a fraction to a whole percent', () => {
    expect(formatPercent(0)).toBe('0%');
    expect(formatPercent(0.944)).toBe('94%');
    expect(formatPercent(1)).toBe('100%');
  });

  it('returns a placeholder for non finite input', () => {
    expect(formatPercent(Infinity)).toBe('—');
  });
});

describe('formatFixed', () => {
  it('fixes to the requested decimals', () => {
    expect(formatFixed(0.123456, 3)).toBe('0.123');
  });

  it('returns a placeholder for missing or NaN values', () => {
    expect(formatFixed(undefined)).toBe('—');
    expect(formatFixed(NaN)).toBe('—');
  });
});
