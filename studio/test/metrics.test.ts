import { describe, it, expect } from 'vitest';
import { perClassMetrics } from '../src/lib/metrics';

describe('perClassMetrics', () => {
  it('computes precision, recall, f1, and support from a confusion matrix', () => {
    // Rows are true class, columns predicted. Class 0: 8 right, 2 called class 1.
    // Class 1: 1 called class 0, 9 right. Predicted-0 column = 8 + 1 = 9.
    const m = perClassMetrics([
      [8, 2],
      [1, 9],
    ]);
    expect(m[0]!.support).toBe(10);
    expect(m[0]!.precision).toBeCloseTo(8 / 9, 5); // 8 of 9 predicted-0 are right
    expect(m[0]!.recall).toBeCloseTo(8 / 10, 5); // 8 of 10 true-0 recovered
    expect(m[0]!.f1).toBeCloseTo((2 * (8 / 9) * 0.8) / (8 / 9 + 0.8), 5);
    expect(m[1]!.precision).toBeCloseTo(9 / 11, 5);
    expect(m[1]!.recall).toBeCloseTo(9 / 10, 5);
  });

  it('gives a perfect classifier 1.0 across the board', () => {
    const m = perClassMetrics([
      [5, 0],
      [0, 7],
    ]);
    expect(m[0]).toEqual({ precision: 1, recall: 1, f1: 1, support: 5 });
    expect(m[1]).toEqual({ precision: 1, recall: 1, f1: 1, support: 7 });
  });

  it('returns 0, not NaN, for a class never predicted or with no support', () => {
    // Class 1 is never predicted (column 1 all zero) and class 2 has no support.
    const m = perClassMetrics([
      [3, 0, 0],
      [2, 0, 0],
      [0, 0, 0],
    ]);
    expect(m[1]!.precision).toBe(0);
    expect(m[1]!.recall).toBe(0);
    expect(m[1]!.f1).toBe(0);
    expect(m[2]!.support).toBe(0);
    expect(m[2]!.recall).toBe(0);
  });
});
