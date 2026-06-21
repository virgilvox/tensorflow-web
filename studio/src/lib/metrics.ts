/**
 * Per-class classification metrics derived from a confusion matrix. Pure and
 * unit tested. The Test stage already shows the confusion matrix from verify;
 * this turns it into the precision, recall, and F1 a reader expects per class,
 * so a model that is strong overall but weak on one class is visible.
 */

/** Precision, recall, F1, and support (true count) for one class. */
export interface ClassMetric {
  precision: number;
  recall: number;
  f1: number;
  support: number;
}

/**
 * Computes per-class precision, recall, F1, and support from a confusion matrix
 * laid out with rows as the true class and columns as the predicted class.
 *
 * @param confusion a square matrix; `confusion[t][p]` is the count of true class
 *   t predicted as p.
 * @returns one metric record per class, in the matrix's class order. A class with
 *   no predictions or no support yields 0 for the affected metric rather than NaN.
 */
export function perClassMetrics(confusion: number[][]): ClassMetric[] {
  const n = confusion.length;
  return confusion.map((row, i) => {
    const tp = row[i] ?? 0;
    const support = row.reduce((a, b) => a + b, 0); // all true class i
    let predicted = 0; // all predicted as i, across true classes
    for (let t = 0; t < n; t++) predicted += confusion[t]?.[i] ?? 0;
    const fp = predicted - tp;
    const fn = support - tp;
    const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
    const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
    const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
    return { precision, recall, f1, support };
  });
}
