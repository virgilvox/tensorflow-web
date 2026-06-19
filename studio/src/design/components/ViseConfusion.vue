<script setup lang="ts">
/**
 * ViseConfusion renders a confusion matrix as a dense HTML table. Rows are the
 * true class and columns are the predicted class, so matrix[true][pred] holds
 * the count of samples of a given true label that were predicted as another.
 * Diagonal cells (correct predictions) carry a caution tint scaled by how large
 * the count is relative to the largest count in that row. Off diagonal non zero
 * cells carry a faint rust tint on the same scale, so misclassification mass is
 * visible at a glance.
 */

const props = defineProps<{
  /**
   * Counts indexed as matrix[trueClass][predictedClass]. Each inner array is
   * one true class row, each position within it is a predicted class column.
   */
  matrix: number[][]
  /**
   * Class names in index order. Used for both the column header and the row
   * header, since the axes share the same set of classes.
   */
  labels: string[]
}>()

/**
 * Largest count in a given row. Used to normalize the tint opacity per row so a
 * sparse class still shows contrast. Returns 0 for an empty or missing row.
 */
function rowMax(rowIndex: number): number {
  const row = props.matrix[rowIndex]
  if (!row || row.length === 0) return 0
  return Math.max(0, ...row)
}

/**
 * Background color for a cell. The diagonal uses the caution accent, off
 * diagonal non zero cells use rust, and empty off diagonal cells stay
 * untinted. Opacity scales 0 to 0.32 by count over the row max, guarded
 * against division by zero.
 */
function cellBackground(rowIndex: number, colIndex: number, count: number): string {
  const max = rowMax(rowIndex)
  const ratio = max > 0 ? count / max : 0
  const alpha = ratio * 0.32
  if (rowIndex === colIndex) {
    return `rgba(204, 242, 62, ${alpha})`
  }
  if (count > 0) {
    return `rgba(224, 89, 63, ${alpha})`
  }
  return 'transparent'
}
</script>

<template>
  <table class="confusion">
    <thead>
      <tr>
        <th class="corner" scope="col"></th>
        <th
          v-for="(label, colIndex) in labels"
          :key="'col-' + colIndex"
          class="col-head"
          scope="col"
        >
          {{ label }}
        </th>
      </tr>
    </thead>
    <tbody>
      <tr v-for="(label, rowIndex) in labels" :key="'row-' + rowIndex">
        <th class="row-head" scope="row">{{ label }}</th>
        <td
          v-for="(count, colIndex) in matrix[rowIndex] ?? []"
          :key="'cell-' + rowIndex + '-' + colIndex"
          class="cell"
          :class="{ diag: rowIndex === colIndex }"
          :style="{ background: cellBackground(rowIndex, colIndex, count) }"
        >
          {{ count }}
        </td>
      </tr>
    </tbody>
  </table>
</template>

<style scoped>
.confusion {
  border-collapse: collapse;
  background: var(--graphite);
  border: 1px solid var(--seam);
  font-family: var(--f-mono);
  font-variant-numeric: tabular-nums;
}
.confusion th,
.confusion td {
  border: 1px solid var(--seam);
  padding: 6px 9px;
  text-align: right;
}
.col-head,
.row-head,
.corner {
  font-family: var(--f-label);
  font-size: var(--t-xs);
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--ash);
  font-weight: 600;
  background: var(--gunmetal);
}
.row-head {
  text-align: left;
}
.cell {
  color: var(--steam);
  font-size: var(--t-sm);
  font-variant-numeric: tabular-nums;
}
.cell.diag {
  color: var(--chalk);
}
</style>
