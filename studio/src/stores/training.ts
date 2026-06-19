/**
 * Training store: the live job status and metric history the Train stage writes
 * and the bench rail reads, plus the artifacts the later stages produce (the
 * trained model handle, the emitted bytes, the verify report, and the device
 * budget result). Phase 1 establishes the shape; later phases fill it in.
 */
import { defineStore } from 'pinia';
import type { VerifyReport } from 'tensorflow-web';

export type JobStatus = 'idle' | 'running' | 'done' | 'cancelled' | 'error';

/** One epoch's metrics, captured for the loss curve and the gauges. */
export interface EpochMetric {
  epoch: number;
  loss: number;
  acc?: number;
  valLoss?: number;
  valAcc?: number;
}

/** The device budget result the meter renders. */
export interface BudgetResult {
  flashBytes: number;
  arenaBytes: number;
  fits: boolean;
}

interface TrainingState {
  status: JobStatus;
  epoch: number;
  totalEpochs: number;
  metrics: EpochMetric[];
  message: string;
  tfliteBytes: number | null;
  report: VerifyReport | null;
  budget: BudgetResult | null;
}

export const useTrainingStore = defineStore('training', {
  state: (): TrainingState => ({
    status: 'idle',
    epoch: 0,
    totalEpochs: 0,
    metrics: [],
    message: '',
    tfliteBytes: null,
    report: null,
    budget: null,
  }),
  getters: {
    latest: (s): EpochMetric | undefined => s.metrics[s.metrics.length - 1],
    running: (s): boolean => s.status === 'running',
    /** Fraction of the run completed, for the progress gauge. */
    progress: (s): number => (s.totalEpochs > 0 ? Math.min(1, s.epoch / s.totalEpochs) : 0),
  },
  actions: {
    begin(totalEpochs: number): void {
      this.status = 'running';
      this.epoch = 0;
      this.totalEpochs = totalEpochs;
      this.metrics = [];
      this.message = 'training';
      this.tfliteBytes = null;
      this.report = null;
      this.budget = null;
    },
    pushEpoch(metric: EpochMetric): void {
      this.epoch = metric.epoch;
      this.metrics.push(metric);
    },
    finish(status: JobStatus, message = ''): void {
      this.status = status;
      this.message = message;
    },
    setArtifacts(tfliteBytes: number, report: VerifyReport | null, budget: BudgetResult | null): void {
      this.tfliteBytes = tfliteBytes;
      this.report = report;
      this.budget = budget;
    },
  },
});
