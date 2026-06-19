/**
 * Project store: the active modality, the class list including the scaffolded
 * negative class, and the captured samples grouped by capture session. Feature
 * and model configuration attach here as later phases fill them in. The bench
 * rail and every stage read this store.
 */
import { defineStore } from 'pinia';
import type { ClassDef, Modality, Sample } from '../types';

let idCounter = 0;
/** Monotonic id source. Avoids Math.random so ids stay stable and ordered. */
function nextId(prefix: string): string {
  idCounter += 1;
  return `${prefix}-${idCounter}`;
}

/** The auto offered negative class label per modality. */
const NEGATIVE_LABEL: Record<Modality, string> = {
  image: 'Neither',
  audio: 'Background Noise',
  motion: 'Idle',
  text: 'Other',
};

interface ProjectState {
  name: string;
  modality: Modality;
  classes: ClassDef[];
  samples: Sample[];
}

export const useProjectStore = defineStore('project', {
  state: (): ProjectState => ({
    name: 'untitled',
    modality: 'image',
    classes: [],
    samples: [],
  }),
  getters: {
    /** Sample count per class id, for the bench rail and balance warnings. */
    countsByClass(state): Record<string, number> {
      const counts: Record<string, number> = {};
      for (const c of state.classes) counts[c.id] = 0;
      for (const s of state.samples) counts[s.classId] = (counts[s.classId] ?? 0) + 1;
      return counts;
    },
    totalSamples: (state): number => state.samples.length,
    /** Distinct positive (non negative) classes, the ones the user names. */
    positiveClasses: (state): ClassDef[] => state.classes.filter((c) => !c.negative),
    negativeClass: (state): ClassDef | undefined => state.classes.find((c) => c.negative),
  },
  actions: {
    setModality(modality: Modality): void {
      this.modality = modality;
    },
    addClass(name: string, negative = false): ClassDef {
      const cls: ClassDef = { id: nextId('class'), name, negative };
      this.classes.push(cls);
      return cls;
    },
    removeClass(id: string): void {
      this.classes = this.classes.filter((c) => c.id !== id);
      this.samples = this.samples.filter((s) => s.classId !== id);
    },
    /** Adds the scaffolded negative class for the current modality if absent. */
    ensureNegativeClass(): ClassDef {
      const existing = this.classes.find((c) => c.negative);
      if (existing) return existing;
      return this.addClass(NEGATIVE_LABEL[this.modality], true);
    },
    addSample(sample: Omit<Sample, 'id' | 'createdAt'>): Sample {
      const full: Sample = { ...sample, id: nextId('sample'), createdAt: idCounter };
      this.samples.push(full);
      return full;
    },
    reset(): void {
      this.classes = [];
      this.samples = [];
      this.name = 'untitled';
    },
  },
});
