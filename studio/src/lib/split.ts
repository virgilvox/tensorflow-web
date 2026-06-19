/**
 * Session aware train and test split. Whole capture sessions are assigned to one
 * side of the split, never individual samples, so the same subject or recording
 * cannot appear in both train and test and inflate the score. Pure and
 * deterministic given a seed, so it is unit testable and reproducible.
 */
import type { Sample } from '../types';

export interface Split {
  train: Sample[];
  test: Sample[];
}

/** A tiny deterministic PRNG so the split is reproducible without a global seed. */
function mulberry32(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Splits samples into train and test by whole capture session, aiming for the
 * requested test fraction within each class so the split stays balanced. A class
 * with a single session keeps that session in train, since a one session class
 * cannot be split without leakage; such classes are reported by the caller as a
 * coverage warning rather than silently leaked.
 *
 * @param samples all samples across all classes.
 * @param testFraction target fraction of samples held out for test, in (0, 1).
 * @param seed optional seed for reproducible session assignment.
 * @returns the train and test sample arrays. Their union is the input set.
 */
export function sessionAwareSplit(samples: Sample[], testFraction = 0.25, seed = 1): Split {
  const rand = mulberry32(seed);
  const train: Sample[] = [];
  const test: Sample[] = [];

  // Group by class, then by session, so the test fraction is per class.
  const byClass = new Map<string, Map<string, Sample[]>>();
  for (const s of samples) {
    let sessions = byClass.get(s.classId);
    if (!sessions) {
      sessions = new Map();
      byClass.set(s.classId, sessions);
    }
    const bucket = sessions.get(s.sessionId);
    if (bucket) bucket.push(s);
    else sessions.set(s.sessionId, [s]);
  }

  for (const sessions of byClass.values()) {
    const sessionIds = [...sessions.keys()];
    // A single session cannot be split without leaking; keep it in train.
    if (sessionIds.length < 2) {
      for (const id of sessionIds) train.push(...sessions.get(id)!);
      continue;
    }
    // Shuffle sessions, then peel sessions into test until the fraction is met.
    for (let i = sessionIds.length - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      [sessionIds[i], sessionIds[j]] = [sessionIds[j]!, sessionIds[i]!];
    }
    const total = [...sessions.values()].reduce((n, b) => n + b.length, 0);
    const targetTest = Math.round(total * testFraction);
    let inTest = 0;
    for (const id of sessionIds) {
      const bucket = sessions.get(id)!;
      // Assign to test while under target and at least one session stays in train.
      const remainingTrainSessions = sessionIds.length - sessionIds.indexOf(id);
      if (inTest < targetTest && remainingTrainSessions > 1) {
        test.push(...bucket);
        inTest += bucket.length;
      } else {
        train.push(...bucket);
      }
    }
  }

  return { train, test };
}

/** Class ids that have only one capture session, so they cannot be split. */
export function singleSessionClasses(samples: Sample[]): string[] {
  const sessionsByClass = new Map<string, Set<string>>();
  for (const s of samples) {
    const set = sessionsByClass.get(s.classId) ?? new Set<string>();
    set.add(s.sessionId);
    sessionsByClass.set(s.classId, set);
  }
  const out: string[] = [];
  for (const [classId, sessions] of sessionsByClass) {
    if (sessions.size < 2) out.push(classId);
  }
  return out;
}
