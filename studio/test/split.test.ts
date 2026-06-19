import { describe, it, expect } from 'vitest';
import { sessionAwareSplit, singleSessionClasses } from '../src/lib/split';
import type { Sample } from '../src/types';

/** Builds a minimal image sample for a class and session. */
function sample(id: string, classId: string, sessionId: string): Sample {
  return {
    id,
    classId,
    sessionId,
    createdAt: 0,
    payload: { kind: 'image', width: 1, height: 1, data: new Uint8ClampedArray(4) },
  };
}

describe('sessionAwareSplit', () => {
  it('never splits a session across train and test', () => {
    const samples: Sample[] = [];
    // Two classes, four sessions each, three samples per session.
    for (const cls of ['a', 'b']) {
      for (let s = 0; s < 4; s++) {
        for (let n = 0; n < 3; n++) samples.push(sample(`${cls}-${s}-${n}`, cls, `${cls}-sess-${s}`));
      }
    }
    const { train, test } = sessionAwareSplit(samples, 0.25, 7);

    const trainSessions = new Set(train.map((s) => s.sessionId));
    const testSessions = new Set(test.map((s) => s.sessionId));
    for (const id of testSessions) {
      expect(trainSessions.has(id)).toBe(false);
    }
    // The union is the whole set, nothing dropped or duplicated.
    expect(train.length + test.length).toBe(samples.length);
    expect(test.length).toBeGreaterThan(0);
  });

  it('keeps a single session class entirely in train', () => {
    const samples = [
      sample('a-1', 'a', 'only'),
      sample('a-2', 'a', 'only'),
      sample('b-1', 'b', 's1'),
      sample('b-2', 'b', 's2'),
    ];
    const { train, test } = sessionAwareSplit(samples, 0.5, 1);
    // Class a has one session, so both of its samples stay in train.
    expect(train.filter((s) => s.classId === 'a').length).toBe(2);
    expect(test.filter((s) => s.classId === 'a').length).toBe(0);
  });
});

describe('singleSessionClasses', () => {
  it('flags classes that have only one capture session', () => {
    const samples = [
      sample('a-1', 'a', 's1'),
      sample('a-2', 'a', 's1'),
      sample('b-1', 'b', 's1'),
      sample('b-2', 'b', 's2'),
    ];
    expect(singleSessionClasses(samples)).toEqual(['a']);
  });
});
