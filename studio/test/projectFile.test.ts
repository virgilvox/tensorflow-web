import { describe, it, expect } from 'vitest';
import { serializeProject, parseProject } from '../src/lib/projectFile';
import type { Sample } from '../src/types';

const imageSample: Sample = {
  id: 's1',
  classId: 'c1',
  sessionId: 'sess1',
  createdAt: 1,
  payload: { kind: 'image', width: 2, height: 1, data: new Uint8ClampedArray([1, 2, 3, 4, 5, 6, 7, 8]) },
};

const textSample: Sample = {
  id: 's2',
  classId: 'c2',
  sessionId: 'sess2',
  createdAt: 2,
  payload: { kind: 'text', text: 'hello world' },
};

describe('project file round trip', () => {
  it('serializes and parses an image project without loss', () => {
    const state = {
      name: 'my project',
      modality: 'image' as const,
      classes: [
        { id: 'c1', name: 'A', negative: false },
        { id: 'c2', name: 'Other', negative: true },
      ],
      samples: [imageSample],
    };
    const file = serializeProject(state);
    // The on disk form holds plain arrays, not typed arrays.
    expect(file.version).toBe(1);
    expect(Array.isArray((file.samples[0]!.payload as { data: number[] }).data)).toBe(true);

    // Round trip through JSON, as a real save and load would.
    const restored = parseProject(JSON.parse(JSON.stringify(file)));
    expect(restored.name).toBe('my project');
    expect(restored.classes).toHaveLength(2);
    expect(restored.samples).toHaveLength(1);
    const img = restored.samples[0]!.payload;
    expect(img.kind).toBe('image');
    if (img.kind === 'image') {
      expect(img.data).toBeInstanceOf(Uint8ClampedArray);
      expect(Array.from(img.data)).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
    }
  });

  it('round trips a text project', () => {
    const file = serializeProject({
      name: 'words',
      modality: 'text',
      classes: [{ id: 'c2', name: 'Other', negative: true }],
      samples: [textSample],
    });
    const restored = parseProject(JSON.parse(JSON.stringify(file)));
    expect(restored.modality).toBe('text');
    expect(restored.samples[0]!.payload).toEqual({ kind: 'text', text: 'hello world' });
  });

  it('rejects a sample whose payload kind does not match the modality', () => {
    expect(() =>
      parseProject({
        version: 1,
        name: 'x',
        modality: 'image',
        classes: [],
        samples: [{ id: 'a', classId: 'c', sessionId: 's', createdAt: 0, payload: { kind: 'text', text: 'hi' } }],
      }),
    ).toThrow();
  });

  it('rejects input that is not a project file', () => {
    expect(() => parseProject({ hello: 'world' })).toThrow();
    expect(() => parseProject({ version: 2, classes: [], samples: [] })).toThrow();
    expect(() => parseProject(null)).toThrow();
  });

  it('rejects a corrupt sample payload instead of loading an undefined payload', () => {
    const base = { version: 1, name: 'x', modality: 'image', classes: [] };
    // Unknown payload kind.
    expect(() =>
      parseProject({
        ...base,
        samples: [{ id: 'a', classId: 'c', sessionId: 's', createdAt: 0, payload: { kind: 'video', data: [] } }],
      }),
    ).toThrow();
    // Missing payload.
    expect(() =>
      parseProject({ ...base, samples: [{ id: 'a', classId: 'c', sessionId: 's', createdAt: 0 }] }),
    ).toThrow();
    // Malformed sample fields.
    expect(() =>
      parseProject({ ...base, samples: [{ id: 5, payload: { kind: 'text', text: 'hi' } }] }),
    ).toThrow();
  });

  it('rejects a payload with non-numeric dimensions', () => {
    const base = { version: 1, name: 'x', modality: 'image', classes: [] };
    expect(() =>
      parseProject({
        ...base,
        samples: [
          { id: 'a', classId: 'c', sessionId: 's', createdAt: 0, payload: { kind: 'image', width: 'big', height: 1, data: [0] } },
        ],
      }),
    ).toThrow();
  });

  it('rejects an unknown or missing modality', () => {
    expect(() => parseProject({ version: 1, name: 'x', modality: 'video', classes: [], samples: [] })).toThrow();
    expect(() => parseProject({ version: 1, name: 'x', classes: [], samples: [] })).toThrow();
  });

  it('rejects a class missing its shape', () => {
    expect(() =>
      parseProject({ version: 1, name: 'x', modality: 'image', classes: [{ id: 'c', name: 'A' }], samples: [] }),
    ).toThrow();
  });
});
