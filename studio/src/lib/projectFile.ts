/**
 * Project file serialization. Converts a project (name, modality, classes, and
 * samples with their typed array payloads) to and from a JSON safe shape so it
 * can be saved to a local file and loaded back, all on the user's machine. Pure
 * and unit tested; the composable wraps the actual download and file read.
 */
import type { ClassDef, Modality, Sample, SamplePayload } from '../types';

/** The on disk project shape. Versioned so future formats can migrate. */
export interface ProjectFile {
  version: 1;
  name: string;
  modality: Modality;
  classes: ClassDef[];
  samples: SerializedSample[];
}

interface SerializedSample {
  id: string;
  classId: string;
  sessionId: string;
  createdAt: number;
  payload: SerializedPayload;
}

type SerializedPayload =
  | { kind: 'image'; width: number; height: number; data: number[] }
  | { kind: 'audio'; sampleRate: number; data: number[] }
  | { kind: 'motion'; hz: number; axes: number; data: number[] }
  | { kind: 'text'; text: string };

/** Converts a payload's typed arrays to plain number arrays for JSON. */
function serializePayload(payload: SamplePayload): SerializedPayload {
  switch (payload.kind) {
    case 'image':
      return { kind: 'image', width: payload.width, height: payload.height, data: Array.from(payload.data) };
    case 'audio':
      return { kind: 'audio', sampleRate: payload.sampleRate, data: Array.from(payload.data) };
    case 'motion':
      return { kind: 'motion', hz: payload.hz, axes: payload.axes, data: Array.from(payload.data) };
    case 'text':
      return { kind: 'text', text: payload.text };
  }
}

/** The error thrown for any file that is not a valid project file. */
function invalid(): never {
  throw new Error('Not a recognized VISE Studio project file.');
}

/** Rebuilds a payload's typed arrays from the JSON shape, validating its shape. */
function deserializePayload(payload: SerializedPayload): SamplePayload {
  if (!payload || typeof payload !== 'object') invalid();
  switch (payload.kind) {
    case 'image':
      if (!Array.isArray(payload.data)) invalid();
      return {
        kind: 'image',
        width: payload.width,
        height: payload.height,
        data: new Uint8ClampedArray(payload.data),
      };
    case 'audio':
      if (!Array.isArray(payload.data)) invalid();
      return { kind: 'audio', sampleRate: payload.sampleRate, data: Float32Array.from(payload.data) };
    case 'motion':
      if (!Array.isArray(payload.data)) invalid();
      return { kind: 'motion', hz: payload.hz, axes: payload.axes, data: Float32Array.from(payload.data) };
    case 'text':
      if (typeof payload.text !== 'string') invalid();
      return { kind: 'text', text: payload.text };
    default:
      // An unknown payload kind means a corrupt or foreign file. Fail loudly
      // here rather than load a sample with an undefined payload that crashes
      // deep in feature extraction later.
      return invalid();
  }
}

/** Serializes the current project state into the file shape. */
export function serializeProject(state: {
  name: string;
  modality: Modality;
  classes: ClassDef[];
  samples: Sample[];
}): ProjectFile {
  return {
    version: 1,
    name: state.name,
    modality: state.modality,
    classes: state.classes.map((c) => ({ ...c })),
    samples: state.samples.map((s) => ({
      id: s.id,
      classId: s.classId,
      sessionId: s.sessionId,
      createdAt: s.createdAt,
      payload: serializePayload(s.payload),
    })),
  };
}

/**
 * Parses and validates a project file into the in memory state.
 *
 * @returns the project state ready to load into the store.
 * @throws if the input is not a recognized project file.
 */
export function parseProject(input: unknown): {
  name: string;
  modality: Modality;
  classes: ClassDef[];
  samples: Sample[];
} {
  const file = input as Partial<ProjectFile>;
  if (!file || file.version !== 1 || !Array.isArray(file.classes) || !Array.isArray(file.samples)) {
    invalid();
  }
  const samples: Sample[] = file.samples.map((s) => {
    if (
      !s ||
      typeof s.id !== 'string' ||
      typeof s.classId !== 'string' ||
      typeof s.sessionId !== 'string' ||
      typeof s.createdAt !== 'number'
    ) {
      invalid();
    }
    return {
      id: s.id,
      classId: s.classId,
      sessionId: s.sessionId,
      createdAt: s.createdAt,
      payload: deserializePayload(s.payload),
    };
  });
  return {
    name: file.name ?? 'untitled',
    modality: (file.modality ?? 'image') as Modality,
    classes: file.classes.map((c) => ({ ...c })),
    samples,
  };
}
