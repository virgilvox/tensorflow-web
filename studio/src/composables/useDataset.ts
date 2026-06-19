/**
 * Dataset CRUD with local persistence. Adds and removes samples through the
 * project store and mirrors them into IndexedDB so a refresh keeps the work and
 * nothing is uploaded. Samples store their raw payload, so features recompute
 * when the feature config changes. Storage failures degrade to in memory only
 * rather than breaking the flow.
 */
import type { RgbaFrame } from '../features/image';
import { useProjectStore } from '../stores/project';
import type { ClassDef, Modality, Sample } from '../types';
import {
  STORES,
  getAll,
  putRecord,
  deleteRecord,
  clearStore,
  storageAvailable,
} from '../lib/storage';

/** The single project meta record persisted alongside the samples. */
interface ProjectMeta {
  key: 'project';
  name: string;
  modality: Modality;
  classes: ClassDef[];
}

export function useDataset() {
  const project = useProjectStore();

  /** Loads any persisted project and samples into the store. Safe to call once. */
  async function init(): Promise<void> {
    if (!storageAvailable()) return;
    try {
      const [metas, samples] = await Promise.all([
        getAll<ProjectMeta>(STORES.meta),
        getAll<Sample>(STORES.samples),
      ]);
      const meta = metas.find((m) => m.key === 'project');
      project.load({
        name: meta?.name,
        modality: meta?.modality,
        classes: meta?.classes,
        samples,
      });
    } catch {
      // A blocked or unavailable database is not fatal; carry on in memory.
    }
  }

  /** Writes the current project meta (name, modality, classes) to storage. */
  async function persistMeta(): Promise<void> {
    if (!storageAvailable()) return;
    const meta: ProjectMeta = {
      key: 'project',
      name: project.name,
      modality: project.modality,
      classes: [...project.classes],
    };
    try {
      await putRecord(STORES.meta, meta);
    } catch {
      // ignore, in memory state is still correct
    }
  }

  /** Adds an image sample, persists it, and returns the stored record. */
  async function addImageSample(classId: string, sessionId: string, frame: RgbaFrame): Promise<Sample> {
    const sample = project.addSample({
      classId,
      sessionId,
      payload: { kind: 'image', width: frame.width, height: frame.height, data: frame.data },
    });
    await persistSample(sample);
    return sample;
  }

  /** Adds an audio sample, persists it, and returns the stored record. */
  async function addAudioSample(
    classId: string,
    sessionId: string,
    clip: { data: Float32Array; sampleRate: number },
  ): Promise<Sample> {
    const sample = project.addSample({
      classId,
      sessionId,
      payload: { kind: 'audio', sampleRate: clip.sampleRate, data: clip.data },
    });
    await persistSample(sample);
    return sample;
  }

  /** Persists one sample record. */
  async function persistSample(sample: Sample): Promise<void> {
    if (!storageAvailable()) return;
    try {
      await putRecord(STORES.samples, sample);
    } catch {
      // ignore
    }
  }

  /** Removes a sample from the store and storage. */
  async function removeSample(id: string): Promise<void> {
    project.removeSample(id);
    if (storageAvailable()) {
      try {
        await deleteRecord(STORES.samples, id);
      } catch {
        // ignore
      }
    }
  }

  /** Clears all samples for a class and removes the class. */
  async function removeClass(id: string): Promise<void> {
    const toDelete = project.samples.filter((s) => s.classId === id).map((s) => s.id);
    project.removeClass(id);
    await persistMeta();
    if (storageAvailable()) {
      for (const sampleId of toDelete) {
        try {
          await deleteRecord(STORES.samples, sampleId);
        } catch {
          // ignore
        }
      }
    }
  }

  /** Empties the whole dataset and the persisted stores. */
  async function clearAll(): Promise<void> {
    project.reset();
    if (storageAvailable()) {
      try {
        await Promise.all([clearStore(STORES.samples), clearStore(STORES.meta)]);
      } catch {
        // ignore
      }
    }
  }

  return { init, persistMeta, addImageSample, addAudioSample, removeSample, removeClass, clearAll };
}
