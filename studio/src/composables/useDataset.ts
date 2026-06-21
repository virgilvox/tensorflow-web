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

// Module level so the one time load survives stage view remounts: the store
// outlives any single view, so reloading from storage on every mount would
// clobber in memory edits (and a just imported project) with stale data.
let loadedOnce = false;

export function useDataset() {
  const project = useProjectStore();

  /** Loads any persisted project and samples into the store once per session. */
  async function init(): Promise<void> {
    if (loadedOnce || !storageAvailable()) return;
    loadedOnce = true;
    try {
      const [metas, samples] = await Promise.all([
        getAll<ProjectMeta>(STORES.meta),
        getAll<Sample>(STORES.samples),
      ]);
      const meta = metas.find((m) => m.key === 'project');
      // Nothing persisted yet: leave the store as is. This also avoids an
      // in flight init from clobbering a project the user just imported.
      if (!meta && samples.length === 0) return;
      project.load({
        name: meta?.name,
        modality: meta?.modality,
        classes: meta?.classes,
        samples,
      });
      // Self heal: drop any persisted sample whose class is missing from the
      // loaded class list. Such orphans cannot be trained or labelled, and they
      // would otherwise crash the Train stage. This also recovers a dataset left
      // inconsistent by an older build.
      const known = new Set(project.classes.map((c) => c.id));
      const orphans = project.samples.filter((s) => !known.has(s.classId));
      if (orphans.length > 0) {
        console.warn(`TF Web Studio: dropping ${orphans.length} sample(s) with no matching class.`);
        for (const orphan of orphans) {
          project.removeSample(orphan.id);
          try {
            await deleteRecord(STORES.samples, orphan.id);
          } catch {
            // best effort cleanup
          }
        }
      }
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
      // Plain objects, not the store's reactive proxies: IndexedDB structured
      // clone throws on a Proxy, which previously made every meta write fail
      // silently and lose the class list on reload.
      classes: project.classes.map((c) => ({ id: c.id, name: c.name, negative: c.negative })),
    };
    try {
      await putRecord(STORES.meta, meta);
    } catch (err) {
      // Surface the failure rather than hide it; the in memory state is still
      // correct but persistence is broken and that must not be silent.
      console.error('TF Web Studio: failed to persist project meta', err);
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

  /** Adds a motion window sample, persists it, and returns the stored record. */
  async function addMotionSample(
    classId: string,
    sessionId: string,
    window: { hz: number; axes: number; data: Float32Array },
  ): Promise<Sample> {
    const sample = project.addSample({
      classId,
      sessionId,
      payload: { kind: 'motion', hz: window.hz, axes: window.axes, data: window.data },
    });
    await persistSample(sample);
    return sample;
  }

  /** Adds a text sample, persists it, and returns the stored record. */
  async function addTextSample(classId: string, sessionId: string, text: string): Promise<Sample> {
    const sample = project.addSample({ classId, sessionId, payload: { kind: 'text', text } });
    await persistSample(sample);
    return sample;
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

  return {
    init,
    persistMeta,
    addImageSample,
    addAudioSample,
    addMotionSample,
    addTextSample,
    removeSample,
    removeClass,
    clearAll,
  };
}
