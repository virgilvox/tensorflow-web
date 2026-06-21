/**
 * Save, load, export, and import a project as a single local file. Export writes
 * the whole project (classes and samples) to a JSON file the user downloads;
 * import reads one back, loads it into the store, and re persists it locally.
 * Everything stays on the machine; there is no network call.
 */
import { useProjectStore } from '../stores/project';
import { serializeProject, parseProject } from '../lib/projectFile';
import { toCIdentifier } from '../lib/cformat';
import { STORES, putRecord, clearStore, storageAvailable } from '../lib/storage';
import type { Sample } from '../types';

export function useProjectFile() {
  const project = useProjectStore();

  /** Downloads the current project as a JSON file. */
  function exportToFile(): void {
    const file = serializeProject({
      name: project.name,
      modality: project.modality,
      classes: project.classes,
      samples: project.samples,
    });
    const blob = new Blob([JSON.stringify(file)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${toCIdentifier(project.name)}.tfwsproj.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }

  /**
   * Imports a project file, replacing the current project and its persisted
   * store.
   *
   * @throws if the file is not a recognized project file.
   */
  async function importFromFile(file: File): Promise<void> {
    const parsed = parseProject(JSON.parse(await file.text()));
    project.load(parsed);
    if (!storageAvailable()) return;
    await clearStore(STORES.samples);
    for (const sample of parsed.samples as Sample[]) {
      await putRecord(STORES.samples, sample);
    }
    await putRecord(STORES.meta, {
      key: 'project',
      name: parsed.name,
      modality: parsed.modality,
      classes: parsed.classes,
    });
  }

  return { exportToFile, importFromFile };
}
