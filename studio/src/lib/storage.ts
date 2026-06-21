/**
 * A small promise based IndexedDB wrapper. The studio persists the project and
 * its samples locally so a refresh does not lose work and nothing is uploaded.
 * Typed array payloads (image pixels, audio, motion) survive the structured
 * clone IndexedDB uses, so samples are stored as is. No Vue, no library.
 */

const DB_NAME = 'tf-web-studio';
const DB_VERSION = 1;

/** The object stores the studio keeps: one for samples, one for project meta. */
export const STORES = { samples: 'samples', meta: 'meta' } as const;
export type StoreName = (typeof STORES)[keyof typeof STORES];

let dbPromise: Promise<IDBDatabase> | null = null;

/** True when IndexedDB is available in this context (it is absent in some workers). */
export function storageAvailable(): boolean {
  return typeof indexedDB !== 'undefined';
}

/** Opens the database, creating the object stores on first use. Cached. */
function openDb(): Promise<IDBDatabase> {
  if (dbPromise) return dbPromise;
  dbPromise = new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORES.samples)) {
        db.createObjectStore(STORES.samples, { keyPath: 'id' });
      }
      if (!db.objectStoreNames.contains(STORES.meta)) {
        db.createObjectStore(STORES.meta, { keyPath: 'key' });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error ?? new Error('Failed to open IndexedDB'));
  });
  return dbPromise;
}

/** Wraps an IDBRequest in a promise. */
function wrap<T>(req: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error ?? new Error('IndexedDB request failed'));
  });
}

/** Stores a record by its key path. Resolves when the write commits. */
export async function putRecord<T>(store: StoreName, value: T): Promise<void> {
  const db = await openDb();
  const tx = db.transaction(store, 'readwrite');
  tx.objectStore(store).put(value);
  await new Promise<void>((resolve, reject) => {
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error ?? new Error('IndexedDB transaction failed'));
  });
}

/** Reads every record in a store. */
export async function getAll<T>(store: StoreName): Promise<T[]> {
  const db = await openDb();
  return wrap(db.transaction(store, 'readonly').objectStore(store).getAll() as IDBRequest<T[]>);
}

/** Deletes one record by key. */
export async function deleteRecord(store: StoreName, key: IDBValidKey): Promise<void> {
  const db = await openDb();
  const tx = db.transaction(store, 'readwrite');
  tx.objectStore(store).delete(key);
  await new Promise<void>((resolve, reject) => {
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error ?? new Error('IndexedDB transaction failed'));
  });
}

/** Empties a store. */
export async function clearStore(store: StoreName): Promise<void> {
  const db = await openDb();
  const tx = db.transaction(store, 'readwrite');
  tx.objectStore(store).clear();
  await new Promise<void>((resolve, reject) => {
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error ?? new Error('IndexedDB transaction failed'));
  });
}
