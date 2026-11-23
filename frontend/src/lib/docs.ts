// Documents: server-side CRUD (/api/docs) + autosave + files on disk.
//
// The editor's working copy also lives in localStorage (so a reload never loses
// typing); this module keeps the server copy in step and switches documents.

import { editor } from './editorController';
import { savedDoc, useApp } from '../store';
import type { TreeState } from '../tree/model';
import type { PangramResult } from '../types';

export interface DocSummary { id: string; title: string; created: number; updated: number; words: number; preview: string }
export interface DocFull { id: string; title: string; html: string; text: string; tree: TreeState | null; pangram: PangramResult | null }

const CURRENT = 'cowrite.current';
const SAVE_IDLE_MS = 800;

async function api<T>(method: string, url: string, body?: unknown): Promise<T> {
  const r = await fetch(url, { method, headers: body ? { 'content-type': 'application/json' } : undefined, body: body ? JSON.stringify(body) : undefined });
  const j = await r.json().catch(() => ({}));
  if (!r.ok || (j && j.error)) throw new Error(j?.error ?? `HTTP ${r.status}`);
  return j as T;
}

export const listDocs = () => api<DocSummary[]>('GET', '/api/docs');
const readDoc = (id: string) => api<DocFull>('GET', `/api/docs/${id}`);
const createDoc = (body: Partial<DocFull>) => api<DocFull>('POST', '/api/docs', body);
const updateDoc = (id: string, body: Partial<DocFull>) => api<DocSummary>('PUT', `/api/docs/${id}`, body);
const removeDoc = (id: string) => api<{ ok: boolean }>('DELETE', `/api/docs/${id}`);

// ------------------------------------------------------------ current document

function remember() {
  const { docId, docTitle } = useApp.getState();
  localStorage.setItem(CURRENT, JSON.stringify({ id: docId, title: docTitle }));
}

let timer: number | undefined;
let loading = false;      // switching documents: don't autosave the half-loaded state
let inflight: Promise<void> | null = null;

function body(): Partial<DocFull> {
  const s = useApp.getState();
  return { title: s.docTitle, html: savedDoc(), text: s.docText, tree: s.tree, pangram: s.pangram.result };
}

/** Save now (create on the server the first time there is something to save). */
export async function saveNow(): Promise<void> {
  clearTimeout(timer);
  if (inflight) await inflight;
  const s = useApp.getState();
  if (!s.docId && !s.docText.trim()) return;           // nothing worth a document yet
  const run = (async () => {
    s.set({ saveState: 'saving' });
    try {
      if (useApp.getState().docId) await updateDoc(useApp.getState().docId!, body());
      else {
        const d = await createDoc(body());
        useApp.getState().set({ docId: d.id });
      }
      useApp.getState().set({ saveState: 'saved', saveError: null });
      remember();
    } catch (e) {
      useApp.getState().set({ saveState: 'error', saveError: (e as Error).message });
    }
  })();
  inflight = run;
  await run;
  inflight = null;
}

function scheduleSave() {
  if (loading) return;
  clearTimeout(timer);
  useApp.getState().set({ saveState: 'unsaved' });
  timer = window.setTimeout(saveNow, SAVE_IDLE_MS);
}

useApp.subscribe((s, p) => {
  if (s.docText !== p.docText || s.tree !== p.tree || s.docTitle !== p.docTitle) scheduleSave();
  if (s.docId !== p.docId || s.docTitle !== p.docTitle) remember();
});
window.addEventListener('beforeunload', () => { if (useApp.getState().saveState === 'unsaved') void saveNow(); });

async function switching(fn: () => Promise<void> | void) {
  if (useApp.getState().saveState === 'unsaved') await saveNow();
  loading = true;
  try { await fn(); } finally { loading = false; }
}

export async function newDocument(title = 'Untitled') {
  await switching(() => {
    useApp.getState().set({ docId: null, docTitle: title, saveState: 'idle', saveError: null });
    editor.reset();
  });
}

export async function openDocument(id: string) {
  await switching(async () => {
    const d = await readDoc(id);
    useApp.getState().set({ docId: d.id, docTitle: d.title, saveState: 'saved', saveError: null });
    editor.loadDocument(d.html, d.tree, d.pangram, d.text);
  });
}

export async function deleteDocument(id: string) {
  await removeDoc(id);
  if (useApp.getState().docId === id) {
    useApp.getState().set({ docId: null });      // don't let a pending save resurrect it
    clearTimeout(timer);
    await newDocument();
  }
}

export function renameDocument(title: string) {
  useApp.getState().set({ docTitle: title.trim() || 'Untitled' });
}

// ------------------------------------------------------------ files on disk

function download(name: string, data: string, type: string) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([data], { type }));
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

const safeName = (t: string) => (t.trim() || 'Untitled').replace(/[\\/:*?"<>|]+/g, '-').slice(0, 80);

/** .cosign.json keeps everything: authorship spans, the decision tree, the Pangram result. */
export function downloadCosign() {
  const s = useApp.getState();
  download(`${safeName(s.docTitle)}.cosign.json`,
    JSON.stringify({ format: 'cosign', version: 1, ...body(), title: s.docTitle }, null, 1), 'application/json');
}

export function downloadText() {
  const s = useApp.getState();
  download(`${safeName(s.docTitle)}.txt`, s.docText, 'text/plain');
}

const stem = (name: string) => name.replace(/(\.cosign|\.cowrite)?\.(json|txt|md|markdown|pdf)$/i, '');

/** Open a file from disk: .cosign.json (or a legacy .cowrite.json) restores everything; .txt/.md import as a chunk chain. */
export async function openFile(f: File) {
  const text = await f.text();
  if (/\.json$/i.test(f.name)) {
    let d: Partial<DocFull> & { format?: string };
    try { d = JSON.parse(text); } catch { throw new Error('not a valid .cosign.json file'); }
    if ((d.format !== 'cosign' && d.format !== 'cowrite') || typeof d.html !== 'string') throw new Error('not a .cosign.json file');
    await switching(() => {
      useApp.getState().set({ docId: null, docTitle: d.title || stem(f.name), saveState: 'idle', saveError: null });
      editor.loadDocument(d.html!, d.tree ?? null, d.pangram ?? null, d.text);
    });
    await saveNow();                                   // becomes a new document on the server
    return 'opened';
  }
  // plain text / markdown: paragraphs become a linear chain, like a PDF import
  const chunks = text.replace(/\r\n?/g, '\n').split(/\n\s*\n/).map((t) => t.trim()).filter(Boolean).map((t) => ({ text: t, page: 1 }));
  if (!chunks.length) throw new Error('that file is empty');
  await importChunks(chunks, f.name);
  return `${chunks.length} paragraphs`;
}

/** Start a new document from imported chunks (PDF via Reducto, or a text file). */
export async function importChunks(chunks: { text: string; page: number }[], file: string) {
  await switching(() => { useApp.getState().set({ docId: null, docTitle: stem(file), saveState: 'idle', saveError: null }); });
  await editor.importDoc(chunks, file);
  await saveNow();
}
