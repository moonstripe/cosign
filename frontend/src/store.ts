import { create } from 'zustand';
import { initTree, type TreeState } from './tree/model';
import { DEFAULT_AXES, type Axis, type WheelPick } from './lib/wheel';
import type { Candidate, DialAxis, PangramResult, Signpost, Src } from './types';

const LS = { signposts: 'cowrite.held.v1', tree: 'cowrite.tree.v2', doc: 'cowrite.doc', axes: 'cowrite.axes' };

export type Action = 'write' | 'choose' | 'sentence';

let sid = 0;
export const newSignpostId = () => `sp${Date.now().toString(36)}${(sid++).toString(36)}`;

function load<T>(key: string, fallback: T): T {
  try {
    const v = localStorage.getItem(key);
    return v ? (JSON.parse(v) as T) : fallback;
  } catch {
    return fallback;
  }
}

export interface Popover { items: Candidate[]; x: number; y: number; sel: number }

export interface AppState {
  conn: { connected: boolean; model?: string; device?: string };
  generating: boolean;
  signposts: Signpost[];    // moods held on with the dial sliders (dial-backed only)
  dialAxes: DialAxis[];     // moods the server has programmed dials for
  tab: 'model' | 'view';
  modelId: string;          // dummy picker for now: only LFM3-1.7B is served
  budget: number;
  temp: number;
  showColors: boolean;
  showSegments: boolean;
  authorship: Record<Src, number>;
  docText: string;
  pangram: { text: string | null; result: PangramResult | null; busy: boolean; error: string | null };
  popover: Popover | null;
  wheel: { action: Action; x: number; y: number; pick: WheelPick } | null;   // open radial picker
  axes: Axis[];             // the wheel's moods (mood / opposite)
  oneShot: Signpost[];      // steering picked on the wheel, for the current action only
  hoverNode: string | null;   // tree node under the mouse (in the tree or the text)
  tree: TreeState;
  // the open document (server copy in documents/, see lib/docs.ts)
  docId: string | null;
  docTitle: string;
  saveState: 'idle' | 'unsaved' | 'saving' | 'saved' | 'error';
  saveError: string | null;
  set: (p: Partial<AppState>) => void;
  setTree: (f: (t: TreeState) => TreeState) => void;
  setPangram: (p: Partial<AppState['pangram']>) => void;
}

function currentDoc(): { docId: string | null; docTitle: string } {
  try {
    const c = JSON.parse(localStorage.getItem('cowrite.current') ?? 'null');
    if (c && typeof c === 'object') return { docId: c.id ?? null, docTitle: c.title || 'Untitled' };
  } catch { /* fall through */ }
  return { docId: null, docTitle: 'Untitled' };
}

function loadTree(): TreeState {
  const t = load<TreeState | null>(LS.tree, null);
  return t && t.nodes && t.nodes[t.cur] ? t : initTree();
}

export const useApp = create<AppState>((set) => ({
  conn: { connected: false },
  generating: false,
  signposts: [],           // steering comes from the wheel only (no held moods)
  dialAxes: [],
  tab: 'model',
  modelId: 'LiquidAI/LFM3-1.7B',
  budget: 8,
  temp: 0.8,
  showColors: true,
  showSegments: true,
  authorship: { human: 0, model: 0, picked: 0, doc: 0 },
  docText: '',
  pangram: { text: null, result: null, busy: false, error: null },
  popover: null,
  wheel: null,
  axes: load<Axis[]>(LS.axes, DEFAULT_AXES),
  oneShot: [],
  hoverNode: null,
  tree: loadTree(),
  ...currentDoc(),
  saveState: 'idle',
  saveError: null,
  set: (p) => set(p),
  setTree: (f) => set((s) => ({ tree: f(s.tree) })),
  setPangram: (p) => set((s) => ({ pangram: { ...s.pangram, ...p } })),
}));

// persistence
useApp.subscribe((s, prev) => {
  if (s.signposts !== prev.signposts) localStorage.setItem(LS.signposts, JSON.stringify(s.signposts));
  if (s.axes !== prev.axes) localStorage.setItem(LS.axes, JSON.stringify(s.axes));
  if (s.tree !== prev.tree) {
    try { localStorage.setItem(LS.tree, JSON.stringify(s.tree)); } catch { /* quota */ }
  }
});

export const savedDoc = () => localStorage.getItem(LS.doc) ?? '';
export const saveDoc = (html: string) => localStorage.setItem(LS.doc, html);
export const clearSaved = () => { localStorage.removeItem(LS.doc); localStorage.removeItem(LS.tree); };
