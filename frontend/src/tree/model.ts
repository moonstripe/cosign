// Decision tree: pure state transitions, no DOM.
//
// Every decision — a run of typing, ⌥Tab, ⌘↵, or a ⇧Tab pick chain — is a node
// holding a snapshot of the document. Going to a node restores its snapshot;
// acting from there grows a new branch. Deleting from the end walks back up.

import type { PangramResult, PgFractions, Snapshot, Src, Via } from '../types';

export type NodeKind = 'root' | Src;

/** A mood dial setting used for a decision: the axis and where on it you aimed. */
export interface SteerChoice { axis: [string, string]; toward: string; strength: number }

/** One paragraph of an imported PDF (Reducto): its position in the import chain and source page. */
export interface ChunkInfo { i: number; n: number; page: number; file: string }

export interface TreeNode extends Snapshot {
  id: string;
  parent: string | null;
  kind: NodeKind;
  via: Via;
  inserted: string;
  removed: string;
  meta: { steer?: SteerChoice[]; chunk?: ChunkInfo; signposts?: string[] };   // signposts: legacy, not shown
  t: number;
  pg: PgFractions | null;          // Pangram on the whole document at this node
  chunkPg?: PgFractions | null;     // imported chunks: this paragraph's score inside the full-document scan
}

export interface TreeState {
  nodes: Record<string, TreeNode>;
  cur: string;
  seq: number;
  // human typing / pick chains merge into the node they started
  session: { kind: NodeKind; id: string } | null;
  pgCache: Record<string, PgFractions>;
  // the full-document Pangram scan made at import: prefixes of it (clicking back along the
  // import chain) reuse its windows instead of re-scoring a shorter text
  importScan?: { text: string; result: PangramResult } | null;
}

export const ROOT = 'n0';

export function initTree(now = Date.now()): TreeState {
  return {
    nodes: {
      [ROOT]: {
        id: ROOT, parent: null, kind: 'root', via: 'root', html: '', text: '', caret: 0,
        inserted: '', removed: '', meta: {}, t: now, pg: null,
      },
    },
    cur: ROOT,
    seq: 1,
    session: null,
    pgCache: {},
  };
}

export function children(s: TreeState, id: string): TreeNode[] {
  return Object.values(s.nodes).filter((n) => n.parent === id);
}

export function pathTo(s: TreeState, id: string): string[] {
  const p: string[] = [];
  for (let x: string | null = id; x; x = s.nodes[x]?.parent ?? null) p.unshift(x);
  return p;
}

export function depth(s: TreeState, id: string): number {
  return pathTo(s, id).length - 1;
}

export function diff(a: string, b: string): { inserted: string; removed: string } {
  let p = 0;
  const m = Math.min(a.length, b.length);
  while (p < m && a[p] === b[p]) p++;
  let q = 0;
  while (q < m - p && a[a.length - 1 - q] === b[b.length - 1 - q]) q++;
  return { removed: a.slice(p, a.length - q), inserted: b.slice(p, b.length - q) };
}

function withNodes(s: TreeState): TreeState {
  return { ...s, nodes: { ...s.nodes } };
}

function addChild(s: TreeState, kind: NodeKind, via: Via, snap: Snapshot, meta: TreeNode['meta'] = {}, now = Date.now()): TreeState {
  const parent = s.nodes[s.cur];
  const id = 'n' + s.seq;
  const n: TreeNode = {
    id, parent: s.cur, kind, via, ...snap, ...diff(parent.text, snap.text),
    meta, t: now, pg: s.pgCache[snap.text] ?? null,
  };
  return { ...s, nodes: { ...s.nodes, [id]: n }, cur: id, seq: s.seq + 1 };
}

function updateNode(s: TreeState, id: string, snap: Snapshot): TreeState {
  const n = s.nodes[id];
  const parentText = n.parent ? s.nodes[n.parent].text : '';
  return {
    ...s,
    nodes: { ...s.nodes, [id]: { ...n, ...snap, ...diff(parentText, snap.text), pg: s.pgCache[snap.text] ?? null } },
  };
}

function remove(s: TreeState, id: string): TreeState {
  const nodes = { ...s.nodes };
  delete nodes[id];
  return { ...s, nodes };
}

// Return to path[i]: remove the childless nodes after it (deepest first).
function popTo(s0: TreeState, path: string[], i: number, snap: Snapshot): TreeState {
  let s = withNodes(s0);
  for (let j = path.length - 1; j > i; j--) {
    if (children(s, path[j]).length) break;
    s = remove(s, path[j]);
  }
  const X = path[i];
  return { ...s, cur: X, session: null, nodes: { ...s.nodes, [X]: { ...s.nodes[X], caret: snap.caret } } };
}

// Deleting from the end: whoever wrote the last remaining character becomes
// current; nodes whose text is fully gone disappear (unless other branches
// hang off them).
function backspace(s0: TreeState, snap: Snapshot): TreeState {
  let s = withNodes(s0);
  const path = pathTo(s, s.cur);
  let xi = 0;
  path.forEach((id, i) => { if (snap.text.startsWith(s.nodes[id].text)) xi = i; });
  for (let i = path.length - 1; i > xi + 1; i--) {
    if (children(s, path[i]).length) break;
    s = remove(s, path[i]);
  }
  const X = path[xi], Y = path[xi + 1];
  const yFree = Y !== undefined && !!s.nodes[Y] && children(s, Y).length === 0;
  if (snap.text === s.nodes[X].text) {
    if (yFree) s = remove(s, Y);
    s = { ...s, cur: X, session: null, nodes: { ...s.nodes, [X]: { ...s.nodes[X], caret: snap.caret } } };
  } else if (yFree) {
    s = updateNode(s, Y, snap);
    s = { ...s, cur: Y, session: s.nodes[Y].kind === 'human' ? { kind: 'human', id: Y } : null };
  } else {
    s = addChild({ ...s, cur: X }, 'human', 'type', snap);
    s = { ...s, session: { kind: 'human', id: s.cur } };
  }
  return s;
}

/** Record a decision. Human edits and pick chains merge into the node they started. */
export function record(s: TreeState, kind: Src, via: Via, snap: Snapshot, meta?: TreeNode['meta']): TreeState {
  const c = s.nodes[s.cur];
  if (snap.text === c.text) {
    return { ...s, nodes: { ...s.nodes, [c.id]: { ...c, html: snap.html, caret: snap.caret } } };
  }
  if (kind === 'human') {
    // An edit that brings the document back to an earlier state on this path
    // (wherever in the text it happened) undoes the nodes after it.
    const path = pathTo(s, s.cur);
    for (let i = path.length - 2; i >= 0; i--) {
      if (s.nodes[path[i]].text === snap.text) return popTo(s, path, i, snap);
    }
    if (snap.text.length < c.text.length && c.text.startsWith(snap.text)) return backspace(s, snap);
  }
  const mergeable = s.session && s.session.kind === kind && s.session.id === s.cur && children(s, s.cur).length === 0;
  if (mergeable) return updateNode(s, s.cur, snap);
  if (kind === 'model') s = { ...s, session: null };
  const n = addChild(s, kind, via, snap, meta);
  return { ...n, session: kind === 'human' || kind === 'picked' ? { kind, id: n.cur } : null };
}

export function goto(s: TreeState, id: string): TreeState {
  if (!s.nodes[id]) return s;
  return { ...s, cur: id, session: null };
}

export function endChain(s: TreeState): TreeState {
  return s.session?.kind === 'picked' ? { ...s, session: null } : s;
}

export function attachPangram(s: TreeState, text: string, pg: PgFractions): TreeState {
  const nodes = { ...s.nodes };
  for (const n of Object.values(nodes)) if (n.text === text) nodes[n.id] = { ...n, pg };
  return { ...s, nodes, pgCache: { ...s.pgCache, [text]: pg } };
}

/** An imported chunk's own score (from the full-document scan); kept apart from the node's whole-document score. */
export function setChunkPangram(s: TreeState, id: string, chunkPg: PgFractions | null): TreeState {
  const n = s.nodes[id];
  return n ? { ...s, nodes: { ...s.nodes, [id]: { ...n, chunkPg } } } : s;
}

/** Fractions over [a, b) from a Pangram result's windows (by character overlap). */
export function fractionsIn(windows: PangramResult['windows'], a: number, b: number): PgFractions | null {
  const acc = { ai: 0, assisted: 0, human: 0 };
  for (const w of windows ?? []) {
    const o = Math.min(b, w.end_index ?? 0) - Math.max(a, w.start_index ?? 0);
    if (o <= 0) continue;
    const lbl = (w.label ?? '').toLowerCase();
    acc[lbl.includes('assisted') ? 'assisted' : lbl.includes('ai') ? 'ai' : 'human'] += o;
  }
  const tot = acc.ai + acc.assisted + acc.human;
  return tot ? { ai: acc.ai / tot, assisted: acc.assisted / tot, human: acc.human / tot } : null;
}

/** The document drifted from the tree (e.g. loaded from an older session). */
export function reconcile(s: TreeState, snap: Snapshot): TreeState {
  return snap.text === s.nodes[s.cur].text ? s : addChild(s, 'human', 'type', snap);
}
