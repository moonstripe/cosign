// Low-level DOM helpers for the contenteditable editor.
//
// The document is a flat list of <span data-src="human|model|picked"
// data-node="n3"> plus a trailing <br class="tail"> (keeps a final newline
// visible; textContent ignores it). data-node links every character to the
// decision-tree node that wrote it. New text is inserted as data-node="pending"
// and tagged once the tree has decided which node it belongs to.
// React never renders inside the editor — typing stays native.

export const PENDING = 'pending';

import type { Snapshot, Src } from '../types';

export function ensureTail(ed: HTMLElement) {
  let tail = ed.querySelector('br.tail');
  if (!tail) {
    tail = document.createElement('br');
    tail.className = 'tail';
    ed.appendChild(tail);
  } else if (tail !== ed.lastChild) ed.appendChild(tail);
}

export const docText = (ed: HTMLElement) => ed.textContent ?? '';

export function caretRange(ed: HTMLElement): Range {
  const sel = getSelection();
  if (!sel || !sel.rangeCount || !ed.contains(sel.anchorNode)) {
    const r = document.createRange();
    const tail = ed.querySelector('br.tail');
    if (tail) r.setStartBefore(tail); else r.setStart(ed, ed.childNodes.length);
    r.collapse(true);
    return r;
  }
  return sel.getRangeAt(0);
}

/** For model actions: never replace a selection — act at its end instead. */
export function collapsedCaret(ed: HTMLElement): Range {
  const r = caretRange(ed).cloneRange();
  if (!r.collapsed) {
    r.collapse(false);
    selectRange(r);
  }
  return r;
}

export function selectRange(r: Range) {
  const sel = getSelection();
  if (!sel) return;
  sel.removeAllRanges();
  sel.addRange(r);
}

export function prefixBefore(ed: HTMLElement, range: Range): string {
  const r = document.createRange();
  r.setStart(ed, 0);
  r.setEnd(range.startContainer, range.startOffset);
  return r.toString();
}

export function placeCaret(node: Node, offset: number) {
  const r = document.createRange();
  r.setStart(node, offset);
  r.collapse(true);
  selectRange(r);
}

/** Insert text at the range as a new pending span attributed to `src`. */
/** A non-text spinner at the caret (not part of the document text). Remove it when done. */
export function insertSpinner(range: Range): HTMLElement {
  const el = document.createElement('span');
  el.className = 'spin';
  el.contentEditable = 'false';
  el.setAttribute('aria-label', 'waiting for the model');
  const r = range.cloneRange();
  r.collapse(true);
  r.insertNode(el);
  return el;
}

export function insertAt(range: Range, text: string, src: Src, cls?: string): { node: Text; span: HTMLElement } {
  if (!range.collapsed) range.deleteContents();
  const node = range.startContainer, off = range.startOffset;
  const span = document.createElement('span');
  span.dataset.src = src;
  span.dataset.node = PENDING;
  if (cls) span.className = cls;
  const t = document.createTextNode(text);
  span.appendChild(t);
  if (node.nodeType === Node.TEXT_NODE) {
    const after = (node as Text).splitText(off);
    const outer = node.parentElement!;
    if (outer.tagName === 'SPAN' && outer.dataset.src) {
      const right = outer.cloneNode(false) as HTMLElement;
      for (let n: ChildNode | null = after; n; ) { const nx: ChildNode | null = n.nextSibling; right.appendChild(n); n = nx; }
      outer.after(right);
      outer.after(span);
    } else (node as Text).after(span);
  } else if ((node as Element).hasAttribute?.('contenteditable')) {   // the editor root itself
    node.insertBefore(span, node.childNodes[off] ?? null);
  } else {
    const outer = (node as Element).closest('span[data-src]') ?? (node as Element);
    if (off === 0) outer.before(span); else outer.after(span);
  }
  placeCaret(t, text.length);
  return { node: t, span };
}

/** Drop empties, merge adjacent same-source spans, wrap stray nodes. */
export function normalize(ed: HTMLElement) {
  for (const n of [...ed.childNodes]) {
    if (n.nodeType === Node.TEXT_NODE) {
      if (!(n as Text).data) { n.remove(); continue; }
      const s = document.createElement('span');
      s.dataset.src = 'human';
      n.before(s);
      s.appendChild(n);
    } else if (n.nodeName === 'BR' && !(n as Element).classList.contains('tail')) {
      const s = document.createElement('span');
      s.dataset.src = 'human';
      s.textContent = '\n';
      n.replaceWith(s);
    } else if (n.nodeName === 'SPAN' && !(n as HTMLElement).dataset.src) (n as HTMLElement).dataset.src = 'human';
  }
  for (const s of ed.querySelectorAll('span[data-src]')) if (!s.textContent) s.remove();
  const sel = getSelection();
  const keep = sel && sel.rangeCount ? sel.getRangeAt(0).cloneRange() : null;
  let prev: HTMLElement | null = null;
  for (const s of [...ed.children] as HTMLElement[]) {
    if (s.nodeName !== 'SPAN') { prev = null; continue; }
    if (prev && prev.dataset.src === s.dataset.src && prev.dataset.node === s.dataset.node
        && !prev.classList.contains('live') && !s.classList.contains('live')) {
      while (s.firstChild) prev.appendChild(s.firstChild);
      s.remove();
    } else prev = s;
  }
  if (keep) { try { selectRange(keep); } catch { /* range went stale */ } }
  ensureTail(ed);
}

/** Map [start, end) character offsets to a DOM Range. */
export function textRange(ed: HTMLElement, start: number, end: number): Range | null {
  const w = document.createTreeWalker(ed, NodeFilter.SHOW_TEXT);
  const r = document.createRange();
  let pos = 0, started = false;
  for (let n = w.nextNode() as Text | null; n; n = w.nextNode() as Text | null) {
    const len = n.data.length;
    if (!started && start <= pos + len) { r.setStart(n, start - pos); started = true; }
    if (started && end <= pos + len) { r.setEnd(n, end - pos); return r; }
    pos += len;
  }
  if (!started) return null;
  r.setEndAfter(ed.lastChild!);
  return r;
}

export function authorship(ed: HTMLElement): Record<Src, number> {
  const c: Record<Src, number> = { human: 0, model: 0, picked: 0, doc: 0 };
  for (const s of ed.querySelectorAll<HTMLElement>('span[data-src]')) c[s.dataset.src as Src] += s.textContent?.length ?? 0;
  return c;
}

export function snapshot(ed: HTMLElement): Snapshot {
  const c = ed.cloneNode(true) as HTMLElement;
  c.querySelectorAll('.live').forEach((s) => s.classList.remove('live'));
  const sel = getSelection();
  const caret = sel && sel.rangeCount && ed.contains(sel.anchorNode) ? prefixBefore(ed, sel.getRangeAt(0)).length : docText(ed).length;
  return { html: c.innerHTML, text: docText(ed), caret };
}

/** Viewport rect of the caret (collapsed ranges often measure as zero). */
export function caretRect(ed: HTMLElement, range: Range): { left: number; bottom: number } {
  const b = range.getBoundingClientRect();
  if (b.height) return { left: b.left, bottom: b.bottom };
  if (range.startContainer.nodeType === Node.TEXT_NODE && range.startOffset > 0) {
    const r = document.createRange();
    r.setStart(range.startContainer, range.startOffset - 1);
    r.setEnd(range.startContainer, range.startOffset);
    const p = r.getBoundingClientRect();
    if (p.height) return { left: p.right, bottom: p.bottom };
  }
  const e = ed.getBoundingClientRect();
  return { left: e.left, bottom: e.top + 30 };
}

/** Give pending (and legacy untagged) spans their tree node. */
export function tagPending(ed: HTMLElement, nodeId: string) {
  for (const s of ed.querySelectorAll<HTMLElement>(`span[data-node="${PENDING}"], span[data-src]:not([data-node])`)) {
    if (!s.classList.contains('live')) s.dataset.node = nodeId;
  }
}

/** Page-y (document coordinates) of the first line each node wrote. */
export function nodeAnchors(ed: HTMLElement): Map<string, number> {
  const out = new Map<string, number>();
  for (const s of ed.querySelectorAll<HTMLElement>('span[data-node]')) {
    const id = s.dataset.node!;
    if (out.has(id) || id === PENDING) continue;
    const y = firstVisibleY(s);
    if (y !== null) out.set(id, y + scrollY);
  }
  return out;
}

/** Viewport-y of the middle of a span's first non-whitespace character
 *  (spans often start with "\n\n", whose first box is the blank line above). */
function firstVisibleY(span: HTMLElement): number | null {
  const w = document.createTreeWalker(span, NodeFilter.SHOW_TEXT);
  for (let t = w.nextNode() as Text | null; t; t = w.nextNode() as Text | null) {
    const i = t.data.search(/\S/);
    if (i < 0) continue;
    const r = document.createRange();
    r.setStart(t, i);
    r.setEnd(t, i + 1);
    const b = r.getBoundingClientRect();
    if (b.height) return b.top + b.height / 2;
  }
  const b = [...span.getClientRects()].find((r) => r.height > 0);   // whitespace-only span
  return b ? b.top + b.height / 2 : null;
}

/** Ranges covering everything a node wrote (for hover highlighting). */
export function nodeRanges(ed: HTMLElement, nodeId: string): Range[] {
  return [...ed.querySelectorAll<HTMLElement>(`span[data-node="${nodeId}"]`)].map((s) => {
    const r = document.createRange();
    r.selectNodeContents(s);
    return r;
  });
}

export function nodeAt(target: EventTarget | null): string | null {
  const el = (target as Element | null)?.closest?.('span[data-node]') as HTMLElement | null;
  const id = el?.dataset.node;
  return id && id !== PENDING ? id : null;
}
