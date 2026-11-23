// Imperative controller for the contenteditable editor.
//
// React owns the chrome; this class owns the editor DOM (typing must stay
// native and fast) and bridges events into the zustand store.

import { clearSaved, saveDoc, savedDoc, useApp, type Action } from '../store';
import { attachPangram, endChain, fractionsIn, goto, initTree, pathTo, reconcile, record, setChunkPangram, type SteerChoice, type TreeNode, type TreeState } from '../tree/model';
import type { PangramResult } from '../types';
import type { Signpost, Src, Via } from '../types';
import * as dom from './editorDom';
import { detect, MIN_WORDS, wordCount } from './pangram';
import { engine } from './socket';
import { pickFromDelta, toServer, wheelSignposts } from './wheel';

const DETECT_IDLE_MS = 2500;
const HOLD_MS = 180;   // hold an action key this long to open the radial wheel


class EditorController {
  ed: HTMLElement | null = null;
  private chaining = false;
  private pgTimer: number | undefined;
  private pgInflight: symbol | null = null;
  private cleanup: (() => void) | null = null;
  private lastCaret: Range | null = null;   // so focus can return to where you were
  private hold: { action: Action; key: string; mod: string; timer: number; opened: boolean } | null = null;
  private mouse: { x: number; y: number } | null = null;

  // ------------------------------------------------------------ lifecycle
  attach(ed: HTMLElement) {
    this.ed = ed;
    ed.innerHTML = savedDoc();
    ed.querySelectorAll('.live').forEach((s) => s.classList.remove('live'));
    this.changed();
    // older documents may not match the tree / have untagged spans
    const t1 = reconcile(this.s.tree, dom.snapshot(ed));
    dom.tagPending(ed, t1.cur);
    this.changed();
    this.s.set({ tree: record(t1, 'human', 'type', dom.snapshot(ed)) });

    const before = (e: InputEvent) => this.onBeforeInput(e);
    const input = () => this.commit('human', 'type');   // native deletions
    const hover = (e: MouseEvent) => {
      const id = dom.nodeAt(e.target);
      if (id !== this.s.hoverNode) this.s.set({ hoverNode: id });
    };
    const leave = () => { if (this.s.hoverNode) this.s.set({ hoverNode: null }); };
    const key = (e: KeyboardEvent) => this.onKeyDown(e);
    const selchange = () => {
      const sel = getSelection();
      if (sel?.rangeCount && ed.contains(sel.anchorNode)) this.lastCaret = sel.getRangeAt(0).cloneRange();
    };
    const keyup = (e: KeyboardEvent) => {
      const h = this.hold;
      if (h && (e.key === h.key || e.key === h.mod)) { e.preventDefault(); this.releaseHold(); }
    };
    const move = (e: MouseEvent) => {
      this.mouse = { x: e.clientX, y: e.clientY };
      const w = this.s.wheel;
      if (w) this.s.set({ wheel: { ...w, pick: pickFromDelta(e.clientX - w.x, e.clientY - w.y) } });
    };
    const blur = () => this.cancelHold();
    const down = (e: MouseEvent) => {
      if (!(e.target as Element).closest?.('.popover') && (this.chaining || this.s.popover)) this.endChain();
    };
    ed.addEventListener('beforeinput', before);
    ed.addEventListener('input', input);
    ed.addEventListener('keydown', key);
    ed.addEventListener('mousemove', hover);
    ed.addEventListener('mouseleave', leave);
    document.addEventListener('mousedown', down);
    document.addEventListener('selectionchange', selchange);
    document.addEventListener('keyup', keyup, true);
    document.addEventListener('mousemove', move);
    window.addEventListener('blur', blur);
    this.cleanup = () => {
      document.removeEventListener('keyup', keyup, true);
      document.removeEventListener('mousemove', move);
      window.removeEventListener('blur', blur);
      document.removeEventListener('selectionchange', selchange);
      ed.removeEventListener('beforeinput', before);
      ed.removeEventListener('input', input);
      ed.removeEventListener('keydown', key);
      ed.removeEventListener('mousemove', hover);
      ed.removeEventListener('mouseleave', leave);
      document.removeEventListener('mousedown', down);
    };
    ed.focus();
    this.caretToEnd();
  }

  private caretToEnd() {
    const tail = this.ed?.querySelector('br.tail');
    if (!tail) return;
    const r = document.createRange();
    r.setStartBefore(tail);
    r.collapse(true);
    dom.selectRange(r);
  }

  /** Focus the editor, restoring the caret to where it last was. */
  focus() {
    const ed = this.ed;
    if (!ed) return;
    ed.focus();
    const r = this.lastCaret;
    if (r && ed.contains(r.startContainer)) dom.selectRange(r);
    else this.caretToEnd();
  }

  detach() { this.cleanup?.(); this.cleanup = null; this.ed = null; }

  private get s() { return useApp.getState(); }

  /** After any change to the editor DOM. */
  private changed() {
    const ed = this.ed!;
    dom.normalize(ed);
    saveDoc(ed.innerHTML);
    const text = dom.docText(ed);
    this.s.set({ authorship: dom.authorship(ed), docText: text });
    if (text !== this.s.pangram.text) this.scheduleDetect();
    else this.drawHighlights();
  }

  /**
   * Record a decision in the tree, then tag the text it inserted with the
   * node that now owns it, and store the tagged snapshot on that node.
   */
  private commit(kind: Src, via: Via, meta?: TreeNode['meta']) {
    const ed = this.ed!;
    const t1 = record(this.s.tree, kind, via, dom.snapshot(ed), meta);
    dom.tagPending(ed, t1.cur);
    this.changed();
    this.s.set({ tree: record(t1, kind, via, dom.snapshot(ed), meta) });   // same text: refreshes html only
  }

  // ------------------------------------------------------------ typing
  private onBeforeInput(e: InputEvent) {
    if (this.s.generating) this.stop();
    this.closePop();
    const t = e.inputType;
    let text: string | null | undefined = null;
    if (t === 'insertText' || t === 'insertReplacementText') text = e.data ?? e.dataTransfer?.getData('text/plain');
    else if (t === 'insertParagraph' || t === 'insertLineBreak') text = '\n';
    else if (t === 'insertFromPaste' || t === 'insertFromDrop') text = e.dataTransfer?.getData('text/plain');
    else if (t.startsWith('format')) { e.preventDefault(); return; }
    if (text == null) return;   // deletions: let the browser do it
    e.preventDefault();
    dom.insertAt(dom.caretRange(this.ed!), text, 'human');
    this.commit('human', 'type');
  }

  // ------------------------------------------------------------ action keys
  // Tap ⌥Tab / ⇧Tab / ⌘↵ to act with your signposts; hold to open the radial
  // wheel, aim with the mouse, release to act with that steering added.

  private actionOf(e: KeyboardEvent): { action: Action; key: string; mod: string } | null {
    if (e.key === 'Tab' && e.altKey) return { action: 'write', key: 'Tab', mod: 'Alt' };
    if (e.key === 'Tab' && e.shiftKey) return { action: 'choose', key: 'Tab', mod: 'Shift' };
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) return { action: 'sentence', key: 'Enter', mod: e.metaKey ? 'Meta' : 'Control' };
    return null;
  }

  private openWheel() {
    const h = this.hold;
    if (!h || !this.ed || this.s.generating) return;
    h.opened = true;
    this.closePop();
    let c = this.mouse;
    if (!c) { const r = dom.caretRect(this.ed, dom.caretRange(this.ed)); c = { x: r.left, y: r.bottom }; }
    // keep the whole wheel on screen
    const m = 170;
    const x = Math.max(m, Math.min(innerWidth - m, c.x)), y = Math.max(m, Math.min(innerHeight - m, c.y));
    this.s.set({ wheel: { action: h.action, x, y, pick: pickFromDelta((this.mouse?.x ?? x) - x, (this.mouse?.y ?? y) - y) } });
  }

  private releaseHold() {
    const h = this.hold;
    if (!h) return;
    clearTimeout(h.timer);
    this.hold = null;
    let steer: Signpost[] | null = null;   // null = tapped, no wheel
    if (h.opened) {
      const w = this.s.wheel;
      steer = w ? wheelSignposts(this.s.axes, w.pick, this.s.dialAxes) : [];
      this.s.set({ wheel: null });
    }
    this.run(h.action, steer);
  }

  cancelHold() {
    if (this.hold) clearTimeout(this.hold.timer);
    this.hold = null;
    if (this.s.wheel) this.s.set({ wheel: null });
  }

  private run(action: Action, steer: Signpost[] | null) {
    if (action === 'choose') {
      // a wheel pick steers the whole pick chain; a tap keeps the chain's steering
      if (steer !== null) this.s.set({ oneShot: steer });
      else if (!this.chaining) this.s.set({ oneShot: [] });
      this.chaining = true;
      this.showCandidates();
      return;
    }
    this.chaining = false;
    this.s.set({ oneShot: steer ?? [] });
    if (action === 'write') this.generate({ n: this.s.budget });
    else this.generate({ until: 'sentence' });
  }

  private endChain() {
    this.chaining = false;
    this.closePop();
    this.s.setTree(endChain);
    if (this.s.oneShot.length) this.s.set({ oneShot: [] });
  }

  private onKeyDown(e: KeyboardEvent) {
    if (this.s.wheel && e.key === 'Escape') { e.preventDefault(); this.cancelHold(); return; }
    const act = this.actionOf(e);
    if (act) {
      e.preventDefault();
      if (e.repeat || this.hold) return;
      this.hold = { ...act, opened: false, timer: window.setTimeout(() => this.openWheel(), HOLD_MS) };
      return;
    }
    const pop = this.s.popover;
    if (pop) {
      if (/^[1-9]$/.test(e.key)) { e.preventDefault(); this.pick(+e.key - 1); return; }
      if (e.key === 'ArrowDown') { e.preventDefault(); this.s.set({ popover: { ...pop, sel: (pop.sel + 1) % pop.items.length } }); return; }
      if (e.key === 'ArrowUp') { e.preventDefault(); this.s.set({ popover: { ...pop, sel: (pop.sel - 1 + pop.items.length) % pop.items.length } }); return; }
      if (e.key === 'Enter') { e.preventDefault(); this.pick(pop.sel); return; }
      if (e.key === 'Escape') { e.preventDefault(); this.endChain(); return; }
    }
    if (e.key === 'Tab') {   // plain Tab = indent, typed by you
      e.preventDefault(); this.closePop(); this.chaining = false;
      dom.insertAt(dom.caretRange(this.ed!), '\t', 'human');
      this.commit('human', 'type');
      return;
    }
    if (e.key === 'Escape') { this.stop(); this.endChain(); }
  }

  // ------------------------------------------------------------ model
  /** Tree metadata: which dial axes steered this decision, and where on them. Empty for plain generation. */
  private steerMeta(): TreeNode['meta'] {
    const steer = this.steering().filter((s) => s.strength > 0).map((s) => {
      const axis = this.s.axes.find(([a, b]) => (a === s.concept && b === s.anti) || (a === s.anti && b === s.concept));
      return { axis: axis ?? [s.concept, s.anti ?? ''], toward: s.concept, strength: s.strength } as SteerChoice;
    });
    return steer.length ? { steer } : {};
  }

  private steering(): Signpost[] { return [...this.s.signposts, ...this.s.oneShot]; }

  private req(prefix: string) {
    // UI strengths are 0–1; the server's scale is 0–10
    return { prefix, signposts: this.steering().map((s) => ({ ...s, strength: toServer(s.strength) })), temp: this.s.temp };
  }

  generate(opts: { n?: number; until?: 'sentence' }) {
    if (this.s.generating || !engine.ready || !this.ed) return;
    this.closePop();
    const range = dom.collapsedCaret(this.ed);
    const prefix = dom.prefixBefore(this.ed, range);
    const via: Via = opts.until ? 'cmd-enter' : 'opt-tab';
    const meta = this.steerMeta();
    const { node, span } = dom.insertAt(range, '', 'model', 'live waiting');   // spinner until the first token
    this.s.set({ generating: true });
    engine.generate({ ...this.req(prefix), ...opts }, {
      token: (t) => { span.classList.remove('waiting'); node.appendData(t); dom.placeCaret(node, node.length); },
      done: (signposts) => {
        span.classList.remove('live', 'waiting');
        // chips keep their remaining ttl; wheel steering was for this action only
        const left = new Map(signposts.map((s) => [s.id, s]));
        this.s.set({
          generating: false,
          oneShot: [],
          signposts: this.s.signposts.filter((c) => left.has(c.id)).map((c) => ({ ...c, ttl: left.get(c.id)!.ttl })),
        });
        if (node.isConnected) dom.placeCaret(node, node.length);
        this.commit('model', via, meta);
      },
    });
  }

  stop() { if (this.s.generating) engine.stop(); }

  async showCandidates() {
    if (this.s.generating || !engine.ready || !this.ed) return;
    const range = dom.collapsedCaret(this.ed);
    const req = { ...this.req(dom.prefixBefore(this.ed, range)), k: 9 };
    const spin = dom.insertSpinner(range);          // at the caret until the candidates arrive
    let items;
    try { items = await engine.candidates(req); } finally { spin.remove(); }
    const r = dom.caretRect(this.ed, dom.caretRange(this.ed));
    this.s.set({ popover: { items, x: r.left + scrollX, y: r.bottom + scrollY + 6, sel: 0 } });
  }

  pick(i: number) {
    const c = this.s.popover?.items[i];
    if (!c || !this.ed) return;
    dom.insertAt(dom.caretRange(this.ed), c.text, 'picked');
    this.commit('picked', 'pick', this.steerMeta());
    this.closePop();
    if (this.chaining) this.showCandidates();
  }

  closePop() { if (this.s.popover) this.s.set({ popover: null }); }

  // ------------------------------------------------------------ tree
  gotoNode(id: string) {
    const ed = this.ed;
    const n = this.s.tree.nodes[id];
    if (!ed || !n || this.s.generating) return;
    if (id !== this.s.tree.cur) {
      this.closePop(); this.chaining = false;
      ed.innerHTML = n.html;
      this.changed();
      ed.focus({ preventScroll: true });
      this.lastCaret = null;
      const r = dom.textRange(ed, n.caret, n.caret);
      if (r) { r.collapse(true); dom.selectRange(r); }
      this.s.setTree((t) => goto(t, id));
    }
    this.scrollToNode(id);
  }

  /**
   * Scroll the page so the node's text sits about a third of the way down the
   * screen. The tree shares the page's coordinates, so it follows.
   */
  scrollToNode(id: string) {
    const ed = this.ed;
    if (!ed) return;
    let y = dom.nodeAnchors(ed).get(id);
    if (y === undefined) {   // wrote nothing visible (e.g. only deleted): use the caret
      const c = dom.caretRect(ed, dom.caretRange(ed));
      y = c.bottom + scrollY;
    }
    scrollTo({ top: Math.max(0, y - innerHeight * 0.33), behavior: 'smooth' });
  }

  /** Replace the editor with a saved document (html with authorship spans + its tree + last Pangram result). */
  loadDocument(html: string, tree: TreeState | null, pangram: PangramResult | null, text?: string) {
    if (!this.ed || this.s.generating) return;
    clearTimeout(this.pgTimer);
    this.closePop();
    this.ed.innerHTML = html;
    this.ed.querySelectorAll('.live, .waiting, .spin').forEach((s) => (s.classList.contains('spin') ? s.remove() : s.classList.remove('live', 'waiting')));
    const docText = dom.docText(this.ed);
    this.s.setPangram({ text: pangram && (text ?? docText) === docText ? docText : null, result: pangram && (text ?? docText) === docText ? pangram : null, busy: false, error: null });
    const t0 = tree && tree.nodes && tree.nodes[tree.cur] ? tree : initTree();
    const t1 = reconcile(t0, dom.snapshot(this.ed));
    dom.tagPending(this.ed, t1.cur);
    this.s.set({ tree: t1 });
    this.changed();
    this.ed.focus();
  }

  /**
   * Load a parsed PDF as a linear chain: one 'doc' node per chunk, no branches.
   * Pangram then scores the whole document once; its windows (character offsets)
   * are split back over the chunks so every node gets its own human -> AI colour.
   */
  async importDoc(chunks: { text: string; page: number }[], file: string) {
    if (!this.ed || this.s.generating || !chunks.length) return;
    clearTimeout(this.pgTimer);
    clearSaved();
    this.ed.innerHTML = '';
    this.s.set({ tree: initTree() });
    this.s.setPangram({ text: null, result: null, error: null, busy: false });
    const n = chunks.length;
    chunks.forEach((c, i) => {
      const r = document.createRange();
      r.selectNodeContents(this.ed!);
      r.collapse(false);
      dom.insertAt(r, (i ? '\n\n' : '') + c.text, 'doc');
      this.commit('doc', 'import', { chunk: { i: i + 1, n, page: c.page, file } });
    });
    this.ed.focus();
    await this.runDetect(true);
    this.scoreChunks();
  }

  /** Per-chunk scores from the full-document windows; the scan is kept so prefixes can reuse it. */
  private scoreChunks() {
    const res: PangramResult | null = this.s.pangram.result;
    const text = this.s.pangram.text;
    if (!res?.windows?.length || !text) return;
    this.s.setTree((t) => {
      let prev = 0;
      for (const id of pathTo(t, t.cur)) {
        const node = t.nodes[id];
        if (node.via !== 'import') continue;
        const a = prev, b = node.text.length;
        prev = b;
        t = setChunkPangram(t, id, fractionsIn(res.windows, a, b));
      }
      return { ...t, importScan: { text, result: res } };
    });
  }

  /** The import scan cut off at `len` characters: what it said about this prefix, in context. */
  private scanPrefix(text: string): PangramResult | null {
    const scan = this.s.tree.importScan;
    if (!scan || !text || !scan.text.startsWith(text)) return null;
    const len = text.length;
    const windows = (scan.result.windows ?? [])
      .filter((w) => (w.start_index ?? 0) < len)
      .map((w) => ({ ...w, end_index: Math.min(w.end_index ?? len, len) }));
    const f = fractionsIn(windows, 0, len) ?? { ai: 0, assisted: 0, human: 1 };
    const verdict = f.ai >= 0.5 ? 'AI Detected' : f.ai + f.assisted >= 0.5 ? 'AI-Assisted' : 'Human Written';
    return {
      ...scan.result, windows,
      fraction_ai: f.ai, fraction_ai_assisted: f.assisted, fraction_human: f.human,
      headline: `${verdict} · import scan`,
      prediction: 'From the full-document Pangram scan made at import (no new check: this text is part of it).',
    };
  }

  reset() {
    if (!this.ed || this.s.generating) return;
    clearSaved();
    this.ed.innerHTML = '';
    clearTimeout(this.pgTimer);
    this.pgInflight = null;
    this.clearHighlights();
    this.s.setPangram({ text: null, result: null, error: null, busy: false });
    this.s.set({ tree: initTree() });
    this.changed();
    this.ed.focus();
  }

  // ------------------------------------------------------------ pangram
  private scheduleDetect() {
    clearTimeout(this.pgTimer);
    this.clearHighlights();
    if (this.ed && this.scanPrefix(dom.docText(this.ed))) { void this.runDetect(false); return; }   // free + instant
    this.pgTimer = window.setTimeout(() => this.runDetect(false), DETECT_IDLE_MS);
  }

  async runDetect(force: boolean) {
    clearTimeout(this.pgTimer);
    if (!this.ed) return;
    if (this.s.generating) { this.pgTimer = window.setTimeout(() => this.runDetect(force), 1000); return; }
    const text = dom.docText(this.ed);
    if (!force && text === this.s.pangram.text) return;
    // back along an imported chain: reuse the import scan so node colours and highlights agree
    const fromScan = this.scanPrefix(text);
    if (fromScan) {
      this.pgInflight = null;
      this.s.setPangram({ text, result: fromScan, busy: false, error: null });
      this.s.setTree((t) => attachPangram(t, text, { ai: fromScan.fraction_ai, assisted: fromScan.fraction_ai_assisted, human: fromScan.fraction_human }));
      this.drawHighlights();
      return;
    }
    if (wordCount(text) < MIN_WORDS) {
      this.s.setPangram({ text: null, result: null, error: null, busy: false });
      return;
    }
    const mine = (this.pgInflight = Symbol());
    this.s.setPangram({ busy: true, error: null });
    try {
      const result = await detect(text);
      if (this.pgInflight !== mine) return;
      this.s.setPangram({ text, result, busy: false });
      this.s.setTree((t) => attachPangram(t, text, {
        ai: result.fraction_ai ?? 0, assisted: result.fraction_ai_assisted ?? 0, human: result.fraction_human ?? 0,
      }));
      this.drawHighlights();
    } catch (err) {
      if (this.pgInflight === mine) this.s.setPangram({ busy: false, error: (err as Error).message });
    }
  }

  /** Tint everything a tree node wrote (hovering the node or its text). */
  highlightNode(id: string | null) {
    if (!CSS.highlights) return;
    CSS.highlights.delete('node-hover');
    if (!id || !this.ed) return;
    CSS.highlights.set('node-hover', new Highlight(...dom.nodeRanges(this.ed, id)));
  }

  clearHighlights() {
    CSS.highlights?.delete('pg-ai');
    CSS.highlights?.delete('pg-assisted');
  }

  drawHighlights() {
    this.clearHighlights();
    const { pangram, showSegments } = this.s;
    if (!this.ed || !CSS.highlights || !pangram.result || !showSegments || dom.docText(this.ed) !== pangram.text) return;
    const ai = new Highlight(), as = new Highlight();
    for (const w of pangram.result.windows ?? []) {
      const lbl = (w.label ?? '').toLowerCase();
      const target = lbl.includes('assisted') ? as : lbl.includes('ai') ? ai : null;
      const r = target && dom.textRange(this.ed, w.start_index, w.end_index);
      if (target && r) target.add(r);
    }
    CSS.highlights.set('pg-ai', ai);
    CSS.highlights.set('pg-assisted', as);
  }
}

export const editor = new EditorController();
