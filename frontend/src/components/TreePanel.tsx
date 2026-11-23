import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { editor } from '../lib/editorController';
import { nodeAnchors } from '../lib/editorDom';
import { cloudColor, PG_COLORS } from '../lib/pangram';
import { WHEEL } from '../lib/wheel';
import { useApp } from '../store';
import { children, pathTo, ROOT, type TreeNode, type TreeState } from '../tree/model';
import { kick, spawnBody, step, type Body, type Link } from '../tree/physics';

const KIND_COLOR: Record<TreeNode['kind'], string> = { root: '#fbfaf7', human: '#1d1b18', model: '#b3541e', picked: '#2b6cb0', doc: '#8a857c' };
const VIA_LABEL: Record<TreeNode['via'], string> = {
  root: 'start', type: 'you typed', 'opt-tab': '⌥Tab · model wrote', 'cmd-enter': '⌘↵ · model finished sentence', pick: '⇧Tab · you picked tokens', import: 'PDF · Reducto chunk',
};

const radius = (n: TreeNode) => (n.kind === 'root' ? 6 : 5 + Math.min(7, Math.sqrt(n.inserted.length + n.removed.length) / 2.2));
const clip = (s: string, n: number) => { s = s.replace(/\n/g, '↵'); return s.length > n ? s.slice(0, n) + '…' : s; };

function links(t: TreeState): Link[] {
  // BFS so parents spawn before children
  const out: Link[] = [];
  const q: [string, number][] = [[ROOT, 0]];
  while (q.length) {
    const [id, d] = q.shift()!;
    out.push({ id, parent: t.nodes[id].parent, depth: d });
    children(t, id).sort((a, b) => a.t - b.t).forEach((c) => q.push([c.id, d + 1]));
  }
  return out;
}

interface Els { g?: SVGGElement | null; line?: SVGLineElement | null; cloud?: SVGCircleElement | null }

export function TreeView() {
  const tree = useApp((s) => s.tree);
  const svgRef = useRef<SVGSVGElement>(null);
  const worldRef = useRef<SVGGElement>(null);
  const tipRef = useRef<HTMLDivElement>(null);
  const bodies = useRef(new Map<string, Body>());
  const els = useRef(new Map<string, Els>());
  const size = useRef({ w: 320, h: 600, top: 0 });
  const cam = useRef({ x: 0, y: 0 });
  // Page-y of each node's first line in the editor; the tree shares the
  // editor's vertical coordinates and scrolls with the page.
  const targets = useRef(new Map<string, number>());
  const rootY = useRef(40);
  const hoverNode = useApp((s) => s.hoverNode);
  const docText = useApp((s) => s.docText);
  const drag = useRef<{ id: string; moved: boolean; sx: number; sy: number } | null>(null);
  const [hover, setHover] = useState<string | null>(null);
  const hoverRef = useRef<string | null>(null);
  hoverRef.current = hover;

  const linkList = useMemo(() => links(tree), [tree]);
  const live = useRef({ tree, linkList });
  live.current = { tree, linkList };
  const onPath = useMemo(() => new Set(pathTo(tree, tree.cur)), [tree]);

  const reg = (id: string, k: keyof Els) => (el: SVGElement | null) => {
    const e = els.current.get(id) ?? {};
    (e as Record<string, unknown>)[k] = el;
    els.current.set(id, e);
  };

  const measureTargets = () => {
    const ed = editor.ed;
    if (!ed) return;
    targets.current = nodeAnchors(ed);
    rootY.current = ed.getBoundingClientRect().top + scrollY - 28;
  };

  // spawn bodies for new nodes, drop removed ones, jiggle on navigation
  const prevCur = useRef(tree.cur);
  useLayoutEffect(() => {
    measureTargets();
    const b = bodies.current;
    for (const l of linkList) {
      if (b.has(l.id)) continue;
      b.set(l.id, spawnBody(l.parent ? b.get(l.parent) : undefined, size.current.w, rootY.current, targets.current.get(l.id)));
      if (l.parent) kick(b.get(l.parent), 2);
    }
    for (const id of [...b.keys()]) if (!tree.nodes[id]) { b.delete(id); els.current.delete(id); }
    if (hover && !tree.nodes[hover]) setHover(null);
    if (prevCur.current !== tree.cur) { kick(b.get(tree.cur), 1.5); prevCur.current = tree.cur; }
  }, [linkList, tree, hover, docText]);

  // text reflows (resize, fonts, edits elsewhere) — re-measure a few times a second
  useEffect(() => {
    const t = setInterval(measureTargets, 300);
    addEventListener('resize', measureTargets);
    return () => { clearInterval(t); removeEventListener('resize', measureTargets); };
  }, []);

  // hovering a node (here) or its text (in the editor) tints the text it wrote
  useEffect(() => { editor.highlightNode(hoverNode); }, [hoverNode, tree]);

  // animation loop
  useEffect(() => {
    let raf = 0;
    const frame = () => {
      const { linkList: ls } = live.current;
      const { w, h } = size.current;
      step(bodies.current, ls, w, ROOT, {
        rootY: rootY.current, targets: targets.current, pinned: drag.current?.moved ? drag.current.id : undefined,
      });
      const c = cam.current;
      c.x = 0;
      c.y = scrollY + size.current.top;   // page coords -> panel coords
      worldRef.current?.setAttribute('transform', `translate(${-c.x},${-c.y})`);
      for (const l of ls) {
        const b = bodies.current.get(l.id), e = els.current.get(l.id);
        if (!b || !e) continue;
        e.g?.setAttribute('transform', `translate(${b.x},${b.y})`);
        e.cloud?.setAttribute('cx', String(b.x)); e.cloud?.setAttribute('cy', String(b.y));
        const p = l.parent ? bodies.current.get(l.parent) : undefined;
        if (p && e.line) {
          e.line.setAttribute('x1', String(p.x)); e.line.setAttribute('y1', String(p.y));
          e.line.setAttribute('x2', String(b.x)); e.line.setAttribute('y2', String(b.y));
        }
      }
      const hv = hoverRef.current, tip = tipRef.current, hb = hv && bodies.current.get(hv);
      if (tip && hb) {
        const x = hb.x - c.x, y = hb.y - c.y, tw = tip.offsetWidth;
        tip.style.left = `${Math.max(6, Math.min(w - tw - 6, x - tw / 2))}px`;
        tip.style.top = `${y + 18 > h - 120 ? y - tip.offsetHeight - 14 : y + 18}px`;
      }
      raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(raf);
  }, []);

  // size + wheel (wheel must be non-passive to preventDefault)
  useEffect(() => {
    const svg = svgRef.current!;
    const ro = new ResizeObserver(() => {
      const r = svg.getBoundingClientRect();
      size.current = { w: r.width || 320, h: r.height || 600, top: r.top };
    });
    ro.observe(svg);
    // scrolling over the tree scrolls the document (they share coordinates)
    const wheel = (e: WheelEvent) => { e.preventDefault(); scrollBy(0, e.deltaY); };
    svg.addEventListener('wheel', wheel, { passive: false });
    return () => { ro.disconnect(); svg.removeEventListener('wheel', wheel); };
  }, []);

  const toWorld = (ev: React.PointerEvent) => {
    const r = svgRef.current!.getBoundingClientRect();
    return { x: ev.clientX - r.left + cam.current.x, y: ev.clientY - r.top + cam.current.y };
  };

  const nodes = linkList.map((l) => tree.nodes[l.id]);
  const hn = hover ? tree.nodes[hover] : null;

  return (
    <div className="tbody">
      <svg ref={svgRef}>
        <defs>
          <filter id="cloudblur" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="11" /></filter>
        </defs>
        <g ref={worldRef}>
          <g filter="url(#cloudblur)">
            {nodes.map((n) => (
              <circle key={n.id} ref={reg(n.id, 'cloud')} className="cloud"
                      r={(n.chunkPg ?? n.pg) ? 30 : 0} fill={(n.chunkPg ?? n.pg) ? cloudColor((n.chunkPg ?? n.pg)!) : 'none'}
                      style={{ opacity: (n.chunkPg ?? n.pg) ? (onPath.has(n.id) ? 0.6 : 0.4) : 0 }} />
            ))}
          </g>
          <g>
            {nodes.filter((n) => n.parent).map((n) => (
              <line key={n.id} ref={reg(n.id, 'line')} className={'te' + (onPath.has(n.id) ? '' : ' off')} stroke={KIND_COLOR[n.kind]} />
            ))}
          </g>
          <g>
            {nodes.map((n) => {
              const r = radius(n);
              return (
                <g key={n.id} ref={reg(n.id, 'g')} data-id={n.id}
                   className={'tn' + (n.id === tree.cur ? ' cur' : '') + (onPath.has(n.id) ? '' : ' off') + (n.id === hoverNode ? ' hl' : '')}
                   onPointerDown={(ev) => { ev.stopPropagation(); drag.current = { id: n.id, moved: false, sx: ev.clientX, sy: ev.clientY }; (ev.currentTarget as Element).setPointerCapture(ev.pointerId); }}
                   onPointerMove={(ev) => {
                     const d = drag.current; if (!d || d.id !== n.id) return;
                     if (Math.hypot(ev.clientX - d.sx, ev.clientY - d.sy) > 4) d.moved = true;
                     if (d.moved) { const p = toWorld(ev), b = bodies.current.get(n.id); if (b) { b.x = p.x; b.y = p.y; } }
                   }}
                   onPointerUp={() => {
                     const d = drag.current; drag.current = null;
                     if (!d || d.id !== n.id) return;
                     if (d.moved) kick(bodies.current.get(n.id), 2); else editor.gotoNode(n.id);
                   }}
                   onPointerEnter={() => { setHover(n.id); useApp.getState().set({ hoverNode: n.id }); }}
                   onPointerLeave={() => {
                     setHover((h) => (h === n.id ? null : h));
                     if (useApp.getState().hoverNode === n.id) useApp.getState().set({ hoverNode: null });
                   }}>
                  <circle className="halo" r={r + 7} />
                  <circle className="dot" r={r} fill={KIND_COLOR[n.kind]} stroke={n.kind === 'root' ? '#9a958c' : 'none'} />
                </g>
              );
            })}
          </g>
        </g>
      </svg>
      {hn && (
        <div className="ttip" ref={tipRef}>
          <div className="via" style={{ color: hn.kind === 'root' ? '#9a958c' : KIND_COLOR[hn.kind] }}>{VIA_LABEL[hn.via]}</div>
          {hn.inserted && <div className="ins">+ {clip(hn.inserted, 600)}</div>}
          {hn.removed && <div className="rem">− {clip(hn.removed, 600)}</div>}
          {hn.meta.steer?.map((c, i) => {
            const v = c.toward === c.axis[0] ? c.strength : -c.strength;   // + toward the axis' first mood
            return (
              <div className="steer" key={i}>
                <span className={v < 0 ? 'on' : ''}>{c.axis[1]}</span>
                <span className="gauge"><i style={{ left: `${50 + 50 * v}%`, background: Math.abs(v) > WHEEL.redline ? 'rgb(205,40,40)' : 'var(--ink)' }} /></span>
                <span className={v > 0 ? 'on' : ''}>{c.axis[0]}</span>
                <span className="amt">{c.strength.toFixed(2)}</span>
              </div>
            );
          })}
          {hn.meta.chunk && <div className="meta">chunk {hn.meta.chunk.i} of {hn.meta.chunk.n} · page {hn.meta.chunk.page} · {hn.meta.chunk.file}</div>}
          {hn.chunkPg && <PgLine label="this chunk" pg={hn.chunkPg} />}
          {hn.pg && <PgLine label={hn.chunkPg ? 'document up to here' : 'pangram'} pg={hn.pg} />}
          <div className="meta">{(hn.text.match(/\S+/g) ?? []).length} words · {hn.id === tree.cur ? 'you are here' : 'click to rewind here'}</div>
        </div>
      )}
    </div>
  );
}

function PgLine({ label, pg }: { label: string; pg: { ai: number; assisted: number; human: number } }) {
  return (
    <div className="meta">{label}:{' '}
      <b style={{ color: PG_COLORS.ai }}>{Math.round(100 * pg.ai)}% AI</b> ·{' '}
      <span style={{ color: '#b7791f' }}>{Math.round(100 * pg.assisted)}% assisted</span> ·{' '}
      <b style={{ color: PG_COLORS.human }}>{Math.round(100 * pg.human)}% human</b>
    </div>
  );
}

export function TreePanel() {
  return (
    <aside id="tree">
      <header>
        <div className="ttl">Decisions
          <button onClick={() => { if (confirm('Clear the whole tree and the document?')) editor.reset(); }}>clear</button>
        </div>
      </header>
      <TreeView />
      <footer>
        <div className="legend">
          <span><i style={{ background: 'var(--ink)' }} />you</span>
          <span><i style={{ background: 'var(--model)' }} />model</span>
          <span><i style={{ background: 'var(--picked)' }} />picked</span>
          <span><i style={{ background: 'var(--doc)' }} />pdf</span>
          <span title="Pangram: an imported chunk shows its own score; other nodes show the whole document at that point"><i className="cloudkey" />pangram human → AI</span>
        </div>
      </footer>
    </aside>
  );
}
