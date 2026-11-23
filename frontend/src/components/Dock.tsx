import { useEffect } from 'react';
import { editor } from '../lib/editorController';
import { PG_COLORS } from '../lib/pangram';
import { useApp, type AppState } from '../store';
import { children } from '../tree/model';

type Tab = AppState['tab'];
const TABS: { id: Tab; label: string; title: string }[] = [
  { id: 'model', label: 'Model', title: 'which model writes, how much and how freely' },
  { id: 'view', label: 'View', title: 'what the page shows' },
];

// Only LFM3-1.7B is served today; the others are placeholders for the picker.
const MODELS = [
  { id: 'LiquidAI/LFM3-1.7B', label: 'LFM3-1.7B', ok: true },
  { id: 'LiquidAI/LFM3-8B', label: 'LFM3-8B (soon)', ok: false },
  { id: 'NousResearch/Hermes-4-14B', label: 'Hermes-4-14B (soon)', ok: false },
];

// ------------------------------------------------------------------ meters

/** A stacked bar; each segment carries its own label in white when there's room. */
function Bar({ parts }: { parts: { color: string; frac: number; label: string }[] }) {
  return (
    <div className="bar">
      {parts.filter((p) => p.frac > 0.001).map((p, i) => {
        const pc = `${Math.round(100 * p.frac)}%`;
        const text = p.frac >= 0.22 ? `${p.label} ${pc}` : p.frac >= 0.09 ? pc : '';
        return <div key={i} style={{ background: p.color, width: `${100 * p.frac}%` }} title={`${p.label} ${pc}`}>{text}</div>;
      })}
    </div>
  );
}

function Meters() {
  const a = useApp((s) => s.authorship);
  const pg = useApp((s) => s.pangram);
  const docText = useApp((s) => s.docText);
  const tree = useApp((s) => s.tree);
  const kids = Object.values(tree.nodes).filter((n) => n.kind !== 'root');
  const branches = Object.values(tree.nodes).filter((n) => children(tree, n.id).length === 0).length;
  const c = { human: 0, model: 0, picked: 0, doc: 0 };
  kids.forEach((n) => { c[n.kind as keyof typeof c]++; });
  const tot = a.human + a.model + a.picked + a.doc || 1;
  const r = pg.result;
  const stale = !!r && docText !== pg.text;


  return (
    <div className="stats">
      <div className="statrow counts">
        <span className="lbl">decisions</span>
        <span>{kids.length} total · {branches} branch{branches === 1 ? '' : 'es'} · you {c.human} · model {c.model} · picks {c.picked}{c.doc ? ` · pdf ${c.doc}` : ''}</span>
      </div>
      <div className="statrow">
        <span className="lbl">authorship</span>
        <Bar parts={tot > 1 ? [{ color: 'var(--ink)', frac: a.human / tot, label: 'you' }, { color: 'var(--picked)', frac: a.picked / tot, label: 'picked' }, { color: 'var(--model)', frac: a.model / tot, label: 'model' }, { color: 'var(--doc)', frac: a.doc / tot, label: 'pdf' }] : []} />
      </div>
      <div className={'statrow pgrow' + (stale || pg.error ? ' stale' : '') + (pg.busy ? ' busy' : '')}
           title={(r?.prediction ?? 'Pangram AI detection') + ' (click to check now)'} onClick={() => editor.runDetect(true)}>
        <span className="lbl">pangram</span>
        <Bar parts={r ? [{ color: PG_COLORS.human, frac: r.fraction_human, label: 'human' }, { color: PG_COLORS.assisted, frac: r.fraction_ai_assisted, label: 'assisted' }, { color: PG_COLORS.ai, frac: r.fraction_ai, label: 'AI' }] : []} />
        <button className="pgrefresh" aria-label="re-check with Pangram"
                title={pg.busy ? 'checking…' : r ? `${r.headline ?? r.prediction_short ?? ''} · re-check` : 'check with Pangram (20+ words)'}
                onClick={(e) => { e.stopPropagation(); editor.runDetect(true); }}>
          <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true">
            <path d="M13.2 8a5.2 5.2 0 1 1-1.52-3.68" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="square" />
            <path d="M13.6 1.8v3.4h-3.4" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="square" />
          </svg>
        </button>
        {pg.error && <span className="pgerr" title={pg.error}>{pg.error}</span>}
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ dock

function Panel({ tab }: { tab: Tab }) {
  const { budget, temp, showColors, showSegments, modelId, conn, generating, set } = useApp();
  if (tab === 'model') return (
    <div className="ctls">
      <label className="ctl"><span className="sub">model</span>
        <select value={modelId} onChange={(e) => set({ modelId: e.target.value })}>
          {MODELS.map((m) => <option key={m.id} value={m.id} disabled={!m.ok}>{m.label}</option>)}
        </select>
        <span className="conn">{generating ? 'writing…' : conn.connected ? `on ${conn.device}` : 'connecting…'}</span>
      </label>
      <label className="ctl"><span className="sub">⌥Tab writes</span>
        <input type="range" min={1} max={64} value={budget} onChange={(e) => set({ budget: +e.target.value })} /> <span className="val">{budget} tokens</span></label>
      <label className="ctl"><span className="sub">temperature</span>
        <input type="range" min={0} max={1.5} step={0.05} value={temp} onChange={(e) => set({ temp: +e.target.value })} /> <span className="val">{temp.toFixed(2)}</span></label>
    </div>
  );
  return (
    <div className="ctls">
      <label className="ctl"><input type="checkbox" checked={showColors} onChange={(e) => set({ showColors: e.target.checked })} /> colour text by author (you / picked / model)</label>
      <label className="ctl"><input type="checkbox" checked={showSegments} onChange={(e) => set({ showSegments: e.target.checked })} /> highlight Pangram's AI segments</label>
    </div>
  );
}

export function Dock() {
  const { tab, showColors, showSegments, set } = useApp();
  useEffect(() => { document.body.classList.toggle('prov', showColors); }, [showColors]);
  useEffect(() => { editor.drawHighlights(); }, [showSegments]);

  return (
    <div id="dock"><div className="inner">
      <nav className="vtabs" role="tablist">
        {TABS.map((t) => (
          <button key={t.id} role="tab" aria-selected={tab === t.id} title={t.title}
                  className={tab === t.id ? 'on' : ''} onClick={() => set({ tab: t.id })}>
            {t.label}
          </button>
        ))}
      </nav>
      <section className="panel" role="tabpanel"><Panel tab={tab} /></section>
      <aside className="side">
        <Meters />
      </aside>
    </div></div>
  );
}
