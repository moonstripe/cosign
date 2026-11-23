import { useEffect, useState } from 'react';
import {
  deleteDocument, downloadCosign, downloadText, importChunks, listDocs, newDocument, openDocument, openFile,
  renameDocument, saveNow, type DocSummary,
} from '../lib/docs';
import { useApp } from '../store';

const ago = (t: number) => {
  const s = Date.now() / 1000 - t;
  if (s < 60) return 'just now';
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  return new Date(t * 1000).toLocaleDateString();
};

const SAVE_LABEL = { idle: '', unsaved: 'editing…', saving: 'saving…', saved: 'saved', error: 'not saved' } as const;

/** Upload a PDF -> Reducto parses it -> a new document: a linear chain of chunk nodes -> Pangram scores each. */
async function importPdf(f: File, say: (s: string) => void) {
  say('parsing with Reducto…');
  const r = await fetch('/api/import', { method: 'POST', body: f, headers: { 'content-type': 'application/pdf', 'x-filename': f.name } });
  const j = await r.json();
  if (!r.ok || j.error) throw new Error(j.error ?? `HTTP ${r.status}`);
  const info = `${j.chunks.length} chunks · ${j.words} words · ${j.pages ?? '?'} pages${j.truncated ? ' (truncated)' : ''}`;
  say(`${info} · checking with Pangram…`);
  await importChunks(j.chunks, f.name);
  const pg = useApp.getState().pangram;
  say(`${info}${pg.error ? ` · pangram: ${pg.error}` : ''}`);
}

function OpenMenu({ close, say }: { close: () => void; say: (s: string) => void }) {
  const [docs, setDocs] = useState<DocSummary[] | null>(null);
  const [err, setErr] = useState('');
  const docId = useApp((s) => s.docId);
  const refresh = () => listDocs().then(setDocs).catch((e) => setErr(e.message));
  useEffect(() => { refresh(); }, []);
  const pick = async (fn: () => Promise<unknown>) => {
    try { await fn(); close(); } catch (e) { setErr((e as Error).message); }
  };
  return (
    <div className="menu open-menu" onMouseDown={(e) => e.stopPropagation()}>
      <div className="mhead">documents</div>
      {err && <div className="merr">{err}</div>}
      {!docs && !err && <div className="mrow muted">loading…</div>}
      {docs?.length === 0 && <div className="mrow muted">no saved documents yet</div>}
      {docs?.map((d) => (
        <div key={d.id} className={'mrow doc' + (d.id === docId ? ' cur' : '')} onClick={() => d.id !== docId && pick(() => openDocument(d.id))}>
          <span className="dtitle">{d.title}</span>
          <span className="dmeta">{d.words} words · {ago(d.updated)}</span>
          <button className="del" title="delete" onClick={async (e) => {
            e.stopPropagation();
            if (!confirm(`Delete “${d.title}”? This can't be undone.`)) return;
            try { await deleteDocument(d.id); refresh(); } catch (er) { setErr((er as Error).message); }
          }}>×</button>
        </div>
      ))}
      <label className="mrow action">open a file… <span className="dmeta">.cosign.json · .txt · .md · .pdf</span>
        <input type="file" accept=".json,.txt,.md,.markdown,.pdf,application/pdf,application/json,text/plain"
               onChange={(e) => {
                 const f = e.target.files?.[0]; e.target.value = '';
                 if (!f) return;
                 pick(async () => {
                   if (/\.pdf$/i.test(f.name)) await importPdf(f, say);
                   else say(await openFile(f));
                 });
               }} />
      </label>
    </div>
  );
}

function SaveMenu({ close }: { close: () => void }) {
  const run = (fn: () => unknown) => { fn(); close(); };
  return (
    <div className="menu save-menu" onMouseDown={(e) => e.stopPropagation()}>
      <div className="mrow action" onClick={() => run(saveNow)}>save now <span className="dmeta">⌘S · autosaves as you write</span></div>
      <div className="mrow action" onClick={() => run(downloadCosign)}>download .cosign.json <span className="dmeta">text + authorship + tree</span></div>
      <div className="mrow action" onClick={() => run(downloadText)}>download .txt <span className="dmeta">plain text</span></div>
    </div>
  );
}

/** Upper left: wordmark, document title + save state, file actions, PDF upload. */
export function Topbar() {
  const { docTitle, saveState, saveError, generating } = useApp();
  const [menu, setMenu] = useState<'open' | 'save' | null>(null);
  const [status, setStatus] = useState('');
  const [title, setTitle] = useState(docTitle);
  useEffect(() => setTitle(docTitle), [docTitle]);
  const docId = useApp((s) => s.docId);
  useEffect(() => { if (!docId) setStatus((x) => (x.includes('…') ? x : '')); }, [docId]);
  useEffect(() => {
    const down = () => setMenu(null);
    const key = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 's') { e.preventDefault(); void saveNow(); }
      if (e.key === 'Escape') setMenu(null);
    };
    document.addEventListener('mousedown', down);
    document.addEventListener('keydown', key);
    return () => { document.removeEventListener('mousedown', down); document.removeEventListener('keydown', key); };
  }, []);
  const toggle = (m: 'open' | 'save') => () => setMenu((x) => (x === m ? null : m));
  const fail = (e: unknown) => setStatus(`failed: ${(e as Error).message}`);

  return (
    <header id="topbar">
      <span className="brand">Cosign</span>
      <input className="doctitle" value={title} spellCheck={false} aria-label="document title"
             onChange={(e) => setTitle(e.target.value)}
             onBlur={() => renameDocument(title)}
             onKeyDown={(e) => {
               if (e.key === 'Enter') (e.target as HTMLInputElement).blur();
               if (e.key === 'Escape') { setTitle(docTitle); setTimeout(() => (e.target as HTMLInputElement).blur()); }
             }} />
      <span className={'savestate ' + saveState} title={saveError ?? ''}>{SAVE_LABEL[saveState]}</span>
      <nav className={generating ? 'off' : ''}>
        <button onClick={() => { setStatus(''); newDocument().catch(fail); }}>new</button>
        <span className="anchor" onMouseDown={(e) => e.stopPropagation()}>
          <button onClick={toggle('open')} className={menu === 'open' ? 'on' : ''}>open</button>
          {menu === 'open' && <OpenMenu close={() => setMenu(null)} say={setStatus} />}
        </span>
        <span className="anchor" onMouseDown={(e) => e.stopPropagation()}>
          <button onClick={toggle('save')} className={menu === 'save' ? 'on' : ''}>save</button>
          {menu === 'save' && <SaveMenu close={() => setMenu(null)} />}
        </span>
        <label className="upload" title="Upload a PDF as a new document: Reducto splits it into chunks, Pangram scores each one">
          ↑ upload PDF
          <input type="file" accept="application/pdf,.pdf" disabled={generating}
                 onChange={(e) => {
                   const f = e.target.files?.[0]; e.target.value = '';
                   if (f) importPdf(f, setStatus).catch(fail);
                 }} />
        </label>
      </nav>
      {status && <div className="upstatus">{status}</div>}
    </header>
  );
}
