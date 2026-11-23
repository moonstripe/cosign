import { useLayoutEffect, useRef, useState } from 'react';
import { editor } from '../lib/editorController';
import { useApp } from '../store';

export function CandidatePopover() {
  const pop = useApp((s) => s.popover);
  const steered = useApp((s) => s.signposts.length > 0);
  const ref = useRef<HTMLDivElement>(null);
  const [left, setLeft] = useState(0);

  useLayoutEffect(() => {
    if (!pop || !ref.current) return;
    const w = ref.current.offsetWidth;
    setLeft(Math.max(8 + scrollX, Math.min(pop.x, innerWidth - w - 8 + scrollX)));
  }, [pop]);

  if (!pop) return null;
  const max = Math.max(...pop.items.map((c) => c.prob));
  return (
    <div ref={ref} className="popover" style={{ left, top: pop.y }}>
      {pop.items.map((c, i) => {
        const diff = c.base_prob == null ? 0 : c.prob - c.base_prob;
        return (
          <div key={c.token_id} className={'row' + (i === pop.sel ? ' sel' : '')}
               onMouseDown={(e) => { e.preventDefault(); editor.pick(i); }}>
            <span className="k">{i + 1}</span>
            <span className="tok">
              {c.text.replace(/\n/g, '↵').replace(/^ /, '·')}
              <span className="bar" style={{ width: `${(100 * c.prob) / max}%` }} />
              {c.base_prob != null && <span className="ghost" style={{ width: `${Math.min(100, (100 * c.base_prob) / max)}%` }} />}
            </span>
            <span className="p">
              {(c.prob * 100).toFixed(1)}%
              {Math.abs(diff) >= 0.005 && (
                <span className={'d ' + (diff > 0 ? 'up' : 'down')}>{diff > 0 ? '+' : '−'}{Math.round(Math.abs(diff) * 100)}</span>
              )}
            </span>
          </div>
        );
      })}
      <div className="foot">
        {steered && 'bar = steered · dotted = without mood dials · '}1–9 / ↵ pick · ⌥Tab let model write · Esc close
      </div>
    </div>
  );
}
