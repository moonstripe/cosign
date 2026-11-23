import { useMemo } from 'react';
import { useApp } from '../store';
import { describe, hasDial, heat, strengthOf, WHEEL, wedges, weights, wheelSignposts } from '../lib/wheel';

// model colour -> red as the setting goes past the redline
const hotColour = (h: number) => `rgb(${Math.round(37 + (205 - 37) * h)},${Math.round(99 + (40 - 99) * h)},${Math.round(235 + (40 - 235) * h)})`;


// annular wedge path, angles in degrees clockwise from up
function wedgePath(r0: number, r1: number, a0: number, a1: number) {
  const p = (r: number, a: number) => {
    const t = ((a - 90) * Math.PI) / 180;
    return `${(r * Math.cos(t)).toFixed(2)} ${(r * Math.sin(t)).toFixed(2)}`;
  };
  const large = a1 - a0 > 180 ? 1 : 0;
  return `M ${p(r1, a0)} A ${r1} ${r1} 0 ${large} 1 ${p(r1, a1)} L ${p(r0, a1)} A ${r0} ${r0} 0 ${large} 0 ${p(r0, a0)} Z`;
}

export function RadialWheel() {
  const wheel = useApp((s) => s.wheel);
  const axes = useApp((s) => s.axes);
  const chips = useApp((s) => s.signposts);
  const dialAxes = useApp((s) => s.dialAxes);
  const ws = useMemo(() => wedges(axes), [axes]);
  if (!wheel) return null;

  const { pick } = wheel;
  const n = ws.length, step = 360 / n, R = WHEEL.radius;
  const active = pick.dist >= WHEEL.dead ? weights(pick.angle, n) : [];
  const weightOf = (i: number) => active.find((a) => a.index === i)?.weight ?? 0;
  const s = strengthOf(pick.dist);
  const steer = wheelSignposts(axes, pick, dialAxes);
  const reach = Math.min(pick.dist, R + 20);
  const t = ((pick.angle - 90) * Math.PI) / 180;
  const rAt = (v: number) => WHEEL.dead + (R - WHEEL.dead) * (v / WHEEL.maxStrength);
  const rRed = rAt(WHEEL.redline);
  const h = heat(s);
  const aim = h > 0 ? hotColour(h) : undefined;

  return (
    <div className="wheel" style={{ left: wheel.x, top: wheel.y }}>
      <svg width={2 * R + 120} height={2 * R + 120} viewBox={`${-R - 60} ${-R - 60} ${2 * R + 120} ${2 * R + 120}`}>
        <defs>
          {/* tachometer redline: clear up to the redline radius, deepening red to the rim */}
          <radialGradient id="redline" gradientUnits="userSpaceOnUse" cx={0} cy={0} r={R}>
            <stop offset={rRed / R} stopColor="rgb(205,40,40)" stopOpacity={0} />
            <stop offset={(rRed / R + 1) / 2} stopColor="rgb(205,40,40)" stopOpacity={0.18} />
            <stop offset={1} stopColor="rgb(205,40,40)" stopOpacity={0.42} />
          </radialGradient>
        </defs>
        <circle r={R + 4} className="wheel-bg" />
        {ws.map((w, i) => {
          const a0 = i * step - step / 2, a1 = i * step + step / 2;
          const live = hasDial(dialAxes, w.concept, w.anti);
          const wt = live ? weightOf(i) : 0;
          const la = ((i * step - 90) * Math.PI) / 180, lr = R * 0.68;
          return (
            <g key={i}>
              <path d={wedgePath(WHEEL.dead, R, a0 + 1, a1 - 1)} className={'wedge' + (live ? '' : ' nodial')} style={{ fillOpacity: live ? 0.05 + 0.4 * wt * (s / WHEEL.maxStrength) : 0.02 }} />
              <text x={lr * Math.cos(la)} y={lr * Math.sin(la)} className={'wlabel' + (wt > 0 ? ' on' : '') + (live ? '' : ' nodial')}
                    style={{ opacity: live ? 0.55 + 0.45 * wt : 0.3 }}>{w.concept}{live ? '' : ' ·'}</text>
            </g>
          );
        })}
        <path d={`M ${R} 0 A ${R} ${R} 0 1 1 ${-R} 0 A ${R} ${R} 0 1 1 ${R} 0 M ${rRed} 0 A ${rRed} ${rRed} 0 1 0 ${-rRed} 0 A ${rRed} ${rRed} 0 1 0 ${rRed} 0 Z`}
              fill="url(#redline)" fillRule="evenodd" pointerEvents="none" />
        {[0.25, 0.5, 0.75].map((v) => <circle key={v} r={rAt(v)} className="wheel-tick" />)}
        <circle r={rRed} className="wheel-redline" />
        {[0.5, 1].map((v) => <text key={v} x={4} y={-rAt(v) + 10} className="wtick">{v}</text>)}
        <circle r={WHEEL.dead} className="wheel-dead" />
        {pick.dist >= WHEEL.dead && (
          <>
            <circle r={reach} className="wheel-reach" style={aim ? { stroke: aim } : undefined} />
            <line x1={WHEEL.dead * Math.cos(t)} y1={WHEEL.dead * Math.sin(t)} x2={reach * Math.cos(t)} y2={reach * Math.sin(t)} className="wheel-aim" style={aim ? { stroke: aim } : undefined} />
            <circle cx={reach * Math.cos(t)} cy={reach * Math.sin(t)} r={5 + 2 * h} className="wheel-dot" style={aim ? { fill: aim } : undefined} />
          </>
        )}
        {pick.dist >= WHEEL.dead && <text y={0} className="wcenter" style={aim ? { fill: aim } : undefined}>{s.toFixed(2)}</text>}
      </svg>
      <div className="wheel-readout">
        <b style={aim ? { color: aim } : undefined}>{describe(steer)}</b>
        {h > 0 && <span className="redline-note"> · red zone: expect frantic, looping text</span>}
        {chips.length > 0 && <span> · plus held: {chips.map((c) => `${c.concept} ${c.strength.toFixed(2)}`).join(', ')}</span>}
        {!steer.length && pick.dist >= WHEEL.dead && <div>no dial for this mood yet — grey wedges aren't programmed</div>}
        <div>release to act · Esc to cancel</div>
      </div>
    </div>
  );
}
