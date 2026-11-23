// Radial "weapon wheel" for steering.
//
// Hold ⌥Tab / ⇧Tab / ⌘↵ and the wheel opens around the mouse. Wedges are moods;
// opposite wedges are opposites (each axis is a contrast, which steers harder
// than a single word). Direction picks the mood — between two wedges it
// blends them — and distance from the centre is intensity. Release to act.

import type { DialAxis, Signpost } from '../types';

export type Axis = [string, string];   // [mood, its opposite]

export const DEFAULT_AXES: Axis[] = [
  ['tender', 'cold'],
  ['playful', 'grave'],
  ['tense', 'calm'],
  ['lush', 'plain'],
];

export const WHEEL = {
  radius: 150,     // px, outer edge of the wedges
  dead: 26,        // px, centre zone: act without steering
  maxStrength: 1,  // 1 = full dial (strongest still-mostly-fluent setting)
  redline: 0.65,   // past this the text starts to come apart (frantic, loops)
  minWeight: 0.15, // drop blend partners lighter than this
};

/** 0 below the redline, rising to 1 at full strength — how "hot" a setting is. */
export const heat = (s: number) => Math.max(0, Math.min(1, (s - WHEEL.redline) / (WHEEL.maxStrength - WHEEL.redline)));

/** UI strength is 0–1; the server's signpost scale is 0–10 (dial alpha = strength/10). */
export const toServer = (s: number) => Math.round(s * 100) / 10;

/** Does the server have a dial for this mood (either direction of an axis)? */
export function hasDial(dialAxes: DialAxis[], concept: string, anti: string | null): boolean {
  return dialAxes.some((d) => (d.concept === concept && d.anti === anti) || (d.concept === anti && d.anti === concept));
}

/** Put every dial-backed axis on the wheel, replacing axes that have no dial. */
export function withDialAxes(axes: Axis[], dialAxes: DialAxis[]): Axis[] {
  const out = axes.slice();
  for (const d of dialAxes) {
    if (!d.anti || out.some(([a, b]) => hasDial([d], a, b))) continue;
    const i = out.findIndex(([a, b]) => !hasDial(dialAxes, a, b));
    if (i >= 0) out[i] = [d.concept, d.anti];
  }
  return out;
}

/** 8 wedges clockwise from the top; wedge i's opposite is i+4. */
export function wedges(axes: Axis[]): { concept: string; anti: string }[] {
  const pos = axes.map(([a, b]) => ({ concept: a, anti: b }));
  const neg = axes.map(([a, b]) => ({ concept: b, anti: a }));
  return [...pos, ...neg];
}

export interface WheelPick {
  angle: number;    // degrees, 0 = up, clockwise
  dist: number;     // px from centre
}

export function pickFromDelta(dx: number, dy: number): WheelPick {
  const angle = ((Math.atan2(dx, -dy) * 180) / Math.PI + 360) % 360;
  return { angle, dist: Math.hypot(dx, dy) };
}

export function strengthOf(dist: number): number {
  const t = (dist - WHEEL.dead) / (WHEEL.radius - WHEEL.dead);
  return Math.round(Math.max(0, Math.min(1, t)) * WHEEL.maxStrength * 20) / 20;
}

/** Wedge weights for a direction: the two nearest wedges, linearly blended. */
export function weights(angle: number, n: number): { index: number; weight: number }[] {
  const step = 360 / n;
  const f = (((angle % 360) + 360) % 360) / step;
  // wedge centres sit at i*step; blend between floor and ceil
  const i = Math.floor(f) % n, j = (i + 1) % n, frac = f - Math.floor(f);
  let ws = [{ index: i, weight: 1 - frac }, { index: j, weight: frac }].filter((w) => w.weight >= WHEEL.minWeight);
  const tot = ws.reduce((a, w) => a + w.weight, 0);
  ws = ws.map((w) => ({ ...w, weight: w.weight / tot }));
  return ws.sort((a, b) => b.weight - a.weight);
}

/** Signposts for a wheel position (empty in the dead zone). */
export function wheelSignposts(axes: Axis[], pick: WheelPick, dialAxes?: DialAxis[]): Signpost[] {
  const s = strengthOf(pick.dist);
  if (pick.dist < WHEEL.dead || s === 0) return [];
  const ws = wedges(axes);
  const live = (i: number) => !dialAxes || hasDial(dialAxes, ws[i].concept, ws[i].anti);
  return weights(pick.angle, ws.length).filter(({ index }) => live(index)).map(({ index, weight }) => ({
    id: `wheel:${index}`,
    concept: ws[index].concept,
    anti: ws[index].anti,
    strength: Math.round(s * weight * 100) / 100,
    ttl: null,
  }));
}

export function describe(sps: Signpost[]): string {
  if (!sps.length) return 'no steering';
  return sps.map((s) => `${s.concept} ${s.strength.toFixed(2)}`).join(' + ');
}
