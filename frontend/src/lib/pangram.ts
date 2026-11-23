import type { PangramResult } from '../types';

export const MIN_WORDS = 20;
export const wordCount = (t: string) => (t.match(/\S+/g) ?? []).length;

export async function detect(text: string): Promise<PangramResult> {
  const r = await fetch('/api/detect', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ text }),
  });
  const j = await r.json();
  if (!r.ok) throw new Error(j.error ?? `HTTP ${r.status}`);
  return j as PangramResult;
}

const RGB = { human: [47, 133, 90], assisted: [214, 158, 46], ai: [197, 48, 48] } as const;
export const PG_COLORS = { human: 'rgb(47,133,90)', assisted: 'rgb(214,158,46)', ai: 'rgb(197,48,48)' };

/** Mix of green (human) / amber (assisted) / red (AI) by Pangram's fractions. */
export function cloudColor(p: { ai: number; assisted: number; human: number }): string {
  const t = p.ai + p.assisted + p.human || 1;
  const c = [0, 1, 2].map((i) => Math.round((p.human * RGB.human[i] + p.assisted * RGB.assisted[i] + p.ai * RGB.ai[i]) / t));
  return `rgb(${c.join(',')})`;
}
