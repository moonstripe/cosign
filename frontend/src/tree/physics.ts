// Force-directed layout for the decision tree. Plain mutable bodies, stepped
// once per animation frame. Lightly under-damped: a calm jiggle.
//
// Coordinates are page coordinates, shared with the editor: a node whose text
// is in the document is pulled to the height of its first line (`targets`), so
// the tree lines up with the text it wrote. Other nodes (other branches,
// fully overwritten text) hang below their parent.

export interface Body { x: number; y: number; vx: number; vy: number }

export interface Link { id: string; parent: string | null; depth: number }

export const PHYS = {
  repulsion: 700,       // k / d²
  repulsionCutoff: 200, // px
  spring: 0.07,
  springRest: 44,
  layer: 46,            // px below the parent, for nodes not in the document
  layerPull: 0.025,
  alignPull: 0.12,      // pull toward the node's line in the editor
  center: 0.003,
  noise: 0.085,          // was 0.25: ~66% less idle jiggle
  damping: 0.80,         // was 0.86: settles faster, less wobble
  maxSpeed: 40,
};

export interface StepOpts {
  rootY: number;
  targets: Map<string, number>;
  pinned?: string;
}

export function spawnBody(parent: Body | undefined, width: number, rootY: number, targetY?: number): Body {
  const b = parent ?? { x: width / 2, y: rootY };
  return {
    x: b.x + (Math.random() - 0.5) * 30,
    y: targetY ?? b.y + 20 + Math.random() * 10,
    vx: (Math.random() - 0.5) * 2,
    vy: 1.4 + Math.random(),
  };
}

export function kick(b: Body | undefined, amount: number) {
  if (!b) return;
  b.vx += (Math.random() - 0.5) * 1.36 * amount;
  b.vy += (Math.random() - 0.5) * 1.36 * amount;
}

export function step(bodies: Map<string, Body>, links: Link[], width: number, rootId: string, { rootY, targets, pinned }: StepOpts) {
  const ids = links.map((l) => l.id).filter((id) => bodies.has(id));
  const fx = new Map<string, number>(), fy = new Map<string, number>();
  for (const id of ids) { fx.set(id, 0); fy.set(id, 0); }
  const cut2 = PHYS.repulsionCutoff ** 2;

  for (let i = 0; i < ids.length; i++) {
    for (let j = i + 1; j < ids.length; j++) {
      const a = bodies.get(ids[i])!, b = bodies.get(ids[j])!;
      let dx = a.x - b.x, dy = a.y - b.y, d2 = dx * dx + dy * dy;
      if (d2 > cut2) continue;
      if (d2 < 1) { dx = Math.random() - 0.5; dy = Math.random() - 0.5; d2 = 1; }
      const d = Math.sqrt(d2), k = PHYS.repulsion / d2;
      // two nodes pinned to lines of text should spread sideways, not off their lines
      const ky = targets.has(ids[i]) && targets.has(ids[j]) ? 0.2 : 1;
      fx.set(ids[i], fx.get(ids[i])! + (k * dx) / d); fy.set(ids[i], fy.get(ids[i])! + (ky * k * dy) / d);
      fx.set(ids[j], fx.get(ids[j])! - (k * dx) / d); fy.set(ids[j], fy.get(ids[j])! - (ky * k * dy) / d);
    }
  }

  for (const l of links) {
    const s = bodies.get(l.id);
    if (!s) continue;
    const p = l.parent ? bodies.get(l.parent) : undefined;
    if (p) {
      const dx = s.x - p.x, dy = s.y - p.y, d = Math.max(1, Math.hypot(dx, dy));
      // aligned pairs: the text decides the vertical gap, so the spring rests at it
      const tc = targets.get(l.id), tp = targets.get(l.parent!);
      const rest = tc !== undefined && tp !== undefined ? Math.max(PHYS.springRest, Math.abs(tc - tp)) : PHYS.springRest;
      const k = PHYS.spring * (d - rest);
      fx.set(l.id, fx.get(l.id)! - (k * dx) / d); fy.set(l.id, fy.get(l.id)! - (k * dy) / d);
      fx.set(l.parent!, fx.get(l.parent!)! + (k * dx) / d / 2); fy.set(l.parent!, fy.get(l.parent!)! + (k * dy) / d / 2);
    }
    const target = targets.get(l.id);
    if (target !== undefined) fy.set(l.id, fy.get(l.id)! + (target - s.y) * PHYS.alignPull);
    else fy.set(l.id, fy.get(l.id)! + ((p ? p.y + PHYS.layer : rootY + l.depth * PHYS.layer) - s.y) * PHYS.layerPull);
    fx.set(l.id, fx.get(l.id)! + (width / 2 - s.x) * PHYS.center + (Math.random() - 0.5) * PHYS.noise);
    fy.set(l.id, fy.get(l.id)! + (Math.random() - 0.5) * PHYS.noise);
  }

  for (const id of ids) {
    const s = bodies.get(id)!;
    if (id === rootId) {
      s.x += (width / 2 - s.x) * 0.2; s.y += (rootY - s.y) * 0.2; s.vx = s.vy = 0;
      continue;
    }
    if (id === pinned) { s.vx = s.vy = 0; continue; }
    s.vx = (s.vx + fx.get(id)!) * PHYS.damping;
    s.vy = (s.vy + fy.get(id)!) * PHYS.damping;
    const v = Math.hypot(s.vx, s.vy);
    if (v > PHYS.maxSpeed) { s.vx *= PHYS.maxSpeed / v; s.vy *= PHYS.maxSpeed / v; }
    s.x += s.vx; s.y += s.vy;
  }
}
