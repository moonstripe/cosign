// WebSocket client for the engine (generate / candidates / stop).

import type { Candidate, DialAxis, Signpost } from '../types';

type ServerMsg =
  | { op: 'ready'; model: string; device: string; axes: DialAxis[] }
  | { op: 'axes'; axes: DialAxis[] }
  | { op: 'token'; id: number; text: string }
  | { op: 'done'; id: number; written: number; signposts: Signpost[] }
  | { op: 'candidates'; id: number; items: Candidate[] };

export interface RequestBase {
  prefix: string;
  signposts: Signpost[];
  temp: number;
}

type Handler = (m: ServerMsg) => void;

export class EngineSocket {
  private ws: WebSocket | null = null;
  private seq = 0;
  private pending = new Map<number, Handler>();
  onStatus: (s: { connected: boolean; model?: string; device?: string }) => void = () => {};
  onAxes: (axes: DialAxis[]) => void = () => {};

  connect() {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const ws = new WebSocket(`${proto}://${location.host}/ws`);
    this.ws = ws;
    ws.onmessage = (ev) => {
      const m = JSON.parse(ev.data) as ServerMsg;
      if (m.op === 'ready') { this.onStatus({ connected: true, model: m.model, device: m.device }); this.onAxes(m.axes ?? []); return; }
      if (m.op === 'axes') { this.onAxes(m.axes); return; }
      this.pending.get(m.id)?.(m);
      if (m.op === 'done' || m.op === 'candidates') this.pending.delete(m.id);
    };
    ws.onclose = () => {
      this.onStatus({ connected: false });
      setTimeout(() => this.connect(), 1500);
    };
  }

  get ready() { return this.ws?.readyState === WebSocket.OPEN; }

  private send(payload: object, handler: Handler): number {
    const id = ++this.seq;
    this.pending.set(id, handler);
    this.ws!.send(JSON.stringify({ id, ...payload }));
    return id;
  }

  generate(req: RequestBase & { n?: number; until?: 'sentence' },
           on: { token: (t: string) => void; done: (signposts: Signpost[]) => void }) {
    return this.send({ op: 'generate', ...req }, (m) => {
      if (m.op === 'token') on.token(m.text);
      else if (m.op === 'done') on.done(m.signposts);
    });
  }

  candidates(req: RequestBase & { k: number }): Promise<Candidate[]> {
    return new Promise((resolve) => this.send({ op: 'candidates', ...req }, (m) => {
      if (m.op === 'candidates') resolve(m.items);
    }));
  }

  /** Ask which moods have programmed dials (the server hot-reloads moods/). */
  refreshAxes() {
    if (this.ready) this.ws!.send(JSON.stringify({ op: 'axes' }));
  }

  stop() { if (this.ready) this.ws!.send(JSON.stringify({ op: 'stop' })); }
}

export const engine = new EngineSocket();
