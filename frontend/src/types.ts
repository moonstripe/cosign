/** Who wrote a span. 'doc' = imported from a PDF, not yet touched by you or the model. */
export type Src = 'human' | 'model' | 'picked' | 'doc';

export interface Signpost {
  id?: string;
  concept: string;
  anti: string | null;
  strength: number;
  ttl: number | null; // tokens left; null = until removed
}

/** A mood axis with a weight-space dial on the server (moods/<concept>__<anti>.pt). */
export interface DialAxis { concept: string; anti: string | null; notes?: string }

export interface Candidate {
  token_id: number;
  text: string;
  prob: number;
  base_prob: number | null; // without signposts
}

export interface PangramWindow {
  label: string;
  ai_assistance_score: number;
  confidence: string;
  start_index: number;
  end_index: number;
}

export interface PangramResult {
  headline: string | null;
  prediction: string | null;
  prediction_short: string | null;
  fraction_ai: number;
  fraction_ai_assisted: number;
  fraction_human: number;
  windows: PangramWindow[];
}

export interface PgFractions {
  ai: number;
  assisted: number;
  human: number;
}

export type Via = 'root' | 'type' | 'opt-tab' | 'cmd-enter' | 'pick' | 'import';

export interface Snapshot {
  html: string;
  text: string;
  caret: number;
}
