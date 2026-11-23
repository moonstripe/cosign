import { describe, expect, it } from 'vitest';
import { attachPangram, children, endChain, goto, initTree, record, ROOT, type TreeState } from './model';

const snap = (text: string) => ({ html: `<span>${text}</span>`, text, caret: text.length });
const kinds = (s: TreeState) => Object.values(s.nodes).map((n) => `${n.id}:${n.kind}:${n.parent ?? '-'}`);

describe('tree model', () => {
  it('merges a run of typing into one node', () => {
    let s = initTree();
    s = record(s, 'human', 'type', snap('Hel'));
    s = record(s, 'human', 'type', snap('Hello'));
    expect(kinds(s)).toEqual(['n0:root:-', 'n1:human:n0']);
    expect(s.nodes.n1.inserted).toBe('Hello');
  });

  it('model writes always make a new node; typing after starts a new one', () => {
    let s = initTree();
    s = record(s, 'human', 'type', snap('A'));
    s = record(s, 'model', 'opt-tab', snap('A B'));
    s = record(s, 'model', 'opt-tab', snap('A B C'));
    s = record(s, 'human', 'type', snap('A B C d'));
    expect(kinds(s)).toEqual(['n0:root:-', 'n1:human:n0', 'n2:model:n1', 'n3:model:n2', 'n4:human:n3']);
  });

  it('pick chains merge until ended', () => {
    let s = initTree();
    s = record(s, 'picked', 'pick', snap(' a'));
    s = record(s, 'picked', 'pick', snap(' a b'));
    s = endChain(s);
    s = record(s, 'picked', 'pick', snap(' a b c'));
    expect(kinds(s)).toEqual(['n0:root:-', 'n1:picked:n0', 'n2:picked:n1']);
  });

  it('goto + act forks a branch', () => {
    let s = initTree();
    s = record(s, 'human', 'type', snap('A'));
    s = record(s, 'model', 'opt-tab', snap('A B'));
    s = goto(s, 'n1');
    s = record(s, 'model', 'opt-tab', snap('A X'));
    expect(children(s, 'n1').map((n) => n.id)).toEqual(['n2', 'n3']);
    // typing after a goto never merges into the old node
    s = goto(s, 'n1');
    s = record(s, 'human', 'type', snap('A, then'));
    expect(children(s, 'n1').length).toBe(3);
  });

  describe('deleting from the end walks back up', () => {
    // typed sentence -> ⌘↵ -> typed sentence
    const build = () => {
      let s = initTree();
      s = record(s, 'human', 'type', snap('One.'));
      s = record(s, 'model', 'cmd-enter', snap('One. Two.'));
      s = record(s, 'human', 'type', snap('One. Two. Three.'));
      return s;
    };

    it('deleting the last typed sentence removes its node', () => {
      let s = build();
      for (const t of ['One. Two. Three', 'One. Two. Th', 'One. Two. ', 'One. Two.']) s = record(s, 'human', 'type', snap(t));
      expect(kinds(s)).toEqual(['n0:root:-', 'n1:human:n0', 'n2:model:n1']);
      expect(s.cur).toBe('n2');
    });

    it('deleting into model text trims the model node and keeps it current', () => {
      let s = build();
      s = record(s, 'human', 'type', snap('One. Tw'));
      expect(kinds(s)).toEqual(['n0:root:-', 'n1:human:n0', 'n2:model:n1']);
      expect(s.cur).toBe('n2');
      expect(s.nodes.n2.text).toBe('One. Tw');
      // typing now starts a new human node under the trimmed model node
      s = record(s, 'human', 'type', snap('One. Tw!'));
      expect(s.nodes[s.cur].parent).toBe('n2');
    });

    it('deleting everything returns to the root', () => {
      let s = build();
      s = record(s, 'human', 'type', snap(''));
      expect(kinds(s)).toEqual(['n0:root:-']);
      expect(s.cur).toBe(ROOT);
    });

    it('undoing a node with edits in the middle of the text also removes it', () => {
      let s = initTree();
      s = record(s, 'human', 'type', snap('One. Two.'));
      s = record(s, 'model', 'opt-tab', snap('X One. Two.'));      // model wrote at the start
      s = record(s, 'human', 'type', snap('X Hi One. Two.'));      // you typed in the middle
      s = record(s, 'human', 'type', snap('X One. Two.'));         // ...and deleted it again
      expect(kinds(s)).toEqual(['n0:root:-', 'n1:human:n0', 'n2:model:n1']);
      expect(s.cur).toBe('n2');
    });

    it('never deletes a node that other branches hang off', () => {
      let s = build();
      s = goto(s, 'n2');
      s = record(s, 'model', 'opt-tab', snap('One. Two. Four.'));  // second child of n2
      s = goto(s, 'n3');
      s = { ...s, nodes: { ...s.nodes } };
      s = record(s, 'human', 'type', snap('One.'));  // deletes n3, but n2 has a branch
      expect(s.nodes.n2).toBeDefined();
      expect(s.nodes.n3).toBeUndefined();
      expect(s.cur).toBe('n1');
    });
  });

  it('attaches pangram scores to every node with that text, and to future nodes', () => {
    let s = initTree();
    s = record(s, 'human', 'type', snap('A'));
    s = attachPangram(s, 'A', { ai: 0.2, assisted: 0.1, human: 0.7 });
    expect(s.nodes.n1.pg?.human).toBe(0.7);
    s = record(s, 'model', 'opt-tab', snap('A B'));
    s = goto(s, 'n1');
    s = record(s, 'human', 'type', snap('A'));  // no change
    s = goto(s, ROOT);
    s = record(s, 'human', 'type', snap('A'));  // same text elsewhere picks up the cached score
    expect(s.nodes[s.cur].pg?.ai).toBe(0.2);
  });
});

import { fractionsIn } from './model';
describe('fractionsIn (import chunk scores)', () => {
  const W = [
    { label: 'AI-Generated', ai_assistance_score: 1, confidence: 'High', start_index: 0, end_index: 100 },
    { label: 'Human Written', ai_assistance_score: 0, confidence: 'High', start_index: 100, end_index: 200 },
  ];
  it('splits windows over a character range', () => {
    expect(fractionsIn(W, 0, 100)).toEqual({ ai: 1, assisted: 0, human: 0 });
    expect(fractionsIn(W, 100, 200)).toEqual({ ai: 0, assisted: 0, human: 1 });
    expect(fractionsIn(W, 50, 150)).toEqual({ ai: 0.5, assisted: 0, human: 0.5 });
  });
  it('is null outside every window', () => expect(fractionsIn(W, 300, 400)).toBeNull());
});
