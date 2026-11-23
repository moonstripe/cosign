import { describe as d, expect, it } from 'vitest';
import { DEFAULT_AXES, hasDial, pickFromDelta, strengthOf, weights, WHEEL, wedges, wheelSignposts, withDialAxes } from './wheel';

d('radial wheel', () => {
  it('lays out 8 wedges with opposites across the wheel', () => {
    const w = wedges(DEFAULT_AXES);
    expect(w.length).toBe(8);
    expect(w[0]).toEqual({ concept: 'tender', anti: 'cold' });
    expect(w[4]).toEqual({ concept: 'cold', anti: 'tender' });
  });

  it('measures angle clockwise from up', () => {
    expect(pickFromDelta(0, -100).angle).toBeCloseTo(0);
    expect(pickFromDelta(100, 0).angle).toBeCloseTo(90);
    expect(pickFromDelta(0, 100).angle).toBeCloseTo(180);
    expect(pickFromDelta(-100, 0).angle).toBeCloseTo(270);
  });

  it('maps distance to strength with a dead zone and a cap', () => {
    expect(strengthOf(WHEEL.dead - 1)).toBe(0);
    expect(strengthOf(WHEEL.radius)).toBe(WHEEL.maxStrength);
    expect(strengthOf(WHEEL.radius * 3)).toBe(WHEEL.maxStrength);
    const mid = strengthOf((WHEEL.dead + WHEEL.radius) / 2);
    expect(mid).toBeGreaterThan(0.4); expect(mid).toBeLessThan(0.6);
  });

  it('points straight at a wedge -> just that mood', () => {
    expect(weights(90, 8)).toEqual([{ index: 2, weight: 1 }]);
  });

  it('between two wedges -> blends them', () => {
    const w = weights(22.5, 8);
    expect(w.map((x) => x.index).sort()).toEqual([0, 1]);
    expect(w[0].weight).toBeCloseTo(0.5);
  });

  it('wraps around the top', () => {
    const w = weights(350, 8);
    expect(w.map((x) => x.index).sort()).toEqual([0, 7]);
  });

  it('builds contrastive signposts scaled by distance and weight', () => {
    expect(wheelSignposts(DEFAULT_AXES, { angle: 90, dist: 5 })).toEqual([]);
    const s = wheelSignposts(DEFAULT_AXES, { angle: 90, dist: WHEEL.radius });
    expect(s).toEqual([{ id: 'wheel:2', concept: 'tense', anti: 'calm', strength: 1, ttl: null }]);
  });
});

d('dial-backed wedges', () => {
  const dials = [{ concept: 'tense', anti: 'calm' }];
  it('knows an axis has a dial in either direction', () => {
    expect(hasDial(dials, 'tense', 'calm')).toBe(true);
    expect(hasDial(dials, 'calm', 'tense')).toBe(true);
    expect(hasDial(dials, 'lush', 'plain')).toBe(false);
  });
  it('wedges without a dial give no steering', () => {
    // straight at "tense" (wedge 2, 90°) steers; straight at "lush" (wedge 3, 135°) does not
    expect(wheelSignposts(DEFAULT_AXES, { angle: 90, dist: WHEEL.radius }, dials)).toHaveLength(1);
    expect(wheelSignposts(DEFAULT_AXES, { angle: 135, dist: WHEEL.radius }, dials)).toEqual([]);
    // the opposite wedge (calm, 270°) runs the same dial backwards
    expect(wheelSignposts(DEFAULT_AXES, { angle: 270, dist: WHEEL.radius }, dials)[0].concept).toBe('calm');
  });
  it('puts dial axes on the wheel, replacing undialled ones', () => {
    const axes = withDialAxes([['a', 'b'], ['c', 'd']], [{ concept: 'tense', anti: 'calm' }]);
    expect(axes).toEqual([['tense', 'calm'], ['c', 'd']]);
    expect(withDialAxes(DEFAULT_AXES, dials)).toEqual(DEFAULT_AXES);  // already there
  });
});
