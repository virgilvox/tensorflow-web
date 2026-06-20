import { describe, it, expect } from 'vitest';
import {
  toCIdentifier,
  toCArray,
  toTFLMSketch,
  resolverCallsForLayers,
} from '../src/lib/cformat';

describe('toCIdentifier', () => {
  it('sanitizes a project name to a C identifier', () => {
    expect(toCIdentifier('My Notes Chat')).toBe('my_notes_chat');
    expect(toCIdentifier('parts-bin 2')).toBe('parts_bin_2');
  });

  it('prefixes a leading digit and falls back to model', () => {
    expect(toCIdentifier('3 things')).toBe('m_3_things');
    expect(toCIdentifier('   ')).toBe('model');
  });
});

describe('toCArray', () => {
  it('formats bytes as a length tagged hex array', () => {
    const bytes = new Uint8Array([0, 15, 255, 16]);
    const c = toCArray(bytes, 'demo');
    expect(c).toContain('demo_tflite[] = {');
    expect(c).toContain('0x00, 0x0f, 0xff, 0x10');
    expect(c).toContain('demo_tflite_len = 4;');
    expect(c).toContain('alignas(16)');
  });

  it('wraps at sixteen bytes per row', () => {
    const bytes = new Uint8Array(20).fill(1);
    const rows = toCArray(bytes, 'm')
      .split('\n')
      .filter((l) => l.trim().startsWith('0x'));
    expect(rows.length).toBe(2);
  });

  it('carries an include guard so it can be included more than once', () => {
    expect(toCArray(new Uint8Array([1]), 'm')).toContain('#pragma once');
  });

  it('throws on empty input rather than emitting uncompilable C', () => {
    expect(() => toCArray(new Uint8Array([]), 'm')).toThrow();
  });
});

describe('resolverCallsForLayers', () => {
  it('always includes the quantize boundary ops and maps each layer', () => {
    const calls = resolverCallsForLayers(['Conv2D', 'MaxPooling2D', 'Flatten', 'Dense'], true);
    expect(calls).toContain('AddQuantize');
    expect(calls).toContain('AddDequantize');
    expect(calls).toContain('AddConv2D');
    expect(calls).toContain('AddMaxPool2D');
    expect(calls).toContain('AddReshape');
    expect(calls).toContain('AddFullyConnected');
    expect(calls).toContain('AddSoftmax');
  });

  it('de duplicates repeated layer types', () => {
    const calls = resolverCallsForLayers(['Conv2D', 'Conv2D', 'Conv2D']);
    expect(calls.filter((c) => c === 'AddConv2D').length).toBe(1);
  });
});

describe('toTFLMSketch', () => {
  it('embeds a padded arena and the resolver calls', () => {
    const sketch = toTFLMSketch({
      varName: 'gesture',
      arenaBytes: 10000,
      inputShape: [1, 48, 48, 1],
      classCount: 3,
      ops: { resolverCalls: ['AddQuantize', 'AddConv2D'] },
    });
    expect(sketch).toContain('MicroMutableOpResolver<2>');
    expect(sketch).toContain('resolver.AddConv2D();');
    // The arena is the estimate padded by 25 percent, rounded to a kilobyte.
    expect(sketch).toContain('kArenaSize = 13312');
    expect(sketch).toContain('gesture_tflite');
  });
});
