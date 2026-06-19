/**
 * Node side tests for the verification layer. structuralCheck parses real
 * emitted bytes and confirms the model is well formed; validateModelShape is the
 * pure range check it delegates to, exercised here against deliberately broken
 * shapes. The browser parity path (verify) loads tfjs-tflite WASM and is not
 * tested here; it runs in the test app.
 */
import { describe, expect, it } from 'vitest';
import { GraphBuilder } from '../src/ir';
import { serialize } from '../src/serialize';
import { f32ToBytes } from '../src/dtype';
import { structuralCheck, validateModelShape } from '../src/verify';

/** A small but real graph: a constant weight feeding an ADD into the output. */
function buildSmallGraph() {
  const g = new GraphBuilder('verify-test');
  const input = g.addActivation('input', [1, 4], 'float32');
  const bias = g.addConst('bias', [1, 4], 'float32', f32ToBytes([1, 2, 3, 4]));
  const output = g.addActivation('output', [1, 4], 'float32');
  g.addOp({
    kind: 'ADD',
    inputs: [input, bias],
    outputs: [output],
    options: { fusedActivation: 'NONE' },
  });
  g.setInputs([input]);
  g.setOutputs([output]);
  return g.build();
}

describe('structuralCheck', () => {
  it('accepts a well formed serialized model and reports the counts', () => {
    const bytes = serialize(buildSmallGraph());
    const result = structuralCheck(bytes);

    expect(result.ok).toBe(true);
    expect(result.issues).toEqual([]);
    // input, bias constant, output.
    expect(result.tensorCount).toBe(3);
    expect(result.operatorCount).toBe(1);
  });

  it('reports a missing TFL3 identifier without throwing', () => {
    const notATflite = new Uint8Array(64);
    const result = structuralCheck(notATflite);

    expect(result.ok).toBe(false);
    expect(result.issues.some((m) => m.includes('TFL3'))).toBe(true);
  });
});

describe('validateModelShape', () => {
  const wellFormed = {
    operatorCodeCount: 1,
    bufferCount: 2,
    tensorCount: 3,
    operatorOpcodeIndices: [0],
    tensorBufferIndices: [0, 1, 0],
    inputs: [0],
    outputs: [2],
  };

  it('returns no issues for a well formed shape', () => {
    expect(validateModelShape(wellFormed)).toEqual([]);
  });

  it('catches an operator opcodeIndex out of range', () => {
    const issues = validateModelShape({ ...wellFormed, operatorOpcodeIndices: [5] });
    expect(issues).toHaveLength(1);
    expect(issues[0]).toMatch(/opcodeIndex 5 is out of range/);
  });

  it('catches a tensor buffer index out of range', () => {
    const issues = validateModelShape({ ...wellFormed, tensorBufferIndices: [0, 9, 0] });
    expect(issues).toHaveLength(1);
    expect(issues[0]).toMatch(/buffer index 9 is out of range/);
  });

  it('catches an input that references a missing tensor', () => {
    const issues = validateModelShape({ ...wellFormed, inputs: [7] });
    expect(issues).toHaveLength(1);
    expect(issues[0]).toMatch(/input 0 references tensor 7/);
  });

  it('catches an output that references a missing tensor', () => {
    const issues = validateModelShape({ ...wellFormed, outputs: [-1] });
    expect(issues).toHaveLength(1);
    expect(issues[0]).toMatch(/output 0 references tensor -1/);
  });

  it('accumulates several issues at once', () => {
    const issues = validateModelShape({
      ...wellFormed,
      operatorOpcodeIndices: [3],
      outputs: [99],
    });
    expect(issues).toHaveLength(2);
  });
});
