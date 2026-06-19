import { describe, it, expect } from 'vitest';
import {
  standardize,
  buildVocabulary,
  textToFeatures,
  textTensorShape,
} from '../src/features/text';

describe('standardize', () => {
  it('lowercases and strips punctuation into tokens', () => {
    expect(standardize('Hello, World!')).toEqual(['hello', 'world']);
    expect(standardize('  spaced   out  ')).toEqual(['spaced', 'out']);
    expect(standardize('')).toEqual([]);
  });
});

describe('buildVocabulary', () => {
  it('keeps the most frequent tokens up to the cap, ties alphabetical', () => {
    const corpus = ['the cat', 'the dog', 'the bird sings'];
    const vocab = buildVocabulary(corpus, 3);
    expect(vocab[0]).toBe('the'); // most frequent
    expect(vocab.length).toBe(3);
  });
});

describe('textToFeatures', () => {
  it('encodes term frequencies over the vocabulary', () => {
    const config = { vocab: ['cat', 'dog', 'bird'] };
    const f = textToFeatures('cat cat dog', config);
    expect(f.length).toBe(3);
    // Two of three tokens are "cat", one is "dog".
    expect(f[0]).toBeCloseTo(2 / 3, 5);
    expect(f[1]).toBeCloseTo(1 / 3, 5);
    expect(f[2]).toBe(0);
  });

  it('ignores out of vocabulary tokens', () => {
    const config = { vocab: ['cat'] };
    const f = textToFeatures('cat zebra zebra', config);
    expect(f[0]).toBeCloseTo(1 / 3, 5); // one of three tokens is in vocab
  });

  it('reports the vocabulary sized shape', () => {
    expect(textTensorShape({ vocab: ['a', 'b', 'c', 'd'] })).toEqual([4]);
  });
});
