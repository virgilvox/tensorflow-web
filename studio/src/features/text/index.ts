/**
 * Pure text feature extraction: standardize, build a capped vocabulary, and
 * encode each string as a fixed length bag of words term frequency vector. This
 * is an honest limit, no embeddings or sequence models, which suits a small MLP
 * and a tiny export. The vocabulary is built from the training set only, so it
 * carries no test leakage. No Vue, no browser.
 */

/** The text preprocessing config. The vocabulary is derived from the data. */
export interface TextFeatureConfig {
  /** Ordered vocabulary; the encoded vector has one slot per term. */
  vocab: string[];
}

/** The cap on vocabulary size, keeping the MLP and the export small. */
export const DEFAULT_VOCAB_CAP = 200;

/**
 * Standardizes a string: lowercase, strip anything that is not a letter or
 * digit, and split into tokens. Returns the token list.
 */
export function standardize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, ' ')
    .trim()
    .split(/\s+/)
    .filter((t) => t.length > 0);
}

/**
 * Builds a vocabulary from a corpus: count token frequencies, then keep the most
 * frequent up to the cap. Ties break alphabetically so the result is
 * deterministic.
 *
 * @returns the ordered vocabulary.
 */
export function buildVocabulary(texts: string[], cap = DEFAULT_VOCAB_CAP): string[] {
  const counts = new Map<string, number>();
  for (const text of texts) {
    for (const token of standardize(text)) {
      counts.set(token, (counts.get(token) ?? 0) + 1);
    }
  }
  return [...counts.entries()]
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .slice(0, cap)
    .map(([token]) => token);
}

/** The tensor shape [vocabSize] a config produces, batch aside. */
export function textTensorShape(config: TextFeatureConfig): [number] {
  return [config.vocab.length];
}

/**
 * Encodes a string as a bag of words term frequency vector over the vocabulary:
 * the count of each vocabulary term divided by the document's total token count,
 * so the scale stays in 0..1 and quantizes cleanly. Out of vocabulary tokens are
 * ignored.
 *
 * @returns a Float32Array of length config.vocab.length.
 */
export function textToFeatures(text: string, config: TextFeatureConfig): Float32Array {
  const index = new Map(config.vocab.map((term, i) => [term, i]));
  const out = new Float32Array(config.vocab.length);
  const tokens = standardize(text);
  for (const token of tokens) {
    const i = index.get(token);
    if (i !== undefined) out[i] += 1;
  }
  if (tokens.length > 0) {
    for (let i = 0; i < out.length; i++) out[i] /= tokens.length;
  }
  return out;
}
