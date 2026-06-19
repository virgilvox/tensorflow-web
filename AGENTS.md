# AGENTS.md

Rules for any contributor working in this repository, human or automated. This is the canonical rules file. It follows the [AGENTS.md](https://agents.md) open standard so that every coding agent and every person reads the same instructions. `CLAUDE.md` exists only to point here.

These rules are binding. When a rule and a convenience conflict, the rule wins.

## Project

`tensorflow-web` is a headless TypeScript library that trains a TensorFlow model in the browser tab and emits a verified `.tflite` file. It sits next to TensorFlow.js as a peer dependency and adds calibration, int8 post training quantization, and FlatBuffers serialization to a real `.tflite`. It has no user interface of its own. The repository holds two things only: the library, and a small test app that drives it.

The full design lives in the project plan. Read it before making structural changes.

## Commits and attribution

1. Never add Claude, any AI assistant, or any AI tool as an author or co-author of a commit.
2. Never add a `Co-Authored-By` trailer that credits Claude or any AI.
3. Never add "Generated with", "Created by AI", or any similar attribution to commit messages, pull request bodies, code comments, or documentation.
4. Commit messages describe the change and the reason for it. They do not mention the tool that produced the change.
5. Author every commit as the human contributor running the work.

There are no exceptions to this section.

## Correctness

This library lives or dies on int8 correctness. A `.tflite` that produces different numbers in the interpreter than the float model produced in TensorFlow.js is worthless and dangerous, because it fails silently. Hold the whole codebase to that standard.

1. Be 100 percent certain before you act. If you are not certain, research until you are, then act.
2. Never make an assumption that is not backed by research, a reference, a test, or the source code in front of you. State the basis for non-obvious claims.
3. When you cannot verify something, say so plainly. Do not present a guess as a fact.
4. Verify before you trust. The `verify` module loading the emitted `.tflite` back into `tfjs-tflite` and checking parity against the float reference is the spine of the project, not an afterthought. Nothing is trusted until it matches.
5. Report outcomes faithfully. If a test fails, say so and show the output. If a step was skipped, say so. Do not soften a failure into a success.
6. Pin the schema version. Regenerate FlatBuffers bindings deliberately, never silently.

## Architecture

1. Smart separation of concerns. Each module owns one job. `train`, `calibrate`, `quantize`, `serialize`, `ops`, `verify`, and `emitC` stay distinct and do not reach into each other's internals.
2. No monolithic files. One converter per supported layer under `src/ops/`. One concern per file. If a file grows past its single responsibility, split it.
3. The op registry is pluggable. Adding a layer is a small, isolated, testable change, not a rewrite.
4. An unsupported layer fails loudly on export with the offending op named. It never writes a silently wrong file.
5. The library stays headless. No DOM, no framework, no opinion about anything visual. Functions return plain data, a promise, or progress events. The UI decides everything visual.
6. The test app in `app/` is never published. It imports the live library source.

## Do not reinvent the wheel

1. Before writing something, check whether a maintained library or an established pattern already solves it. Prefer the proven option.
2. Use the public FlatBuffers schema and the `flatbuffers` runtime. Do not hand roll serialization that the schema already defines.
3. Use `tfjs-tflite` as the parity oracle. Do not write a second interpreter.
4. Use TensorFlow.js for model building and training. The library adds the half that TensorFlow.js does not have, and nothing more.
5. Follow established best practices for TypeScript, ESM, and the browser target.

## Writing style

This applies to code comments, commit messages, documentation, the README, and every other piece of prose in the repository.

1. No emojis. Anywhere.
2. No emoji used in place of a proper icon. If an interface needs an icon, use a real icon asset or an SVG.
3. No em-dashes, ever.
4. No double hyphens standing in for an em-dash.
5. No typical AI or large language model phrasing. Avoid filler openers, hollow hedging, inflated adjectives, and the habit of restating the prompt. Write direct, plain, specific prose. Say the thing.
6. Document well. Every public function carries a doc comment that states what it does, what it returns, and what it can throw. Non-obvious internal logic carries a comment that explains why, not what.

## Documentation and the README

1. The README targets an open source audience: someone who has never seen the project and wants to install it, understand what it does, and use it.
2. Lead with what the library is and the one honest constraint: training is general, export is bounded by the op registry, and a model outside the registry trains fine and then fails loudly on export.
3. Show the public API and a one shot convenience path. Keep examples runnable and true to the current code. Do not document a feature that does not exist yet. If something is planned, mark it as planned.
4. State the verification story. A reader needs to know that the emitted file is checked against the float reference before it is trusted.
5. Credit any work that was used or leveraged to reach a solution at the bottom of the README. This includes libraries, specifications, articles, prior art, and any reference that materially shaped a design or fixed a problem. Keep this section current as the project grows.

## License

The project is licensed under Apache License 2.0. See `LICENSE` and `NOTICE`.

This choice is deliberate. TensorFlow, TensorFlow.js, `tfjs-tflite`, the TFLite schema, and Google FlatBuffers are all Apache-2.0. Apache-2.0 keeps the library compatible with every dependency it builds on, it is permissive enough for wide adoption, and it carries an explicit patent grant that matters for a model format and quantization library. Do not relicense without a documented reason and the maintainer's approval.

## Working with research

1. Leverage web research when a fact is outside the repository. Cite the source.
2. Prefer primary sources: the specification, the schema, the library's own documentation, the source code.
3. When research informs a design decision or fixes a bug, record the reference so it can be credited in the README.
