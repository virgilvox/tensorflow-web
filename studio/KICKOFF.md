# VISE Studio continuation kickoff

VISE Studio is already built: all five phases in `studio/PLAN.md` are complete,
verified, and committed, then audited and hardened. This file is the paste-ready
prompt to continue the work in a fresh session. The full state is in
[`HANDOFF.md`](./HANDOFF.md).

Paste the block below into a fresh session opened in this repository. It is self
contained and assumes no prior conversation.

---

You are continuing VISE Studio, a browser native, local first machine learning
training studio in this repository
(`/Users/obsidian/Projects/ossuary-projects/tensorflow-web`). It is already built
end to end and is in good shape; you are extending and maintaining it, not
starting over.

Read these first, in order, before changing anything:
1. `studio/HANDOFF.md` the current state: what is done, how to verify, the
   invariants that must not regress, the hard boundary with the library, and
   what remains.
2. `studio/PLAN.md` the full product spec, build phases, and exit tests.
3. `AGENTS.md` the binding rules. Follow every one.
4. `DESIGN.md` the library architecture and the supported layer set.
5. `src/index.ts` and `src/types.ts` the `tensorflow-web` public API the studio
   calls (train, calibrate, quantize, toTFLite, verify, supportedLayers).
6. `studio/scripts/*.mjs` the headless smokes; they show the verification pattern
   and the interpreter setup. `app/index.html` and `examples/cdn.html` show the
   TFLite interpreter CDN setup the studio reuses.

What exists now: four modalities (image, audio, motion, text) run the full loop,
collect, optional features and model, train, test live, and export a parity
verified int8 `.tflite` plus a C array and a TFLite Micro sketch. The Guided,
Standard, Expert altitude control drives disclosure. Projects persist to
IndexedDB and save to and load from a local file.

Hard rules from AGENTS.md, non negotiable:
- Never add Claude or any AI as an author or co author, and no AI attribution in
  any commit, comment, or document.
- No emojis. Use inline SVG icons. No emoji as an icon.
- No em-dashes, ever, and no double hyphen substitutes. A middot is the no value
  placeholder already in use.
- No typical AI or large language model phrasing. Write plain, direct prose.
- Smart separation of concerns. No monolithic files. Document public functions
  and composables with what they do, return, and throw.
- Apache 2.0. Do not reinvent the wheel: use Vue, Pinia, Vue Router, the proven
  FFT, and the existing library.
- Be certain before acting. Verify before you claim. Report outcomes faithfully;
  if a test fails, show it.

The hard boundary with the library, do not cross it:
- Do not modify the library's `src/`, root `package.json`, `tsup.config.ts`, or
  the existing `app/`. The studio is its own package in `studio/` and imports the
  library source through a Vite alias to `../src/index.ts`. It is never published
  with the library.
- Run all studio commands from `studio/`. Use `git -C <repo>` for git so the
  shell working directory does not drift to the repo root. A stray `npm install`
  in the repo root pollutes the library's `package.json`.

Invariants that must not regress (each was a real bug; see HANDOFF.md for detail):
1. One `@tensorflow/tfjs` copy, via `resolve.dedupe: ['@tensorflow/tfjs']` and a
   tsconfig path. Do not dedupe the tfjs sub packages.
2. The TFLite interpreter is the alpha.8 CDN global, never bundled; setWasmPath
   before verify. alpha.10 omits the WASM.
3. Never write Vue reactive proxies to IndexedDB; persist plain objects.
4. Parity is enforced: only a verify passing `.tflite` is downloadable or loaded
   for live inference.
5. A project is single modality; switching modality with data clears it after a
   confirm.
6. The dataset loads from storage once per session and self heals orphaned
   samples; do not reload on every view remount.
7. Dispose a trained model and its optimizer together.
8. The model builder restricts to the twelve exportable layers and fails loud on
   anything else, mirroring the library export guard.

How to work:
- Verify before claiming success. Run from `studio/`: `npm run typecheck` (expect
  0), `npm run test` (expect 52 passing), and the smokes
  `node scripts/smoke.mjs`, `smoke-image.mjs`, `smoke-audio.mjs`, `smoke-mt.mjs`,
  `smoke-expert.mjs`, and `smoke-persist.mjs` (expect all green). The smokes use
  the system Chrome through Playwright, start their own Vite server, and inject
  data through the real file or text import UI. Paste real output. A smoke that
  touches the dataset must reload the page after capture, since that is how a
  past persistence regression was caught.
- Commit at each meaningful step, authored as the human contributor, with no AI
  attribution. Ask before pushing. Note: at handoff there is one unpushed commit
  (`889b064`, the persistence fix); confirm the suite is green before pushing it.
- To run it for a person: `npm run dev` from `studio/`, then open the printed
  localhost URL. Camera, microphone, and motion need a real device and a secure
  context (localhost qualifies).

What to do next, if not directed otherwise. None of these block; pick what the
maintainer asks, or from HANDOFF.md "What remains": the Expert architecture
search under the device budget, a full editable layer editor (today it is a read
only operator inspector), replacing the native confirm prompts with styled
modals, validating the real device capture paths on hardware, flashing the C
array and sketch to an actual ESP32, and per class precision, recall, and F1 for
text. Whatever you build, hold the same verification bar and keep every invariant
above true.
