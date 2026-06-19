# VISE Studio kickoff prompt

Paste the block below into a fresh session opened in this repository. It is self
contained and assumes no prior conversation.

---

Build VISE Studio, a browser native, local first machine learning training studio,
in this repository (/Users/obsidian/Projects/ossuary-projects/tensorflow-web). The
complete specification is `studio/PLAN.md`. It is your source of truth. Read it in
full before writing anything.

Read these first, in order:
1. `studio/PLAN.md` — the full spec, build phases, and exit tests.
2. `AGENTS.md` — the binding rules. Follow every one.
3. `DESIGN.md` — the library architecture and the supported layer set.
4. `src/index.ts` and `src/types.ts` — the `tensorflow-web` public API you will call
   (train, calibrate, quantize, toTFLite, verify, convert, supportedLayers).
5. `examples/cdn.html`, `app/main.ts`, `app/index.html`, `app/parity.ts`,
   `scripts/run-parity.mjs` — working reference patterns for the full pipeline and
   the TFLite interpreter setup. Reuse what they do.
6. `/Users/obsidian/Downloads/vise-design-system.html` — the VISE design system.
   Copy its `:root` tokens into `studio/src/design/tokens.css` and implement its
   components (buttons, segmented altitude control, toggles, fields, sliders,
   status, badges, cards, gauges, the jaw grip, nav rail, bench rail) as small Vue
   components. Match the steel surfaces, the single caution accent `#ccf23e`, square
   corners, hard cut shadows, the three typefaces, and the grain overlay.

Hard rules from AGENTS.md, non negotiable:
- Never add Claude or any AI as a commit author or co author. No AI attribution in
  any commit, comment, or document.
- No emojis. Use inline SVG icons. No emoji used as an icon.
- No em-dashes, ever. No double hyphen substitutes.
- No typical AI or large language model phrasing. Write plain, direct prose.
- Smart separation of concerns. No monolithic files. Document public functions and
  composables with what they do, return, and throw.
- Apache 2.0. Do not reinvent the wheel: use Vue, Pinia, Vue Router, a proven FFT,
  and the existing library, not hand rolled equivalents.

Decisions already made, do not relitigate them:
- Stage labels are plain words: Data, Features, Model, Train, Test, Export. The VISE
  machine shop character is visual only.
- Lead with image, audio, and motion. Include bag of words text alongside them.
- Transfer learning is out of scope; if ever added, flag clearly that it will not
  fit a microcontroller.
- The Expert architecture search is a deferable stretch.

Critical integration gotchas, already solved, do not rediscover them (see
`studio/PLAN.md` section 20 for detail):
1. The TFLite interpreter is browser WASM and the alpha.10 npm package omits the
   WASM. For verify and live inference, load the interpreter from the alpha.8 CDN as
   a global script, load `@tensorflow/tfjs@4.9.0` as a global first, and call
   `setWasmPath` to the alpha.8 dist before verify. Copy the setup from
   `app/index.html` and `examples/cdn.html`.
2. Vite cannot bundle `@tensorflow/tfjs-tflite`. Exclude it with
   `optimizeDeps.exclude` and use the CDN global.
3. The emitted `.tflite` has a fixed input batch of one. Run live inference one
   sample at a time.
4. The library trains CNN and MLP from scratch, int8, float input and output. Only
   these twelve layers export: Conv2D, DepthwiseConv2D, Dense, MaxPooling2D,
   AveragePooling2D, GlobalAveragePooling2D, Flatten, Reshape, Softmax, Activation,
   ReLU, Add. The model builder must restrict to these; an unsupported layer fails
   loud on export.
5. The studio is its own package in `studio/` with its own `package.json`. Do not
   modify the library's `src`, `package.json`, or `tsup.config`. Import the library
   through a Vite alias to `../src/index.ts`. The library publishes only `dist`, so
   the studio never ships. Leave the existing `app/` in place; it backs the parity
   tests.

How to work:
- Build phase by phase exactly as `studio/PLAN.md` lists. Do not skip ahead.
- At each phase exit, verify before claiming success: run the studio typecheck and
  the vitest unit tests, and run a headless browser smoke with Playwright using the
  system Chrome (`chromium.launch({ channel: 'chrome', headless: true })`, the same
  approach as `scripts/run-parity.mjs`) to drive the loop and confirm it works.
  Paste real output. Do not claim a phase works without running it.
- Commit at each phase exit, authored as the human, with no AI attribution. Ask
  before pushing.

Start now with phase 1, the shell: scaffold the Vue 3, Vite, TypeScript, Pinia, and
Vue Router project in `studio/`; implement `tokens.css` and `base.css` from the VISE
design system; build the core design components and the bench shell (top bar with
the altitude control, nav rail, jaw gripped work panel, bench rail); and confirm the
altitude control switches a visible disclosure state. Then move to phase 2, the
image flow end to end, which is the flagship. Report honestly at each step: what
works, what you verified and how, and any limitation.
