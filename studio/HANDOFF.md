# TF Web Studio handoff

State of the work for whoever picks it up next, human or agent. The paste-ready
prompt for a fresh session is in [`KICKOFF.md`](./KICKOFF.md); this file is the
reference it points at.

## What this is

TF Web Studio is a browser native, local first studio that collects data, trains a
small model, tests it live, and exports a verified int8 `.tflite` plus a C array
and a TFLite Micro sketch. It is built on the headless `tensorflow-web` library
in this same repository and is a separate package in `studio/` that is never
published with the library. The full product spec is [`PLAN.md`](./PLAN.md). The
library is published on npm as `tensorflow-web` (v0.1.2); the studio is the live
demo deployed at [tensorflow.tools](https://tensorflow.tools).

## Status: complete through phase 5, audited and hardened

All five build phases in `PLAN.md` section 16 are done, verified, and committed.
Four modalities (image, audio, motion, text) run the full loop end to end:
collect, optional features and model, train, test live, export a parity verified
int8 `.tflite`. The Guided, Standard, Expert altitude control drives disclosure.

Added after the original five phases (audit driven and on request):
- A standalone Playground (`/playground`, `PlaygroundView.vue`, `usePlayground.ts`)
  that runs a model live on fresh input independent of the project, loading the
  current session model, a self-contained bundle (`.tfwsmodel.json`,
  `lib/modelBundle.ts`), or a bare `.tflite` configured by hand. It holds its own
  interpreter instance, separate from the Test stage's shared live inference. It
  listens continuously on a rolling window (audio) and predicts every frame
  (image), driven by `useMicrophone.latestWindow` and a guarded poll loop, rather
  than capturing one clip per click. A bare `.tflite` whose preprocessing cannot
  be made to match the interpreter's reported input shape is refused with a clear
  message instead of failing at predict.
- A Data stage readiness cue: per class sample progress and a Train action that
  enables when the dataset clears the bar.
- Selectable (and arbitrary) microphone clip length. The audio extractor anchors
  every clip to a fixed `clipSeconds` (carried in `AudioFeatureConfig` and the
  bundle): each clip is padded or truncated to that length and the hop is derived
  from it, so the same sound always maps to the same grid and training and live
  inference stay consistent. A longer configured clip spans more time across the
  same fixed-size grid, so a longer recording is used, not truncated.
- An in-flight guard (`usePipeline.busy`): train and export refuse to overlap, and
  New project / Import / modality switch are blocked while a job runs, so reset()
  can never free tensors out from under a running fit or export.
- Export now also emits the Playground bundle alongside the `.tflite`, C array,
  and sketch.
- Fixes from the audit: a model trained on one dataset no longer survives a New
  project, Import, or modality switch (a `usePipeline.reset()` disposes it and
  resets the training store); import failure now surfaces an inline error instead
  of failing silently; the class row delete is a real button and the modality and
  class controls carry `aria-pressed`.
- A styled in-app confirm dialog (`useConfirm`, `TwConfirm.vue`, mounted in
  `App.vue`) replaces the native `window.confirm` for New project and modality
  switch: focus managed, Escape and backdrop cancel, destructive button styled.
  No `window.confirm` remains.
- Project file parsing hardened: `deserializePayload` validates the numeric
  payload fields, and `parseProject` validates the modality, each class shape, and
  that every sample's payload kind matches the project modality, so a corrupt or
  foreign file fails loudly at load rather than crashing later.
- The mel low-band "collapse" flagged in an early audit was checked numerically
  and does not occur at the studio's config (the points at 32 bands / 512 FFT /
  20 Hz are already strictly increasing: 0, 2, 4, 6, 8, 11, ...). A guard was tried
  and reverted because it was a no-op there and would relocate the collapse to the
  top band at extreme configs. An invariant test asserts the default config stays
  collapse-free.
- Real Standard-altitude controls: the Features stage edits the per-modality knobs
  (image size and channels, audio clip length / mel-or-mfcc / mel bands, motion
  steps, text vocabulary cap) and the Model stage edits a size lever (compact /
  standard / large), all threaded through `buildFeatureConfig`/`presetFor` from the
  settings store. Defaults equal the auto pipeline, so Guided is unchanged. The
  Features stage shows a live input-shape preview; the smoke proves the controls
  are real (changing image size moves the shape).
- Per-class precision, recall, and F1 on the Test stage, computed from the verify
  confusion matrix (`lib/metrics.ts`), for every modality.
- Capture hardening: camera and microphone surface a clear secure-context message
  on an insecure origin instead of a raw TypeError. Interpreter models are now
  disposed before being replaced in live inference and the Playground.
- Audit fix: live capture and the exported bundle take the audio clip length from
  the model's FROZEN `featureCfg.audio.clipSeconds`, not the mutable
  `settings.audioSeconds`, so changing the clip control after training cannot feed
  the model a differently anchored clip.
- Deployed as a static site (DigitalOcean) at [tensorflow.tools](https://tensorflow.tools).
  Getting there needed two studio-only build fixes (see Deployment below). The top
  bar links to the GitHub repo (`virgilvox/tensorflow-web`) and the bench-rail
  footer credits hack.build.

Everything above is committed and pushed to `origin/main`: the original five build
phases, the rename to TF Web Studio, the audio and motion fixed-scale fixes, the
Playground, the Standard-altitude controls, the deploy fixes, and the README and
site links. Run `git log --oneline` for the detailed history.

## How to verify (run from `studio/`)

```
npm install
npm run typecheck                # vue-tsc, expect 0 errors
npm run test                     # Vitest unit tests, expect 69 passing
node scripts/smoke.mjs           # shell smoke (13 checks)
node scripts/smoke-image.mjs     # image flow end to end (7)
node scripts/smoke-audio.mjs     # audio keyword spotting (6)
node scripts/smoke-mt.mjs        # motion and text (10)
node scripts/smoke-expert.mjs    # expert depth + project save/load (5)
node scripts/smoke-persist.mjs   # reload persistence regression (4)
node scripts/smoke-playground.mjs # playground: current model + reloaded bundle (6)
node scripts/smoke-audio-bg.mjs  # audio: loud keyword vs quiet background (5)
node scripts/smoke-motion-idle.mjs # motion: shake vs near-still idle (5)
npm run dev                      # http://localhost:5173 (or next free port)
```

Last known good: typecheck 0, 69 unit tests, and all nine smokes green. The smokes
use the system Chrome through Playwright (`channel: 'chrome', headless: true`),
start their own Vite server, and inject learnable data through the real file or
text import UI, since a headless browser has no camera, microphone, or motion
sensor. Each modality smoke trains in Guided mode, confirms parity against the
real TFLite interpreter, an int8 accuracy at or above the floor, a device fit,
and a correct live prediction. Verify and live inference need network for the
TFLite WASM interpreter loaded from a CDN.

## Deployment

The studio is a static Vite SPA, deployed to DigitalOcean App Platform as a
**Static Site**: source directory `studio`, build command `npm run build`, output
directory `dist`, no run command. It uses hash routing (`createWebHashHistory`),
so deep links work on static hosting with no rewrite rules. At runtime the client
loads tfjs and the TFLite WASM interpreter from a CDN (see `index.html`), so a
visitor needs normal outbound network; collect/train/quantize/export work offline,
verify and live inference need the CDN.

Two studio-only fixes were needed to build on a studio-rooted deploy (where only
`studio/`'s deps are installed, not the repo root's), both in invariant form below:

1. The deploy build is `vite build`, not `vue-tsc --noEmit && vite build`. Running
   `vue-tsc` re-typechecked the whole aliased library and demanded the library's
   own dev deps (flatbuffers and tfjs-tflite type decls). Typecheck stays a
   separate `npm run typecheck`.
2. `flatbuffers` is a studio dependency AND in `resolve.dedupe`. The library source
   at `../src` imports it at runtime; the bundler resolves that import relative to
   `../src` (the repo root node_modules, absent on a studio-only install), so
   deduping forces it to the studio's own copy, the same trick used for tfjs.

## Layout

```
studio/
  index.html                 loads the tfjs and tfjs-tflite CDN globals, the fonts
  vite.config.ts             alias tensorflow-web -> ../src/index.ts; tfjs and
                             flatbuffers dedupe; tfjs-tflite external
  src/
    main.ts App.vue router.ts
    design/ tokens.css base.css components/Tw*.vue   (20 components, incl. TwConfirm)
    stages/ Data Features Model Train Test Export Playground views
    stores/ project.ts training.ts settings.ts          (Pinia)
    composables/  the only place that touches the library and browser devices:
      useCamera useMicrophone useMotion useDataset useInterpreter useTrainer
      useQuantizer useVerifier useDeviceBudget useLiveInference useExport
      useProjectFile usePlayground usePipeline (the orchestrator)
    features/  pure extractors: image/ audio/ motion/ text/ + index.ts dispatch
    models/    builder.ts presets.ts types.ts
    lib/       dsp tensors storage split cformat format modalities imageDecode
               audioDecode motionImport projectFile modelBundle
    types.ts
  test/        Vitest unit tests for the pure pieces
  scripts/     headless smokes (one per modality plus shell and persistence)
```

Separation holds: the design system knows nothing about machine learning, the
feature extractors are pure functions with no Vue, the composables own all
library and device calls, the stores hold state, the views compose.

## Invariants that must not regress

These were each a bug or a near miss. Keep them true.

1. One `@tensorflow/tfjs` copy. The studio and the aliased library source both
   import tfjs; two copies give two nominal `Tensor` types and two runtime
   instances. Enforced by `resolve.dedupe: ['@tensorflow/tfjs']` in
   `vite.config.ts` and a `paths` entry in `tsconfig.json`. Do NOT dedupe the
   tfjs sub packages (core, layers, converter); that mixes versions and breaks
   their internal imports.
2. The TFLite interpreter is the alpha.8 CDN global from `index.html`, never
   bundled. `optimizeDeps.exclude` plus `build.rollupOptions.external` keep it out
   of the bundle; `setWasmPath` is called before verify. alpha.10 omits the WASM.
3. Never write Vue reactive proxies to IndexedDB. Structured clone throws on a
   Proxy. Persist plain objects (see `useDataset.persistMeta`). This was the cause
   of the orphaned sample crash on reload.
4. Parity is enforced, not just shown. Only a `.tflite` whose verify reports
   parity is downloadable or loaded for live inference (`usePipeline.exportModel`,
   gated downloads in `ExportView.vue`).
5. A project is single modality. Switching modality with data clears it after a
   confirm, so no sample is ever attached to the wrong modality.
6. `useDataset.init` loads from storage once per session and self heals orphans
   (drops any sample whose class is missing). Do not reload from storage on every
   stage view remount, and do not load samples without reconciling against the
   classes.
7. Dispose a trained model AND its optimizer. tfjs does not free an instance
   optimizer on `model.dispose`; `usePipeline.disposeModel` frees both.
8. The model builder restricts to the twelve exportable layers and
   `assertExportable` mirrors the library's export guard. An unsupported layer
   fails loud with its name.
9. A smoke must reload the page after capture to catch persistence regressions
   (this is what `smoke-persist.mjs` exists for).
10. A trained model must never outlive the dataset it was trained on. New project,
   Import, and modality switch all call `usePipeline.reset()`, which disposes the
   model and resets the training store, so a stale model can never be tested or
   downloaded against a different dataset. The Playground's `usePlayground` holds
   its OWN interpreter instance; loading a model there must not touch the Test
   stage's shared live inference.
11. One job at a time. `usePipeline.busy` gates train and export against each other
   and gates reset() (New project, Import, modality switch) against a running job,
   so the shared module-level tensors and model are never freed mid-fit or
   mid-export. Keep these guards when adding any action that mutates pipeline state.
12. Audio extraction is anchored to a fixed `clipSeconds`, not the recorded clip
   length. Keep the pad/truncate-to-clipSeconds step so the same sound maps to the
   same grid; deriving the hop from a single clip's length silently warps the time
   axis and breaks train/live parity.
13. Audio log mel is mapped to 0..1 against a FIXED reference (`LOG_FLOOR`/`LOG_CEIL`
   in `features/audio`), never a per-clip min-max. Per-clip min-max stretched a
   quiet Background Noise clip to full contrast, so a keyword model predicted the
   keyword for near-silence (a real, reported bug). `smoke-audio-bg.mjs` guards it:
   a quiet clip must predict the background class, not the keyword. Do not
   reintroduce per-clip min-max for mel features.
14. Motion features use a per-axis FIXED scale set from the training set (each
   axis's max magnitude, floored at a fraction of the overall max), carried in
   `MotionFeatureConfig.scales`, never a per-window min-max. Per-window min-max
   stretched a near-still window to full contrast, so an Idle class looked like a
   gesture. `smoke-motion-idle.mjs` guards it: a still window must predict Idle,
   not the gesture. Do not reintroduce per-window min-max for motion.
15. Live inference and export read the FROZEN `featureCfg` captured at train time,
   never the mutable settings. The Standard knobs (clip length, image size, etc.)
   change `settings` and apply only on the next train; Test, the Playground, and
   the exported bundle must derive their config (notably the audio clip length)
   from `featureCfg`, or a settings change after training silently feeds the model
   input it was not trained on.
16. The deploy must build from `studio/` alone (the repo root's deps are not
   installed there). Keep `build` as `vite build` only (not `vue-tsc && vite build`,
   which re-typechecks the library and needs its dev deps), and keep `flatbuffers`
   both as a studio dependency and in `resolve.dedupe` so the library's runtime
   import resolves from the studio's node_modules. Verified by building with the
   repo-root `flatbuffers` hidden.

## Hard boundary with the library

Do not modify the library's `src/`, root `package.json`, `tsup.config.ts`, or the
existing `app/`. The studio imports the library source through the Vite alias.
Run all studio commands from `studio/`, and use `git -C <repo>` for git so the
shell working directory does not drift to the repo root; a stray `npm install` in
the repo root pollutes the library's `package.json` (it happened once and was
reverted).

## What remains (all optional, none blocking)

- The Expert architecture search under the device budget. The deferable stretch
  in `PLAN.md` section 16.6. Not started.
- A full editable layer editor. Today the Expert path shows a read only operator
  inspector; the plan flags the editor as a stretch.
- Bare `.tflite` support in the Playground is pragmatic: the user picks the
  modality, preprocessing, and labels (image preprocessing is prefilled from the
  interpreter's reported input shape when available). Audio and motion fall back
  to the default feature config, so a foreign model whose preprocessing differs
  will misread; a studio bundle avoids this. Text needs a bundle for its
  vocabulary. Deriving more of the config from a bare model is a future step.
- Real device validation. The webcam, microphone, and DeviceMotion capture paths
  are wired and typecheck, but only the file and text import paths are exercised
  by the headless smokes. Test capture on real hardware.
- Flash the generated C array and TFLite Micro sketch to an actual ESP32. The
  output is op compatible and arena estimated, not yet flashed.
- Deferred audit nits, confirmed to have no functional or parity impact:
  calibration uses one representative batch (a memory concern only at large
  dataset sizes), the audio and motion resamplers use different endpoint
  conventions, and `persistMeta` has a harmless last write wins ordering under
  rapid edits.
- Quantization aware training is not in the library; the studio shows the int8
  accuracy delta and offers mitigations instead.

## Decisions already made (do not relitigate)

- Stage labels are plain words: Data, Features, Model, Train, Test, Export. The
  machine shop character is visual only.
- Image, audio, and motion lead; text is bag of words, included with its limits
  stated.
- Transfer learning is out of scope; if added, flag that it will not fit an MCU.
- Int8 post training quantization only.

## Rules

Binding rules live in [`../AGENTS.md`](../AGENTS.md) and are summarized in the
kickoff prompt. The ones that bite most often: no AI attribution anywhere
(commits, comments, docs), no emojis (inline SVG icons only), no em-dashes (a
middot is used as the no value placeholder), plain direct prose, documented
public functions, Apache 2.0, commit authored as the human, and ask before
pushing.
