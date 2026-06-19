# VISE Studio plan

A browser native, local first studio for training small models and shipping a
verified int8 `.tflite`, built on the `tensorflow-web` library and styled with
the VISE design system. It is a separate application in this repository that is
never published with the library.

## 0. The thesis

Research across Edge Impulse, Google Teachable Machine, Microsoft Lobe, Roboflow,
NanoEdge AI Studio, and SensiML found one open gap: a tool with Teachable Machine
and Lobe level simplicity that actually ships an int8 `.tflite` for bare metal
edge devices. The simple tools (Teachable Machine, Lobe) do not target
microcontrollers or hide deployment; the capable tools (Edge Impulse, SensiML,
NanoEdge) require accounts, cloud upload, or paid deployment, and bury the user
in jargon and expert decisions. VISE Studio fills that gap: everything runs in
the browser, nothing is uploaded, no account is needed, and the output is a
verified int8 `.tflite` plus a ready to flash C array for an ESP32.

## 1. Goals and non goals

Goals:
- A beginner can collect data, train, test live, and download a working `.tflite`
  in a few minutes without knowing any machine learning terms.
- An expert can open every lever: feature parameters, a layer editor, quantization
  options, calibration, tolerance, and an optional architecture search.
- Cover four modalities end to end: image, audio (keyword and sound), motion
  (accelerometer gesture), and simple text classification.
- Stay light: browser native, no heavy framework beyond Vue, no cloud.
- Be readable: small files, clear separation, so a developer can open the code and
  understand or modify any one piece.

Non goals:
- Not a general deep learning IDE. It trains the small CNN and MLP models the
  `tensorflow-web` library supports, quantized to int8, for edge deployment.
- Not a cloud service. No accounts, no servers, no telemetry.
- Not a replacement for the library. The library stays headless; the studio is one
  user interface on top of it.

## 2. Principles drawn from the research

Pain points to avoid, each turned into a rule:

1. No accounts, no cloud upload, nothing leaves the browser. This is the headline.
2. No invented jargon. Stages are named in plain words, not machine shop or DSP
   terms. See the naming decision in section 4.
3. Never surface an unguided expert decision to a beginner. Feature pipeline and
   architecture are auto chosen, shown read only with a one line reason, and only
   editable behind the Expert level.
4. No compute caps. Training is local and unlimited.
5. Nothing the user makes is paywalled or forced public. It is their data and
   their model, on their machine.
6. No hardware lock in. The output is a standard `.tflite` that runs on any TFLite
   or TFLite Micro target.
7. Always show the deployment target fit. A budget meter shows flash size, an
   estimated runtime arena, and whether it fits the selected device.
8. Browser native, no install. An optional offline mode is a later enhancement.
9. The negative or unknown class is a scaffolded, first class step, not optional.
10. Quantization accuracy loss is shown, never hidden. The float versus int8
    accuracy delta is displayed before export.

Patterns to adopt:

- One button local training, results forming live.
- Capture in the browser by webcam burst, microphone, motion, or drag and drop,
  and label in the same motion.
- A live test stage with real time confidence bars on fresh input.
- Zero configuration auto architecture, with the choice visible and explainable.
- A short, plain, linear progress spine.
- Device constraint awareness shown inline.
- An optional architecture search for experts, never a blocker.
- Export buttons named by target, not by file format.
- Progressive disclosure between a beginner front door and an expert front door.

## 3. Positioning

| Need | Teachable Machine | Edge Impulse | VISE Studio |
| --- | --- | --- | --- |
| No account, fully local | Yes | No | Yes |
| Beginner can finish in minutes | Yes | No | Yes |
| Feature and architecture control | No | Yes | Yes, behind Expert |
| Ships int8 .tflite for a microcontroller | No | Yes (account) | Yes |
| Shows float vs int8 accuracy delta | No | Partly | Yes |
| Shows "fits your ESP32" budget | No | Yes | Yes |
| C array plus inference snippet for the board | No | Yes (account) | Yes, generated in browser |
| Verified parity against the real interpreter | No | No | Yes |
| Open source, readable, hackable | No | No | Yes |

## 4. Design system and naming

The studio uses the VISE design system verbatim for its visual language: the steel
surface scale, chalk and ash ink, the single caution accent `#ccf23e`, the patina
secondary, square corners, hard cut shadows, the three typefaces (Chakra Petch for
display and numbers, JetBrains Mono for body and data, Space Mono for labels), the
grain overlay, the jaw grip on the primary work panel, instrument gauges, the
hazard stripe for live state, and the status squares.

The design tokens become a single `tokens.css` (the `:root` block from the design
system), and each demonstrated component becomes a small Vue component.

Naming decision. The VISE brand and machine shop personality live in the visuals,
the typography, the jaw grip, and the gauges, not in the labels. The research is
clear that invented stage names (Edge Impulse uses "impulse", "processing block",
"learning block") force a glossary and slow beginners down. So the nav rail and
stage titles use plain words: Data, Features, Model, Train, Test, Export. The VISE
character carries the identity without costing clarity. Decided: stage labels are
plain words. The machine shop personality stays in the visuals only.

The altitude control from the design system (the Guided, Standard, Expert
segmented control) is the core of the beginner to expert experience and lives in
the top bar. It drives progressive disclosure across every stage.

The bench shell, mapped from the design system layout anatomy:
- Top bar: VISE mark, the altitude control, the active project, the target device
  selector, and a local only indicator.
- Nav rail: the workflow stages, the active one marked with the live left border.
- Work panel: the current stage, gripped by the jaws. This is the one bold move,
  used once per screen.
- Bench rail: persistent live state, the training job, sample counts, model size,
  and the device fit meter.

## 5. The workflow

The plain four step spine the user always sees: Collect, Train, Test, Export.
Two configuration stages, Features and Model, sit between Collect and Train and are
hidden in Guided, shown read only in Standard, and fully editable in Expert.

Stages, in nav rail order:
1. Data: choose a preset or modality, manage classes including the scaffolded
   negative class, capture or import samples, see per class counts and balance and
   leakage warnings.
2. Features: the preprocessing that turns raw samples into tensors, with a live
   preview of the processed result. Auto chosen per modality with a one line
   reason. Editable in Expert.
3. Model: the architecture, auto sized from the modality and data, shown with its
   parameter count and estimated size. A layer editor in Expert, restricted to the
   operators the library supports, failing loud on anything else.
4. Train: one button training with live loss and accuracy gauges, a loss curve,
   the hazard rail while running, an epoch readout, early stop, and cancel.
5. Test: live inference on fresh camera, microphone, motion, or text input with
   real time confidence bars, plus held out test accuracy, a confusion matrix, and
   per class precision and recall.
6. Export: calibrate, quantize to int8, emit the `.tflite`, verify parity in the
   real interpreter, show the float versus int8 accuracy delta and the device fit,
   then offer the named downloads.

Altitude behavior:
- Guided: Features and Model are automatic and hidden. The flow is Data, then a
  prominent Train button that lights up when the minimum data is met, then Test,
  then Export. Coaching and reason badges are everywhere.
- Standard: all stages visible, defaults applied, the main knobs editable
  (image size, epochs, architecture preset).
- Expert: every lever, the layer editor, quantization scheme, calibration method,
  tolerance, an operator inspector, and an optional architecture search that
  respects the device budget.

## 6. Modalities

Each modality defines: how capture works, how raw samples become a tensor, the
default model, the coaching shown during capture, and the scaffolded negative
class. Feature extraction is a set of pure functions so it is unit testable without
a camera or microphone.

Image, audio, and motion are the lead modalities and are built first. Text, as a
bag of words MLP, is included alongside them rather than deferred, with its honest
limits called out in the interface.

Image:
- Capture: webcam burst, file or folder drag and drop, or a draw pad.
- Features: resize to a small square (default 48 by 48), grayscale or three
  channel, normalize to zero to one, optional augmentation (flip, small crop,
  brightness). Tensor shape is side by side by channels.
- Model: a small CNN from scratch (Conv, pool, Conv, pool, flatten, dense,
  softmax). The library trains from scratch, not by transfer learning, which suits
  tiny edge models. Transfer learning stays out of scope. If it is ever added as an
  Expert option, it must be flagged clearly that the result will likely not fit a
  microcontroller.
- Coaching: aim for roughly fifty per class, vary angle, lighting, and background,
  add a Neither class.
- Negative class: Neither, auto offered.

Audio:
- Capture: microphone, one second clips by default at sixteen kilohertz.
- Features: framed short time Fourier transform to a log mel spectrogram (default
  thirty two mel bands by about thirty two frames), treated as a single channel
  image. MFCC available in Expert; default to MFCC for spoken keywords and the mel
  spectrogram for general sounds, the choice hidden in Guided with a reason badge.
- Model: a small CNN over the spectrogram.
- Coaching: record varied speakers, avoid single syllable keywords, capture about
  ten minutes of background noise.
- Negative class: Background Noise, auto created and required, plus an Other Words
  prompt for keyword tasks.

Motion:
- Capture: the device motion sensor (accelerometer, optionally gyroscope), a window
  of one to two seconds, with a live trace shown during capture so the gesture is
  visibly distinct.
- Features: windowing, then either a reshaped two dimensional tensor for a small
  CNN or statistical and spectral features for an MLP. Default window two seconds,
  sampling at least twice the fastest motion of interest.
- Model: a small CNN or MLP.
- Coaching: perform the gesture continuously, vary speed and orientation, capture
  about three minutes per class.
- Negative class: Idle, auto included.

Text:
- Capture: typed or pasted strings, or import a list.
- Features: standardize (lowercase, strip punctuation), build a capped vocabulary,
  encode as a fixed length bag of words or hashed vector. This is an honest limit:
  no embeddings or sequence models, which suits a small MLP and tiny export. For
  small datasets, optional easy data augmentation (synonym replace, insert, swap,
  delete).
- Model: a small MLP over the encoded vector.
- Coaching: keep classes balanced, show per class counts, report per class
  precision, recall, and F1 rather than only accuracy.
- Negative class: Other, offered.

Cross modality rules baked into the flow: the negative or unknown or idle class is
scaffolded for every modality, and the train and test split is by capture session,
not by random row, so the same subject or recording does not leak across the split
and inflate the score.

## 7. Grounding in the tensorflow-web API

The studio only uses what the library actually provides, and surfaces its
constraints honestly.

- Build a `tf.LayersModel` from the model builder, using only supported layers
  (Conv2D, DepthwiseConv2D, Dense, MaxPooling2D, AveragePooling2D,
  GlobalAveragePooling2D, Flatten, Reshape, Softmax, Activation, ReLU, Add). An
  unsupported layer fails loud on export with the offending class named; the layer
  editor mirrors this and refuses to add an unsupported operator.
- `train(model, { data, epochs, batchSize, onEpoch, onBatch, signal })` drives the
  Train stage. `onEpoch` and `onBatch` feed the gauges and curve; `signal` powers
  cancel.
- `calibrate(model, representativeTensors)` and `quantize(model, calibration)`
  produce the int8 model; `toTFLite` produces the bytes.
- `verify(tflite, model, testData, { tolerance })` returns parity, max absolute
  error, and the float and int8 accuracy used for the delta and the confusion
  matrix. Verify needs the WASM interpreter, loaded as a global from the alpha.8
  CDN with the WASM path set, exactly as the example and parity harness do.
- Live inference loads the emitted `.tflite` into the interpreter and runs it on
  captured input for the Test stage.

Honest limits surfaced in the interface and the README: CNN and MLP only, no
recurrent or transformer models, no batch normalization, classification first,
int8 post training quantization only (no quantization aware training yet), small
models, and text limited to bag of words. The on device path uses standard TFLite
Micro through the generated C array and snippet; it is op compatible and arena
estimated, but not yet flashed to hardware by this project.

## 8. Differentiators we can actually build

All of these are achievable with the current library and browser APIs:

- A loud local only badge and the fact that data never leaves the browser.
- The float versus int8 accuracy delta, taken directly from the verify report
  (float accuracy and int8 accuracy), shown side by side before export.
- A device fit meter: flash size is the exact `.tflite` byte count; the runtime
  arena is estimated from the model (a heuristic sum of the largest concurrent
  activation tensors, refined by the interpreter where it exposes the arena), shown
  against a selectable target (ESP32, ESP32 S3, generic Cortex M) as green or red.
- A C array export generated in the browser from the `.tflite` bytes (a standard
  byte to array formatting), plus a ready TFLite Micro sketch with the op resolver
  and the arena size pre filled. This makes the train to ESP32 story real without
  any library change.
- Verified parity against the real interpreter, a trust signal no competitor shows.
- Capture coaching overlays per modality, and the scaffolded negative class, so the
  documented best practices are built into the flow rather than left in docs.
- Leakage aware split by capture session.

## 9. Tech stack

- Vue 3 with the Composition API and `<script setup>`.
- Vite for the dev server and build.
- TypeScript, strict.
- Pinia for state, Vue Router for the stage routes. Both are small and official.
- Peer and runtime: `@tensorflow/tfjs` for training, the local `tensorflow-web`
  source for the pipeline, `@tensorflow/tfjs-tflite` (loaded as a CDN global) for
  verify and live inference.
- Browser APIs: getUserMedia for camera and microphone, the Web Audio API and a
  small FFT for spectrograms, DeviceMotion for the accelerometer, Canvas for image
  work, IndexedDB for local persistence.
- A small, well tested FFT dependency for the spectrogram, with the mel filterbank
  and MFCC and DCT hand written and unit tested against known values. We do not
  reinvent the FFT; we keep the rest minimal and lightweight.
- Charts are hand drawn SVG (a loss curve and the design system gauges), no heavy
  chart library.
- The VISE design system implemented as Vue components, no third party UI kit.

## 10. Architecture and folder layout

The studio is a self contained Vite and Vue project in `studio/` with its own
`package.json`, so it never enters the library build. The library publishes only
`dist`, so `studio/` is excluded already; the separate package makes the boundary
explicit and keeps Vue out of the library's dependencies. During development the
studio imports the library through a Vite alias to `../src/index.ts`; it can be
switched to the published package.

```
studio/
  package.json            own deps; never published with the library
  vite.config.ts          alias tensorflow-web -> ../src/index.ts
  tsconfig.json
  index.html
  README.md               how to run, architecture, credits
  src/
    main.ts               bootstrap
    App.vue               the bench shell (top bar, nav rail, grip, bench rail)
    router.ts             one route per stage
    design/
      tokens.css          the VISE :root tokens
      base.css            reset, fonts, grain overlay, scrollbars
      components/         one file per design system component
        ViseTopBar.vue ViseNavRail.vue ViseBenchRail.vue ViseGrip.vue
        ViseButton.vue ViseSegmented.vue ViseToggle.vue ViseSlider.vue
        ViseField.vue ViseSelect.vue ViseStatus.vue ViseBadge.vue
        ViseCard.vue ViseGauge.vue ViseHazard.vue ViseSectionHead.vue
        ViseConfusion.vue ViseLossChart.vue ViseIcon.vue
    stages/
      DataView.vue FeaturesView.vue ModelView.vue
      TrainView.vue TestView.vue ExportView.vue
    stores/
      project.ts          modality, classes, samples meta, feature and model config
      training.ts         job state, metrics, artifacts
      settings.ts         altitude level, target device
    composables/
      useCamera.ts useMicrophone.ts useMotion.ts
      useDataset.ts        sample CRUD and IndexedDB persistence
      useInterpreter.ts    load the global tflite, set the WASM path
      useTrainer.ts        wraps train, emits progress, supports cancel
      useQuantizer.ts      calibrate, quantize, toTFLite
      useVerifier.ts       verify, build the delta and confusion matrix
      useLiveInference.ts  run the tflite on live capture
      useExport.ts         download tflite, C array, metadata, snippet
      useDeviceBudget.ts   flash and arena estimate versus target
    features/             pure, testable extractors per modality
      image/ audio/ motion/ text/
    models/
      builder.ts           presets and a constrained custom builder
      presets.ts           per modality default architectures
    presets/
      index.ts             task presets: image classifier, keyword spotter,
                           gesture, text intent, sound classifier
    lib/
      dsp.ts               fft wrapper, mel filterbank, mfcc, dct
      tensors.ts           tensor helpers
      cformat.ts           bytes to C array and the TFLM snippet
      storage.ts           IndexedDB wrapper
      split.ts             session aware train and test split
      format.ts            number and size formatting
    types.ts
  test/                   vitest unit tests for features, dsp, builder, split,
                          cformat, presets
```

Separation of concerns: the design system knows nothing about machine learning;
the feature extractors are pure functions with no Vue; the composables are the only
place that touches the library and the browser devices; the stores hold state; the
views compose. No file owns more than one job, and nothing is monolithic.

## 11. State and data model

Pinia stores:
- project: the active modality and preset, the class list including the negative
  class, sample metadata grouped by capture session, the feature config, and the
  model config.
- training: the job status, live metrics history, the trained model handle, the
  emitted bytes, the verify report, and the device budget result.
- settings: the altitude level and the selected target device.

Data model:
- A Sample has an id, a class id, a capture session id, a created time, the raw
  payload (image blob, audio buffer, motion window, or text), and an optional
  cached feature tensor. Storing the raw payload lets features be recomputed when
  the feature config changes.
- A Dataset is the set of samples plus the class list. The session aware split
  assigns whole sessions to train or test.
- Everything persists to IndexedDB so a project survives a refresh, and can be
  exported to and imported from a single local file.

## 12. Persistence and privacy

All data stays in the browser in IndexedDB. There is no network call for data, no
upload, and no account. The only network use is loading the library peer scripts
and the WASM interpreter from a CDN, which carry no user data. A project can be
saved, loaded, exported to a file, and imported, all locally. The local only badge
states this plainly.

## 13. Accessibility and performance

- Keyboard reachable controls, visible focus rings (the design system already
  defines a live focus ring), and aria labels on the readouts.
- Respect prefers reduced motion, which the design system already honors.
- Keep the main thread responsive: run the FFT and feature extraction in a worker
  where it matters, and let training yield to the event loop through the library's
  per batch callbacks.
- Lazy load each stage and the interpreter so the first paint is fast.

## 14. Honest limitations, shown in the interface

- Models are CNN and MLP only; no recurrent, transformer, or batch normalization
  layers. An unsupported layer fails loud with its name.
- Classification first. Regression through an MLP is a later option.
- Int8 post training quantization only; quantization aware training is not in the
  library yet, so the studio shows the accuracy delta and offers the mitigations it
  can (more calibration data, per channel weights, retrain), not QAT.
- Text is bag of words, not embeddings.
- The microcontroller path is op compatible and arena estimated and produces the C
  array and a TFLite Micro snippet, but this project has not flashed it to hardware.

## 15. Rules compliance, from AGENTS.md

- No Claude or AI attribution in any commit, comment, or document.
- No emojis. Icons are inline SVG, matching the design system. No emoji used as an
  icon.
- No em-dashes, ever. No double hyphen substitutes.
- No typical AI or large language model phrasing.
- Smart separation of concerns, no monolithic files, as the layout enforces.
- Public functions and composables carry doc comments that say what they do, what
  they return, and what they throw.
- Apache 2.0, inherited from the repository.
- The studio README credits the work it leverages.
- Research was leveraged and is cited where it shaped a decision.
- Do not reinvent the wheel: Vue, Pinia, Vue Router, a proven FFT, and the existing
  library, rather than hand rolled equivalents.
- Best practices: strict TypeScript, the Composition API, accessibility, and a real
  test suite.
- The studio is a separate package and never ships with the library.

## 16. Build phases, each with a hard exit test

1. Scaffold and shell. Stand up the Vue, Vite, TypeScript, Pinia, and Router
   project in `studio/`. Implement `tokens.css`, `base.css`, and the core design
   components (button, segmented, card, status, gauge, field, slider), and the
   bench shell with the top bar, nav rail, jaw gripped work panel, and bench rail.
   Exit: the app runs, renders in the VISE language, and the altitude control
   switches a visible disclosure state.

2. Image, end to end, the flagship. Camera capture, class management with the
   negative class, the dataset in IndexedDB, image features with a live preview,
   the CNN builder, training with live gauges and a loss curve, quantize and
   export, verify with the accuracy delta, the device budget meter, the C array and
   snippet, and live test. Exit: in Guided mode, train a two or three class webcam
   image classifier and download a verified `.tflite`; live test predicts correctly;
   the budget meter reports a fit for an ESP32 S3.

3. Audio, keyword spotting. Microphone capture, the spectrogram features with a
   live preview, the auto Background Noise class, the CNN, and the full loop. Exit:
   train a two or three keyword spotter and confirm it live.

4. Motion and text. Accelerometer gesture with the live trace and the Idle class;
   bag of words text intent with per class metrics. Exit: both loops complete end
   to end and export a verified `.tflite`.

5. Expert depth and polish. The full feature and architecture and quantization
   levers, the operator inspector, the confusion matrix, project save, load,
   export, and import, presets, the accessibility pass, the README and credits, and
   the unit tests. Exit: a non expert and an expert each complete a project
   successfully, the rules checklist passes, and the unit tests for the feature
   extractors, DSP, builder, split, and C formatting pass.

6. Optional stretch, deferable. The Expert architecture search under the device
   budget. Not required for the studio to be complete; pick it up only after phase
   5 lands.

## 17. Testing

- Vitest unit tests for the pure pieces: image resize and normalize, the FFT and
  mel and MFCC against known values, the model builder presets and the unsupported
  operator guard, the session aware split, and the C array formatting.
- A Playwright smoke per modality that drives the browser loop, mirroring the parity
  harness already in the repository.

## 18. Credits, for the studio README

TensorFlow.js and the `tensorflow-web` library; the TensorFlow Lite WASM
interpreter for verification and live inference; the FFT library used for
spectrograms; Vue, Pinia, and Vite; the VISE design system; and an acknowledgement
that the experience design was informed by studying Edge Impulse, Google Teachable
Machine, and Microsoft Lobe, and by the documented best practices in the
TensorFlow, LiteRT, and TinyML literature.

## 19. Decisions, resolved

1. Stage labels are plain words: Data, Features, Model, Train, Test, Export. The
   VISE machine shop character is visual only.
2. Image, audio, and motion lead and are built first. Text, as a bag of words MLP,
   is included alongside them, not deferred, with its limits called out.
3. Transfer learning for image stays out of scope. Any future Expert path for it
   must be flagged clearly that it will likely not fit a microcontroller.
4. The Expert architecture search is a deferable stretch, after phase 5.

## 20. Known gotchas and reference implementations

These were learned while building the library and its existing demos. Reuse them;
do not rediscover them.

- The TFLite interpreter (`@tensorflow/tfjs-tflite`) is a browser WASM build, and
  the alpha.10 npm package omits the WASM binaries. For verify and live inference,
  load the interpreter from the alpha.8 CDN as a global script, and load
  `@tensorflow/tfjs@4.9.0` as a global first because the interpreter wraps its
  outputs with a global tf. Then call `setWasmPath` to the alpha.8 dist before
  verify. The working setup is in `app/index.html`, `app/main.ts`, and
  `examples/cdn.html`.
- Vite's dependency optimizer cannot bundle `@tensorflow/tfjs-tflite`. Exclude it
  with `optimizeDeps.exclude` and rely on the CDN global, as the existing
  `vite.config.ts` does.
- The emitted `.tflite` has a fixed input batch of one. Run the interpreter one
  sample at a time for live inference. The library's `verify` already does this.
- The library trains CNN and MLP from scratch, int8, with float input and output.
  Only twelve layers export: Conv2D, DepthwiseConv2D, Dense, MaxPooling2D,
  AveragePooling2D, GlobalAveragePooling2D, Flatten, Reshape, Softmax, Activation,
  ReLU, Add. The model builder must restrict to these; an unsupported layer fails
  loud on export with its name.
- The studio is its own package in `studio/`. Do not modify the library's `src`,
  `package.json`, or `tsup.config`. The existing test app in `app/` stays as is, it
  backs the parity tests; the studio is additive.
- Reference implementations to read before building: `examples/cdn.html` (the full
  pipeline from a clean entry point), `app/main.ts` and `app/index.html` (capture
  to train to quantize to verify to download), `app/parity.ts` and
  `scripts/run-parity.mjs` (the browser test harness and interpreter setup).
