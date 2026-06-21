# TF Web Studio

A browser native, local first studio for training small models and shipping a
verified int8 `.tflite` for edge devices. It collects data, trains a small model,
tests it live, and exports a real `.tflite` plus a ready to flash C array, a
TFLite Micro sketch, and a self-contained bundle you can replay in the
Playground, all in the browser. Nothing is uploaded, no account is
needed, and the output is checked against the float reference in the actual
TFLite interpreter before it is trusted.

TF Web Studio is a separate application built on the headless
[`tensorflow-web`](../README.md) library. It is never published with the library.

## The one honest constraint

Training is general, but export is bounded by the library's operator registry.
The studio trains small convolutional and dense models from scratch, quantizes
them to int8 by post training quantization, and serializes a real `.tflite`. Only
these twelve layers export: Conv2D, DepthwiseConv2D, Dense, MaxPooling2D,
AveragePooling2D, GlobalAveragePooling2D, Flatten, Reshape, Softmax, Activation,
ReLU, and Add. The model builder restricts itself to these, and an unsupported
layer fails loudly on export with its name. There is no recurrent, transformer,
or batch normalization support, and text is bag of words, not embeddings.

## Run it

```
cd studio
npm install
npm run dev
```

The verify and live inference steps load the TFLite WASM interpreter from a CDN
(`@tensorflow/tfjs` and `@tensorflow/tfjs-tflite`, see `index.html`), so those two
steps need network access. Collecting data, training, calibrating, quantizing,
and exporting all work offline.

## What it does

Four modalities, each with capture, a feature stage, a model, and the same export
path:

- Image: webcam burst, file import, or a draw pad, resized grayscale, a small CNN.
- Audio: microphone or clip import, a log mel spectrogram, a small CNN. The
  microphone clip length is selectable (1, 2, 4 seconds, or any value); the
  spectrogram spans the whole clip across a fixed grid, so the model stays small.
  Loudness is scaled against a fixed reference, not each clip's own range, so a
  quiet background stays distinguishable from a spoken keyword.
- Motion: device accelerometer or window import, a resampled window, a small MLP.
- Text: typed or pasted strings, a bag of words encoding, a small MLP.

The workflow is a plain spine: Data, then optionally Features and Model, then
Train, Test, and Export. The Data stage shows a clear readiness cue: how many
samples each class still needs and a Train action that lights up when the dataset
clears the bar. The altitude control in the top bar drives progressive
disclosure: Guided hides the configuration stages and chooses everything for you,
Standard shows the main knobs, and Expert opens the operator inspector and the
rest of the levers. The negative class (Neither, Background Noise, Idle, Other)
is a scaffolded first class step for every modality, and the train and test split
is by capture session, never a random row, so the same subject does not leak
across the split.

## Playground

The Playground is a standalone view, off the workflow spine, for running a model
live against fresh camera, microphone, motion, or text input. It loads three
ways: the model you just trained, a self-contained bundle (`.tfwsmodel.json`)
exported from any earlier project, or a bare `.tflite` you configure by hand
(pick the modality, preprocessing, and class labels; a studio bundle carries all
of that for you). Audio listens continuously on a rolling window and image runs
live on every frame, rather than one clip at a time. It is where you confirm an
exported model actually behaves before you flash it, with no project open.

## Privacy

Everything stays in the browser. Samples and the project persist in IndexedDB, a
project can be exported to and imported from a single local file
(`<name>.tfwsproj.json`), and there is no network call for any user data. The only network use is loading the peer scripts
and the WASM interpreter from a CDN, which carry nothing of yours.

## Verification and device fit

The emitted `.tflite` is loaded back into the real TFLite interpreter and checked
against the float reference before it is shown as shippable. Parity is enforced,
not just displayed: if the int8 model does not match the float reference within
tolerance, the artifact is not offered for download and is not loaded for live
inference, and the failure is shown plainly. The float versus int8 accuracy delta
is displayed, never hidden, and a device budget meter reports the exact flash size
and an estimated runtime arena against a selected target (ESP32 S3, ESP32, Cortex
M7, or Cortex M4), green when it fits and red when it does not.

## Architecture

```
src/
  App.vue, main.ts, router.ts   the bench shell and stage routes
  design/                       tokens, base layer, and the studio components
  stages/                       one view per stage
  stores/                       project, training, and settings (Pinia)
  composables/                  the only place that touches the library and the
                                browser devices (camera, microphone, motion,
                                dataset, interpreter, trainer, quantizer,
                                verifier, device budget, live inference, export,
                                playground, and the pipeline orchestrator)
  features/                     pure, testable extractors per modality
  models/                       presets and the constrained, guarded builder
  lib/                          dsp, tensors, storage, split, cformat, format,
                                modelBundle
```

The design system knows nothing about machine learning, the feature extractors
are pure functions with no Vue, the composables own the library and device calls,
the stores hold state, and the views compose. No file owns more than one job.

## Testing

Unit tests (Vitest) cover the pure pieces: image resize and normalization, the
FFT, mel filterbank, and DCT against known values, the model builder presets and
the unsupported operator guard, the session aware split, the C array and sketch
formatting, and the project file and model bundle round trips. A headless Chrome
smoke per modality drives the full browser loop and confirms parity, the int8
accuracy, the device fit, and a correct live prediction; a Playground smoke runs
the current model and a reloaded bundle on a held out image.

```
npm run test            # unit tests
npm run smoke           # the phase 1 shell smoke
node scripts/smoke-image.mjs   # image flow, end to end
node scripts/smoke-audio.mjs   # audio keyword spotting
node scripts/smoke-audio-bg.mjs    # audio: loud keyword vs quiet background
node scripts/smoke-mt.mjs      # motion and text
node scripts/smoke-expert.mjs  # expert depth, project save and load
node scripts/smoke-persist.mjs # reload persistence
node scripts/smoke-playground.mjs  # run the current model and a reloaded bundle
```

## Limitations, stated plainly

- Small CNN and MLP only. No recurrent, transformer, or batch normalization layers.
- Classification first. Regression is a later option.
- Int8 post training quantization only. The studio shows the accuracy delta and
  offers the mitigations it can (more calibration data, per channel weights,
  retrain), not quantization aware training.
- Text is bag of words, not embeddings.
- The microcontroller path is operator compatible and arena estimated and emits
  the C array and a TFLite Micro sketch, but this project has not flashed it to
  hardware.

## Credits

TF Web Studio leverages and is grateful to:

- [TensorFlow.js](https://www.tensorflow.org/js) for model building and training,
  and the `tensorflow-web` library in this repository for calibration, int8
  quantization, and `.tflite` serialization.
- The TensorFlow Lite WASM interpreter (`@tensorflow/tfjs-tflite`) as the parity
  oracle for verification and for live inference.
- [fft.js](https://github.com/indutny/fft.js) for the FFT behind the audio
  spectrograms; the mel filterbank, MFCC, and DCT are hand written and unit tested.
- [Vue](https://vuejs.org), [Pinia](https://pinia.vuejs.org),
  [Vue Router](https://router.vuejs.org), and [Vite](https://vite.dev).
- The studio design system for the visual language.
- Experience design informed by studying Edge Impulse, Google Teachable Machine,
  and Microsoft Lobe, and by the documented best practices in the TensorFlow,
  LiteRT, and TinyML literature.

## License

Apache License 2.0, inherited from the repository. See [`LICENSE`](../LICENSE).
