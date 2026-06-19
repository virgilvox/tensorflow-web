/**
 * Static descriptors for each modality: the plain label, the icon name, the
 * capture method, the auto chosen feature summary with its one line reason, the
 * default architecture summary, and the capture coaching. The stages read these
 * to render read only auto decisions with a why badge. No Vue, no library.
 */
import type { Modality } from '../types';

export interface ModalityInfo {
  id: Modality;
  label: string;
  /** ViseIcon name. */
  icon: string;
  /** How samples are captured, shown in the Data stage. */
  capture: string;
  /** The auto chosen feature pipeline summary. */
  feature: string;
  /** One line reason for the auto feature choice, shown behind a why badge. */
  featureReason: string;
  /** The auto chosen default architecture summary. */
  model: string;
  /** One line reason for the auto model choice. */
  modelReason: string;
  /** Capture coaching shown during the Data stage. */
  coaching: string[];
  /** The auto offered negative class label. */
  negative: string;
}

export const MODALITIES: Record<Modality, ModalityInfo> = {
  image: {
    id: 'image',
    label: 'Image',
    icon: 'image',
    capture: 'Webcam burst, drag and drop, or a draw pad.',
    feature: 'Resize to 48 by 48, grayscale, normalize 0 to 1.',
    featureReason:
      'A small square keeps the model tiny enough for a microcontroller while still learning shape and texture.',
    model: 'Small CNN: Conv, pool, Conv, pool, flatten, dense, softmax.',
    modelReason:
      'A from scratch CNN suits tiny edge image models. Transfer learning is out of scope; it would not fit an MCU.',
    coaching: [
      'Aim for roughly fifty samples per class.',
      'Vary angle, lighting, and background.',
      'Add a Neither class for everything else.',
    ],
    negative: 'Neither',
  },
  audio: {
    id: 'audio',
    label: 'Audio',
    icon: 'mic',
    capture: 'Microphone, one second clips at sixteen kilohertz.',
    feature: 'Framed STFT to a log mel spectrogram, treated as one channel.',
    featureReason:
      'A mel spectrogram is the standard compact time frequency view that a small CNN can read as an image.',
    model: 'Small CNN over the spectrogram.',
    modelReason: 'A CNN over the spectrogram is the proven keyword spotting baseline at this size.',
    coaching: [
      'Record varied speakers and distances.',
      'Avoid single syllable keywords.',
      'Capture about ten minutes of background noise.',
    ],
    negative: 'Background Noise',
  },
  motion: {
    id: 'motion',
    label: 'Motion',
    icon: 'motion',
    capture: 'Device accelerometer, a one to two second window with a live trace.',
    feature: 'Window the signal, then a reshaped 2D tensor for a small CNN.',
    featureReason:
      'A windowed tensor lets a small CNN pick up the gesture shape across time and axes.',
    model: 'Small CNN or MLP over the window.',
    modelReason: 'Both are tiny; the CNN reads the windowed signal, the MLP reads summary features.',
    coaching: [
      'Perform the gesture continuously.',
      'Vary speed and orientation.',
      'Capture about three minutes per class.',
    ],
    negative: 'Idle',
  },
  text: {
    id: 'text',
    label: 'Text',
    icon: 'text',
    capture: 'Typed or pasted strings, or import a list.',
    feature: 'Standardize, build a capped vocabulary, encode as a bag of words.',
    featureReason:
      'Bag of words suits a small MLP and a tiny export. This is an honest limit: no embeddings or sequence models.',
    model: 'Small MLP over the encoded vector.',
    modelReason: 'An MLP over a fixed length vector is the right size for a bag of words classifier.',
    coaching: [
      'Keep classes balanced.',
      'Watch the per class counts.',
      'Report precision and recall, not only accuracy.',
    ],
    negative: 'Other',
  },
};

/** The modalities in lead order: image, audio, motion, then text. */
export const MODALITY_ORDER: readonly Modality[] = ['image', 'audio', 'motion', 'text'];
