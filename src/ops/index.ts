/**
 * Imports every layer converter for its registration side effect. Importing this
 * module ensures the converter registry is populated before a graph is built.
 * It re-exports nothing; the registry is the public surface.
 */
import './conv2d';
import './depthwise';
import './dense';
import './pooling';
import './global-pool';
import './reshape';
import './softmax';
import './activation';
import './add';
