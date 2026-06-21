/**
 * One route per workflow stage, in nav rail order. Each stage view is lazy
 * loaded so the first paint stays fast and the interpreter and heavy device code
 * load only when their stage is opened.
 */
import { createRouter, createWebHashHistory, type RouteRecordRaw } from 'vue-router';
import type { StageId } from './types';

export interface StageMeta {
  id: StageId;
  /** Plain word shown in the nav rail and the stage title. No invented jargon. */
  label: string;
  /** Short caption shown under the label in the rail. */
  caption: string;
  /** Hidden in Guided when true. Features and Model are the configuration stages. */
  configuration?: boolean;
}

export const STAGES: readonly StageMeta[] = [
  { id: 'data', label: 'Data', caption: 'collect and label' },
  { id: 'features', label: 'Features', caption: 'raw to tensor', configuration: true },
  { id: 'model', label: 'Model', caption: 'architecture', configuration: true },
  { id: 'train', label: 'Train', caption: 'fit the model' },
  { id: 'test', label: 'Test', caption: 'live and held out' },
  { id: 'export', label: 'Export', caption: 'verify and ship' },
];

const routes: RouteRecordRaw[] = [
  { path: '/', redirect: '/data' },
  {
    path: '/data',
    name: 'data',
    component: () => import('./stages/DataView.vue'),
  },
  {
    path: '/features',
    name: 'features',
    component: () => import('./stages/FeaturesView.vue'),
  },
  {
    path: '/model',
    name: 'model',
    component: () => import('./stages/ModelView.vue'),
  },
  {
    path: '/train',
    name: 'train',
    component: () => import('./stages/TrainView.vue'),
  },
  {
    path: '/test',
    name: 'test',
    component: () => import('./stages/TestView.vue'),
  },
  {
    path: '/export',
    name: 'export',
    component: () => import('./stages/ExportView.vue'),
  },
  {
    // Standalone, outside the staged spine: runs the current model, a bundle, or
    // an uploaded .tflite live, with no project required.
    path: '/playground',
    name: 'playground',
    component: () => import('./stages/PlaygroundView.vue'),
  },
];

export const router = createRouter({
  history: createWebHashHistory(),
  routes,
});
