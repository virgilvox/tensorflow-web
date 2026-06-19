/**
 * Studio bootstrap. Wires Pinia and the router into the bench shell and mounts
 * it. The library and the browser devices are touched only from composables, so
 * nothing machine learning specific happens here.
 */
import { createApp } from 'vue';
import { createPinia } from 'pinia';
import { router } from './router';
import App from './App.vue';

import './design/tokens.css';
import './design/base.css';

const app = createApp(App);
app.use(createPinia());
app.use(router);
app.mount('#app');
