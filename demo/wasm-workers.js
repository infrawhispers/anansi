import { threads } from 'wasm-feature-detect';
import * as Comlink from 'comlink';

async function initHandlers() {
  let [singleThread, multiThread] = await Promise.all([
    (async () => {
      if (!await threads()) {
        console.log("we support threads");
      }
      
      const singleThread = await import('./pkg/wasm.js');
      await singleThread.default();
      let numThreads = 6;
      if (navigator !== undefined && navigator.hardwareConcurrency !== undefined) {
        numThreads = Math.min(6, navigator.hardwareConcurrency);
      }
      await singleThread.initThreadPool(numThreads);
      return singleThread;
    })(),
    (async () => {
      // If threads are unsupported in this browser, skip this handler.
      // if (!(await threads())) return;
      // const multiThread = await import(
      //   '../pkg/wasm.js'
      // );
      // await multiThread.default();
      // await multiThread.initThreadPool(navigator.hardwareConcurrency);
      // return multiThread;
      // return SingleThread;
    })()
  ]);
  return Comlink.proxy({
    singleThread,
    supportsThreads: !!multiThread,
    multiThread
  });
}

Comlink.expose({ handlers: initHandlers() });
