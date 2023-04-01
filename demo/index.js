import ReactDOM from 'react-dom'
import React from 'react'
import * as Comlink from 'comlink';
import { downloadBlobWithProgress, makeTextFileLineIterator, cosineSimilarity } from "./utils";
import Tokenizer from './clip_bpe/mod';

const MODEL_NAME = "clip_vit_32"
// REDDIT_EMBEDDS_URI = "https://huggingface.co/datasets/rocca/top-reddit-posts/resolve/main/clip_embeddings_top_50_images_per_subreddit.tsv.gz"
const REDDIT_EMBEDDS_URI = "https://d1wz516niig2xr.cloudfront.net/reddit_clip_embedds.tsv.gz"

const EMBEDD_DIMENSIONS = 512;
const MAX_NUM_OF_EMBEDDS = 150000;
const NUM_IMAGE_WORKERS = 2;

let onnxImageSessions = [];
let onnxTextSession;
let imageWorkers = [];
let vips;
let ann_idx;
let directoryHandle;
let path_by_eid = {};


async function getFileHandleByPath(path) {
    let handle = directoryHandle;
    let chunks = path.split("/").slice(1);
    for (let i = 0; i < chunks.length; i++) {
        let chunk = chunks[i];
        if (i === chunks.length - 1) {
            handle = await handle.getFileHandle(chunk);
        } else {
            handle = await handle.getDirectoryHandle(chunk);
        }
    }
    return handle;
}

async function bicubicResizeAndCenterCrop(blob) {
    let im1 = vips.Image.newFromBuffer(await blob.arrayBuffer());
    // Resize so smallest side is 224px:
    const scale = 224 / Math.min(im1.height, im1.width);
    let im2 = im1.resize(scale, { kernel: vips.Kernel.cubic });
    // crop to 224x224:
    let left = (im2.width - 224) / 2;
    let top = (im2.height - 224) / 2;
    let im3 = im2.crop(left, top, 224, 224)
    let outBuffer = new Uint8Array(im3.writeToBuffer('.png'));
    im1.delete(), im2.delete(), im3.delete();
    return new Blob([outBuffer], { type: 'image/png' });
}
async function getRgbData(blob) {
    // let blob = await fetch(imgUrl, {referrer:""}).then(r => r.blob());
    let resizedBlob = await bicubicResizeAndCenterCrop(blob);
    let img = await createImageBitmap(resizedBlob);
    let canvas = new OffscreenCanvas(224, 224);
    let ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0);
    let imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    let rgbData = [[], [], []]; // [r, g, b]
    // remove alpha and put into correct shape:
    let d = imageData.data;
    for (let i = 0; i < d.length; i += 4) {
        let x = (i / 4) % canvas.width;
        let y = Math.floor((i / 4) / canvas.width)
        if (!rgbData[0][y]) rgbData[0][y] = [];
        if (!rgbData[1][y]) rgbData[1][y] = [];
        if (!rgbData[2][y]) rgbData[2][y] = [];
        rgbData[0][y][x] = d[i + 0] / 255;
        rgbData[1][y][x] = d[i + 1] / 255;
        rgbData[2][y][x] = d[i + 2] / 255;
        // From CLIP repo: Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        rgbData[0][y][x] = (rgbData[0][y][x] - 0.48145466) / 0.26862954;
        rgbData[1][y][x] = (rgbData[1][y][x] - 0.4578275) / 0.26130258;
        rgbData[2][y][x] = (rgbData[2][y][x] - 0.40821073) / 0.27577711;
    }
    rgbData = Float32Array.from(rgbData.flat().flat());
    return rgbData;
}


const modelData = {
    clip_vit_32: {
        image: {
            modelURI: (quantized) => `https://huggingface.co/rocca/openai-clip-js/resolve/main/clip-image-vit-32-${quantized ? "uint8" : "float32"}.onnx`,
            embed: async function (blob, session) {
                let rgbData = await getRgbData(blob);
                const feeds = { input: new ort.Tensor('float32', rgbData, [1, 3, 224, 224]) };
                const results = await session.run(feeds);
                const embedVec = results["output"].data; // Float32Array
                return embedVec;
            }
        },
        text: {
            modelURI: (quantized) => `https://huggingface.co/rocca/openai-clip-js/resolve/main/clip-text-vit-32-${quantized ? "uint8" : "float32-int32"}.onnx`,
            embed: async function (text, session) {
                if (!window.textTokenizerClip) {
                    // let Tokenizer = (await import("https://deno.land/x/clip_bpe@v0.0.6/mod.js")).default;
                    window.textTokenizerClip = new Tokenizer();
                }
                let textTokens = window.textTokenizerClip.encodeForCLIP(text);
                textTokens = Int32Array.from(textTokens);
                const feeds = { input: new ort.Tensor('int32', textTokens, [1, 77]) };
                const results = await session.run(feeds);
                return [...results["output"].data];
            },
        },
    }
};

function getMLModels(target, updater) {
    if (target == "image") {
        let imageOnnxBlobPromise = downloadBlobWithProgress(modelData[MODEL_NAME].image.modelURI(false), function (e) {
            let ratio = e.loaded / e.total;
            updater({ "val": ratio, "mbs": Math.round(ratio * e.total / 1e6) + " MB" });
        });
        return Promise.all([imageOnnxBlobPromise])
    } else {
        let textOnnxBlobPromise = downloadBlobWithProgress(modelData[MODEL_NAME].text.modelURI(false), function (e) {
            let ratio = e.loaded / e.total;
            updater({ "val": ratio, "mbs": Math.round(ratio * e.total / 1e6) + " MB" });
        });
        return Promise.all([textOnnxBlobPromise]);
    }
}


const ModelLoader = () => {
    const [imageMBs, setImageMbs] = React.useState({ val: 0, mbs: "0 MB" });
    const [textMBs, setTextMbs] = React.useState({ val: 0, mbs: "0 MB" });
    const [pending, setPending] = React.useState(false);
    const initMLModels = async function initMLModels() {
        if (pending) {
            return
        }
        let worker = await Comlink.wrap(
            new Worker(new URL('./wasm-workers.js', import.meta.url), {
                type: 'module'
            })
        ).handlers;
        ann_idx = await new worker.singleThread.Index(EMBEDD_DIMENSIONS, MAX_NUM_OF_EMBEDDS);
        setPending({ image: true, text: true });
        if (imageMBs.val == 100 && textMBs.val == 100) {
            return
        }
        let openRequest = indexedDB.open("anansi-demo", 1);
        openRequest.onupgradeneeded = function (event) {
            const db = event.target.result;
            const objectStore = db.createObjectStore("models", { keyPath: "id" });
            console.log("[anansi] created a new IndexedDB database");
        };
        openRequest.onerror = function () {
            console.error("Error", openRequest.error);
        };
        openRequest.onsuccess = function () {
            let db = openRequest.result;
            const transaction = db.transaction(["models"], "readwrite");
            transaction.onerror = (event) => {
                console.log(event);
            }
            const modelStore = transaction.objectStore("models");
            modelStore.get("clip-image").onsuccess = async (event) => {
                let modelBlob;
                if (event.target.result === undefined) {
                    console.log("[anansi] model:image - initating download");
                    let [imageOnnxBlob] = await getMLModels("image", setImageMbs);
                    db.transaction(["models"], "readwrite").objectStore("models").put({ val: imageOnnxBlob, id: "clip-image" });
                    modelBlob = imageOnnxBlob;
                } else {
                    console.log("[anansi] model:image - already have model in IndexedDB");
                    modelBlob = event.target.result.val;
                }
                let imageModelUrl = window.URL.createObjectURL(modelBlob);
                for (let i = 0; i < NUM_IMAGE_WORKERS; i++) {
                    let session = await ort.InferenceSession.create(imageModelUrl, { executionProviders: ["wasm"] });
                    onnxImageSessions.push(session);
                    imageWorkers.push({
                        session,
                        busy: false,
                    });
                }
                console.log("[anansi] model:image - fully loaded.");
                window.URL.revokeObjectURL(imageModelUrl);
                setImageMbs({ val: 100, mbs: 'Loaded' });
                setPending((prevState) => ({
                    ...prevState,
                    image: false,
                }));
            }
            modelStore.get("clip-text").onsuccess = async (event) => {
                let modelBlob;
                if (event.target.result === undefined) {
                    console.log("[anansi] model:text - initating download");
                    let [textOnnxBlob] = await getMLModels("text", setTextMbs);
                    db.transaction(["models"], "readwrite").objectStore("models").put({ val: textOnnxBlob, id: "clip-text" });
                    modelBlob = textOnnxBlob;
                } else {
                    console.log("[anansi] model:text - already have model in IndexedDB");
                    modelBlob = event.target.result.val;
                }
                let textModelUrl = window.URL.createObjectURL(modelBlob);
                onnxTextSession = await ort.InferenceSession.create(textModelUrl, { executionProviders: ["wasm"] });
                console.log("[anansi] model:text - fully loaded.");
                window.URL.revokeObjectURL(textModelUrl);
                setTextMbs({ val: 100, mbs: 'Loaded' });
                setPending((prevState) => ({
                    ...prevState,
                    text: false,
                }));
            }
        };
        vips = await Vips();
        vips.EMBIND_AUTOMATIC_DELETELATER = false
    }
    const loadedModels = imageMBs.val == 100 && textMBs.val == 100;
    const opacity = loadedModels ? "0.5" : "1";
    return (
        <div id="initCtnEl" style={{ "padding": "0.5rem", "background": "lightgrey", "margin": "0.5rem", "opacity": opacity }}>
            <b>Step 1:</b> Download and initialize the models.
            <p>
                Image Model Progress: <progress value={imageMBs['val']}></progress> <span
                >{imageMBs['mbs']}</span>
            </p>
            <p>
                Text Model Progress: <progress value={textMBs['val']}></progress> <span
                >{textMBs['mbs']}</span>
            </p>
            <div>
                <div style={{ display: "inline-block", paddingRight: "0.5rem" }} >
                    <button disabled={loadedModels || pending.image || pending.text} onClick={() => { initMLModels() }}>Initialize
                        Models</button>
                </div>
            </div>
        </div>
    );
}

const ImageLoader = ({ setDataSource }) => {
    const [selection, setSelection] = React.useState(undefined);
    const [embeddStatus, setEmbeddStatus] = React.useState({ imagesEmbedded: 0, totalEmbeddingsCount: 0, imagesBeingProcessedNow: 0 });
    const [indexStatus, setIndexStatus] = React.useState({ timeTaken: "-", status: undefined, numIndexed: "-" });
    const [loaded, setLoaded] = React.useState(false);
    const [embeddsMBs, setEmbeddsMB] = React.useState({ val: 0, mbs: "0 MB" })
    const opacity = loaded ? "0.5" : "1";

    const insertEmbedds = async (embedds) => {
        let eidCnt = 0;
        let eids = [];
        let vecData = [];
        for (let [filePath, embeddingVec] of Object.entries(embedds)) {
            path_by_eid[eidCnt.toString()] = filePath;
            eids.push(eidCnt.toString());
            vecData.push(...embeddingVec);
            eidCnt += 1
        }
        setIndexStatus((prevState) => ({ ...prevState, status: "INDEXING" }));
        var startTime = performance.now();
        await ann_idx.insert(eids, vecData);
        var endTime = performance.now()
        setIndexStatus((prevState) => ({ ...prevState, status: "DONE", numIndexed: eidCnt, timeTaken: `${(endTime - startTime).toFixed(3)}ms` }));
    }

    const insertEmbeddsFrFile = async (file, opts) => {
        let eidOffset = 0;
        let eidCnt = 0;
        let eids = [];
        let vecData = [];
        let totalTimeSpent = 0;
        // chrome does *not* like a massive array of 100000 items!
        // so let us batch things!
        const batchSize = 50000;
        setIndexStatus((prevState) => ({ ...prevState, status: "INDEXING" }));
        for await (let line of makeTextFileLineIterator(file, opts)) {
            if (eidCnt >= MAX_NUM_OF_EMBEDDS) {
                break
            }
            if (!line || !line.trim()) continue;
            let [filePath, embeddingVec] = line.split("\t");
            path_by_eid[eidCnt.toString()] = filePath;
            eids.push(eidCnt.toString());
            vecData.push(...JSON.parse(embeddingVec));
            // we index in bathces since chrome doesn't like the massive arrays!
            if (eids.length >= batchSize) {
                var startTime = performance.now();
                await ann_idx.insert(eids, vecData);
                var endTime = performance.now()
                totalTimeSpent += endTime - startTime;
                eidOffset += eids.length;
                eidCnt = eidOffset;
                eids = [];
                vecData = [];
                setIndexStatus((prevState) => ({ ...prevState, numIndexed: eidOffset }));
            }
            eidCnt += 1;
        }
        if (eids.length != 0) {
            var startTime = performance.now();
            await ann_idx.insert(eids, vecData);
            var endTime = performance.now()
            totalTimeSpent += endTime - startTime;
        }
        setIndexStatus((prevState) => ({ ...prevState, status: "DONE", numIndexed: eidCnt, timeTaken: `${totalTimeSpent.toFixed(3)}ms` }));
    }

    const loadOrPopulateFrReddit = async function () {
        let openRequest = indexedDB.open("anansi-demo", 1);
        openRequest.onupgradeneeded = function (event) {
            const db = event.target.result;
            const objectStore = db.createObjectStore("models", { keyPath: "id" });
        };
        openRequest.onerror = function () {
            console.error("Error", openRequest.error);
        };
        openRequest.onsuccess = function () {
            let db = openRequest.result;
            const transaction = db.transaction(["models"], "readwrite");
            transaction.onerror = (event) => {
                console.log(`[anansi] error creating the IndexedDb database: ${event}`);
            }
            let opts = { decompress: "gzip" };
            const modelStore = transaction.objectStore("models");
            modelStore.get("reddit-embedds").onsuccess = async (event) => {
                if (event.target.result === undefined) {
                    let redditEmbeddingsBlob = await downloadBlobWithProgress(REDDIT_EMBEDDS_URI, function (e) {
                        let ratio = e.loaded / e.total;
                        setEmbeddsMB({ "val": ratio, "mbs": Math.round(ratio * e.total / 1e6) + " MB" });
                    });
                    db.transaction(["models"], "readwrite").objectStore("models").put({ val: redditEmbeddingsBlob, id: "reddit-embedds" });
                    setLoaded(true);
                    await insertEmbeddsFrFile(redditEmbeddingsBlob, opts);
                } else {
                    setEmbeddsMB({ "val": 100, "mbs": "Loaded Fr Cache" });
                    setLoaded(true);
                    await insertEmbeddsFrFile(event.target.result.val, opts);
                }
            }
        }
    }

    const computeImageEmbeddings = async (dirHandle) => {
        let embeddings = {};
        let recentEmbeddingTimes = [];
        try {
            await recursivelyProcessImagesInDir(dirHandle, embeddings, recentEmbeddingTimes);
        } catch (e) {
            console.error(e);
            alert(e.message);
        }
        setLoaded(true);
        await insertEmbedds(embeddings);
    }


    const sleep = ms => new Promise(res => setTimeout(res, ms));
    const recursivelyProcessImagesInDir = async (dirHandle, embeddings, recentEmbeddingTimes, currentPath = "") => {
        let processors = [];
        for await (let [name, handle] of dirHandle) {
            let path = `${currentPath}/${name}`;
            if (handle.kind === 'directory') {
                await recursivelyProcessImagesInDir(handle, embeddings, recentEmbeddingTimes, path,);
            } else {
                let isImage = /\.(png|jpg|jpeg|JPG|JPEG|webp)$/.test(path);
                if (!isImage) continue;
                while (imageWorkers.filter(w => !w.busy).length === 0) await sleep(1);
                let worker = imageWorkers.filter(w => !w.busy)[0];
                worker.busy = true;
                setEmbeddStatus((prevState) => ({ ...prevState, imagesBeingProcessedNow: prevState.imagesBeingProcessedNow + 1 }));
                processors.push((async function () {
                    let startTime = Date.now();
                    let blob = await handle.getFile();
                    const embedVec = await modelData[MODEL_NAME].image.embed(blob, worker.session);
                    embeddings[path] = [...embedVec];
                    worker.busy = false;
                    setEmbeddStatus((prevState) => ({ ...prevState, imagesEmbedded: prevState.imagesEmbedded + 1 }));
                    setEmbeddStatus((prevState) => ({ ...prevState, totalEmbeddingsCount: prevState.totalEmbeddingsCount + 1 }));
                    recentEmbeddingTimes.push(Date.now() - startTime);
                    if (recentEmbeddingTimes.length > 100) recentEmbeddingTimes = recentEmbeddingTimes.slice(-50);
                    if (recentEmbeddingTimes.length > 10) computeEmbeddingsSpeedEl.innerHTML = Math.round(recentEmbeddingTimes.slice(-20).reduce((a, v) => a + v, 0) / 20);
                    setEmbeddStatus((prevState) => ({ ...prevState, imagesBeingProcessedNow: prevState.imagesBeingProcessedNow - 1 }));
                })());
            }
        }
        // wait on processing for all files in this subdirectory - this is really a block
        // (since we cannot proceed lower) but ensures that we process everything!
        await Promise.all(processors);
    }

    const loadOrPopulateFrFolder = async function () {
        if (!window.showDirectoryPicker) return alert("Your browser does not support some modern features (specifically, File System Access API) required to use this web app. Please try updating your browser, or switching to Chrome, Edge, or Brave.");
        directoryHandle = await window.showDirectoryPicker();
        await computeImageEmbeddings(directoryHandle);
    }

    const handleSelection = async (newVal) => {
        if (selection !== undefined) {
            return
        }
        setSelection(newVal);
        if (newVal == "REDDIT") {
            await loadOrPopulateFrReddit();
            setDataSource("REDDIT");
        } else {
            await loadOrPopulateFrFolder();
            setDataSource("DIRECTORY");
        }
    }

    return (
        <>
            <div style={{ "padding": "0.5rem", "background": "lightgrey", "margin": "0.5rem", "opacity": opacity }}>
                <p><b>Step 2a:</b> Pick a directory of images to use or default to ~150k Reddit Images.</p>
                <p><b><i>If you select a directory, no images you use will leave your device - <a href="https://github.com/infrawhispers/anansi/tree/main/demo">code is here</a> [feel free to run it yourself! :)]</i></b></p>
                <button disabeld={loaded} onClick={() => { handleSelection("REDDIT") }}>Use Reddit Images</button>
                <span style={{ "paddingRight": "0.5rem" }}></span>
                <button disabled={loaded} onClick={() => { handleSelection("DIRECTORY") }}>Use Directory Images </button>
                {selection === "REDDIT" &&
                    <>
                        <br />
                        <br />
                        Reddit Download Progress: <progress value={embeddsMBs['val']}></progress> <span
                        >{embeddsMBs['mbs']}</span>
                    </>
                }
                {selection === "DIRECTORY" &&
                    <>
                        <br />
                        <p> Images Embedded:{embeddStatus.imagesEmbedded} | Images Being Processed: {embeddStatus.imagesBeingProcessedNow} </p>
                    </>
                }
            </div>
            <div style={{ "padding": "0.5rem", "background": "lightgrey", "margin": "0.5rem", "opacity": opacity }}>
                <p><b>Step 2b:</b>[automatic] Build an Approximate Nearest Neighbor (ANN) index to make lookups snappy!</p>
                <p>note: this takes around ~70s for 150k Images with numThreads == 6</p>
                <p>perf on M1 Macs is not great due to <a href="https://github.com/GoogleChromeLabs/wasm-bindgen-rayon/issues/16" target="_blank">wasm-bindgen-rayon:16</a>
                    ...which rolls into <a href="https://bugs.chromium.org/p/chromium/issues/detail?id=1228686&q=reporter%3Arreverser%40google.com&can=1" target="_blank">chromium-issue:1228686</a>.
                    We recommend uploading a batch of demo images to play around with and apologize for the hassle.
                </p>
                {indexStatus.status == "INDEXING" && <>
                    <p>Currently Indexing...Num Indexed: {indexStatus.numIndexed}</p><div className='loader'></div>
                </>}

                {indexStatus.status == "DONE" && <>
                    <p>Num Indexed: {indexStatus.numIndexed} | Total Time Taken: {indexStatus.timeTaken}</p>
                </>}
            </div>
        </>
    );
}

const Searcher = ({ dataSource }) => {
    const [results, setResults] = React.useState([]);
    const [query, setQuery] = React.useState('');
    const [latency, setLatency] = React.useState({ embedd: "-", search: "-" });
    const show = true;
    const opacity = show ? "1" : "0.5"

    const issueSearch = async () => {
        let startEmbeddTime = performance.now();
        let embedd = await modelData[MODEL_NAME].text.embed(query, onnxTextSession);
        let endEmbeddTime = performance.now();
        let startSearchTime = performance.now();
        let nns = await ann_idx.search(embedd, 50);
        let endSearchTime = performance.now();
        setLatency({
            embedd: `${(endEmbeddTime - startEmbeddTime).toFixed(3)}ms`,
            search: `${(endSearchTime - startSearchTime).toFixed(3)}ms`,
        })
        let result = [];
        if (dataSource === "REDDIT") {
            for (let i = 0; i < nns.length; i++) {
                let nn = nns[i];
                let path = path_by_eid[nn.eid];
                if (path !== undefined) {
                    let imageUrl = `./fetcher?uri=${path.split("__")[1]}`;
                    let postUrl = `https://reddit.com/comments/${path.split("__")[0].split("/")[1]}`;
                    result.push(
                        <a href={postUrl} target="_blank" key={i}>
                            <img
                                src={imageUrl}
                                onload="this.style.height='';this.style.width='';this.style.border='';"
                                style={{ "maxHeight": "400px", "height": "300px", "width": "300px", "border": "1px solid black" }}
                                title={`${path}: ${nn.distance}`}
                                loading="lazy" />
                        </a>);
                }
            }
        } else {
            for (let i = 0; i < nns.length; i++) {
                let nn = nns[i];
                let path = path_by_eid[nn.eid];
                if (path !== undefined) {
                    let handle = await getFileHandleByPath(path);
                    let url = URL.createObjectURL(await handle.getFile());
                    result.push(<img key={i} src={url} style={{ "maxHeight": "400px" }} title={`${path}: ${nn.distance}`} loading="lazy" />);
                }
            }
        }
        setResults(result)
    }
    return (
        <>
            <div style={{ "padding": "0.5rem", "background": "lightgrey", "margin": "0.5rem", "opacity": opacity }}>
                <p><b>Step 3:</b> Enter a search term, we will yield the nearest 50 results</p>
                <div>
                    <div style={{ "display": "inline-block", "paddingRight": "0.5rem" }}>
                        <input onChange={(e) => setQuery(e.target.value)} value={query} ></input>
                    </div>
                    <div style={{ "display": "inline-block" }}>
                        <button disabled={!show} onClick={() => { issueSearch() }}>Search</button>
                    </div>
                </div>
                {latency.embedd != "-" && (
                    <p>Embedd Time: {latency.embedd} | Search Time: {latency.search}</p>
                )}
            </div>
            {results}
        </>
    )
}
const App = () => {
    const [dataSource, setDataSource] = React.useState("");
    return (<div>
        <h1 style={{ "font-size": "1rem" }}>CLIP Image + ANN Search In Your Browser!</h1>
        <p>This small demo allows you to search a batch of images using <a href="https://openai.com/research/clip"
            target="_blank">OpenAI's CLIP model</a> via the <a href="https://onnxruntime.ai/docs/tutorials/web/"
                target="_blank">ONXX</a> runtime.
        </p>
        <p>Model translation and embeddings generation was pulled from <a
            href="https://github.com/josephrocca/clip-image-sorter" target="_blank">clip-image-sorter</a>.
            We have added an implementation of DiskANN (in Rust, compiles down to WASM) <a
                href="https://github.com/infrawhispers/anansi" target="_blank">here</a>, which allows us to speed up the
            search speed of
            the nearest neighbors.</p>
        <ModelLoader />
        <ImageLoader setDataSource={setDataSource} />
        <Searcher dataSource={dataSource} />
    </div>
    );
}

(async function init() {
    if (window.Worker) {
        console.log("[anansi] web: we support web workers");
    } else {
        alert("This demo requires web workers, please use a recent version of Chrome, Edge or Brave.")
    }

    const root = ReactDOM.createRoot(document.getElementById('root'));
    root.render(<><App /></>);
})();

