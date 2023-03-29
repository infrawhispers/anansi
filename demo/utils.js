export function downloadBlobWithProgress(url, onProgress) {
    return new Promise((res, rej) => {
        var blob;
        var xhr = new XMLHttpRequest();
        xhr.open('GET', url, true);
        xhr.responseType = 'arraybuffer';
        xhr.onload = function (e) {
            blob = new Blob([this.response]);
        };
        xhr.onprogress = onProgress;
        xhr.onloadend = function (e) {
            res(blob);
        }
        xhr.send();
    });
}
// for filtering the NSFW items our side - we would need to keep a bunch of them
// in memory to do so and since we are _already_ running the index. we will rely
// on people to be reasonable (famous last words)
export function cosineSimilarity(A, B) {
    if (A.length !== B.length) throw new Error("A.length !== B.length");
    let dotProduct = 0, mA = 0, mB = 0;
    for (let i = 0; i < A.length; i++) {
        dotProduct += A[i] * B[i];
        mA += A[i] * A[i];
        mB += B[i] * B[i];
    }
    mA = Math.sqrt(mA);
    mB = Math.sqrt(mB);
    let similarity = dotProduct / (mA * mB);
    return similarity;
}

// Tweaked version of example from here: https://developer.mozilla.org/en-US/docs/Web/API/ReadableStreamDefaultReader/read
export async function* makeTextFileLineIterator(blob, opts = {}) {
    const utf8Decoder = new TextDecoder("utf-8");
    let stream = await blob.stream();

    if (opts.decompress === "gzip") stream = stream.pipeThrough(new DecompressionStream("gzip"));

    let reader = stream.getReader();

    let { value: chunk, done: readerDone } = await reader.read();
    chunk = chunk ? utf8Decoder.decode(chunk, { stream: true }) : "";

    let re = /\r\n|\n|\r/gm;
    let startIndex = 0;

    while (true) {
        let result = re.exec(chunk);
        if (!result) {
            if (readerDone) {
                break;
            }
            let remainder = chunk.substr(startIndex);
            ({ value: chunk, done: readerDone } = await reader.read());
            chunk = remainder + (chunk ? utf8Decoder.decode(chunk, { stream: true }) : "");
            startIndex = re.lastIndex = 0;
            continue;
        }
        yield chunk.substring(startIndex, result.index);
        startIndex = re.lastIndex;
    }
    if (startIndex < chunk.length) {
        // last line didn't end in a newline char
        yield chunk.substr(startIndex);
    }
}
