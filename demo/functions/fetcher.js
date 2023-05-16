export async function onRequest(context) {
    // we need this in order to get around pesky CORS issues
    // as we are being strict to enable web-workers + the reddit
    // image server does not set the necessary headers.
    const { pathname, search } = new URL(context.request.url);
    if (pathname.startsWith("/fetcher")) {
        let uri = search.substring(5)
        return await fetch("https://i.redd.it/" + uri)
    }
    return new Response(err.stack, { status: 500 })
}