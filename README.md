# anansi
embedding search and all the associated tooling to embedded AI in your applications, wherever they run.
# commands
cargo nextest run --package base
cd lib/base && wasm-pack test --firefox --headless
in order to build the wasm target, we need to do the following:

## running the web demo
1. cd core/lib/base/wasm && wasm-pack build --target web
2. cd demo && npm ci && npm run start
3. demo will be available @ https://localhost:9000

## pushing the demo to cloudflare
1. cd demo && wrangler pages publish dist --commit-dirty

## Resources
[Faster Population Counts Using AVX2 Instructions](https://arxiv.org/pdf/1611.07612.pdf)