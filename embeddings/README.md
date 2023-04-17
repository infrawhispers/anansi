# Embedding Server


head -c 50M </dev/urandom >download.bin
--http-probe-cmd-file=probe.json
cargo run -- -a 0.0.0.0 -f .cache -c runtime/config.yaml