FROM debian:bullseye-slim as envoy-runtime
RUN rm -rf /var/lib/apt/lists/* && apt update
RUN apt update && apt-get -y install  debian-keyring debian-archive-keyring apt-transport-https \
    curl lsb-release && \
    curl -sL 'https://deb.dl.getenvoy.io/public/gpg.8115BA8E629CC074.key' | gpg --dearmor -o /usr/share/keyrings/getenvoy-keyring.gpg && \
    echo a077cb587a1b622e03aa4bf2f3689de14658a9497a9af2c427bba5f4cc3c4723 /usr/share/keyrings/getenvoy-keyring.gpg | sha256sum --check && \
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/getenvoy-keyring.gpg] https://deb.dl.getenvoy.io/public/deb/debian $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/getenvoy.list
RUN apt update && apt-get -y install getenvoy-envoy


FROM infrawhispers/anansi:base-rust-1.69 as planner
RUN apt update && apt-get -y install openssl build-essential libc6 \
    libc-bin musl-tools libssl-dev pkg-config python3 \
    protobuf-compiler
WORKDIR /app
# we only pay the installation cost once, 
# it will be cached from the second build onwards
RUN cargo install cargo-chef
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

# this allows us to do development in a repetable manner
FROM infrawhispers/anansi:base-rust-1.69 as cacher
RUN apt update && apt-get clean && apt-get -y -f install openssl build-essential libc6 \
    libc-bin musl-tools libssl-dev libvips-dev pkg-config python3 \
    protobuf-compiler supervisor llvm-dev libclang-dev clang
WORKDIR /app
RUN cargo install cargo-chef
COPY --from=planner /app/recipe.json recipe.json
RUN cargo chef cook --release --recipe-path recipe.json
COPY . .
COPY --from=envoy-runtime /usr/bin/envoy /usr/bin/envoy
RUN cargo build --release
# RUN ln -s /app/target/debug/embedding-server /app/embedding-server
RUN cp /app/target/release/libonnxruntime* /usr/lib/

FROM debian:buster-slim as runtime
RUN apt update && apt-get -y install openssl build-essential libc6 \
    libc-bin musl-tools ca-certificates supervisor gettext libvips-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN mkdir .cache && mkdir /var/log/supervisord
COPY --from=cacher /app/target/release/embedding-server embedding-server
COPY --from=cacher /app/init.sh init.sh
# copy in the libonnxruntime* dylibs that we need. 
COPY --from=cacher /app/target/release/libonnxruntime* /usr/lib/
COPY --from=envoy-runtime /usr/bin/envoy /usr/bin/envoy
COPY runtime/ runtime/
# envoy binds the HTTP listener to 50052
# rust binds the HTTP2 listener to 50051
EXPOSE 50051
EXPOSE 50052 
ENTRYPOINT ["./init.sh"]