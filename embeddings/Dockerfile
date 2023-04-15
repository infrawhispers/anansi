FROM rust:1.68-slim-buster as planner
RUN apt update && apt-get -y install openssl build-essential libc6 \
    libc-bin musl-tools libssl-dev pkg-config python3 \
    protobuf-compiler
WORKDIR app
# we only pay the installation cost once, 
# it will be cached from the second build onwards
RUN cargo install cargo-chef
COPY . .
RUN cargo chef prepare  --recipe-path recipe.json

FROM rust:1.68-slim-buster as cacher
RUN apt update && apt-get -y install openssl build-essential libc6 \
    libc-bin musl-tools libssl-dev pkg-config python3 \
    protobuf-compiler
WORKDIR app
RUN cargo install cargo-chef
COPY --from=planner /app/recipe.json recipe.json
RUN cargo chef cook --release --recipe-path recipe.json
COPY . .
RUN cargo build --release


FROM debian:buster-slim as runtime
RUN apt update && apt-get -y install openssl build-essential libc6 \
    libc-bin musl-tools ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR app
RUN mkdir .cache
COPY --from=cacher /app/target/release/embedding-server embedding-server
COPY --from=cacher /app/config.yaml config.yaml
# copy in the libonnxruntime* dylibs that we need. 
COPY --from=cacher /app/target/release/libonnxruntime* /usr/lib/
EXPOSE 50051
ENTRYPOINT ["./embedding-server", "-c", "config.yaml"]