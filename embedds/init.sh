#!/bin/bash

# substitute for the envoy-config
EMBEDDS_CONFIG_FILE="${EMBEDDS_CONFIG_FILE:-/app/runtime/config.yaml}" \
EMBEDDS_CACHE_FOLDER="${EMBEDDS_CACHE_FOLDER:-/app/.cache}" \
EMBEDDS_GRPC_PORT="${EMBEDDS_GRPC_PORT:-50051}" \
EMBEDDS_HTTP_PORT="${EMBEDDS_HTTP_PORT:-50052}" \
EMEBEDDS_ALLOW_ADMIN="${EMBEDDS_ALLOW_ADMIN:-false}" \
envsubst < runtime/templates/envoy-config.yaml > runtime/envoy-config.yaml;

# substitute for the supervisord.conf
EMBEDDS_CONFIG_FILE="${EMBEDDS_CONFIG_FILE:-/app/runtime/config.yaml}" \
EMBEDDS_CACHE_FOLDER="${EMBEDDS_CACHE_FOLDER:-/app/.cache}" \
EMBEDDS_GRPC_PORT="${EMBEDDS_GRPC_PORT:-50051}" \
EMBEDDS_HTTP_PORT="${EMBEDDS_HTTP_PORT:-50052}" \
EMEBEDDS_ALLOW_ADMIN="${EMBEDDS_ALLOW_ADMIN:-false}" \
envsubst < runtime/templates/supervisord.conf > runtime/supervisord.conf;

supervisord -c runtime/supervisord.conf