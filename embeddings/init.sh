#!/bin/bash

# substitute for the envoy-config
CONFIG_FILE="${CONFIG_FILE:-/app/runtime/config.yaml}" \
CACHE_FOLDER="${CACHE_FOLDER:-/app/.cache}" \
GRPC_PORT="${GRPC_PORT:-50051}" \
HTTP_PORT="${HTTP_PORT:-50052}" \
envsubst < runtime/templates/envoy-config.yaml > runtime/envoy-config.yaml;

# substitute for the supervisord.conf
CONFIG_FILE="${CONFIG_FILE:-/app/runtime/config.yaml}" \
CACHE_FOLDER="${CACHE_FOLDER:-/app/.cache}" \
GRPC_PORT="${GRPC_PORT:-50051}" \
HTTP_PORT="${HTTP_PORT:-50052}" \
envsubst < runtime/templates/supervisord.conf > runtime/supervisord.conf;

supervisord -c runtime/supervisord.conf