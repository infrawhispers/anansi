gen-pb:
	protoc -I$(GOOGLEAPIS_DIR) -I ./proto \
		--descriptor_set_out=runtime/api.pb --include_source_info \
		--include_imports ./proto/api.proto 
		
build-n-run:
	docker build . -t anansi-embeddings && \
	docker run -v $(PWD)/.cache:/app/.cache -p 0.0.0.0:50051:50051 \
		-p 0.0.0.0:50052:50052 anansi-embeddings:latest

run:
	docker ps -aq | xargs docker stop && \
	docker run -v $(PWD)/.cache:/app/.cache -p 0.0.0.0:50051:50051 \
		-p 0.0.0.0:50052:50052 anansi-embeddings.slim:latest

run-cpu:
	docker run -p 0.0.0.0:50051:50051 -p 0.0.0.0:50052:50052 \
		-v $(PWD)/.cache:/app/.cache anansi.embeddings:latest-cpu

run-dev:
	docker run -p  0.0.0.0:50051:50051 -p 0.0.0.0:50052:50052 \
		-v $(PWD)/src:/app/src \
		-v $(PWD)/proto:/app/proto \
		-v $(PWD)/.cache:/app/.cache \
		-v $(PWD)/runtime:/app/runtime \
		-it --entrypoint bash anansi-embeddings:cacher 

build-slim-cpu:
	# docker build . -t anansi-embeddings:raw && \
	slim build --http-probe=false --publish-port 0.0.0.0:50051:50051 \
		--publish-port 0.0.0.0:50052:50052 --mount $(PWD)/.cache:/app/.cache --include-path \
		/var/log/supervisord --include-path=/app/runtime \
		--include-path=/bin/bash \
		--include-path=/usr/lib/libonnxruntime.so \
		--include-path=/usr/lib/libonnxruntime.so.1.14.1 \
		--exclude-mounts=true \
		--exclude-pattern=/app/.cache/clip/*/*.onnx \
		--exclude-pattern=/app/.cache/instructor/* \
		--exclude-pattern=/usr/lib/libonnxruntime_providers_cuda.so \
		--exclude-pattern=/usr/lib/libonnxruntime_shared.so \
		--tag anansi.embeddings:latest-cpu \
		--target anansi-embeddings:raw

build-slim:
	slim build --http-probe=false --publish-port 0.0.0.0:50051:50051 \
		--publish-port 0.0.0.0:50052:50052 --mount $(PWD)/.cache:/app/.cache --include-path \
		/var/log/supervisord --include-path=/app/runtime \
		--include-path=/bin/bash \
		--include-path=/usr/lib/libonnxruntime.so \
		--include-path=/usr/lib/libonnxruntime.so.1.14.1 \
		--include-path=/usr/lib/libonnxruntime_providers_cuda.so \
		--include-path=/usr/lib/libonnxruntime_shared.so \
		--exclude-mounts=true \
		--exclude-pattern=/app/.cache/clip/*/*.onnx \
		--exclude-pattern=/app/.cache/instructor/* \
		--tag anansi.embeddings:latest \
		--target anansi-embeddings:raw

