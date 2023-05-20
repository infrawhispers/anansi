gen-pb:
	protoc -I proto/google -I ./proto \
		--descriptor_set_out=runtime/api.pb --include_source_info \
		--include_imports ./proto/api.proto 
gen-docs:
	cd proto && buf generate && \
	swagger2openapi --yaml --outfile swagger/api.swagger.yaml swagger/api.swagger.json

run:
	docker build . -t  anansi:latest && \
	docker run -v $(PWD)/.cache:/app/.cache \
		-p 50051:50051 \
		-p 50052:50052 anansi:latest-slim

push-to-ecr:
	aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.$(REGION).amazonaws.com && \
	docker tag $(IMG) ${AWS_ACCOUNT_ID}.dkr.ecr.$(REGION).amazonaws.com/$(IMG) && \
	docker push ${AWS_ACCOUNT_ID}.dkr.ecr.$(REGION).amazonaws.com/$(IMG)

# build-slim-cpu:
# 	docker build . -t anansi:latest && \
# 	slim build --http-probe=false --publish-port 0.0.0.0:50051:50051 \
# 		--publish-port 0.0.0.0:50052:50052 --mount $(PWD)/.cache:/app/.cache --include-path \
# 		/var/log/supervisord --include-path=/app/runtime \
# 		--include-path=/bin/bash \
# 		--include-path=/usr/lib/libonnxruntime.so \
# 		--include-path=/usr/lib/libonnxruntime.so.1.14.1 \
# 		--exclude-mounts=true \
# 		--exclude-pattern=/app/.cache/clip/*/*.onnx \
# 		--exclude-pattern=/app/.cache/instructor/* \
# 		--exclude-pattern=/usr/lib/libonnxruntime_providers_cuda.so \
# 		--exclude-pattern=/usr/lib/libonnxruntime_shared.so \
# 		--tag anansi:latest-slim \
# 		--target anansi-embeddings:raw

build-slim:
	# docker build . -t anansi:latest-fat && \
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
		--tag anansi:latest-slim \
		--target anansi:latest-fat

push-img:
	docker tag anansi:latest-slim infrawhispers/anansi:latest
	docker push infrawhispers/anansi:latest