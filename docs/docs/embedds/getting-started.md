---
sidebar_position: 2
---

# Getting Started

## Docker Images

The easiest way to get started locally is to use one of the docker images we publish on DockerHub:

- [latest](https://hub.docker.com/repository/docker/infrawhispers/anansi/tags?page=1&ordering=last_updated&name=embeddings-latest) - includes CUDA and libcudnn bindings to support GPU and CPU accelerated inference.
- [latest-cpu](https://hub.docker.com/repository/docker/infrawhispers/anansi/tags?page=1&ordering=last_updated&name=embeddings-latest-cpu) - a minimial image, lacking CUDA + libcuddn dylibs that allows for **only** CPU inference.

Once you have a docker image, you can then simply run it using `docker-compose` or `docker`:

```bash
docker run -p 0.0.0.0:50051:50051 \
           -p 0.0.0.0:50052:50052 \
		   -v $(PWD)/.cache:/app/.cache \
		   anansi:embeddings-latest-cpu
```

With the above setup, you can then send HTTP1 requests to 50051 or issue gRPC requests to 50052.

## Configuration

### Environment Variables

embedds can be configured using environment variables. All environment variables are prefixed with `EMBEDDS_` and are outlined below, along with their effects:

The list of environment variables that are supported are as follows:

<table>
<tr>
<td><b>Environment Variable</b></td>
<td><b>Usage</b></td>
</tr>
<tr>
<td>

`EMBEDDS_GRPC_PORT`

</td>
<td><p>port to listen for and server gRPC requests [default=50051]</p></td>
</tr>
<tr>
<td>

`EMBEDDS_HTTP_PORT`

</td>
<td><p>port to listen for and server HTTP requests [default=50052]</p></td>
</tr>
<tr>
<td>

`EMBEDDS_CONFIG_FILE`

</td>
<td><p>filepath to store the runtime configuration for models - more on this file is available below [default=/app/config.yaml] </p></td>
</tr>
<tr>
<td>

`EMBEDDS_CACHE_FOLDER`

</td>
<td><p>folder in which to store the cached model files - these are typically on the order of ~100s of MBs and can grow to GBs if you bin-pack different types of models [default=/app/.cache] </p></td>
</tr>

<tr>
<td>

`EMBEDDS_ALLOW_ADMIN`

</td>
<td>

<p>

whether or not to honor `Initalize(..)` requests, which load a model into the current ONNX runtime. we recommend that models be loaded once on startup; however, this conveninece is included for experimentation [default=false]

</p>
</td>
</tr>
</table>

### Config Files

<p>
The `EMBEDDS_CONFIG_FILE` points to an accessible filepath that stores a list of models that should be instantiated on startup of the process.

<b>
If these models are missing, embedds will attempt to download them and store them</b>

in the filepath pointed to by `EMBEDDS_CACHE_FOLDER`. An example configuration is outlined below:

</p>

```yaml
models:
  # class must match one of the available models, defined at:
  # https://github.com/infrawhispers/anansi/blob/main/embeddings/proto/api.proto
  - name: VIT_L_14_336_OPENAI
    class: ModelClass_CLIP
    # [optional] set to zero or leave empty for parallelism to be determined
    num_threads: 4
    # [optional] enable | disable parallel execution of the onnx graph, which may improve
    # performance at the cost of memory usage.
    parallel_execution: true
  - name: INSTRUCTOR_LARGE
    class: ModelClass_INSTRUCTOR
  - name: INSTRUCTOR_LARGE
    class: ModelClass_INSTRUCTOR
```

This configuration would create ONE `VIT_L_14_336_OPENAI` and TWO `INSTRUCTOR_LARGE` models. This is useful for running multiple embedding models on a single GPU. The list of devices and available models can be found [here](https://github.com/infrawhispers/anansi/blob/main/embeddings/proto/api.proto). By default, embedds will instantiate one instance of [INSTRUCTOR_LARGE](https://huggingface.co/hkunlp/instructor-large).

## Issuing Requests

With a running server, you can now issue requests against the server. Take a look at the <a href="/swagger-api/embedds" target="_blank">swagger-api</a> docs for the HTTP methods available to you.
<br/>
You can also pull the <a href="https://github.com/infrawhispers/anansi/blob/main/embedds/proto/api.proto" target="_blank">proto definition</a> from source to build your grpc client. Native clients are on the roadmap, we are also open for pull-requests 🤩.
