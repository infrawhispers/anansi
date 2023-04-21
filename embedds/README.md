# embedds 🛏

emebedds is a general-purpose embedding service that converts text and images into multi-dimensional vectors. It is focused on providing turn-key access to embedding models available on the [Massive Text Embedding](https://huggingface.co/spaces/mteb/leaderboard) leaderboard.

## Quick Evaluation

A server is available at <b>api.embeddings.getanansi.com</b> loaded with `M_INSTRUCTOR_LARGE` for testing purposes that accepts gRPC requests. Here is an example using grpcurl:

<table>
<tr>
<td> request 🏄 </td>
<td> response 🚀 </td>
</tr>
<tr>
<td>

```bash
# brew install grpcurl
grpcurl -d '{
    "data":[{
        "model":"M_INSTRUCTOR_LARGE",
        "text":[
            "3D ActionSLAM: wearable person tracking ...",
            "Inside Gohar World and the Fine, Fantastical Art"
        ],
        "instructions":[
            "Represent the Science title:",
            "Represent the Magazine title:"
        ]
    }]}' \
    api.embeddings.getanansi.com:50051 api.Api/Encode
```

</td>
<td>

```json
{
    "results": [
        {
            "embedding": [-0.06240629777312279, 0.025188930332660675, ...]
        },
        {
            "embedding": [-0.018718170002102852, -0.03428122401237488, ...]
        }
    ]
}
```

</td>
</tr>
</table>

embedds also provides a HTTP endpoint (via [envoy](https://www.envoyproxy.io/docs/envoy/v1.26.0/)) allowing you to accomplish the above using a simple curl:

```
curl \
-X POST http://host.docker.internal:50052/encode \
-H 'Content-Type: application/json' \
-d '{"data": [{
        "model": "M_INSTRUCTOR_LARGE",
        "text": [
            "3D ActionSLAM: wearable person tracking ...",
            "Inside Gohar World and the Fine, Fantastical Art"
        ],
        "instructions": [
            "Represent the Science title:",
            "Represent the Magazine title:"
        ]}
    ]}
'
```

## Getting Started

### Image & Env Variables

The easiest way to get started locally is to use one of the docker images we publish on DockerHub:

- [latest](https://hub.docker.com/repository/docker/infrawhispers/anansi/tags?page=1&ordering=last_updated&name=embeddings-latest) - includes CUDA and libcudnn bindings to support GPU and CPU accelerated inference.
- [latest-cpu](https://hub.docker.com/repository/docker/infrawhispers/anansi/tags?page=1&ordering=last_updated&name=embeddings-latest-cpu) - a minimial image, lacking CUDA + libcuddn dylibs that allows for **only** CPU inference.

<!-- Both options are loaded with envoy, which provides JSON <-> GRPC transcoding. We will include details on building from
source and packaging for even lighter images below. -->

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
<td><p>port to listen for and server gRPC requests [default=50051]</td>
</tr>
<tr>
<td>

`EMBEDDS_HTTP_PORT`

</td>
<td><p>port to listen for and server HTTP requests [default=50052]</td>
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
</table>

### Config File

The `EMBEDDS_CONFIG_FILE` points to an accessible filepath that stores a list of models that should be instantiated on startup of the process.

<b>If these models are missing, embedds will attempt to download them and store them</b> in the filepath pointed to by `EMBEDDS_CACHE_FOLDER`. An example configuration is outlined below:

```yaml
models:
  # MODEL_NAME must match one of the available models, defined at:
  # https://github.com/infrawhispers/anansi/blob/main/embeddings/proto/api.proto#L27
  - name: M_CLIP_VIT_L_14_336_OPENAI
    # [optional] set to zero or leave empty for parallelism to be determined
    num_threads: 4
    # [optional] enable | disable parallel execution of the onnx graph, which may improve
    # performance at the cost of memory usage.
    parallel_execution: true
  - name: M_INSTRUCTOR_LARGE
  - name: M_INSTRUCTOR_LARGE
```

This configuration would create ONE `CLIP_VIT_L_14_336` and TWO `M_INSTRUCTOR_LARGE` models. This is useful for running multiple embedding models on a single GPU. The list of devices and available models can be found [here](https://github.com/infrawhispers/anansi/blob/main/embeddings/proto/api.proto). By default, embedds will instantiate one instance of [M_INSTRUCTOR_LARGE](https://huggingface.co/hkunlp/instructor-large).

### Getting It Running

We recommend having your `EMBEDDS_CONFIG_FILE` volume mounted in order to speed up startup and prevent repeated downloads. Spinning up a local server can be done via docker:

```bash
docker run -p 0.0.0.0:50051:50051 -p 0.0.0.0:50052:50052 \
		-v $(PWD)/.cache:/app/.cache \
		anansi:embeddings-latest-cpu
```
### API Documentation 