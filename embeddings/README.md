# Embedds

Emebedds is a general-purpose embedding service that converts text and images into multi-dimensional vectors. We are focused on providing turn-key access to embedding models available on the [Massive Text Embedding](https://huggingface.co/spaces/mteb/leaderboard) leaderboard.

## Getting Started
A server is available at embedds.getanansi.com loaded with `M_CLIP_VIT_L_14_336_OPENAI` for testing purposes. We provide HTTPS and gRPC access out of the box; though it is recommended that you utilize a GRPC connection when handling large amounts of text. Here are a couple examples using both HTTPS and gRPC:

<table>
<tr>
<td> via HTTPS 🏄 </td>
<td> via gRPC 🚀 </td>
</tr>
<tr>
<td>

```bash
curl \
-X POST https://embedds.getanansi:50052/encode \
-H 'Content-Type: application/json' \
-H 'Authorization: <your access token>' \
-d '{"data": [{
        "model": "M_CLIP_VIT_L_14_336_OPENAI",
        "text": [
            "3D ActionSLAM: wearable person tracking ...",
            "Tracking early lung cancer metastatic..."
        ],
        "instructions": [
            "Represent the Science title:",
            "Represent the Nature title:"
        ]}
    ]}
'
```
</td>
<td>

```python
# pip install embedds-client
from embedds_client import Client

c = Client(
    'grpcs://embedds.getanansi.com:50051',
    credential={'Authorization': '<access token>'}
)
r = c.encode(
    [
        'The quick brown fox',
        'jumps over the lazy dog',
        'and runs all the way to the finish line',
    ]
)
print(r)
```
</td>
</tr>
</table>

## Getting Started
The easiest way to get started locally is to use one of the docker images we publish here. There are two primary options:
* latest - includes CUDA and libcudnn bindings to support GPU and CPU accelerated inference.
* latest-cpu - a minimial image, lacking CUDA + libcuddn dylibs that allows for **only** CPU inference.

Both options are loaded with envoy, which provides JSON <-> GRPC transcoding. We will include details on building from
source and packaging for even lighter images below.

The list of environment variables that are supported are as follows:
<table>
<tr>
<td>Environment Variable</td>
<td>Usage</td>
</tr>
<tr>
<td>

```EMBEDDS_GRPC_PORT```
</td>
<td><p>port to listen for and server gRPC requests.</td>
</tr>
<tr>
<td>

```EMBEDDS_HTTP_PORT```
</td>
<td><p>port to listen for and server HTTP requests.</td>
</tr>
<tr>
<td>

```EMBEDDS_CONFIG_FILE```
</td>
<td><p>filepath to store the runtime configuration for models - more on this file is available below</p></td>
</tr>
<tr>
<td>

```EMBEDDS_CACHE_FOLDER```
</td>
<td><p>folder in which to store the cached model files - these are typically on the order of ~100s of MBs so it makes sense to keep it on a dedicated volume. </p></td>
</tr>
</table>
