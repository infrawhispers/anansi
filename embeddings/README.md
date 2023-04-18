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
            "3D ActionSLAM: wearable person tracking in multi-floor environments is the best thing that we can possibly do",
            "3D ActionSLAM: wearable person tracking in multi-floor environments is the best thing that we can possibly do"
        ],
        "instructions": [
            "Represent the Science title:",
            "Represent the Magazine title:"
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
    'grpcs://embedds.getanansi.com:50051', credential={'Authorization': '<your access token>'}
)
r = c.encode(
    [
        'First do it',
        'then do it right',
        'then do it better',
        'https://picsum.photos/200',
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
