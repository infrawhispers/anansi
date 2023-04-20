# Embedds

Emebedds is a general-purpose embedding service that converts text and images into multi-dimensional vectors. We are focused on providing turn-key access to embedding models available on the [Massive Text Embedding](https://huggingface.co/spaces/mteb/leaderboard) leaderboard.

## Getting Started
A server is available at api.embeddings.getanansi.com loaded with `M_CLIP_VIT_L_14_336_OPENAI` for testing purposes that accepts gRPC requests. Here is an example using grpcurl: 

<table>
<tr>
<td> grpcurl 🏄 </td>
<td> response 🚀 </td>
</tr>
<tr>
<td>

```bash
grpcurl -d '{
    "data":[{
        "model":"M_CLIP_VIT_L_14_336_OPENAI",
        "text":[
            "3D ActionSLAM: wearable person tracking ...",
            "Tracking early lung cancer metastatic..."
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
            "embedding": [
                0.2052011638879776,
                -0.1430814117193222,
                -0.12958881258964539,
                0.16942952573299408,
                0.17204207181930542,
                0.20820102095603943,
                -0.31196773052215576,
                0.11654488742351532,
                ...
            ]
        },
        {
            
            "embedding": [
                -0.33970779180526733,
                0.14125438034534454,
                0.06866287440061569,
                -0.10105855762958527,
                -0.6169333457946777,
                0.03621787577867508,
                -0.008955627679824829,
                -0.37703391909599304,
                ...
            ]
        }
    ]
}
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
<td><p>folder in which to store the cached model files - these are typically on the order of ~100s of MBs </p></td>
</tr>
</table>
