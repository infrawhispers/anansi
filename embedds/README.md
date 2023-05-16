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
        "model_name":"INSTRUCTOR_LARGE",
        "model_class":"ModelClass_INSTRUCTOR",
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
        "model_name": "INSTRUCTOR_LARGE",
        "model_class": "ModelClass_INSTRUCTOR",
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

## Documentation
We use docusaurus to generate our documenation, please either refer to the READMEs <a href="https://github.com/infrawhispers/anansi/tree/main/docs/docs/embedds/getting-started.md" target="_blank">here</a> or check out the documentation website.