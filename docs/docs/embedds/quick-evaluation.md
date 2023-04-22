---
sidebar_position: 1
---

# Quick Evaluation

A server is available at <b>api.embeddings.getanansi.com</b> loaded with `INSTRUCTOR_LARGE` for testing purposes - it accepts gRPC requests. Here is an example using grpcurl:

<table>
<tr>

</tr>
<tr>
<td>

```bash
# brew install grpcurl
grpcurl -d '{
    "data":[{
        "model_name":"INSTRUCTOR_LARGE",
        "model_class": "ModelClass_INSTRUCTOR",
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
</tr>
<tr>
<td>

```json
{
    "results": [
        {
            "embedding": [
                -0.06240629777312279,
                0.025188930332660675,
                ...,
            ]
        },
        {
            "embedding": [
                -0.018718170002102852,
                -0.03428122401237488,
                ...
            ]
        }
    ]
}
```

</td>
</tr>
</table>

embedds also provides a HTTP endpoint (via envoy) allowing you to accomplish the above using a call to `curl`:

<table>
<tr>
<td>

```bash
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

</td>
</tr>
</table>
