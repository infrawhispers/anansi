# anansi 🕸️

<div>
<p align="center">
  <a href="https://discord.gg/xNyytmxrWh" target="_blank">
      <img src="https://img.shields.io/discord/1098724607864864839" alt="Discord">
  </a>
  <a href="" target="_blank">
      <img src="https://img.shields.io/static/v1?label=license&message=Apache 2.0&color=red" alt="License">
  </a> 
</p>
</div>
<p>
anansi is a fully featured content vectorization system aimed at providing the latest
advances in embedding generation, in-domain tuning and vector storage in an easy to use package.
</p>

### Core Features
#### 🏎️ Performance
* Rust implementation of [FreshDiskANN](https://arxiv.org/abs/2105.09613) with support for scalar quantization
* Configurable RocksDB based storage engine
* ONNX runtime support for CUDA accelerated embedding models

#### 🗒️ Developer Experience
* Build indices on unstructured data without worrying about whether or not it is text, image or video
* Support for gRPC and HTTP clients
* Single installation binary that can cross-compile to non-Linux targets

#### 💡 Machine Learning
* Utilize cutting-edge embeddings models that are listed on the [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
* Bin-pack model inference on the CPU or GPU, supporting request batching with little effort
* Fine tune embedding generation with in-domain samples

### Getting Started

```bash
docker pull infrawhispers/anansi:latest
docker run --name anansi -it -p 50051:50051 -p 50052:50052 -v /.cache:/app/.cache infrawhispers/anansi:latest
```

[1] standalone embedding generation using [INSTRUCTOR](https://github.com/HKUNLP/instructor-embedding)
```bash
curl \
-X POST http://172.17.0.1:50052/encode \
-H 'Content-Type: application/json' \
-d '{
    "batches":[{
        "model_name":"INSTRUCTOR_LARGE",
        "model_class":"ModelClass_INSTRUCTOR",
        "text":{
            "data": [
                {
                    "instruction": "Represent the Science title:",
                    "value": "3D ActionSLAM: wearable person tracking ..."
                },
                {
                    "instruction": "Represent the Nature title:",
                    "value": "Inside Gohar World and the Fine, Fantastical Art"
                }
            ]
        }
    }]}
'
```

---
## Documentation
We use docusaurus to generate our documenation, please either refer to the READMEs <a href="https://github.com/infrawhispers/anansi/tree/main/docs/docs/embedds/getting-started.md" target="_blank">here</a> or check out the documentation website.

---
### FAQ

#### What's with the name?

<p>
<b>anansi</b> (/əˈnɑːnsi/ ə-NAHN-see; literally translates to spider) is an <a href="https://en.wikipedia.org/wiki/Anansi" target="_blank">Akan folktale character</a> and god of stories, wisdom and knowledge. We thought it was an apt name as we aim to provide ML applications with turn-key memory and persistence.
</p>

#### How do I contact the developers?

<p>
Hop onto Discord via this <a href=https://discord.gg/xNyytmxrWh>invite link</a> or shoot an email to infrawhispers@proton.me
</p>

#### How do I contribute?

<p>
We welcome contributions of all sizes and contributors at all levels! Please take a look at open issues or look at #contributions in the Discord. 
</p>
