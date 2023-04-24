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
anansi is an open source project aimed at accelerating the adoption of ML applications by providing packaged solutions for content encoding and memory.
official docker images are available here.
</p>

## Project Overview

<table>
<thead>
<tr>
      <th>Project Name</th>
      <th>Git Folder</th>
      <th>Summary</th>
      <th>Usage</th>
</tr>
</thead>

<tbody>
<tr style="vertical-align:top">
<td>embedds 🛏</td>
<td>

`/embedds`

</td>
<td>
<p>turn-key generation of embeddings using models from the <a href="https://huggingface.co/spaces/mteb/leaderboard">MTEB Leaderboard</a>, with bin-packing and CPU + GPU support.</p>
</td>

<td>

```
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
</tr>
</tbody>
</table>


## What can anansi be used for?
anansi is an LLM tooling project that has a vector storage and content encoding libraries developed in Rust, which allows the code to be compiled into WebAssembly (WASM) for use on different platforms, including web browsers and smartphones. The library includes a conversion technique that decreases the model size while also improving CPU and hardware accelerator latency through a process called quantization.

By employing quantization, the memory usage of a vector dimension is reduced from 4 bytes to 1 byte, saving **75%** of memory. This is done by converting f32 to u8 using scalar quantization. Additionally, anansi uses the ONNX runtime to compress embedding models, leading to at least a 2X reduction in the model size. These are some of the use cases of anansi
- **Resource-constrained environments**: anansi can be particularly useful for applications running on devices with limited computational power and memory, such as smartphones or IoT devices. The reduced model size and improved latency from quantization enable better performance on these devices compared to other similarity search engines.
- **Cross-platform deployment**: Since anansi can be compiled to WebAssembly (WASM), it can be easily deployed in web browsers, offering a broader range of platform compatibility than Faiss and Milvus, which mainly target server and desktop environments.
- **Energy efficiency**: Applications that require energy-efficient solutions, such as those running on battery-powered devices or in data centers with strict power limitations, could benefit from anansi's smaller models and reduced computational requirements. This could lead to longer battery life and lower energy consumption compared to using others.
- **Edge computing**: In edge computing scenarios where data processing is performed near the source of data generation, anansi's ability to run on various platforms and lower resource requirements make it a more suitable choice than Faiss and Milvus, which may have difficulty operating effectively on some edge devices.
- **Low-bandwidth environments**: In situations where there are constraints on network bandwidth, such as remote or rural areas with limited connectivity, anansi's reduced model sizes can be beneficial. Smaller models require less data transfer, making it easier to deploy and update applications even with limited bandwidth.
- **Privacy-preserving applications**: For applications that involve processing sensitive user data, anansi's ability to run efficiently on various platforms, including web browsers, smartphones and edge devices, can be valuable (think **GDPR**). This allows for local data processing, reducing the need to send sensitive information to remote servers for analysis, thus better preserving user privacy. Faiss and Milvus, with their focus on server and desktop environments, might not be as well-suited for these privacy-conscious scenarios.


## Architecture overview


## FAQ

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

## License

