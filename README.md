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

### Project Overview

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

`/embeddings`

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
