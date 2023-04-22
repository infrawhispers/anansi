---
sidebar_position: 1
slug: /
title: Overview
hide_title: true
tags:
  - documentation
  - getting started
  - quickstart
---

<div align="center">
<img width="150" src="/img/anansi.jpg" />
</div>
<div align="center">
<p>
Open Source Tooling for Applied Machine Learning

[![license](https://img.shields.io/badge/license-Apache_2.0-red.svg)](https://github.com/infrawhispers/anansi/blob/HEAD/LICENSE)

</p>
</div>

<div>
<p align="center">

</p>
</div>

## Overview

<p>
<b>anansi</b> is a collection of open source tools aimed at easing the adoption of machine learning within your applications. Currently
the project is focused on two primary workstreams:
<br/>
</p>
<ul>
<li>Creating and evaluating embeddings using the <a href="https://huggingface.co/spaces/mteb/leaderboard" target="_blank">Massive Text Embedding</a> leaderboard, on both CPUs and GPUs.</li>
<li>Providing a correct and highly-performant embedding storage that runs within WASM.</li>
</ul>
<br/>
With one or both of the aforementioned pieces, developers can quickly augment their applications with machine learning <b>without relying on Python</b>. anansi allows one to:

<ul>
<li>Generate embeddings for text or images and semantically search through them.</li>
<li>Create recommendations by storing multi-dimensional features + searching using Hamming distance.</li>
<li>Fine tune their embeddings via a live continuous process.</li>
</ul>

<i>In the coming weeks (Q3 2023) anansi will be providing a managed service to further reduce adoption and setup costs.</i>

## Getting Started

Please refer to the detailed documenation for:

<ul>
<li><a href="/embedds">Embedds</a> (embedding generation)</li>
<li><a href="/horus">Horus</a> (embedding search)</li>
</ul>

## Contributing

We welcome contributions of all sizes and contributors at all levels!
Please take a look at open issues on <a href="https://github.com/infrawhispers/anansi">GitHub</a> or hop into the <a href="https://discord.gg/xNyytmxrWh">Discord</a> and take a look at `#contributions`.
