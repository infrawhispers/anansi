---
sidebar_position: 3
hide_title: true
title: Supported Models
---

## Supported Models

embedds architecture is rooted in the concept of a `ModelClass` - these represent a collection of fine tuned models that have a similar graph architecture and positioning in the market. Non-exhaustive list of available models is as follows:

<table>
<thead>
<tr>
<th>
Model Class
</th>
<th>
Model Name
</th>
<th>
Size
</th>
</tr>
</thead>
<tbody>

<tr>
<td>
ModelClass_E5
</td>
<td>
E5_LARGE
</td>
<td>
1.2GB
</td>
</tr>

<tr>
<td>
ModelClass_E5
</td>
<td>
E5_BASE
</td>
<td>
418MB
</td>
</tr>

<tr>
<td>
ModelClass_E5
</td>
<td>
E5_SMALL
</td>
<td>
128MB
</td>
</tr>

<tr>
<td>
ModelClass_INSTRUCTOR
</td>
<td>
INSTRUCTOR_LARGE
</td>
<td>
1.5GB
</td>
</tr>


<tr>
<td>
ModelClass_INSTRUCTOR
</td>
<td>
INSTRUCTOR_BASE
</td>
<td>
421MB
</td>
</tr>


<tr>
<td>
ModelClass_CLIP
</td>
<td>
RN50_OPENAI
</td>
<td>
textual: 255MB | visual: 153MB
</td>
</tr>

<tr>
<td>
ModelClass_CLIP
</td>
<td>
RN50_YFCC15M
</td>
<td>
textual: 255MB | visual: 153MB
</td>
</tr>

<tr>
<td>
ModelClass_CLIP
</td>
<td>
RN50_CC12M
</td>
<td>
textual: 255MB | visual: 153MB
</td>
</tr>

<tr>
<td>
ModelClass_CLIP
</td>
<td>
VIT_L_14_336_OPENAI
</td>
<td>
textual: 473MB | visual: 1.2GB
</td>
</tr>


</tbody>
</table>

Note: For more information about CLIP class models, please look at <a href="https://github.com/infrawhispers/anansi/blob/main/embedds/src/clip_models.rs">clip_models.rs</a>.

### Including a Fine Tuned Model

<p>
this is currently a WIP - see this <a href="https://github.com/infrawhispers/anansi/issues/2">GitHub
</a> issue for progress.
</p>
