<div align="center">
<img src="assets/93cc350bee4c4e43f38c9a078acdde29.png" width="680" alt="Teaser PTP" />
<h2>Perceive-then-Plan: Layout-as-Policy for Monocular 3D Scene Layout Estimation</h2>
<!-- <p><b>ECCV 2026</b></p> -->
<p>
<a href="https://colezwhy.github.io/" target="_blank">Junwei Zhou</a>,
<a href="https://yuwingtai.github.io/" target="_blank">Yu-Wing Tai</a>
</p>
<p>Dartmouth College</p>
</div>


>**TL;DR**: <em> We propose a perceive-then-plan framework with two VLMs (Perceiver and LaP Planner) for monocular 3D layout estimation, enabling both visual alignment and physical plausibility.</em>

<p align="center">
  <a href="https://colezwhy.github.io/perceivethenplan/">
    <img src="https://img.shields.io/badge/Project-Website-green">
  </a>
<a href="https://arxiv.org/abs/2605.25326">
  <img src="https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=white" alt="arXiv">
</a>
    <a href="#">
    <img src="https://visitor-badge.laobi.icu/badge?page_id=Colezwhy.PTP" alt="Visitors">
  </a>
</p>


Official implementation for paper 'Perceive-then-Plan: Layout-as-Policy for Monocular 3D Scene Layout Estimation'.

In this work, we formulate monocular 3D layout estimation as a perceive-then-plan problem with vision-language models, where a Perceiver first grounds the 3D objects and then a Planner iteratively refines the scene hypothesis through actions that improve physical plausibility while preserving consistency with the input image.

https://github.com/user-attachments/assets/6f4b36e1-c50d-436a-9241-bd0e700c809e


## Updates and TODOs
- ✔️ 06/15/2025: Initialize the project page.
- 🔲 TODO: The code will soon be released upon acceptance. Please stay tuned!


## Method 
<p align="center">
  <img src="assets/11fee07ae661ec3e0f6e4941ea41e31e.png" width="800" alt="Overall pipeline" />
</p>

<p align="center">
  GENA3D bridges the 2D amodal completion with 3D generation using deliberaely designed View-Wise Cross Attention and Stereo-Conditioned Cross Attention in the Sparse Structure Generation Stage, with synthesized sparse-view 3D consistent occlusions as training data.
</p>

<p align="center">
  <img src="assets/308b495e4ed18e608a635ee2861486bb.png" width="540" alt="Module design" />
</p>

<p align="center">
  A detailed illustration of our proposed ViewWise Cross Attention and Stereo-Conditioned Cross Attention modules.
</p>

## Results
<p align="center">
  <img src="assets/4851b41a68ad377dfcef3a6ac9ad03a1.png" width="720" alt="Object-level" />
</p>

<p align="center">
  Results on GSO object-level synthetic dataset.
</p>

<p align="center">
  <img src="assets/d8797c493af58b11cb24b4fe7fa56e7e.png" width="580" alt="In-the-wild" />
</p>

<p align="center">
  Results on in-the-wild real-world captures.
</p>

## Citation
Here is the bibtex reference. If you find our work interesting or useful, please give us a :star: or cite our paper!
```
@misc{zhou2026perceivethenplan,
      title={Perceive-then-Plan: Layout-as-Policy for Monocular 3D Scene Layout Estimation}, 
      author={Junwei Zhou and Yu-Wing Tai},
      year={2026},
      eprint={2605.25326},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2605.25326}, 
}
```
