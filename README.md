## Seperated DDPM

A pytorch implementation of Seperated DDPM.

The codes are based on Huggingface diffusers from https://github.com/huggingface/diffusers.git

Author: Da Eun Lee

## Method

<p align="center">
  <img src="resource/1.png" />
</p>

<p align="center">
  <img src="resource/2.png" />
</p>

<p align="center">
  <img src="resource/3.png" />
</p>

<p align="center">
  <img src="resource/4.png" />
</p>

## Experiments

Trajectories of mean values of noisy images following time steps in reverse process.

1. Linear

<p align="center">
  <img src="resource/5.png" />
</p>

2. Sqaure

<p align="center">
  <img src="resource/6.png" />
</p>

3. Sqaure root

<p align="center">
  <img src="resource/7.png" />
</p>

4. Sigmoid

<p align="center">
  <img src="resource/8.png" />
</p>

5. Shifted Diffusion 

<p align="center">
  <img src="resource/9.png" />
</p>

## References

PRDC codes are cited by the paper "https://proceedings.mlr.press/v119/naeem20a/naeem20a.pdf"

Related Works
DDPM: "https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf"
PriorGrad: "https://arxiv.org/pdf/2106.06406"
Shifted Diffusion: "https://openaccess.thecvf.com/content/CVPR2023/papers/Zhou_Shifted_Diffusion_for_Text-to-Image_Generation_CVPR_2023_paper.pdf"
ShiftDDPMs: "https://ojs.aaai.org/index.php/AAAI/article/download/25465/25237"

