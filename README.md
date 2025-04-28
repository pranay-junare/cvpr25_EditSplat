<p align="center">
  <h1 align="center"><strong>[CVPR 2025] EditSplat: Multi-View Fusion & Attention-Guided Optimization for View-Consistent 3D Scene Editing</strong></h1>
</p>

<p align="center">
  Dong In Lee<sup>1</sup>, Hyeongcheol Park<sup>1</sup>, Jiyoung Seo<sup>1</sup>, Eunbyung Park<sup>2</sup>,<br>
  Hyunje Park<sup>1</sup>, Ha Dam Baek<sup>1</sup>, Sangheon Shin<sup>3</sup>, Sangmin Kim<sup>3</sup>, Sangpil Kim<sup>1†</sup><br><br>
  <sup>1</sup>Korea University, &nbsp; <sup>2</sup>Yonsei University, &nbsp; <sup>3</sup>Hanwha Systems
</p>

<div align="center">
  <a href="https://arxiv.org/abs/2412.11520">
    <img src="https://img.shields.io/badge/arXiv-2412.11520-red?logo=arxiv" alt="arXiv Badge">
  </a>
  <a href="https://kuai-lab.github.io/editsplat2024/">
    <img src="https://img.shields.io/badge/Project-Page-blue?logo=website" alt="Project Page">
  </a>
</div>

<p align="center">
  <img src="./assets/teaser.png" alt="EditSplat Teaser" style="width:100%;"/>
</p>

## **⚙️ Installation**

Tested on Ubuntu 22.04 + CUDA 11.8 + Python 3.9 (RTX A6000 / RTX 3090).

> **Note**: The GPU memory requirement depends on your dataset size.

```bash
conda env create -f environment.yml
conda activate editsplat
```

## **📂 Dataset**

We provide a preprocessed **Face** dataset.

- 📥 Download here: [Drive Link](https://drive.google.com/drive/folders/1zpkYAJsJxcs13J0bZa-jThuiWgStdpWX?usp=sharing)

After downloading, move the dataset into the cvpr25_EditSplat/dataset/ directory.

If you want to edit your own dataset, you must first pre-train a 3D Gaussian Splatting (3DGS) model from your custom dataset using COLMAP for camera poses.

> *We are planning to release more datasets with detailed instructions soon — stay tuned!*

## **🎨 Editing**
<p align="center">
  <img src="./assets/pipeline.png" alt="EditSplat Teaser" style="width:100%;"/>
</p>
To run the editing pipeline:

```bash
./script/editing_face_to_marble_sculpture.sh
```

The edited 3D Gaussian Splatting outputs will be saved under `cvpr25_EditSplat/output`.

You can render novel views from the updated 3D scene stored in `cvpr25_EditSplat/output/point_cloud/`.

## **📜 Citation**

If you find our work useful, please consider citing:

```tex
@article{lee2024editsplat,
  title={EditSplat: Multi-View Fusion and Attention-Guided Optimization for View-Consistent 3D Scene Editing with 3D Gaussian Splatting},
  author={Dong In Lee and Hyeongcheol Park and Jiyoung Seo and Eunbyung Park and Hyunje Park and Ha Dam Baek and Sangheon Shin and Sangmin Kim and Sangpil Kim},
  journal={arXiv preprint arXiv:2412.11520},
  year={2024},
}
```
